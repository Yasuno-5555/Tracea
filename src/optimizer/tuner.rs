use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use crate::runtime::manager::{RuntimeManager, DeviceBackend};
use crate::optimizer::{AutoTuner, HardwareProfile, ProblemDescriptor, OptimizationGoal, TuningStats};
use crate::optimizer::history::{TuningHistory, HistoryEntry};
use crate::core::config::PipelineConfig;
use crate::policy::types::{GraphTopology, OperatorTopology};
use crate::optimizer::benchmark::MicroBenchmark;

#[derive(Debug)]
pub struct MetaTuner {
    runtime: std::sync::Weak<RuntimeManager>,
    tuners: HashMap<DeviceBackend, Mutex<AutoTuner>>,
    history: Mutex<TuningHistory>,
    pub evolution: std::sync::Arc<crate::optimizer::evolution::EvolutionaryEngine>,
}

impl MetaTuner {
    pub fn new(runtime: std::sync::Weak<RuntimeManager>) -> Self {
        let mut tuners = HashMap::new();
        
        // We can't check devices cleanly with Weak ref during construction (upgrade returns None)
        // So we just assume common backends or we require a later init step.
        // Or we just add all potential backends and they will fail fast if device not present during tune.
        
        // Cuda
        {
             let profile = HardwareProfile::rtx3070(); 
             let mut t = AutoTuner::new(profile);
             t.runtime = Some(runtime.clone());
             tuners.insert(DeviceBackend::Cuda, Mutex::new(t));
        }

        // Metal
        {
             let profile = HardwareProfile::apple_m1(); 
             let mut t = AutoTuner::new(profile);
             t.runtime = Some(runtime.clone());
             tuners.insert(DeviceBackend::Metal, Mutex::new(t));
        }
        
        // Rocm
         {
             let profile = HardwareProfile::mi250(); 
             let mut t = AutoTuner::new(profile);
             t.runtime = Some(runtime.clone());
             tuners.insert(DeviceBackend::Rocm, Mutex::new(t));
        }

        Self {
            runtime,
            tuners,
            history: Mutex::new(TuningHistory::load_or_create("tuning_history.json")),
            evolution: std::sync::Arc::new(crate::optimizer::evolution::EvolutionaryEngine::new("dna_database.json")),
        }
    }
    
    pub fn tune_operator(
        &self, 
        op: &OperatorTopology, 
        backend: DeviceBackend,
        iterations: usize
    ) -> Option<PipelineConfig> {
        let runtime_arc = self.runtime.upgrade()?;

        let problem = match op {
            OperatorTopology::Gemm { m, n, k, name, .. } => {
                ProblemDescriptor::new_gemm(*m as usize, *n as usize, *k as usize).named(name.clone())
            },
            OperatorTopology::Conv2d { n, c, h, w, k, r, s, stride, padding, name, .. } => {
                ProblemDescriptor::new_conv2d(
                    *n as usize, *h as usize, *w as usize, *c as usize, *k as usize,
                    *r as usize, *s as usize, *stride as usize, *padding as usize,
                    crate::optimizer::Layout::NHWC
                ).named(name.clone())
            },
            _ => return None, // Not tuneable yet
        };

        // 1. Check History 
        let key = format!("{:?}:{}", backend, problem.name); 
        {
            let history = self.history.lock().unwrap();
            if let Some(entry) = history.query(&key) {
                return Some(entry.config.clone());
            }
        }
        
        // 2. Tune using Live Runtime
        eprintln!("[MetaTuner] Tuning operator: {}", key);
        let mut tuner_guard = self.tuners.get(&backend)?.lock().unwrap();
        
        let best_config = match op {
            OperatorTopology::Gemm { m, n, k, .. } => {
                use crate::optimizer::benchmark::NVRTCBenchmark;
                let benchmark = NVRTCBenchmark::new(runtime_arc.clone(), *m, *n, *k);
                tuner_guard.optimize_v2(&benchmark, &problem, iterations, OptimizationGoal::MaximizeTFLOPS)
            },
            OperatorTopology::Conv2d { n, c, h, w, k, r, s, stride, padding, .. } => {
                use crate::optimizer::benchmark::NVRTCConvBenchmark;
                use crate::optimizer::benchmark::Conv2dProblem;
                use crate::optimizer::ConvBenchmarkAdapter;
                use crate::core::config::MagicNumberStrategy;
                
                let conv_problem = Conv2dProblem::new(
                    &problem.name, 
                    *n as usize, *h as usize, *w as usize, *c as usize, *k as usize,
                    *r as usize, *s as usize, *stride as usize, *padding as usize, 
                    1 // dilation default
                );
                
                let benchmark = NVRTCConvBenchmark::new(runtime_arc.clone(), conv_problem.clone());
                let hw_out = conv_problem.h_out() * conv_problem.w_out();
                let magic = MagicNumberStrategy::select_for(hw_out);
                
                let adapter = ConvBenchmarkAdapter {
                    inner: &benchmark,
                    magic_strategy: magic,
                };
                
                tuner_guard.optimize_v2(&adapter, &problem, iterations, OptimizationGoal::MaximizeTFLOPS)
            },
            _ => return None,
        };
        
        // 3. Record in History
        {
            let mut history = self.history.lock().unwrap();
            history.record(key.clone(), HistoryEntry {
                backend,
                config: best_config.clone(),
                tflops: 0.0, 
                timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                explanation: "Auto-tuned via MetaTuner".to_string(),
            });
        }
        
        Some(best_config)
    }

    pub fn get_tuning_stats(&self) -> HashMap<DeviceBackend, TuningStats> {
        let mut stats = HashMap::new();
        for (backend, tuner) in &self.tuners {
            let t = tuner.lock().unwrap();
            stats.insert(*backend, t.stats.clone());
        }
        stats
    }

    /// Primary evolution entry point for a ComputeAtom
    pub fn tune_atom(
        &self,
        atom: &crate::core::manifold::ComputeAtom,
        backend: DeviceBackend,
        generations: usize,
    ) -> Option<crate::core::mapper::MappingStrategy> {
        let runtime_arc = self.runtime.upgrade()?;
        let lattice = runtime_arc.doctor.synthesize_hardware_lattice();
        
        eprintln!("[MetaTuner] 🧬 Starting evolution for atom: {} on lattice: {}", atom.name, lattice.name);
        
        let best_strategy = self.evolution.search(atom, &lattice, generations, |strategy| {
            // Evaluator: Heuristic + Basic Occupancy check for now in POC
            // A full implementation would compile and run here using NVRTCBenchmark-like logic
            let (grid, block) = strategy.get_launch_params(atom);
            let threads = grid.0 * grid.1 * grid.2 * block.0 * block.1 * block.2;
            
            if threads == 0 { return 0.0; }
            
            // Simulated Fitness: Prefer larger tiles and higher occupancy
            let tile_score = strategy.tile_sizes.values().sum::<usize>() as f32;
            let occupancy_multiplier = if block.0 % 32 == 0 { 1.2 } else { 0.8 };
            
            tile_score * occupancy_multiplier
        });
        
        Some(best_strategy)
    }
}
