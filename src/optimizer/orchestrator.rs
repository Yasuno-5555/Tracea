use crate::optimizer::history::{TuningHistory, HistoryEntry};
use crate::runtime::manager::{RuntimeManager, DeviceBackend};
use crate::optimizer::{AutoTuner, HardwareProfile, ProblemDescriptor, OptimizationGoal};
use crate::core::config::PipelineConfig;
use crate::optimizer::benchmark::MicroBenchmark;
use std::sync::Arc;
use std::collections::HashMap;

pub struct CrossBackendTuner {
    runtime: Arc<RuntimeManager>,
    tuners: HashMap<DeviceBackend, AutoTuner>,
    history: TuningHistory,
}

impl CrossBackendTuner {
    pub fn new(runtime: Arc<RuntimeManager>) -> Self {
        let mut tuners = HashMap::new();
        let backends: Vec<_> = {
            let devices = runtime.devices.lock().unwrap();
            devices.keys().cloned().collect()
        };
        
        for backend in backends {
            let mut profile = match backend {
                DeviceBackend::Cuda => HardwareProfile::rtx3070(),
                DeviceBackend::Rocm => HardwareProfile::mi250(),
                _ => HardwareProfile::rtx3070(),
            };
            profile.backend = backend;
            tuners.insert(backend, AutoTuner::new(profile).with_runtime(runtime.clone()));
        }
        
        Self { 
            runtime, 
            tuners, 
            history: TuningHistory::load_or_create("tuning_history.json") 
        }
    }

    pub fn find_best_backend<B: MicroBenchmark>(
        &mut self, 
        benchmark: &B, 
        problem: &ProblemDescriptor,
        iterations: usize
    ) -> (DeviceBackend, PipelineConfig, f32) {
        let key = format!("all:{}:{}", problem.name, "FP16");
        if let Some(entry) = self.history.query(&key) {
            println!("[Orchestrator] 💎 History Hit! {}", entry.explanation);
            return (entry.backend, entry.config.clone(), entry.tflops);
        }

        let mut best_backend = DeviceBackend::Cpu;
        let mut best_config = PipelineConfig::new(2, 64, 64, 32);
        let mut best_tflops = -1.0;

        for (backend, tuner) in self.tuners.iter_mut() {
            println!("[Orchestrator] Tuning on Backend: {:?}", backend);
            let config = tuner.optimize_v2(benchmark, problem, iterations, OptimizationGoal::MaximizeTFLOPS);
            
            let res = benchmark.measure(&config);
            if res.tflops > best_tflops {
                best_tflops = res.tflops;
                best_backend = *backend;
                best_config = config;
            }
        }

        let explanation = TuningHistory::generate_explanation(
            best_backend, 
            best_tflops, 
            "Selected based on highest benchmarking score in real-time."
        );
        
        self.history.record(key, HistoryEntry {
            backend: best_backend,
            config: best_config.clone(),
            tflops: best_tflops,
            timestamp: 0, // Placeholder
            explanation: explanation.clone(),
        });

        println!("[Orchestrator] Winner: {:?} ({:.2} TFLOPS) - {}", best_backend, best_tflops, explanation);
        (best_backend, best_config, best_tflops)
    }
}
