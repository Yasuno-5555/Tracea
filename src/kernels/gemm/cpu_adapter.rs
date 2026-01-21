use crate::core::tuning::{TunableKernel, SearchSpace};
use crate::backend::cpu::CpuBackend;
use crate::backend::Backend;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GemmProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl GemmProblem {
    pub fn signature(&self) -> String {
        format!("m{}_n{}_k{}", self.m, self.n, self.k)
    }
}

/// Configuration for CPU GEMM execution.
/// Unlike PipelineConfig (which is GPU focused), this is CPU specific.
/// Or we could try to reuse PipelineConfig if we map concepts?
/// For now, distinct config is safer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuGemmConfig {
    pub m_block: usize,
    pub n_block: usize,
    pub k_block: usize,
    pub num_threads: usize, 
    pub vector_width: usize, // e.g. 8 for AVX2 (f32), 16 for AVX512
}

pub struct GemmAdapter {
    pub backend: CpuBackend,
    pub problem: GemmProblem,
}

impl GemmAdapter {
    pub fn new(backend: CpuBackend, problem: GemmProblem) -> Self {
        Self { backend, problem }
    }
}

impl TunableKernel for GemmAdapter {
    type Config = CpuGemmConfig;

    fn name(&self) -> &'static str {
        "cpu_gemm"
    }

    fn search_space(&self) -> SearchSpace<Self::Config> {
        let mut candidates = Vec::new();
        
        // Rule C: Alignment Priority
        // Prioritize block sizes that divide the problem dimensions evenly.
        let block_candidates = [32, 64, 128, 256];
        let threads = [1, 2, 4, 8, 16]; 
        
        for &bs in &block_candidates {
            // Check alignment (Rule C)
            let is_aligned = (self.problem.m % bs == 0) && (self.problem.n % bs == 0) && (self.problem.k % bs == 0);
            
            for &t in &threads {
                let mut cfg = CpuGemmConfig {
                    m_block: bs,
                    n_block: bs,
                    k_block: bs,
                    num_threads: t,
                    vector_width: 8,
                };
                
                if is_aligned {
                    // Place at front or prioritize in real tuner
                    candidates.insert(0, cfg);
                } else {
                    candidates.push(cfg);
                }
            }
        }

        SearchSpace::new(candidates)
    }

    fn is_feasible(&self, cfg: &Self::Config) -> bool {
        // CPU feasibility is loose, mostly about memory.
        if cfg.num_threads > self.backend.logical_cores {
            return false;
        }
        true
    }

    fn benchmark(&self, cfg: &Self::Config) -> Option<f32> {
        let m = self.problem.m;
        let n = self.problem.n;
        let k = self.problem.k;
        
        // Allocate buffers (random init ideally, but zeros/ones ok for perf)
        // Flattened row-major
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        let start = std::time::Instant::now();
        
        let num_threads = cfg.num_threads.max(1);
        let m_per_thread = (m + num_threads - 1) / num_threads;

        let a = std::sync::Arc::new(a);
        let b = std::sync::Arc::new(b);
        // We can't easily write to C in parallel without UnsafeCell or splitting.
        // For simplicity in this tuner demo, let's use scoped threads or just simple split.
        // Since we don't have scoped threads in std 2018 (available in 1.63+ but let's be safe), 
        // we will use crossbeam or just spawn and join on chunks if we can moved data. 
        // OR, simply unsafe wrap for C.
        
        // Helper for raw pointer access to C to allow parallel writes to disjoint slices
        struct UnsafeSlice(*mut f32);
        unsafe impl Send for UnsafeSlice {}
        unsafe impl Sync for UnsafeSlice {}
        let c_ptr = UnsafeSlice(c.as_mut_ptr());

        let handles: Vec<_> = (0..num_threads).map(|tid| {
            let a = a.clone();
            let b = b.clone();
            let c_ptr = UnsafeSlice(c_ptr.0);
            let m_block = cfg.m_block;
            let n_block = cfg.n_block;
            let k_block = cfg.k_block;
            
            std::thread::spawn(move || {
                let m_start = tid * m_per_thread;
                let m_end = std::cmp::min(m_start + m_per_thread, m);
                
                if m_start >= m_end { return; }

                unsafe {
                    let c_raw = c_ptr.0;
                    // Tiled Loop
                    for i_blk in (m_start..m_end).step_by(m_block) {
                        let i_end = std::cmp::min(i_blk + m_block, m_end);
                        for j_blk in (0..n).step_by(n_block) {
                            let j_end = std::cmp::min(j_blk + n_block, n);
                            for l_blk in (0..k).step_by(k_block) {
                                let l_end = std::cmp::min(l_blk + k_block, k);
                                
                                // Micro-kernel (naive for now)
                                for i in i_blk..i_end {
                                    for j in j_blk..j_end {
                                        let mut sum = 0.0f32;
                                        for l in l_blk..l_end {
                                            sum += a[i * k + l] * b[l * n + j];
                                        }
                                        *c_raw.add(i * n + j) += sum;
                                    }
                                }
                            }
                        }
                    }
                }
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }

        let elapsed = start.elapsed();
        let nanos = elapsed.as_nanos() as f32;
        
        let flops = 2.0 * (m as f32) * (n as f32) * (k as f32);
        let tflops = (flops / 1e12) / (nanos / 1e9);

        // Optional: verify C[0] to prevent optimization? 
        // println!("DEBUG: C[0] = {}", c[0]);

        Some(tflops)
    }

    fn cache_key(&self) -> String {
        format!("{}:{}:{}", self.name(), self.backend.device_id(), self.problem.signature())
    }
}
