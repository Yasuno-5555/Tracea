use crate::core::tuning::{TunableKernel, SearchSpace};
use crate::core::config::{PipelineConfig, SpecializedInstruction, LayoutPolicy};
use crate::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use crate::emitter::universal::UniversalEmitter;
use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CudaGemmProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl CudaGemmProblem {
    pub fn signature(&self) -> String {
        format!("m{}_n{}_k{}", self.m, self.n, self.k)
    }
}

pub struct CudaGemmAdapter {
    pub runtime: Arc<RuntimeManager>,
    pub problem: CudaGemmProblem,
    // Buffers
    pub a_buf: crate::runtime::manager::BufferId,
    pub b_buf: crate::runtime::manager::BufferId,
    pub c_buf: crate::runtime::manager::BufferId,
}

impl CudaGemmAdapter {
    pub fn new(runtime: Arc<RuntimeManager>, problem: CudaGemmProblem) -> Self {
        let size_a = problem.m * problem.k * 2; // f16
        let size_b = problem.k * problem.n * 2; // f16
        let size_c = problem.m * problem.n * 4; // f32 accumulation

        let a = runtime.alloc(size_a, DeviceBackend::Cuda).unwrap();
        let b = runtime.alloc(size_b, DeviceBackend::Cuda).unwrap();
        let c = runtime.alloc(size_c, DeviceBackend::Cuda).unwrap();

        Self { runtime, problem, a_buf: a, b_buf: b, c_buf: c }
    }
}

impl TunableKernel for CudaGemmAdapter {
    type Config = PipelineConfig;

    fn name(&self) -> &'static str {
        "cuda_gemm_mma"
    }

    fn search_space(&self) -> SearchSpace<Self::Config> {
        let mut candidates = Vec::new();
        
        // "Golden" configurations for RTX 3070 (Ampere)
        // 9 warps = 1 Prod + 8 Cons. 128 / 8 = 16 (1 frag/warp).
        let goldens = vec!(
            (128, 128, 32, 9, 2),
            (128, 128, 32, 5, 3), // 128 / 4 = 32 (2 frags/warp)
            (64, 64, 32, 5, 2),   // 64 / 4 = 16 (1 frag/warp)
        );

        for (m, n, k, w, s) in goldens {
            let mut cfg = PipelineConfig::new(s, m, n, k);
            cfg.instruction = SpecializedInstruction::CudaMMA;
            cfg.force_num_warps = Some(w);
            cfg.layout_policy = Some(LayoutPolicy::RowMajor); 
            candidates.push(cfg);
        }
        
        SearchSpace::new(candidates)
    }

    fn is_feasible(&self, cfg: &Self::Config) -> bool {
        // Tile size must be multiple of MMA size (16)
        if cfg.m_tile % 16 != 0 || cfg.n_tile % 16 != 0 || cfg.k_tile % 16 != 0 {
            return false;
        }

        // Warp tiling: Must divide perfectly into 16-row chunks per consumer
        let num_warps = cfg.force_num_warps.unwrap_or(9);
        if num_warps < 2 { return false; }
        
        let cons_warps = num_warps - 1;
        if cfg.m_tile % (cons_warps * 16) != 0 {
            return false;
        }

        // Vectorization: Global N and Tile NT must be 16-byte aligned (8 halfs)
        if cfg.n_tile % 8 != 0 {
            return false;
        }

        // Shared Memory Limit: 96KB for sm_8x
        let a_stride = cfg.k_tile + 8;
        let b_stride = cfg.n_tile + 8;
        let smem_a_bytes = cfg.m_tile * a_stride * 2;
        let smem_b_bytes = cfg.k_tile * b_stride * 2;
        let required_smem = (smem_a_bytes + smem_b_bytes) as usize * cfg.num_stages as usize + 128; // Header 128B
        
        if required_smem > 96000 {
            return false;
        }

        true
    }

    fn benchmark(&self, cfg: &Self::Config) -> Option<f32> {
        // --- v3 Optimization: Check for Specialized Template First ---
        let num_warps = cfg.force_num_warps.unwrap_or(8);
        let block = (num_warps * 32, 1, 1); 
        let grid = (
            (self.problem.m as u32 + cfg.m_tile - 1) / cfg.m_tile,
            (self.problem.n as u32 + cfg.n_tile - 1) / cfg.n_tile,
            1
        );
        let a_stride = cfg.m_tile * (cfg.k_tile + 8) * 2;
        let b_stride = cfg.k_tile * (cfg.n_tile + 8) * 2;
        let smem_bytes = ((a_stride + b_stride) as usize * cfg.num_stages as usize + 512) as u32;

        let a_ptr = self.runtime.get_device_ptr(self.a_buf).ok()?;
        let b_ptr = self.runtime.get_device_ptr(self.b_buf).ok()?;
        let c_ptr = self.runtime.get_device_ptr(self.c_buf).ok()?;

        let start = std::time::Instant::now();

        /*
        // Attempt v3 Dispatch (Pre-compiled kernels)
        let dispatched = crate::kernels::gpu::gpu_dispatch::dispatch_gpu_gemm(
            cfg,
            a_ptr, b_ptr, c_ptr,
            self.problem.m as i32, self.problem.n as i32, self.problem.k as i32,
            grid, block, smem_bytes,
            std::ptr::null_mut() // stream
        );

        if dispatched {
            self.runtime.synchronize();
            let nanos = start.elapsed().as_nanos() as f32;
            let gflops = (2.0 * self.problem.m as f32 * self.problem.n as f32 * self.problem.k as f32) / nanos;
            return Some(gflops);
        }
        */

        // --- Fallback: NVRTC String Emitter ---
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Gemm { 
                m: self.problem.m as u32,
                n: self.problem.n as u32,
                k: self.problem.k as u32,
                batch: 1,
                epilogue: vec![],
            },
            precison: "f16".to_string(),
            tiling: cfg.clone(),
            conv_magic_strategy: None,
            polyhedral_strategy: None,
        };

        let emitter = UniversalEmitter::new(DeviceBackend::Cuda);
        let source = emitter.generate(ir);
        let kernel_name = "gemm_mma_kernel";

        let kernel_id = match self.runtime.compile(&source, kernel_name, DeviceBackend::Cuda) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("Compile Failed: {}", e);
                return None;
            }
        };

        let args = vec![
            KernelArg::Buffer(self.a_buf),
            KernelArg::Buffer(self.b_buf),
            KernelArg::Buffer(self.c_buf),
            KernelArg::Int(self.problem.m as i32),
            KernelArg::Int(self.problem.n as i32),
            KernelArg::Int(self.problem.k as i32),
        ];

        if let Err(e) = self.runtime.launch(kernel_id, grid, block, smem_bytes, args) {
             eprintln!("Launch Failed: {}", e);
             return None;
        }
        self.runtime.synchronize();
        let nanos = start.elapsed().as_nanos() as f32;

        let ops = 2.0 * self.problem.m as f64 * self.problem.n as f64 * self.problem.k as f64;
        let tflops = ops / (nanos as f64 * 1000.0); // Ops / (nanos * 1e3) = TFLOPS
        Some(tflops as f32)
    }

    fn cache_key(&self) -> String {
        format!("cuda_gemm_{}", self.problem.signature())
    }
}

impl crate::optimizer::benchmark::MicroBenchmark for CudaGemmAdapter {
    fn m(&self) -> u32 { self.problem.m as u32 }
    fn n(&self) -> u32 { self.problem.n as u32 }
    fn k(&self) -> u32 { self.problem.k as u32 }
    
    fn device_info(&self) -> crate::optimizer::benchmark::EnvironmentInfo {
        crate::optimizer::benchmark::EnvironmentInfo {
            backend: DeviceBackend::Cuda,
            api_version: "13.1".to_string(),
            driver_version: "unknown".to_string(),
            arch: "Ampere".to_string(),
        }
    }

    fn validate_config(&self, config: &PipelineConfig) -> bool {
        self.is_feasible(config)
    }

    fn measure(&self, config: &PipelineConfig) -> crate::optimizer::benchmark::BenchmarkResult {
        let tflops = self.benchmark(config).unwrap_or(0.0);
        crate::optimizer::benchmark::BenchmarkResult {
            tflops,
            mean_tflops: tflops,
            std_dev: 0.0,
            latency_ms: if tflops > 0.0 { (2.0 * self.problem.m as f32 * self.problem.n as f32 * self.problem.k as f32) / (tflops * 1e9) * 1000.0 } else { 0.0 },
            observation: self.observe_hardware(config),
        }
    }

    fn observe_hardware(&self, _config: &PipelineConfig) -> Option<crate::optimizer::model::HardwareObservation> { None }
}
