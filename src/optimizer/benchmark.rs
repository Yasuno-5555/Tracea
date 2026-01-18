use crate::PipelineConfig;
use crate::runtime::{RuntimeManager, BufferId, KernelArg};
use std::sync::Arc;

pub struct BenchmarkResult {
    pub tflops: f32,
    pub latency_ms: f32,
}

/// Trait for measuring the actual performance of a generated kernel
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    pub cuda_version: String,
    pub driver_version: String,
    pub sm_arch: u32,
}

pub trait MicroBenchmark {
    fn measure(&self, config: &PipelineConfig) -> BenchmarkResult;
    fn m(&self) -> u32;
    fn n(&self) -> u32;
    fn k(&self) -> u32;
    fn device_info(&self) -> EnvironmentInfo;
}

/// Simulated benchmark for testing the Auto-tuner without a physical GPU
pub struct SimulatedBenchmark {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl MicroBenchmark for SimulatedBenchmark {
    fn measure(&self, config: &PipelineConfig) -> BenchmarkResult {
        let base = 10.0;
        let scale = (config.num_stages as f32 * config.m_tile as f32 * config.n_tile as f32).log2();
        let tflops = base + scale * 0.5;
        let latency_ms = (2.0 * self.m as f32 * self.n as f32 * self.k as f32) / (tflops * 1e9);
        BenchmarkResult { tflops, latency_ms }
    }
    fn m(&self) -> u32 { self.m }
    fn n(&self) -> u32 { self.n }
    fn k(&self) -> u32 { self.k }
    fn device_info(&self) -> EnvironmentInfo { 
        EnvironmentInfo {
            cuda_version: "simulated".to_string(),
            driver_version: "simulated".to_string(),
            sm_arch: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub config: PipelineConfig,
    pub score: f32,
}

pub(crate) struct NVRTCBenchmark {
    pub runtime: Arc<RuntimeManager>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    // Cached buffers handles
    pub buffers_u16: (BufferId, BufferId),
    pub buffer_c: BufferId,
}

impl NVRTCBenchmark {
    pub fn new(runtime: Arc<RuntimeManager>, m: u32, n: u32, k: u32) -> Self {
        let size_a = (m * k) as usize;
        let size_b = (k * n) as usize;
        let size_c = (m * n) as usize;
        
        let a = runtime.alloc_u16(size_a).expect("Alloc A Failed");
        let b = runtime.alloc_u16(size_b).expect("Alloc B Failed");
        let c = runtime.alloc_f32(size_c).expect("Alloc C Failed");
        
        Self {
            runtime,
            m, n, k,
            buffers_u16: (a, b),
            buffer_c: c,
        }
    }
}

impl MicroBenchmark for NVRTCBenchmark {
    fn measure(&self, config: &PipelineConfig) -> BenchmarkResult {
        let emitter = crate::emitter::cuda::CUDAEmitter::new();
        let source = emitter.generate_pipelined_gemm(config.clone());
        let kernel_name = "gemm_pipelined_kernel";

        let kernel_id = match self.runtime.compile(&source, kernel_name) {
            Ok(k) => k,
            Err(_) => return BenchmarkResult { tflops: 0.0, latency_ms: 1e9 },
        };

        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;
        let n_stages = config.num_stages;
        
        let element_size = if config.use_tensor_cores { 2 } else { 4 };
        let s_a = if config.use_tensor_cores { kt } else { kt + 4 };
        let s_b = if config.use_tensor_cores { nt } else { nt + 4 };
        let smem_size = (n_stages * mt * s_a + n_stages * kt * s_b) * element_size;

        let grid_dim = ((self.m + mt - 1) / mt, (self.n + nt - 1) / nt, 1);
        let block_dim = if config.use_tensor_cores { (128, 1, 1) } else { (16, 16, 1) };
        
        // Prepare Args
        let (buf_a, buf_b) = self.buffers_u16;
        let buf_c = self.buffer_c;
        
        let args = vec![
            KernelArg::Buffer(buf_a),
            KernelArg::Buffer(buf_b),
            KernelArg::Buffer(buf_c),
            KernelArg::Int(self.m as i32),
            KernelArg::Int(self.n as i32),
            KernelArg::Int(self.k as i32),
        ];

        let start = std::time::Instant::now();
        let iterations = 5;
        for i in 0..iterations {
             if let Err(e) = self.runtime.launch(kernel_id, grid_dim, block_dim, smem_size as u32, args.clone()) {
                  eprintln!("[Tracea] ⚠️  Launch Failed on iteration {}: {:?}", i, e);
                  return BenchmarkResult { tflops: 0.0, latency_ms: 1e9 };
             }
        }
        
        if let Err(e) = self.runtime.get_device().synchronize() {
             eprintln!("[Tracea] ⚠️  Sync Failed: {:?}", e);
             return BenchmarkResult { tflops: 0.0, latency_ms: 1e9 };
        }
        let dur = start.elapsed().as_secs_f32() / iterations as f32;
        let latency_ms = dur * 1000.0;
        let tflops = (2.0 * self.m as f32 * self.n as f32 * self.k as f32) / (dur * 1e12);
        
        BenchmarkResult { tflops, latency_ms }
    }
    
    fn m(&self) -> u32 { self.m }
    fn n(&self) -> u32 { self.n }
    fn k(&self) -> u32 { self.k }
    
    #[allow(deprecated)] // We are using low-level API here for info
    fn device_info(&self) -> EnvironmentInfo { 
        let d = self.runtime.get_device();
        let mut driver_ver: i32 = 0;
        unsafe { cudarc::driver::sys::lib().cuDriverGetVersion(&mut driver_ver) };
        
        // Get NVRTC Version (Toolkit)
        let mut nvrtc_major = 0;
        let mut nvrtc_minor = 0;
        unsafe { cudarc::nvrtc::sys::lib().nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) };

        let sm_major = d.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(0);
        let sm_minor = d.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0);
        let sm = sm_major * 10 + sm_minor;
        
        EnvironmentInfo {
            cuda_version: format!("{}.{}", nvrtc_major, nvrtc_minor),
            driver_version: format!("{}.{}", driver_ver / 1000, (driver_ver % 1000) / 10),
            sm_arch: sm as u32,
        }
    }
}
