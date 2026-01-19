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
    pub backend: crate::runtime::DeviceBackend,
    pub api_version: String,
    pub driver_version: String,
    pub arch: String,
}

pub trait MicroBenchmark {
    fn measure(&self, config: &PipelineConfig) -> BenchmarkResult;
    fn validate_config(&self, config: &PipelineConfig) -> bool;
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
    fn validate_config(&self, _config: &PipelineConfig) -> bool { true }
    fn m(&self) -> u32 { self.m }
    fn n(&self) -> u32 { self.n }
    fn k(&self) -> u32 { self.k }
    fn device_info(&self) -> EnvironmentInfo { 
        EnvironmentInfo {
            backend: crate::runtime::DeviceBackend::Cuda, 
            api_version: "simulated".to_string(),
            driver_version: "simulated".to_string(),
            arch: "simulated".to_string(),
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
    pub backend: crate::runtime::DeviceBackend,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub buffers_in: (BufferId, BufferId),
    pub buffer_c: BufferId,
}

impl NVRTCBenchmark {
    pub fn new(runtime: Arc<RuntimeManager>, m: u32, n: u32, k: u32) -> Self {
        let backend = if crate::emitter::rocm_driver::RocmDriverApi::get().is_some() {
            crate::runtime::DeviceBackend::Rocm
        } else {
            crate::runtime::DeviceBackend::Cuda
        };

        let size_a = (m * k) as usize;
        let size_b = (k * n) as usize;
        let size_c = (m * n) as usize;
        
        let a = runtime.alloc_f32(size_a, backend).expect("Alloc A Failed");
        let b = runtime.alloc_f32(size_b, backend).expect("Alloc B Failed");
        let c = runtime.alloc_f32(size_c, backend).expect("Alloc C Failed");
        
        Self {
            runtime,
            backend,
            m, n, k,
            buffers_in: (a, b),
            buffer_c: c,
        }
    }

    fn get_launch_params(&self, config: &PipelineConfig) -> ((u32, u32, u32), (u32, u32, u32), u32) {
        let (mt, nt, kt) = (config.m_tile, config.n_tile, config.k_tile);
        let grid_dim = ((self.m + mt - 1) / mt, (self.n + nt - 1) / nt, 1);
        
        let (block_dim, smem_size) = match self.backend {
            crate::runtime::DeviceBackend::Cuda => {
                let block = (128, 1, 1);
                let element_size = if config.use_tensor_cores() { 2 } else { 4 };
                let s_a = if config.use_tensor_cores() { kt } else { kt + 4 };
                let s_b = if config.use_tensor_cores() { nt } else { nt + 4 };
                let smem = (config.num_stages * mt * s_a + config.num_stages * kt * s_b) * element_size;
                (block, smem)
            }
            crate::runtime::DeviceBackend::Rocm => {
                let block = (128, 1, 1);
                let smem = (config.num_stages * mt * kt + config.num_stages * kt * nt) * 2;
                (block, smem)
            }
            crate::runtime::DeviceBackend::Metal => {
                let block = (128, 1, 1); 
                let smem = (mt * kt + kt * nt) * 2;
                (block, smem)
            }
        };

        (grid_dim, block_dim, smem_size)
    }
}

impl MicroBenchmark for NVRTCBenchmark {
    fn validate_config(&self, config: &PipelineConfig) -> bool {
         let emitter = crate::emitter::universal::UniversalEmitter::new(self.backend);
         let source = emitter.generate(crate::emitter::traits::UnifiedOpIR {
             op_type: crate::emitter::traits::UnifiedOpType::Gemm { m: self.m, n: self.n, k: self.k },
             precison: "f32".to_string(),
             tiling: config.clone(),
         });
         
         let kernel_id = match self.runtime.compile(&source, "gemm_kernel", self.backend) {
             Ok(k) => k,
             Err(_) => return false,
         };

         let (grid_dim, block_dim, smem_size) = self.get_launch_params(config);
         
         let args = vec![
             KernelArg::Buffer(self.buffers_in.0),
             KernelArg::Buffer(self.buffers_in.1),
             KernelArg::Buffer(self.buffer_c),
             KernelArg::Int(self.m as i32),
             KernelArg::Int(self.n as i32),
             KernelArg::Int(self.k as i32),
         ];

         if let Err(_) = self.runtime.launch(kernel_id, grid_dim, block_dim, smem_size as u32, args) {
              return false;
         }
         
         self.runtime.synchronize(self.backend).is_ok()
    }

    fn measure(&self, config: &PipelineConfig) -> BenchmarkResult {
        let emitter = crate::emitter::universal::UniversalEmitter::new(self.backend);
        let source = emitter.generate(crate::emitter::traits::UnifiedOpIR {
            op_type: crate::emitter::traits::UnifiedOpType::Gemm { m: self.m, n: self.n, k: self.k },
            precison: "f32".to_string(),
            tiling: config.clone(),
        });

        let kernel_id = match self.runtime.compile(&source, "gemm_kernel", self.backend) {
            Ok(k) => k,
            Err(_) => return BenchmarkResult { tflops: 0.0, latency_ms: 1e9 },
        };

        let (grid_dim, block_dim, smem_size) = self.get_launch_params(config);
        
        let args = vec![
            KernelArg::Buffer(self.buffers_in.0),
            KernelArg::Buffer(self.buffers_in.1),
            KernelArg::Buffer(self.buffer_c),
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
        
        if let Err(e) = self.runtime.synchronize(self.backend) {
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
    
    fn device_info(&self) -> EnvironmentInfo { 
        use crate::runtime::DeviceBackend;
        match self.backend {
            crate::runtime::DeviceBackend::Cuda => {
                let d = self.runtime.get_device(DeviceBackend::Cuda).expect("No CUDA Device");
                let mut driver_ver: i32 = 0;
                unsafe { cudarc::driver::sys::lib().cuDriverGetVersion(&mut driver_ver) };
                let mut nvrtc_major = 0; let mut nvrtc_minor = 0;
                unsafe { cudarc::nvrtc::sys::lib().nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) };
                let sm_major = d.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(0);
                let sm_minor = d.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0);
                EnvironmentInfo {
                    backend: DeviceBackend::Cuda,
                    api_version: format!("{}.{}", nvrtc_major, nvrtc_minor),
                    driver_version: format!("{}.{}", driver_ver / 1000, (driver_ver % 1000) / 10),
                    arch: format!("sm_{}{}", sm_major, sm_minor),
                }
            }
            crate::runtime::DeviceBackend::Rocm => {
                EnvironmentInfo {
                    backend: DeviceBackend::Rocm,
                    api_version: "ROCm 6.0".to_string(), // Detect properly
                    driver_version: "ROCm Driver 6.0".to_string(),
                    arch: "gfx90a".to_string(),
                }
            }
            crate::runtime::DeviceBackend::Metal => {
                EnvironmentInfo {
                    backend: DeviceBackend::Metal,
                    api_version: "Metal 3.0".to_string(),
                    driver_version: "macOS 14".to_string(),
                    arch: "Apple M3".to_string(),
                }
            }
        }
    }
}
