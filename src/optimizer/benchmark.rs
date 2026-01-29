use crate::core::config::{PipelineConfig, MagicNumberStrategy};
use crate::runtime::{RuntimeManager, BufferId, KernelArg, DeviceBackend};
use crate::runtime::ttg_builder::TTGBuilder; 
use std::sync::Arc;

pub struct BenchmarkResult {
    pub tflops: f32,      // Best/Robust score (Mean - k*StdDev)
    pub mean_tflops: f32,
    pub std_dev: f32,
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
        BenchmarkResult {
             tflops, 
             mean_tflops: tflops,
             std_dev: 0.0,
             latency_ms 
        }
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

pub struct NVRTCBenchmark {
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
        let backend = if cfg!(target_os = "macos") {
            crate::runtime::DeviceBackend::Metal
        } else if crate::emitter::rocm_driver::RocmDriverApi::get().is_some() {
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
        let grid_dim = ((self.n + nt - 1) / nt, (self.m + mt - 1) / mt, 1);
        
        let (block_dim, smem_size) = match self.backend {
            crate::runtime::DeviceBackend::Cuda => {
                let nw = config.force_num_warps.unwrap_or(4);
                let block = (nw * 32, 1, 1);
                // CUDA kernel hardcoded to f16 (2 bytes) and padding + 8
                let element_size = 2; 
                let s_a = kt + 8;
                let s_b = nt + 8;
                let smem_pipe = (config.num_stages * mt * s_a + config.num_stages * kt * s_b) * element_size;
                let smem_epi = mt * nt * 4; // float accumulator
                let smem = smem_pipe.max(smem_epi) + 256; 
                (block, smem)
            }
            crate::runtime::DeviceBackend::Rocm => {
                let block = (128, 1, 1);
                let smem = (config.num_stages * mt * kt + config.num_stages * kt * nt) * 2;
                (block, smem)
            }
            crate::runtime::DeviceBackend::Metal => {
                let block = (128, 1, 1); 
                // Buffer multiplier: 1 for single, 2 for double buffer
                let buf_mult = if config.double_buffer { 2 } else { 1 };
                let smem = buf_mult * (mt * kt + kt * nt) * 2;
                (block, smem)
            }
            crate::runtime::DeviceBackend::Cpu => {
                ((1, 1, 1), 0)
            }
        };

        (grid_dim, block_dim, smem_size)
    }
}

impl MicroBenchmark for NVRTCBenchmark {
    fn validate_config(&self, config: &PipelineConfig) -> bool {
         let emitter = crate::emitter::universal::UniversalEmitter::new(self.backend);
         let ir = crate::emitter::traits::UnifiedOpIR {
             op_type: crate::emitter::traits::UnifiedOpType::Gemm { m: self.m, n: self.n, k: self.k, batch: 1 },
             precison: "f16".to_string(),
             tiling: config.clone(),
             conv_magic_strategy: None,
         };
         let source = emitter.generate(ir.clone());
         
          let kernel_name = if let crate::emitter::traits::UnifiedOpType::Gemm { .. } = ir.op_type { "unified_gemm_kernel" } else { "gemm_mma_kernel" };
          
          let kernel_id = match self.runtime.compile(&source, kernel_name, self.backend) {
              Ok(k) => k,
              Err(e) => {
                  eprintln!("[Benchmark] Validation Compile Failed: {}", e);
                  return false;
              }
          };


         let (mut grid_dim, block_dim, smem_size) = self.get_launch_params(config);
         
         let mut args = vec![
             KernelArg::Buffer(self.buffers_in.0),
             KernelArg::Buffer(self.buffers_in.1),
             KernelArg::Buffer(self.buffer_c),
             KernelArg::Int(self.m as i32),
             KernelArg::Int(self.n as i32),
             KernelArg::Int(self.k as i32),
         ];

         // TTG Handling
         let mut _l1_buf = None;
         let mut _l2_buf = None;
         if config.ttg_enabled {
             let layout = TTGBuilder::from_dense(self.m, self.n, self.k, config.m_tile, config.n_tile, config.k_tile);
             
             let l1_bytes: Vec<u8> = layout.l1_map.iter().flat_map(|x| x.to_ne_bytes().to_vec()).collect();
             // TileMetadata(20 bytes) iter
             // Using unsafe to cast struct to bytes or manual mapping
             let l2_bytes: Vec<u8> = layout.l2_table.iter().flat_map(|meta| {
                 let mut b = Vec::with_capacity(20);
                 b.extend_from_slice(&meta.region_m.to_ne_bytes());
                 b.extend_from_slice(&meta.region_n.to_ne_bytes());
                 b.extend_from_slice(&meta.k_start.to_ne_bytes());
                 b.extend_from_slice(&meta.k_end.to_ne_bytes());
                 b.extend_from_slice(&meta.role.to_ne_bytes());
                 b
             }).collect();

             let l1 = match self.runtime.alloc(l1_bytes.len(), self.backend) {
                 Ok(x) => x,
                 Err(_) => return false,
             };
             let l2 = match self.runtime.alloc(l2_bytes.len(), self.backend) {
                 Ok(x) => x,
                 Err(_) => return false,
             };
             if self.runtime.copy_to_device(l1, &l1_bytes).is_err() { return false; }
             if self.runtime.copy_to_device(l2, &l2_bytes).is_err() { return false; }
             
             args.push(KernelArg::Buffer(l1));
             args.push(KernelArg::Buffer(l2));
             
             // Update Grid
             grid_dim = (layout.num_active_tiles, 1, 1);
             
             _l1_buf = Some(l1);
             _l2_buf = Some(l2);
         }

         if let Err(e) = self.runtime.launch(kernel_id, grid_dim, block_dim, smem_size as u32, args) {
             eprintln!("[Benchmark] Validation Launch Failed: {}", e);
             return false;
         }
         
         self.runtime.synchronize();
         true
    }

    fn measure(&self, config: &PipelineConfig) -> BenchmarkResult {
        let emitter = crate::emitter::universal::UniversalEmitter::new(self.backend);
        let ir = crate::emitter::traits::UnifiedOpIR {
            op_type: crate::emitter::traits::UnifiedOpType::Gemm { m: self.m, n: self.n, k: self.k, batch: 1 },
            precison: "f16".to_string(),
            tiling: config.clone(),
            conv_magic_strategy: None,
        };
        let source = emitter.generate(ir.clone());

        let kernel_name = if let crate::emitter::traits::UnifiedOpType::Gemm { .. } = ir.op_type { "unified_gemm_kernel" } else { "gemm_mma_kernel" };

        let kernel_id = match self.runtime.compile(&source, kernel_name, self.backend) {
            Ok(k) => k,
            Err(_) => return BenchmarkResult { tflops: 0.0, mean_tflops: 0.0, std_dev: 0.0, latency_ms: 1e9 },
        };

        let (mut grid_dim, block_dim, smem_size) = self.get_launch_params(config);
        
        let mut args = vec![
            KernelArg::Buffer(self.buffers_in.0),
            KernelArg::Buffer(self.buffers_in.1),
            KernelArg::Buffer(self.buffer_c),
            KernelArg::Int(self.m as i32),
            KernelArg::Int(self.n as i32),
            KernelArg::Int(self.k as i32),
        ];

         // TTG Handling
         let mut _l1_buf = None;
         let mut _l2_buf = None;
         if config.ttg_enabled {
             let layout = TTGBuilder::from_dense(self.m, self.n, self.k, config.m_tile, config.n_tile, config.k_tile);
             
             let l1_bytes: Vec<u8> = layout.l1_map.iter().flat_map(|x| x.to_ne_bytes().to_vec()).collect();
             let l2_bytes: Vec<u8> = layout.l2_table.iter().flat_map(|meta| {
                 let mut b = Vec::with_capacity(20);
                 b.extend_from_slice(&meta.region_m.to_ne_bytes());
                 b.extend_from_slice(&meta.region_n.to_ne_bytes());
                 b.extend_from_slice(&meta.k_start.to_ne_bytes());
                 b.extend_from_slice(&meta.k_end.to_ne_bytes());
                 b.extend_from_slice(&meta.role.to_ne_bytes());
                 b
             }).collect();

             // Unwrap unsafe/expect in measure block for simplicity or handle gracefully
             let l1 = self.runtime.alloc(l1_bytes.len(), self.backend).expect("TTG L1 Alloc");
             let l2 = self.runtime.alloc(l2_bytes.len(), self.backend).expect("TTG L2 Alloc");
             self.runtime.copy_to_device(l1, &l1_bytes).expect("TTG L1 Copy");
             self.runtime.copy_to_device(l2, &l2_bytes).expect("TTG L2 Copy");
             
             args.push(KernelArg::Buffer(l1));
             args.push(KernelArg::Buffer(l2));
             
             grid_dim = (layout.num_active_tiles, 1, 1);
             
             _l1_buf = Some(l1);
             _l2_buf = Some(l2);
         }

        // Warmup
        for _ in 0..10 {
            let _ = self.runtime.launch(kernel_id, grid_dim, block_dim, smem_size as u32, args.clone());
        }
        self.runtime.synchronize();

        let start = std::time::Instant::now();
        let iterations = 10;
        for i in 0..iterations {
             if let Err(e) = self.runtime.launch(kernel_id, grid_dim, block_dim, smem_size as u32, args.clone()) {
                  eprintln!("[Tracea] ⚠️  Launch Failed on iteration {}: {:?}", i, e);
                  return BenchmarkResult { tflops: 0.0, mean_tflops: 0.0, std_dev: 0.0, latency_ms: 1e9 };
             }
        }
        
        self.runtime.synchronize();
        let dur = start.elapsed().as_secs_f32() / iterations as f32;
        let latency_ms = dur * 1000.0;
        let tflops = (2.0 * self.m as f32 * self.n as f32 * self.k as f32) / (dur * 1e12);
        
        BenchmarkResult {
             tflops, 
             mean_tflops: tflops,
             std_dev: 0.0,
             latency_ms 
        }
    }
    
    fn m(&self) -> u32 { self.m }
    fn n(&self) -> u32 { self.n }
    fn k(&self) -> u32 { self.k }
    
    fn device_info(&self) -> EnvironmentInfo { 
        use crate::runtime::DeviceBackend;
        match self.backend {
            crate::runtime::DeviceBackend::Cuda => {
                let d_handle = self.runtime.get_device(crate::runtime::DeviceBackend::Cuda).expect("No CUDA Device");
                let d = d_handle.cuda_dev.as_ref().expect("No CUDA Device");
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
            crate::runtime::DeviceBackend::Cpu => {
                EnvironmentInfo {
                    backend: DeviceBackend::Cpu,
                    api_version: "0.1.0".to_string(),
                    driver_version: "0.1.0".to_string(),
                    arch: "x86_64".to_string(),
                }
            }
        }
    }
}

// ============================================================================
// Conv2d Benchmark Infrastructure (Phase 5)
// ============================================================================

/// Convolution problem specification
#[derive(Debug, Clone)]
pub struct Conv2dProblem {
    pub name: String,
    pub batch: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub pad: usize,
    pub dilation: usize,
}

impl Conv2dProblem {
    pub fn new(name: &str, batch: usize, h: usize, w: usize, c_in: usize, c_out: usize, 
               r: usize, s: usize, stride: usize, pad: usize, dilation: usize) -> Self {
        Self {
            name: name.to_string(),
            batch, h_in: h, w_in: w, c_in, c_out,
            kernel_h: r, kernel_w: s,
            stride, pad, dilation,
        }
    }

    pub fn h_out(&self) -> usize {
        (self.h_in + 2 * self.pad - self.kernel_h) / self.stride + 1
    }

    pub fn w_out(&self) -> usize {
        (self.w_in + 2 * self.pad - self.kernel_w) / self.stride + 1
    }

    /// GEMM dimensions for implicit GEMM
    pub fn gemm_dims(&self) -> (usize, usize, usize) {
        let m = self.batch * self.h_out() * self.w_out();
        let n = self.c_out;
        let k = self.c_in * self.kernel_h * self.kernel_w;
        (m, n, k)
    }

    /// Total FLOPs for this convolution
    pub fn flops(&self) -> f64 {
        let (m, n, k) = self.gemm_dims();
        2.0 * m as f64 * n as f64 * k as f64
    }

    // ========== ResNet-50 Standard Layers ==========
    
    /// 3x3 conv in residual block (inference, batch=1)
    pub fn resnet50_conv3x3_64() -> Self {
        Self::new("ResNet50-Conv3x3-64-B1", 1, 56, 56, 64, 64, 3, 3, 1, 1, 1)
    }

    /// 1x1 conv in bottleneck
    pub fn resnet50_conv1x1_256() -> Self {
        Self::new("ResNet50-Conv1x1-256-B1", 1, 56, 56, 64, 256, 1, 1, 1, 0, 1)
    }

    /// 7x7 stem layer with stride=2 (stress test)
    pub fn resnet50_stem_7x7() -> Self {
        Self::new("ResNet50-Stem-7x7-B1", 1, 224, 224, 3, 64, 7, 7, 2, 3, 1)
    }

    // ========== Training Scenarios (Large Batch) ==========
    
    /// 3x3 conv with batch=32 (training)
    pub fn resnet50_conv3x3_64_batch32() -> Self {
        Self::new("ResNet50-Conv3x3-64-B32", 32, 56, 56, 64, 64, 3, 3, 1, 1, 1)
    }

    /// 3x3 conv with batch=64 (training, Tensor Core optimal)
    pub fn resnet50_conv3x3_64_batch64() -> Self {
        Self::new("ResNet50-Conv3x3-64-B64", 64, 56, 56, 64, 64, 3, 3, 1, 1, 1)
    }

    /// Full benchmark suite
    pub fn resnet50_suite() -> Vec<Self> {
        vec![
            Self::resnet50_conv3x3_64(),
            Self::resnet50_conv1x1_256(),
            Self::resnet50_stem_7x7(),
            Self::resnet50_conv3x3_64_batch32(),
            Self::resnet50_conv3x3_64_batch64(),
        ]
    }
}

// ============================================================================
// ConvTranspose2d Benchmark Infrastructure (Phase F)
// ============================================================================

/// ConvTranspose2d (Deconvolution) problem specification
#[derive(Debug, Clone)]
pub struct ConvTranspose2dProblem {
    pub name: String,
    pub batch: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub pad: usize,
    pub output_padding: usize,
}

impl ConvTranspose2dProblem {
    pub fn new(name: &str, batch: usize, h: usize, w: usize, c_in: usize, c_out: usize,
               r: usize, s: usize, stride: usize, pad: usize, output_padding: usize) -> Self {
        Self {
            name: name.to_string(),
            batch, h_in: h, w_in: w, c_in, c_out,
            kernel_h: r, kernel_w: s,
            stride, pad, output_padding,
        }
    }

    pub fn h_out(&self) -> usize {
        (self.h_in - 1) * self.stride - 2 * self.pad + self.kernel_h + self.output_padding
    }

    pub fn w_out(&self) -> usize {
        (self.w_in - 1) * self.stride - 2 * self.pad + self.kernel_w + self.output_padding
    }

    /// GEMM dimensions for implicit GEMM (transposed)
    pub fn gemm_dims(&self) -> (usize, usize, usize) {
        let m = self.batch * self.h_out() * self.w_out();
        let n = self.c_out;
        let k = self.c_in * self.kernel_h * self.kernel_w;
        (m, n, k)
    }

    /// Total FLOPs for this transposed convolution
    pub fn flops(&self) -> f64 {
        let (m, n, k) = self.gemm_dims();
        2.0 * m as f64 * n as f64 * k as f64
    }

    // ========== VAE Decoder Standard Layers ==========
    
    /// 4x4 conv_transpose in VAE decoder (upsampling by 2x)
    pub fn vae_conv_transpose_4x4() -> Self {
        Self::new("VAE-ConvT-4x4-B1", 1, 32, 32, 512, 256, 4, 4, 2, 1, 0)
    }

    /// 4x4 conv_transpose with batch=8 (training)
    pub fn vae_conv_transpose_4x4_batch8() -> Self {
        Self::new("VAE-ConvT-4x4-B8", 8, 32, 32, 512, 256, 4, 4, 2, 1, 0)
    }

    /// Full benchmark suite
    pub fn vae_suite() -> Vec<Self> {
        vec![
            Self::vae_conv_transpose_4x4(),
            Self::vae_conv_transpose_4x4_batch8(),
        ]
    }
}


// MagicNumberStrategy moved to src/core/config.rs

/// Conv2d-specific configuration (extends PipelineConfig)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConvConfig {
    pub base: PipelineConfig,
    pub use_tensor_core: bool,
    pub use_nhwc: bool,
    pub magic_strategy: MagicNumberStrategy,
}

impl ConvConfig {
    pub fn default_for_problem(problem: &Conv2dProblem) -> Self {
        let (m, n, _k) = problem.gemm_dims();
        
        // Heuristic tile selection
        let m_tile = if m >= 128 { 128 } else { 64 };
        let n_tile = if n >= 128 { 128 } else { 64 };
        let k_tile = 32;
        
        let mut base = PipelineConfig::new(2, m_tile, n_tile, k_tile);
        base.instruction = crate::core::config::SpecializedInstruction::CudaMMA;
        base.layout_policy = Some(crate::core::config::LayoutPolicy::NHWC);
        base.force_num_warps = Some(4);
        
        // Select optimal magic number strategy
        let hw_out = problem.h_out() * problem.w_out();
        let magic_strategy = MagicNumberStrategy::select_for(hw_out);
        
        Self {
            base,
            use_tensor_core: true,
            use_nhwc: true,
            magic_strategy,
        }
    }
}

/// Trait for Conv2d-specific benchmarking
pub trait Conv2dBenchmark {
    fn measure(&self, config: &ConvConfig) -> BenchmarkResult;
    fn validate_config(&self, config: &ConvConfig) -> bool;
    fn problem(&self) -> &Conv2dProblem;
    fn device_info(&self) -> EnvironmentInfo;
}

/// NVRTC-based Conv2d benchmark using implicit GEMM
pub struct NVRTCConvBenchmark {
    pub runtime: Arc<RuntimeManager>,
    pub backend: DeviceBackend,
    pub problem: Conv2dProblem,
    pub d_input: BufferId,
    pub d_weight: BufferId,
    pub d_output: BufferId,
}

impl NVRTCConvBenchmark {
    pub fn new(runtime: Arc<RuntimeManager>, problem: Conv2dProblem) -> Self {
        let backend = if cfg!(target_os = "macos") {
            crate::runtime::DeviceBackend::Metal
        } else {
            crate::runtime::DeviceBackend::Cuda
        };
        
        let input_size = problem.batch * problem.h_in * problem.w_in * problem.c_in;
        let weight_size = problem.c_out * problem.kernel_h * problem.kernel_w * problem.c_in;
        let output_size = problem.batch * problem.h_out() * problem.w_out() * problem.c_out;
        
        let d_input = runtime.alloc_u16(input_size, backend).expect("Alloc Input Failed");
        let d_weight = runtime.alloc_u16(weight_size, backend).expect("Alloc Weight Failed");
        let d_output = runtime.alloc_u16(output_size, backend).expect("Alloc Output Failed");
        
        // Initialize with 1.0 (0x3C00 in half precision)
        let h_input = vec![0x3C00u16; input_size];
        let h_weight = vec![0x3C00u16; weight_size];
        runtime.copy_to_device(d_input, &h_input).expect("Copy Input Failed");
        runtime.copy_to_device(d_weight, &h_weight).expect("Copy Weight Failed");
        
        Self {
            runtime,
            backend,
            problem,
            d_input,
            d_weight,
            d_output,
        }
    }

    fn get_launch_params(&self, config: &ConvConfig) -> ((u32, u32, u32), (u32, u32, u32), u32) {
        let (m_gemm, n_gemm, _k_gemm) = self.problem.gemm_dims();
        let mt = config.base.m_tile as usize;
        let nt = config.base.n_tile as usize;
        let kt = config.base.k_tile as usize;
        let num_warps = config.base.force_num_warps.unwrap_or(4) as usize;
        let stages = config.base.num_stages as usize;
        
        // Metal kernel uses bid.y for M dimension, bid.x for N dimension
        let grid_m = (m_gemm + mt - 1) / mt;  // Y dimension in Metal
        let grid_n = (n_gemm + nt - 1) / nt;  // X dimension in Metal
        
        // Shared memory logic must match emitter in src/emitter/conv.rs
        let a_stride = kt + 8;
        let b_stride = nt + 8;
        let smem_a_bytes = mt * a_stride * 2;
        let smem_b_bytes = kt * b_stride * 2;

        let a_smem_offset = 128; // Header
        let smem_compute = a_smem_offset + (smem_a_bytes + smem_b_bytes) * stages;
        let smem_epilogue = mt * nt * 4;
        
        let smem = (std::cmp::max(smem_compute, smem_epilogue) as u32 + 255) & !255;
        
        // Grid: (X=N, Y=M, Z=1) to match Metal kernel's bid.x/bid.y usage
        ((grid_n as u32, grid_m as u32, 1), 
         ((num_warps * 32) as u32, 1, 1), 
         smem)
    }
}

impl Conv2dBenchmark for NVRTCConvBenchmark {
    fn validate_config(&self, config: &ConvConfig) -> bool {
        use crate::emitter::universal::UniversalEmitter;
        use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
        use crate::core::config::LayoutPolicy;
        
        let backend = self.backend;
        let emitter = UniversalEmitter::new(backend);
        
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Conv2d {
                n: self.problem.batch,
                h: self.problem.h_in,
                w: self.problem.w_in,
                c: self.problem.c_in,
                k: self.problem.c_out,
                r: self.problem.kernel_h,
                s: self.problem.kernel_w,
                stride: self.problem.stride,
                pad: self.problem.pad,
                dilation: self.problem.dilation,
                layout: LayoutPolicy::NHWC,
            },
            precison: "f16".to_string(),
            tiling: config.base.clone(),
            conv_magic_strategy: Some(config.magic_strategy),
        };
        
        let source = emitter.generate(ir);
        
        match self.runtime.compile(&source, "conv2d_implicit_gemm", backend) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("[Conv2dBenchmark] Compilation failed: {:?}", e);
                false
            }
        }
    }

    fn measure(&self, config: &ConvConfig) -> BenchmarkResult {
        use crate::emitter::universal::UniversalEmitter;
        use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType};
        use crate::core::config::LayoutPolicy;
        
        let backend = self.backend;
        let emitter = UniversalEmitter::new(backend);
        
        let ir = UnifiedOpIR {
            op_type: UnifiedOpType::Conv2d {
                n: self.problem.batch,
                h: self.problem.h_in,
                w: self.problem.w_in,
                c: self.problem.c_in,
                k: self.problem.c_out,
                r: self.problem.kernel_h,
                s: self.problem.kernel_w,
                stride: self.problem.stride,
                pad: self.problem.pad,
                dilation: self.problem.dilation,
                layout: LayoutPolicy::NHWC,
            },
            precison: "f16".to_string(),
            tiling: config.base.clone(),
            conv_magic_strategy: Some(config.magic_strategy),
        };
        
        let source = emitter.generate(ir);
        
        let kernel_id = match self.runtime.compile(&source, "conv2d_implicit_gemm", backend) {
            Ok(k) => k,
            Err(_) => return BenchmarkResult { tflops: 0.0, mean_tflops: 0.0, std_dev: 0.0, latency_ms: 1e9 },
        };
        
        let (grid, block, smem) = self.get_launch_params(config);
        
        // Build ConvParams structure matching Metal kernel's expectation
        // struct ConvParams { uint batch, h_in, w_in, c_in, k_out, h_out, w_out, r_sz, s_sz, stride, pad, dilation; }
        let h_out = self.problem.h_out();
        let w_out = self.problem.w_out();
        let conv_params: Vec<u8> = [
            self.problem.batch as u32,
            self.problem.h_in as u32,
            self.problem.w_in as u32,
            self.problem.c_in as u32,
            self.problem.c_out as u32,
            h_out as u32,
            w_out as u32,
            self.problem.kernel_h as u32,
            self.problem.kernel_w as u32,
            self.problem.stride as u32,
            self.problem.pad as u32,
            self.problem.dilation as u32,
        ].iter().flat_map(|v| v.to_le_bytes()).collect();
        
        let args = vec![
            KernelArg::Buffer(self.d_input),
            KernelArg::Buffer(self.d_weight),
            KernelArg::Buffer(self.d_output),
            KernelArg::Bytes(conv_params.clone()),
        ];
        
        // Pass 1: Warmup + Rapid 1st Measurement
        for _ in 0..2 {
            if let Err(_) = self.runtime.launch(kernel_id, grid, block, smem, args.clone()) {
                return BenchmarkResult { tflops: 0.0, mean_tflops: 0.0, std_dev: 0.0, latency_ms: 1e9 };
            }
        }
        self.runtime.synchronize();
        
        let start = std::time::Instant::now();
        if let Err(_) = self.runtime.launch(kernel_id, grid, block, smem, args.clone()) {
            return BenchmarkResult { tflops: 0.0, mean_tflops: 0.0, std_dev: 0.0, latency_ms: 1e9 };
        }
        self.runtime.synchronize();
        let first_dur = start.elapsed().as_secs_f64();
        let first_tflops = (self.problem.flops() / first_dur / 1e12) as f32;

        let mut tflops_samples = vec![first_tflops];

        // Adaptive Sampling: Only measure more if it's promising (> 5 TFLOPS for now)
        // In v1.1, this threshold should ideally be based on the current best.
        let is_promising = first_tflops > 3.0; 

        if is_promising {
            let iterations = 5;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                if let Err(_) = self.runtime.launch(kernel_id, grid, block, smem, args.clone()) {
                    break;
                }
            }
            self.runtime.synchronize();
            let dur = start.elapsed().as_secs_f64() / iterations as f64;
            let refined_tflops = (self.problem.flops() / dur / 1e12) as f32;
            
            // For simplicity, we use the refined average and a small mock variance for now
            // if we only did one more batch. Practical implementation would individualize samples.
            tflops_samples.push(refined_tflops);
        }

        let mean = tflops_samples.iter().sum::<f32>() / tflops_samples.len() as f32;
        let std_dev = if tflops_samples.len() > 1 {
            let var = tflops_samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / tflops_samples.len() as f32;
            var.sqrt()
        } else {
            0.0
        };

        BenchmarkResult { 
            tflops: mean, 
            mean_tflops: mean,
            std_dev,
            latency_ms: (self.problem.flops() / (mean as f64 * 1e12 + 1e-6) * 1000.0) as f32
        }
    }

    fn problem(&self) -> &Conv2dProblem {
        &self.problem
    }

    fn device_info(&self) -> EnvironmentInfo {
        let d = self.runtime.get_device(crate::runtime::DeviceBackend::Cuda)
            .expect("No CUDA Device");
        let dev = d.cuda_dev.as_ref().expect("No CUDA Device");
        
        let mut driver_ver: i32 = 0;
        unsafe { cudarc::driver::sys::lib().cuDriverGetVersion(&mut driver_ver) };
        
        let sm_major = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(0);
        let sm_minor = dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0);
        
        EnvironmentInfo {
            backend: crate::runtime::DeviceBackend::Cuda,
            api_version: format!("{}.{}", driver_ver / 1000, (driver_ver % 1000) / 10),
            driver_version: format!("{}.{}", driver_ver / 1000, (driver_ver % 1000) / 10),
            arch: format!("sm_{}{}", sm_major, sm_minor),
        }
    }
}

// ============================================================================
// FlashAttention Benchmark Infrastructure (Phase 6)
// ============================================================================

#[derive(Debug, Clone)]
pub struct FlashAttentionProblem {
    pub name: String,
    pub b: usize,
    pub h: usize,
    pub s: usize,
    pub d: usize,
    pub is_causal: bool,
}

impl FlashAttentionProblem {
    pub fn new(b: usize, h: usize, s: usize, d: usize, is_causal: bool) -> Self {
        Self {
            name: format!("FA2-B{}-H{}-S{}-D{}-{}", b, h, s, d, if is_causal {"Causal"} else {"Bidirectional"}),
            b, h, s, d, is_causal
        }
    }
    
    pub fn flops(&self) -> f64 {
        // 4 * B * H * S * S * D (Approx)
        4.0 * self.b as f64 * self.h as f64 * self.s as f64 * self.s as f64 * self.d as f64
    }
}

pub struct FlashAttentionBenchmark {
    pub runtime: Arc<RuntimeManager>,
    pub problem: FlashAttentionProblem,
    pub d_q: BufferId,
    pub d_k: BufferId,
    pub d_v: BufferId,
    pub d_o: BufferId,
}

impl FlashAttentionBenchmark {
    pub fn new(runtime: Arc<RuntimeManager>, problem: FlashAttentionProblem) -> Self {
        let backend = if cfg!(target_os = "macos") {
            crate::runtime::DeviceBackend::Metal
        } else {
            crate::runtime::DeviceBackend::Cuda
        };
        let size = problem.b * problem.h * problem.s * problem.d; // in elements
        
        // 256MB allocations logic check
        let d_q = runtime.alloc_u16(size, backend).expect("Alloc Q Failed");
        let d_k = runtime.alloc_u16(size, backend).expect("Alloc K Failed");
        let d_v = runtime.alloc_u16(size, backend).expect("Alloc V Failed");
        let d_o = runtime.alloc_u16(size, backend).expect("Alloc O Failed");
        
        // Initialize (Dummy)
        let host_data = vec![0x3C00u16; size];
        runtime.copy_to_device(d_q, &host_data).expect("Copy Q Failed");
        runtime.copy_to_device(d_k, &host_data).expect("Copy K Failed");
        runtime.copy_to_device(d_v, &host_data).expect("Copy V Failed");
        
        Self { runtime, problem, d_q, d_k, d_v, d_o }
    }
}

impl MicroBenchmark for FlashAttentionBenchmark {
    fn m(&self) -> u32 { self.problem.s as u32 } 
    fn n(&self) -> u32 { self.problem.d as u32 } 
    fn k(&self) -> u32 { self.problem.d as u32 }

    fn validate_config(&self, config: &PipelineConfig) -> bool {
        let backend = if cfg!(target_os = "macos") {
            crate::runtime::DeviceBackend::Metal
        } else {
            crate::runtime::DeviceBackend::Cuda
        };
        let emitter = crate::emitter::universal::UniversalEmitter::new(backend);
        let source = emitter.generate(crate::emitter::traits::UnifiedOpIR {
            op_type: crate::emitter::traits::UnifiedOpType::FusedAttention {
                b: self.problem.b as u32,
                s: self.problem.s as u32,
                d: self.problem.d as u32,
                h: self.problem.h as u32,
                dh: self.problem.d as u32, // Simplified dh estimation
                causal: self.problem.is_causal,
            },
            precison: "f16".to_string(),
            tiling: config.clone(),
            conv_magic_strategy: None,
        });
        
        match self.runtime.compile(&source, "unified_attention_kernel", backend) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    fn measure(&self, config: &PipelineConfig) -> BenchmarkResult {
        let backend = if cfg!(target_os = "macos") {
            crate::runtime::DeviceBackend::Metal
        } else {
            crate::runtime::DeviceBackend::Cuda
        };
        let emitter = crate::emitter::universal::UniversalEmitter::new(backend);
        let source = emitter.generate(crate::emitter::traits::UnifiedOpIR {
            op_type: crate::emitter::traits::UnifiedOpType::FusedAttention {
                b: self.problem.b as u32,
                s: self.problem.s as u32,
                d: self.problem.d as u32,
                h: self.problem.h as u32,
                dh: self.problem.d as u32,
                causal: self.problem.is_causal,
            },
            precison: "f16".to_string(),
            tiling: config.clone(),
            conv_magic_strategy: None,
        });

        let kernel_id = match self.runtime.compile(&source, "unified_attention_kernel", backend) {
            Ok(k) => k,
            Err(_) => return BenchmarkResult { tflops: 0.0, mean_tflops: 0.0, std_dev: 0.0, latency_ms: 1e9 },
        };
        
        // Launch Params
        let mt = config.m_tile as usize;
        let nt = config.n_tile as usize; 
        
        let ticks = (self.problem.s + mt - 1) / mt;
        let grid = (ticks as u32, self.problem.h as u32, self.problem.b as u32);
        
        let smem = 4096; // Baseline for simplified unified kernel
        
        // Args match unified_attention_kernel: Q, K, V, O, S, D, scale
        let args = vec![
            KernelArg::Buffer(self.d_q),
            KernelArg::Buffer(self.d_k),
            KernelArg::Buffer(self.d_v),
            KernelArg::Buffer(self.d_o),
            KernelArg::Int(self.problem.s as i32),
            KernelArg::Int(self.problem.d as i32),
            KernelArg::Float(1.0 / (self.problem.d as f32).sqrt()),
        ];
        
        let nw = config.force_num_warps.unwrap_or(4);
        let block = (nw * 32, 1, 1);
        
        // Execution
        self.runtime.synchronize();
        let start = std::time::Instant::now();
        let iterations = 100;
        
        for _ in 0..iterations {
             if let Err(_) = self.runtime.launch(kernel_id, grid, block, smem, args.clone()) {
                 return BenchmarkResult { tflops: 0.0, mean_tflops: 0.0, std_dev: 0.0, latency_ms: 1e9 };
             }
        }
        self.runtime.synchronize();
        
        let dur = start.elapsed().as_secs_f64() / iterations as f64;
        let tflops = (self.problem.flops() / dur / 1e12) as f32;
        
        BenchmarkResult {
            tflops,
            mean_tflops: tflops,
            std_dev: 0.0,
            latency_ms: (dur * 1000.0) as f32,
        }
    }
    
    fn device_info(&self) -> EnvironmentInfo { 
        let backend = if cfg!(target_os = "macos") {
            crate::runtime::DeviceBackend::Metal
        } else {
            crate::runtime::DeviceBackend::Cuda
        };
        EnvironmentInfo {
            backend,
            api_version: "1.0".to_string(),
            driver_version: "Unknown".to_string(),
            arch: "M1/Ampere".to_string(),
        }
    }
}
