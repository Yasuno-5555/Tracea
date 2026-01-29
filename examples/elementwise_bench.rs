use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::emitter::universal::UniversalEmitter;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use tracea::core::op::{ElementwiseType};
use tracea::core::config::{PipelineConfig, SpecializedInstruction, SwizzleMode, QuantizationMode, LayoutPolicy};

fn main() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).unwrap();
    let n = 1024 * 1024 * 64; // 64M floats = 256MB
    let size_bytes = n * 4;

    println!("Allocating buffers ({} MB)...", size_bytes / 1024 / 1024);
    let a_buf = runtime.alloc_f32(n, DeviceBackend::Cuda).unwrap();
    let b_buf = runtime.alloc_f32(n, DeviceBackend::Cuda).unwrap();
    let c_buf = runtime.alloc_f32(n, DeviceBackend::Cuda).unwrap();

    // Init Data
    let a_data = vec![1.0f32; n];
    let b_data = vec![2.0f32; n];
    runtime.copy_to_device(a_buf, &a_data).unwrap();
    runtime.copy_to_device(b_buf, &b_data).unwrap();

    // ADD Test
    println!("Generating ADD Kernel...");
    let ir_add = UnifiedOpIR {
        op_type: UnifiedOpType::Elementwise { op_type: ElementwiseType::Add, n },
        precison: "f32".to_string(),
        tiling: dummy_config(),
    };
    
    let source = UniversalEmitter::new(DeviceBackend::Cuda).generate(ir_add);
    // println!("{}", source); // Debug
    let kid_add = runtime.compile(&source, "elementwise_add", DeviceBackend::Cuda).unwrap();

    let grid = ((n as u32 + 255) / 256, 1, 1);
    let block = (256, 1, 1);
    
    println!("Launching ADD...");
    let args = vec![
        KernelArg::Buffer(a_buf),
        KernelArg::Buffer(b_buf),
        KernelArg::Buffer(c_buf),
        KernelArg::Int(n as i32),
    ];
    
    let start = std::time::Instant::now();
    runtime.launch(kid_add, grid, block, 0, args).unwrap();
    runtime.synchronize();
    let duration = start.elapsed();
    
    println!("ADD Time: {:?}", duration);
    let bw = (3.0 * size_bytes as f64) / duration.as_secs_f64() / 1e9;
    println!("ADD Bandwidth: {:.2} GB/s", bw);

    // Verify
    let mut c_host = vec![0.0f32; n];
    runtime.copy_from_device(c_buf, &mut c_host).unwrap();
    if (c_host[0] - 3.0).abs() < 1e-5 && (c_host[n-1] - 3.0).abs() < 1e-5 {
        println!("ADD Correctness: PASS");
    } else {
         println!("ADD Correctness: FAIL (Expected 3.0, got {})", c_host[0]);
    }

    // GELU Test
    println!("Generating GELU Kernel...");
    let ir_gelu = UnifiedOpIR {
        op_type: UnifiedOpType::Elementwise { op_type: ElementwiseType::Gelu, n },
        precison: "f32".to_string(),
        tiling: dummy_config(),
    };
    let source = UniversalEmitter::new(DeviceBackend::Cuda).generate(ir_gelu);
    let kid_gelu = runtime.compile(&source, "elementwise_gelu", DeviceBackend::Cuda).unwrap();
    println!("Launching GELU...");
    let args_gelu = vec![
        KernelArg::Buffer(a_buf), // Input
        KernelArg::Buffer(c_buf), // Output reusing C
        KernelArg::Int(n as i32),
    ];
    runtime.launch(kid_gelu, grid, block, 0, args_gelu).unwrap();
    runtime.synchronize();

    let mut gelu_host = vec![0.0f32; n];
    runtime.copy_from_device(c_buf, &mut gelu_host).unwrap();
    // GELU(1.0) approx 0.8413
    if (gelu_host[0] - 0.8413).abs() < 1e-3 {
        println!("GELU Correctness: PASS (Input 1.0 -> {:.4})", gelu_host[0]);
    } else {
        println!("GELU Correctness: FAIL (Expected 0.8413, got {})", gelu_host[0]);
    }
}

fn dummy_config() -> PipelineConfig {
    PipelineConfig {
        num_stages: 2, m_tile: 16, n_tile: 16, k_tile: 16,
        instruction: SpecializedInstruction::None,
        swizzle_mode: SwizzleMode::None,
        intrinsic_shape: IntrinsicShape::None,
        vectorize_epilogue: true,
        ttg_enabled: false,
        attention_variant: Default::default(),
        force_num_warps: None,
    }
}
