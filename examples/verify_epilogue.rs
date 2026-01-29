use half::f16;
use ndarray::{Array2, Array1};
use tracea::core::config::{PipelineConfig, SpecializedInstruction};
use tracea::core::op::EpilogueOp;
use tracea::emitter::cuda::CUDAEmitter;
use tracea::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};

fn cpu_gemm_bias_relu(a: &Array2<f32>, b: &Array2<f32>, bias: &Array1<f32>) -> Array2<f32> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    assert_eq!(k, k2);
    let mut c = a.dot(b);
    for j in 0..n {
        let b_val = bias[j];
        for i in 0..m {
            let val = c[[i, j]] + b_val;
            c[[i, j]] = if val > 0.0 { val } else { 0.0 };
        }
    }
    c
}

fn main() {
    println!("=== Epilogue Fusion Verification (GEMM + Bias + ReLU) ===");
    let manager = RuntimeManager::init(None).expect("Failed to init");

    let m = 128;
    let n = 128;
    let k = 128;
    
    let a = Array2::<f32>::from_shape_fn((m, k), |_| rand::random::<f32>() * 0.1);
    let b = Array2::<f32>::from_shape_fn((k, n), |_| rand::random::<f32>() * 0.1);
    let bias = Array1::<f32>::from_shape_fn(n, |_| rand::random::<f32>() * 0.1);
    
    // CPU Reference
    println!("Computing CPU Reference...");
    let ref_c = cpu_gemm_bias_relu(&a, &b, &bias);

    // GPU Setup
    let mut config = PipelineConfig::new(2, 64, 64, 32);
    config.instruction = SpecializedInstruction::CudaMMA;
    config.epilogue = vec![
        EpilogueOp::BiasAdd { bias_ptr: 0 }, // indices are symbolic here, logic uses valid index
        EpilogueOp::ReLU,
    ];

    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm { m: m as u32, n: n as u32, k: k as u32 },
        tiling: config.clone(),
        precison: "f16".to_string(),
        conv_magic_strategy: None,
    };

    let mut emitter = CUDAEmitter::new();
    let source = emitter.generate_from_ir(&ir);
    let kernel_name = "gemm_mma_kernel"; 
    // Note: generate_gemm hardcodes "gemm_mma_kernel". 
    // If we run multiple tests we might need unique names, but for this example it's fine.
    
    println!("Compiling Kernel...");
    let kernel_id = manager.compile(&source, kernel_name, DeviceBackend::Cuda).expect("Compile failed");

    let a_gpu = manager.alloc_u16(m * k, DeviceBackend::Cuda).expect("Alloc A");
    let b_gpu = manager.alloc_u16(k * n, DeviceBackend::Cuda).expect("Alloc B");
    let c_gpu = manager.alloc_u16(m * n, DeviceBackend::Cuda).expect("Alloc C");
    let bias_gpu = manager.alloc_f32(n, DeviceBackend::Cuda).expect("Alloc Bias"); // Bias is float in epilogue functor?
    // Wait, Epilogue definitions: "const float* bias". 
    // Functor: "return x + bias[channel]".
    // x comes from accumulator which is float (in cuda.rs) or cast to float.
    
    let a_f16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b_f16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    // Bias is float because accumulation is float
    let bias_vec = bias.to_vec();

    manager.copy_to_device(a_gpu, &a_f16).unwrap();
    manager.copy_to_device(b_gpu, &b_f16).unwrap();
    manager.copy_to_device(bias_gpu, &bias_vec).unwrap();

    let grid = ((n as u32 + 63) / 64, (m as u32 + 63) / 64, 1); // 64 is micro tile size based on config
    let block = (5 * 32, 1, 1); // 5 warps (1 producer + 4 consumers) for 64x64 tile
    let smem = 32768; // Estimate

    println!("Launching Kernel...");
    // Args: A, B, C, M, N, K, Bias
    manager.launch(kernel_id, grid, block, smem, vec![
        KernelArg::Buffer(a_gpu), 
        KernelArg::Buffer(b_gpu), 
        KernelArg::Buffer(c_gpu),
        KernelArg::Int(m as i32),
        KernelArg::Int(n as i32),
        KernelArg::Int(k as i32),
        KernelArg::Buffer(bias_gpu), // Appended epilogue arg
    ]).expect("Launch failed");

    let mut c_res_f16 = vec![f16::ZERO; m * n];
    manager.copy_from_device(c_gpu, &mut c_res_f16).unwrap();

    println!("Verifying...");
    let mut max_err = 0.0;
    for i in 0..m {
        for j in 0..n {
            let val = c_res_f16[i * n + j].to_f32();
            let ref_val = ref_c[[i, j]];
            let err = (val - ref_val).abs();
            if err > max_err { max_err = err; }
        }
    }
    
    println!("Max Absolute Error: {:.6}", max_err);
    if max_err < 1e-2 {
        println!("✅ SUCCESS");
    } else {
        println!("❌ FAIL");
    }
}
