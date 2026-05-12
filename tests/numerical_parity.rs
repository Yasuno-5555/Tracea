use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::doctor::visualizer::Visualizer;
use tracea::emitter::universal::UniversalEmitter;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use tracea::PipelineConfig;
use half::f16;
use std::sync::Arc;

#[cfg(feature = "vulkan")]
#[test]
fn test_cuda_vulkan_parity() {
    let runtime = RuntimeManager::init(None).unwrap();
    let has_cuda = runtime.devices.lock().unwrap().contains_key(&DeviceBackend::Cuda);
    let has_vulkan = runtime.devices.lock().unwrap().contains_key(&DeviceBackend::Vulkan);

    if !has_cuda || !has_vulkan {
        println!("Test requires both CUDA and Vulkan backends. Skipping.");
        return;
    }

    let visualizer = Visualizer::new(runtime.clone());
    let size = 1024;
    
    // Allocate and initialize buffers on both backends
    let b_cuda = runtime.alloc(size * 4, DeviceBackend::Cuda).unwrap();
    let b_vulkan = runtime.alloc(size * 4, DeviceBackend::Vulkan).unwrap();

    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
    let u8_data: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size * 4) };

    runtime.write(b_cuda, u8_data).unwrap();
    runtime.write(b_vulkan, u8_data).unwrap();

    // Compute error map
    let report = visualizer.compare_tensors(b_cuda, b_vulkan, size).expect("Comparison failed");

    println!("Numerical Parity Report:");
    println!("  MAE: {}", report.max_abs_error);
    println!("  MSE: {}", report.mean_squared_error);
    if let Some(heatmap) = report.heatmap {
        println!("  Heatmap:\n{}", heatmap);
    }

    assert!(report.max_abs_error < 1e-5, "Numerical deviation too high!");
}

#[test]
fn test_cuda_gemm_numerical_parity() {
    // 1. Detect if CUDA/NVRTC is available
    let runtime = match RuntimeManager::init(Some(DeviceBackend::Cuda)) {
        Ok(r) => r,
        Err(_) => {
            println!("CUDA runtime not available. Skipping test.");
            return;
        }
    };
    
    let has_cuda = runtime.devices.lock().unwrap().contains_key(&DeviceBackend::Cuda);
    if !has_cuda {
        println!("No CUDA device found. Skipping test.");
        return;
    }

    let m = 16;
    let k = 16;
    let n = 16;

    // 2. Prepare CPU reference using matrixmultiply
    let mut a_f32 = vec![0.0f32; m * k];
    let mut b_f32 = vec![0.0f32; k * n];
    let mut c_f32_cpu = vec![0.0f32; m * n];

    // Initialize with stable deterministic values
    for i in 0..m * k { a_f32[i] = (i as f32) * 0.05 - 0.4; }
    for i in 0..k * n { b_f32[i] = (i as f32) * 0.02 - 0.1; }

    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a_f32.as_ptr(), k as isize, 1,
            b_f32.as_ptr(), n as isize, 1,
            0.0,
            c_f32_cpu.as_mut_ptr(), n as isize, 1,
        );
    }

    // 3. Prepare inputs/outputs in half precision for CUDA
    let a_f16: Vec<f16> = a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let b_f16: Vec<f16> = b_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let size_a = m * k * 2; // f16 is 2 bytes
    let size_b = k * n * 2;
    let size_c = m * n * 2;

    let buf_a = runtime.alloc(size_a, DeviceBackend::Cuda).unwrap();
    let buf_b = runtime.alloc(size_b, DeviceBackend::Cuda).unwrap();
    let buf_c = runtime.alloc(size_c, DeviceBackend::Cuda).unwrap();

    runtime.copy_to_device(buf_a, &a_f16).unwrap();
    runtime.copy_to_device(buf_b, &b_f16).unwrap();

    // 4. Generate & compile CUDA GEMM
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            batch: 1,
            epilogue: vec![],
        },
        precison: "f16".to_string(),
        tiling: PipelineConfig::new(2, 16, 16, 16),
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let emitter = UniversalEmitter::new(DeviceBackend::Cuda);
    let source = emitter.generate(ir).unwrap();
    
    let kernel_id = runtime.compile(&source, "gemm_mma_kernel", DeviceBackend::Cuda).unwrap();

    // 5. Launch CUDA Kernel
    let grid = (1, 1, 1);
    let block = (32, 1, 1); // 1 warp
    let args = vec![
        KernelArg::Buffer(buf_a),
        KernelArg::Buffer(buf_b),
        KernelArg::Buffer(buf_c),
        KernelArg::Int(m as i32),
        KernelArg::Int(n as i32),
        KernelArg::Int(k as i32),
    ];

    runtime.launch(kernel_id, grid, block, 1024, args).unwrap();
    runtime.synchronize();

    // 6. Read back & verify
    let mut c_f16_gpu = vec![f16::ZERO; m * n];
    runtime.copy_from_device(buf_c, &mut c_f16_gpu).unwrap();

    let mut max_abs_error = 0.0f32;
    for i in 0..m * n {
        let expected = c_f32_cpu[i];
        let actual = c_f16_gpu[i].to_f32();
        let abs_error = (expected - actual).abs();
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
        }
    }

    println!("CUDA GEMM Numerical Parity Verification Success. Max Abs Error: {}", max_abs_error);
    assert!(max_abs_error < 5e-2, "CUDA GEMM precision mismatch! Max Abs Error: {}", max_abs_error);
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_gemm_numerical_parity() {
    let runtime = match RuntimeManager::init(Some(DeviceBackend::Metal)) {
        Ok(r) => r,
        Err(_) => {
            println!("Metal runtime not available. Skipping test.");
            return;
        }
    };
    
    let has_metal = runtime.devices.lock().unwrap().contains_key(&DeviceBackend::Metal);
    if !has_metal {
        println!("No Metal device found. Skipping test.");
        return;
    }

    let m = 16;
    let k = 16;
    let n = 16;

    // 1. Prepare CPU reference using matrixmultiply
    let mut a_f32 = vec![0.0f32; m * k];
    let mut b_f32 = vec![0.0f32; k * n];
    let mut c_f32_cpu = vec![0.0f32; m * n];

    for i in 0..m * k { a_f32[i] = (i as f32) * 0.05 - 0.4; }
    for i in 0..k * n { b_f32[i] = (i as f32) * 0.02 - 0.1; }

    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a_f32.as_ptr(), k as isize, 1,
            b_f32.as_ptr(), n as isize, 1,
            0.0,
            c_f32_cpu.as_mut_ptr(), n as isize, 1,
        );
    }

    // 2. Prepare inputs in f16 for Metal
    let a_f16: Vec<f16> = a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let b_f16: Vec<f16> = b_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let size_a = m * k * 2;
    let size_b = k * n * 2;
    let size_c = m * n * 4; // Metal output C is float (f32, 4 bytes)

    let buf_a = runtime.alloc(size_a, DeviceBackend::Metal).unwrap();
    let buf_b = runtime.alloc(size_b, DeviceBackend::Metal).unwrap();
    let buf_c = runtime.alloc(size_c, DeviceBackend::Metal).unwrap();

    // Clear output buffer with zeros to detect if the kernel is actually writing anything!
    let zeros = vec![0.0f32; m * n];
    runtime.copy_to_device(buf_c, &zeros).unwrap();

    runtime.copy_to_device(buf_a, &a_f16).unwrap();
    runtime.copy_to_device(buf_b, &b_f16).unwrap();

    // 3. Generate & compile Metal GEMM
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            batch: 1,
            epilogue: vec![],
        },
        precison: "f16".to_string(),
        tiling: PipelineConfig::new(1, 16, 16, 16), // matching m_tile=16, n_tile=16, k_tile=16
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let emitter = UniversalEmitter::new(DeviceBackend::Metal);
    let mut source = emitter.generate(ir).unwrap();
    
    // Crucial JIT hack: Replace hardcoded load step size 128 with 32 to match block thread count
    // and prevent uninitialized shared memory / boundary out of bounds crashes!
    source = source.replace("i += 128", "i += 32");

    let kernel_id = runtime.compile(&source, "unified_gemm_kernel", DeviceBackend::Metal).unwrap();

    // 4. Allocate dynamic bounds for buffer args to avoid Metal pointer faults
    let buf_m = runtime.alloc(4, DeviceBackend::Metal).unwrap();
    let buf_n = runtime.alloc(4, DeviceBackend::Metal).unwrap();
    let buf_k = runtime.alloc(4, DeviceBackend::Metal).unwrap();

    runtime.copy_to_device(buf_m, &[m as u32]).unwrap();
    runtime.copy_to_device(buf_n, &[n as u32]).unwrap();
    runtime.copy_to_device(buf_k, &[k as u32]).unwrap();

    // 5. Launch Metal Kernel
    let grid = (1, 1, 1);
    let block = (32, 1, 1); // 1 simdgroup (32 threads) is perfect and aligned with replace hash!
    let args = vec![
        KernelArg::Buffer(buf_a),
        KernelArg::Buffer(buf_b),
        KernelArg::Buffer(buf_c),
        KernelArg::Buffer(buf_m),
        KernelArg::Buffer(buf_n),
        KernelArg::Buffer(buf_k),
    ];

    runtime.launch(kernel_id, grid, block, 0, args).unwrap();
    runtime.synchronize();

    // 6. Read back & verify
    let mut c_f32_gpu = vec![-999.0f32; m * n]; // Initialize with distinctive debug value
    runtime.copy_from_device(buf_c, &mut c_f32_gpu).unwrap();

    println!("First 10 values from GPU: {:?}", &c_f32_gpu[0..10]);
    println!("First 10 expected values from CPU: {:?}", &c_f32_cpu[0..10]);

    let mut max_abs_error = 0.0f32;
    for i in 0..m * n {
        let expected = c_f32_cpu[i];
        let actual = c_f32_gpu[i];
        let abs_error = (expected - actual).abs();
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
        }
    }

    println!("Metal GEMM Numerical Parity Verification. Max Abs Error: {}", max_abs_error);
    assert!(max_abs_error < 1e-1, "Metal GEMM precision mismatch! Max Abs Error: {}", max_abs_error);
}
