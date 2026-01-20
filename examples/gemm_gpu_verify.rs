use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::kernels::gemm::cuda_gemm::{CudaGemmAdapter, CudaGemmProblem};
use tracea::core::tuning::TunableKernel;
use std::sync::Arc;
use half::f16;
use rand::Rng;

fn cpu_reference(m: usize, n: usize, k: usize, a: &[f16], b: &[f16]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p].to_f32() * b[p * n + j].to_f32();
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn main() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).unwrap();
    
    // Choose dimensions compatible with our tiles
    let m = 128;
    let n = 128;
    let k = 128;
    
    let problem = CudaGemmProblem { m, n, k };
    let adapter = CudaGemmAdapter::new(Arc::clone(&runtime), problem);
    
    let mut rng = rand::thread_rng();
    let a_data: Vec<f16> = (0..m*k).map(|_| f16::from_f32(rng.gen_range(-1.0..1.0))).collect();
    let b_data: Vec<f16> = (0..k*n).map(|_| f16::from_f32(rng.gen_range(-1.0..1.0))).collect();
    
    runtime.copy_to_device(adapter.a_buf, &a_data).unwrap();
    runtime.copy_to_device(adapter.b_buf, &b_data).unwrap();
    
    let space = adapter.search_space();
    let config = &space.candidates[0]; // Take the first candidate (e.g., 128x128, 2 stages)
    
    println!("Benchmarking Config: {:?}", config);
    let gflops = adapter.benchmark(config).expect("Benchmark failed");
    println!("Performance: {:.2} GFLOPS", gflops);
    
    // Read back C
    let mut c_gpu_half = vec![f16::ZERO; m * n];
    runtime.copy_from_device(adapter.c_buf, &mut c_gpu_half).unwrap();
    
    let c_ref = cpu_reference(m, n, k, &a_data, &b_data);
    
    let mut max_err = 0.0f32;
    for i in 0..m*n {
        let diff = (c_gpu_half[i].to_f32() - c_ref[i]).abs();
        if diff > max_err { max_err = diff; }
    }
    
    println!("Max Absolute Error: {:.6}", max_err);
    if max_err < 0.1 { // Allow some error for f16
        println!("✅ Numerical Verification Passed!");
    } else {
        println!("❌ Numerical Verification Failed!");
        // Print some samples
        for i in 0..5 {
            println!("  [{}] GPU: {:.4}, REF: {:.4}", i, c_gpu_half[i].to_f32(), c_ref[i]);
        }
    }
}
