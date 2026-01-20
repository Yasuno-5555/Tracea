use tracea::core::tuning::TunableKernel;
use tracea::kernels::gemm::cuda_gemm::CudaGemmAdapter;
use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use half::f16;
use std::io::Write;

fn main() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Failed to init runtime");
    
    let m = 2048;
    let n = 2048;
    let k = 2048;
    
    println!("Benchmarking GEMM {}x{}x{}", m, n, k);
    
    let problem = tracea::kernels::gemm::cuda_gemm::CudaGemmProblem { m, n, k };
    let mut adapter = CudaGemmAdapter::new(runtime.clone(), problem);
    
    // Initialize with some data to avoid NaNs/Underflows if it matters
    let a_init = vec![f16::from_f32(0.1); (m*k) as usize];
    let b_init = vec![f16::from_f32(0.1); (k*n) as usize];
    runtime.copy_to_device(adapter.a_buf, &a_init).unwrap();
    runtime.copy_to_device(adapter.b_buf, &b_init).unwrap();

    // Search for best config
    let candidates = adapter.search_space();
    println!("Found {} candidates", candidates.candidates.len());
    std::io::stdout().flush().unwrap();
    
    let mut best_tflops = 0.0;
    let mut best_cfg = None;
    
    for (i, cfg) in candidates.candidates.iter().enumerate() {
        if !adapter.is_feasible(cfg) {
            continue;
        }
        
        let warmups = 3;
        let iters = 10;
        let mut total_gflops = 0.0;
        
        for j in 0..(warmups + iters) {
            if let Some(gflops) = adapter.benchmark(cfg) {
                if j >= warmups {
                    total_gflops += gflops;
                }
            } else {
                println!("  Iteration {} failed", j);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
        }
        
        let avg_gflops = total_gflops / iters as f32;
        let tflops = avg_gflops as f64 / 1000.0;
        
        println!("[{}/{}] Config: {:?} -> {:.2} TFLOPS", i+1, candidates.candidates.len(), cfg, tflops);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        if tflops > best_tflops {
            best_tflops = tflops;
            best_cfg = Some(cfg.clone());
        }
    }
    
    if let Some(cfg) = best_cfg {
        println!("Best Configuration Found: {:?}", cfg);
        println!("Best Performance: {:.2} TFLOPS", best_tflops);
    }
}
