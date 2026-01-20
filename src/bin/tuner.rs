use std::env;
use std::sync::Arc;
use tracea::core::tuning::{tune_kernel, SearchMode};
use tracea::kernels::attention::cuda_adapter::{Fa2Adapter, Fa2Problem};
use tracea::kernels::gemm::cuda_gemm::{CudaGemmAdapter, CudaGemmProblem};
use tracea::kernels::gemm::rocm_gemm::{RocmGemmAdapter, RocmGemmProblem};
use tracea::kernels::gemm::metal_gemm::{MetalGemmAdapter, MetalGemmProblem};
use tracea::kernels::gemm::cpu_adapter::{GemmAdapter, GemmProblem, CpuGemmConfig};
use tracea::backend::cuda::CudaBackend;
use tracea::backend::cpu::CpuBackend;
use tracea::runtime::manager::RuntimeManager;
use tracea::backend::Backend;

fn help() {
    println!("Tracea Tuning OS");
    println!("Usage: tuner --backend=<cuda|cpu> --kernel=<fa2|gemm>");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut backend_arg = "cuda".to_string();
    let mut kernel_arg = "fa2".to_string();

    for arg in &args[1..] {
        if arg.starts_with("--backend=") {
            backend_arg = arg.trim_start_matches("--backend=").to_string();
        } else if arg.starts_with("--kernel=") {
            kernel_arg = arg.trim_start_matches("--kernel=").to_string();
        } else {
            help();
            return;
        }
    }

    println!("Starting Tuning OS...");
    println!("Backend: {}", backend_arg);
    println!("Kernel:  {}", kernel_arg);

    match backend_arg.as_str() {
        "cuda" => {
            // #[cfg(feature = "cuda")] // Removed
            {
                // Initialize RuntimeManager (Device Abstraction)
                // In a real app we might pass specific device ID.
                // RuntimeManager::init uses lazy static or similar, 
                // but let's assume we can get an instance.
                let runtime = RuntimeManager::init(Some(tracea::runtime::manager::DeviceBackend::Cuda))
                    .expect("Failed to initialize RuntimeManager");

                // Initialize Backend Adapter (for device info)
                // Note: RuntimeManager encapsulates the device, but CudaBackend is our "Tuning OS" view of it.
                // We might need to construct CudaBackend separately or extract from runtime.
                // For now, let's create a CudaBackend just to implement the trait, 
                // even though RuntimeManager holds the real context.
                // This duality will need unification later.
                let _backend = CudaBackend::new(0).expect("Failed to init CudaBackend"); 
                
                if kernel_arg == "fa2" {
                    let problem = Fa2Problem {
                        b: 1, s: 128, h: 8, d: 64, is_causal: true
                    };
                    let adapter = Fa2Adapter::new(runtime, problem);
                    
                    println!("Tuning FA2...");
                    let best_config = tune_kernel(&adapter, SearchMode::GridSearch);
                    println!("Best Config Found: {:?}", best_config);
                } else if kernel_arg == "gemm" {
                    let problem = CudaGemmProblem { m: 1024, n: 1024, k: 1024 };
                    let adapter = CudaGemmAdapter::new(runtime, problem);

                    println!("Tuning CUDA GEMM (MMA)...");
                    let best_config = tune_kernel(&adapter, SearchMode::GridSearch);
                    println!("Best Config Found: {:?}", best_config);
                } else {
                    println!("Unknown kernel for CUDA: {}", kernel_arg);
                }
            }
            /*
            #[cfg(not(feature = "cuda"))]
            {
                println!("CUDA feature not enabled.");
            }
            */
        },
        "rocm" => {
            let runtime = RuntimeManager::init(Some(tracea::runtime::manager::DeviceBackend::Rocm))
                .expect("Failed to init Runtime for ROCm");
            
            if kernel_arg == "gemm" {
                 let problem = RocmGemmProblem { m: 1024, n: 1024, k: 1024 };
                 let adapter = RocmGemmAdapter::new(runtime, problem);
                 println!("Tuning ROCm GEMM (MFMA)...");
                 let best = tune_kernel(&adapter, SearchMode::GridSearch);
                 println!("Best Config: {:?}", best);
            } else {
                 println!("Unknown kernel for ROCm: {}", kernel_arg);
            }
        },
        "metal" => {
            let runtime = RuntimeManager::init(Some(tracea::runtime::manager::DeviceBackend::Metal))
                .expect("Failed to init Runtime for Metal");

            if kernel_arg == "gemm" {
                 let problem = MetalGemmProblem { m: 1024, n: 1024, k: 1024 };
                 let adapter = MetalGemmAdapter::new(runtime, problem);
                 println!("Tuning Metal GEMM (SIMD)...");
                 let best = tune_kernel(&adapter, SearchMode::GridSearch);
                 println!("Best Config: {:?}", best);
            } else {
                 println!("Unknown kernel for Metal: {}", kernel_arg);
            }
        },
        "cpu" => {
            let backend = CpuBackend::new();
            if kernel_arg == "gemm" {
                 let problem = GemmProblem { m: 1024, n: 1024, k: 1024 };
                 let adapter = GemmAdapter::new(backend, problem);
                 
                 println!("Tuning CPU GEMM...");
                 let best_config = tune_kernel(&adapter, SearchMode::GridSearch);
                 println!("Best Config Found: {:?}", best_config);
            } else {
                println!("Unknown kernel for CPU: {}", kernel_arg);
            }
        },
        _ => {
            println!("Unsupported backend: {}", backend_arg);
        }
    }
}
