use tracea::optimizer::{ProblemDescriptor, AutoTuner, OptimizationGoal, GPUInfo};
use tracea::core::backend::{Device, CpuArch};
use tracea::runtime::manager::DeviceBackend;
use tracea::MicroBenchmark;
use std::time::Instant;

fn main() {
    println!("--- Tracea CPU GEMM Tuning Benchmark ---");

    // 1. Define Problem
    let m = 1024;
    let n = 1024;
    let k = 1024;
    let problem = ProblemDescriptor::new_gemm(m, n, k)
        .with_device(Device::Cpu(CpuArch::Avx2));

    // 2. Setup Tuner for CPU
    let mut cpu_info = GPUInfo::rtx3070(); // Placeholder
    cpu_info.backend = DeviceBackend::Cpu; 
    cpu_info.name = "Intel/AMD AVX2 Processor".to_string();

    let mut tuner = AutoTuner::new(cpu_info);

    // 3. Real Benchmark Adapter for CPU
    struct CpuBenchmark {
        m: usize, n: usize, k: usize,
        a: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
    }
    
    impl tracea::optimizer::benchmark::MicroBenchmark for CpuBenchmark {
        fn measure(&self, config: &tracea::PipelineConfig) -> tracea::optimizer::benchmark::BenchmarkResult {
            let mut c = vec![0.0f32; self.m * self.n];
            
            // Warmup
            tracea::kernels::cpu::gemm::gemm_cpu_packed(
                &ProblemDescriptor::new_gemm(self.m, self.n, self.k),
                config,
                &self.a,
                &self.b,
                &mut c,
            );

            let iterations = 3;
            let start = Instant::now();
            for _ in 0..iterations {
                tracea::kernels::cpu::gemm::gemm_cpu_packed(
                    &ProblemDescriptor::new_gemm(self.m, self.n, self.k),
                    config,
                    &self.a,
                    &self.b,
                    &mut c,
                );
            }
            let dur = start.elapsed().as_secs_f32() / iterations as f32;
            let latency_ms = dur * 1000.0;
            let tflops = (2.0 * self.m as f32 * self.n as f32 * self.k as f32) / (dur * 1e12);
            
            println!("  -> Mc={}, Nc={}, Kc={}, Mr={}x{} | Latency: {:.2}ms | Result: {:.4} TFLOPS", 
                config.m_tile, config.n_tile, config.k_tile, 
                config.micro_m, config.micro_n,
                latency_ms, tflops);

            tracea::optimizer::benchmark::BenchmarkResult {
                latency_ms,
                tflops,
                mean_tflops: tflops,
                std_dev: 0.0,
            }
        }
        fn validate_config(&self, _config: &tracea::PipelineConfig) -> bool { true }
        fn m(&self) -> u32 { self.m as u32 }
        fn n(&self) -> u32 { self.n as u32 }
        fn k(&self) -> u32 { self.k as u32 }
        fn device_info(&self) -> tracea::optimizer::benchmark::EnvironmentInfo {
            tracea::optimizer::benchmark::EnvironmentInfo {
                backend: tracea::runtime::manager::DeviceBackend::Cpu,
                api_version: "0.1.0".to_string(),
                driver_version: "0.1.0".to_string(),
                arch: "x86_64".to_string(),
            }
        }
    }

    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let c = vec![0.0f32; m * n];
    let bench = CpuBenchmark { m, n, k, a, b, c };

    // 4. Run Reference Baseline (matrixmultiply)
    println!("\n[Baseline] Running matrixmultiply reference...");
    let mut c_ref = vec![0.0f32; m * n];
    let iters = 5;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                1.0, 
                bench.a.as_ptr(), k as isize, 1,
                bench.b.as_ptr(), n as isize, 1,
                1.0,
                c_ref.as_mut_ptr(), n as isize, 1,
            );
        }
    }
    let dur_ref = start.elapsed().as_secs_f32() / iters as f32;
    let tflops_ref = (2.0 * m as f32 * n as f32 * k as f32) / (dur_ref * 1e12);
    println!("Reference (matrixmultiply) TFLOPS: {:.4}", tflops_ref);

    // 5. Run Tuning
    println!("\n[Tracea] Starting tuning for CPU...");
    let best_config = tuner.optimize_v2(&bench, &problem, 3, OptimizationGoal::MaximizeTFLOPS);

    println!("\n--- Final Results ---");
    println!("Best Tiling: {}x{}x{}", best_config.m_tile, best_config.n_tile, best_config.k_tile);
    println!("Micro-kernel: {}x{}", best_config.micro_m, best_config.micro_n);
    
    // Final measure of best config
    let final_res = bench.measure(&best_config);
    println!("Tracea Optimized TFLOPS: {:.4}", final_res.tflops);
    println!("Speedup vs Reference: {:.2}x", final_res.tflops / tflops_ref);
}
