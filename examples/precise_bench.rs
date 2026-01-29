use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::optimizer::benchmark::{NVRTCBenchmark, NVRTCConvBenchmark, FlashAttentionBenchmark, Conv2dProblem, FlashAttentionProblem, MicroBenchmark, Conv2dBenchmark};
use tracea::optimizer::{AutoTuner, OptimizationGoal, GPUInfo, ProblemDescriptor, ConvBenchmarkAdapter};
use tracea::core::config::MagicNumberStrategy;
use std::sync::Arc;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Tracea Masterpiece Verification (V2 Tuner)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda))
        .expect("Failed to initialize CUDA runtime");
    
    let mut tuner = AutoTuner::new(GPUInfo::rtx3070()).with_runtime(runtime.clone());

    // --- 1. GEMM Large ---
    {
        println!("📝 [1/5] GEMM Large (2048x2048x2048)");
        let m = 2048; let n = 2048; let k = 2048;
        let bench = NVRTCBenchmark::new(runtime.clone(), m, n, k);
        let prob = ProblemDescriptor::new_gemm(m as usize, n as usize, k as usize);
        let config = tuner.optimize_v2(&bench, &prob, 5, OptimizationGoal::MaximizeTFLOPS);
        let res = bench.measure(&config);
        println!("   >> Result: {:.2} TFLOPS | Config: {:?}", res.tflops, config);
        println!();
    }

    // --- 2. Conv2d B32 NHWC ---
    {
        println!("📝 [2/5] Conv2d B32 NHWC (ResNet-50 3x3)");
        let p = Conv2dProblem::resnet50_conv3x3_64_batch32();
        let bench = NVRTCConvBenchmark::new(runtime.clone(), p.clone());
        let prob = ProblemDescriptor::new_conv2d(p.batch, p.h_in, p.w_in, p.c_in, p.c_out, tracea::optimizer::Layout::NHWC);
        
        let adapter = ConvBenchmarkAdapter {
            inner: &bench,
            magic_strategy: MagicNumberStrategy::select_for(p.h_out() * p.w_out()),
        };
        
        let base_config = tuner.optimize_v2(&adapter, &prob, 5, OptimizationGoal::MaximizeTFLOPS);
        let res = adapter.measure(&base_config);
        println!("   >> Result: {:.2} TFLOPS | Config: {:?}", res.tflops, base_config);
        println!();
    }

    // --- 3. Conv2d B64 NHWC ---
    {
        println!("📝 [3/5] Conv2d B64 NHWC (ResNet-50 3x3)");
        let p = Conv2dProblem::resnet50_conv3x3_64_batch64();
        let bench = NVRTCConvBenchmark::new(runtime.clone(), p.clone());
        let prob = ProblemDescriptor::new_conv2d(p.batch, p.h_in, p.w_in, p.c_in, p.c_out, tracea::optimizer::Layout::NHWC);
        
        let adapter = ConvBenchmarkAdapter {
            inner: &bench,
            magic_strategy: MagicNumberStrategy::select_for(p.h_out() * p.w_out()),
        };

        let base_config = tuner.optimize_v2(&adapter, &prob, 5, OptimizationGoal::MaximizeTFLOPS);
        let res = adapter.measure(&base_config);
        println!("   >> Result: {:.2} TFLOPS | Config: {:?}", res.tflops, base_config);
        println!();
    }

    // --- 4. FA2 S1024 ---
    {
        println!("📝 [4/5] FlashAttention-2 S1024");
        let p = FlashAttentionProblem::new(1, 12, 1024, 64, true);
        let bench = FlashAttentionBenchmark::new(runtime.clone(), p.clone());
        let prob = ProblemDescriptor::new_fa2(p.b as usize, p.s as usize, p.d as usize, tracea::optimizer::Fa2Variant::Causal);
        let config = tuner.optimize_v2(&bench, &prob, 5, OptimizationGoal::MaximizeTFLOPS);
        let res = bench.measure(&config);
        println!("   >> Result: {:.2} TFLOPS | Config: {:?}", res.tflops, config);
        println!();
    }

    // --- 5. FA2 S2048 ---
    {
        println!("📝 [5/5] FlashAttention-2 S2048 (Large S Trigger)");
        let p = FlashAttentionProblem::new(1, 12, 2048, 64, true);
        let bench = FlashAttentionBenchmark::new(runtime.clone(), p.clone());
        let prob = ProblemDescriptor::new_fa2(p.b as usize, p.s as usize, p.d as usize, tracea::optimizer::Fa2Variant::Causal);
        let config = tuner.optimize_v2(&bench, &prob, 5, OptimizationGoal::MaximizeTFLOPS);
        let res = bench.measure(&config);
        println!("   >> Result: {:.2} TFLOPS | Config: {:?}", res.tflops, config);
        println!();
    }
}
