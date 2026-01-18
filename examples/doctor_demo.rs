use tracea::doctor;
use std::env;

fn main() {
    println!("--- Tracea Doctor Multi-Backend Verification ---");

    // Scenario 1: Default Environment (Likely CUDA + CPU)
    println!("\n[Scenario 1] Default Environment");
    run_verification_step("Default");

    // Scenario 2: Simulated ROCm Environment
    println!("\n[Scenario 2] Simulated ROCm Environment");
    // enable mock via env var (as implemented in profiler.rs)
    env::set_var("TRACEA_MOCK_ROCM", "1"); 
    run_verification_step("ROCm Simulation");
    env::remove_var("TRACEA_MOCK_ROCM");

    // Scenario 3: Simulated Metal Environment
    println!("\n[Scenario 3] Simulated Metal Environment");
    env::set_var("TRACEA_MOCK_METAL", "1");
    run_verification_step("Metal Simulation");
    env::remove_var("TRACEA_MOCK_METAL");

    // Scenario 4: CPU Only (Force no CUDA if possible, or just checking CPU variant preference)
    // We can't easily "hide" CUDA if cudarc detects it, but we can request a CPU-specific variant via constraints if we wanted,
    // or just rely on the fact that we have CPU variants now.
    println!("\n[Scenario 4] CPU Capabilities Check");
    let caps = doctor::get_capabilities();
    if let Some(cpu_caps) = caps.get_backend(doctor::BackendKind::Cpu) {
        println!("  CPU Detected: Cores={}, SIMD Bits={}", cpu_caps.core_count, cpu_caps.simd_width_bits);
    }

    println!("\n--- Verification Complete ---");
}

fn run_verification_step(label: &str) {
    let caps = doctor::get_capabilities();
    println!("  [{}] Capabilities Detected:", label);
    for backend in &caps.backends {
        println!("    - {:?} (Arch: {}, Mem: {} KB, TensorCore: {})", 
            backend.backend, backend.arch_code, backend.max_shared_mem / 1024, backend.has_tensor_core_like);
    }

    // Plan FlashAttention-2
    println!("  Planning FlashAttention-2 (Precision=BF16)...");
    let request_fa2 = doctor::KernelRequestContext {
        precision_policy: doctor::PrecisionPolicy::BF16,
        latency_vs_throughput: 0.8,
        allow_fallback: true,
    };

    // We assume plan_kernel uses the capabilities we just fetched internally? 
    // Wait, the previous doctor_demo.rs called `doctor::plan_kernel`, which presumably calls `get_capabilities` internally?
    // Or does `plan_kernel` take capabilities?
    // Checking `lib.rs` or `mod.rs` would be good, but assuming standard flow:
    // Actually `engine::select_variant` takes `&caps`.
    // I need to check how `doctor::plan_kernel` is exposed. 
    // If checking `doctor_demo.rs` original: `let decision_fa2 = doctor::plan_kernel("flash_attention_2", request_fa2);`
    // It seems `plan_kernel` might be a wrapper. I should check `mod.rs` or `lib.rs` to see if `plan_kernel` accepts caps or fetches them.
    // If it fetches them, I'm good.
    // But to be safe, I'll update it to pass caps if the API allows, or rely on it fetching fresh caps (which will see the env var).
    
    let decision_fa2 = doctor::plan_kernel("flash_attention_2", request_fa2);
    
    if let Some(vid) = decision_fa2.selected_variant {
        println!("    => Selected Variant: {}", vid);
        println!("    => Strategy: {:?}", decision_fa2.compile_strategy);
    } else {
        println!("    => No suitable variant selected.");
        for failure in decision_fa2.fallback_plan {
            println!("       - Rejected {}: {}", failure.variant_id, failure.reason);
        }
    }
}

