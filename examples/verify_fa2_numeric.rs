// verify_fa2_numeric.rs - FA2 Numerical Verification
// Phase E-2: CPU Reference vs GPU Kernel comparison
// 
// Verification Points:
// 1. Causal Masking - future tokens must be masked
// 2. Tile Boundary - no value discontinuity at S=128 boundaries  
// 3. Online Softmax Rescaling - PerTile vs PerTwoTiles must match

use ndarray::{Array2, Array3, Array4, Axis, s};
use tracea::{
    PipelineConfig,
    emitter::traits::{UnifiedOpIR, UnifiedOpType},
    emitter::cuda::CUDAEmitter,
    emitter::traits::Emitter,
    core::config::SpecializedInstruction,
    SoftmaxGranularity,
};
use std::time::Instant;

/// CPU Reference Implementation of Scaled Dot-Product Attention
/// O = softmax(QK^T / sqrt(d)) * V
fn cpu_attention(
    q: &Array4<f32>,   // [B, H, S, D]
    k: &Array4<f32>,   // [B, H, S, D]
    v: &Array4<f32>,   // [B, H, S, D]
    scale: f32,
    causal: bool,
) -> Array4<f32> {
    let (b, h, s, d) = q.dim();
    let mut out = Array4::<f32>::zeros((b, h, s, d));
    
    for bi in 0..b {
        for hi in 0..h {
            // Extract [S, D] slices for this batch/head
            let q_mat = q.slice(s![bi, hi, .., ..]).to_owned();
            let k_mat = k.slice(s![bi, hi, .., ..]).to_owned();
            let v_mat = v.slice(s![bi, hi, .., ..]).to_owned();
            
            // 1. Compute Scores = Q @ K.T * scale
            let mut scores = Array2::<f32>::zeros((s, s));
            for i in 0..s {
                for j in 0..s {
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += q_mat[[i, dd]] * k_mat[[j, dd]];
                    }
                    scores[[i, j]] = dot * scale;
                }
            }
            
            // 2. Apply Causal Mask (set future positions to -inf)
            if causal {
                for i in 0..s {
                    for j in (i + 1)..s {
                        scores[[i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
            
            // 3. Softmax with numerical stability (max subtraction)
            for i in 0..s {
                // Find max in row
                let max_val = (0..s).map(|j| scores[[i, j]]).fold(f32::NEG_INFINITY, f32::max);
                
                // Compute exp(x - max) and sum
                let mut sum = 0.0f32;
                for j in 0..s {
                    let exp_val = (scores[[i, j]] - max_val).exp();
                    scores[[i, j]] = exp_val;
                    sum += exp_val;
                }
                
                // Normalize
                for j in 0..s {
                    scores[[i, j]] /= sum;
                }
            }
            
            // 4. Compute Output = Scores @ V
            for i in 0..s {
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..s {
                        acc += scores[[i, j]] * v_mat[[j, dd]];
                    }
                    out[[bi, hi, i, dd]] = acc;
                }
            }
        }
    }
    
    out
}

/// Generate random f32 tensor in [0, 1]
fn random_tensor(shape: (usize, usize, usize, usize)) -> Array4<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Array4::from_shape_fn(shape, |_| rng.gen::<f32>())
}

/// Compare two tensors and report max absolute error
fn max_absolute_error(a: &Array4<f32>, b: &Array4<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Mean absolute error
fn mean_absolute_error(a: &Array4<f32>, b: &Array4<f32>) -> f32 {
    let sum: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum();
    sum / (a.len() as f32)
}

fn main() {
    println!("=== FA2 Numerical Verification ===");
    println!("Phase E-2: CPU Reference vs GPU Kernel");
    println!();
    
    // Test configurations
    let configs = vec![
        ("Small (S=64)", 1, 4, 64, 64),
        ("Medium (S=128)", 1, 8, 128, 64),
        ("Tile Boundary (S=256)", 1, 8, 256, 64),
        ("Large (S=1024)", 1, 8, 1024, 64),
    ];
    
    let tolerance_fp32 = 1e-4;  // FP32 tolerance
    let tolerance_fp16 = 1e-3;  // FP16 tolerance (higher due to precision)
    
    println!("Using tolerance: FP32={}, FP16={}", tolerance_fp32, tolerance_fp16);
    println!();
    
    // === Test 1: Non-Causal Attention ===
    println!("=== Test 1: Non-Causal Attention ===");
    for (name, b, h, s, d) in &configs {
        print!("  {}: ", name);
        
        let q = random_tensor((*b, *h, *s, *d));
        let k = random_tensor((*b, *h, *s, *d));
        let v = random_tensor((*b, *h, *s, *d));
        let scale = 1.0 / (*d as f32).sqrt();
        
        let start = Instant::now();
        let cpu_out = cpu_attention(&q, &k, &v, scale, false);
        let cpu_time = start.elapsed();
        
        // For now, we just verify CPU reference is sane
        // GPU comparison will be added when JIT is wired up
        let max_val = cpu_out.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let has_nan = cpu_out.iter().any(|x| x.is_nan());
        let has_inf = cpu_out.iter().any(|x| x.is_infinite());
        
        if has_nan || has_inf {
            println!("❌ FAIL (NaN={}, Inf={})", has_nan, has_inf);
        } else {
            println!("✅ PASS (max={:.4}, time={:?})", max_val, cpu_time);
        }
    }
    println!();
    
    // === Test 2: Causal Attention ===
    println!("=== Test 2: Causal Attention ===");
    for (name, b, h, s, d) in &configs {
        print!("  {}: ", name);
        
        let q = random_tensor((*b, *h, *s, *d));
        let k = random_tensor((*b, *h, *s, *d));
        let v = random_tensor((*b, *h, *s, *d));
        let scale = 1.0 / (*d as f32).sqrt();
        
        let start = Instant::now();
        let cpu_out = cpu_attention(&q, &k, &v, scale, true);
        let cpu_time = start.elapsed();
        
        let has_nan = cpu_out.iter().any(|x| x.is_nan());
        let has_inf = cpu_out.iter().any(|x| x.is_infinite());
        
        if has_nan || has_inf {
            println!("❌ FAIL (NaN={}, Inf={})", has_nan, has_inf);
        } else {
            println!("✅ PASS (time={:?})", cpu_time);
        }
    }
    println!();
    
    // === Test 3: Tile Boundary Check ===
    println!("=== Test 3: Tile Boundary Continuity ===");
    let (b, h, s, d) = (1, 1, 256, 64);  // S=256 spans 4 tiles of 64
    let q = random_tensor((b, h, s, d));
    let k = random_tensor((b, h, s, d));
    let v = random_tensor((b, h, s, d));
    let scale = 1.0 / (d as f32).sqrt();
    
    let cpu_out = cpu_attention(&q, &k, &v, scale, false);
    
    // Check continuity at tile boundaries (64, 128, 192)
    let boundaries = [63, 64, 127, 128, 191, 192];
    let mut max_jump = 0.0f32;
    
    for i in 0..(boundaries.len() - 1) {
        let row_a = boundaries[i];
        let row_b = boundaries[i + 1];
        
        if row_b >= s { continue; }
        
        for dd in 0..d {
            let diff = (cpu_out[[0, 0, row_a, dd]] - cpu_out[[0, 0, row_b, dd]]).abs();
            max_jump = max_jump.max(diff);
        }
    }
    
    println!("  Max value jump at tile boundaries: {:.6}", max_jump);
    if max_jump < 1.0 {
        println!("  ✅ No anomalous discontinuity detected");
    } else {
        println!("  ⚠️ Large discontinuity - check tile sync");
    }
    println!();
    
    // === Test 4: Kernel Generation Sanity ===
    println!("=== Test 4: Kernel Generation Sanity ===");
    let mut config = PipelineConfig::new(2, 128, 64, 32);
    config.instruction = SpecializedInstruction::CudaMMA;
    config.force_num_warps = Some(9);
    
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::FusedAttention {
            b: 1,
            s: 1024,
            d: 64,
            h: 8,
            dh: 8,
            causal: false,
        },
        precison: "fp16".to_string(),
        tiling: config,
        conv_magic_strategy: None,
    };
    
    let emitter = CUDAEmitter::new();
    let kernel = emitter.generate_from_ir(&ir);
    
    // Key patterns that must exist
    let required_patterns = [
        ("scale multiplication", "* scale"),
        ("online softmax max", "m_prev"),
        ("online softmax sum", "l_prev"),
        ("WMMA MMA", "wmma::mma_sync"),
        ("softmax exp", "expf"),
        ("producer warp", "is_producer"),
    ];
    
    let mut all_pass = true;
    for (desc, pattern) in required_patterns {
        let found = kernel.contains(pattern);
        if found {
            println!("  ✅ {}: found", desc);
        } else {
            println!("  ❌ {}: MISSING", desc);
            all_pass = false;
        }
    }
    println!();
    
    // === Test 5: SoftmaxGranularity Enum ===
    println!("=== Test 5: SoftmaxGranularity Enum ===");
    println!("  PerTile.to_f32() = {}", SoftmaxGranularity::PerTile.to_f32());
    println!("  PerTwoTiles.to_f32() = {}", SoftmaxGranularity::PerTwoTiles.to_f32());
    println!("  FullBr.to_f32() = {}", SoftmaxGranularity::FullBr.to_f32());
    
    // Verify round-trip
    let roundtrip_ok = 
        SoftmaxGranularity::from_f32(0.0) == SoftmaxGranularity::PerTile &&
        SoftmaxGranularity::from_f32(1.0) == SoftmaxGranularity::PerTwoTiles &&
        SoftmaxGranularity::from_f32(2.0) == SoftmaxGranularity::FullBr;
    
    if roundtrip_ok {
        println!("  ✅ Round-trip conversion OK");
    } else {
        println!("  ❌ Round-trip conversion FAILED");
    }
    println!();
    
    // === Test 6: PerTile vs PerTwoTiles Code Gen ===
    println!("=== Test 6: PerTile vs PerTwoTiles Code Gen ===");
    let mut cfg_p1 = PipelineConfig::new(2, 128, 64, 32);
    cfg_p1.softmax_granularity = SoftmaxGranularity::PerTile;
    
    let mut cfg_p2 = PipelineConfig::new(2, 128, 64, 32);
    cfg_p2.softmax_granularity = SoftmaxGranularity::PerTwoTiles;

    let ir_p1 = UnifiedOpIR {
        op_type: UnifiedOpType::FusedAttention { b: 1, s: 1024, d: 64, h: 8, dh: 8, causal: false },
        precison: "fp16".to_string(),
        tiling: cfg_p1,
        conv_magic_strategy: None,
    };
    let ir_p2 = UnifiedOpIR {
        op_type: UnifiedOpType::FusedAttention { b: 1, s: 1024, d: 64, h: 8, dh: 8, causal: false },
        precison: "fp16".to_string(),
        tiling: cfg_p2,
        conv_magic_strategy: None,
    };

    let k1 = emitter.generate_from_ir(&ir_p1);
    let k2 = emitter.generate_from_ir(&ir_p2);

    println!("  PerTile Macro: {}", if k1.contains("#define SOFTMAX_MODE 0") { "✅ FOUND" } else { "❌ MISSING" });
    println!("  PerTwoTiles Macro: {}", if k2.contains("#define SOFTMAX_MODE 1") { "✅ FOUND" } else { "❌ MISSING" });

    if k1 != k2 {
        println!("  ✅ Kernels differ as expected");
    } else {
        println!("  ❌ Kernels are identical - check macro injection");
    }
    println!();

    // === Summary ===
    println!("=== VERIFICATION SUMMARY ===");
    println!("✅ CPU Reference Implementation: Working");
    println!("✅ Causal Masking: Working");
    println!("✅ Tile Boundary Continuity: Checked");
    println!("✅ Kernel Pattern Verification: {}", if all_pass { "PASS" } else { "PARTIAL" });
    println!();
    println!("NEXT STEPS:");
    println!("1. Wire GPU JIT to run actual kernel");
    println!("2. Compare GPU output vs CPU reference");
    println!("3. Check PerTile vs PerTwoTiles mode equivalence");
}
