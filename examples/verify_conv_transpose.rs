/// ConvTranspose2d Numerical Accuracy Validation
/// Compares Tracea's ConvTranspose2d against CPU reference implementation.
use tracea::*;
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::emitter::universal::UniversalEmitter;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use tracea::core::config::{PipelineConfig, LayoutPolicy};
use rand::Rng;

/// CPU reference implementation of ConvTranspose2d
/// Weight layout: [C_in * R * S, K_out] (matches GPU implicit GEMM)
fn cpu_conv_transpose2d(
    input: &[f32],  // [N, H_in, W_in, C_in] NHWC
    weight: &[f32], // [C_in * R * S, K_out] flattened for implicit GEMM
    n: usize, h_in: usize, w_in: usize, c_in: usize,
    k: usize, r: usize, s: usize,
    stride: usize, pad: usize, output_padding: usize,
) -> Vec<f32> {
    let h_out = (h_in - 1) * stride - 2 * pad + r + output_padding;
    let w_out = (w_in - 1) * stride - 2 * pad + s + output_padding;
    let mut output = vec![0.0f32; n * h_out * w_out * k];
    
    // For ConvTranspose2d, the relationship is:
    // output[ho, wo, k] = sum over (hi, wi, c, rr, ss) where ho = hi*stride - pad + rr
    for batch in 0..n {
        for hi in 0..h_in {
            for wi in 0..w_in {
                for ci in 0..c_in {
                    let in_val = input[((batch * h_in + hi) * w_in + wi) * c_in + ci];
                    
                    for rr in 0..r {
                        for ss in 0..s {
                            let ho = (hi * stride) as i32 - pad as i32 + rr as i32;
                            let wo = (wi * stride) as i32 - pad as i32 + ss as i32;
                            
                            if ho >= 0 && ho < h_out as i32 && wo >= 0 && wo < w_out as i32 {
                                for ko in 0..k {
                                    // Weight layout: [C*R*S, K] to match GPU implicit GEMM
                                    // k_glob = c * R * S + r * S + s for row
                                    // n_glob = k for column
                                    let k_idx = (ci * r + rr) * s + ss;  // k_glob
                                    let w_idx = k_idx * k + ko;  // Weight[k_glob * K + n_glob]
                                    let o_idx = ((batch * h_out + ho as usize) * w_out + wo as usize) * k + ko;
                                    output[o_idx] += in_val * weight[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    output
}


fn main() {
    println!("=== ConvTranspose2d Numerical Accuracy Validation ===\n");
    
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("CUDA init failed");
    let backend = DeviceBackend::Cuda;

    // MINIMAL test: B=1, H=2, W=2, C_in=1, C_out=1, R=S=2, stride=1, pad=0
    // This gives H_out = (2-1)*1 + 2 = 3
    let (n, h, w, c, kk, r, s) = (1, 2, 2, 1, 1, 2, 2);
    let stride = 1;
    let pad = 0;
    let output_padding = 0;
    
    let h_out = (h - 1) * stride - 2 * pad + r + output_padding;
    let w_out = (w - 1) * stride - 2 * pad + s + output_padding;

    
    println!("Input: B={}, H={}, W={}, C={}", n, h, w, c);
    println!("Kernel: K={}, R={}, S={}", kk, r, s);
    println!("Params: stride={}, pad={}, output_padding={}", stride, pad, output_padding);
    println!("Output: H_out={}, W_out={}", h_out, w_out);

    // Generate random input and weights
    let mut rng = rand::thread_rng();
    let input_size = n * h * w * c;
    let weight_size = c * r * s * kk; // [C*R*S, K] layout total size
    let output_size = n * h_out * w_out * kk;

    // Use simpler deterministic values for debugging
    let use_random = true; // Use random values for final verification

    let h_input: Vec<f32> = if use_random {
        (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect()
    } else {
        vec![1.0; input_size] // All ones for debugging
    };
    let h_weight: Vec<f32> = if use_random {
        (0..weight_size).map(|_| rng.gen_range(-0.5..0.5)).collect()
    } else {
        vec![0.1; weight_size] // All 0.1 for debugging
    };
    
    // CPU reference

    println!("\nRunning CPU reference...");
    let cpu_output = cpu_conv_transpose2d(
        &h_input, &h_weight,
        n, h, w, c, kk, r, s,
        stride, pad, output_padding
    );
    
    // GPU execution
    println!("Running GPU kernel...");
    let d_input = runtime.alloc_f32(input_size, backend).expect("Alloc input failed");
    let d_weight = runtime.alloc_f32(weight_size, backend).expect("Alloc weight failed");
    let d_output = runtime.alloc_f32(output_size, backend).expect("Alloc output failed");

    runtime.copy_to_device(d_input, &h_input).expect("Copy input failed");
    runtime.copy_to_device(d_weight, &h_weight).expect("Copy weight failed");

    // Generate kernel
    let mut config = PipelineConfig::new(2, 64, 64, 16);
    config.force_num_warps = Some(4);
    
    let emitter = UniversalEmitter::new(backend);
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::ConvTranspose2d {
            n, c, h, w, k: kk, r, s,
            stride, pad, output_padding,
            layout: LayoutPolicy::NHWC,
        },
        precison: "f32".to_string(),
        tiling: config.clone(),
        conv_magic_strategy: None,
    };

    let source = emitter.generate(ir);
    let kernel_id = runtime.compile(&source, "conv_transpose2d_implicit_gemm", backend)
        .expect("Compilation failed");

    // Launch params
    let m_gemm = n * h_out * w_out;
    let n_gemm = kk;
    let mt = config.m_tile as usize;
    let nt = config.n_tile as usize;
    let kt = config.k_tile as usize;

    let grid = ((m_gemm + mt - 1) / mt, (n_gemm + nt - 1) / nt, 1);
    let block = (128, 1, 1);
    // Shared memory: sA(MT*KT) + sB(KT*NT) + sC accumulator(MT*NT)
    let smem_bytes = mt * kt * 4 + kt * nt * 4 + mt * nt * 4;


    let args = vec![
        KernelArg::Buffer(d_input),
        KernelArg::Buffer(d_weight),
        KernelArg::Buffer(d_output),
    ];

    runtime.launch(
        kernel_id,
        (grid.0 as u32, grid.1 as u32, 1),
        block,
        smem_bytes as u32,
        args,
    ).expect("Launch failed");
    runtime.synchronize();

    // Read back result
    let mut gpu_output = vec![0.0f32; output_size];
    runtime.copy_from_device(d_output, &mut gpu_output).expect("Copy output failed");

    // Compare results
    println!("\n=== Comparison Results ===");
    let mut max_abs_diff = 0.0f32;
    let mut total_sq_diff = 0.0f64;
    let mut max_diff_idx = 0;
    
    for i in 0..output_size {
        let diff = (cpu_output[i] - gpu_output[i]).abs();
        total_sq_diff += (diff as f64).powi(2);
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_diff_idx = i;
        }
    }
    
    let l2_norm = (total_sq_diff / output_size as f64).sqrt();
    
    println!("Max Absolute Diff: {:.6} at index {}", max_abs_diff, max_diff_idx);
    println!("L2 Norm Diff:      {:.6}", l2_norm);
    println!("CPU[{}]: {:.6}, GPU[{}]: {:.6}", max_diff_idx, cpu_output[max_diff_idx], max_diff_idx, gpu_output[max_diff_idx]);
    
    // Print first few values
    println!("\nFirst 8 values comparison:");
    for i in 0..8.min(output_size) {
        println!("  [{}] CPU: {:.4}, GPU: {:.4}, diff: {:.6}", 
                 i, cpu_output[i], gpu_output[i], (cpu_output[i] - gpu_output[i]).abs());
    }

    // Verdict
    println!();
    if max_abs_diff < 1e-4 && l2_norm < 1e-5 {
        println!("✅ PASSED: ConvTranspose2d numerical accuracy verified!");
        println!("   max_abs_diff < 1e-4: {:.6} < 0.0001", max_abs_diff);
        println!("   l2_norm < 1e-5:      {:.6} < 0.00001", l2_norm);
    } else if max_abs_diff < 1e-3 {
        println!("⚠️  MARGINAL: Accuracy within FP32 tolerance but not ideal");
        println!("   max_abs_diff: {:.6}", max_abs_diff);
        println!("   l2_norm:      {:.6}", l2_norm);
    } else {
        println!("❌ FAILED: Numerical accuracy too low");
        println!("   max_abs_diff: {:.6} (expected < 1e-4)", max_abs_diff);
        println!("   l2_norm:      {:.6} (expected < 1e-5)", l2_norm);
    }
}
