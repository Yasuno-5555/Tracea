use half::f16;
use ndarray::{Array2, Array4, s};
use tracea::core::config::{PipelineConfig, SpecializedInstruction, SoftmaxGranularity};
use tracea::kernels::attention::cuda_emitter::FlashAttentionEmitter;
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};

/// CPU Reference for FlashAttention-2
fn cpu_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    scale: f32,
    is_causal: bool,
) -> Array2<f32> {
    let (s, d) = (q.nrows(), q.ncols());
    let mut o = Array2::zeros((s, d));

    for i in 0..s {
        let mut scores = Array2::zeros((1, s));
        let mut max_val = f32::NEG_INFINITY;

        //  Dot product + Max
        for j in 0..s {
            if is_causal && j > i {
                scores[[0, j]] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0;
                for k_idx in 0..d {
                    dot += q[[i, k_idx]] * k[[j, k_idx]];
                }
                let val = dot * scale;
                scores[[0, j]] = val;
                if val > max_val {
                    max_val = val;
                }
            }
        }

        // Softmax
        let mut sum = 0.0;
        for j in 0..s {
            if scores[[0, j]] != f32::NEG_INFINITY {
                let exp_val = (scores[[0, j]] - max_val).exp();
                scores[[0, j]] = exp_val;
                sum += exp_val;
            } else {
                scores[[0, j]] = 0.0;
            }
        }

        // Weighted Sum
        for j in 0..s {
            let p_val = scores[[0, j]] / (sum + 1e-6);
            for k_idx in 0..d {
                o[[i, k_idx]] += p_val * v[[j, k_idx]];
            }
        }
    }
    o
}

fn main() {
    println!("=== FA2 GPU Numerical Verification ===");
    
    let manager = RuntimeManager::init(None).expect("Failed to init RuntimeManager");

    let b = 1;
    let h = 4;
    let s = 512;
    let d = 64;
    let scale = 1.0 / (d as f32).sqrt(); 
    let causal = true;
    
    println!("Shapes: B={}, H={}, S={}, D={}, Causal={}", b, h, s, d, causal);
    
    let q = Array4::<f32>::from_shape_fn((b, h, s, d), |_| rand::random::<f32>() * 0.2);
    let k = Array4::<f32>::from_shape_fn((b, h, s, d), |_| rand::random::<f32>() * 0.2);
    let v = Array4::<f32>::from_shape_fn((b, h, s, d), |_| rand::random::<f32>() * 0.2);
    
    println!("Computing CPU Reference...");
    let mut gold_results = Vec::new();
    for hi in 0..h {
        let q_slice = q.slice(s![0, hi, .., ..]).to_owned();
        let k_slice = k.slice(s![0, hi, .., ..]).to_owned();
        let v_slice = v.slice(s![0, hi, .., ..]).to_owned();
        gold_results.push(cpu_attention(&q_slice, &k_slice, &v_slice, scale, causal));
    }

    let mut config = PipelineConfig::new(2, 128, 64, 32); 
    config.instruction = SpecializedInstruction::CudaMMA;
    config.force_num_warps = Some(9);
    config.softmax_granularity = SoftmaxGranularity::PerTile;
    
    let emitter = FlashAttentionEmitter::new(config.clone());
    let kernel_name = "flash_attention_v2_kernel";
    let source = emitter.generate_kernel(h, d, causal);
    
    let kernel_id = manager.compile(&source, kernel_name, DeviceBackend::Cuda).expect("JIT Compilation Failed");

    let grid = ((s as u32 + 127) / 128, h as u32, b as u32);
    let block = (9 * 32, 1, 1);
    let (smem_bytes, _, _, _, _) = FlashAttentionEmitter::calculate_smem_layout(&config, d);

    let q_gpu = manager.alloc_u16(b * h * s * d, DeviceBackend::Cuda).expect("Alloc Q");
    let k_gpu = manager.alloc_u16(b * h * s * d, DeviceBackend::Cuda).expect("Alloc K");
    let v_gpu = manager.alloc_u16(b * h * s * d, DeviceBackend::Cuda).expect("Alloc V");
    let o_gpu = manager.alloc_u16(b * h * s * d, DeviceBackend::Cuda).expect("Alloc O");

    let q_f16: Vec<f16> = q.iter().map(|&x| f16::from_f32(x)).collect();
    let k_f16: Vec<f16> = k.iter().map(|&x| f16::from_f32(x)).collect();
    let v_f16: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();

    manager.copy_to_device(q_gpu, &q_f16).unwrap();
    manager.copy_to_device(k_gpu, &k_f16).unwrap();
    manager.copy_to_device(v_gpu, &v_f16).unwrap();

    manager.launch(kernel_id, grid, block, smem_bytes as u32, vec![
        KernelArg::Buffer(q_gpu), KernelArg::Buffer(k_gpu), KernelArg::Buffer(v_gpu), KernelArg::Buffer(o_gpu),
        KernelArg::Usize(b), KernelArg::Usize(h), KernelArg::Usize(s), KernelArg::Usize(d), KernelArg::Float(scale),
    ]).expect("Kernel Launch Failed");

    let mut o_res_f16 = vec![f16::ZERO; b * h * s * d];
    manager.copy_from_device(o_gpu, &mut o_res_f16).unwrap();

    let mut gpu_res = Array4::zeros((b, h, s, d));
    for hi in 0..h {
        for si in 0..s {
            for di in 0..d {
                let idx = (hi * s + si) * d + di;
                gpu_res[[0, hi, si, di]] = o_res_f16[idx].to_f32();
            }
        }
    }

    println!("\n=== VERIFICATION RESULTS ===");
    let mut max_abs_err = 0.0f32;
    let mut total_rel_err = 0.0f32;
    let mut count = 0;
    for bi in 0..b {
        for hi in 0..h {
            let ref_res = &gold_results[hi];
            for i in 0..s {
                for j in 0..d {
                    let g_val = gpu_res[[bi, hi, i, j]];
                    let r_val = ref_res[[i, j]];
                    let err = (g_val - r_val).abs();
                    if err > max_abs_err { max_abs_err = err; }
                    total_rel_err += err / (r_val.abs() + 1e-4);
                    count += 1;
                }
            }
        }
    }

    println!("Max Absolute Error: {:.6}", max_abs_err);
    println!("Avg Relative Error: {:.6}", total_rel_err / (count as f32));

    if max_abs_err < 2e-3 {
        println!("✅ SUCCESS: Numerical accuracy verified!");
    } else {
        println!("❌ FAIL: Numerical accuracy out of tolerance");
    }
}
