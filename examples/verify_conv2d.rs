use tracea::core::config::{PipelineConfig, LayoutPolicy};
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use tracea::runtime::manager::{RuntimeManager, KernelArg, DeviceBackend};
use tracea::doctor::{Doctor, DoctorConfig};
use rand::Rng;

fn cpu_conv2d(
    input: &[f32], weight: &[f32],
    n: usize, h: usize, w: usize, c: usize,
    kk: usize, r: usize, s: usize,
    stride: usize, pad: usize, dilation: usize
) -> Vec<f32> {
    let h_out = (h + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
    let w_out = (w + 2 * pad - dilation * (s - 1) - 1) / stride + 1;
    let mut output = vec![0.0f32; n * h_out * w_out * kk];

    for bi in 0..n {
        for hi in 0..h_out {
            for wi in 0..w_out {
                for ki in 0..kk {
                    let mut acc = 0.0f32;
                    for ri in 0..r {
                        for si in 0..s {
                            for ci in 0..c {
                                let cur_h = (hi * stride) as i32 - pad as i32 + (ri * dilation) as i32;
                                let cur_w = (wi * stride) as i32 - pad as i32 + (si * dilation) as i32;
                                
                                if cur_h >= 0 && cur_h < h as i32 && cur_w >= 0 && cur_w < w as i32 {
                                    let in_idx = ((bi * h + cur_h as usize) * w + cur_w as usize) * c + ci;
                                    let w_idx = ((ri * s + si) * c + ci) * kk + ki;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    let out_idx = ((bi * h_out + hi) * w_out + wi) * kk + ki;
                    output[out_idx] = acc;
                }
            }
        }
    }
    output
}

fn main() {
    println!("=== Conv2d Numerical Accuracy Validation ===");

    // Initialize Doctor
    let _doctor = Doctor::new(DoctorConfig::default());

    // Test Case: Larger Tensor Core compatible
    let (n, h, w, c, kk, r, s) = (1, 8, 8, 32, 32, 3, 3);
    let stride = 1;
    let pad = 1;
    let dilation = 1;

    let h_out = (h + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
    let w_out = (w + 2 * pad - dilation * (s - 1) - 1) / stride + 1;

    println!("Input: B={}, H={}, W={}, C={}", n, h, w, c);
    println!("Kernel: K={}, R={}, S={}", kk, r, s);
    let mut rng = rand::thread_rng();
    let input_size = n * h * w * c;
    let weight_size = kk * r * s * c;
    let output_size = n * h_out * w_out * kk;

    let h_input: Vec<f32> = (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let h_weight: Vec<f32> = (0..weight_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let h_output_cpu = cpu_conv2d(&h_input, &h_weight, n, h, w, c, kk, r, s, stride, pad, dilation);

    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("Failed to init runtime");
    
    // Convert to f16
    let h_input_f16: Vec<half::f16> = h_input.iter().map(|&x| half::f16::from_f32(x)).collect();
    let h_weight_f16: Vec<half::f16> = h_weight.iter().map(|&x| half::f16::from_f32(x)).collect();

    let d_input = runtime.alloc(input_size * 2, DeviceBackend::Cuda).unwrap();
    let d_weight = runtime.alloc(weight_size * 2, DeviceBackend::Cuda).unwrap();
    let d_output = runtime.alloc(output_size * 2, DeviceBackend::Cuda).unwrap();

    runtime.copy_to_device(d_input, &h_input_f16).unwrap();
    runtime.copy_to_device(d_weight, &h_weight_f16).unwrap();

    let config = PipelineConfig::new(2, 64, 64, 16);
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Conv2d {
            n, h, w, c, k: kk, r, s, stride, pad, dilation,
            layout: LayoutPolicy::NHWC,
        },
        precison: "f16".to_string(), // Typo in codebase
        tiling: config.clone(),
        conv_magic_strategy: None,
    };

    let source = tracea::emitter::conv::generate_conv(&ir);

    let m_gemm = n * h_out * w_out;
    let n_gemm = kk;
    
    let grid = ((m_gemm as u32 + config.m_tile - 1) / config.m_tile, (n_gemm as u32 + config.n_tile - 1) / config.n_tile, 1);
    let block = (config.force_num_warps.unwrap_or(4) * 32, 1, 1);
    let smem_bytes = 49152; 

    let kernel_id = runtime.compile(&source, "conv2d_implicit_gemm", DeviceBackend::Cuda).unwrap();
    runtime.launch(kernel_id, grid, block, smem_bytes, vec![
        KernelArg::Buffer(d_input),
        KernelArg::Buffer(d_weight),
        KernelArg::Buffer(d_output),
    ]).unwrap();

    let mut h_output_gpu_f16 = vec![half::f16::ZERO; output_size];
    runtime.copy_from_device(d_output, &mut h_output_gpu_f16).unwrap();

    let h_output_gpu: Vec<f32> = h_output_gpu_f16.iter().map(|x| x.to_f32()).collect();

    println!("\n=== Comparison Results ===");
    let mut max_abs_diff = 0.0f32;
    for i in 0..output_size {
        let diff = (h_output_cpu[i] - h_output_gpu[i]).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    println!("Max Absolute Diff: {:.6}", max_abs_diff);

    if max_abs_diff < 5e-2 {
        println!("✅ PASSED: Conv2d numerical accuracy verified!");
    } else {
        println!("❌ FAILED: Numerical accuracy too low");
        for i in 0..8.min(output_size) {
            println!("  [{}] CPU: {:.4}, GPU: {:.4}, diff: {:.6}", i, h_output_cpu[i], h_output_gpu[i], (h_output_cpu[i] - h_output_gpu[i]).abs());
        }
    }
}
