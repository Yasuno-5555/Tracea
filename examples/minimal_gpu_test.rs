/// Minimal ConvTranspose2d test - just verify data loading works
use tracea::*;
use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};

fn main() {
    println!("=== Minimal ConvTranspose2d Data Loading Test ===");
    
    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda)).expect("CUDA init failed");
    let backend = DeviceBackend::Cuda;

    // Simple test: just copy input to output to verify buffers work
    let size = 64;
    let d_input = runtime.alloc_f32(size, backend).expect("Alloc input failed");
    let d_output = runtime.alloc_f32(size, backend).expect("Alloc output failed");

    // Initialize with sequential values
    let h_input: Vec<f32> = (0..size).map(|i| i as f32).collect();
    runtime.copy_to_device(d_input, &h_input).expect("Copy input failed");

    // Minimal copy kernel
    let source = r#"
extern "C" __global__ void copy_test(const float* input, float* output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < 64) output[i] = input[i] * 2.0f; // Multiply by 2 to verify processing
}
"#;

    let kernel_name = "copy_test";
    let kernel_id = match runtime.compile(source, kernel_name, backend) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Compilation failed: {:?}", e);
            return;
        }
    };

    let args = vec![
        KernelArg::Buffer(d_input),
        KernelArg::Buffer(d_output),
    ];

    match runtime.launch(kernel_id, (1, 1, 1), (64, 1, 1), 0, args) {
        Ok(_) => println!("Kernel launched successfully"),
        Err(e) => {
            eprintln!("Launch failed: {:?}", e);
            return;
        }
    }

    runtime.synchronize();

    // Read back result
    let mut h_output = vec![0.0f32; size];
    runtime.copy_from_device(d_output, &mut h_output).expect("Copy output failed");

    // Print sample output
    println!("\nOutput sample (first 8 elements):");
    for (i, v) in h_output.iter().take(8).enumerate() {
        println!("  [{}]: {:.2} (expected: {:.2})", i, v, (i as f32) * 2.0);
    }

    let sum: f32 = h_output.iter().sum();
    if sum > 0.0 {
        println!("\n✅ Data transfer works: sum={:.2}", sum);
    } else {
        println!("\n❌ Data transfer failed: output is zeros");
    }
}
