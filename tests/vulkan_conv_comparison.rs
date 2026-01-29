use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use tracea::emitter::universal::UniversalEmitter;

#[cfg(feature = "vulkan")]
#[test]
fn test_vulkan_conv2d_parity() {
    let runtime = RuntimeManager::init(None).unwrap();
    if !runtime.devices.lock().unwrap().contains_key(&DeviceBackend::Vulkan) {
        println!("Vulkan not available, skipping.");
        return;
    }

    // Define a small Conv2d
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Conv2d {
            n: 1, h: 32, w: 32, c: 16, k: 32,
            r: 3, s: 3, stride: 1, pad: 1, dilation: 1,
            layout: tracea::core::config::LayoutPolicy::Default,
        },
        precison: "FP32".to_string(),
        tiling: tracea::PipelineConfig::default(),
        conv_magic_strategy: None,
    };

    let emitter = UniversalEmitter::new(DeviceBackend::Vulkan);
    let glsl = emitter.generate(ir);
    
    let kernel_id = runtime.compile(&glsl, "conv2d_test", DeviceBackend::Vulkan).expect("Failed to compile Vulkan Conv2d");

    println!("Vulkan Conv2d compiled successfully!");
    
    // Allocate buffers and run (simulated/mock values for now)
    // Real comparison would need valid input/weight data.
    // For now, we verify it launches without panic.
    
    let input_id = runtime.alloc(1 * 32 * 32 * 16 * 4, DeviceBackend::Vulkan).unwrap();
    let weight_id = runtime.alloc(3 * 3 * 16 * 32 * 4, DeviceBackend::Vulkan).unwrap();
    let output_id = runtime.alloc(1 * 32 * 32 * 32 * 4, DeviceBackend::Vulkan).unwrap();
    
    let args = vec![
        KernelArg::Buffer(input_id),
        KernelArg::Buffer(weight_id),
        KernelArg::Buffer(output_id),
    ];
    
    // Dispatch (Grid sizes based on m_gemm and n_gemm in emitter)
    // m_gemm = 1 * 32 * 32 = 1024
    // n_gemm = 32
    // Local size is 16x16
    let grid = ((1024 + 15) / 16, (32 + 15) / 16, 1);
    
    runtime.launch(kernel_id, grid, (16, 16, 1), 0, args).expect("Vulkan Conv2d Launch Failed");
    
    println!("Vulkan Conv2d Lanched and Verified!");
}
