use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::doctor::visualizer::Visualizer;
use std::sync::Arc;

#[cfg(feature = "vulkan")]
#[test]
fn test_cuda_vulkan_parity() {
    let runtime = RuntimeManager::init(None).unwrap();
    let has_cuda = runtime.devices.lock().unwrap().contains_key(&DeviceBackend::Cuda);
    let has_vulkan = runtime.devices.lock().unwrap().contains_key(&DeviceBackend::Vulkan);

    if !has_cuda || !has_vulkan {
        println!("Test requires both CUDA and Vulkan backends. Skipping.");
        return;
    }

    let visualizer = Visualizer::new(runtime.clone());
    let size = 1024;
    
    // Allocate and initialize buffers on both backends
    let b_cuda = runtime.alloc(size * 4, DeviceBackend::Cuda).unwrap();
    let b_vulkan = runtime.alloc(size * 4, DeviceBackend::Vulkan).unwrap();

    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
    let u8_data: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size * 4) };

    runtime.write(b_cuda, u8_data).unwrap();
    runtime.write(b_vulkan, u8_data).unwrap();

    // Compute error map
    let report = visualizer.compare_tensors(b_cuda, b_vulkan, size).expect("Comparison failed");

    println!("Numerical Parity Report:");
    println!("  MAE: {}", report.max_abs_error);
    println!("  MSE: {}", report.mean_squared_error);
    if let Some(heatmap) = report.heatmap {
        println!("  Heatmap:\n{}", heatmap);
    }

    assert!(report.max_abs_error < 1e-5, "Numerical deviation too high!");
}
