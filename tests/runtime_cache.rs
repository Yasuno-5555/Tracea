use tracea::runtime::manager::{RuntimeManager, DeviceBackend};

#[test]
fn test_runtime_kernel_compilation_cache() {
    let runtime = match RuntimeManager::init(None) {
        Ok(r) => r,
        Err(_) => {
            println!("No device runtime available. Skipping cache test.");
            return;
        }
    };

    let mut backend = if cfg!(target_os = "macos") {
        DeviceBackend::Metal
    } else if tracea::emitter::rocm_driver::RocmDriverApi::get().is_some() {
        DeviceBackend::Rocm
    } else {
        DeviceBackend::Cuda
    };

    #[cfg(feature = "vulkan")]
    {
        if runtime.devices.lock().unwrap().contains_key(&DeviceBackend::Vulkan) {
            backend = DeviceBackend::Vulkan;
        }
    }

    let source = if backend == DeviceBackend::Metal {
        r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void mock_kernel(device float* out [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) {
            out[tid] = 1.0f;
        }
        "#
    } else {
        r#"
        extern "C" __global__ void mock_kernel(float* out) {
            out[0] = 1.0f;
        }
        "#
    };

    // First compilation
    let id1 = runtime.compile(source, "mock_kernel", backend);
    if id1.is_err() {
        println!("Compilation not supported on this platform/backend. Skipping test.");
        return;
    }
    let id1 = id1.unwrap();

    // Second compilation (with exact same source and name)
    let id2 = runtime.compile(source, "mock_kernel", backend).expect("Second compile failed");

    // Verify cache hit: The generated KernelIds must be identical
    assert_eq!(id1, id2, "Kernel compilation cache missed! IDs must be identical.");
    println!("Cache hit verified. Kernel ID: {:?}", id1);
}
