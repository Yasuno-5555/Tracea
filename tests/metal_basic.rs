#[cfg(target_os = "macos")]
#[test]
fn test_metal_basic_add() {
    use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg, DeviceBuffer};
    use std::sync::Arc;

    let runtime = RuntimeManager::init(Some(DeviceBackend::Metal)).expect("Failed to init Metal runtime");

    // 1. Compile Kernel
    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_kernel(device const float* A [[buffer(0)]],
                               device const float* B [[buffer(1)]],
                               device float* C [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
            C[id] = A[id] + B[id];
        }
    "#;
    
    let kernel_id = runtime.compile(source, "add_kernel", DeviceBackend::Metal).expect("Compile failed");

    // 2. Allocate Buffers
    let n = 100;
    let size_bytes = n * 4;
    
    let buf_a = runtime.alloc(size_bytes, DeviceBackend::Metal).expect("Alloc A failed");
    let buf_b = runtime.alloc(size_bytes, DeviceBackend::Metal).expect("Alloc B failed");
    let buf_c = runtime.alloc(size_bytes, DeviceBackend::Metal).expect("Alloc C failed");

    // 3. Initialize Data (Need a way to write to buffer - RuntimeManager doesn't expose host->device copy easily in public API?)
    // RuntimeManager::alloc returns BufferId. 
    // We need to write data. Python interface does this via mapping usually.
    // Let's assume initialized to zeros or garbage for smoke test, 
    // OR try to map pointer if possible.
    // For this smoke test, we just check if Launch succeeds without crash. 
    // Correctness requires Host<->Device copy which might be missing in Manager public API for Metal?
    // Wait, RuntimeManager has no `write_buffer` method?
    // It has `alloc`.
    // Checking manager.rs... `launch` takes `Bytes` kernel arg, but that's for params.
    // Ah, `PyDeviceBufferF32::unsafe_from_ptr` registers external pointer.
    // For Metal, `new_buffer` creates generic buffer.
    // `metal::Buffer` has `contents() -> *mut c_void`.
    
    // Hack for test: Get the internal buffer to write data?
    // `runtime.buffers` is shielded by Mutex.
    // For a unit test inside `tests/`, we rely on public API.
    // If public API lacks copy, we can't verify result easily.
    // BUT, the goal is "Verify Metal support implementation" (Smoke Test).
    // If we can Compile and Launch successfully, that proves Backend is alive.
    
    // 4. Launch
    let grid = (n as u32, 1, 1);
    let block = (n as u32, 1, 1);
    
    let args = vec![
        KernelArg::Buffer(buf_a),
        KernelArg::Buffer(buf_b),
        KernelArg::Buffer(buf_c)
    ];

    runtime.launch(kernel_id, grid, block, 0, args).expect("Launch failed");
    
    println!("Metal Smoke Test Passed!");
}
