use tracea::runtime::manager::{RuntimeManager, DeviceBackend};

#[cfg(feature = "vulkan")]
#[test]
fn test_vulkan_init_and_alloc() {
    let runtime = RuntimeManager::init(None).unwrap();
    
    // Check if Vulkan is available
    let has_vulkan = runtime.devices.lock().unwrap().contains_key(&DeviceBackend::Vulkan);
    if !has_vulkan {
        println!("Vulkan not available on this system, skipping test.");
        return;
    }
    
    println!("Vulkan detected, proceeding with allocation test...");
    
    // Test Allocation
    let size = 1024;
    let buf_id = runtime.alloc(size, DeviceBackend::Vulkan).expect("Failed to allocate Vulkan buffer");
    
    // Test Write
    let data_vec: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    runtime.write(buf_id, &data_vec).expect("Failed to write to Vulkan buffer");
    
    // Test Read
    let mut out_vec = vec![0u8; size];
    runtime.read(buf_id, &mut out_vec).expect("Failed to read from Vulkan buffer");
    
    assert_eq!(data_vec, out_vec, "Read data does not match written data");
    println!("Vulkan Read/Write verification successful!");
}
