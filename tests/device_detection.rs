use tracea::core::device::{DeviceProfile, BackendType};

#[test]
fn test_device_detection() {
    let profile = DeviceProfile::detect();
    println!("Detected Device Profile: {:?}", profile);

    #[cfg(target_os = "macos")]
    {
        // On macOS (development machine), we expect Metal detection
        assert_eq!(profile.backend, BackendType::Metal);
        assert!(profile.name.len() > 0);
        assert_eq!(profile.simd_width, 32);
        assert!(profile.max_threads_per_block >= 1024);
        
        // Check for Apple Silicon specific features if applicable
        if profile.name.contains("Apple") {
            assert!(profile.has_tensor_cores, "Apple Silicon should have AMX/TensorCores enabled");
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        // On other platforms, checking for CUDA or CPU fallback
    }
}
