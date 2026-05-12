use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};
use tracea::emitter::universal::UniversalEmitter;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType};

#[test]
#[cfg(target_os = "macos")]
fn test_metal_elementwise_relu() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Metal)).expect("Failed to init Metal");
    let n = 1024;
    let size = n * 4; // float

    let buf_in = runtime.alloc(size, DeviceBackend::Metal).unwrap();
    let buf_out = runtime.alloc(size, DeviceBackend::Metal).unwrap();

    let mut data_in = vec![0.0f32; n];
    for i in 0..n {
        data_in[i] = (i as f32) - 512.0;
    }
    runtime.copy_to_device(buf_in, &data_in).unwrap();

    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Elementwise {
            op_type: tracea::core::op::ElementwiseType::Relu,
            n,
        },
        precison: "f32".to_string(),
        tiling: tracea::PipelineConfig::new(1, 1, 1, 1), // Tiles not used for elementwise currently
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let emitter = UniversalEmitter::new(DeviceBackend::Metal);
    let source = emitter.generate(ir).unwrap();
    
    let kernel_id = runtime.compile(&source, "elementwise_relu", DeviceBackend::Metal).unwrap();
    
    // Allocate l1_map and l2_table (TileMetadata) buffers required by the Metal elementwise kernel
    let l1_data = vec![0u32];
    let l1_bytes: Vec<u8> = l1_data.iter().flat_map(|x| x.to_ne_bytes().to_vec()).collect();
    let buf_l1 = runtime.alloc(l1_bytes.len(), DeviceBackend::Metal).unwrap();
    runtime.copy_to_device(buf_l1, &l1_bytes).unwrap();

    // TileMetadata: 5 * u32 = 20 bytes (region_m = 0)
    let l2_bytes = vec![0u8; 20];
    let buf_l2 = runtime.alloc(l2_bytes.len(), DeviceBackend::Metal).unwrap();
    runtime.copy_to_device(buf_l2, &l2_bytes).unwrap();

    let threads_per_grid = (n as u32, 1, 1);
    let threads_per_group = (1024, 1, 1); // 1 threadgroup of 1024 threads matches 1024 elements
    
    let args = vec![
        KernelArg::Buffer(buf_in),
        KernelArg::Buffer(buf_out),
        KernelArg::Int(n as i32),
        KernelArg::Buffer(buf_l1),
        KernelArg::Buffer(buf_l2),
    ];

    runtime.launch(kernel_id, threads_per_grid, threads_per_group, 0, args).unwrap();
    runtime.synchronize();

    let mut data_out = vec![0.0f32; n];
    runtime.copy_from_device(buf_out, &mut data_out).unwrap();

    for i in 0..n {
        let expected = if data_in[i] > 0.0 { data_in[i] } else { 0.0 };
        assert!((data_out[i] - expected).abs() < 1e-5, "Mismatch at {}: got {}, special {}", i, data_out[i], expected);
    }
    println!("Metal Elementwise ReLU Passed!");
}

#[test]
#[cfg(target_os = "macos")]
fn test_metal_softmax() {
    let runtime = RuntimeManager::init(Some(DeviceBackend::Metal)).expect("Failed to init Metal");
    let dim_size = 128;
    let total_elements = 128;
    let size = total_elements * 4;

    let buf_in = runtime.alloc(size, DeviceBackend::Metal).unwrap();
    let buf_out = runtime.alloc(size, DeviceBackend::Metal).unwrap();

    let data_in = vec![1.0f32; total_elements];
    runtime.copy_to_device(buf_in, &data_in).unwrap();

    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Softmax {
            axis: -1,
            dim_size,
            stride: 1,
            total_elements,
        },
        precison: "f32".to_string(),
        tiling: tracea::PipelineConfig::new(1, 1, 1, 1),
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    let emitter = UniversalEmitter::new(DeviceBackend::Metal);
    let source = emitter.generate(ir).unwrap();
    
    let kernel_id = runtime.compile(&source, "softmax_kernel", DeviceBackend::Metal).unwrap();
    
    let args = vec![
        KernelArg::Buffer(buf_in),
        KernelArg::Buffer(buf_out),
    ];

    runtime.launch(kernel_id, (1, 1, 1), (1, 1, 1), 0, args).unwrap();
    runtime.synchronize();

    let mut data_out = vec![0.0f32; total_elements];
    runtime.copy_from_device(buf_out, &mut data_out).unwrap();

    let expected = 1.0 / (dim_size as f32);
    for i in 0..total_elements {
        assert!((data_out[i] - expected).abs() < 1e-5, "Softmax mismatch at {}: got {}, expected {}", i, data_out[i], expected);
    }
    println!("Metal Softmax Passed!");
}
