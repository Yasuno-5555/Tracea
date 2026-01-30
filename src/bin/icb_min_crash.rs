#[cfg(target_os = "macos")]
use metal;

#[cfg(target_os = "macos")]
fn main() {
    println!("[ICB Min] Starting minimal ICB test...");
    let device = metal::Device::system_default().expect("No Metal device");
    println!("[ICB Min] Device: {}", device.name());

    let desc = metal::IndirectCommandBufferDescriptor::new();
    desc.set_command_types(metal::MTLIndirectCommandType::ConcurrentDispatch);
    desc.set_inherit_buffers(false);
    desc.set_inherit_pipeline_state(false);
    desc.set_max_kernel_buffer_bind_count(8);

    println!("[ICB Min] Creating ICB...");
    let icb = device.new_indirect_command_buffer_with_descriptor(
        &desc, 
        1, 
        metal::MTLResourceOptions::StorageModeShared
    );
    
    println!("[ICB Min] ICB size: {}", icb.size());
    
    // Create a dummy library/kernel to get a PSO
    let source = "
        #include <metal_stdlib>
        using namespace metal;
        kernel void dummy(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            out[id] = 1.0;
        }
    ";
    let options = metal::CompileOptions::new();
    let library = device.new_library_with_source(source, &options).expect("Compile failed");
    let func = library.get_function("dummy", None).expect("Function not found");
    let pipeline = device.new_compute_pipeline_state_with_function(&func).expect("PSO failed");

    println!("[ICB Min] Getting command 0...");
    // We use the binding first, if it crashes, we know for sure.
    let cmd = icb.indirect_compute_command_at_index(0);
    println!("[ICB Min] Setting pipeline...");
    cmd.set_compute_pipeline_state(&pipeline);
    
    println!("[ICB Min] SUCCESS. ICB might actually work if simple?");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("ICB test is only supported on macOS.");
}
