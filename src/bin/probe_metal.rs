fn main() {
    #[cfg(target_os = "macos")]
    {
        use metal::*;
        let device = Device::system_default().expect("No device");
        let queue = device.new_command_buffer(); // Wait, new_command_queue
        let queue = device.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(MTLIndirectCommandType::ConcurrentDispatch);
        let icb = device.new_indirect_command_buffer_with_descriptor(&desc, 10, MTLResourceOptions::StorageModeShared);
        
        enc.execute_commands_in_buffer(&icb, NSRange::new(0, 1));
        
        enc.end_encoding();
        println!("Probe successful");
    }
}
