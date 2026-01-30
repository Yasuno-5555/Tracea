#[cfg(target_os = "macos")]
use metal;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct MetalGraphCapture {
    pub icb: metal::IndirectCommandBuffer,
    pub arg_buffer: metal::Buffer,
    pub count: usize,
}

#[cfg(target_os = "macos")]
impl MetalGraphCapture {
    pub fn new(device: &metal::Device, max_commands: usize, arg_buffer_size: usize) -> Self {
        let desc = metal::IndirectCommandBufferDescriptor::new();
        desc.set_command_types(metal::MTLIndirectCommandType::ConcurrentDispatch);
        desc.set_inherit_buffers(false);
        desc.set_inherit_pipeline_state(false);
        desc.set_max_kernel_buffer_bind_count(31);

        let icb = device.new_indirect_command_buffer_with_descriptor(&desc, max_commands as u64, metal::MTLResourceOptions::StorageModeShared);
        
        let arg_buffer = device.new_buffer(arg_buffer_size as u64, metal::MTLResourceOptions::StorageModeShared);
        
        Self { icb, arg_buffer, count: max_commands }
    }
    
    pub fn encode_dispatch(
        &self,
        command_index: usize,
        pipeline: &metal::ComputePipelineState,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        buffers: &[(usize, &metal::Buffer, usize)], // (bind_index, buffer, offset)
    ) {
        let cmd = self.icb.indirect_compute_command_at_index(command_index as u64);
        cmd.set_compute_pipeline_state(pipeline);
        
        for (bind_index, buffer, offset) in buffers {
            cmd.set_kernel_buffer(*bind_index as u64, Some(buffer), *offset as u64);
        }
        
        cmd.concurrent_dispatch_threadgroups(
            metal::MTLSize { width: grid.0 as u64, height: grid.1 as u64, depth: grid.2 as u64 },
            metal::MTLSize { width: block.0 as u64, height: block.1 as u64, depth: block.2 as u64 },
        );
    }
}
