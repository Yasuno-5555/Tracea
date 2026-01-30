use crate::runtime::manager::DeviceBackend;
use crate::runtime::manager::RuntimeManager;
use crate::runtime::plan::ExecutionPlan;

#[derive(Debug)]
pub struct CapturedGraph {
    pub backend: DeviceBackend,
    #[cfg(target_os = "macos")]
    pub icb: Option<std::sync::Arc<crate::backend::metal::capture::MetalGraphCapture>>,
}

impl CapturedGraph {
    pub fn execute(&self, manager: &RuntimeManager) -> Result<(), String> {
        match self.backend {
            DeviceBackend::Metal => {
                #[cfg(target_os = "macos")]
                {
                    if let Some(icb_wrapper) = &self.icb {
                        // Get Metal Queue
                        if let Some(device_handle) = manager.devices.lock().unwrap().get(&DeviceBackend::Metal) {
                            if let Some(metal_backend) = &device_handle.metal_dev {
                                let command_buffer = metal_backend.queue.new_command_buffer();
                                let encoder = command_buffer.new_compute_command_encoder();
                                
                                // Execute ICB
                                // We need range. The wrapper knows it? No, wrapper has `icb` object.
                                // We need to know how many commands.
                                // Let's assume full range for now or store it.
                                let count = icb_wrapper.icb.len(); 
                                encoder.execute_commands_in_indirect_command_buffer(&icb_wrapper.icb, 0..count);
                                
                                encoder.end_encoding();
                                command_buffer.commit();
                                command_buffer.wait_until_completed();
                                return Ok(());
                            }
                        }
                    }
                    return Err("Metal Capture execution failed: Device or ICB not found".into());
                }
                #[cfg(not(target_os = "macos"))]
                {
                    return Err("Metal not supported on this OS".into());
                }
            },
            _ => Err("Capture not implemented for this backend".into()),
        }
    }
}
