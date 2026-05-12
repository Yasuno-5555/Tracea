use crate::runtime::manager::{BufferId, DeviceBackend, RuntimeManager};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum FastBinding {
    Arena { index: u32, offset: u64 },
    Input { index: u32, op_id: u64 },
    Static { index: u32, offset: u64 },
}

#[derive(Debug, Clone)]
pub struct LinearStep {
    pub pipeline: metal::ComputePipelineState,
    pub grid: metal::MTLSize,
    pub block: metal::MTLSize,
    pub bindings: Vec<FastBinding>,
}

#[derive(Debug)]
pub struct CapturedGraph {
    pub steps: Vec<LinearStep>,
    pub arena_size: usize,
    pub static_args: Option<metal::Buffer>,
}

impl CapturedGraph {
    /// Execute captured graph via direct Metal command encoding.
    /// Processes all steps sequentially — no ICB overhead but no ICB limitation either.
    pub fn execute(&self, manager: &RuntimeManager, inputs: &HashMap<u64, BufferId>) -> Result<(), String> {
        #[cfg(target_os = "macos")]
        {
            let devices = manager.devices.lock().unwrap();
            let metal_dev = devices.get(&DeviceBackend::Metal)
                .and_then(|h| h.metal_dev.as_ref())
                .ok_or("Metal device not initialized")?;

            let command_buffer = metal_dev.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Resolve arena buffer once
            let (arena_id, _) = manager.get_arena_metal_buffer(self.arena_size)?;
            let bufs = manager.buffers.lock().unwrap();
            let arena_buf = match bufs.get(&arena_id) {
                Some(crate::runtime::manager::memory::DeviceBuffer::Metal(b)) => b,
                _ => return Err("Arena is not a Metal buffer".into()),
            };

            for step in &self.steps {
                encoder.set_compute_pipeline_state(&step.pipeline);

                // Bind resolved buffers
                for binding in &step.bindings {
                    let buf = match binding {
                        FastBinding::Arena { offset, .. } => {
                            Some((arena_buf, *offset))
                        }
                        FastBinding::Input { op_id, .. } => {
                            let buf_id = inputs.get(op_id).ok_or("Missing input")?;
                            match bufs.get(buf_id) {
                                Some(crate::runtime::manager::memory::DeviceBuffer::Metal(b)) => {
                                    Some((b, 0u64))
                                }
                                _ => None,
                            }
                        }
                        FastBinding::Static { offset, .. } => {
                            self.static_args.as_ref().map(|b| (b, *offset))
                        }
                    };
                    if let Some((buf, offset)) = buf {
                        encoder.set_buffer(binding.index() as u64, Some(buf), offset);
                    }
                }

                encoder.dispatch_thread_groups(step.grid, step.block);
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = manager; let _ = inputs;
            Err("Capture execution only supported on macOS".into())
        }
    }

}

// Helper for FastBinding
impl FastBinding {
    pub fn index(&self) -> u32 {
        match self {
            FastBinding::Arena { index, .. } => *index,
            FastBinding::Input { index, .. } => *index,
            FastBinding::Static { index, .. } => *index,
        }
    }
}
