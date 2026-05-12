use crate::runtime::plan::*;
use crate::runtime::manager::{RuntimeManager, BufferId, DeviceBackend};
use std::collections::HashMap;

pub mod capture;

#[derive(Debug)]
pub struct Executor {
    pub backend: DeviceBackend,
}

impl Executor {
    pub fn new(backend: DeviceBackend) -> Self {
        Self { backend }
    }

    /// Execute the plan with the given inputs.
    ///
    /// # Safety
    /// This function assumes the ExecutionPlan is valid and that input buffers match
    /// the shapes expected by the plan. No validation is performed here.
    pub fn execute(
        &self,
        plan: &ExecutionPlan,
        inputs: &HashMap<u64, BufferId>,
        manager: &RuntimeManager,
    ) -> Result<HashMap<u64, BufferId>, String> {
        // 1. Allocate Arena
        let (arena_id, _arena_ptr) = if self.backend == DeviceBackend::Metal {
            #[cfg(target_os = "macos")]
            { manager.get_arena_metal_buffer(plan.arena_size)? }
            #[cfg(not(target_os = "macos"))]
            { return Err("Metal not supported on this platform".into()); }
        } else {
            (BufferId(0), 0)
        };

        // 2. Dispatch Steps
        for step in &plan.steps {
            match step {
                ExecutionStep::LaunchKernel { kernel_id, grid_size, block_size, shared_mem_bytes, args } => {
                    let mut kernel_args = Vec::with_capacity(args.len());
                    for arg in args {
                        match arg {
                            KernelArgSpec::ArenaOffset(offset) => {
                                kernel_args.push(crate::runtime::manager::KernelArg::BufferOffset(arena_id, *offset));
                            },
                            KernelArgSpec::ExternalInput(op_id) => {
                                let buf_id = inputs.get(op_id)
                                    .ok_or_else(|| format!("Missing input for op_id {}", op_id))?;
                                kernel_args.push(crate::runtime::manager::KernelArg::Buffer(*buf_id));
                            },
                            KernelArgSpec::ScalarInt(val) => {
                                kernel_args.push(crate::runtime::manager::KernelArg::Int(*val));
                            },
                            KernelArgSpec::ScalarFloat(val) => {
                                kernel_args.push(crate::runtime::manager::KernelArg::Float(*val));
                            },
                            KernelArgSpec::Bytes(bytes) => {
                                kernel_args.push(crate::runtime::manager::KernelArg::Bytes(bytes.clone()));
                            },
                        }
                    }

                    manager.launch_kernel_by_id(
                        crate::runtime::manager::KernelId(*kernel_id as u64),
                        *grid_size,
                        *block_size,
                        *shared_mem_bytes,
                        kernel_args,
                    )?;
                },
                ExecutionStep::Memcpy { src_offset, dst_offset, size } => {
                    manager.memcpy_d2d(arena_id, *src_offset, arena_id, *dst_offset, *size)?;
                }
            }
        }

        // 3. Retrieve Outputs
        let mut results = HashMap::new();
        for (op_id, (offset, size)) in &plan.output_map {
             let output_buf = manager.alloc(*size, self.backend)?;
             manager.memcpy_d2d(arena_id, *offset, output_buf, 0, *size)?;
             results.insert(*op_id, output_buf);
        }

        Ok(results)
    }

    /// Capture the execution plan as a CapturedGraph for zero-overhead replay.
    pub fn capture(
        &self,
        plan: &ExecutionPlan,
        _inputs: &HashMap<u64, BufferId>,
        manager: &RuntimeManager,
    ) -> Result<capture::CapturedGraph, String> {
        #[cfg(target_os = "macos")]
        {
            use metal;
            let metal_dev = {
                let devices = manager.devices.lock().unwrap();
                let device_handle = devices.get(&DeviceBackend::Metal).ok_or("Metal backend not available")?;
                device_handle.metal_dev.as_ref().ok_or("Metal device not initialized")?.clone()
            };

            let mut static_size = 0;
            for step in &plan.steps {
                if let ExecutionStep::LaunchKernel { args, .. } = step {
                    for arg in args {
                        match arg {
                            KernelArgSpec::ScalarInt(_) | KernelArgSpec::ScalarFloat(_) => static_size += 4,
                            KernelArgSpec::Bytes(b) => static_size += b.len(),
                            _ => {}
                        }
                        static_size = (static_size + 255) & !255;
                    }
                }
            }

            let static_buf = if static_size > 0 {
                Some(metal_dev.device.new_buffer(static_size as u64, metal::MTLResourceOptions::StorageModeShared))
            } else {
                None
            };
            let static_ptr = static_buf.as_ref().map(|b| b.contents() as *mut u8);
            let mut current_static_offset = 0;

            let mut linear_steps = Vec::new();
            for step in &plan.steps {
                if let ExecutionStep::LaunchKernel { kernel_id, grid_size, block_size, args, .. } = step {
                    let pso = {
                        let kernels = manager.kernels.lock().unwrap();
                        let record = kernels.get(&crate::runtime::manager::KernelId(*kernel_id as u64)).ok_or("Kernel not found")?;
                        match &record.handle {
                            crate::runtime::manager::kernel::KernelHandle::Metal { pipeline, .. } => pipeline.clone(),
                            _ => return Err("Expected Metal kernel".into()),
                        }
                    };

                    let grid = metal::MTLSize::new(grid_size.0 as u64, grid_size.1 as u64, grid_size.2 as u64);
                    let block = metal::MTLSize::new(block_size.0 as u64, block_size.1 as u64, block_size.2 as u64);

                    let mut bindings = Vec::new();
                    for (i, arg) in args.iter().enumerate() {
                        match arg {
                            KernelArgSpec::ArenaOffset(offset) => {
                                bindings.push(capture::FastBinding::Arena { index: i as u32, offset: *offset as u64 });
                            }
                            KernelArgSpec::ExternalInput(op_id) => {
                                bindings.push(capture::FastBinding::Input { index: i as u32, op_id: *op_id });
                            }
                            KernelArgSpec::ScalarInt(val) => {
                                if let Some(ptr) = static_ptr {
                                    unsafe { std::ptr::copy_nonoverlapping(val as *const i32 as *const u8, ptr.add(current_static_offset), 4); }
                                    bindings.push(capture::FastBinding::Static { index: i as u32, offset: current_static_offset as u64 });
                                    current_static_offset = (current_static_offset + 4 + 255) & !255;
                                }
                            }
                            KernelArgSpec::ScalarFloat(val) => {
                                if let Some(ptr) = static_ptr {
                                    unsafe { std::ptr::copy_nonoverlapping(val as *const f32 as *const u8, ptr.add(current_static_offset), 4); }
                                    bindings.push(capture::FastBinding::Static { index: i as u32, offset: current_static_offset as u64 });
                                    current_static_offset = (current_static_offset + 4 + 255) & !255;
                                }
                            }
                            KernelArgSpec::Bytes(bytes) => {
                                if let Some(ptr) = static_ptr {
                                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(current_static_offset), bytes.len()); }
                                    bindings.push(capture::FastBinding::Static { index: i as u32, offset: current_static_offset as u64 });
                                    current_static_offset = (current_static_offset + bytes.len() + 255) & !255;
                                }
                            }
                        }
                    }

                    linear_steps.push(capture::LinearStep {
                        pipeline: pso,
                        grid,
                        block,
                        bindings,
                    });
                }
            }

            Ok(capture::CapturedGraph {
                steps: linear_steps,
                arena_size: plan.arena_size,
                static_args: static_buf,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = plan; let _ = inputs; let _ = manager;
            Err("Capture only supported on macOS".into())
        }
    }
}
