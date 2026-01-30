use crate::runtime::plan::*;
use crate::runtime::manager::{RuntimeManager, BufferId, DeviceBackend};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub mod capture {
    use crate::runtime::manager::{RuntimeManager, BufferId, DeviceBackend};
    use std::collections::HashMap;

    #[cfg(target_os = "macos")]
    use metal;

    #[derive(Debug, Clone)]
    pub enum FastBinding {
        /// Binding from the arena buffer at a fixed offset
        Arena { index: u32, offset: u64 },
        /// Binding from a user-provided input/output tensor
        Input { index: u32, op_id: u64 },
        /// Binding from a pre-cooked static argument buffer (scalars, etc.)
        Static { index: u32, offset: u64 },
    }

    #[derive(Debug, Clone)]
    pub struct LinearStep {
        #[cfg(target_os = "macos")]
        pub pipeline: metal::ComputePipelineState,
        #[cfg(target_os = "macos")]
        pub grid: metal::MTLSize,
        #[cfg(target_os = "macos")]
        pub block: metal::MTLSize,
        pub bindings: Vec<FastBinding>,
    }

    #[derive(Debug)]
    pub struct CapturedGraph {
        pub steps: Vec<LinearStep>,
        pub arena_size: usize,
        #[cfg(target_os = "macos")]
        pub static_args: Option<metal::Buffer>,
        #[cfg(target_os = "macos")]
        pub icb: Option<std::sync::Arc<crate::backend::metal::capture::MetalGraphCapture>>,
    }

    impl CapturedGraph {
        pub fn execute(&self, manager: &RuntimeManager, inputs: &HashMap<u64, BufferId>) -> Result<(), String> {
            #[cfg(target_os = "macos")]
            {
                // 1. Get Buffer Locks (minimizing scope)
                let bufs_lock = manager.buffers.lock().unwrap();
                let (arena_id, _) = manager.get_arena_metal_buffer(self.arena_size)?;
                let arena_buf = match bufs_lock.get(&arena_id).ok_or("Arena buffer missing")? {
                    crate::runtime::manager::memory::DeviceBuffer::Metal(b) => b,
                    _ => return Err("Invalid arena backend".into()),
                };

                // 2. Get Metal Queue
                let devices = manager.devices.lock().unwrap();
                let metal_dev = devices.get(&DeviceBackend::Metal)
                    .and_then(|h| h.metal_dev.as_ref())
                    .ok_or("Metal device not initialized")?;
                
                let command_buffer = metal_dev.queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                // 3. Ultra-Linear Loop (Zero branching on critical path)
                for step in &self.steps {
                    encoder.set_compute_pipeline_state(&step.pipeline);
                    for binding in &step.bindings {
                        match binding {
                            FastBinding::Arena { index, offset } => {
                                encoder.set_buffer(*index as u64, Some(arena_buf), *offset);
                            }
                            FastBinding::Input { index, op_id } => {
                                let bid = inputs.get(op_id).ok_or("Input missing")?;
                                if let crate::runtime::manager::memory::DeviceBuffer::Metal(b) = bufs_lock.get(bid).unwrap() {
                                    encoder.set_buffer(*index as u64, Some(b), 0);
                                }
                            }
                            FastBinding::Static { index, offset } => {
                                if let Some(ref sbuf) = self.static_args {
                                    encoder.set_buffer(*index as u64, Some(sbuf), *offset);
                                }
                            }
                        }
                    }
                    encoder.dispatch_thread_groups(step.grid, step.block);
                }

                encoder.end_encoding();
                command_buffer.commit();
                return Ok(());
            }

            #[cfg(not(target_os = "macos"))]
            {
                let _ = manager; let _ = inputs;
                Err("Metal not supported".into())
            }
        }
    }
}

#[derive(Debug)]
pub struct GraphExecutor {
    // Backend handles (e.g., Metal, CUDA)
    // For now, we assume Metal is the primary backend
    backend: crate::runtime::manager::DeviceBackend,
}

impl GraphExecutor {
    pub fn new(backend: crate::runtime::manager::DeviceBackend) -> Self {
        Self {
            backend,
        }
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
        manager: &crate::runtime::manager::RuntimeManager,
    ) -> Result<HashMap<u64, BufferId>, String> {
        // 1. Allocate Arena
        let (arena_id, _arena_ptr) = if self.backend == DeviceBackend::Metal {
            #[cfg(target_os = "macos")]
            { manager.get_arena_metal_buffer(plan.arena_size)? }
            #[cfg(not(target_os = "macos"))]
            { return Err("Metal not supported on this platform".into()); }
        } else if self.backend == DeviceBackend::Cuda {
            // For CUDA, we use the unified alloc/arena if implemented, 
            // or just skip if kernels handle their own memory for now.
            (BufferId(0), 0) 
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
                                // Pass the arena buffer + offset
                                kernel_args.push(crate::runtime::manager::KernelArg::BufferOffset(arena_id, *offset));
                            },
                            KernelArgSpec::ExternalInput(op_id) => {
                                // Resolve external input buffer
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

                    // Launch via Manager
                    manager.launch_kernel_by_id(
                        crate::runtime::manager::KernelId(*kernel_id as u64),
                        *grid_size,
                        *block_size,
                        *shared_mem_bytes,
                        kernel_args,
                    )?;
                },
                ExecutionStep::Memcpy { src_offset, dst_offset, size } => {
                    // Intra-arena copy
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
    pub fn capture(
        &self,
        plan: &ExecutionPlan,
        inputs: &HashMap<u64, BufferId>,
        manager: &crate::runtime::manager::RuntimeManager,
    ) -> Result<capture::CapturedGraph, String> {
        #[cfg(target_os = "macos")]
        {
            use metal;
            let metal_dev = {
                let devices = manager.devices.lock().unwrap();
                let device_handle = devices.get(&DeviceBackend::Metal).ok_or("Metal backend not available")?;
                device_handle.metal_dev.as_ref().ok_or("Metal device not initialized")?.clone()
            };

            // 1. Pre-calculate Static Arguments Size
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

            // 2. Pre-resolve Steps
            let mut linear_steps = Vec::new();
            for step in &plan.steps {
                if let ExecutionStep::LaunchKernel { kernel_id, grid_size, block_size, args, .. } = step {
                    // Resolve Pipeline
                    let pso = {
                        let kernels = manager.kernels.lock().unwrap();
                        let record = kernels.get(&crate::runtime::manager::KernelId(*kernel_id as u64)).ok_or("Kernel not found")?;
                        match &record.handle {
                            crate::runtime::manager::kernel::KernelHandle::Metal { pipeline, .. } => pipeline.clone(),
                            _ => return Err("Expected Metal kernel".into()),
                        }
                    };

                    // Pre-convert grid/block to MTLSize
                    let grid = metal::MTLSize::new(grid_size.0 as u64, grid_size.1 as u64, grid_size.2 as u64);
                    let block = metal::MTLSize::new(block_size.0 as u64, block_size.1 as u64, block_size.2 as u64);

                    // Pre-resolve Bindings
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
                icb: None,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = plan; let _ = inputs; let _ = manager;
            Err("Capture only supported on macOS".into())
        }
    }
}
