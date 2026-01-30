use std::ffi::c_void;
use super::{KernelId, RuntimeManager, DeviceBackend, KernelHandle, BufferId};
use crate::doctor::KernelLaunchInfo;
#[cfg(feature = "vulkan")]
use ash::vk;

impl RuntimeManager {
    pub fn launch(&self, id: KernelId, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: Vec<KernelArg>) -> Result<(), String> {
        let recorded = self.kernels.lock().map_err(|_| "Lock")?.get(&id).ok_or("No kernel")?.clone();
        
        let profiler = crate::doctor::profiler::TraceProfiler::get();
        profiler.record(recorded.name.clone(), "B");
        // println!("[Runtime] Launching {}: Grid{:?}, Block{:?}, Smem: {}", recorded.name, grid, block, smem);

        let mut arg_store = [0u64; 64]; 
        let mut kernel_params = [std::ptr::null_mut() as *mut c_void; 64];

        for (i, arg) in args.iter().enumerate() {
            if i >= 64 { break; }
            match arg {
                KernelArg::Int(x) => {
                    let ptr = &mut arg_store[i] as *mut u64 as *mut i32;
                    unsafe { *ptr = *x; }
                    kernel_params[i] = ptr as *mut c_void;
                }
                KernelArg::Float(x) => {
                    let ptr = &mut arg_store[i] as *mut u64 as *mut f32;
                    unsafe { *ptr = *x; }
                    kernel_params[i] = ptr as *mut c_void;
                }
                KernelArg::Usize(x) => {
                    let ptr = &mut arg_store[i] as *mut u64;
                    unsafe { *ptr = *x as u64; }
                    kernel_params[i] = ptr as *mut c_void;
                }
                KernelArg::Buffer(bid) => {
                    let ptr_val = self.get_device_ptr(*bid)?;
                    let ptr = &mut arg_store[i] as *mut u64;
                    unsafe { *ptr = ptr_val; }
                    kernel_params[i] = ptr as *mut c_void;
                }
                KernelArg::Bytes(bytes) => {
                    kernel_params[i] = bytes.as_ptr() as *mut c_void;
                }
                KernelArg::BufferOffset(bid, offset) => {
                    let ptr_val = self.get_device_ptr(*bid)?;
                    let ptr = &mut arg_store[i] as *mut u64;
                    unsafe { *ptr = ptr_val + (*offset as u64); }
                    kernel_params[i] = ptr as *mut c_void;
                }
            }
        }

        match &recorded.handle {
            KernelHandle::Cuda { func, .. } => {
                unsafe {
                    let lib = cudarc::driver::sys::lib();
                    
                    let res = lib.cuLaunchKernel(
                        func.0,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        smem, std::ptr::null_mut(),
                        kernel_params.as_ptr() as *mut *mut c_void,
                        std::ptr::null_mut()
                    );

                    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS { 
                        let msg = format!("{:?}", res);
                        self.doctor.on_kernel_launch(KernelLaunchInfo {
                             backend: crate::doctor::BackendKind::Cuda,
                             kernel_name: recorded.name.clone(),
                             return_code: res as i32,
                             last_runtime_error: Some(msg),
                             grid: (grid.0, grid.1, grid.2), 
                             block: (block.0, block.1, block.2),
                             smem,
                        });
                        return Err(format!("CUDA Launch Failed: {:?}", res)); 
                    }
                    self.doctor.on_kernel_launch(KernelLaunchInfo {
                             backend: crate::doctor::BackendKind::Cuda,
                             kernel_name: recorded.name.clone(),
                             return_code: 0,
                             last_runtime_error: None,
                             grid: (grid.0, grid.1, grid.2), 
                             block: (block.0, block.1, block.2),
                             smem,
                    });
                }
            }
            KernelHandle::Rocm { func, .. } => {
                let api = crate::emitter::rocm_driver::RocmDriverApi::get().ok_or("ROCm API not found")?;
                unsafe {
                    let res = (api.hipModuleLaunchKernel)(
                        func.0,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        smem, std::ptr::null_mut(),
                        kernel_params.as_ptr() as *mut *mut c_void,
                        std::ptr::null_mut()
                    );
                    if res != 0 { return Err(format!("ROCm Launch Failed: {}", res)); }
                }
            }
            #[cfg(feature = "vulkan")]
            KernelHandle::Vulkan { func, .. } => {
                let vk_backend = self.devices.lock().map_err(|_| "Lock")?.get(&DeviceBackend::Vulkan).ok_or("Vulkan Device not found")?.vulkan_dev.clone().ok_or("Vulkan Backend not found")?;
                unsafe {
                    let device = &vk_backend.device;
                    let command_pool_info = vk::CommandPoolCreateInfo::builder()
                        .queue_family_index(vk_backend.queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
                    let pool = device.create_command_pool(&command_pool_info, None).map_err(|e| e.to_string())?;
                    
                    let alloc_info = vk::CommandBufferAllocateInfo::builder()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1);
                    let cmd_buf = device.allocate_command_buffers(&alloc_info).map_err(|e| e.to_string())?[0];
                    
                    device.begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| e.to_string())?;
                    device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, func.pipeline);
                    
                    device.cmd_dispatch(cmd_buf, grid.0, grid.1, grid.2);
                    
                    device.end_command_buffer(cmd_buf).map_err(|e| e.to_string())?;
                    device.queue_submit(vk_backend.queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd_buf]).build()], vk::Fence::null()).map_err(|e| e.to_string())?;
                    device.queue_wait_idle(vk_backend.queue).map_err(|e| e.to_string())?;
                    
                    device.destroy_command_pool(pool, None);
                }
            }
            #[cfg(target_os = "macos")]
            KernelHandle::Metal { pipeline, .. } => {
                let devices = self.devices.lock().map_err(|_| "Lock")?;
                let handle = devices.get(&DeviceBackend::Metal).ok_or("No Metal Device")?;
                let backend = handle.metal_dev.as_ref().ok_or("No Metal Backend instance")?;
                
                let command_buffer = backend.queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                
                encoder.set_compute_pipeline_state(pipeline);
                
                // Bind Args
                for (i, arg) in args.iter().enumerate() {
                    match arg {
                        KernelArg::Buffer(bid) => {
                            if *bid == BufferId(0) {
                                encoder.set_buffer(i as u64, None, 0);
                            } else {
                                let buf_guards = self.buffers.lock().map_err(|_| "Lock")?;
                                if let Some(buf) = buf_guards.get(bid) {
                                    match buf {
                                        #[cfg(target_os = "macos")]
                                        crate::runtime::manager::memory::DeviceBuffer::Metal(b) => {
                                            encoder.set_buffer(i as u64, Some(b), 0);
                                        }
                                        _ => return Err(format!("Arg {} is not a Metal buffer", i)),
                                    }
                                } else {
                                    return Err(format!("Arg {} (BufferId {:?}) not found", i, bid));
                                }
                            }
                        }
                        KernelArg::BufferOffset(bid, offset) => {
                            if *bid == BufferId(0) {
                                encoder.set_buffer(i as u64, None, *offset as u64);
                            } else {
                                let buf_guards = self.buffers.lock().map_err(|_| "Lock")?;
                                if let Some(buf) = buf_guards.get(bid) {
                                    match buf {
                                        #[cfg(target_os = "macos")]
                                        crate::runtime::manager::memory::DeviceBuffer::Metal(b) => {
                                            encoder.set_buffer(i as u64, Some(b), *offset as u64);
                                        }
                                        _ => return Err(format!("Arg {} is not a Metal buffer", i)),
                                    }
                                } else {
                                    return Err(format!("Arg {} (BufferId {:?}) not found", i, bid));
                                }
                            }
                        }
                        KernelArg::Int(val) => {
                             encoder.set_bytes(i as u64, std::mem::size_of::<i32>() as u64, val as *const i32 as *const _);
                        }
                        KernelArg::Float(val) => {
                             encoder.set_bytes(i as u64, std::mem::size_of::<f32>() as u64, val as *const f32 as *const _);
                        }
                        KernelArg::Usize(val) => {
                             encoder.set_bytes(i as u64, std::mem::size_of::<u64>() as u64, val as *const usize as *const _);
                        }
                        KernelArg::Bytes(data) => {
                             encoder.set_bytes(i as u64, data.len() as u64, data.as_ptr() as *const _);
                        }
                    }
                }
                
                let thread_group_count = metal::MTLSize { width: grid.0 as u64, height: grid.1 as u64, depth: grid.2 as u64 };
                let thread_group_size = metal::MTLSize { width: block.0 as u64, height: block.1 as u64, depth: block.2 as u64 };
                
                encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
                encoder.end_encoding();
                
                command_buffer.commit();
                
                self.doctor.on_kernel_launch(KernelLaunchInfo {
                    backend: crate::doctor::BackendKind::Metal,
                    kernel_name: recorded.name.clone(),
                    return_code: 0,
                    last_runtime_error: None,
                    grid: (grid.0, grid.1, grid.2), 
                    block: (block.0, block.1, block.2),
                    smem,
                });
                // command_buffer.wait_until_completed(); // Removed to enable async pipelining
            }
            _ => return Err("Unsupported backend".to_string()),
        }

        profiler.record(recorded.name.clone(), "E");
        Ok(())
    }

    pub fn launch_ttg(
        &self,
        kernel_id: KernelId,
        block: (u32, u32, u32),
        smem: u32,
        base_args: Vec<KernelArg>,
        ttg: &crate::runtime::ttg::DeviceTTG,
        epilogue_args: Vec<KernelArg>
    ) -> Result<(), String> {
        let mut final_args = base_args;
        final_args.push(KernelArg::BufferOffset(ttg.l1_buffer, ttg.l1_offset));
        final_args.push(KernelArg::BufferOffset(ttg.l2_buffer, ttg.l2_offset));
        final_args.extend(epilogue_args);

        let grid = (ttg.num_active_tiles, 1, 1);
        self.launch(kernel_id, grid, block, smem, final_args)
    }

    pub fn launch_with_policy(
        &self,
        kernel_id: KernelId,
        args: Vec<KernelArg>,
        op: &crate::policy::types::OperatorTopology,
        t_policy: &crate::policy::types::TilePolicy,
        e_policy: &crate::policy::types::ExecPolicy,
        epilogue_args: Vec<KernelArg>,
        backend: DeviceBackend,
    ) -> Result<(), String> {
        let layout = crate::runtime::ttg_builder::TTGBuilder::from_policy(op, t_policy);
        let device_ttg = crate::runtime::ttg::DeviceTTG::new(self, &layout, backend)?;
        let block = e_policy.backend_hint.preferred_block_dim;
        let smem = 48 * 1024;
        self.launch_ttg(kernel_id, block, smem as u32, args, &device_ttg, epilogue_args)
    }

    /// Launch with pre-allocated TTG workspace in arena (zero runtime malloc)
    pub fn launch_with_arena(
        &self,
        kernel_id: KernelId,
        args: Vec<KernelArg>,
        op: &crate::policy::types::OperatorTopology,
        t_policy: &crate::policy::types::TilePolicy,
        e_policy: &crate::policy::types::ExecPolicy,
        epilogue_args: Vec<KernelArg>,
        arena_buffer: BufferId,
        ttg_l1_offset: usize,
        ttg_l2_offset: usize,
    ) -> Result<(), String> {
        let layout = crate::runtime::ttg_builder::TTGBuilder::from_policy(op, t_policy);
        
        // Use arena-based TTG (no malloc)
        let device_ttg = crate::runtime::ttg::DeviceTTG::new_from_arena(
            self,
            &layout,
            arena_buffer,
            ttg_l1_offset,
            ttg_l2_offset,
        )?;
        
        let block = e_policy.backend_hint.preferred_block_dim;
        let smem = 48 * 1024;
        self.launch_ttg(kernel_id, block, smem as u32, args, &device_ttg, epilogue_args)
    }
    pub fn launch_kernel_by_id(
        &self, 
        kernel_id: KernelId, 
        grid: (u32, u32, u32), 
        block: (u32, u32, u32), 
        smem: u32, 
        args: Vec<KernelArg>
    ) -> Result<(), String> {
        self.launch(kernel_id, grid, block, smem, args)
    }
}

#[derive(Debug, Clone)]
pub enum KernelArg {
    Buffer(BufferId),
    BufferOffset(BufferId, usize),
    Int(i32),
    Float(f32),
    Usize(usize),
    Bytes(Vec<u8>),
}
