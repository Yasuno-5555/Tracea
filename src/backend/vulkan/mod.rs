pub mod primitives;
use crate::backend::Backend;
use std::sync::Arc;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;

pub struct VulkanBackend {
    pub instance: Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub allocator: Arc<std::sync::Mutex<Allocator>>,
    pub device_id: String,
    pub subgroup_size: u32,
    pub supported_subgroup_operations: vk::SubgroupFeatureFlags,
}

impl VulkanBackend {
    /// # Safety
    ///
    /// This function is unsafe because it loads Vulkan drivers, creates Vulkan instances/devices,
    /// and manipulates raw pointers (e.g. reading physical device names). The caller must ensure
    /// that a compatible Vulkan driver is available on the host system to prevent undefined behavior.
    pub unsafe fn new() -> Result<Self, String> {
        // Safety: Entry::load dynamically loads vulkan driver which is unsafe.
        let entry = unsafe { Entry::load().map_err(|e| e.to_string())? };
        let app_info = vk::ApplicationInfo::builder()
            .application_name(std::ffi::CStr::from_bytes_with_nul(b"Tracea\0").unwrap())
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        // Safety: Creating vulkan instance is an FFI call that relies on a valid driver.
        let instance = unsafe { entry.create_instance(&create_info, None).map_err(|e| e.to_string())? };

        // Safety: Querying devices is safe assuming instance is valid.
        let p_devices = unsafe { instance.enumerate_physical_devices().map_err(|e| e.to_string())? };
        let (p_device, queue_family_index) = p_devices.iter().find_map(|&p_device| {
            // Safety: Querying queue properties requires a valid physical device handle.
            unsafe {
                instance.get_physical_device_queue_family_properties(p_device)
                    .iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        if info.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                            Some((p_device, index as u32))
                        } else {
                            None
                        }
                    })
            }
        }).ok_or("No suitable Vulkan device found")?;

        let device_extension_names = [];
        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names);

        // Safety: Device creation initializes hardware structures and is unsafe.
        let device = unsafe { instance.create_device(p_device, &device_create_info, None).map_err(|e| e.to_string())? };
        // Safety: Getting device queue is safe assuming queue index exists.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Safety: Allocator initialization uses driver handles and is unsafe.
        let allocator = unsafe {
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: p_device,
                debug_settings: Default::default(),
                buffer_device_address: true,
            }).map_err(|e| e.to_string())?
        };

        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut subgroup_props);
        // Safety: Querying device properties requires a valid physical device handle.
        unsafe { instance.get_physical_device_properties2(p_device, &mut props2) };

        // Safety: Querying device properties is unsafe but valid here with valid p_device.
        let props = unsafe { instance.get_physical_device_properties(p_device) };
        // Safety: Converting null-terminated raw pointer device name to CStr is unsafe.
        let device_name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy() };
        let device_id = format!("vulkan_{}", device_name.replace(" ", "_"));

        Ok(Self {
            instance,
            device,
            physical_device: p_device,
            queue,
            queue_family_index,
            allocator: Arc::new(std::sync::Mutex::new(allocator)),
            device_id,
            subgroup_size: subgroup_props.subgroup_size,
            supported_subgroup_operations: subgroup_props.supported_operations,
        })
    }
}

impl Backend for VulkanBackend {
    fn device_id(&self) -> String {
        self.device_id.clone()
    }

    fn driver_version(&self) -> String {
        "vulkan_driver_1.2".to_string()
    }

    fn runtime_version(&self) -> String {
        "vulkan_runtime".to_string()
    }

    fn max_shared_memory(&self) -> usize {
        // Safety: Querying physical device properties is safe once instance is successfully initialized and physical_device is a valid handle.
        unsafe {
            let props = self.instance.get_physical_device_properties(self.physical_device);
            props.limits.max_compute_shared_memory_size as usize
        }
    }
 
    fn max_threads_per_block(&self) -> usize {
        // Safety: Querying physical device properties is safe once instance is successfully initialized and physical_device is a valid handle.
        unsafe {
            let props = self.instance.get_physical_device_properties(self.physical_device);
            props.limits.max_compute_work_group_invocations as usize
        }
    }

    pub fn get_primitive_defs() -> String {
        primitives::VulkanPrimitives::get_subgroup_defs()
    }
}
