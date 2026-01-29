
use libloading::{Library, Symbol};
use std::ffi::c_void;

#[allow(non_snake_case)]
pub type HipGetDeviceCount = unsafe extern "system" fn(*mut i32) -> i32;
#[allow(non_snake_case)]
pub type HipDeviceGetAttribute = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;
#[allow(non_snake_case)]
pub type HipGetDeviceProperties = unsafe extern "system" fn(*mut c_void, i32) -> i32;
#[allow(non_snake_case)]
pub type HipMalloc = unsafe extern "system" fn(*mut u64, usize) -> i32;
#[allow(non_snake_case)]
pub type HipFree = unsafe extern "system" fn(u64) -> i32;
#[allow(non_snake_case)]
pub type HipMemcpyHtoD = unsafe extern "system" fn(u64, *const c_void, usize) -> i32;
#[allow(non_snake_case)]
pub type HipMemcpyDtoD = unsafe extern "system" fn(u64, u64, usize) -> i32;
#[allow(non_snake_case)]
pub type HipModuleLoadData = unsafe extern "system" fn(*mut *mut c_void, *const c_void) -> i32;
#[allow(non_snake_case)]
pub type HipModuleGetFunction = unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
#[allow(non_snake_case)]
pub type HipModuleLaunchKernel = unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> i32;
#[allow(non_snake_case)]
pub type HipDeviceSynchronize = unsafe extern "system" fn() -> i32;
#[allow(non_snake_case)]
pub type HipModuleUnload = unsafe extern "system" fn(*mut c_void) -> i32;

pub struct RocmDriverApi {
    pub lib: &'static Library,
    pub hipGetDeviceCount: Symbol<'static, HipGetDeviceCount>,
    pub hipDeviceGetAttribute: Symbol<'static, HipDeviceGetAttribute>,
    pub hipGetDeviceProperties: Symbol<'static, HipGetDeviceProperties>,
    pub hipMalloc: Symbol<'static, HipMalloc>,
    pub hipFree: Symbol<'static, HipFree>,
    pub hipMemcpyHtoD: Symbol<'static, HipMemcpyHtoD>,
    pub hipMemcpyDtoD: Symbol<'static, HipMemcpyDtoD>,
    pub hipModuleLoadData: Symbol<'static, HipModuleLoadData>,
    pub hipModuleGetFunction: Symbol<'static, HipModuleGetFunction>,
    pub hipModuleLaunchKernel: Symbol<'static, HipModuleLaunchKernel>,
    pub hipDeviceSynchronize: Symbol<'static, HipDeviceSynchronize>,
    pub hipModuleUnload: Symbol<'static, HipModuleUnload>,
}

static mut ROCM_DRIVER_API: Option<RocmDriverApi> = None;

impl RocmDriverApi {
    pub fn get() -> Option<&'static Self> {
        unsafe {
            if ROCM_DRIVER_API.is_some() {
                return ROCM_DRIVER_API.as_ref();
            }

            let lib_res = Library::new("amdhip64.dll"); // Windows name
            if let Ok(lib) = lib_res {
                let lib_ref: &'static Library = Box::leak(Box::new(lib));
                
                let get_count = lib_ref.get(b"hipGetDeviceCount");
                let get_attr = lib_ref.get(b"hipDeviceGetAttribute");
                let get_props = lib_ref.get(b"hipGetDeviceProperties");
                let malloc = lib_ref.get(b"hipMalloc");
                let free = lib_ref.get(b"hipFree");
                let h2d = lib_ref.get(b"hipMemcpyHtoD");
                let d2d = lib_ref.get(b"hipMemcpyDtoD");
                let load = lib_ref.get(b"hipModuleLoadData");
                let get_func = lib_ref.get(b"hipModuleGetFunction");
                let launch = lib_ref.get(b"hipModuleLaunchKernel");
                let sync = lib_ref.get(b"hipDeviceSynchronize");
                let unload = lib_ref.get(b"hipModuleUnload");
                
                if let (Ok(count), Ok(attr), Ok(props), Ok(m), Ok(f), Ok(h), Ok(d), Ok(l), Ok(gf), Ok(launch), Ok(sync), Ok(unload)) = 
                   (get_count, get_attr, get_props, malloc, free, h2d, d2d, load, get_func, launch, sync, unload) {
                    ROCM_DRIVER_API = Some(RocmDriverApi {
                        lib: lib_ref,
                        hipGetDeviceCount: count,
                        hipDeviceGetAttribute: attr,
                        hipGetDeviceProperties: props,
                        hipMalloc: m,
                        hipFree: f,
                        hipMemcpyHtoD: h,
                        hipMemcpyDtoD: d,
                        hipModuleLoadData: l,
                        hipModuleGetFunction: gf,
                        hipModuleLaunchKernel: launch,
                        hipDeviceSynchronize: sync,
                        hipModuleUnload: unload,
                    });
                    return ROCM_DRIVER_API.as_ref();
                }
            }
            None
        }
    }
}

// HIP attributes
pub const HIP_DEVICE_ATTRIBUTE_WAVEFRONT_SIZE: i32 = 28;
pub const HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 13;
pub const HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 31;
pub const HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 32;
