use std::sync::Once;
use std::ffi::c_void;
use libloading::{Library, Symbol};

#[allow(non_snake_case)]
pub type CuLaunchKernel = unsafe extern "system" fn(
    *mut c_void, 
    u32, u32, u32, 
    u32, u32, u32, 
    u32, 
    *mut c_void, 
    *mut *mut c_void, 
    *mut *mut c_void
) -> i32;

#[allow(non_snake_case)]
pub type CuFuncSetAttribute = unsafe extern "system" fn(*mut c_void, i32, i32) -> i32;
#[allow(non_snake_case)]
pub type CuFuncGetAttribute = unsafe extern "system" fn(*mut i32, i32, *mut c_void) -> i32;
#[allow(non_snake_case)]
pub type CuDriverGetVersion = unsafe extern "system" fn(*mut i32) -> i32;
#[allow(non_snake_case)]
pub type CuDeviceGetAttribute = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;

pub struct DriverApi {
    pub lib: &'static Library,
    pub launch_kernel: Symbol<'static, CuLaunchKernel>,
    pub cu_func_set_attribute: Symbol<'static, CuFuncSetAttribute>,
    pub cu_func_get_attribute: Symbol<'static, CuFuncGetAttribute>,
    pub cu_driver_get_version: Symbol<'static, CuDriverGetVersion>,
    pub cu_device_get_attribute: Symbol<'static, CuDeviceGetAttribute>,
}

static mut DRIVER_API: Option<DriverApi> = None;
static INIT: Once = Once::new();

pub fn get_driver_api() -> Result<&'static DriverApi, String> {
    unsafe {
        INIT.call_once(|| {
            let lib_res = Library::new("nvcuda.dll");
            match lib_res {
                Ok(lib) => {
                    // We must leak the library to ensure symbols are static
                    let lib_ref = Box::leak(Box::new(lib)); 
                    let launch_res: Result<Symbol<CuLaunchKernel>, _> = lib_ref.get(b"cuLaunchKernel");
                    let set_attr_res: Result<Symbol<CuFuncSetAttribute>, _> = lib_ref.get(b"cuFuncSetAttribute");
                    let get_attr_res: Result<Symbol<CuFuncGetAttribute>, _> = lib_ref.get(b"cuFuncGetAttribute");
                    let get_ver_res: Result<Symbol<CuDriverGetVersion>, _> = lib_ref.get(b"cuDriverGetVersion");
                    let get_dev_attr_res: Result<Symbol<CuDeviceGetAttribute>, _> = lib_ref.get(b"cuDeviceGetAttribute");
                    
                    if let (Ok(launch), Ok(set_attr), Ok(get_attr), Ok(get_ver), Ok(get_dev_attr)) = (launch_res, set_attr_res, get_attr_res, get_ver_res, get_dev_attr_res) {
                        DRIVER_API = Some(DriverApi {
                            lib: lib_ref, 
                            launch_kernel: std::mem::transmute(launch),
                            cu_func_set_attribute: std::mem::transmute(set_attr),
                            cu_func_get_attribute: std::mem::transmute(get_attr),
                            cu_driver_get_version: std::mem::transmute(get_ver),
                            cu_device_get_attribute: std::mem::transmute(get_dev_attr),
                        });
                    } else {
                        println!("Tracea Error: Some Driver symbols could not be loaded.");
                    }
                },
                Err(e) => println!("Tracea Error: Failed to load nvcuda.dll: {}", e),
            }
        });
        
        match &DRIVER_API {
            Some(api) => Ok(api),
            None => Err("Driver API not initialized".to_string()),
        }
    }
}

// Safety wrapper for global statics
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
