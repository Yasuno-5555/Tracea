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

pub struct DriverApi {
    pub lib: &'static Library,
    pub launch_kernel: Symbol<'static, CuLaunchKernel>,
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
                    match launch_res {
                        Ok(sym) => {
                            DRIVER_API = Some(DriverApi {
                                lib: lib_ref, 
                                launch_kernel: std::mem::transmute(sym),
                            });
                        }
                        Err(e) => println!("Tracea Error: Failed to load cuLaunchKernel: {}", e),
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
