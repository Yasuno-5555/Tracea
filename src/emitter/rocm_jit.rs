
use libloading::{Library, Symbol};
use std::ffi::{CString, c_void};
use std::ptr;

#[allow(non_snake_case)]
pub type HiprtcCreateProgram = unsafe extern "system" fn(*mut *mut c_void, *const i8, *const i8, i32, *const *const i8, *const *const i8) -> i32;
#[allow(non_snake_case)]
pub type HiprtcCompileProgram = unsafe extern "system" fn(*mut c_void, i32, *const *const i8) -> i32;
#[allow(non_snake_case)]
pub type HiprtcGetCodeSize = unsafe extern "system" fn(*mut c_void, *mut usize) -> i32;
#[allow(non_snake_case)]
pub type HiprtcGetCode = unsafe extern "system" fn(*mut c_void, *mut i8) -> i32;
#[allow(non_snake_case)]
pub type HiprtcDestroyProgram = unsafe extern "system" fn(*mut *mut c_void) -> i32;

#[allow(non_snake_case)]
pub struct RocmJitApi {
    pub lib: &'static Library,
    pub hiprtcCreateProgram: Symbol<'static, HiprtcCreateProgram>,
    pub hiprtcCompileProgram: Symbol<'static, HiprtcCompileProgram>,
    pub hiprtcGetCodeSize: Symbol<'static, HiprtcGetCodeSize>,
    pub hiprtcGetCode: Symbol<'static, HiprtcGetCode>,
    pub hiprtcDestroyProgram: Symbol<'static, HiprtcDestroyProgram>,
}

static ROCM_JIT_API: std::sync::OnceLock<Option<RocmJitApi>> = std::sync::OnceLock::new();

impl RocmJitApi {
    pub fn get() -> Option<&'static Self> {
        ROCM_JIT_API.get_or_init(|| {
            unsafe {
                let lib_res = Library::new("hiprtc.dll"); 
                if let Ok(lib) = lib_res {
                    let lib_ref: &'static Library = Box::leak(Box::new(lib));
                    
                    if let (Ok(create), Ok(compile), Ok(get_size), Ok(get_code), Ok(destroy)) = (
                        lib_ref.get(b"hiprtcCreateProgram"),
                        lib_ref.get(b"hiprtcCompileProgram"),
                        lib_ref.get(b"hiprtcGetCodeSize"),
                        lib_ref.get(b"hiprtcGetCode"),
                        lib_ref.get(b"hiprtcDestroyProgram"),
                    ) {
                        return Some(RocmJitApi {
                            lib: lib_ref,
                            hiprtcCreateProgram: create,
                            hiprtcCompileProgram: compile,
                            hiprtcGetCodeSize: get_size,
                            hiprtcGetCode: get_code,
                            hiprtcDestroyProgram: destroy,
                        });
                    }
                }
                None
            }
        }).as_ref()
    }
}

pub struct ROCMJITCompiler {
    pub api: &'static RocmJitApi,
}

impl ROCMJITCompiler {
    pub fn new() -> Option<Self> {
        RocmJitApi::get().map(|api| Self { api })
    }

    pub fn compile(&self, source: &str, name: &str, options: Vec<String>) -> Result<Vec<u8>, String> {
        let source_c = CString::new(source).unwrap();
        let name_c = CString::new(name).unwrap();
        
        let mut prog: *mut c_void = ptr::null_mut();
        unsafe {
            let res = (self.api.hiprtcCreateProgram)(&mut prog, source_c.as_ptr(), name_c.as_ptr(), 0, ptr::null(), ptr::null());
            if res != 0 { return Err(format!("hiprtcCreateProgram failed: {}", res)); }
            
            let opts_c: Vec<CString> = options.into_iter().map(|s| CString::new(s).unwrap()).collect();
            let opts_ptr: Vec<*const i8> = opts_c.iter().map(|s| s.as_ptr()).collect();
            
            let res = (self.api.hiprtcCompileProgram)(prog, opts_ptr.len() as i32, opts_ptr.as_ptr());
            if res != 0 {
                // TODO: Log compile log
                (self.api.hiprtcDestroyProgram)(&mut prog);
                return Err(format!("hiprtcCompileProgram failed: {}", res));
            }
            
            let mut size: usize = 0;
            (self.api.hiprtcGetCodeSize)(prog, &mut size);
            let mut code = vec![0u8; size];
            (self.api.hiprtcGetCode)(prog, code.as_mut_ptr() as *mut i8);
            
            (self.api.hiprtcDestroyProgram)(&mut prog);
            Ok(code)
        }
    }
}
