// Expose the new C-API bindings
#[cfg(feature = "cpp")]
pub mod c_bindings;


#[derive(Debug, Clone)]
pub struct TensorView {
    pub shape: Vec<u32>,
    pub data_ptr: *mut f32,
}

pub mod nn;
#[cfg(feature = "python")]
pub mod python;
#[cfg(feature = "cpp")]
pub mod cpp;
