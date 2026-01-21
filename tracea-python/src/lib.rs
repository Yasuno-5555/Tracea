//! Tracea Python Extension Module
//! 
//! Zero-copy PyTorch backend for Stable Diffusion / diffusers

use pyo3::prelude::*;

mod device;
mod tensor_view;
mod conv2d;

#[pymodule]
fn tracea(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create ops submodule
    let ops = PyModule::new_bound(m.py(), "ops")?;
    ops.add_function(wrap_pyfunction!(conv2d::conv2d, &ops)?)?;
    m.add_submodule(&ops)?;

    // Expose Core Classes from tracea::interface::python
    m.add_class::<tracea::interface::python::PyContext>()?;
    m.add_class::<tracea::interface::python::PyTuner>()?;
    m.add_class::<tracea::interface::python::PyDeviceBufferF32>()?;
    m.add_class::<tracea::interface::python::PyDeviceBufferU16>()?;
    
    // Version info
    m.add("__version__", "0.1.0")?;
    
    Ok(())
}
