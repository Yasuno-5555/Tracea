use pyo3::prelude::*;
use crate::interface::python::{PyGraph, PyEpilogueOp, PyEpilogueType};
use crate::core::op::EpilogueOp;

#[pyclass]
#[derive(Clone)]
pub struct Linear {
    in_features: u32,
    out_features: u32,
    bias: bool,
    activation: Vec<EpilogueOp>,
}

#[pymethods]
impl Linear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true, activation=None))]
    fn new(in_features: u32, out_features: u32, bias: bool, activation: Option<PyEpilogueOp>) -> PyResult<Self> {
        let mut rust_epilogue = Vec::new();
        
        if let Some(py_epi) = activation {
            for (op_type, ptr_opt) in py_epi.ops {
                 let op = match op_type {
                    PyEpilogueType::ReLU => EpilogueOp::ReLU,
                    PyEpilogueType::Gelu => EpilogueOp::Gelu,
                    PyEpilogueType::BiasAdd => {
                        if let Some(ptr) = ptr_opt {
                            EpilogueOp::BiasAdd { bias_ptr: ptr }
                        } else {
                             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("BiasAdd requires pointer"));
                        }
                    }
                };
                rust_epilogue.push(op);
            }
        }

        Ok(Self {
            in_features,
            out_features,
            bias,
            activation: rust_epilogue,
        })
    }

    /// Adds the Linear layer to the graph.
    /// Input: 
    /// - graph: The PyGraph instance to add to.
    /// - input_node: The ID of the input node in the graph.
    /// Returns: The ID of the new Linear node.
    fn __call__(&self, graph: &mut PyGraph, input_node: usize) -> PyResult<usize> {
        // In the future, we might infer shapes or check inputs here.
        let id = graph.inner.add_linear(
            crate::core::op::DimExpr::Symbol("B".to_string()),
            self.in_features, 
            self.out_features, 
            self.bias, 
            self.activation.clone(), 
            vec![input_node]
        );
        Ok(id)
    }
    
    // Support functional style usage?
}

#[pyclass]
#[derive(Clone)]
pub struct Attention {
    embed_dim: u32,
    num_heads: u32,
    head_dim: u32,
    causal: bool,
}

#[pymethods]
impl Attention {
    #[new]
    #[pyo3(signature = (embed_dim, num_heads, causal=false))]
    fn new(embed_dim: u32, num_heads: u32, causal: bool) -> Self {
        Self {
            embed_dim,
            num_heads,
            head_dim: embed_dim / num_heads,
            causal,
        }
    }

    fn __call__(&self, graph: &mut PyGraph, input_node: usize) -> PyResult<usize> {
        let id = graph.inner.add_attention(
            self.embed_dim,
            self.num_heads,
            self.head_dim,
            self.causal,
            vec![input_node]
        );
        Ok(id)
    }
}

// Module registration helper
pub fn register_nn_module(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn_module = PyModule::new_bound(py, "nn")?;
    nn_module.add_class::<Linear>()?;
    nn_module.add_class::<Attention>()?;
    parent_module.add_submodule(&nn_module)?;
    Ok(())
}
