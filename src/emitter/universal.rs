use crate::emitter::traits::{Emitter, UnifiedOpIR};
use crate::emitter::cuda::CUDAEmitter;
use crate::emitter::rocm::ROCMEmitter;
use crate::emitter::metal::MetalEmitter;
use crate::runtime::manager::DeviceBackend;

pub struct UniversalEmitter {
    pub backend: DeviceBackend,
}

impl UniversalEmitter {
    pub fn new(backend: DeviceBackend) -> Self {
        Self { backend }
    }

    pub fn generate_from_strategy(
        &self,
        atom: &crate::core::manifold::ComputeAtom,
        lattice: &crate::core::lattice::HardwareLattice,
        strategy: &crate::core::mapper::MappingStrategy,
    ) -> String {
        match self.backend {
            DeviceBackend::Cuda => self.generate_cuda(atom, lattice, strategy),
            _ => format!("// Universal generation not yet supported for {:?}", self.backend),
        }
    }

    fn generate_cuda(
        &self,
        atom: &crate::core::manifold::ComputeAtom,
        lattice: &crate::core::lattice::HardwareLattice,
        strategy: &crate::core::mapper::MappingStrategy,
    ) -> String {
        let mut source = String::new();
        source.push_str("#include <cuda_fp16.h>\n#include <mma.h>\nusing namespace nvcuda;\n\n");

        // 1. Generate Parameters Struct from Atom Tensors
        source.push_str(&self.generate_params_struct(atom));

        // 2. Kernel Signature
        source.push_str(&format!(
            "extern \"C\" __global__ void {}(Params p) {{\n",
            atom.name
        ));

        // 3. Hierarchical Thread Indexing based on Lattice & Strategy
        source.push_str(&self.generate_indexing_logic(lattice, strategy));

        // 4. Memory Staging & Tiled Loop Nest
        source.push_str(&self.generate_loop_nest(atom, strategy));

        source.push_str("}\n");
        source
    }

    fn generate_params_struct(&self, atom: &crate::core::manifold::ComputeAtom) -> String {
        let mut s = "struct Params {\n".to_string();
        for read in &atom.reads {
            s.push_str(&format!("    const half* __restrict__ tensor_{};\n", read.tensor_id));
        }
        s.push_str(&format!("    half* __restrict__ tensor_{};\n", atom.write.tensor_id));
        
        // Add dynamic bounds/strides
        for dim in &atom.domain.dimensions {
            s.push_str(&format!("    int dim_{};\n", dim.id));
        }
        s.push_str("};\n\n");
        s
    }

    fn generate_indexing_logic(
        &self,
        _lattice: &crate::core::lattice::HardwareLattice,
        strategy: &crate::core::mapper::MappingStrategy,
    ) -> String {
        let mut s = "    int tid = threadIdx.x;\n".to_string();
        for (dim_id, level) in &strategy.spatial_map {
            match level.0.as_str() {
                "BlockX" => s.push_str(&format!("    int block_{} = blockIdx.x;\n", dim_id)),
                "BlockY" => s.push_str(&format!("    int block_{} = blockIdx.y;\n", dim_id)),
                "ThreadX" => s.push_str(&format!("    int thread_{} = threadIdx.x;\n", dim_id)),
                _ => {}
            }
        }
        s
    }

    fn generate_affine_expression(&self, matrix: &crate::core::manifold::AffineMatrix, atom: &crate::core::manifold::ComputeAtom) -> String {
        let mut parts = Vec::new();
        for (i, &coeff) in matrix.coeffs.iter().enumerate() {
            if coeff != 0 {
                let dim_name = &atom.domain.dimensions[i].id;
                let var_name = if dim_name == "n" || dim_name == "oh" || dim_name == "ow" || dim_name == "oc" {
                    format!("block_{}", dim_name)
                } else if dim_name == "ic" || dim_name == "r" || dim_name == "s" {
                    format!("thread_{}", dim_name)
                } else {
                    format!("iter_{}", dim_name)
                };

                if coeff == 1 {
                    parts.push(var_name);
                } else if coeff == -1 {
                    parts.push(format!("-{}", var_name));
                } else {
                    parts.push(format!("({} * {})", coeff, var_name));
                }
            }
        }
        if matrix.constant != 0 {
            parts.push(matrix.constant.to_string());
        }
        if parts.is_empty() { "0".to_string() } else { parts.join(" + ") }
    }

    fn generate_linear_index(&self, access: &crate::core::manifold::AccessMap, atom: &crate::core::manifold::ComputeAtom) -> String {
        let mut stride_exprs = Vec::new();
        let mut group_stride = 1;
        
        // Recompute strides based on Dimension ranges
        let mut strides = vec![0i64; access.index_expressions.len()];
        for i in (0..access.index_expressions.len()).rev() {
            strides[i] = group_stride;
            // Simplified: range_max represents the count for this dimension
            group_stride *= atom.domain.dimensions[i].range_max; 
        }

        for (i, expr_matrix) in access.index_expressions.iter().enumerate() {
            let expr = self.generate_affine_expression(expr_matrix, atom);
            if strides[i] == 1 {
                stride_exprs.push(expr);
            } else {
                stride_exprs.push(format!("(({}) * {})", expr, strides[i]));
            }
        }
        stride_exprs.join(" + ")
    }

    fn generate_predication(&self, atom: &crate::core::manifold::ComputeAtom) -> String {
        let mut checks = Vec::new();
        for dim in &atom.domain.dimensions {
            checks.push(format!("(block_{} < p.dim_{})", dim.id, dim.id));
        }
        format!("    if (!({})) return;\n", checks.join(" && "))
    }

    fn generate_loop_nest(
        &self,
        atom: &crate::core::manifold::ComputeAtom,
        _strategy: &crate::core::mapper::MappingStrategy,
    ) -> String {
        let mut s = String::new();
        
        // 1. Predication (Bounds Check)
        s.push_str(&self.generate_predication(atom));

        // 2. Declare Shared Memory
        s.push_str("    __shared__ half smem_A[128*32];\n");
        s.push_str("    __shared__ half smem_B[32*128];\n\n");

        // 2. Reduction Loop
        if let Some(k_dim) = atom.get_dim_by_id("ic") {
             s.push_str(&format!("    for (int k_outer = 0; k_outer < {}; k_outer += 32) {{\n", k_dim.range_max));
             
             // 3. Staging: Global -> Shared (simplified)
             s.push_str("        // Collective Global -> Shared Load\n");
             s.push_str("        if (tid < 128) {\n");
             s.push_str("            smem_A[tid] = p.tensor_0[tid]; // Placeholder for async copy\n");
             s.push_str("        }\n");
             s.push_str("        __syncthreads();\n\n");

             // 4. Inner Compute
             s.push_str("        float acc = 0.0f;\n");
             s.push_str("        for (int k_inner = 0; k_inner < 32; ++k_inner) {\n");
             s.push_str("            acc += (float)smem_A[k_inner];\n");
             s.push_str("        }\n");
             
             s.push_str("        __syncthreads();\n");
             s.push_str("    }\n");
        }

        // 5. Write Back (simplified)
        let write_expr = self.generate_linear_index(&atom.write, atom);
        s.push_str(&format!("    p.tensor_{}[{}] = (half)0.0f; // Final Result\n", atom.write.tensor_id, write_expr));
        
        s
    }

    pub fn generate(&self, ir: UnifiedOpIR) -> String {
        if let crate::emitter::traits::UnifiedOpType::Elementwise { .. } = ir.op_type {
            return crate::emitter::elementwise::generate_elementwise(&ir, self.backend);
        }
        if let crate::emitter::traits::UnifiedOpType::Conv2d { .. } = ir.op_type {
            match self.backend {
                #[cfg(feature = "vulkan")]
                DeviceBackend::Vulkan => {
                    return crate::emitter::vulkan::generate_vulkan_conv(&ir);
                }
                DeviceBackend::Metal => {
                    let emitter = MetalEmitter::detect();
                    return emitter.generate_from_ir(&ir);
                }
                _ => return crate::emitter::conv::generate_conv(&ir),
            }
        }
        if let crate::emitter::traits::UnifiedOpType::FusedAttention { .. } = ir.op_type {
            return crate::emitter::attention::generate_attention(&ir, self.backend);
        }
        if let crate::emitter::traits::UnifiedOpType::Gemm { .. } = ir.op_type {
            return crate::emitter::gemm::generate_gemm(&ir, self.backend);
        }
        if let crate::emitter::traits::UnifiedOpType::MatrixCore { .. } = ir.op_type {
            #[cfg(feature = "vulkan")]
            if self.backend == DeviceBackend::Vulkan {
                return crate::emitter::vulkan::generate_vulkan_mma(&ir);
            }
            // CUDA/Metal fallbacks to Gemm or specialized emitters
        }

        match self.backend {
            DeviceBackend::Cuda => {
                let emitter = CUDAEmitter::new();
                emitter.generate_from_ir(&ir)
            }
            DeviceBackend::Rocm => {
                let emitter = ROCMEmitter::detect();
                emitter.generate_from_ir(&ir)
            }
            DeviceBackend::Metal => {
                let emitter = MetalEmitter::detect();
                emitter.generate_from_ir(&ir)
            }
            DeviceBackend::Cpu => {
                "/* CPU implementation is static */".to_string()
            }
        }
    }
}
