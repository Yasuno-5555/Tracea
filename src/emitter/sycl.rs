use crate::emitter::traits::Emitter;
use crate::semantic::transition::SyncRequirement;
use crate::semantic::fusion::EpilogueOp;

pub struct SYCLEmitter;

impl SYCLEmitter {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_pipelined_gemm(&self, config: crate::PipelineConfig) -> String {
        let n = config.num_stages;
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;

        format!(r#"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

void gemm_intel_pipelined_kernel(
    queue& q, float* A, float* B, float* C, 
    int M, int N, int K
) {{
    // Intel XMX Pipeline: {n} stages
    // Tile size: {mt}x{nt}x{kt}
    
    q.submit([&](handler& h) {{
        h.parallel_for(nd_range<2>{{range<2>{{M/{mt}, N/{nt}}}, range<2>{{1, 32}}}}, [=](nd_item<2> item) {{
            auto sg = item.get_sub_group();
            
            // Register fragmentation for XMX
            ext::intel::sub_group_matrix<float, matrix_type::c, {mt}, {nt}, matrix_layout::row_major> acc;
            ext::intel::sub_group_matrix<float, matrix_type::a, {mt}, {kt}, matrix_layout::row_major> frag_a;
            ext::intel::sub_group_matrix<float, matrix_type::b, {kt}, {nt}, matrix_layout::row_major> frag_b;

            sub_group_matrix_fill(sg, acc, 0.0f);

            // --- Phase Transition Loop (Z/NZ) ---
            for (int k_outer = 0; k_outer < K; k_outer += {kt}) {{
                sub_group_matrix_load(sg, frag_a, A + (item.get_group(0) * {mt} * K + k_outer), K);
                sub_group_matrix_load(sg, frag_b, B + (k_outer * N + item.get_group(1) * {nt}), N);
                
            sub_group_matrix_mad(sg, acc, frag_a, frag_b, acc);
            }}

            // Write back with Fusion
            float val; // Simplified: SYCL matrix doesn't easily allow register-level access like CUDA
            // In a real impl, we'd use a temporary float or specialized intrinsics
            // For now, we apply fusion logic conceptually or via specialized emitters
            {epilogue}

            sub_group_matrix_store(sg, acc, C + (item.get_group(0) * {mt} * N + item.get_group(1) * {nt}), N);
        }});
    }});
}}
"#, n = n, mt = mt, nt = nt, kt = kt, epilogue = self.emit_epilogue(config.epilogue.as_slice(), "val"))
    }
}

impl Emitter for SYCLEmitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String {
        match req {
            SyncRequirement::WaitAsyncLoad { .. } => {
                // SYCL uses group barriers
                "sycl::group_barrier(group);".to_string()
            }
            SyncRequirement::Barrier => "sycl::group_barrier(group);".to_string(),
            SyncRequirement::None => "".to_string(),
        }
    }

    fn emit_epilogue(&self, ops: &[EpilogueOp], acc_name: &str) -> String {
        let mut code = String::new();
        for op in ops {
            match op {
                EpilogueOp::BiasAdd { bias_ptr } => {
                    code.push_str(&format!("  {acc} += ((float*){ptr})[global_n];\n", acc = acc_name, ptr = bias_ptr));
                }
                EpilogueOp::ReLU => {
                    code.push_str(&format!("  {acc} = ({acc} > 0.0f) ? {acc} : 0.0f;\n", acc = acc_name));
                }
                _ => {}
            }
        }
        code
    }
}
