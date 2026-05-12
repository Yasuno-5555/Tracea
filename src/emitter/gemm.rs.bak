use crate::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};
use crate::runtime::manager::DeviceBackend;

pub fn generate_gemm(ir: &UnifiedOpIR, backend: DeviceBackend) -> String {
    if let UnifiedOpType::Gemm { m, n, k, .. } = ir.op_type {
        let mt = ir.tiling.m_tile;
        let nt = ir.tiling.n_tile;
        let kt = ir.tiling.k_tile;

        match backend {
            #[cfg(feature = "vulkan")]
            DeviceBackend::Vulkan => {
                crate::emitter::vulkan::generate_vulkan_mma(&UnifiedOpIR {
                    op_type: UnifiedOpType::MatrixCore { m: mt, n: nt, k: kt },
                    ..ir.clone()
                })
            }
            DeviceBackend::Cuda => {
                format!(r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define MT {mt}
#define NT {nt}
#define KT {kt}

extern "C" __global__ void unified_gemm_kernel(const half* A, const half* B, float* C, int M, int N, int K) {{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int row = blockIdx.y * MT;
    int col = blockIdx.x * NT;

    for (int k_step = 0; k_step < K; k_step += KT) {{
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
        
        wmma::load_matrix_sync(frag_a, A + row * K + k_step, K);
        wmma::load_matrix_sync(frag_b, B + k_step * N + col, N);
        wmma::mma_sync(acc, frag_a, frag_b, acc);
    }}

    wmma::store_matrix_sync(C + row * N + col, acc, N, wmma::mem_row_major);
}}
"#, mt=mt, nt=nt, kt=kt)
            }
            DeviceBackend::Metal => {
                let emitter = crate::emitter::metal::MetalEmitter::detect();
                emitter.generate_from_ir(ir)
            }
            _ => "// GEMM not yet unified for this backend\n".to_string(),
        }
    } else {
        panic!("GEMM emitter called with non-gemm op");
    }
}
