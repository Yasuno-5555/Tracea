use crate::emitter::traits::Emitter;
use crate::semantic::transition::SyncRequirement;
use crate::semantic::swizzle::SwizzleMode;
use crate::core::op::EpilogueOp;

pub struct CUDAEmitter;

impl CUDAEmitter {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_pipelined_gemm(&self, config: crate::PipelineConfig) -> String {
        if config.use_tensor_cores {
            return self.generate_tensor_core_gemm(config);
        }
        let n = config.num_stages;
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;
        let epilogue_code = self.emit_epilogue(config.epilogue.as_slice(), "sum", "global_n");
        
        let reg_m = mt / 16;
        let reg_n = nt / 16;
        
        let s_a = kt + 4;
        let s_b = nt + 4;

        format!(r#"
extern "C" __device__ __forceinline__ unsigned int get_smem_ptr(const void* ptr) {{
    unsigned int ret;
    asm volatile("{{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }}" : "=r"(ret) : "l"(ptr));
    return ret;
}}

extern "C" __global__ void gemm_pipelined_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {{
    extern __shared__ __align__(16) float smem[];
    float* As = &smem[0];
    float* Bs = &smem[{mt} * {s_a} * {n}];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;

    float frag_a[2][{reg_m}];
    float frag_b[2][{reg_n}];
    float acc[{reg_m}][{reg_n}];
    
    #pragma unroll
    for (int i = 0; i < {reg_m}; ++i) 
        for (int j = 0; j < {reg_n}; ++j) 
            acc[i][j] = 0.0f;
            
    #define IDX_A(s, r, c) ( (s) * {mt} * {s_a} + (r) * {s_a} + (c) )
    #define SWZ_B(s, r, c) ( (s) * {kt} * {s_b} + (r) * {s_b} + (c) )
    
    __syncthreads();

    // PROLOGUE
    for (int s = 0; s < {n} - 1; ++s) {{
        int k_offset = s * {kt};
        if (k_offset < K) {{
            #pragma unroll
            for (int i = 0; i < 4; ++i) {{
                int idx4 = i * 256 + tid;
                int r = (idx4 * 4) / {kt};
                int c = (idx4 * 4) % {kt};
                long long gr = (long long)by * {mt} + r;
                long long gc = (long long)k_offset + c;
                float* dst = &As[IDX_A(s, r, c)];
                if (gr < M && gc < K) {{
                    const float* src = &A[gr * K + gc];
                    unsigned int sp = get_smem_ptr(dst);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sp), "l"((unsigned long long)src));
                }} else {{ *((float4*)dst) = make_float4(0,0,0,0); }}
            }}
            #pragma unroll
            for (int i = 0; i < 4; ++i) {{
                int idx4 = i * 256 + tid;
                int r = (idx4 * 4) / {nt};
                int c = (idx4 * 4) % {nt};
                long long gr = (long long)k_offset + r;
                long long gc = (long long)bx * {nt} + c;
                float* dst = &Bs[SWZ_B(s, r, c)];
                if (gr < K && gc < N) {{
                    const float* src = &B[gr * N + gc];
                    unsigned int sp = get_smem_ptr(dst);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sp), "l"((unsigned long long)src));
                }} else {{ *((float4*)dst) = make_float4(0,0,0,0); }}
            }}
            asm volatile("cp.async.commit_group;");
        }}
    }}
    
    // MAIN LOOP
    #pragma unroll 1
    for (int k_step = 0; k_step < K / {kt}; ++k_step) {{
        int load_s = (k_step + {n} - 1) % {n};
        int k_next = (k_step + {n} - 1) * {kt};
        if (k_next < K) {{
             #pragma unroll
             for (int i = 0; i < 4; ++i) {{
                int idx4 = i * 256 + tid;
                int r = (idx4 * 4) / {kt};
                int c = (idx4 * 4) % {kt};
                long long gr = (long long)by * {mt} + r;
                long long gc = (long long)k_next + c;
                float* dst = &As[IDX_A(load_s, r, c)];
                if (gr < M && gc < K) {{
                    const float* src = &A[gr * K + gc];
                    unsigned int sp = get_smem_ptr(dst);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sp), "l"((unsigned long long)src));
                }} else {{ *((float4*)dst) = make_float4(0,0,0,0); }}
            }}
            #pragma unroll
            for (int i = 0; i < 4; ++i) {{
                int idx4 = i * 256 + tid;
                int r = (idx4 * 4) / {nt};
                int c = (idx4 * 4) % {nt};
                long long gr = (long long)k_next + r;
                long long gc = (long long)bx * {nt} + c;
                float* dst = &Bs[SWZ_B(load_s, r, c)];
                if (gr < K && gc < N) {{
                    const float* src = &B[gr * N + gc];
                    unsigned int sp = get_smem_ptr(dst);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sp), "l"((unsigned long long)src));
                }} else {{ *((float4*)dst) = make_float4(0,0,0,0); }}
            }}
            asm volatile("cp.async.commit_group;");
        }}

        asm volatile("cp.async.wait_group {wait_stages};");
        __syncthreads();
        
        int comp_s = k_step % {n};
        
        // Zero-Integer-Math: Direct pointers to thread-local rows in SMEM
        float* a_tile_ptr[{reg_m}];
        #pragma unroll
        for (int i = 0; i < {reg_m}; ++i) {{
            a_tile_ptr[i] = &As[IDX_A(comp_s, ty * {reg_m} + i, 0)];
        }}
        float* b_tile_ptr = &Bs[SWZ_B(comp_s, 0, tx * {reg_n})];
        
        // Initial Prefetch for k_inner=0
        #pragma unroll
        for(int i=0; i<{reg_m}; ++i) frag_a[0][i] = *a_tile_ptr[i]++;
        *((float4*)&frag_b[0][0]) = *((float4*)&b_tile_ptr[0]);
        *((float4*)&frag_b[0][4]) = *((float4*)&b_tile_ptr[4]);
        b_tile_ptr += {s_b};
        
        #pragma unroll
        for (int k_inner = 0; k_inner < {kt}; ++k_inner) {{
            int cur = k_inner % 2;
            int next = (k_inner + 1) % 2;
            
            // Prefetch next k-inner
            if (k_inner < {kt} - 1) {{
                 #pragma unroll
                 for(int i=0; i<{reg_m}; ++i) frag_a[next][i] = *a_tile_ptr[i]++;
                 *((float4*)&frag_b[next][0]) = *((float4*)&b_tile_ptr[0]);
                 *((float4*)&frag_b[next][4]) = *((float4*)&b_tile_ptr[4]);
                 b_tile_ptr += {s_b};
            }}
            
            // Core FMA
            #pragma unroll
            for(int i=0; i<{reg_m}; ++i) {{
                float val_a = frag_a[cur][i];
                #pragma unroll
                for(int j=0; j<{reg_n}; ++j) {{
                    acc[i][j] += val_a * frag_b[cur][j];
                }}
            }}
        }}

        __syncthreads(); // MANDATORY SYNC AT END OF STAGE
    }}
    
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    
    // Epilogue
    for (int i = 0; i < {reg_m}; ++i) {{
        for (int j = 0; j < {reg_n}; ++j) {{
            int row_local = ty * {reg_m} + i;
            int col_local = tx * {reg_n} + j;
            long long global_m = (long long)by * {mt} + row_local;
            long long global_n = (long long)bx * {nt} + col_local;
            if (global_m < M && global_n < N) {{
                float sum = acc[i][j];
                {epilogue}
                C[global_m * N + global_n] = sum;
            }}
        }}
    }}
}}
"#, n=n, mt=mt, nt=nt, kt=kt, wait_stages=n-1, epilogue=epilogue_code, reg_m=reg_m, reg_n=reg_n, s_a=s_a, s_b=s_b)
    }

    pub fn generate_tensor_core_gemm(&self, config: crate::PipelineConfig) -> String {
        let n = config.num_stages;
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;
        
        // Automatic Warp Layout Determination
        // Target 128 threads (4 warps)
        let num_warps = 4;
        let warp_rows = if mt >= nt { 2 } else { 1 };
        let warp_cols = num_warps / warp_rows;
        
        let warp_mt = mt / warp_rows;
        let warp_nt = nt / warp_cols;
        
        // MMA Shape (m16n8k16)
        let mma_m = 16;
        let mma_n = 8;
        let mma_k = 16;
        
        let tiles_m = warp_mt / mma_m;
        let tiles_n = warp_nt / mma_n;
        let tiles_k = kt / mma_k;
        let loads_a = (mt * kt + 128 * 8 - 1) / (128 * 8);
        let loads_b = (kt * nt + 128 * 8 - 1) / (128 * 8);

        
        format!(r#"
extern "C" __device__ __forceinline__ unsigned int get_smem_ptr(const void* ptr) {{
    unsigned int ret;
    asm volatile("{{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }}" : "=r"(ret) : "l"(ptr));
    return ret;
}}

extern "C" __global__ void gemm_pipelined_kernel(
    const unsigned short* __restrict__ A, 
    const unsigned short* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K
) {{
    extern __shared__ __align__(16) unsigned short smem_h[];
    unsigned short* As = &smem_h[0];
    unsigned short* Bs = &smem_h[{mt} * {kt} * {n}];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    unsigned int frag_a[{tiles_m}][4];
    unsigned int frag_b[{tiles_n}][2];
    float acc[{tiles_m}][{tiles_n}][4];

    #pragma unroll
    for (int i = 0; i < {tiles_m}; ++i) 
        for (int j = 0; j < {tiles_n}; ++j) 
            for (int k = 0; k < 4; ++k)
                acc[i][j][k] = 0.0f;

    int warp_m = (warp_id / {warp_cols}) * {warp_mt};
    int warp_n = (warp_id % {warp_cols}) * {warp_nt};

    __syncthreads();

    // PROLOGUE
    for (int s = 0; s < {n} - 1; ++s) {{
        int k_offset = s * {kt};
        if (k_offset < K) {{
            #pragma unroll
            for (int i = 0; i < {loads_a}; ++i) {{
                int idx = i * 128 + tid;
                int r = (idx * 8) / {kt};
                int c = (idx * 8) % {kt};
                if (r < {mt}) {{
                    long long gr = (long long)by * {mt} + r;
                    long long gc = (long long)k_offset + c;
                    unsigned short* dst = &As[s * {mt} * {kt} + r * {kt} + c];
                    if (gr < M && gc < K) {{
                        const unsigned short* src = &A[gr * K + gc];
                        unsigned int sp = get_smem_ptr(dst);
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sp), "l"((unsigned long long)src));
                    }} else {{ *((float4*)dst) = make_float4(0,0,0,0); }}
                }}
            }}
            #pragma unroll
            for (int i = 0; i < {loads_b}; ++i) {{
                int idx = i * 128 + tid;
                int r = (idx * 8) / {nt};
                int c = (idx * 8) % {nt};
                if (r < {kt}) {{
                    long long gr = (long long)k_offset + r;
                    long long gc = (long long)bx * {nt} + c;
                    unsigned short* dst = &Bs[s * {kt} * {nt} + r * {nt} + c];
                    if (gr < K && gc < N) {{
                        const unsigned short* src = &B[gr * N + gc];
                        unsigned int sp = get_smem_ptr(dst);
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sp), "l"((unsigned long long)src));
                    }} else {{ *((float4*)dst) = make_float4(0,0,0,0); }}
                }}
            }}
            asm volatile("cp.async.commit_group;");
        }}
    }}

    // MAIN LOOP
    for (int k_step = 0; k_step < K / {kt}; ++k_step) {{
        int load_s = (k_step + {n} - 1) % {n};
        int k_next = (k_step + {n} - 1) * {kt};
        if (k_next < K) {{
             #pragma unroll
             for (int i = 0; i < {loads_a}; ++i) {{
                int idx = i * 128 + tid;
                int r = (idx * 8) / {kt};
                int c = (idx * 8) % {kt};
                if (r < {mt}) {{
                    long long gr = (long long)by * {mt} + r;
                    long long gc = (long long)k_next + c;
                    unsigned short* dst = &As[load_s * {mt} * {kt} + r * {kt} + c];
                    if (gr < M && gc < K) {{
                        const unsigned short* src = &A[gr * K + gc];
                        unsigned int sp = get_smem_ptr(dst);
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sp), "l"((unsigned long long)src));
                    }} else {{ *((float4*)dst) = make_float4(0,0,0,0); }}
                }}
            }}
            #pragma unroll
            for (int i = 0; i < {loads_b}; ++i) {{
                int idx = i * 128 + tid;
                int r = (idx * 8) / {nt};
                int c = (idx * 8) % {nt};
                if (r < {kt}) {{
                    long long gr = (long long)k_next + r;
                    long long gc = (long long)bx * {nt} + c;
                    unsigned short* dst = &Bs[load_s * {kt} * {nt} + r * {nt} + c];
                    if (gr < K && gc < N) {{
                        const unsigned short* src = &B[gr * N + gc];
                        unsigned int sp = get_smem_ptr(dst);
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sp), "l"((unsigned long long)src));
                    }} else {{ *((float4*)dst) = make_float4(0,0,0,0); }}
                }}
            }}
            asm volatile("cp.async.commit_group;");
        }}

        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        int comp_s = k_step % {n};
        unsigned short* a_warp_ptr = &As[comp_s * {mt} * {kt} + warp_m * {kt}];
        unsigned short* b_warp_ptr = &Bs[comp_s * {kt} * {nt} + warp_n];

        #pragma unroll
        for (int k_inner = 0; k_inner < {tiles_k}; ++k_inner) {{
            #pragma unroll
            for (int i = 0; i < {tiles_m}; ++i) {{
                unsigned int addr = get_smem_ptr(&a_warp_ptr[(i * 16) * {kt} + k_inner * 16 + (lane_id % 16) * {kt} + (lane_id / 16) * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];"
                    : "=r"(frag_a[i][0]), "=r"(frag_a[i][1]), "=r"(frag_a[i][2]), "=r"(frag_a[i][3]) : "r"(addr));
            }}
            #pragma unroll
            for (int j = 0; j < {tiles_n}; ++j) {{
                unsigned int addr = get_smem_ptr(&b_warp_ptr[(k_inner * 16) * {nt} + j * 8 + (lane_id % 16) * {nt}]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%0, %1}}, [%2];"
                    : "=r"(frag_b[j][0]), "=r"(frag_b[j][1]) : "r"(addr));
            }}

            #pragma unroll
            for (int i = 0; i < {tiles_m}; ++i) {{
                #pragma unroll
                for (int j = 0; j < {tiles_n}; ++j) {{
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%10, %11, %12, %13}};"
                        : "=f"(acc[i][j][0]), "=f"(acc[i][j][1]), "=f"(acc[i][j][2]), "=f"(acc[i][j][3])
                        : "r"(frag_a[i][0]), "r"(frag_a[i][1]), "r"(frag_a[i][2]), "r"(frag_a[i][3]),
                          "r"(frag_b[j][0]), "r"(frag_b[j][1]),
                          "f"(acc[i][j][0]), "f"(acc[i][j][1]), "f"(acc[i][j][2]), "f"(acc[i][j][3]));
                }}
            }}
        }}
        __syncthreads();
    }}

    #pragma unroll
    for (int i = 0; i < {tiles_m}; ++i) {{
        #pragma unroll
        for (int j = 0; j < {tiles_n}; ++j) {{
            int row_base = warp_m + i * 16;
            int col_base = warp_n + j * 8;
            int r0 = row_base + (lane_id / 4);
            int c0 = col_base + (lane_id % 4) * 2;
            int r1 = row_base + (lane_id / 4) + 8;
            
            float v0 = acc[i][j][0];
            float v1 = acc[i][j][1];
            float v2 = acc[i][j][2];
            float v3 = acc[i][j][3];
            
            {e0}
            {e1}
            {e2}
            {e3}

            if (r0 < M && c0 < N) C[r0 * N + c0] = v0;
            if (r0 < M && (c0 + 1) < N) C[r0 * N + c0 + 1] = v1;
            if (r1 < M && c0 < N) C[r1 * N + c0] = v2;
            if (r1 < M && (c0 + 1) < N) C[r1 * N + c0 + 1] = v3;
        }}
    }}
}}
"#, n=n, mt=mt, nt=nt, kt=kt, tiles_m=tiles_m, tiles_n=tiles_n, tiles_k=tiles_k, warp_cols=warp_cols, warp_mt=warp_mt, warp_nt=warp_nt,
    e0=self.emit_epilogue(config.epilogue.as_slice(), "v0", "c0"),
    e1=self.emit_epilogue(config.epilogue.as_slice(), "v1", "(c0 + 1)"),
    e2=self.emit_epilogue(config.epilogue.as_slice(), "v2", "c0"),
    e3=self.emit_epilogue(config.epilogue.as_slice(), "v3", "(c0 + 1)"))
    }
}

impl Emitter for CUDAEmitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String {
        match req {
            SyncRequirement::WaitAsyncLoad { stages_behind } => {
                format!("asm volatile(\"cp.async.wait_group %0;\" : : \"n\"({}));", stages_behind)
            }
            SyncRequirement::Barrier => "__syncthreads();".to_string(),
            SyncRequirement::None => "".to_string(),
        }
    }

    fn emit_epilogue(&self, ops: &[EpilogueOp], acc_name: &str, global_n: &str) -> String {
        let mut code = String::new();
        for op in ops {
            match op {
                EpilogueOp::BiasAdd { bias_ptr } => {
                    code.push_str(&format!("  {{ \n    float* b_ptr = (float*)(unsigned long long)0x{ptr:x};\n    {acc} += b_ptr[{gn}];\n  }}\n", acc = acc_name, ptr = bias_ptr, gn = global_n));
                }
                EpilogueOp::ReLU => {
                    code.push_str(&format!("  {acc} = ({acc} > 0.0f) ? {acc} : 0.0f;\n", acc = acc_name));
                }
                EpilogueOp::Gelu => {
                    code.push_str(&format!("  {acc} *= 0.5f * (1.0f + tanhf(0.79788456f * ({acc} + 0.044715f * {acc} * {acc} * {acc})));\n", acc = acc_name));
                }
                EpilogueOp::None => {}
            }
        }
        code
    }

    fn emit_fragment_op(&self, op: crate::semantic::fragment::FragmentOp, _frags: &[crate::semantic::fragment::Fragment]) -> String {
        match op {
            crate::semantic::fragment::FragmentOp::LoadTC { is_x4, transposed } => {
                let suffix = if is_x4 { ".x4" } else { ".x1" };
                let trans_suffix = if transposed { ".trans" } else { "" };
                format!("ldmatrix.sync.aligned.m8n8{}.shared.b16", suffix)
            }
            crate::semantic::fragment::FragmentOp::MMA { m, n, k } => {
                format!("mma.sync.aligned.m{}n{}k{}.row.col.f32.f16.f16.f32", m, n, k)
            }
            crate::semantic::fragment::FragmentOp::FMA => "fma".to_string(),
        }
    }
}
