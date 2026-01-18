use crate::emitter::traits::Emitter;
use crate::semantic::transition::SyncRequirement;
use crate::core::config::SwizzleMode;
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
        
        // Assume non-Tensor Core is FP32
        let swizzle_shift = match config.swizzle_mode { SwizzleMode::Xor2 => 1, SwizzleMode::Xor4 => 2, SwizzleMode::Xor8 => 3, _ => 0};
        let swizzle_mask = match config.swizzle_mode { SwizzleMode::Xor2 => 1, SwizzleMode::Xor4 => 3, SwizzleMode::Xor8 => 7, _ => 0};

        format!(r#"
#include <cuda_fp16.h>

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
    #define SWZ_B(s, r, c) ( (s) * {kt} * {s_b} + (r) * {s_b} + ((c) ^ (((r) >> {swizzle_shift}) & {swizzle_mask})) )
    
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
                {epilogue_code}
                C[global_m * N + global_n] = sum;
            }}
        }}
    }}
}}
"#, n=n, mt=mt, nt=nt, kt=kt, wait_stages=if n > 1 { n - 2 } else { 0 }, reg_m=reg_m, reg_n=reg_n, s_a=s_a, s_b=s_b, 
        swizzle_shift=swizzle_shift, swizzle_mask=swizzle_mask, epilogue_code=epilogue_code
        )
    }

    pub fn generate_tensor_core_gemm(&self, config: crate::PipelineConfig) -> String {
        let n = config.num_stages;
        let mt = config.m_tile;
        let nt = config.n_tile;
        let kt = config.k_tile;
        let quant = config.quantization;
        
        let is_int4 = quant == crate::core::config::QuantizationMode::Int4;
        let b_type = if is_int4 { "int" } else { "unsigned short" };
        let b_elem_bits = if is_int4 { 4 } else { 16 };
        let elems_per_128b_b = if is_int4 { 32 } else { 8 };
        let loads_per_128b_b = if is_int4 { 4 } else { 8 }; 
        
        let num_warps = 4;
        let warp_rows = if mt >= nt { 2 } else { 1 };
        let warp_cols = num_warps / warp_rows;
        
        let warp_m = mt / warp_rows;
        let warp_n = nt / warp_cols;
        
        let mma_m = 16;
        let mma_n = 8;
        let mma_k = 16;
        let n_stages = config.num_stages;

        let tiles_m = (mt / 2) / 16;  
        let tiles_n = (nt / 2) / 8;
        let tiles_k = kt / 16; 

        let warp_rows = mt / 2;
        let warp_cols = nt / 2;

        let loads_a = (mt * kt + 128 * 8 - 1) / (128 * 8); 
        
        let b_bytes = (kt * nt * b_elem_bits) / 8;
        let loads_b = (b_bytes + 15) / 16;

        let unpack_func = if is_int4 {
            r#"
__device__ __forceinline__ void unpack_int4_to_half2(int packed, half2* out) {
    int i0 = (packed) & 0xF;
    int i1 = (packed >> 4) & 0xF;
    int i2 = (packed >> 8) & 0xF;
    int i3 = (packed >> 12) & 0xF;
    
    int i4 = (packed >> 16) & 0xF;
    int i5 = (packed >> 20) & 0xF;
    int i6 = (packed >> 24) & 0xF;
    int i7 = (packed >> 28) & 0xF;
    
    out[0] = __floats2half2_rn((float)i0, (float)i1);
    out[1] = __floats2half2_rn((float)i2, (float)i3);
    out[2] = __floats2half2_rn((float)i4, (float)i5);
    out[3] = __floats2half2_rn((float)i6, (float)i7);
}
            "#
        } else { "" };

        let smem_b_size_bytes = (mt * kt * n_stages * 2) + (kt * nt * n_stages * b_elem_bits / 8);
        
        let smem_decl = if is_int4 {
             format!(r#"
    extern __shared__ __align__(16) unsigned char smem_bytes[];
    unsigned short* As = (unsigned short*)smem_bytes;
    int* Bs = (int*)&smem_bytes[{mt} * {kt} * {n_stages} * 2]; // 2 bytes per A
             "#, mt=mt, kt=kt, n_stages=n_stages)
        } else {
             format!(r#"
    extern __shared__ __align__(16) unsigned short smem_h[];
    unsigned short* As = &smem_h[0];
    unsigned short* Bs = &smem_h[{mt} * {kt} * {n_stages}];
             "#, mt=mt, kt=kt, n_stages=n_stages)
        };
        
        let load_b_template = if is_int4 {
             // Template using {k_val} and {stage_val}
             // loads_b is number of 128-bit chunks.
             // We use 4 byte copies for safety in Int4 mode for now
             let iters = loads_b * 4; 
             format!(r#"
            #pragma unroll
            for (int i = 0; i < {iters}; ++i) {{
                int idx = i * 128 + tid;
                int r_pack = idx / {nt}; 
                int c = idx % {nt};
                
                if (r_pack < ({kt}/8)) {{
                    long long mk_packed_offset = (long long){{k_val}} / 8; 
                    long long gr = mk_packed_offset + r_pack;
                    long long gc = (long long)bx * {nt} + c;
                    
                    int* dst = &Bs[{{stage_val}} * ({kt}/8) * {nt} + r_pack * {nt} + c];
                    
                    if ((gr * 8) < K && gc < N) {{
                        const int* src = &B[gr * N + gc];
                        unsigned int sp = get_smem_ptr(dst);
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sp), "l"((unsigned long long)src));
                    }} else {{ *dst = 0; }}
                }}
            }}
             "#, iters=iters, kt=kt, nt=nt)
        } else {
             format!(r#"
            #pragma unroll
            for (int i = 0; i < {loads_b}; ++i) {{
                int idx = i * 128 + tid;
                int r = (idx * 8) / {nt};
                int c = (idx * 8) % {nt};
                if (r < {kt}) {{
                    long long gr = (long long){{k_val}} + r;
                    long long gc = (long long)bx * {nt} + c;
                    unsigned short* dst = &Bs[{{stage_val}} * {kt} * {nt} + r * {nt} + c];
                    if (gr < K && gc < N) {{
                        const unsigned short* src = &B[gr * N + gc];
                        unsigned int sp = get_smem_ptr(dst);
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(sp), "l"((unsigned long long)src));
                    }} else {{ *((float4*)dst) = make_float4(0,0,0,0); }}
                }}
            }}
             "#, loads_b=loads_b, kt=kt, nt=nt)
        };
        
        let loads_b_prologue = load_b_template.replace("{k_val}", "k_offset").replace("{stage_val}", "s");
        let loads_b_main = load_b_template.replace("{k_val}", "k_next").replace("{stage_val}", "load_s");
        
        let b_frag_load = if is_int4 {
            format!(r#"
            #pragma unroll
            for (int j = 0; j < {tiles_n}; ++j) {{
                // Interleaved Mapping:
                // T0 loads Col 0 and Col 1.
                // Pack: Reg0=(C0.x, C1.x), Reg1=(C0.y, C1.y).
                // Requires n=(lane%4)*2.
                
                int n_base = (lane_id % 4) * 2; 
                int sub_idx = (lane_id / 4) % 4; // K-group distribution
                
                int n0 = j * 8 + n_base;
                int n1 = n0 + 1;
                
                // Load Col 0
                unsigned int addr0 = get_smem_ptr(&b_warp_ptr[(k_inner*2) * ({nt}) + n0]); 
                int packed0; 
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(packed0) : "r"(addr0));
                half2 unp0[4]; unpack_int4_to_half2(packed0, unp0);
                
                // Load Col 1
                unsigned int addr1 = get_smem_ptr(&b_warp_ptr[(k_inner*2) * ({nt}) + n1]);
                int packed1;
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(packed1) : "r"(addr1));
                half2 unp1[4]; unpack_int4_to_half2(packed1, unp1);
                
                half2 vals0 = unp0[sub_idx]; // (C0 K0, C0 K1)
                half2 vals1 = unp1[sub_idx]; // (C1 K0, C1 K1)
                
                half k0_c0 = __low2half(vals0);
                half k1_c0 = __high2half(vals0);
                half k0_c1 = __low2half(vals1);
                half k1_c1 = __high2half(vals1);
                
                // Pack interleaved: (C0, C1)
                half2 res0 = __halves2half2(k0_c0, k0_c1);
                half2 res1 = __halves2half2(k1_c0, k1_c1);
                
                frag_b[j][0] = *(unsigned int*)&res0;
                frag_b[j][1] = *(unsigned int*)&res1;

                if (bx==0 && by==0 && tid==0 && k_inner==0 && j==0) {{
                     printf("[Thread 0 Interleaved] C0K0: %f C1K0: %f\n", __half2float(k0_c0), __half2float(k0_c1));
                }}
            }}
            "#, tiles_n=tiles_n, nt=nt)
        } else {
             format!(r#"
            #pragma unroll
            for (int j = 0; j < {tiles_n}; ++j) {{
                unsigned int addr = get_smem_ptr(&b_warp_ptr[(k_inner * 16) * {nt} + j * 8 + (lane_id % 16) * {nt}]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%0, %1}}, [%2];"
                    : "=r"(frag_b[j][0]), "=r"(frag_b[j][1]) : "r"(addr));
            }}
             "#, tiles_n=tiles_n, nt=nt)
        };

        format!(r#"
#include <cuda_fp16.h>

{unpack_func}

__device__ __forceinline__ unsigned int get_smem_ptr(const void *ptr) {{
    unsigned int ret;
    asm volatile("{{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }}" : "=r"(ret) : "l"(ptr));
    return ret;
}}

extern "C" __global__ void gemm_pipelined_kernel(
    const unsigned short* __restrict__ A, 
    const {b_type}* __restrict__ B, 
    float* __restrict__ C,
    int M_in, int N_in, int K_in
) {{
    int M = M_in;
    int N = N_in;
    int K = K_in;
    
    {smem_decl}

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = (warp_id / 2) * {warp_rows}; 
    int warp_n = (warp_id % 2) * {warp_cols};

    unsigned int frag_a[{tiles_m}][4];
    unsigned int frag_b[{tiles_n}][2];
    float acc[{tiles_m}][{tiles_n}][4];

    #pragma unroll
    for (int i = 0; i < {tiles_m}; ++i) {{
        #pragma unroll
        for (int j = 0; j < {tiles_n}; ++j) {{
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        }}
    }}

    // PROLOGUE
    #pragma unroll
    for (int s = 0; s < {n_stages} - 1; ++s) {{
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
            
            {loads_b_prologue}
            
            asm volatile("cp.async.commit_group;");
        }}
    }}

    // MAIN LOOP
    for (int k_step = 0; k_step < K / {kt}; ++k_step) {{
        int load_s = (k_step + {n_stages} - 1) % {n_stages};
        int k_next = (k_step + {n_stages} - 1) * {kt};
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
            
            {loads_b_main}

            asm volatile("cp.async.commit_group;");
        }}

        asm volatile("cp.async.wait_group {wait_stages};");
        __syncthreads();

        int comp_s = k_step % {n_stages};
        unsigned short* a_warp_ptr = &As[comp_s * {mt} * {kt} + warp_m * {kt}];
        {b_type}* b_warp_ptr = ({b_type}*)&Bs[comp_s * ({kt_scaled}) * {nt} + warp_n]; 

        #pragma unroll
        for (int k_inner = 0; k_inner < {tiles_k}; ++k_inner) {{
            #pragma unroll
            for (int i = 0; i < {tiles_m}; ++i) {{
                unsigned int addr = get_smem_ptr(&a_warp_ptr[(i * 16) * {kt} + k_inner * 16 + (lane_id % 16) * {kt} + (lane_id / 16) * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];"
                    : "=r"(frag_a[i][0]), "=r"(frag_a[i][1]), "=r"(frag_a[i][2]), "=r"(frag_a[i][3]) : "r"(addr));
            }}
            
            {b_frag_load}

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
                    
                    if (bx==0 && by==0 && tid==0 && k_inner==0 && i==0 && j==0) {{
                    }}
                }}
            }}
        }}
        __syncthreads();
    }}
    
    if (bx==0 && by==0 && tid==0) {{
    }}

    #pragma unroll
    for (int i = 0; i < {tiles_m}; ++i) {{
        #pragma unroll
        for (int j = 0; j < {tiles_n}; ++j) {{
            int rb = warp_m + i * 16;
            int cb = warp_n + j * 8;
            
            int r0 = by * {mt} + rb + (lane_id / 4);
            int r1 = r0 + 8;
            // Correct Col Logic (Standard):
            // T0..T3 -> Cols 0,1,2,3? (Stride 2?)
            // Standard: c_local = (lane % 4) * 2;
            int c_local = (lane_id % 4) * 2;
            int c0 = bx * {nt} + cb + c_local;
            int c1 = c0 + 1; 

            float v0 = acc[i][j][0];
            float v1 = acc[i][j][1];
            float v2 = acc[i][j][2];
            float v3 = acc[i][j][3];

            {e0}
            {e1}
            {e2}
            {e3}

            if (r0 < M && c0 < N) {{
                C[(long long)r0 * N + c0] = v0;
            }}
            if (r0 < M && (c0+1) < N) C[(long long)r0 * N + c0 + 1] = v1;
            if (r1 < M && c0 < N) C[(long long)r1 * N + c0] = v2;
            if (r1 < M && (c0+1) < N) C[(long long)r1 * N + c0 + 1] = v3;
        }}
    }}
}}
"#
, n_stages=n_stages, mt=mt, nt=nt, kt=kt, tiles_m=tiles_m, tiles_n=tiles_n, tiles_k=tiles_k, 
  wait_stages=0,
  warp_rows=warp_rows, warp_cols=warp_cols,
  e0=self.emit_epilogue(config.epilogue.as_slice(), "v0", "c0"),
  e1=self.emit_epilogue(config.epilogue.as_slice(), "v1", "(c0 + 1)"),
  e2=self.emit_epilogue(config.epilogue.as_slice(), "v2", "c0"),
  e3=self.emit_epilogue(config.epilogue.as_slice(), "v3", "(c0 + 1)"),
  unpack_func=unpack_func, b_type=b_type, smem_decl=smem_decl, loads_b_prologue=loads_b_prologue, loads_b_main=loads_b_main, b_frag_load=b_frag_load,
  loads_a=loads_a,
  kt_scaled=if is_int4 { kt/8 } else { kt * 16 / 16 }  
  )
    }

    pub fn generate_fused_attention(&self, op: crate::core::op::FusedAttentionOp) -> String {
        let br = 64; 
        let bc = 32;
        let dh = op.dh;
        
        let scale_val = if op.scale_inv_sqrt_d {
             1.0 / (dh as f32).sqrt()
        } else {
             1.0
        };

    let causal_logic = if op.causal {
        format!(r#"
            int row_off = (m == 2 || m == 3 || m == 6 || m == 7) ? 8 : 0;
            int col_off = (m >= 4) ? 8 : 0;
            int r_idx_global = q_tile_idx * {br} + warp_id * 16 + (lane_id / 4) + row_off;
            int c_idx_global = j * {bc} + n * 16 + (lane_id % 4) * 2 + (m % 2) + col_off;
            if (c_idx_global > r_idx_global) S_frag[n][m] = -1e9f;
        "#, br=br, bc=bc)
    } else {
        "".to_string()
    };

        format!(r#"
#include <cuda_fp16.h>

__device__ __forceinline__ unsigned int get_smem_ptr(const void *ptr) {{
    unsigned int ret;
    asm volatile("{{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }}" : "=r"(ret) : "l"(ptr));
    return ret;
}}

extern "C" __global__ void flash_attention_v2_kernel(
    const half* __restrict__ Q,  
    const half* __restrict__ K,  
    const half* __restrict__ V,  
    half* __restrict__ O,        
    int B_size, int H, int S, int D_size,
    float scale
) {{
    int tid = threadIdx.x;
    int q_tile_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    long long head_offset = (long long)batch_idx * H * S * D_size + (long long)head_idx * S * D_size;
    const half* q_base = Q + head_offset + (long long)q_tile_idx * {br} * D_size;
    const half* k_ptr_base = K + head_offset;
    const half* v_ptr_base = V + head_offset;
    half* o_base = O + head_offset + (long long)q_tile_idx * {br} * D_size;

    __shared__ __align__(16) half sQ[{br} * {dh}]; 
    __shared__ __align__(16) half sK[{bc} * {dh}];
    __shared__ __align__(16) half sV[{dh} * {bc}]; // Transposed for ldmatrix

    // Per-row stats: Thread handles rows (lane_id/4) and (lane_id/4 + 8)
    float m_prev[2] = {{-1e9f, -1e9f}};
    float l_prev[2] = {{0.0f, 0.0f}};
    
    float O_reg[{dh}/16][8]; 
    #pragma unroll
    for(int i=0; i<{dh}/16; ++i) for(int k=0; k<8; k++) O_reg[i][k] = 0.0f;

    for (int i = tid; i < {br} * D_size; i += blockDim.x) sQ[i] = q_base[i];
    __syncthreads();

    int Tc = (S + {bc} - 1) / {bc};
    for (int j = 0; j < Tc; ++j) {{
        for (int i = tid; i < {bc} * D_size; i += blockDim.x) {{
            sK[i] = k_ptr_base[j * {bc} * D_size + i];
            // Transpose V into sV for ldmatrix
            int row = i / D_size;
            int col = i % D_size;
            sV[col * {bc} + row] = v_ptr_base[j * {bc} * D_size + i];
        }}
        __syncthreads();

        unsigned int q_frag[4]; 
        unsigned int k_frag[2][4]; 
        float S_frag[2][8]; 

        #pragma unroll
        for(int m=0; m<2; ++m) for(int n=0; n<8; n++) S_frag[m][n] = 0.0f;

        #pragma unroll
        for (int k_step = 0; k_step < {dh}/16; ++k_step) {{
            unsigned int addr_q = get_smem_ptr(&sQ[(warp_id * 16 + (lane_id % 16)) * D_size + k_step * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];"
                : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3]) : "r"(addr_q));

            #pragma unroll
            for (int n_step = 0; n_step < 2; ++n_step) {{
                unsigned int addr_k = get_smem_ptr(&sK[(n_step * 16 + (lane_id % 16)) * D_size + k_step * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];"
                    : "=r"(k_frag[n_step][0]), "=r"(k_frag[n_step][1]), "=r"(k_frag[n_step][2]), "=r"(k_frag[n_step][3]) : "r"(addr_k));

                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%10, %11, %12, %13}};"
                    : "=f"(S_frag[n_step][0]), "=f"(S_frag[n_step][1]), "=f"(S_frag[n_step][2]), "=f"(S_frag[n_step][3])
                    : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]),
                      "r"(k_frag[n_step][0]), "r"(k_frag[n_step][1]),
                      "f"(S_frag[n_step][0]), "f"(S_frag[n_step][1]), "f"(S_frag[n_step][2]), "f"(S_frag[n_step][3]));

                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%10, %11, %12, %13}};"
                    : "=f"(S_frag[n_step][4]), "=f"(S_frag[n_step][5]), "=f"(S_frag[n_step][6]), "=f"(S_frag[n_step][7])
                    : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]),
                      "r"(k_frag[n_step][2]), "r"(k_frag[n_step][3]),
                      "f"(S_frag[n_step][4]), "f"(S_frag[n_step][5]), "f"(S_frag[n_step][6]), "f"(S_frag[n_step][7]));
            }}
        }}

        float m_curr[2] = {{-1e9f, -1e9f}};
        #pragma unroll
        for(int n=0; n<2; n++) for(int m=0; m<8; m++) {{
            S_frag[n][m] *= scale;
            {causal_logic}
            int r_idx = (m == 2 || m == 3 || m == 6 || m == 7) ? 1 : 0;
            if (S_frag[n][m] > m_curr[r_idx]) m_curr[r_idx] = S_frag[n][m];
        }}
        // Row-wise Max Reduction across threads 0-3 (group of 4 threads handles 16 elements of 2 rows)
        m_curr[0] = fmaxf(m_curr[0], __shfl_xor_sync(0xffffffff, m_curr[0], 1));
        m_curr[0] = fmaxf(m_curr[0], __shfl_xor_sync(0xffffffff, m_curr[0], 2));
        m_curr[1] = fmaxf(m_curr[1], __shfl_xor_sync(0xffffffff, m_curr[1], 1));
        m_curr[1] = fmaxf(m_curr[1], __shfl_xor_sync(0xffffffff, m_curr[1], 2));

        float m_next[2];
        float alpha[2];
        float l_curr[2] = {{0.0f, 0.0f}};
        unsigned int p_frag_half[2][4]; 

        #pragma unroll
        for(int r=0; r<2; r++) {{
            m_next[r] = fmaxf(m_prev[r], m_curr[r]);
            alpha[r] = expf(m_prev[r] - m_next[r]);
        }}

        #pragma unroll
        for(int n=0; n<2; ++n) {{
            half2 ph0 = __halves2half2(__float2half(expf(S_frag[n][0] - m_next[0])), __float2half(expf(S_frag[n][1] - m_next[0])));
            half2 ph1 = __halves2half2(__float2half(expf(S_frag[n][2] - m_next[1])), __float2half(expf(S_frag[n][3] - m_next[1])));
            half2 ph2 = __halves2half2(__float2half(expf(S_frag[n][4] - m_next[0])), __float2half(expf(S_frag[n][5] - m_next[0])));
            half2 ph3 = __halves2half2(__float2half(expf(S_frag[n][6] - m_next[1])), __float2half(expf(S_frag[n][7] - m_next[1])));
            p_frag_half[n][0] = *(unsigned int*)&ph0;
            p_frag_half[n][1] = *(unsigned int*)&ph1;
            p_frag_half[n][2] = *(unsigned int*)&ph2;
            p_frag_half[n][3] = *(unsigned int*)&ph3;
            l_curr[0] += __half2float(ph0.x) + __half2float(ph0.y) + __half2float(ph2.x) + __half2float(ph2.y);
            l_curr[1] += __half2float(ph1.x) + __half2float(ph1.y) + __half2float(ph3.x) + __half2float(ph3.y);
        }}
        // Row-wise Sum Reduction
        l_curr[0] += __shfl_xor_sync(0xffffffff, l_curr[0], 1);
        l_curr[0] += __shfl_xor_sync(0xffffffff, l_curr[0], 2);
        l_curr[1] += __shfl_xor_sync(0xffffffff, l_curr[1], 1);
        l_curr[1] += __shfl_xor_sync(0xffffffff, l_curr[1], 2);

        #pragma unroll
        for(int i=0; i<{dh}/16; ++i) {{
            #pragma unroll
            for(int k=0; k<8; k++) {{
                int r_idx = (k == 2 || k == 3 || k == 6 || k == 7) ? 1 : 0;
                O_reg[i][k] *= alpha[r_idx];
            }}
        }}

        // --- MMC 2: O += P * V ---
        unsigned int v_frag[4]; 
        #pragma unroll
        for (int d_step = 0; d_step < {dh}/16; ++d_step) {{
             #pragma unroll
             for (int k_idx = 0; k_idx < 2; ++k_idx) {{
                 unsigned int addr_v = get_smem_ptr(&sV[(d_step * 16 + (lane_id % 16)) * {bc} + k_idx * 16]);
                 asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];"
                    : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3]) : "r"(addr_v));
                 
                 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%10, %11, %12, %13}};"
                    : "=f"(O_reg[d_step][0]), "=f"(O_reg[d_step][1]), "=f"(O_reg[d_step][2]), "=f"(O_reg[d_step][3])
                    : "r"(p_frag_half[k_idx][0]), "r"(p_frag_half[k_idx][1]), "r"(p_frag_half[k_idx][2]), "r"(p_frag_half[k_idx][3]),
                      "r"(v_frag[0]), "r"(v_frag[1]),
                      "f"(O_reg[d_step][0]), "f"(O_reg[d_step][1]), "f"(O_reg[d_step][2]), "f"(O_reg[d_step][3]));

                 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%10, %11, %12, %13}};"
                    : "=f"(O_reg[d_step][4]), "=f"(O_reg[d_step][5]), "=f"(O_reg[d_step][6]), "=f"(O_reg[d_step][7])
                    : "r"(p_frag_half[k_idx][0]), "r"(p_frag_half[k_idx][1]), "r"(p_frag_half[k_idx][2]), "r"(p_frag_half[k_idx][3]),
                      "r"(v_frag[2]), "r"(v_frag[3]),
                      "f"(O_reg[d_step][4]), "f"(O_reg[d_step][5]), "f"(O_reg[d_step][6]), "f"(O_reg[d_step][7]));
             }}
        }}

        #pragma unroll
        for(int r=0; r<2; r++) {{
            m_prev[r] = m_next[r];
            l_prev[r] = l_prev[r] * alpha[r] + l_curr[r];
        }}
        __syncthreads();
    }}

    // Final Normalize and Store
    float inv_l[2];
    #pragma unroll
    for(int r=0; r<2; r++) inv_l[r] = 1.0f / (l_prev[r] + 1e-6f);

    #pragma unroll
    for (int d_step = 0; d_step < {dh}/16; ++d_step) {{
        int row_base = warp_id * 16;
        int c_base = d_step * 16;
        int r0 = row_base + (lane_id / 4);
        int r1 = r0 + 8;
        int c_off = (lane_id % 4) * 2;
        
        if (r0 < S) {{
            o_base[r0 * D_size + c_base + c_off]   = __float2half(O_reg[d_step][0] * inv_l[0]);
            o_base[r0 * D_size + c_base + c_off+1] = __float2half(O_reg[d_step][1] * inv_l[0]);
            o_base[r0 * D_size + c_base + c_off+8] = __float2half(O_reg[d_step][4] * inv_l[0]);
            o_base[r0 * D_size + c_base + c_off+9] = __float2half(O_reg[d_step][5] * inv_l[0]);
        }}
        if (r1 < S) {{
            o_base[r1 * D_size + c_base + c_off]   = __float2half(O_reg[d_step][2] * inv_l[1]);
            o_base[r1 * D_size + c_base + c_off+1] = __float2half(O_reg[d_step][3] * inv_l[1]);
            o_base[r1 * D_size + c_base + c_off+8] = __float2half(O_reg[d_step][6] * inv_l[1]);
            o_base[r1 * D_size + c_base + c_off+9] = __float2half(O_reg[d_step][7] * inv_l[1]);
        }}
    }}
}}
"#)

    }

    pub fn emit_epilogue(&self, ops: &[EpilogueOp], val_reg: &str, global_idx_reg: &str) -> String {
        let mut code = String::new();
        for op in ops {
            match op {
                EpilogueOp::ReLU => {
                    code.push_str(&format!("{} = fmaxf({}, 0.0f);\n", val_reg, val_reg));
                },
                EpilogueOp::Gelu => {
                    code.push_str(&format!("{{ float x = {}; {} = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }}\n", val_reg, val_reg));
                },
                EpilogueOp::BiasAdd { bias_ptr: _ } => {
                    // Bias pointer is usually passed as extra arg?
                    // For now, we assume bias is handled externally or not implemented in basic template
                    // But we generated code.
                    // Ideally we should pass bias ptr.
                },
                EpilogueOp::None => {}
            }
        }
        code
    }
}

impl Emitter for CUDAEmitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String {
        match req {
            SyncRequirement::WaitAsyncLoad { stages_behind } => {
                format!("asm volatile(\"cp.async.wait_group %0;\" : : \"n\"({}));", stages_behind)
            }
            SyncRequirement::Barrier => "__syncthreads();".to_string(),
            SyncRequirement::None => String::new(),
        }
    }
}
