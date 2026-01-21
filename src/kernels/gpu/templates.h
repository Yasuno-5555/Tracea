#ifndef TRACEA_TEMPLATES_H
#define TRACEA_TEMPLATES_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" {

void launch_gemm_v3_s2_sw0_b0_m128_n128_k32(const __half *A, const __half *B,
                                            float *C, int M, int N, int K,
                                            int gridX, int gridY, int gridZ,
                                            int blockX, int blockY, int blockZ,
                                            int smem, void *stream);

void launch_gemm_v3_s3_sw1_b1_m128_n128_k32(const __half *A, const __half *B,
                                            float *C, int M, int N, int K,
                                            int gridX, int gridY, int gridZ,
                                            int blockX, int blockY, int blockZ,
                                            int smem, void *stream);
}

#endif
