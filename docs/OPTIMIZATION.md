# Tracea Optimization: Bayesian Tuning & Pipelining 🚀

This document details the high-performance techniques used in Tracea to achieve and exceed cuBLAS performance.

## 1. Phasic Pipelining (L3/L5)
Tracea uses **Software Pipelining** to overlap memory transfers with computation.
- **Stage 0**: Global Memory -> Shared Memory (Asynchronous).
- **Stage 1**: Shared Memory -> Registers.
- **Stage 2**: Compute (FMA).

The `SyncRequirement` enum manages the necessary wait-states (`cp.async.wait_group` or barriers) to ensure data integrity without over-synchronizing.

## 2. Implicit GEMM Optimizations (v3.2)
Tracea's Conv2d engine completely avoids `im2col` memory expansion by computing indices on the fly.
- **Magic Number Division**: Replaces expensive integer division (`/`, `%`) with high-throughput 32-bit fixed-point multiplication for coordinate recovery.
- **Hybrid Hoisting**: Automatically decides whether to pre-compute coordinates into Shared Memory (for speed) or re-compute them (to save SMEM for larger tiles), based on the `smem_size < 96KB` threshold.
- **Alignment-Safe Epilogues**: Vectorized `uint4` stores for aligned channels, falling back to conditional scalar stores for odd shapes ($C \pmod 8 \neq 0$), ensuring safety without penalizing the common case.

## 3. Backend-Aware Variant Selection (Tracea Doctor)
Tracea doesn't just tune; it *diagnoses*. The **Doctor** performs a multi-objective search:
- **Requirement Filtering**: Discards variants that exceed physical hardware limits (e.g., shared memory).
- **Priority Scoring**: Prefers Tensor Cores on NVIDIA and Matrix Cores on AMD.
- **Precision Fallback**: Automatically downgrades from BF16 to FP16/FP32 if the hardware lacks native support.

## 4. Register Double Buffering & SIMD (Cross-Backend)
- **CUDA**: Interleaves register loads with FMA.
- **Metal/ROCm**: Leverages `simdgroup` and `wavefront` level parallelism to hide memory latency.
- **CPU**: Uses multi-threaded blocking and AVX512/AVX2 intrinsics to maximize host-side throughput.

## 5. Swizzle & Bank Conflict Resolution
Tracea automatically generates **XOR-swizzled shared memory addresses**. The `BankConflictSimulator` in L3 verifies that the generated access patterns result in zero bank conflicts for the target hardware's bank count (typically 32).
