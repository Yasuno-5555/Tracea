# Tracea GEMM Design Specification

> [!IMPORTANT]
> This document serves as the "Constitution" for Tracea's kernel implementations. Deviations should be justified and documented.

## 1. Core Architecture

The Tracea GEMM kernel is designed around the NVIDIA Ampere (sm_80+) architecture, leveraging asynchronous data movement (`cp.async`) and Tensor Cores (`mma.sync`).

### 1.1 Warp Roles
We utilize a **Producer-Consumer** model within a single Thread Block (CTA).
*   **Producer Warp (Warp 0)**: Sole responsibility is ensuring data is loaded from Global Memory (GMEM) to Shared Memory (SMEM). It issues `cp.async` instructions and manages barriers (`cp.async.mbarrier` or grouping).
*   **Consumer Warps (Warps 1..N)**: Responsible for:
    *   Loading fragments from SMEM to Registers.
    *   Executing INT4/FP16/TF32 Tensor Core Math (`mma.sync`).
    *   Storing results to Accumulators.
    *   Writing final output to GMEM (Epilogue).

**Default Configuration:**
*   Total Warps: 8 or 9 (Configurable).
*   Producer Warps: 1.
*   Consumer Warps: 7 or 8.

### 1.2 Pipeline Structure
We implement an **N-Stage Asynchronous Pipeline**.
*   **Prologue**: Pre-loads `STAGES - 1` tiles into SMEM.
*   **Main Loop**:
    *   **Consumer**: Computes on Tile `k % STAGES`.
    *   **Producer**: Issues loads for Tile `k + STAGES - 1` into SMEM buffer `(k + STAGES - 1) % STAGES`.
    *   **Synchronization**:
        *   Producer commits `cp_async_commit_group()`.
        *   Producer waits `cp_async_wait_group<STAGES - 2>()` to ensure the buffer needed by Consumers in the *next* iteration is ready.
*   **Epilogue (Loop Drain)**:
    *   No new loads issued.
    *   Producer simply waits `cp_async_wait_group<0>()` to drain remaining in-flight groups.

### 1.3 Tiling Strategy
The "Hero" configuration for Ampere (e.g., RTX 3070/3090) targets:
*   **Block Shape**: `128x128x32` (M=128, N=128, K=32).
*   **Warps**: 8 (256 threads) or 4 (128 threads).
*   **Stages**: 3 (Optimal occupancy vs SMEM usage balance).

## 2. Memory Layout

### 2.1 Shared Memory (SMEM)
To maximize throughput and minimize bank conflicts, we strictly control SMEM layout.

*   **Bank Padding**: We append **8 elements** (16 bytes for f16) to the leading dimension stride of each tile.
    *   `A_STRIDE = K_TILE + 8`
    *   `B_STRIDE = N_TILE + 8` (Assumes ColMajor B or Transposed load? Default is RowMajor A, RowMajor B? Check code).
    *   *Correction*: Implementation uses `A_STRIDE = kt + 8` and `B_STRIDE = nt + 8`.
*   **Double/Triple Buffering**: Implicitly handled by the `STAGES` dimension in the SMEM array.
    *   `smem_size = (A_Tile_Size + B_Tile_Size) * STAGES`

### 2.2 Layout Policy Philosophy
*   **RowMajor**: The default. Simplest to reason about. Compatible with most C layouts.
*   **ColumnMajor**: Supported if needed by specific BLAS bindings.
*   **Layout Responsibility**: The Operating System (or Host Runtime) ensures data is in a usable state. The Kernel expects contiguous buffers (or strided if supported, but currently contiguous within tiles).
    > "The Layout's responsibility is on the OS."

## 3. Future Alignment (FA2 / Conv)
This design is a template for FlashAttention-2 and Convolution kernels.
*   **FA2**: Will inherit the Producer-Consumer split. Q, K, V loads will be handled by the Producer. Attention Score computation (S = QK^T) and Context (O = SV) will be Consumer tasks.
*   **Conv**: Im2Col or implicit GEMM strategies will map to this 128x128x32 tiling strategy where possible.

