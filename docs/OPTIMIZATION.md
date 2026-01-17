# Tracea Optimization: Bayesian Tuning & Pipelining 🚀

This document details the high-performance techniques used in Tracea to achieve and exceed cuBLAS performance.

## 1. Phasic Pipelining (L3/L5)
Tracea uses **Software Pipelining** to overlap memory transfers with computation.
- **Stage 0**: Global Memory -> Shared Memory (Asynchronous).
- **Stage 1**: Shared Memory -> Registers.
- **Stage 2**: Compute (FMA).

The `SyncRequirement` enum manages the necessary wait-states (`cp.async.wait_group` or barriers) to ensure data integrity without over-synchronizing.

## 2. Bayesian Auto-Tuning (L6)
Instead of a simple grid search, Tracea uses a **Gaussian Process (GP)** surrogate model.
- **Search Space**: $(N\_stages, M\_tile, N\_tile, K\_tile, SwizzleMode)$.
- **Acquisition Function**: Expected Improvement (EI) is used to balance exploration of unknown configurations and exploitation of known good ones.
- **Pruning**: Configurations that exceed hardware limits (e.g., shared memory capacity) are analytically pruned before benchmarks run.

## 3. Register Double Buffering (CUDA)
To maximize Arithmetic Unit utilization, Tracea interleaves the *next* register load with the *current* FMA instructions. This effectively hides the register load latency and minimizes stalls in the GPU scheduler.

## 4. Swizzle & Bank Conflict Resolution
Tracea automatically generates **XOR-swizzled shared memory addresses**. The `BankConflictSimulator` in L3 verifies that the generated access patterns result in zero bank conflicts for the target hardware's bank count (typically 32).
