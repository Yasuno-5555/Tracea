# Tracea Architecture: The 6-Layer Semantic Model 🏛️

Tracea is built on a strictly layered architecture that separates "Meaning" (Semantic IR) from "Physics" (Emitter/Hardware).

## Layer Overview

### L1: User Interface
- **Python**: PyO3 bindings via `tracea-python` crate. Zero-copy PyTorch integration.
- **C++**: RAII header (`tracea.hpp`) wrapping stable C-FFI from `tracea-ffi` crate.
- **Rust**: Native entry points for kernel dispatch and hardware diagnostics.

### L2: Core IR & TTG (Topological Tile Graph)
- Defines logical operations: `GemmOp`, `Conv2dOp`, `AttentionOp`.
- **TTG Layout**: A hardware-agnostic tiling plan consisting of an L1 Map (logical assignment) and L2 Table (tile metadata).
- **Shape-based Problem Descriptors**: Unified dimension handling for arbitrary tensor shapes.

### L3: Tracea Doctor (The Brain)
- **Capability Profiler**: Queries hardware specifics (Shared Mem, Warp size, Tensor Core availability).
- **Variant Registry**: Multi-backend catalog of kernel implementations.
- **Decision Engine**: Requirement-matching and scoring for optimal variant selection.

### L4: Semantic IR (The Heart)
- **TTG Builder**: Translates high-level logical intent or Policy Decisions into concrete `TTGLayout` objects.
- **Phase Transition**: Models Z/NZ cyclicity in asynchronous pipes.
- **Lane Mapping**: Matrix Core / Tensor Core register layout abstractions.
- **Swizzle Mode**: Algebraic bank conflict resolution (XOR swizzle).
- **BarrierMode**: `mbarrier` integration for producer-consumer patterns.

### L5: Universal Emitters
- **CUDA**: PTX pipelining, Register Double Buffering, Tensor Core MMA.
- **Implicit GEMM**: Zero-im2col convolution via magic-number coordinate mapping.
- **HIP**: AMD GCN/CDNA intrinsics (v_mfma).
- **Metal**: Apple Silicon simdgroup support.
- **CPU**: SIMD (AVX512, AVX2, NEON) with packed data layouts.

### L6: Optimizer & Policy Engine
- **Policy Engine**: High-level planner that decides Tiling Kind (Dense, Sparse, Low-Rank) and Execution Order based on Context.
- **Bayesian Auto-tuner**: Gaussian Processes + UCB exploration for fine-tuning kernel parameters.
- **HeroScope v3**: Architecture-aware pre-computed hero configurations.
- **Persistent Cache**: Hardware-fingerprinted tuning results.

---

## v3.2 Module Additions

| Module | Purpose |
|--------|---------|
| `src/policy/` | Policy Engine and Decision Types |
| `src/runtime/ttg*` | TTG construction and device buffer management |
| `src/semantic/` | Tiling patterns and Semantic IR abstractions |
| `tracea-ffi/` | C ABI with panic containment |

---

## Data Flow: From User to Kernel

1. **Capture**: Logical intent captured via Python/C++/Rust APIs.
2. **Diagnostics**: Doctor profiles environment (CUDA Compute Capability, etc).
3. **Policy Planning**: Policy Engine selects Tiling strategy (e.g., Low-Rank for MLP).
4. **TTG Generation**: TTG Builder creates the topological tile graph.
5. **Tuning**: AutoTuner optimizes the specific kernel parameters for the selected TTG.
6. **Emission**: Universal emitter generates target code (PTX/MSL/HIP).
7. **Launch**: RuntimeManager uploads TTG buffers and executes the kernel.
