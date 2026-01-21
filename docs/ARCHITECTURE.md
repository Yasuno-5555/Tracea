# Tracea Architecture: The 6-Layer Semantic Model 🏛️

Tracea is built on a strictly layered architecture that separates "Meaning" (Semantic IR) from "Physics" (Emitter/Hardware).

## Layer Overview

### L1: User Interface
- **Python**: PyO3 bindings via `tracea-python` crate. Zero-copy PyTorch integration.
- **C++**: RAII header (`tracea.hpp`) wrapping stable C-FFI from `tracea-ffi` crate.
- **Rust**: Native entry points for kernel dispatch and hardware diagnostics.

### L2: Core IR
- Defines logical operations: `GemmOp`, `Conv2dOp`, `AttentionOp`.
- **Shape-based Problem Descriptors** (v3.1): Unified dimension handling via `Shape` struct.

### L3: Tracea Doctor (The Brain)
- **Capability Profiler**: Queries hardware specifics (Shared Mem, Warp size, Tensor Core availability).
- **Variant Registry**: Multi-backend catalog of kernel implementations.
- **Decision Engine**: Requirement-matching and scoring for optimal variant selection.

### L4: Semantic IR (The Heart)
- **Phase Transition**: Models Z/NZ cyclicity in asynchronous pipes.
- **Lane Mapping**: Matrix Core / Tensor Core register layout abstractions.
- **Swizzle Mode**: Algebraic bank conflict resolution (XOR swizzle).
- **BarrierMode** (v3.1): `mbarrier` integration for producer-consumer patterns.

### L5: Universal Emitters
- **CUDA**: PTX pipelining, Register Double Buffering, Tensor Core MMA.
- **HIP**: AMD GCN/CDNA intrinsics (v_mfma).
- **Metal**: Apple Silicon simdgroup support.
- **CPU**: SIMD (AVX512, AVX2, NEON) with packed data layouts.

### L6: Optimizer
- **Bayesian Auto-tuner**: Gaussian Processes + UCB exploration.
- **HeroScope v3**: Architecture-aware pre-computed hero configurations.
- **Persistent Cache**: Hardware-fingerprinted tuning results.

---

## v3.1 Module Additions

| Module | Purpose |
|--------|---------|
| `tracea-python/` | PyO3 extension with zero-copy TensorView |
| `tracea-ffi/` | C ABI with panic containment |
| `src/optimizer/problem.rs` | Shape-based ProblemDescriptor |
| `src/core/config.rs` | BarrierMode enum |

---

## Data Flow: From User to Kernel

1. **Capture**: Python call or C++ invocation defines logical intent.
2. **Diagnostics**: Doctor profiles environment and backends.
3. **Planning**: Selects optimal Kernel Variant.
4. **Optimization**: AutoTuner searches for best PipelineConfig.
5. **Emission**: Emitter generates kernel (PTX/HIP/MSL/SIMD).
6. **Launch**: RuntimeManager executes with zero-copy buffers.
