# Tracea Architecture: The 6-Layer Semantic Model 🏛️

Tracea is built on a strictly layered architecture that separates "Meaning" (Semantic IR) from "Physics" (Emitter/Hardware).

## Layer Overview

### L1: User Interface
- **Python**: PyO3 bindings + TorchDynamo backend. Captures FX Graphs and translates them to L2.
- **C++**: RAII header (`tracea.hpp`) wrapping a stable C-FFI.
- **Core API**: Rust entry points for manual kernel dispatch and hardware diagnostics.

### L2: Core IR
- Defines the logical operation: `GemmOp(m, n, k)`, `FusedGemmOp`, `SoftmaxOp`.
- Represents the goal without specifying the implementation details.

### L3: Tracea Doctor (The Brain)
- **Capability Profiler**: Queries hardware specifics (Shared Mem, Warp size, SIMD width, Tensor Core availability).
- **Variant Registry**: Maintains a multi-backend catalog of kernel implementations.
- **Decision Engine**: Performs requirement-matching and scoring to select the optimal variant for the current environment.

### L4: Semantic IR (The Heart)
- **Phase Transition**: Models $Z/NZ$ cyclicity in asynchronous pipes.
- **Lane Mapping**: Matrix Core / Tensor Core / XMX register layout abstractions.
- **Swizzle Mode**: Algebraic bank conflict resolution (e.g., XOR swizzle).

### L4: Optimized IR
- A specific configuration of tiles, stages, and swizzles that has been validated for a specific hardware target.

### L5: Universal Emitters
- **CUDA Emitter**: PTX-level pipelining, Register Double Buffering.
- **HIP Emitter**: AMD GCN/CDNA intrinsics ($v\_mfma$).
- **Metal Emitter**: Apple Silicon `simdgroup` support.
- **CPU Emitter**: Host-side SIMD (AVX512, NEON) via specialized intrinsics.

### L6: Optimizer
- **Bayesian Auto-tuner**: Uses Gaussian Processes to explore the L4 space.
- **Micro-benchmarks**: Hardware-aware data collection.

---

## Data Flow: From Python to Kernel

1.  **Capture**: `torch.compile` or manual API call defines a logical intent.
2.  **Diagnostics**: **Tracea Doctor** profiles the environment and available backends.
3.  **Planning**: The Doctor selects the most performant **Kernel Variant** (e.g., CUDA Tensor Core vs. CPU SIMD).
4.  **Optimization**: `AutoTuner` (if active) searches for the best `PipelineConfig` for the chosen variant.
5.  **Emission**: The corresponding `Emitter` generates the kernel source or selects a precompiled binary.
6.  **Launch**: The `RuntimeManager` executes the kernel with zero-copy buffer management.
