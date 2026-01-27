# Topological Tile Graph (TTG) Design 🏛️

The **Topological Tile Graph (TTG)** is Tracea's unified representation for mapping logical operations (GEMM, Conv2d, Attention) onto physical hardware tiles. It enables high-performance sparse, windowed, and irregular tiling patterns with zero-copy overhead.

## Core Concepts

A TTG consists of two primary data structures that are uploaded to GPU constant or global memory:

### 1. L1 Map (Tile Assignment)
The L1 Map is a 1D or 2D array that maps a **Linear Block ID** (the index of the hardware thread block) to a **Logical Tile ID**.
- For a dense $M \times N$ GEMM with tile sizes $TM, TN$, the L1 Map is usually an identity mapping.
- For sparse or windowed operations, the L1 Map acts as an indirection table, allowing thread blocks to skip "inactive" regions of the computation.

### 2. L2 Table (Tile Metadata)
The L2 Table stores the physical and logical metadata for each unique tile defined in the L1 Map. Each entry is a `TileMetadata` struct:

```rust
#[repr(C)]
pub struct TileMetadata {
    pub region_m: u32,  // Logical coordinate M (starting row)
    pub region_n: u32,  // Logical coordinate N (starting column)
    pub k_start: u32,   // K-dimension start (for split-K or partial tiles)
    pub k_end: u32,     // K-dimension end
    pub role: u32,      // 0=Main, 1=Boundary, 2=Speculative
}
```

## Benefits of TTG

### Zero-Copy Sparsity
Unlike traditional sparse formats (CSR, COO) which often require complex index decoding inside the inner loop, TTG handles sparsity at the **tile level**. The kernel's inner loop remains a high-performance dense GEMM/FMA, while the TTG indirection happens once per thread block setup.

### Flexible Execution Order
By reordering the L1 Map, Tracea can implement various execution strategies without changing the kernel code:
- **Z-order / Hilbert Curve**: Improves L2 cache hit rates by increasing spatial locality.
- **Wavefront**: Enables pipeline synchronization for dependent operations.

### Multi-Role Tiles
The `role` field allows a single kernel to handle both "main" (full) tiles and "boundary" (truncated) tiles by branching once at the start of the kernel, maintaining high warp occupancy for the common case.

## Integration with Policy Engine

The **Policy Engine** analyzes the `ModelTopology` and `DeviceProfile` to produce a `PolicyDecision`. This decision is then "baked" into a `TTGLayout` by the `TTGBuilder`:

1. **Policy**: "Use windowed attention with window size 256 for this 4096x4096 layer."
2. **TTGBuilder**: Generates an L1 Map containing only tiles $(i, j)$ where $|i - j| \times TS < 256$.
3. **Runtime**: Allocates GPU buffers for the L1 Map and L2 Table and launches the kernel.
