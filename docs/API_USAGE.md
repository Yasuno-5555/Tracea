# Tracea API Usage Guide

## Python API

### Installation
```bash
cd tracea-python && maturin develop
```

### Basic Usage
```python
import tracea

# Check version
print(tracea.__version__)  # "0.1.0"

# Access ops module
print(tracea.ops)  # <module 'tracea.ops'>
```

### Conv2d with Automatic Fallback
```python
import torch
from tracea import ops

# Create tensors (FP32, contiguous)
x = torch.randn(1, 64, 224, 224, device='cuda')
w = torch.randn(128, 64, 3, 3, device='cuda')
b = torch.randn(128, device='cuda')

# Call Tracea conv2d
out = ops.conv2d(x, w, b, stride=(1, 1), padding=(1, 1))
```

### Monkey Patch for PyTorch
```python
from tracea import patch_conv2d, TraceaConv2d

# Replace all Conv2d with Tracea backend
patch_conv2d()

# Or use directly
conv = TraceaConv2d(64, 128, kernel_size=3, padding=1)
out = conv(x)
```

### Fallback Conditions
Tracea automatically falls back to PyTorch when:
- `groups != 1`
- `dilation != (1, 1)`
- `dtype != float32`
- Tensor is non-contiguous

---

## C++ API

### Include Header
```cpp
#include "tracea.hpp"
```

### TensorView
```cpp
// Create from raw pointer (GPU)
tracea::TensorView x(
    gpu_ptr,
    {1, 64, 224, 224},  // shape
    tracea::DType::Float32,
    0  // device_id (0 = CUDA:0, -1 = CPU)
);

// Create from CPU data
tracea::TensorView w(
    cpu_ptr,
    {128, 64, 3, 3},
    tracea::DType::Float32,
    -1  // CPU
);
```

### Conv2d
```cpp
tracea::Conv2dParams params;
params.stride_h = 1;
params.stride_w = 1;
params.padding_h = 1;
params.padding_w = 1;
params.stream = my_cuda_stream;  // Optional

try {
    tracea::conv2d(x, w, nullptr, out, params);
} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

### Error Handling
```cpp
// Get last error message
std::string err = tracea::get_last_error();

// Version info
std::string ver = tracea::version();  // "0.1.0"
```

---

## Rust API

### AutoTuner
```rust
use tracea::{AutoTuner, GPUInfo, ProblemDescriptor, Shape, Device, LayerType};

let gpu = GPUInfo::rtx3070();
let tuner = AutoTuner::new(gpu);

let problem = ProblemDescriptor {
    device: Device::Cuda(0),
    layer_type: LayerType::Gemm,
    shape: Shape { m: 2048, n: 2048, k: 2048, batch: 1 },
    ..Default::default()
};

let config = tuner.optimize(&problem);
```

### SearchSpace
```rust
use tracea::core::tuning::SearchSpace;
use tracea::core::config::{SwizzleMode, BarrierMode};

let space = SearchSpace::new()
    .tile_m(&[64, 128])
    .tile_n(&[64, 128])
    .tile_k(&[32])
    .stages(&[2, 3])
    .swizzles(&[SwizzleMode::None, SwizzleMode::Xor4])
    .barrier_modes(&[BarrierMode::None, BarrierMode::ProducerConsumer]);
```

### Policy-Driven Launch
v3.2 introduces the **Policy Engine** for automated tiling and strategy selection.

```rust
use tracea::policy::{PolicyEngine, OperatorTopology, TopologyKind};
use tracea::runtime::{RuntimeManager, DeviceBackend};

let runtime = RuntimeManager::new(DeviceBackend::Cuda(0)).unwrap();

// Describe the operator
let op = OperatorTopology {
    op_id: 101,
    name: "low_rank_mlp".into(),
    op_type: "Gemm".into(),
    m: 4096, n: 4096, k: 512,
    kind: TopologyKind::LowRank { r: 32 },
};

// Generate decision and launch
runtime.launch_with_policy(&[op]).unwrap();
```

### Manual TTG Construction
For custom sparse patterns, you can build a **Topological Tile Graph (TTG)** directly.

```rust
use tracea::runtime::ttg_builder::TTGBuilder;
use tracea::core::ttg::TileMetadata;

let mut builder = TTGBuilder::new(4096, 4096);

// Manually add active tiles
builder.add_tile(TileMetadata {
    region_m: 0, region_n: 0,
    k_start: 0, k_end: 512,
    role: 0, // Main
});

let layout = builder.build_dense(); // or build_sparse()
```

---

## C FFI (tracea.h)

### Types
```c
typedef enum {
    TRACEA_SUCCESS = 0,
    TRACEA_INVALID_PARAMS = 1,
    TRACEA_UNSUPPORTED_CONFIG = 2,
    TRACEA_CUDA_ERROR = 3,
    TRACEA_CPU_ERROR = 4,
} TraceaStatus;

typedef struct {
    void* ptr;
    uint32_t rank;
    const uint64_t* shape;
    const int64_t* stride;
    TraceaDType dtype;
    int32_t device_id;
} TraceaTensorView;
```

### Functions
```c
TraceaStatus tracea_conv2d(
    TraceaTensorView x,
    TraceaTensorView w,
    const TraceaTensorView* b,  // NULL for no bias
    TraceaTensorView* out,
    TraceaConv2dParams params
);

int32_t tracea_get_last_error(uint8_t* buf, size_t len);
void tracea_clear_error(void);
```

---

## Linking

### Python
```bash
pip install maturin
cd tracea-python && maturin develop
```

### C++ (CMake)
```cmake
add_library(tracea_ffi STATIC IMPORTED)
set_target_properties(tracea_ffi PROPERTIES
    IMPORTED_LOCATION "${TRACEA_ROOT}/tracea-ffi/target/release/libtracea_ffi.a"
)
target_include_directories(your_app PRIVATE ${TRACEA_ROOT}/include)
target_link_libraries(your_app tracea_ffi)
```
