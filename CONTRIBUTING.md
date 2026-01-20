# Contributing to Tracea

## Development Workflow

### 1. Building
Tracea uses **Rust** (Cargo) for the core library and **Python** (PyO3) for bindings.

```bash
# Build the Rust library
cargo build --release

# Build and install Python bindings
cargo build --release --features python
copy target\release\tracea.dll tracea.pyd  # Windows
cp target/release/libtracea.so tracea.so   # Linux
```

### 2. Project Structure
The codebase is organized into domain-specific modules:

- **src/kernels/**: Optimized kernel implementations.
    - `attention/`: FlashAttention-2 logic (CUDA, ROCm, Metal).
    - `gemm/`: General Matrix Multiplication.
- **src/bindings/**: Language bindings.
    - `python.rs`: PyO3 exports.
    - `c_bindings.rs`: C ABI exports.
- **src/core/**: High-level IR and graph logic.
- **src/optimizer/**: Bayesian auto-tuner and benchmarks.

### 3. Adding a New Kernel
1.  Create a new adapter in `src/kernels/<op>/<backend>_adapter.rs`.
2.  Implement `TunableKernel` for the adapter.
3.  Expose it in `src/bindings/python.rs`.

### 4. Testing
- **Unit Tests**: `cargo test`
- **Integration Tests**: `python tests/edge_case_fa2.py`
