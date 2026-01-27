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
Tracea follows a strict layered architecture:

- **src/core/**: IR definitions, TTG types, and fundamental shapes.
- **src/semantic/**: Tiling patterns and Semantic IR abstractions.
- **src/policy/**: Policy Engine for strategy planning.
- **src/runtime/**: TTG Builder and Runtime Manager (buffer management).
- **src/kernels/**: Multi-backend kernel implementations (CUDA, HIP, Metal, CPU).
- **src/emitter/**: Code generation logic for various architectures.
- **src/optimizer/**: Bayesian auto-tuner and Hero configurations.
- **src/doctor/**: Environment diagnostics and hardware profiling.

### 3. Development Mode
For Python development, use `maturin`:
```bash
cd tracea-python && maturin develop
```

### 4. Testing
- **Rust Unit Tests**: `cargo test`
- **Benchmarks**: `cargo run --example gemm_bench`
- **Tracea Doctor**: `cargo run --bin tracea-doctor`
