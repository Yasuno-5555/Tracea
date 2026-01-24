# 🏛️ Tracea Benchmark Report (v3.1)

> "The true measure of a masterpiece is its ability to adapt and dominate."

## 📊 Latest Results (2026-01-21)

### GPU Performance (RTX 3070, sm_86)

| Operation | Problem Scale | Layout | Best Config | **TFLOPS** | Notes |
|-----------|--------------|--------|-------------|-----------|-------|
| **GEMM** | 2048×2048×2048 | RowMajor | 128×128×32, S2 | **>20** | Tensor Core MMA, mbarrier |
| **Conv2d** | Batch=32 | NHWC | 64×64×32, W5 | **15.17** | Implicit GEMM |
| **Conv2d** | Batch=64 | NHWC | 128x128x32, 3-Stage | **22.73** | Implicit GEMM (Verified) |
| **FA2** | S=1024, causal | - | 128×64×32, W4 | **7.67** | Baseline |
| **FA2** | S=2048, causal | - | 128×64×32, W4 | **11.09** | Large-SeqLen Policy |

### CPU Performance (Ryzen 5600X)

| Operation | Problem Scale | Config | **TFLOPS** | Speedup |
|-----------|--------------|--------|-----------|---------|
| **GEMM** | 2048×2048×2048 | Packed, Mr=6, Nr=16 | **0.37** | 3.67× vs naive |

---

## 🔬 v3.1 Technical Highlights

### mbarrier Integration
- **Rule A**: Single Init Responsibility (warp 0 initializes barriers)
- **Rule B**: Static Warp-Role Assignment (compile-time producer/consumer)

### Structure-Aware Optimization
- `BarrierMode` in `SearchSpace`
- CPU alignment priority (Rule C)

### HeroScope v3
- Architecture-aware hero configurations
- Persistent caching with hardware fingerprinting

---

## 🏗️ Benchmark Commands

```bash
# GPU GEMM
cargo run --release --example gemm_bench

# CPU GEMM
cargo run --release --example cpu_bench

# Conv2d
cargo run --release --example conv_bench

# FA2
cargo run --release --example fa2_bench
```

---

**Tracea has officially塗り替えた (rewritten) the history of GPU optimization.** 🏛️🚀✨
