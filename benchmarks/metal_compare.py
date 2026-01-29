
import time
import sys
import os
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not found or broken. Skipping Torch benchmarks.")
    TORCH_AVAILABLE = False
except OSError:
    print("Warning: PyTorch installation corrupted (OSError). Skipping Torch benchmarks.")
    TORCH_AVAILABLE = False

try:
    import tracea
except ImportError:
    print("Tracea module not found. Please build and ensure tracea.so is in path.")
    tracea = None

def benchmark_gemm(m, n, k, iterations=10):
    print(f"\n--- Benchmarking GEMM ({m}x{n}x{k}) ---")
    
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)
    
    if TORCH_AVAILABLE:
        if torch.backends.mps.is_available():
            a_mps = torch.from_numpy(a_np).to("mps")
            b_mps = torch.from_numpy(b_np).to("mps")
            _ = a_mps @ b_mps
            torch.mps.synchronize()
            t0 = time.time()
            for _ in range(iterations):
                _ = a_mps @ b_mps
                torch.mps.synchronize()
            dt_mps = (time.time() - t0) / iterations
            print(f"PyTorch MPS: {dt_mps*1000:.2f} ms")

    if tracea:
        ctx = tracea.Context()
        buf_a = ctx.alloc_f16(m * k)
        buf_b = ctx.alloc_f16(k * n)
        buf_c = ctx.alloc_f32(m * n)
        
        # Test Variants: 0=Naive, 1=Tiled
        for v in [0, 1]:
            v_name = "Naive" if v == 0 else "Tiled"
            try:
                # Warmup
                # Signature: (a, b, c, m, n, k, m_tile, n_tile, k_tile, epilogue, bias, residual, variant)
                ctx.gemm(buf_a, buf_b, buf_c, m, n, k, None, None, None, None, None, None, v)
                ctx.synchronize()
                
                t0 = time.time()
                for _ in range(iterations):
                    ctx.gemm(buf_a, buf_b, buf_c, m, n, k, None, None, None, None, None, None, v)
                ctx.synchronize()
                dt = (time.time() - t0) / iterations
                print(f"Tracea Metal ({v_name}): {dt*1000:.2f} ms")
            except Exception as e:
                print(f"Tracea Metal ({v_name}): Failed ({e})")

def benchmark_attention(b, s, h, d, iterations=10):
    print(f"\n--- Benchmarking Attention (B={b}, S={s}, H={h}, D={d}) ---")
    
    if TORCH_AVAILABLE:
        if torch.backends.mps.is_available():
            q_mps = torch.randn(b, h, s, d, device="mps", dtype=torch.float16)
            k_mps = torch.randn(b, h, s, d, device="mps", dtype=torch.float16)
            v_mps = torch.randn(b, h, s, d, device="mps", dtype=torch.float16)
            # Warmup in FP32 MPS baseline
            _ = F.scaled_dot_product_attention(q_mps.float(), k_mps.float(), v_mps.float())
            torch.mps.synchronize()
            t0 = time.time()
            for _ in range(iterations):
                _ = F.scaled_dot_product_attention(q_mps.float(), k_mps.float(), v_mps.float())
                torch.mps.synchronize()
            dt_mps = (time.time() - t0) / iterations
            print(f"PyTorch MPS (FP32): {dt_mps*1000:.2f} ms")

    if tracea:
        ctx = tracea.Context()
        num_el = b * h * s * d
        buf_q = ctx.alloc_f16(num_el)
        buf_k = ctx.alloc_f16(num_el)
        buf_v = ctx.alloc_f16(num_el)
        buf_o = ctx.alloc_f16(num_el)
        
        # Variants: 0=Naive, 2=SimdQK, 4=FlashV2
        for v in [0, 2, 4]:
            v_names = {0: "Naive", 2: "SimdQK", 4: "FlashV2"}
            v_name = v_names.get(v, "Unknown")
            try:
                # Warmup
                # Signature: (q, k, v, o, b_in, h_in, s_in, d_in, dh_in, causal, scale_sqrt, m_tile, n_tile, stages, warps, softmax_mode, variant)
                ctx.attention(buf_q, buf_k, buf_v, buf_o, b, h, s, d, d, False, True, None, None, None, None, None, v)
                ctx.synchronize()
                
                t0 = time.time()
                for _ in range(iterations):
                    ctx.attention(buf_q, buf_k, buf_v, buf_o, b, h, s, d, d, False, True, None, None, None, None, None, v)
                ctx.synchronize()
                dt = (time.time() - t0) / iterations
                print(f"Tracea Metal ({v_name}): {dt*1000:.2f} ms")
            except Exception as e:
                print(f"Tracea Metal ({v_name}): Failed ({e})")

if __name__ == "__main__":
    print("=" * 60)
    print("Metal Performance Benchmark Suite")
    print("=" * 60)
    
    # 1. Attention Sweep
    for s in [1024, 2048, 4096]:
        benchmark_attention(1, s, 8, 64)
    
    # 2. GEMM Sweep
    for size in [1024, 2048, 4096]:
        benchmark_gemm(size, size, size)
    
    print("\n" + "=" * 60)
    print("Benchmark Complete")
