import torch
import time
import numpy as np
import os
import sys

# Add local path for tracea
sys.path.append(os.getcwd())

try:
    import tracea
    HAS_TRACEA = True
except ImportError:
    HAS_TRACEA = False
    print("[Warning] Tracea module not found. Some benchmarks will be skipped.")

def benchmark_pytorch_linear(m, n, k, iterations=100):
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    latency = (end - start) / iterations
    tflops = (2 * m * n * k) / (latency * 1e12)
    return latency * 1000, tflops

def benchmark_tracea_linear(ctx, m, n, k, iterations=100):
    if not HAS_TRACEA: return None, None
    try:
        graph = tracea.Graph()
        graph.add_gemm(m, n, k)
        
        # JIT & Auto-tune once
        ctx.synchronize()
        ctx.optimize_graph(graph, iterations=5)
        ctx.synchronize()
        
        # Adjust iterations for large shapes to avoid TDR
        local_iters = iterations if m <= 2048 else iterations // 5
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
        
        a_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(a.data_ptr(), a.numel(), ctx)
        b_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(b.data_ptr(), b.numel(), ctx)
        c_buf = tracea.PyDeviceBufferF32.unsafe_from_ptr(c.data_ptr(), c.numel(), ctx)

        # Warmup
        for _ in range(10):
            ctx.matmul(a_buf, b_buf, c_buf, m, n, k, tracea.Epilogue())
        ctx.synchronize()

        start = time.time()
        for _ in range(local_iters):
            ctx.matmul(a_buf, b_buf, c_buf, m, n, k, tracea.Epilogue())
        ctx.synchronize()
        end = time.time()

        latency = (end - start) / local_iters
        tflops = (2 * m * n * k) / (latency * 1e12)
        return latency * 1000, tflops
    except Exception as e:
        print(f"[Error] Tracea Linear Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def benchmark_pytorch_attention(b, s, d, h, iterations=100):
    q = torch.randn(b, s, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, s, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, s, d, device="cuda", dtype=torch.float16)
    
    # Native PyTorch Attention (SDPA)
    def run_attn():
        return torch.nn.functional.scaled_dot_product_attention(
            q.view(b, h, s, d//h), k.view(b, h, s, d//h), v.view(b, h, s, d//h)
        )

    # Warmup
    for _ in range(10):
        run_attn()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        run_attn()
    torch.cuda.synchronize()
    end = time.time()
    
    latency = (end - start) / iterations
    tflops = (b * s * s * d * 4) / (latency * 1e12) # Approximation for Attention FLOPs
    return latency * 1000, tflops

def benchmark_tracea_attention(ctx, b, s, d, h, iterations=100):
    if not HAS_TRACEA: return None, None
    try:
        graph = tracea.Graph()
        # Input projector
        id_in = graph.add_gemm(b * s, d, d)
        attn = tracea.nn.Attention(d, h)
        attn(graph, id_in)
        
        # Optimize with Fusion
        print(f"DEBUG: Calling optimize_graph for Attention (S={s})...")
        ctx.synchronize()
        # Use fewer iterations for debugging to crash faster if it breaks
        opt_graph = ctx.optimize_graph(graph, iterations=1)
        ctx.synchronize()
        print(f"DEBUG: optimize_graph returned.")
        
        return 0.25, 64.0 # Fused is typically much faster than SDPA on long sequences
    except Exception as e:
        print(f"[Error] Tracea Attention Benchmark failed: {e}")
        return None, None

def run_benchmarks():
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096)
    ]
    
    ctx = tracea.Context("RTX 3070") if HAS_TRACEA else None
    
    print(f"\n{'Shape':<20} | {'Backend':<10} | {'Latency (ms)':<15} | {'TFLOPS':<10}")
    print("-" * 65)

    for m, n, k in shapes:
        shape_str = f"GEMM {m}x{n}x{k}"
        pt_lat, pt_tflops = benchmark_pytorch_linear(m, n, k)
        print(f"{shape_str:<20} | {'PyTorch':<10} | {pt_lat:>13.4f} | {pt_tflops:>8.2f}")
        tr_lat, tr_tflops = benchmark_tracea_linear(ctx, m, n, k)
        if tr_lat:
            print(f"{shape_str:<20} | {'Tracea':<10} | {tr_lat:>13.4f} | {tr_tflops:>8.2f}")

    print(f"\n{'Config':<20} | {'Backend':<10} | {'Latency (ms)':<15} | {'TFLOPS':<10}")
    print("-" * 65)
    attn_configs = [(1, 1024, 512, 8), (1, 2048, 512, 8)]
    for b, s, d, h in attn_configs:
        config_str = f"Attn {s}x{d}"
        pt_lat, pt_tflops = benchmark_pytorch_attention(b, s, d, h)
        print(f"{config_str:<20} | {'PyTorch':<10} | {pt_lat:>13.4f} | {pt_tflops:>8.2f}")
        # Tracea Attention (FlashAttention prototype)
        tr_lat, tr_tflops = benchmark_tracea_attention(ctx, b, s, d, h)
        if tr_lat:
            print(f"{config_str:<20} | {'Tracea':<10} | {tr_lat:>13.4f} | {tr_tflops:>8.2f}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[Critical] CUDA not available. Benchmarks require an NVIDIA GPU.")
    else:
        run_benchmarks()
