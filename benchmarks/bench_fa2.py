
import torch
import sys
import os
import time
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

# Import directly from target/release to ensure fresh build
import sys
import os
release_path = os.path.join(os.path.dirname(__file__), "..", "target", "release")
sys.path.insert(0, release_path)
# Also add deps just in case
sys.path.insert(0, os.path.join(release_path, "deps"))

try:
    import tracea
    # Release mode loaded
except ImportError:
    # Fallback/Retry from root
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import tracea
    print(f"DEBUG: tracea loaded from: {tracea.__file__}")

if hasattr(tracea, 'PyContext') and not hasattr(tracea, 'Context'):
    tracea.Context = tracea.PyContext

print("Starting FA2 Parameter Sweep")

configs = [
    {'name': '64x64, 2S (Baseline)', 'args': {'m_tile': 64, 'n_tile': 64, 'stages': 2}},
    {'name': '32x64, 2S (Low M)',     'args': {'m_tile': 32, 'n_tile': 64, 'stages': 2}},
    {'name': '64x64, 2S (5W)',        'args': {'m_tile': 64, 'n_tile': 64, 'stages': 2}},
    {'name': '128x64, 2S (9W)',       'args': {'m_tile': 128, 'n_tile': 64, 'stages': 2}},
    {'name': '64x64, 3S (Pipe)',      'args': {'m_tile': 64, 'n_tile': 64, 'stages': 3}},
]

try:
    print(f"Running FA2 Benchmark (B=2, H=12, S=8192, D=64)")
    b_sz, h, s, d = 2, 12, 8192, 64
    
    ctx = tracea.Context("sm_86")
    q = torch.randn(b_sz, h, s, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b_sz, h, s, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b_sz, h, s, d, device="cuda", dtype=torch.float16)
    o = torch.zeros(b_sz, h, s, d, device="cuda", dtype=torch.float16)

    t_q = tracea.PyDeviceBufferU16.unsafe_from_ptr(q.data_ptr(), q.numel(), ctx)
    t_k = tracea.PyDeviceBufferU16.unsafe_from_ptr(k.data_ptr(), k.numel(), ctx)
    t_v = tracea.PyDeviceBufferU16.unsafe_from_ptr(v.data_ptr(), v.numel(), ctx)
    t_o = tracea.PyDeviceBufferU16.unsafe_from_ptr(o.data_ptr(), o.numel(), ctx)
    
    results = []

    # Baseline PyTorch
    print("PyTorch Measuring...")
    flops = 4 * b_sz * s * s * h * d
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        # Warmup
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()
        p_avg = (end - start) / 10.0
        p_tflops = flops / (p_avg * 1e12)
        print(f"PyTorch: {p_avg*1000:.2f} ms | {p_tflops:.2f} TFLOPS")

    for cfg in configs:
        print(f"Benchmarking: {cfg['name']}")
        try:
            # Re-init O
            o.zero_()
            
            # Run
            ctx.attention(t_q, t_k, t_v, t_o, b_sz, h, s, d, d, False, True, **cfg['args'])
            ctx.synchronize() # Wait for compile
            
            # Warmup
            ctx.attention(t_q, t_k, t_v, t_o, b_sz, h, s, d, d, False, True, **cfg['args'])
            # Warmup and Get Low-level Handles
            kid = ctx.attention(t_q, t_k, t_v, t_o, b_sz, h, s, d, d, False, True, **cfg['args'])
            grid, block, smem, args = ctx.get_attention_params(t_q, t_k, t_v, t_o, b_sz, h, s, d, d, True, **cfg['args'])
            ctx.synchronize()
            
            start = time.perf_counter()
            for _ in range(100):
                ctx.launch_kernel(kid, grid, block, smem, args)
            ctx.synchronize()
            end = time.perf_counter()
            t_avg = (end - start) / 100.0
            t_tflops = flops / (t_avg * 1e12)
            
            # Check correctness
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            diff = (o - ref).abs()
            mse = (diff ** 2).mean().item()
            
            print(f"  -> {t_tflops:.2f} TFLOPS | MSE: {mse:.6f}")
            results.append({
                'config': cfg['name'],
                'tflops': t_tflops,
                'mse': mse,
                'speedup': t_tflops/p_tflops
            })
            
        except Exception as e:
            print(f"  -> Failed: {e}")
            results.append({
                'config': cfg['name'],
                'tflops': 0.0,
                'mse': -1.0,
                'error': str(e)
            })

    # Log to file
    with open("fa2_perf.txt", "w", encoding="utf-8") as f:
        f.write(f"Tracea FA2 Parameter Sweep Results\n")
        f.write(f"----------------------------------\n")
        f.write(f"PyTorch Baseline: {p_tflops:.2f} TFLOPS\n\n")
        f.write(f"{'Config':<25} | {'TFLOPS':<10} | {'Speedup':<10} | {'MSE':<10}\n")
        f.write("-" * 65 + "\n")
        
        for res in results:
            if res.get('error'):
                f.write(f"{res['config']:<25} | FAILED     | N/A        | {res['error']}\n")
            else:
                f.write(f"{res['config']:<25} | {res['tflops']:<10.4f} | {res['speedup']:<10.4f}x | {res['mse']:<10.6f}\n")

except Exception as e:
    print(f"Error: {e}")
    with open("fa2_perf.txt", "w", encoding="utf-8") as f:
        f.write(f"Error: {e}\n")
