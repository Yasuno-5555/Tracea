

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
    
    # Baseline: NumPy (CPU)
    # Using float32 for fair comparison with SGEMM
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)
    
    t0 = time.time()
    for _ in range(iterations):
        _ = a_np @ b_np
    dt_np = (time.time() - t0) / iterations
    print(f"NumPy CPU: {dt_np*1000:.2f} ms")

    if TORCH_AVAILABLE:
        # 1. PyTorch CPU
        try:
            a_cpu = torch.from_numpy(a_np)
            b_cpu = torch.from_numpy(b_np)
            
            # Warmup
            _ = a_cpu @ b_cpu
            
            t0 = time.time()
            for _ in range(iterations):
                _ = a_cpu @ b_cpu
            dt_cpu = (time.time() - t0) / iterations
            print(f"PyTorch CPU: {dt_cpu*1000:.2f} ms")

            # 2. PyTorch MPS
            if torch.backends.mps.is_available():
                a_mps = a_cpu.to("mps")
                b_mps = b_cpu.to("mps")
                
                # Warmup
                _ = a_mps @ b_mps
                torch.mps.synchronize()
                
                t0 = time.time()
                for _ in range(iterations):
                    _ = a_mps @ b_mps
                    torch.mps.synchronize()
                dt_mps = (time.time() - t0) / iterations
                print(f"PyTorch MPS: {dt_mps*1000:.2f} ms")
            else:
                print("PyTorch MPS: N/A")
        except Exception as e:
            print(f"PyTorch Benchmark Failed: {e}")

    # 3. Tracea Metal
    if tracea:
        try:
            ctx = tracea.Context()
            
            # Alloc via new API
            buf_a = ctx.alloc_f32(m * k)
            buf_b = ctx.alloc_f32(k * n)
            buf_c = ctx.alloc_f32(m * n)
            
            # Note: We are not copying data for perf benchmark to avoid PCIE overhead measurement
            # Ideally we copy a_np -> buf_a but `PyDeviceBufferF32` needs `copy_from` exposed?
            # For now, running on uninit memory is fine for speed measurement.
            
            # Warmup compilation & launch
            ctx.gemm(buf_a, buf_b, buf_c, m, n, k)
            ctx.synchronize()
            
            t0 = time.time()
            for _ in range(iterations):
                ctx.gemm(buf_a, buf_b, buf_c, m, n, k)
            ctx.synchronize()
            dt_metal = (time.time() - t0) / iterations
            print(f"Tracea Metal: {dt_metal*1000:.2f} ms")
            
        except Exception as e:
            print(f"Tracea Metal: Failed ({e})")
            import traceback
            traceback.print_exc()
    else:
        print("Tracea Metal: Skipped (Module not loaded)")

def benchmark_attention(b, s, h, d, iterations=10):
    print(f"\n--- Benchmarking Attention (B={b}, S={s}, H={h}, D={d}) ---")
    
    # Baseline: NumPy (Calculate FLOPS only, no impl)
    # ops = 4 * b * h * s * s * d
    
    if TORCH_AVAILABLE:
        # 1. PyTorch CPU
        try:
            # Shape: [B, H, S, D]
            q_cpu = torch.randn(b, h, s, d, dtype=torch.float32)
            k_cpu = torch.randn(b, h, s, d, dtype=torch.float32)
            v_cpu = torch.randn(b, h, s, d, dtype=torch.float32)
            
            # Warmup
            _ = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
            
            t0 = time.time()
            for _ in range(iterations):
                _ = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
            dt_cpu = (time.time() - t0) / iterations
            print(f"PyTorch CPU: {dt_cpu*1000:.2f} ms")

            # 2. PyTorch MPS
            if torch.backends.mps.is_available():
                q_mps = q_cpu.to("mps")
                k_mps = k_cpu.to("mps")
                v_mps = v_cpu.to("mps")
                
                # Warmup
                _ = F.scaled_dot_product_attention(q_mps, k_mps, v_mps)
                torch.mps.synchronize()
                
                t0 = time.time()
                for _ in range(iterations):
                    _ = F.scaled_dot_product_attention(q_mps, k_mps, v_mps)
                    torch.mps.synchronize()
                dt_mps = (time.time() - t0) / iterations
                print(f"PyTorch MPS: {dt_mps*1000:.2f} ms")
            else:
                print("PyTorch MPS: N/A")
        except Exception as e:
            print(f"PyTorch Benchmark Failed: {e}")

    # 3. Tracea Metal
    if tracea:
        try:
            ctx = tracea.Context()
            
            # Convert to Half for Metal Input
            q_half = q_cpu.to(torch.float16).numpy()
            k_half = k_cpu.to(torch.float16).numpy()
            v_half = v_cpu.to(torch.float16).numpy()
            
            num_el = b * h * s * d
            
            # Alloc F16 Buffers
            buf_q = ctx.alloc_f16(num_el)
            buf_k = ctx.alloc_f16(num_el)
            buf_v = ctx.alloc_f16(num_el)
            buf_o = ctx.alloc_f16(num_el)
            
            # Initialize Data
            buf_q.copy_from_bytes(q_half.tobytes())
            buf_k.copy_from_bytes(k_half.tobytes())
            buf_v.copy_from_bytes(v_half.tobytes())
            
            # Initialize Output with zeros
            zeros = np.zeros(num_el, dtype=np.float16)
            buf_o.copy_from_bytes(zeros.tobytes())

            # Warmup
            # Note: m_tile=32, n_tile=32 to match 'naive' kernel assumptions if any
            # The current naive kernel assumes BLOCK_N=32. See metal.rs.
            ctx.attention(buf_q, buf_k, buf_v, buf_o, b, h, s, d, d, False, True, 32, 32, 2)
            ctx.synchronize()
            
            # Bench Loop
            t0 = time.time()
            for _ in range(iterations):
                ctx.attention(buf_q, buf_k, buf_v, buf_o, b, h, s, d, d, False, True, 32, 32, 2)
            ctx.synchronize()
            dt_metal = (time.time() - t0) / iterations
            print(f"Tracea Metal: {dt_metal*1000:.2f} ms")
            
            # Correctness Check
            out_bytes = buf_o.to_bytes()
            out_np = np.frombuffer(out_bytes, dtype=np.float16).reshape(b, h, s, d)
            # PyTorch Reference is in FP32. Convert result to FP32 for comparison.
            out_torch = torch.from_numpy(out_np).float()
            
            # Ref is q_ref (FP32 computed)
            # The naive kernel accumulates in FP32 but inputs/outputs in FP16.
            # So it should be close.
            
            # Reference:
            ref_out = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
            
            diff = (out_torch - ref_out).abs()
            print(f"NumPy/MP Check: Max Diff: {diff.max().item():.6f}, Mean Diff: {diff.mean().item():.6f}")
            
        except Exception as e:
            print(f"Tracea Metal: Failed ({e})")
            import traceback
            traceback.print_exc()
    else:
        print("Tracea Metal: Skipped (Module not loaded)")

if __name__ == "__main__":
    # benchmark_attention(1, 1024, 8, 128)
    benchmark_attention(1, 1024, 8, 64) # Use 64 head dim for simple 8x8 matrix fit (D/8 = 8)
    benchmark_gemm(2048, 2048, 2048)
