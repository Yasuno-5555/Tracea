
import torch
import sys
import os
import time
import pandas as pd
import numpy as np
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import tracea
except ImportError as e:
    print(f"[ERROR] Could not import 'tracea': {e}")
    sys.exit(1)

# Shim for PyContext/Context mismatch if present
if hasattr(tracea, 'PyContext') and not hasattr(tracea, 'Context'):
    tracea.Context = tracea.PyContext
if hasattr(tracea, 'PyEpilogueOp') and not hasattr(tracea, 'Epilogue'):
    tracea.Epilogue = tracea.PyEpilogueOp

class BenchmarkSuite:
    def __init__(self, device_id=0):
        self.device = torch.device(f"cuda:{device_id}")
        self.ctx = tracea.Context("sm_86") # Default to sm_86 for RTX 3070
        self.results = []
        
        # Verify CUDA availability
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available for PyTorch. Benchmarks will be simulated or fail.")
            
        # Warmup Context
        print("[Setup] Warming up Tracea Context...")
        self.ctx.synchronize()

    def _get_tracea_buffer(self, tensor: torch.Tensor):
        """Zero-copy wrap torch tensor into Tracea buffer"""
        ptr = tensor.data_ptr()
        numel = tensor.numel()
        
        if tensor.dtype == torch.float32:
            return tracea.PyDeviceBufferF32.unsafe_from_ptr(ptr, numel, self.ctx)
        elif tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
            # Note: treating BF16 as U16/F16 for buffer wrapping purposes
            return tracea.PyDeviceBufferU16.unsafe_from_ptr(ptr, numel, self.ctx)
        elif tensor.dtype == torch.int32:
            return tracea.PyDeviceBufferI32.unsafe_from_ptr(ptr, numel, self.ctx)
        else:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")

    def run_gemm_benchmark(self):
        print("\n" + "="*60)
        print("[Phase A] GEMM Benchmark (Tracea vs cuBLAS)")
        print("="*60)
        
        sizes = [4096, 8192] #, 16384] # 16384 might OOM on 8GB card (3070)
        dtypes = [
            ("FP16", torch.float16),
            ("FP32", torch.float32),
            # ("BF16", torch.bfloat16) # BF16 logic needs verification
        ]
        
        for name, dtype in dtypes:
            for n in sizes:
                m, k = n, n
                print(f"\n[Bench] {name} {m}x{n}x{k}...")
                
                try:
                    # Allocate Tensors
                    a = torch.randn(m, k, device=self.device, dtype=dtype)
                    b = torch.randn(k, n, device=self.device, dtype=dtype)
                    c = torch.zeros(m, n, device=self.device, dtype=dtype) # Accumulator
                    
                    # 1. PyTorch (cuBLAS) Benchmark
                    # Warmup
                    torch.matmul(a, b, out=c)
                    torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    for _ in range(5):
                        torch.matmul(a, b, out=c)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    cublas_avg = (end - start) / 5.0
                    cublas_tflops = (2 * m * n * k) / (cublas_avg * 1e12)
                    
                    self.results.append({
                        "Phase": "GEMM",
                        "Size": n,
                        "Dtype": name,
                        "Backend": "cuBLAS",
                        "Time(ms)": cublas_avg * 1000,
                        "TFLOPS": cublas_tflops
                    })
                    print(f"  cuBLAS: {cublas_avg*1000:.2f} ms | {cublas_tflops:.2f} TFLOPS")

                    # 2. Tracea Benchmark
                    t_a = self._get_tracea_buffer(a)
                    t_b = self._get_tracea_buffer(b)
                    t_c = self._get_tracea_buffer(c)
                    
                    # Warmup (Compiles kernel if needed)
                    self.ctx.matmul(t_a, t_b, t_c, m, n, k, tracea.Epilogue())
                    self.ctx.synchronize()
                    
                    start = time.perf_counter()
                    for _ in range(5):
                        self.ctx.matmul(t_a, t_b, t_c, m, n, k, tracea.Epilogue())
                    self.ctx.synchronize()
                    end = time.perf_counter()
                    
                    tracea_avg = (end - start) / 5.0
                    tracea_tflops = (2 * m * n * k) / (tracea_avg * 1e12)
                    
                    self.results.append({
                        "Phase": "GEMM",
                        "Size": n,
                        "Dtype": name,
                        "Backend": "Tracea",
                        "Time(ms)": tracea_avg * 1000,
                        "TFLOPS": tracea_tflops
                    })
                    print(f"  Tracea: {tracea_avg*1000:.2f} ms | {tracea_tflops:.2f} TFLOPS")
                    
                    speedup = tracea_tflops / cublas_tflops
                    print(f"  > Speedup: {speedup:.2f}x")
                    
                except RuntimeError as e:
                    print(f"  [Error] {e}")
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()

    def run_fa2_benchmark(self):
        print("\n" + "="*60)
        print("[Phase C] FlashAttention-2 Benchmark")
        print("="*60)
        
        # S = Sequence Length
        sequences = [512, 1024, 2048, 4096]
        b_sz = 2 # Batch size
        h = 12   # Heads
        d = 64   # Head Dim
        dtype = torch.float16
        
        for s in sequences:
            print(f"\n[Bench] FA2 B={b_sz} H={h} S={s} D={d}...")
            
            try:
                q = torch.randn(b_sz, h, s, d, device=self.device, dtype=dtype)
                k = torch.randn(b_sz, h, s, d, device=self.device, dtype=dtype)
                v = torch.randn(b_sz, h, s, d, device=self.device, dtype=dtype)
                o = torch.zeros(b_sz, h, s, d, device=self.device, dtype=dtype)
                
                # 1. PyTorch SDPA (Scaled Dot Product Attention)
                # Ensure we are using Flash Attention backend
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    # Warmup
                    torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    for _ in range(10):
                        torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    
                    torch_avg = (end - start) / 10.0
                    # TFLOPS approx calculation for FA
                    flops = 4 * b_sz * s * s * h * d 
                    torch_tflops = flops / (torch_avg * 1e12)
                    
                    self.results.append({
                        "Phase": "FA2",
                        "Size": s,
                        "Dtype": "FP16",
                        "Backend": "PyTorch(SDPA)",
                        "Time(ms)": torch_avg * 1000,
                        "TFLOPS": torch_tflops
                    })
                    print(f"  PyTorch: {torch_avg*1000:.2f} ms | ~{torch_tflops:.2f} TFLOPS")

                # 2. Tracea FA2
                t_q = self._get_tracea_buffer(q)
                t_k = self._get_tracea_buffer(k)
                t_v = self._get_tracea_buffer(v)
                t_o = self._get_tracea_buffer(o)
                
                # Warmup
                self.ctx.attention(t_q, t_k, t_v, t_o, b_sz, h, s, d, d, False, True)
                self.ctx.synchronize()
                
                start = time.perf_counter()
                for _ in range(10):
                    self.ctx.attention(t_q, t_k, t_v, t_o, b_sz, h, s, d, d, False, True)
                self.ctx.synchronize()
                end = time.perf_counter()
                
                tracea_avg = (end - start) / 10.0
                tracea_tflops = flops / (tracea_avg * 1e12)
                
                self.results.append({
                    "Phase": "FA2",
                    "Size": s,
                    "Dtype": "FP16",
                    "Backend": "Tracea",
                    "Time(ms)": tracea_avg * 1000,
                    "TFLOPS": tracea_tflops
                })
                print(f"  Tracea:  {tracea_avg*1000:.2f} ms | ~{tracea_tflops:.2f} TFLOPS")
                
                speedup = tracea_tflops / torch_tflops
                print(f"  > Speedup: {speedup:.2f}x")

            except RuntimeError as e:
                print(f"  [Error] {e}")

    def save_report(self):
        df = pd.DataFrame(self.results)
        print("\n\n" + "="*60)
        print("[Result] Final Benchmark Report")
        print("="*60)
        print(df)
        df.to_csv("tracea_comprehensive_bench.csv", index=False)
        print("\nReport saved to 'tracea_comprehensive_bench.csv'")

if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_gemm_benchmark()
    suite.run_fa2_benchmark()
    suite.save_report()
