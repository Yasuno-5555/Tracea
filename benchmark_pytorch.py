
import torch
import time
import math

def benchmark_conv2d_fused():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📍 PyTorch Round 1: Conv2d + Bias + ReLU (ResNet Block)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # N=64, C=64, H=56, W=56, K=64, R=3, S=3
    N, C, H, W = 64, 64, 56, 56
    K, R, S = 64, 3, 3
    
    # PyTorch uses NCHW by default
    input = torch.randn(N, C, H, W, device='cuda', dtype=torch.float16)
    weight = torch.randn(K, C, R, S, device='cuda', dtype=torch.float16)
    bias = torch.randn(K, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        x = torch.nn.functional.conv2d(input, weight, bias=bias, padding=1)
        x = torch.nn.functional.relu(x)
    torch.cuda.synchronize()
    
    start = time.time()
    iters = 100
    for _ in range(iters):
        x = torch.nn.functional.conv2d(input, weight, bias=bias, padding=1)
        x = torch.nn.functional.relu(x)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters
    flops = 2.0 * N * K * H * W * C * R * S
    tflops = (flops / avg_time) / 1e12
    
    print(f"[PyTorch] Conv2d+Bias+ReLU: {avg_time*1000:.3f} ms | {tflops:.2f} TFLOPS")
    return avg_time, tflops

def benchmark_gemm():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📍 PyTorch Round 3: Pure GEMM (4096^3)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    M, N, K = 4096, 4096, 4096
    
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    start = time.time()
    iters = 50
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters
    flops = 2.0 * M * N * K
    tflops = (flops / avg_time) / 1e12
    
    print(f"[PyTorch] GEMM 4096^3: {avg_time*1000:.3f} ms | {tflops:.2f} TFLOPS")
    return avg_time, tflops

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        exit(1)
        
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    conv_ms, conv_tflops = benchmark_conv2d_fused()
    gemm_ms, gemm_tflops = benchmark_gemm()
    
    print("\n[SUMMARY]")
    print(f"CONV: {conv_tflops:.2f} TFLOPS")
    print(f"GEMM: {gemm_tflops:.2f} TFLOPS")
