import torch
import ops
import time
import math
import tracea # For exception catching if needed

def benchmark_gemm():
    print("\n=== GEMM Benchmark (M=4096, N=4096, K=4096) ===")
    M, N, K = 4096, 4096, 4096
    
    print(f"Tracea File: {tracea.__file__}")
    
    print("Testing Small GEMM (128x128)...")
    a_s = torch.randn(128, 128, device='cuda', dtype=torch.float16)
    b_s = torch.randn(128, 128, device='cuda', dtype=torch.float16)
    c_s = torch.zeros(128, 128, device='cuda', dtype=torch.float16)
    torch.cuda.synchronize()
    print("Allocated small buffers.")
    
    try:
        ops.gemm(a_s, b_s) 
        torch.cuda.synchronize()
        print("Small GEMM Passed!")
    except Exception as e:
        print(f"Small GEMM Failed: {e}")
        import traceback
        traceback.print_exc()

    print("Allocating Large Buffers...")
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        torch.matmul(a, b)
        ops.gemm(a, b) # High level API
    torch.cuda.synchronize()
    
    # 1. PyTorch
    start = time.time()
    for _ in range(20):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    pt_time = (time.time() - start) / 20.0
    pt_tflops = (2 * M * N * K) / (pt_time * 1e12)
    
    # 2. Tracea
    start = time.time()
    for _ in range(20):
        ops.gemm(a, b)
    torch.cuda.synchronize()
    tr_time = (time.time() - start) / 20.0
    tr_tflops = (2 * M * N * K) / (tr_time * 1e12)
    
    print(f"PyTorch: {pt_time*1000:.2f} ms ({pt_tflops:.2f} TFLOPS)")
    print(f"Tracea : {tr_time*1000:.2f} ms ({tr_tflops:.2f} TFLOPS)")
    print(f"Speedup: {pt_time/tr_time:.2f}x")

def benchmark_conv2d_fused():
    print("\n=== Conv2d Fused Epilogue Benchmark ===")
    N, C, H, W = 32, 64, 56, 56
    K, R, S = 64, 3, 3
    stride, pad = 1, 1
    
    x = torch.randn(N, C, H, W, device='cuda', dtype=torch.float16)
    w = torch.randn(K, C, R, S, device='cuda', dtype=torch.float16) 
    
    
    # Ops.py expects LOGICAL NHWC and KRSC? 
    # Let's permute to match ops.py expectation (which unpacks shape).
    x_tr = x.permute(0, 2, 3, 1).contiguous() # N, H, W, C
    w_tr = w.permute(0, 2, 3, 1).contiguous() # K, R, S, C (assuming PT (K, C, R, S))
    
    # Output shape calc done inside ops.py
    
    bias = torch.randn(K, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        y = torch.nn.functional.conv2d(x, w, bias, stride=stride, padding=pad)
        y = torch.relu(y)
        ops.conv2d(x_tr, w_tr, stride=stride, padding=pad, dilation=1, epilogue="bias+relu", bias=bias)
    torch.cuda.synchronize()
    
    # 1. PyTorch 
    start = time.time()
    for _ in range(20):
        y = torch.nn.functional.conv2d(x, w, bias, stride=stride, padding=pad)
        y = torch.relu(y)
    torch.cuda.synchronize()
    pt_time = (time.time() - start) / 20.0
    
    # 2. Tracea
    start = time.time()
    for _ in range(20):
        ops.conv2d(x_tr, w_tr, stride=stride, padding=pad, dilation=1, epilogue="bias+relu", bias=bias)
    torch.cuda.synchronize()
    tr_time = (time.time() - start) / 20.0
    
    ops_cnt = 2 * N * K * C * R * S * H_out * W_out
    print(f"PyTorch (Unfused): {pt_time*1000:.2f} ms ({ops_cnt/pt_time/1e12:.2f} TFLOPS)")
    print(f"Tracea  (Fused)  : {tr_time*1000:.2f} ms ({ops_cnt/tr_time/1e12:.2f} TFLOPS)")
    print(f"Speedup          : {pt_time/tr_time:.2f}x")

def benchmark_conv_transpose2d():
    print("\n=== ConvTranspose2d Benchmark (VAE Decoder) ===")
    N, Cin, Hin, Win = 1, 64, 32, 32
    Cout = 64
    R, S = 4, 4
    stride, pad = 2, 1
    
    # F32 for now
    x = torch.randn(N, Cin, Hin, Win, device='cuda', dtype=torch.float32)
    w = torch.randn(Cin, Cout, R, S, device='cuda', dtype=torch.float32)
    
    # Ops.py expects logical NHWC
    x_tr = x.permute(0, 2, 3, 1).contiguous()
    w_tr = w.permute(1, 2, 3, 0).contiguous() # Transpose weight is (Cin, Cout, R, S). 
    # Tracea python.rs expects w: (K, R, S, C). Wait.
    # ConvTranspose logic: input C, weight K (out?), R, S?
    # Let's check ops.py for Transpose.
    # ops.py: k, r, s, c_w = w.shape. 
    # If we pass (Cin, R, S, Cout), then k=Cin, r=R, s=S, c_w=Cout.
    # And it calls ctx.conv_transpose2d with k=k (Cin), c=c_w (Cout).
    # python.rs conv_transpose2d: n, c, h, w_in, k. 
    # It maps c -> c (Cin), k -> k (Cout).
    # So if ops.py passes k (which is Cin) as k, and c_w (Cout) as c-something?
    # python.rs args: n, c, h, w_in, k, r, s
    # ops.py call: ctx.conv_transpose2d(..., c, h, w_in, k, r, s)
    # ops.py gets 'c' from x.shape (N, H, W, C) -> C=Cin.
    # It gets 'k' from w.shape (K, R, S, Cw). 
    # If we pass (Cout, R, S, Cin) -> K=Cout, Cw=Cin.
    # Then ops.py passes k=Cout.
    # Correct!
    # So w_tr must be (Cout, R, S, Cin).
    # PT weight is (Cin, Cout, R, S).
    # Correct permute: (1, 2, 3, 0). -> (Cout, R, S, Cin).
    w_tr = w.permute(1, 2, 3, 0).contiguous()

    # Warmup
    try:
        ops.conv_transpose2d(x_tr, w_tr, stride=stride, padding=pad)
    except Exception as e:
        print(f"Tracea Warmup Failed: {e}")
        return

    torch.cuda.synchronize()

    # 1. PyTorch 
    start = time.time()
    for _ in range(20):
        torch.nn.functional.conv_transpose2d(x, w, stride=stride, padding=pad)
    torch.cuda.synchronize()
    pt_time = (time.time() - start) / 20.0
    
    # 2. Tracea 
    start = time.time()
    for _ in range(20):
        ops.conv_transpose2d(x_tr, w_tr, stride=stride, padding=pad)
    torch.cuda.synchronize()
    tr_time = (time.time() - start) / 20.0

    print(f"PyTorch: {pt_time*1000:.2f} ms")
    print(f"Tracea : {tr_time*1000:.2f} ms")
    print(f"Speedup: {pt_time/tr_time:.2f}x")

if __name__ == "__main__":
    benchmark_gemm()
    benchmark_conv2d_fused()
    benchmark_conv_transpose2d()
