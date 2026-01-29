import torch
import tracea.ops
import time
import math

def benchmark_gemm():
    print("\n=== GEMM Benchmark (M=4096, N=4096, K=4096) ===")
    M, N, K = 4096, 4096, 4096
    
    # Init Data (FP16)
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        torch.matmul(a, b)
        tracea.ops.gemm(a, b, c, M, N, K)
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
        tracea.ops.gemm(a, b, c, M, N, K)
    torch.cuda.synchronize()
    tr_time = (time.time() - start) / 20.0
    tr_tflops = (2 * M * N * K) / (tr_time * 1e12)
    
    print(f"PyTorch: {pt_time*1000:.2f} ms ({pt_tflops:.2f} TFLOPS)")
    print(f"Tracea : {tr_time*1000:.2f} ms ({tr_tflops:.2f} TFLOPS)")
    print(f"Speedup: {pt_time/tr_time:.2f}x")

def benchmark_conv2d_fused():
    print("\n=== Conv2d Fused Epilogue Benchmark ===")
    # ResNet-50 Layer: N=32, C=64, H=56, W=56, K=64, R=3, S=3
    N, C, H, W = 32, 64, 56, 56
    K, R, S = 64, 3, 3
    stride, pad = 1, 1
    
    x = torch.randn(N, C, H, W, device='cuda', dtype=torch.float16)
    w = torch.randn(K, C, R, S, device='cuda', dtype=torch.float16) # Tracea follows PT NCHW layout for kernel? 
    # Wait, Tracea python.rs maps (x, w, o) kernel args directly. 
    # UniversalEmitter usually expects input NHWC, weight KRSC/RSCK?
    # But current Python API (UnfiedOpType::NHWC) expects NHWC?
    # PyTorch is NCHW by default.
    # To be fair, let's use Channels Last (NHWC) for both if PyTorch supports it, or transpose for Tracea?
    # But Tracea's `gemm` used standard pointers.
    # If Tracea config is NHWC, we MUST pass NHWC tensors.
    
    x_nhwc = x.to(memory_format=torch.channels_last)
    w_nhwc = w.to(memory_format=torch.channels_last) 
    # Note: `w` for Conv2d in PT is (Out, In/Groups, kH, kW). 
    # Channels Last on weights usually means KCRS -> KRSC? No, usually not applied to weights in same way.
    # Let's assume Tracea emitter handles what we give it, BUT 
    # layout: NHWC means we iterate C in inner loop.
    # If we pass NCHW data but say NHWC, stride calc is wrong.
    # So we MUST pass NHWC contiguous data.
    
    x_tr = x_nhwc.contiguous()
    w_tr = w_nhwc.contiguous()
    
    # Output shape
    H_out = (H + 2*pad - R) // stride + 1
    W_out = (W + 2*pad - S) // stride + 1
    o_tr = torch.empty(N, K, H_out, W_out, device='cuda', dtype=torch.float16).to(memory_format=torch.channels_last)
    
    bias = torch.randn(K, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        # PT Unfused
        y = torch.nn.functional.conv2d(x, w, bias, stride=stride, padding=pad)
        y = torch.relu(y)
        # Tracea Fused
        tracea.ops.conv2d(x_tr, w_tr, o_tr, N, C, H, W, K, R, S, stride, pad, 1, epilogue="bias+relu", bias=bias)
    torch.cuda.synchronize()
    
    # 1. PyTorch (Unfused: Conv + Add + Relu)
    start = time.time()
    for _ in range(20):
        y = torch.nn.functional.conv2d(x, w, bias, stride=stride, padding=pad)
        y = torch.relu(y)
    torch.cuda.synchronize()
    pt_time = (time.time() - start) / 20.0
    
    # 2. Tracea (Fused)
    start = time.time()
    for _ in range(20):
        tracea.ops.conv2d(x_tr, w_tr, o_tr, N, C, H, W, K, R, S, stride, pad, 1, epilogue="bias+relu", bias=bias)
    torch.cuda.synchronize()
    tr_time = (time.time() - start) / 20.0
    
    ops = 2 * N * K * C * R * S * H_out * W_out
    print(f"PyTorch (Unfused): {pt_time*1000:.2f} ms ({ops/pt_time/1e12:.2f} TFLOPS)")
    print(f"Tracea  (Fused)  : {tr_time*1000:.2f} ms ({ops/tr_time/1e12:.2f} TFLOPS)")
    print(f"Speedup          : {pt_time/tr_time:.2f}x")
    
    # Verify Correctness
    # Note: o_tr is NHWC, PT result 'y' is NCHW.
    error = (o_tr.transpose(1, 3).transpose(1, 2) - y).abs().mean() 
    print(f"Mean Error: {error.item():.6f}")

def benchmark_conv_transpose2d():
    print("\n=== ConvTranspose2d Benchmark (VAE Decoder) ===")
    # SD VAE: 1->64, 32x32 -> 64x64 (Stride 2)
    N, Cin, Hin, Win = 1, 64, 32, 32
    K, R, S = 64, 4, 4 # K is Out Channels? No, for Transpose: In=K, Out=C?
    # PyTorch: In=Cin, Out=Cout. Weight: (Cin, Cout/Groups, kH, kW)
    # Tracea python.rs expected: n, c, h, w_in, k, r, s
    # n=N, c=Cin, k=Cout? 
    # Gemm logic: M = N*H*W, N=K, K=C*R*S...
    # For Transpose, it's weird.
    # Usually Transpose is "Grad Input" of Conv.
    # Let's assume standard definitions match.
    Cout = 64
    stride, pad = 2, 1
    
    x = torch.randn(N, Cin, Hin, Win, device='cuda', dtype=torch.float32) # Using f32 as per plan constraint
    # PyTorch weight: (Cin, Cout, R, S)
    w = torch.randn(Cin, Cout, R, S, device='cuda', dtype=torch.float32) 
    
    # Tracea needs NHWC data? LayoutPolicy::NHWC was set.
    x_tr = x.to(memory_format=torch.channels_last).contiguous()
    w_tr = w.to(memory_format=torch.channels_last).contiguous() 
    # Weight layout for Transpose? 
    
    # Output
    Hout = (Hin - 1) * stride - 2 * pad + R 
    Wout = (Win - 1) * stride - 2 * pad + S
    o_tr = torch.zeros(N, Cout, Hout, Wout, device='cuda', dtype=torch.float32).to(memory_format=torch.channels_last)

    # Warmup
    for _ in range(5):
        torch.nn.functional.conv_transpose2d(x, w, stride=stride, padding=pad)
        try:
             tracea.ops.conv_transpose2d(x_tr, w_tr, o_tr, N, Cin, Hin, Win, Cout, R, S, stride, pad)
        except Exception as e:
            print(f"Tracea Warmup Failed: {e}")
            break
            
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
        try:
            tracea.ops.conv_transpose2d(x_tr, w_tr, o_tr, N, Cin, Hin, Win, Cout, R, S, stride, pad)
        except: pass
    torch.cuda.synchronize()
    tr_time = (time.time() - start) / 20.0

    print(f"PyTorch: {pt_time*1000:.2f} ms")
    print(f"Tracea : {tr_time*1000:.2f} ms")
    print(f"Speedup: {pt_time/tr_time:.2f}x")

if __name__ == "__main__":
    try:
        benchmark_gemm()
        benchmark_conv2d_fused()
        benchmark_conv_transpose2d()
    except Exception as e:
        print(f"Benchmark Failed: {e}")
