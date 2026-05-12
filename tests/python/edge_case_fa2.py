import torch
import tracea
import time

def verify_fa2(B, H, S, D, causal=False, label=""):
    print(f"\n--- Verifying {label} (B={B}, H={H}, S={S}, D={D}, causal={causal}) ---")
    
    q = torch.randn(B, H, S, D, dtype=torch.half, device="cuda")
    k = torch.randn(B, H, S, D, dtype=torch.half, device="cuda")
    v = torch.randn(B, H, S, D, dtype=torch.half, device="cuda")
    o_tracea = torch.zeros(B, H, S, D, dtype=torch.half, device="cuda")

    # PyTorch Reference
    # is_causal=causal argument in SDPA
    ref_o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # Tracea Execution
    ctx = tracea.Context()
    # Note: Tracea attention takes pointers and dimensions
    # Current API: attention(q, k, v, o, b, h, s, d, dh, causal, scale_sqrt, m_tile, n_tile, stages, warps)
    # Current API: attention(q, k, v, o, b, h, s, d, dh, causal, scale_sqrt, m_tile, n_tile, stages, warps)
    # Use 128x64 for baseline if it's large enough
    mt, nt = (128, 64) if S >= 128 else (16, 16)
    stages = 2 if S >= 128 else 1
    
    # Warmup
    ctx.attention(q, k, v, o_tracea, B, H, S, D, D, causal, True, mt, nt, stages, 1 + mt // 16)
    
    start = time.time()
    iters = 10
    for _ in range(iters):
        ctx.attention(q, k, v, o_tracea, B, H, S, D, D, causal, True, mt, nt, stages, 1 + mt // 16)
    ctx.synchronize()
    end = time.time()
    
    avg_ms = (end - start) * 1000 / iters
    flops = 4 * B * H * S * S * D
    tflops = (flops / 1e12) / (avg_ms / 1000)
    
    print(f"Latency: {avg_ms:.3f} ms, TFLOPS: {tflops:.3f}")

    # Compare
    mse = torch.mean((ref_o - o_tracea)**2).item()
    max_diff = torch.max(torch.abs(ref_o - o_tracea)).item()
    
    print(f"MSE: {mse:.6f}, Max Diff: {max_diff:.6f}")
    if mse < 1e-4:
        print("PASS")
    else:
        print("FAIL")
        # Check if it's just zero
        if torch.all(o_tracea == 0):
            print("ERROR: Output is all zeros!")

if __name__ == "__main__":
    # 1. Powers of 2 (Safe)
    verify_fa2(1, 4, 16, 64, False, "Minimal S=16")
    
    # 2. Non-multiple of TC=64
    verify_fa2(1, 4, 77, 64, False, "Non-multiple S=77")
    verify_fa2(1, 4, 123, 64, False, "Non-multiple S=123")
    
    # 3. Causal + Power of 2
    verify_fa2(1, 4, 128, 64, True, "Causal baseline")
    
    # 4. Causal + Non-multiple
    verify_fa2(1, 4, 77, 64, True, "Causal Non-multiple S=77")
    
    # 5. Large non-multiple
    verify_fa2(1, 4, 1025, 64, False, "Large S=1025")
