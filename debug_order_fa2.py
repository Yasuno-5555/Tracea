import torch
import tracea
import time

def verify_case(B, H, S, D, causal, label, stages, warps, mt=128, nt=64):
    print(f"\n--- Verifying {label} (B={B}, H={H}, S={S}, D={D}, causal={causal}) Stages={stages} Warps={warps} MT={mt} NT={nt} ---")
    q = torch.randn(B, H, S, D, dtype=torch.half, device="cuda")
    k = torch.randn(B, H, S, D, dtype=torch.half, device="cuda")
    v = torch.randn(B, H, S, D, dtype=torch.half, device="cuda")
    o_tracea = torch.zeros(B, H, S, D, dtype=torch.half, device="cuda")
    
    # 1. Warmup
    ctx = tracea.Context()
    ctx.attention(q, k, v, o_tracea, B, H, S, D, D, causal, True, mt, nt, stages, warps)
    ctx.synchronize()
    
    # 2. Ref
    ref_o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    
    # 3. Perf
    start = time.time()
    for _ in range(10):
        ctx.attention(q, k, v, o_tracea, B, H, S, D, D, causal, True, mt, nt, stages, warps)
    ctx.synchronize()
    end = time.time()
    
    mse = torch.mean((ref_o - o_tracea)**2).item()
    print(f"MSE: {mse}")
    
    avg_ms = (end - start) * 1000 / 10
    flops = 4 * B * H * S * S * D
    tflops = (flops / 1e12) / (avg_ms / 1000)
    print(f"Latency: {avg_ms:.3f} ms, TFLOPS: {tflops:.3f}")

if __name__ == "__main__":
    try:
        verify_case(1, 4, 256, 64, False, "S=256 STAGES=2", 2, 9, 128, 64)
    except Exception as e:
        print(f"Caught Exception: {e}")
        import traceback
        traceback.print_exc()
