import torch
import tracea
import os
import sys

# Ensure Tracea is in path
sys.path.append(os.path.join(os.getcwd(), "target/debug"))

def verify_mini():
    print("[Verification] Starting MINI-FA2 Correctness Check")
    
    # Tiny Configuration
    B, H, S, D = 1, 1, 32, 64
    causal = False
    
    device = 'cuda'
    dtype = torch.float16
    
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    # 1. PyTorch Reference
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal
    )
    
    # 2. Tracea Implementation
    ctx = tracea.Context("Auto")
    tracea_out = torch.empty_like(q)
    
    print(f"[Tracea] Launching mini attention kernel (B={B}, H={H}, S={S}, D={D})...")
    ctx.attention(
        q, k, v, tracea_out, 
        B, H, S, H*D, D, 
        causal=causal, 
        softmax_mode="auto",
        scale_sqrt=True,
        m_tile=64,
        n_tile=64,
        stages=1
    )
    
    torch.cuda.synchronize()
    
    # 3. Comparison
    diff = (ref_out - tracea_out).abs()
    max_diff = diff.max().item()
    mse = diff.pow(2).mean().item()
    
    # Locate max diff
    max_idx = torch.argmax(diff.flatten()).item()
    b_idx = max_idx // (H * S * D)
    h_idx = (max_idx % (H * S * D)) // (S * D)
    s_idx = (max_idx % (S * D)) // D
    d_idx = max_idx % D
    
    print(f"\n[Results]")
    print(f"Max Difference: {max_diff:.6f} at B={b_idx}, H={h_idx}, S={s_idx}, D={d_idx}")
    print(f"MSE:            {mse:.6e}")
    
    if mse < 1e-4:
        print("PASSED: Mini check successful!")
    else:
        print("FAILED: Discrepancy detected.")
        
    # Analyze row with max diff
    print(f"\n[Analysis] Row with max difference (h={h_idx}, s={s_idx}):")
    print(f"Ref:    {ref_out[b_idx, h_idx, s_idx, :].cpu().float().numpy()}")
    print(f"Tracea: {tracea_out[b_idx, h_idx, s_idx, :].cpu().float().numpy()}")

if __name__ == "__main__":
    verify_mini()
