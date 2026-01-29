import torch
import tracea
import os
import sys

# Ensure Tracea is in path
sys.path.append(os.path.join(os.getcwd(), "target/debug"))

def verify_fa2():
    print("[Verification] Starting FA2 Correctness Check (Phase 2)")
    
    # Configuration
    B, H, S, D = 1, 8, 128, 64
    causal = False
    
    device = 'cuda'
    dtype = torch.float16
    
    # 1. Ground Truth (PyTorch Native)
    # Using specific seed for reproducibility
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    # PyTorch Reference
    # Note: scaled_dot_product_attention applies 1/sqrt(d) scaling
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal
    )
    
    # 2. Tracea Implementation
    ctx = tracea.Context("Auto")
    
    # High-level API usage
    graph = tracea.Graph()
    # In Tracea high-level API, Attention usually takes (B, S, D) and handles H internally?
    # Or matches the node signature. 
    # Let's use the low-level ctx.attention for direct comparison first to isolate kernel bugs.
    
    tracea_out = torch.empty_like(q)
    
    # Sig: (q, k, v, o, b, h, s, d, dh, causal, scale_sqrt, ...)
    # python.rs: attention(q, k, v, o, B, H, S, D*H, D, causal, ...)
    # Wait, in the emitter: D_VAL is HeadDim (64).
    # d_model is B*H*S*D? No, d_model in launch logs was 512.
    # d_model = H * D = 8 * 64 = 512.
    
    print(f"[Tracea] Launching low-level attention kernel...")
    ctx.attention(
        q, k, v, tracea_out, 
        B, H, S, H*D, D, 
        causal=causal, 
        softmax_mode="auto",
        scale_sqrt=True
    )
    
    torch.cuda.synchronize()
    
    # 3. Comparison
    diff = (ref_out - tracea_out).abs()
    max_diff = diff.max().item()
    mse = diff.pow(2).mean().item()
    
    print(f"\n[Results]")
    print(f"Max Difference: {max_diff:.6f}")
    print(f"MSE:            {mse:.6e}")
    
    # Precision thresholds
    if mse < 1e-4:
        print("PASSED: Tracea matches PyTorch implementation (MSE < 1e-4)")
    else:
        print("FAILED: Difference is too large!")
        if max_diff > 0.1:
            print("CAUTION: Large discrepancy detected. Check Scaling or Softmax rescaling.")

if __name__ == "__main__":
    verify_fa2()
