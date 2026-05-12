import torch
import sys
import os
import math

# Ensure we can import tracea from the local build
sys.path.append(os.getcwd())

try:
    import tracea
    print("[Tracea] Successfully imported tracea")
except ImportError as e:
    print(f"[Tracea] Failed to import tracea: {e}")
    sys.exit(1)

def verify_fa2():
    print("\n--- FA2 Verification (S=128, D=64) ---")
    
    # Config
    B, H, S, D = 1, 4, 128, 64
    dtype = torch.float16
    device = "cuda"

    torch.manual_seed(42)
    
    # Create inputs in (B, S, H, D) layout which is often standard for custom kernels (batch, seq, head, head_dim)
    # PyTorch SDPA prefers (B, H, S, D). We will transpose for PyTorch reference.
    
    # Shape: [B, S, H, D]
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    o = torch.zeros(B, S, H, D, device=device, dtype=dtype)
    
    # 1. PyTorch Reference
    # Input needed: (B, H, S, D)
    q_pt = q.transpose(1, 2).contiguous() # (B, H, S, D)
    k_pt = k.transpose(1, 2).contiguous()
    v_pt = v.transpose(1, 2).contiguous()
    
    print("Running PyTorch SDPA (Causal=True)...")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
         res_pt = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=True)
    
    # Result is (B, H, S, D), transpose back to compare with (B, S, H, D)
    res_pt = res_pt.transpose(1, 2).contiguous()
    
    # 2. Tracea Execution
    print("Running Tracea Attention...")
    try:
        ctx = tracea.Context("RTX 3070") # Or just "RTX 4090", generic name
        
        # ctx.attention(q, k, v, o, b, h, s, d, dh, causal, scale_sqrt)
        # Note: Tracea python bindings signature: 
        # attention(self, q, k, v, o, b, h, s, d, dh, causal=False, scale_sqrt=True)
        # It seems "d" in arguments might be embedding dim or head dim?
        # looking at python.rs: `d_in, dh_in`
        # and `op.d = d_in`, `op.dh = dh_in`.
        # Usually D is model dimension (H * Dh) and Dh is head dimension.
        # Let's pass:
        # b=B, h=H, s=S, d=H*D, dh=D
        
        ctx.attention(q, k, v, o, B, H, S, H*D, D, causal=True, scale_sqrt=True)
        ctx.synchronize()
        
    except Exception as e:
        print(f"Tracea Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Compare
    print("\n--- Comparison ---")
    
    # Convert to float for better comparison of error
    diff = (o.float() - res_pt.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Absolute Difference: {max_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    
    # FA2 can have some numerical divergence, but 1e-3 or 1e-2 usually expected.
    if max_diff < 0.05:
        print("✅ Result matches PyTorch!")
    else:
        print("❌ Result mismatch.")
        print("Detailed stats:")
        print(f"Pt Mean: {res_pt.float().mean().item():.4f}, Tracea Mean: {o.float().mean().item():.4f}")
        print(f"Pt Std:  {res_pt.float().std().item():.4f}, Tracea Std:  {o.float().std().item():.4f}")

if __name__ == "__main__":
    verify_fa2()
