import torch
import numpy as np
import tracea

def test_minimal():
    B, H, S, D = 1, 1, 16, 16
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    ctx = tracea.Context(arch="86") # RTX 3070
    d_Q = q.clone()
    d_K = k.clone()
    d_V = v.clone()
    d_O = torch.zeros_like(q)
    
    ctx.attention(d_Q, d_K, d_V, d_O, B, H, S, D, D, causal=False, scale_sqrt=True)
    
    # PyTorch Reference
    ref_O = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    
    print(f"Ref SDPA (Row 0): {ref_O[0,0,0,:8]}")
    print(f"Tracea Outputs (Row 0): {d_O[0,0,0,:8]}")
    
    diff = (ref_O - d_O).abs()
    print(f"Max Diff (Full): {diff.max().item()}")

if __name__ == "__main__":
    test_minimal()
