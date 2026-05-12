
import torch
import tracea
import torch.nn.functional as F
import math

def test_flash_attention():
    print("--- Testing FlashAttention-2 Numerical Correctness ---")
    
    # Parameters
    B, H, S, D = 1, 8, 128, 64  # Small seq for initial debug
    scale = 1.0 / math.sqrt(D)
    
    # Inputs
    q = torch.randn((B, H, S, D), dtype=torch.float16, device='cuda')
    k = torch.randn((B, H, S, D), dtype=torch.float16, device='cuda')
    v = torch.randn((B, H, S, D), dtype=torch.float16, device='cuda')
    
    # Reference
    # Note: PyTorch expectes (B, H, S, D) for SDPA
    ref_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    
    # Tracea
    ctx = tracea.Context("RTX 3070")
    
    # Wrap pointers
    d_Q = tracea.PyDeviceBufferU16.unsafe_from_ptr(q.data_ptr(), q.numel(), ctx)
    d_K = tracea.PyDeviceBufferU16.unsafe_from_ptr(k.data_ptr(), k.numel(), ctx)
    d_V = tracea.PyDeviceBufferU16.unsafe_from_ptr(v.data_ptr(), v.numel(), ctx)
    
    o_out = torch.zeros_like(q)
    d_O = tracea.PyDeviceBufferU16.unsafe_from_ptr(o_out.data_ptr(), o_out.numel(), ctx)
    
    print(f"Launching Attention (S={S}, D={D})...")
    ctx.attention(d_Q, d_K, d_V, d_O, B, H, S, D, D, causal=False, scale_sqrt=True)
    ctx.synchronize()
    
    # Compare
    diff = (o_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Diff: {max_diff}")
    print(f"Mean Diff: {mean_diff}")
    
    if max_diff < 0.1:
        print("SUCCESS! Attention values Match.")
    else:
        print("FAILURE: Large discrepancy detected.")
        print("Reference (first 4 elements of first head):")
        print(ref_out[0, 0, 0, :4])
        print("Tracea (first 4 elements of first head):")
        print(o_out[0, 0, 0, :4])
        exit(1)

if __name__ == "__main__":
    test_flash_attention()
