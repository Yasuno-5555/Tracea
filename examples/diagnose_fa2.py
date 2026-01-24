
import tracea
import torch

def diagnose():
    print("Initializing Tracea Context...")
    try:
        ctx = tracea.Context("Auto")
    except Exception as e:
        print(f"Failed to init context: {e}")
        return

    # Dimensions
    B, S, H, D = 1, 128, 8, 64
    d_model = H * D # 512

    print(f"Config: B={B}, S={S}, H={H}, D={D}, d_model={d_model}")
    print(f"Context Attributes: {dir(ctx)}")

    # Dummy Tensors (using data_ptr via torch)
    q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    o = torch.zeros(B, S, H, D, device='cuda', dtype=torch.float16)

    # Call attention
    # Signature: q, k, v, o, b_in, h_in, s_in, d_in, dh_in, ...
    # Wait, python.rs signature: (..., b_in, h_in, s_in, d_in, dh_in, ...)
    # d_in = model dimension? dh_in = head dimension?
    # Let's pass both explicitly and distinctive values to see which one identifies 'D' in kernel.
    
    print("Calling ctx.attention...")
    try:
        # Note: d_in=512, dh_in=64
        kernel_id = ctx.attention(
            q, k, v, o, 
            B, H, S, d_model, D,
            causal=True,
            scale_sqrt=True,
            m_tile=64, n_tile=64, stages=2, warps=4
        )
        print(f"Kernel ID: {kernel_id}")
        
        print("Synchronizing...")
        ctx.synchronize()
        print("Success!")
        
    except Exception as e:
        print(f"Error during attention call: {e}")

if __name__ == "__main__":
    diagnose()
