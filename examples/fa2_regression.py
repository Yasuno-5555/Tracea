import torch
import tracea
import math
import numpy as np

def run_test(B, H, S, D, causal):
    ctx = tracea.Context()
    scale = 1.0 / math.sqrt(D)
    
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    
    # Trace with FA2
    # Note: ensure we use a config that uses 9 warps and 3 stages for MT=128
    # In practice Trace picks from Tuner, here we manually launch via low-level or rely on default
    # For regression, we use the tracea.Context which uses the optimized emitter.
    
    # We can use the graph interface to ensure it hits the specialized FA2 op
    # This is a bit complex without the full high-level, so we'll use a direct kernel test for strict regression.
    
    # Since we want to check the EMITTER, we use ctx.compile_custom with the emitter's logic
    # But even better, we just test the end-to-end Tracea API if it's ready.
    # Currently we rely on JIT verify script for kernel-level.
    
    # Let's use the Verify logic but with a loop over sizes.
    pass

def matrix_test():
    configs = [
        (1, 1, 512, 64),
        (1, 8, 1024, 64),
        (1, 16, 2048, 64),
        (1, 32, 4096, 64),
    ]
    
    for B, H, S, D in configs:
        print(f"\n--- Testing B={B}, H={H}, S={S}, D={D} ---")
        # Reuse verification script logic for now to ensure strict kernel regression
        # I will update the fa2_verify to be a library or just copy-paste for simplicity in this artifact
        pass

if __name__ == "__main__":
    # For now, let's just make sure jit_fa2_verify can handle different sizes easily.
    # I will modify it to accept arguments.
    print("Regression suite initialized. See jit_fa2_verify.py updates.")
