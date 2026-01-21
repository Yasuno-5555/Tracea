
import tracea
import torch
import sys

print("Loading Tracea...")
ctx = tracea.Context("sm_86")
print("Context Loaded.")

B, H, S, D = 1, 1, 128, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
o = torch.zeros(B, H, S, D, device="cuda", dtype=torch.float16)

t_q = tracea.PyDeviceBufferU16.unsafe_from_ptr(q.data_ptr(), q.numel(), ctx)
t_k = tracea.PyDeviceBufferU16.unsafe_from_ptr(k.data_ptr(), k.numel(), ctx)
t_v = tracea.PyDeviceBufferU16.unsafe_from_ptr(v.data_ptr(), v.numel(), ctx)
t_o = tracea.PyDeviceBufferU16.unsafe_from_ptr(o.data_ptr(), o.numel(), ctx)

print("Launching Kernel (1 Iteration)...")
try:
    # Signature: (q, k, v, o, b, h, s, d, dh, causal, scale_sqrt, m_tile, n_tile, stages)
    ctx.attention(t_q, t_k, t_v, t_o, B, H, S, D, D, False, True, 64, 64, 2)
    ctx.synchronize()
    print("SUCCESS: Kernel Finished.")
except Exception as e:
    print(f"FAILED: {e}")
