import os
import sys
# Ensure local tracea is loaded
sys.path.insert(0, os.path.dirname(__file__))

import tracea
print(f"[Test] Tracea loaded from: {os.path.abspath(tracea.__file__)}")
import torch

# Ensure output is visible
sys.stdout.reconfigure(encoding='utf-8')

print("[Test] Loading Tracea...")
ctx = tracea.Context("sm_86")
print("[Test] Context Loaded.")

# Benchmarking problem size
B, H, S, D = 1, 1, 128, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
o = torch.zeros(B, H, S, D, device="cuda", dtype=torch.float16)

t_q = tracea.PyDeviceBufferU16.unsafe_from_ptr(q.data_ptr(), q.numel(), ctx)
t_k = tracea.PyDeviceBufferU16.unsafe_from_ptr(k.data_ptr(), k.numel(), ctx)
t_v = tracea.PyDeviceBufferU16.unsafe_from_ptr(v.data_ptr(), v.numel(), ctx)
t_o = tracea.PyDeviceBufferU16.unsafe_from_ptr(o.data_ptr(), o.numel(), ctx)

print("[Test] Starting FlashAttention Tuning...")
# By passing None or minimal tiling, we trigger the tuner IF we have a way to call it from Python.
# Currently py_bindings.rs handles the call. Let's see if we can trigger "Search".
# Python binding for attention takes m_tile, n_tile, stages as Option.
# If they are None, it uses PipelineConfig::new(2, 64, 64, dh_in).
# Wait, let's check py_bindings.rs to see if it calls `tune_kernel`.

try:
    # Explicitly testing the "Search" if possible. 
    # Current py_bindings for attention doesn't call tune_kernel yet, 
    # it just creates a default PipelineConfig.
    # I should update py_bindings.rs to call tune_kernel if parameters are not provided.
    
    print("[Test] Note: python bindings currently use defaults if tiles are missing. Triggering a run...")
    # Signature: q, k, v, o, b, h, s, d, dh, causal, scale_sqrt
    ctx.attention(t_q, t_k, t_v, t_o, B, H, S, D, D, False, True)
    ctx.synchronize()
    print("[Test] Execution successful.")
except Exception as e:
    print(f"[Test] FAILED: {e}")
