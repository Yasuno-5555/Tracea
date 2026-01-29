
import torch
import sys
import os
import time

release_path = os.path.join(os.path.dirname(__file__), "..", "target", "release")
sys.path.insert(0, release_path)
import tracea

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

b_sz, h, s, d = 1, 1, 128, 64
ctx = tracea.Context("sm_86")

q = torch.randn(b_sz, h, s, d, device="cuda", dtype=torch.float16)
k = torch.randn(b_sz, h, s, d, device="cuda", dtype=torch.float16)
v = torch.randn(b_sz, h, s, d, device="cuda", dtype=torch.float16)
o = torch.zeros(b_sz, h, s, d, device="cuda", dtype=torch.float16)

t_q = tracea.PyDeviceBufferU16.unsafe_from_ptr(q.data_ptr(), q.numel(), ctx)
t_k = tracea.PyDeviceBufferU16.unsafe_from_ptr(k.data_ptr(), k.numel(), ctx)
t_v = tracea.PyDeviceBufferU16.unsafe_from_ptr(v.data_ptr(), v.numel(), ctx)
t_o = tracea.PyDeviceBufferU16.unsafe_from_ptr(o.data_ptr(), o.numel(), ctx)

print("Running single small FA2...")
try:
    ctx.attention(t_q, t_k, t_v, t_o, b_sz, h, s, d, d, False, True, m_tile=64, n_tile=32, stages=2, warps=4)
    ctx.synchronize()
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
