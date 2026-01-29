
import torch
import sys
import os
import time

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tracea

print("Starting Debug Bench")
try:
    ctx = tracea.Context("sm_86")
    print("Context Created!")
    
    # Alloc
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    c = torch.zeros(1024, 1024, device="cuda", dtype=torch.float32)
    
    t_a = tracea.PyDeviceBufferF32.unsafe_from_ptr(a.data_ptr(), a.numel(), ctx)
    print("Buffer A Created")
    
    ctx.matmul(t_a, t_a, t_a, 1024, 1024, 1024, tracea.Epilogue())
    print("Matmul Dispatched")
    
    ctx.synchronize()
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
