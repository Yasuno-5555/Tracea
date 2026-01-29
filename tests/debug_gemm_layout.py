import torch
import tracea
import time

def debug_layout():
    m, n, k = 4096, 4096, 4096
    print(f"Debug Gemm Layout: {m}x{n}x{k} (Ones Test)")
    import os, hashlib
    fname = tracea.__file__
    print(f"Loaded Tracea from: {fname}")
    print(f"File Size: {os.path.getsize(fname)} bytes")
    print(f"Last Modified: {time.ctime(os.path.getmtime(fname))}")
    with open(fname, "rb") as f:
        print(f"MD5: {hashlib.md5(f.read()).hexdigest()}")
    
    ctx = tracea.Context("RTX 3070")
    
    # 1. Ones Experiment
    # A = 1.0, B = 1.0 -> C should be K (4096.0)
    a = torch.ones(m, k, device="cuda", dtype=torch.float16)
    b = torch.ones(k, n, device="cuda", dtype=torch.float16)
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    
    epilogue = tracea.Epilogue()
    a_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(a.data_ptr(), a.numel(), ctx)
    b_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(b.data_ptr(), b.numel(), ctx)
    c_buf = tracea.PyDeviceBufferF32.unsafe_from_ptr(c.data_ptr(), c.numel(), ctx)

    print("Running Matmul (Ones)...")
    ctx.matmul(a_buf, b_buf, c_buf, m, n, k, epilogue)
    ctx.synchronize()
    
    print("Sample C (Top 4x4):")
    print(c[:4, :4])
    
    mean_val = c.mean().item()
    std_val = c.std().item()
    min_val = c.min().item()
    max_val = c.max().item()
    
    print(f"Stats: Mean={mean_val:.2f}, Std={std_val:.2f}, Min={min_val:.2f}, Max={max_val:.2f}")
    print(f"Expected: 4096.0")
    
    diff = (c - 4096.0).abs().max().item()
    print(f"Max Diff (C - 4096.0): {diff:.6f}")

    # No need for experiment 2
    return

if __name__ == "__main__":
    debug_layout()
