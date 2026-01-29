import tracea
import torch
import sys

def capture_err():
    ctx = tracea.Context("RTX 3070")
    m, n, k = 128, 128, 128
    a = torch.ones(m, k, device="cuda", dtype=torch.float16)
    b = torch.ones(k, n, device="cuda", dtype=torch.float16)
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    
    a_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(a.data_ptr(), a.numel(), ctx)
    b_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(b.data_ptr(), b.numel(), ctx)
    c_buf = tracea.PyDeviceBufferF32.unsafe_from_ptr(c.data_ptr(), c.numel(), ctx)

    try:
        ctx.matmul(a_buf, b_buf, c_buf, m, n, k)
    except Exception as e:
        print("CATCHED ERROR:")
        msg = str(e)
        # NVRTC errors usually contain \n. Let's make sure they are printed.
        print(msg.replace("\\n", "\n"))

if __name__ == "__main__":
    capture_err()
