
import torch
import tracea
import ctypes

def pack_int4(tensor):
    # tensor: [K, N] int32 (vals 0..15)
    # Pack 8 rows into one int32 (along K)
    # Output: [K/8, N] int32
    K, N = tensor.shape
    assert K % 8 == 0
    packed = torch.zeros((K // 8, N), dtype=torch.int32)
    for i in range(8):
        # bitmask 0..15
        vals = tensor[i::8, :] & 0xF
        packed |= (vals << (i * 4))
    return packed

def debug_int4_layout():
    M, N, K = 32, 16, 16
    print(f"--- Debugging Int4 Layout ({M}x{N}x{K}) ---")

    # A: Ones (Float16)
    # A = torch.ones((M, K), dtype=torch.float16, device='cuda')
    # A = torch.zeros((M, K), dtype=torch.float16, device='cuda')
    # Use pattern to identify rows?
    # For layout debug, use Ones or Diagonal.
    # Let's use Ones to verify accumulation.
    a_host = torch.ones((M, K), dtype=torch.float16)
    # a_host[0, :] = 1.0 
    
    # B: Identity-like (Int4)
    # Row 0: 1, 0, 0...
    # Row 1: 0, 1, 0...
    # ...
    # But packed.
    # Let's simply make B = 1 everywhere for now to verify column presence?
    # No, we want to distinguish columns.
    # B[:, 0] = 1, B[:, 1] = 2 ...
    b_host = torch.zeros((K, N), dtype=torch.int32)
    for c in range(N):
        b_host[:, c] = c % 16 # Col 0=0, Col 1=1...

    # Let's use Identity logic for K compatibility?
    # Just all 1s is easiest to check "Zero vs Non-Zero".
    b_host.fill_(1) 
    
    # Pack B
    b_packed = pack_int4(b_host).cuda()
    # b_packed shape: [2, 16] (Since K=16 / 8 = 2 packs).

    a_uint16 = a_host.view(torch.uint16).cuda()
    
    # C: Output
    c_out = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    
    # Pointers
    print(f"C Ptr (Python): {hex(c_out.data_ptr())}")

    ctx = tracea.Context("RTX 3070")

    # unsafe_from_ptr buffers
    d_A = tracea.PyDeviceBufferU16.unsafe_from_ptr(a_uint16.data_ptr(), a_uint16.numel(), ctx)
    d_B = tracea.PyDeviceBufferI32.unsafe_from_ptr(b_packed.data_ptr(), b_packed.numel(), ctx)
    d_C = tracea.PyDeviceBufferF32.unsafe_from_ptr(c_out.data_ptr(), c_out.numel(), ctx)

    print("Running Kernel...")
    ctx.matmul(d_A, d_B, d_C, M, N, K)
    
    print("Comparing...")
    # Expected: C = A * B.
    # A = 1 (MxK). B = 1 (KxN).
    # C = K (16).
    expected_val = 16.0
    
    # Or if we used range?
    # Let's assume B=1. output should be 16 everywhere.
    
    c_cpu = c_out.cpu()
    print("Output C (Top Left 8x8):")
    print(c_cpu[:8, :8])
    
    expected = torch.full((8, 8), expected_val)
    print("Expected C (Top Left 8x8):")
    print(expected)
    
    diff = (c_cpu[:8, :8] - expected).abs().max().item()
    if diff > 0.1:
        print(f"FAILURE: Max Diff {diff}")
        print("Difference pattern:")
        print((c_cpu[:8, :8] - expected).abs())
        exit(1)
    else:
        print("SUCCESS! Output Matches.")
    
if __name__ == "__main__":
    debug_int4_layout()
