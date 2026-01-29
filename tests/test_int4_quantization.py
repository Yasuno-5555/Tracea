
import sys
import os
import torch
import numpy as np

# Add src to path if needed, assuming tracea installed or in pythonpath
# but here we load dynamic lib? 
sys.path.append("target/release") 

try:
    import tracea
except ImportError:
    # Try local import
    print("Trying to import tracea from local build...")
    sys.path.append(".")
    import tracea 


def test_int4_quantization():
    print(f"Tracea File: {tracea.__file__}")
    print(f"Tracea Dir: {dir(tracea)}")
    print("Testing Int4 Weight-Only Quantization...")
    
    # 1. Setup
    M, N, K = 128, 128, 128
    ctx = tracea.Context("RTX 3070") # Or auto-detect
    
    # 2. Create Data
    # A: FP16
    # A - Random Floats, converted to Half representation (uint16)
    a_float = torch.randn(M, K, dtype=torch.float16, device='cuda')
    # View as int16/uint16
    a_uint16 = a_float.view(torch.uint16)
    
    # B - Int4 Weights (0..15)
    # Generate random int4s
    b_int4 = torch.randint(0, 16, (K, N), dtype=torch.int32, device='cuda') # Use int32 for logic
    
    # Pack B into Int32 (K//8, N) 
    b_packed = torch.zeros((K // 8, N), dtype=torch.int32, device='cuda')
    
    for i in range(8):
        # Slice: rows i, i+8, i+16...
        # Packing logic: B_packed gets 8 rows packed into one.
        # Shift and OR
        rows = b_int4[i::8, :] # Extract every 8th row starting at i
        b_packed |= (rows & 0xF) << (i * 4)
        
    # C - Output (Float32)
    c_out = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    
    # 3. Create Trace Buffers
    print("Getting pointers...")
    a_ptr = a_uint16.data_ptr()
    b_ptr = b_packed.data_ptr()
    c_ptr = c_out.data_ptr()
    
    print("Converting pointers to buffer wrappers...")
    try:
        # Create Tracea Buffers
        # A is F16 -> Use U16 buffer (size matches)
        print(f"[Test] Allocating A (U16/F16) numel={a_uint16.numel()}")
        d_A = tracea.PyDeviceBufferU16.unsafe_from_ptr(a_uint16.data_ptr(), a_uint16.numel(), ctx)
        
        # B is Int32 -> Use I32 buffer
        print(f"[Test] Allocating B (I32) numel={b_packed.numel()}")
        d_B = tracea.PyDeviceBufferI32.unsafe_from_ptr(b_packed.data_ptr(), b_packed.numel(), ctx)
        
        # C is F32 -> Use F32 buffer
        print(f"[Test] Allocating C (F32) numel={c_out.numel()}")
        d_C = tracea.PyDeviceBufferF32.unsafe_from_ptr(c_out.data_ptr(), c_out.numel(), ctx)
        
        print("Buffers Created.")
    except Exception as e:
        print(f"Error creating buffer: {e}")
        raise e
    
    # 4. Run Matmul
    print(f"Running Tracea Matmul (M={M}, N={N}, K={K})...")
    import traceback
    try:
        ctx.matmul(d_A, d_B, d_C, M, N, K)
        ctx.synchronize()
        print("Matmul execution finished and synchronized.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    
    # 5. Reference Verification
    print("Computing Reference...")
    # Dequantize B for reference math
    # b_int4 values are 0..15 floats.
    b_float = b_int4.to(torch.float16)
    
    c_ref = torch.matmul(a_float, b_float)
    c_ref = c_ref.to(torch.float32)
    
    # 6. Compare
    print("Comparing...")
    diff = (c_out - c_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Diff: {max_diff}")
    print(f"Mean Diff: {mean_diff}")
    
    if max_diff > 10.0: # Tolerance for half precision + accumulation might be higher with ints
        # With int4 weights 0..15, outputs can be large. 
        # But random A ~ N(0,1).
        print("Verification Failed! Diff too large.")
        sys.exit(1)
    else:
        print("Verification Passed!")

if __name__ == "__main__":
    test_int4_quantization()
