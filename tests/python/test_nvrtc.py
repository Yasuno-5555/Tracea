
import sys
import os
sys.path.append(os.getcwd())
import tracea
import time

try:
    ctx = tracea.Context("RTX 3070") # This initializes JIT which might print debug info if I added it? No.
    print(f"Context created.")
    
    # Try valid minimal kernel
    src = """
    extern "C" __global__ void test_kernel(float* out) {
        if (threadIdx.x == 0) out[0] = 123.0f;
    }
    """
    try:
        kernel_id = ctx.runtime.compile(src, "test_kernel")
        print(f"Minimal kernel compiled. ID: {kernel_id}")
    except Exception as e:
        print(f"Minimal kernel FAILED: {e}")

except Exception as e:
    print(f"Global Failure: {e}")
