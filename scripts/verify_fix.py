
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tracea

def verify_fix():
    print("--- Verifying Safe API Fix ---")
    try:
        ctx = tracea.Context("GeForce RTX 3070")
    except Exception as e:
        print(f"Failed to create context: {e}")
        # Could be failing due to missing DLL/pyd update, or no GPU.
        # If no GPU, we can't fully test, but we can test API surface if we mock it? 
        # No, Context wants real GPU. Assuming user has GPU as they asked for 'jit' audit.
        return

    print("Context created.")
    
    # Test 1: Valid Matmul with Device Buffers (Default is Tensor Cores -> Requires U16/Half)
    print("Test 1: Valid Matmul...")
    try:
        a = ctx.scratch_a_h
        b = ctx.scratch_b_h
        c = ctx.scratch_c
        print(f"Got buffers: A={a}, B={b}, C={c}")
    except AttributeError:
        print("FAILED: scratch_a/b/c are not exposed as attributes or are missing.")
        return

    # Call matmul
    print("Calling matmul with safe buffers...")
    try:
        ctx.matmul(a, b, c, 1024, 1024, 1024, tracea.Epilogue.empty())
        print("SUCCESS: matmul called successfully.")
    except Exception as e:
        print(f"FAILED: matmul raised exception: {e}")
        return

    # Verify Failure Case: Passing int (Should fail)
    print("Testing Failure Case (Passing int)...")
    try:
        # This simulated the old unsafe API
        fake_ptr = 123456
        ctx.matmul(fake_ptr, fake_ptr, c, 1024, 1024, 1024, tracea.Epilogue.empty())
        print("FAILED: matmul accepted int (Should have rejected it).")
    except TypeError:
        print("SUCCESS: matmul correctly rejected int.")
    except Exception as e:
        print(f"SUCCESS (Partial): matmul rejected int with {type(e).__name__}: {e}")

if __name__ == "__main__":
    verify_fix()
