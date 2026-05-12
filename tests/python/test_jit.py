import pytest
import tracea
import time

def test_jit_matmul():
    print("\n[TEST] Initializing Context...")
    try:
        ctx = tracea.Context("A100") # Device init
    except RuntimeError as e:
        pytest.skip(f"Skipping test: CUDA Init failed: {e}")
        return

    print(f"[TEST] Scratch Pointers: A=0x{ctx.a_ptr:x}, B=0x{ctx.b_ptr:x}, C=0x{ctx.c_ptr:x}")
    assert ctx.a_ptr != 0
    assert ctx.b_ptr != 0
    assert ctx.c_ptr != 0

    # Dimensions
    m, n, k = 1024, 1024, 1024
    
    # Epilogue
    epilogue = tracea.Epilogue()
    # epilogue = epilogue.relu() # Optional: Test fusion later
    
    print(f"[TEST] Launching JIT Matmul {m}x{n}x{k}...")
    start_time = time.time()
    
    # This will trigger compilation on first run
    ctx.matmul(ctx.a_ptr, ctx.b_ptr, ctx.c_ptr, m, n, k, epilogue)
    
    end_time = time.time()
    print(f"[TEST] First Run (Compile+exec) took: {end_time - start_time:.4f}s")
    
    # Second run (Cache hit)
    start_time = time.time()
    ctx.matmul(ctx.a_ptr, ctx.b_ptr, ctx.c_ptr, m, n, k, epilogue)
    end_time = time.time()
    print(f"[TEST] Second Run (Cache hit) took: {end_time - start_time:.4f}s")
    
    print("[TEST] Success!")

if __name__ == "__main__":
    test_jit_matmul()
