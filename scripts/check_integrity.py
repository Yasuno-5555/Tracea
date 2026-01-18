
import sys
sys.path.append("target/release")
try:
    import tracea
    print("Imported tracea successfully.")
    print(f"Dir: {dir(tracea)}")
    
    ctx = tracea.Context("RTX 3070")
    print("Context created.")
    
    buf = tracea.PyDeviceBufferF32.unsafe_from_ptr(0, 1024, ctx)
    print("Buffer F32 created (from null ptr, purely for alloc test logic if not checking ptr).")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
