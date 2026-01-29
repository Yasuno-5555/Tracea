import torch
import sys
import os
import glob

sys.path.append(os.getcwd())

try:
    import tracea
    print(f"[Tracea] Imported from {tracea.__file__}")
except ImportError as e:
    print(f"[Tracea] Failed to import tracea: {e}")
    sys.exit(1)

def verify_fa2():
    print("\n--- FA2 Verification (S=16, D=16) ---") # Running small test
    
    device = "cuda"
    
    # Check for PTX dumps before run
    print("Pre-run PTX files:", glob.glob("*.ptx"))
    
    try:
        ctx = tracea.Context("RTX 3070") 
        print("Context created.")
        
        a = torch.randn(128, 128, device=device, dtype=torch.float16)
        b = torch.randn(128, 128, device=device, dtype=torch.float16)
        c = torch.zeros(128, 128, device=device, dtype=torch.float32)
        ctx.matmul(a, b, c, 128, 128, 128)
        ctx.synchronize()
        print("GEMM Passed!")
        
        # Check files again
        print("Post-GEMM PTX files:", glob.glob("*.ptx"))

        B, H, S, D = 1, 4, 16, 16
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, S, H, D, device=device, dtype=dtype)
        k = torch.randn(B, S, H, D, device=device, dtype=dtype)
        v = torch.randn(B, S, H, D, device=device, dtype=dtype)
        o = torch.zeros(B, S, H, D, device=device, dtype=dtype)
        
        print("Running Tracea Attention...")
        ctx.attention(q, k, v, o, B, H, S, H*D, D, causal=True, scale_sqrt=True)
        ctx.synchronize()
        print("Attention Executed!")
        
    except Exception as e:
        print(f"Tracea Execution failed: {e}")
    
    print("Final PTX files:", glob.glob("*.ptx"))

if __name__ == "__main__":
    verify_fa2()
