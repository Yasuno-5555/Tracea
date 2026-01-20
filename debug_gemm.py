
import tracea
import torch
import time

def verify_gemm(m, n, k, m_tile=128, n_tile=128, k_tile=32):
    print(f"\n[Tracea] Verifying GEMM M={m}, N={n}, K={k} Tile={m_tile}x{n_tile}x{k_tile}")
    
    ctx = tracea.Context()
    
    # Initialize tensors
    a = torch.randn((m, k), device='cuda', dtype=torch.float16)
    b = torch.randn((k, n), device='cuda', dtype=torch.float16) # Tracea expects KxN for B? Or NxK? 
    # Standard GEMM A(MxK) * B(KxN) = C(MxN).
    # Assuming RowMajor B for now.
    c = torch.zeros((m, n), device='cuda', dtype=torch.float16)
    
    # Run Tracea
    try:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        # Note: B should be passed as KxN (RowMajor) or Transposed? 
        # UniversalEmitter normally expects ColMajor B? Or RowMajor?
        # Let's assume RowMajor for now.
        ctx.gemm(a, b, c, m, n, k, m_tile, n_tile, k_tile)
        end_evt.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_evt.elapsed_time(end_evt)
        tflops = (2 * m * n * k) / (elapsed_ms * 1e-3) / 1e12
        print(f"[Tracea] Execution Time: {elapsed_ms:.3f} ms, TFLOPS: {tflops:.3f}")

        # Verification with PyTorch
        c_ref = torch.mm(a, b)
        
        # Compare
        diff = torch.abs(c - c_ref)
        mae = diff.mean().item()
        max_diff = diff.max().item()
        
        print(f"[Tracea] MAE: {mae:.6f}, Max Diff: {max_diff:.6f}")
        
        if max_diff > 5e-2: # Relaxed tolerance for FP16
            print("❌ GEMM Verification FAILED")
        else:
            print("✅ GEMM Verification PASSED")

    except Exception as e:
        print(f"❌ GEMM Verification CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Startup Tracea GEMM Verification...")
    
    # Small GEMM (Health Check)
    verify_gemm(128, 128, 32, m_tile=16, n_tile=16, k_tile=16)
    
    # Larger GEMM (Performance Check)
    # verify_gemm(1024, 1024, 1024)
