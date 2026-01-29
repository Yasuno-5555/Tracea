import torch
import sys
import os

# Add path for tracea
release_path = os.path.join(os.path.dirname(__file__), "target", "release")
sys.path.insert(0, release_path)
import tracea

def run_diagnostic():
    print("FA2 Diagnostic: Argument Passing Test")
    # Use distinct numbers to avoid confusion
    print(f"Loading tracea from {release_path}", file=sys.stderr)
    try:
        print(f"Files in release: {os.listdir(release_path)}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to list release: {e}", file=sys.stderr)

    b, h, s, d = 2, 4, 128, 64
    ctx = tracea.Context("sm_86")
    print("Context created successfully.", file=sys.stderr)
    
    print("\n--- Doctor Diagnosis ---", file=sys.stderr)
    try:
        report = ctx.doctor.diagnose()
        print(f"Status: {report.status}", file=sys.stderr)
        print(f"CUDA: {report.cuda_version}", file=sys.stderr)
        print(f"PTXAS: {report.ptxas_version}", file=sys.stderr)
        print(f"Log Dir: .tracea/doctor", file=sys.stderr)
    except Exception as e:
        print(f"Doctor diagnosis failed: {e}", file=sys.stderr)
    print("------------------------\n", file=sys.stderr)
    
    q = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    o = torch.zeros(b, h, s, d, device="cuda", dtype=torch.float16)
    
    # Reference
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, scale=0.125)
    
    t_q = tracea.PyDeviceBufferU16.unsafe_from_ptr(q.data_ptr(), q.numel(), ctx)
    t_k = tracea.PyDeviceBufferU16.unsafe_from_ptr(k.data_ptr(), k.numel(), ctx)
    t_v = tracea.PyDeviceBufferU16.unsafe_from_ptr(v.data_ptr(), v.numel(), ctx)
    t_o = tracea.PyDeviceBufferU16.unsafe_from_ptr(o.data_ptr(), o.numel(), ctx)
    
    # Pass 0.125 as scale
    ctx.attention(t_q, t_k, t_v, t_o, b, h, s, d, d, False, True, m_tile=64, n_tile=64, stages=2)
    ctx.synchronize()
    
    print("\nNumerical Verification:")
    diff = (o - ref_out).abs()
    mse = (diff ** 2).mean().item()
    max_diff = diff.max().item()
    
    print(f"MSE: {mse:.6e}")
    print(f"Max Diff: {max_diff:.6e}")
    
    if mse < 1e-4:
        print("✨ SUCCESS: Numerical Accuracy Verified!")
    else:
        print("❌ FAILURE: Numerical Accuracy Check Failed.")
        print(f"Sample O[0,0,0,:5]: {o[0,0,0,:5]}")
        print(f"Sample Ref[0,0,0,:5]: {ref_out[0,0,0,:5]}")

if __name__ == "__main__":
    run_diagnostic()
