
import tracea
import torch
import time

def bench_linear_optimization():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    m, n, k = 4096, 4096, 4096
    print(f"Benchmarking Linear Optimization {m}x{n}x{k}...")
    
    ctx = tracea.Context("RTX 3070")
    
    # Run Auto-Tuner
    graph = tracea.Graph()
    id_a = graph.add_gemm(m, n, k)
    
    print("SKIPPING Auto-Tuning (Using Default 64x64x32 Stage-2)...")
    # ctx.optimize_graph(graph, iterations=5, goal=tracea.OptimizationGoal.MaximizeTFLOPS)
    
    # print(f"Tuning took {time.time() - start:.2f}s")
    
    print("Creating buffers for measurement...")
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float32)

    print("Running Measurement...")
    # Wrap tensors
    a_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(a.data_ptr(), a.numel(), ctx)
    b_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(b.data_ptr(), b.numel(), ctx)
    c_buf = tracea.PyDeviceBufferF32.unsafe_from_ptr(c.data_ptr(), c.numel(), ctx)

    epilogue = tracea.Epilogue()
    # Warmup
    ctx.matmul(a_buf, b_buf, c_buf, m, n, k, epilogue)
    ctx.synchronize()
    print("Verifying correctness...")
    c_ref = torch.matmul(a.float(), b.float()).to(torch.float32)
    ctx.synchronize()
    
    # Check permutations
    diff = (c - c_ref).abs().max().item()
    print(f"Diff (A @ B): {diff:.6f}")
    
    c_ref_t = c_ref.T
    diff_t = (c - c_ref_t).abs().max().item()
    print(f"Diff (A @ B).T: {diff_t:.6f}")
    
    c_ref_bt = torch.matmul(a.float(), b.T.float()).to(torch.float32)
    diff_bt = (c - c_ref_bt).abs().max().item()
    print(f"Diff (A @ B.T): {diff_bt:.6f}")
    
    c_ref_at = torch.matmul(a.T.float(), b.float()).to(torch.float32)
    diff_at = (c - c_ref_at).abs().max().item()
    print(f"Diff (A.T @ B): {diff_at:.6f}")
    
    # Copy c back to host for inspection
    print("Sample C (Top Left 4x4):")
    print(c[:4,:4])
    print("Ref C (Top Left 4x4):")
    print(c_ref[:4,:4])
    
    max_diff = diff
    if max_diff < 1e-2:
        print("✅ Correctness Verified!")
    else:
        print("❌ Numerical Mismatch!")

if __name__ == "__main__":
    bench_linear_optimization()
