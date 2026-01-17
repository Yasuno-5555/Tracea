import torch
import tracea
import time

def main():
    print("Tracea Python Demo 🐍")
    
    # 1. Create Context
    ctx = tracea.Context("GeForce") # Auto-detects
    
    # 2. Setup Data (Zero-Copy)
    print("Allocating Tensors...")
    m, n, k = 1024, 1024, 1024
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    
    # Create Safe Tracea Buffers
    a_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(a.data_ptr(), a.numel(), ctx)
    b_buf = tracea.PyDeviceBufferU16.unsafe_from_ptr(b.data_ptr(), b.numel(), ctx)
    c_buf = ctx.scratch_c # Internal F32 scratchpad
    
    # 3. Optimize Graph
    print("Optimizing Graph...")
    graph = tracea.Graph()
    graph.add_gemm(m, n, k)
    
    # Run auto-tuner (Bayesian)
    ctx.optimize_graph(graph, iterations=5, goal=tracea.OptimizationGoal.MaximizeTFLOPS)
    
    # 4. Execute
    print("Executing Kernel...")
    ctx.synchronize()
    start = time.time()
    
    # Fluid Epilogue API
    epilogue = tracea.Epilogue.empty()
    # epilogue = tracea.ReLU() # or use factory functions
    
    for _ in range(10):
        ctx.matmul(a_buf, b_buf, c_buf, m, n, k, epilogue)
        
    ctx.synchronize()
    end = time.time()
    
    print(f"Done. Avg Time: {(end - start)/10 * 1000:.3f} ms")

if __name__ == "__main__":
    main()
