
import torch
import tracea
import time

def test_flash_attention():
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available")
        return

    print("Initializing Tracea Context...")
    ctx = tracea.Context("RTX 3070") # Fresh context
    
    B, S, H, D = 1, 1024, 8, 512
    Dh = D // H
    
    print(f"Testing FlashAttention-2 with B={B}, S={S}, H={H}, D={D} (Dh={Dh})...")
    
    # Create input tensors
    q = torch.randn(B, S, H, Dh, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, H, Dh, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, H, Dh, device="cuda", dtype=torch.float32)
    
    # PyTorch Reference
    print("Running PyTorch SDPA...")
    q_pt = q.transpose(1, 2) # (B, H, S, Dh)
    k_pt = k.transpose(1, 2)
    v_pt = v.transpose(1, 2)
    
    ref_out = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=True)
    
    # Tracea Graph
    print("Building Tracea Graph...")
    graph = tracea.Graph()
    
    # Note: Tracea's Python API currently builds graphs from "Gemm" inputs.
    # We need a way to feed Q, K, V directly.
    # Current hack: define 3 dummy Gemms that produce Q, K, V sized outputs?
    # Or rely on internal "test" nodes?
    # As per implementation plan, we added `benchmark_tracea_attention` which adds an Input Projector.
    # That creates Q/K/V from X. Here we have Q, K, V.
    
    # Let's use the Input Projector style but with Identity weights?
    # X (B*S, D) -> Q, K, V.
    # This is complex to setup ref.
    
    # Strategy: Just trigger the COMPILATION for now using the same graph as comprehensive_bench.
    # But ONLY that.
    
    id_in = graph.add_gemm(B * S, D, D) # Input "X"
    attn = tracea.nn.Attention(D, H)
    attn(graph, id_in)
    
    print("Compiling/Optimizing Graph...")
    # This invokes our modified optimize_graph which calls compile_fused_attention
    ctx.optimize_graph(graph, iterations=0) 
    
    print("Optimization finished without crash!")
    
    # TODO: Once we expose a way to inject "Tensor" nodes (Phase NN-6?), we can run data flow.
    # For Phase NN-5, proving the Kernel Compiles and potentially benchmarking the KERNEL via Rust is the goal.

if __name__ == "__main__":
    test_flash_attention()
