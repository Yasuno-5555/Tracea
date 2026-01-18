import sys
import os

# Ensure we can import tracea from the local build
sys.path.append(os.getcwd())

try:
    import tracea
    # Access submodules directly from tracea
    print("[Tracea] Successfully imported tracea")
except ImportError as e:
    print(f"[Tracea] Failed to import tracea: {e}")
    sys.exit(1)

def demo_fusion():
    print("\n--- Tracea Advanced Graph Fusion (FlashAttention) Demo ---")
    
    # 1. Create a Context and Graph
    try:
        ctx = tracea.Context()
        print("[Tracea] Context initialized")
    except Exception as e:
        print(f"[Tracea] Could not initialize Context (expected if CUDA is missing): {e}")
        ctx = None

    graph = tracea.Graph()
    
    # 2. Add an Attention Layer
    # Attention(embed_dim=512, num_heads=8)
    attn = tracea.nn.Attention(512, 8)
    
    # Input node
    id_in = graph.add_gemm(128, 512, 512, [])
    
    # Apply Attention
    id_out = attn(graph, id_in)
    
    print(f"[Tracea] Original Graph Node Count: {len(graph)}")
    
    # 3. Optimize the Graph (Lowering + Fusion)
    # verify via graph.lower().optimize_fusion() even without Context
    lowered = graph.lower()
    print(f"[Tracea] Lowered Graph Node Count: {len(lowered)}")
    # Expected: 1 (Input) + 6 (Attention Decomposed) = 7 nodes
    
    fused = lowered.optimize_fusion()
    print(f"[Tracea] Fused Graph Node Count: {len(fused)}")
    # Expected: 1 (Input) + 3 (Projections) + 1 (FusedAttention) = 5 nodes
    
    if len(fused) == 5:
        print("[Tracea] ✅ Fusion Verified: 3 nodes (QK + Softmax + Out) collapsed into 1 FusedAttention node.")
    else:
        print(f"[Tracea] ❌ Fusion mismatch: Expected 5 nodes, got {len(fused)}")

    if ctx:
        # Also test the context-based optimization path
        opt_graph = ctx.optimize_graph(graph)
        print(f"[Tracea] Context Optimized Graph Node Count: {len(opt_graph)}")

if __name__ == "__main__":
    demo_fusion()
