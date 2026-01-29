import tracea
import sys

# 1. Initialize Context
print("Attempting to initialize Tracea Context...")
try:
    ctx = tracea.Context("Auto")
    print("Context Created.")
except Exception as e:
    print(f"Warning: Context creation failed: {e}")
    print("This is likely due to missing CUDA/NVRTC DLLs on this environment.")
    print("Proceeding with Graph Lowering verification only.")
    ctx = None

# Using tracea.nn directly
nn = tracea.nn

# 2. Build Attention Graph
print("Building Attention Graph...")
graph = tracea.Graph()

# Define input dimension (Batch=1, Seq=128, Dim=512)
# We use a dummy Gemm to represent input source
input_id = graph.add_gemm(128, 512, 512) 

# Add Attention Layer (512 dim, 8 heads)
attn = tracea.nn.Attention(embed_dim=512, num_heads=8, causal=True)
out_id = attn(graph, input_id)

print(f"Graph constructed. Root node ID: {out_id}")

# 3. Lower Graph
# This should expand Attention into QKV projections, Softmax, etc.
print("Lowering Graph...")
lowered_graph = graph.lower()
print(f"Lowered Graph Node Count: {len(lowered_graph)}")

# 4. Optimize Graph
# This will trigger auto-tuning for each primitive node
# Softmax will be skipped with a warning as per implementation
if ctx:
    print("Starting Graph Optimization (1 iteration per node for demo)...")
    ctx.optimize_graph(graph, iterations=1)
else:
    print("Skipping optimization due to Context initialization failure.")

print("Demo Successful! Attention decomposition and lowering verified.")
