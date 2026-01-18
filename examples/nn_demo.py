import tracea
import sys
import os

# Ensure we can import tracea
try:
    import tracea
    print(f"Imported tracea. Version/Dir: {dir(tracea)}")
    if hasattr(tracea, 'nn'):
        print("tracea.nn found in tracea")
    else:
        print("tracea.nn NOT found in tracea")
except ImportError as e:
    print(f"Error: Could not import tracea: {e}")
    sys.exit(1)

# try import tracea.nn
try:
    import tracea.nn
    print("Imported tracea.nn directly")
except ImportError:
    print("Could not import tracea.nn directly, trying to use tracea.nn attribute")
    if not hasattr(tracea, 'nn'):
        sys.exit(1)

print(f"Stats: tracea.nn content: {dir(tracea.nn)}")

# Create Linear Layer
try:
    # Use tracea.Gelu() factory instead of tracea.Epilogue.gelu()
    lin = tracea.nn.Linear(128, 64, bias=True, activation=tracea.Gelu())
    print("Created Linear Layer with Gelu")
except Exception as e:
    print(f"Failed to create Linear: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create Context (Mock or Real)
try:
    # We rely on tracea to detect GPU. If no GPU, it might fail?
    # Context constructor: "Initialize JIT ... allocates ..."
    # If build is successful but no GPU, this might fail.
    # But usually user has GPU.
    ctx = tracea.Context("Auto")
    print("Context Created")
except Exception as e:
    print(f"Context creation failed (might be expected if no GPU): {e}")

# Graph Test
try:
    graph = tracea.Graph()
    
    # Add dummy input node (simulated by a Gemm for now, specific deps needed?)
    # add_gemm(m,n,k) returns id.
    id_in = graph.add_gemm(128, 128, 128)
    print(f"Input Node ID: {id_in}")
    
    # Add Linear Node
    # linear(graph, input_node_id)
    id_lin = lin(graph, id_in)
    print(f"Linear Node ID: {id_lin}")
    
    # Optimization (Should skip NN node as implemented)
    if 'ctx' in locals():
        print("Attempting Optimization...")
        ctx.optimize_graph(graph, iterations=1, goal=tracea.OptimizationGoal.MaximizeTFLOPS)

except Exception as e:
    print(f"Graph operation failed: {e}")
    import traceback
    traceback.print_exc()

print("NN Demo Finished")
