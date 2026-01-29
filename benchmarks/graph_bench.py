import sys
import os
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tracea

def run_graph_test():
    print("\n--- Phase E: Graph-Level Optimization (The Symphony) ---")
    ctx = tracea.Context("GeForce RTX 3070")
    
    # Define a simple graph: 3 GEMMs (Smallest first to test priority)
    graph = tracea.Graph()
    graph.add_gemm(1024, 1024, 1024)
    graph.add_gemm(2048, 2048, 2048)
    graph.add_gemm(4096, 4096, 4096)
    
    start = time.perf_counter()
    ctx.optimize_graph(graph, iterations=5)
    end = time.perf_counter()
    
    print(f"\nGraph Optimized in {end-start:.2f} seconds.")

if __name__ == "__main__":
    run_graph_test()
