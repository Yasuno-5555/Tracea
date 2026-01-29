import tracea
import numpy as np
import json
import torch

def test_gemm_graph():
    print("=== Testing Metal Graph Execution ===")
    ctx = tracea.Context()
    
    # dimensions
    M, N, K = 512, 512, 512
    
    # 1. Create input buffers
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    
    a_torch = torch.from_numpy(a_np).to("mps")
    b_torch = torch.from_numpy(b_np).to("mps")
    
    a_ptr = a_torch.data_ptr()
    b_ptr = b_torch.data_ptr()
    
    a_buf = tracea.PyDeviceBufferF32.unsafe_from_ptr(a_ptr, a_np.nbytes, ctx)
    b_buf = tracea.PyDeviceBufferF32.unsafe_from_ptr(b_ptr, b_np.nbytes, ctx)
    
    # 2. Define Graph Topology
    # Structure: operators is a list of OperatorTopology, dependencies is list of [producer_id, consumer_id]
    graph = {
        "operators": [
            { "Input": { "op_id": 10, "name": "input_a" } },
            { "Input": { "op_id": 11, "name": "input_b" } },
            {
                "Gemm": {
                    "op_id": 0,
                    "name": "gemm_0",
                    "m": M, "n": N, "k": K,
                    "kind": "Dense"
                }
            }
        ],
        "dependencies": [
            [10, 0], # Input A -> Gemm
            [11, 0]  # Input B -> Gemm
        ]
    }
    
    # 3. Execute Graph
    input_buffers = {
        10: a_buf,
        11: b_buf
    }
    
    print(f"Executing GEMM graph (M={M}, N={N}, K={K})...")
    output_mapping = ctx.execute_graph(json.dumps(graph), input_buffers, backend="metal")
    print(f"Output Mapping: {output_mapping}")
    
    # 4. Verify Output
    # output_mapping should contain {op_id: buffer_id}
    # Note: dict keys in Python from Rust u64 keys might be strings depending on pyo3 version/setting,
    # but here let's try direct access.
    
    # Handle both string and int keys just in case
    out_buf_id = output_mapping.get(0) or output_mapping.get("0")
    if out_buf_id is None:
        print(f"Error: No output buffer found for op_id 0. Keys: {list(output_mapping.keys())}")
        return

    print(f"Success! Output Buffer ID: {out_buf_id}")

if __name__ == "__main__":
    test_gemm_graph()
