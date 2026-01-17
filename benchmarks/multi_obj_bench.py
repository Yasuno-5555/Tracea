import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tracea

def run_multi_obj_test():
    # GeForce RTX 3070
    ctx = tracea.Context("GeForce RTX 3070")
    
    # Large shape: TFLOPS focus
    print("\n--- Tuning for Large Shape (4096, TFLOPS Focus) ---")
    ctx.auto_tune(4096, 4096, 4096, iterations=5, goal=tracea.OptimizationGoal.MaximizeTFLOPS)
    
    # Small shape: Latency focus
    print("\n--- Tuning for Small Shape (512, Latency Focus) ---")
    ctx.auto_tune(512, 512, 512, iterations=10, goal=tracea.OptimizationGoal.MinimizeLatency)
    
    # Small shape: TFLOPS focus (to compare)
    print("\n--- Tuning for Small Shape (512, TFLOPS Focus) ---")
    ctx.auto_tune(512, 512, 512, iterations=10, goal=tracea.OptimizationGoal.MaximizeTFLOPS)

if __name__ == "__main__":
    run_multi_obj_test()
