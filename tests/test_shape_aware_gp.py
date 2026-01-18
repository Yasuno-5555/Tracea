import tracea
import torch
import time

def run_test(m, n, k):
    print(f"\n--- Testing Shape: {m}x{n}x{k} ---")
    ctx = tracea.Context("RTX 3070")
    
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    
    # First call - might trigger tuning
    start = time.time()
    ctx.matmul(a, b, c, m, n, k)
    torch.cuda.synchronize()
    end = time.time()
    print(f"First Call Latency: {(end - start)*1000:.2f} ms")
    
    # Second call - should be fast (cache hit)
    start = time.time()
    ctx.matmul(a, b, c, m, n, k)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Second Call Latency: {(end - start)*1000:.2f} ms")

if __name__ == "__main__":
    # Test Shape 1
    run_test(1024, 1024, 1024)
    # Test Shape 2 (Different)
    run_test(2048, 2048, 2048)
    # Test Shape 3 (Scale up)
    run_test(4096, 4096, 4096)
