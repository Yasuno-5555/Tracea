#include "../../include/tracea.hpp"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Raw CUDA empty kernel
__global__ void empty_raw(int dummy) {}

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main() {
  // 1. Setup
  std::cout << "[Bench] Initializing CUDA..." << std::endl;
  CUDA_CHECK(cudaFree(0)); // Context init

  const int ITERATIONS = 10000;
  const int WARMUP = 100;

  // --- Raw Benchmark ---
  std::cout << "[Bench] Running Raw CUDA..." << std::endl;
  for (int i = 0; i < WARMUP; ++i) {
    empty_raw<<<1, 32>>>(0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  auto start_cpu = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < ITERATIONS; ++i) {
    empty_raw<<<1, 32>>>(0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end_cpu = std::chrono::high_resolution_clock::now();
  double raw_cpu_avg_us =
      std::chrono::duration<double, std::micro>(end_cpu - start_cpu).count() /
      ITERATIONS;

  std::cout << "Raw Latency: " << raw_cpu_avg_us << " us" << std::endl;

  // 2. Tracea Setup
  std::cout << "[Bench] creating Tracea context..." << std::endl;
  tracea::Context ctx(
      "GeForce"); // Device name doesn't matter for this microbench

  // Warmup JIT
  try {
    ctx.compile_empty();
  } catch (const std::exception &e) {
    std::cerr << "Tracea compile failed: " << e.what() << std::endl;
    // Report Raw only
    std::cout << "\n=== Results (Partial) ===" << std::endl;
    std::cout << "Raw CUDA: " << raw_cpu_avg_us << " us" << std::endl;
    std::cout << "Tracea:   FAILED" << std::endl;
    return 1;
  }

  // --- Tracea Benchmark ---
  std::cout << "[Bench] Running Tracea..." << std::endl;
  for (int i = 0; i < WARMUP; ++i) {
    ctx.launch_empty();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  start_cpu = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < ITERATIONS; ++i) {
    ctx.launch_empty();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  end_cpu = std::chrono::high_resolution_clock::now();
  double tracea_cpu_avg_us =
      std::chrono::duration<double, std::micro>(end_cpu - start_cpu).count() /
      ITERATIONS;

  // --- Report ---
  std::cout << "\n=== Results (CPU Latency) ===" << std::endl;
  std::cout << "Raw CUDA: " << raw_cpu_avg_us << " us" << std::endl;
  std::cout << "Tracea:   " << tracea_cpu_avg_us << " us" << std::endl;
  std::cout << "Overhead: " << (tracea_cpu_avg_us - raw_cpu_avg_us) << " us"
            << std::endl;

  return 0;
}
