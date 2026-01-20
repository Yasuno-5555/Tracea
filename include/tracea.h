#ifndef TRACEA_H
#define TRACEA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>


/**
 * @brief Tracea Tuning Result
 */
typedef struct {
  bool success;
  char *error_msg;
  float score;
  void *config_ptr; // JSON string (char*) containing the optimal configuration
} TraceaResult;

/**
 * @brief Initialize the Tracea Runtime (CUDA Backend).
 * @return 0 on success, non-zero on failure.
 */
int tracea_init();

/**
 * @brief Shutdown the Tracea Runtime.
 */
void tracea_shutdown();

/**
 * @brief Tune a GEMM Kernel (Matrix Multiplication).
 * @param m M dimension
 * @param n N dimension
 * @param k K dimension
 * @return TraceaResult containing the optimal config JSON. Caller must free
 * config_ptr using tracea_free_string.
 */
TraceaResult tracea_tune_gemm(size_t m, size_t n, size_t k);

/**
 * @brief Tune a FlashAttention-2 Kernel.
 * @param b Batch size
 * @param h Number of heads
 * @param s Sequence length
 * @param d Head dimension
 * @param causal Whether to apply causal mask
 * @return TraceaResult containing the optimal config JSON.
 */
TraceaResult tracea_tune_fa2(size_t b, size_t h, size_t s, size_t d,
                             bool causal);

/**
 * @brief Tune a Conv2d Kernel (Implicit GEMM).
 * @param n Batch size
 * @param c Input channels
 * @param h Input height
 * @param w Input width
 * @param k Output channels (Filters)
 * @param r Kernel height
 * @param s_kernel Kernel width
 * @param stride Stride
 * @param pad Padding
 * @param dilation Dilation
 * @return TraceaResult containing the optimal config JSON.
 */
TraceaResult tracea_tune_conv2d(size_t n, size_t c, size_t h, size_t w,
                                size_t k, size_t r, size_t s_kernel,
                                size_t stride, size_t pad, size_t dilation);

/**
 * @brief Free a string returned by Tracea (e.g., config_ptr or error_msg).
 * @param s Pointer to the string to free.
 */
void tracea_free_string(char *s);

#ifdef __cplusplus
}
#endif

#endif // TRACEA_H
