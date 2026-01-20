#ifndef TRACEA_H
#define TRACEA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Opaque struct for configuration result
// Actually we return a simple struct with C-compatible layout.
typedef struct {
    bool success;
    char* error_msg;
    float score;
    void* config_ptr; // Generic pointer to config (serialized string for now)
} TraceaResult;

// Initialize the Tracea Runtime. Returns 0 on success.
int tracea_init();

// Shutdown the Tracea Runtime.
void tracea_shutdown();

// Tune GEMM on CPU.
// Returns a TraceaResult containing the score and a pointer to the JSON config string.
// Important: You must free the config_ptr string using tracea_free_string if success is true.
TraceaResult tracea_tune_gemm(size_t m, size_t n, size_t k);

// Tune FlashAttention-2 on GPU.
// Returns a TraceaResult containing the score and a pointer to the JSON config string.
TraceaResult tracea_tune_fa2(size_t b, size_t h, size_t s, size_t d, bool causal);

// Free the string returned in TraceaResult.config_ptr
void tracea_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif // TRACEA_H
