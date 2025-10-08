// config.cu - Configuration assignment and defaults
#include "../include/config.h"
#include <string.h>  // for strcmp
#include <cuda_runtime.h>  // for dim3

// ============================================================================
// GLOBAL CONFIGURATION VARIABLES
// ============================================================================

// Matrix sizes for benchmarking and testing
const int SIZES[] = DEFAULT_SIZES;
const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
