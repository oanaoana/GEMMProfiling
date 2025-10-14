// config.h - Configuration structs for GEMM system
#pragma once
#include <stdbool.h>
#include <string.h>  // for strcmp
#include <cuda_runtime.h>  // for dim3
#include <cuda_fp16.h>  // for half precision types
#include <cuda_bf16.h>  // for bfloat16 types
#include <type_traits>  // for template metaprogramming

// ============================================================================
// COMPILE-TIME CONFIGURATION CONSTANTS
// ============================================================================
// These can be overridden by defining them before including this header

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

#ifndef TILE_M
#define TILE_M 32
#endif

#ifndef TILE_N
#define TILE_N 32
#endif

#ifndef TILE_K
#define TILE_K 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef BLOCK_M
#define BLOCK_M 32
#endif

#ifndef BLOCK_N
#define BLOCK_N 32
#endif

// Split-K configuration
#ifndef SPLIT_K_SLICES
#define SPLIT_K_SLICES 16
#endif

// Matrix sizes for benchmarking and testing
#ifndef DEFAULT_SIZES
#define DEFAULT_SIZES {256, 512, 1024, 2048, 4096}
#endif

// Multi-sample analysis configuration
#ifndef DEFAULT_NUM_SAMPLES
#define DEFAULT_NUM_SAMPLES 80
#endif

// Error analysis reproducibility configuration
#ifndef ERROR_REPRODUCIBLE
#define ERROR_REPRODUCIBLE true  // Set to false for fully random behavior
#endif

#ifndef ERROR_SEED
#define ERROR_SEED 42  // Fixed seed for reproducible analysis
#endif

// SVD matrix generation parameters
#ifndef WELL_COND_NUMBER
#define WELL_COND_NUMBER 1.0f
#endif

#ifndef ILL_COND_NUMBER
#define ILL_COND_NUMBER 1e6f
#endif

#ifndef MAX_LEVELS
#define MAX_LEVELS 10
#endif

// ============================================================================
// TYPE DEFINITIONS AND STRUCTS
// ============================================================================

// Kernel type enumeration for systematic evaluation
typedef enum {
    KERNEL_NAIVE = 0,
    KERNEL_TILED,
    KERNEL_TILED_OPT,
    KERNEL_TILED_PAIRWISE,
    KERNEL_TILED_RECT,
    KERNEL_TILED_MIXPREC,
    KERNEL_TILED_PAIRWISE_MIXPREC,
    KERNEL_CUBLAS,
    KERNEL_CUBLAS_TENSOR,
    KERNEL_CUTLASS,
    KERNEL_CUTLASS_TENSOR,
    KERNEL_CUTLASS_SPLITK_FLAT,
    KERNEL_CUTLASS_SPLITK_PAIRWISE,
    KERNEL_HELPER_1D,                    // Add this for error analysis dispatch
    NUM_KERNELS
} KernelType;

// Matrix type enumeration for different matrix generation strategies
typedef enum {
    MATRIX_ODO_WELL_CONDITIONED,
    MATRIX_ODO_ILL_CONDITIONED,
    MATRIX_ZEROMEAN,
    MATRIX_UNIFORM_POSITIVE,
    MATRIX_SCALED_2POWERS,
    MATRIX_RADEMACHER,
    MATRIX_SANITY,          // Original Rademacher (exact ±1) for debugging/verification
    MATRIX_LOGNORMAL,
    MATRIX_FROM_FILE
} MatrixType;

// Distribution type enumeration for matrix generation
typedef enum {
    DIST_UNIFORM,       // Uniform distribution [min, max)
    DIST_NORMAL,        // Normal distribution (mean, std_dev)
    DIST_LOG_NORMAL     // Log-normal distribution
} DistributionType;

// Keep only the template declaration in config.h:
template<KernelType kernel_type>
void compute_kernel_dimensions_template(int n, dim3* threadsPerBlock, dim3* numBlocks);

// ============================================================================
// GLOBAL CONFIGURATION VARIABLES
// ============================================================================

// Matrix sizes for benchmarking and testing
extern const int SIZES[];
extern const int NUM_SIZES;

// ============================================================================
// MIXED PRECISION CONFIGURATION - Simple compile-time selection
// ============================================================================

// Set these to experiment with different precision combinations
#ifndef COMPUTE_TYPE
#define COMPUTE_TYPE float          // Options: float, __half, __nv_bfloat16
#endif

#ifndef ACCUMULATE_TYPE
#define ACCUMULATE_TYPE float       // Options: float, double
#endif

// ============================================================================
// ERROR ANALYSIS CONFIGURATION
// ============================================================================

// Beta factor computation options
#ifndef INCLUDE_CROSS_TERMS
#define INCLUDE_CROSS_TERMS true        // Include O(u²) cross terms in error bounds
#endif

#ifndef COLLAPSE_SAME_PRECISION
#define COLLAPSE_SAME_PRECISION false   // Use condensed form when compute == accumulate precision
#endif

// Error analysis reproducibility
#ifndef ERROR_REPRODUCIBLE
#define ERROR_REPRODUCIBLE true
#endif

#ifndef ERROR_SEED
#define ERROR_SEED 42
#endif

// ============================================================================
// PRECISION AND UNIT ROUNDOFF CONSTANTS - DECLARATIONS
// ============================================================================

// Unit roundoff function declarations
double unit_roundoff_fp32();
double unit_roundoff_fp64();
double unit_roundoff_tf32();
double unit_roundoff_bf16();
double unit_roundoff_fp16();

// Template function declaration only
template<typename T>
double get_unit_roundoff();

// Helper functions for configured types
double unit_roundoff_from_type();
double unit_roundoff_accumulate_type();
