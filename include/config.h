// config.h - Configuration structs for GEMM system
#pragma once
#include <stdbool.h>
#include <string.h>  // for strcmp
#include <cuda_runtime.h>  // for dim3
#include "error_analysis.cuh"  // For MatrixType
#include "generate_test_matrix.cuh"  // For DistributionType

// ============================================================================
// COMPILE-TIME CONFIGURATION CONSTANTS
// ============================================================================
// These can be overridden by defining them before including this header

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef TILE_M
#define TILE_M 16
#endif

#ifndef TILE_N
#define TILE_N 16
#endif

#ifndef TILE_K
#define TILE_K 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#ifndef BLOCK_M
#define BLOCK_M 16
#endif

#ifndef BLOCK_N
#define BLOCK_N 16
#endif

// Matrix sizes for benchmarking and testing
#ifndef DEFAULT_SIZES
#define DEFAULT_SIZES {256, 512, 1024, 2048, 4096}
#endif

// ============================================================================
// TYPE DEFINITIONS AND STRUCTS
// ============================================================================

// Kernel type enumeration for systematic evaluation
typedef enum {
    KERNEL_NAIVE,
    KERNEL_TILED,
    KERNEL_TILED_RECT,
    KERNEL_CUBLAS,
    KERNEL_CUBLAS_TENSOR,
    KERNEL_CUTLASS,
    KERNEL_CUTLASS_TENSOR,
    KERNEL_COUNT  // For iteration
} KernelType;

// Numerical metrics enumeration
typedef enum {
    METRIC_ABSOLUTE_ERROR,
    METRIC_RELATIVE_ERROR,
    METRIC_FROBENIUS_NORM,
    METRIC_CONDITION_NUMBER,
    METRIC_FORWARD_ERROR,
    METRIC_BACKWARD_ERROR,
    METRIC_COUNT  // For iteration
} MetricType;

// Matrix type configuration
typedef struct {
    MatrixType matrix_type;      // MatrixType enum value
    const char* name;            // Human-readable name
    bool enabled;                // Whether this type is enabled
    float param1, param2;        // Parameters for generation
} MatrixTypeConfig;

// Kernel type configuration
typedef struct {
    KernelType kernel_type;      // KernelType enum value
    const char* name;            // Human-readable name
    bool enabled;                // Whether this kernel is enabled
} KernelTypeConfig;

// Kernel allocation (grid and block dimensions)
typedef struct {
    int block_x, block_y;    // Thread block dimensions
    int grid_x, grid_y;      // Grid dimensions
    bool use_dynamic_grid;   // Whether to calculate grid size dynamically
} KernelAllocation;

// Tile configuration
typedef struct {
    int tile_size;           // Square tile size
    int tile_m, tile_n;      // Rectangular tile dimensions
    int tile_k;              // K dimension for tiling
} TileConfig;

// Set routines that take user params or defaults
void set_matrix_config(MatrixTypeConfig* config, MatrixType matrix_type, const char* name,
                       bool enabled, float param1, float param2);
void set_kernel_config(KernelTypeConfig* config, KernelType kernel_type, const char* name, bool enabled);
void set_allocation(KernelAllocation* alloc, int block_x, int block_y,
                    int grid_x, int grid_y, bool use_dynamic_grid);
void set_tiles(TileConfig* tiles, int tile_size, int tile_m, int tile_n, int tile_k);

// Template-based kernel dimension computation for compile-time efficiency
template<KernelType kernel_type>
inline void compute_kernel_dimensions_template(int n, dim3* threadsPerBlock, dim3* numBlocks);

// Template specializations for each kernel type
template<>
inline void compute_kernel_dimensions_template<KERNEL_NAIVE>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
inline void compute_kernel_dimensions_template<KERNEL_TILED>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(TILE_SIZE, TILE_SIZE);
    *numBlocks = dim3((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
}

template<>
inline void compute_kernel_dimensions_template<KERNEL_TILED_RECT>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_N, BLOCK_M);
    *numBlocks = dim3((n + TILE_N - 1) / TILE_N, (n + TILE_M - 1) / TILE_M);
}

template<>
inline void compute_kernel_dimensions_template<KERNEL_CUBLAS>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
inline void compute_kernel_dimensions_template<KERNEL_CUBLAS_TENSOR>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
inline void compute_kernel_dimensions_template<KERNEL_CUTLASS>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
inline void compute_kernel_dimensions_template<KERNEL_CUTLASS_TENSOR>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// Runtime dispatch function that calls the appropriate template specialization
inline void compute_kernel_dimensions_dispatch(KernelType kernel_type, int n, dim3* threadsPerBlock, dim3* numBlocks) {
    switch(kernel_type) {
        case KERNEL_NAIVE:
            compute_kernel_dimensions_template<KERNEL_NAIVE>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_TILED:
            compute_kernel_dimensions_template<KERNEL_TILED>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_TILED_RECT:
            compute_kernel_dimensions_template<KERNEL_TILED_RECT>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_CUBLAS:
            compute_kernel_dimensions_template<KERNEL_CUBLAS>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_CUBLAS_TENSOR:
            compute_kernel_dimensions_template<KERNEL_CUBLAS_TENSOR>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_CUTLASS:
            compute_kernel_dimensions_template<KERNEL_CUTLASS>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_CUTLASS_TENSOR:
            compute_kernel_dimensions_template<KERNEL_CUTLASS_TENSOR>(n, threadsPerBlock, numBlocks);
            break;
        default:
            // Default fallback
            *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
            *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            break;
    }
}

// Helper function to convert string name to KernelType enum
inline KernelType string_to_kernel_type(const char* kernel_name) {
    if (strcmp(kernel_name, "naive") == 0) return KERNEL_NAIVE;
    if (strcmp(kernel_name, "tiled") == 0) return KERNEL_TILED;
    if (strcmp(kernel_name, "tiled_rect") == 0) return KERNEL_TILED_RECT;
    if (strcmp(kernel_name, "cublas") == 0) return KERNEL_CUBLAS;
    if (strcmp(kernel_name, "cublas_tensor") == 0) return KERNEL_CUBLAS_TENSOR;
    if (strcmp(kernel_name, "cutlass") == 0) return KERNEL_CUTLASS;
    if (strcmp(kernel_name, "cutlass_tensor") == 0) return KERNEL_CUTLASS_TENSOR;
    return KERNEL_NAIVE; // Default fallback
}

// Efficient string-based wrapper that uses templates internally
inline void compute_dimensions(const char* kernel_name, int n, dim3* threadsPerBlock, dim3* numBlocks) {
    KernelType kernel_type = string_to_kernel_type(kernel_name);
    compute_kernel_dimensions_dispatch(kernel_type, n, threadsPerBlock, numBlocks);
}

// ============================================================================
// GLOBAL CONFIGURATION VARIABLES
// ============================================================================

// Matrix sizes for benchmarking and testing
extern const int SIZES[];
extern const int NUM_SIZES;
