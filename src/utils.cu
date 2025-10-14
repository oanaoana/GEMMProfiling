#include "../include/utils.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/gemms.cuh"
#include "../include/benchmark.h"  // For BLOCK_SIZE, TILE_SIZE constants

void fill_matrix(float *mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to print CUDA device properties including theoretical performance
void printDevicePerformanceInfo() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);

    if (err != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("\n===== DEVICE INFORMATION =====\n");
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);

    // Calculate peak memory bandwidth (GB/s)
    // For RTX 4080, memory clock is ~21 GHz
    double memory_clock_rate = prop.memoryClockRate / 1000000.0; // Convert from kHz to GHz
    double bus_width = prop.memoryBusWidth;
    double peak_bandwidth = 2.0 * memory_clock_rate * (bus_width / 8); // GB/s

    // Calculate theoretical peak FLOPS for single precision
    // For RTX 4080, CUDA cores = 9728
    int cuda_cores = prop.multiProcessorCount * 128; // Approximate cores based on SM count
    double gpu_clock_ghz = prop.clockRate / 1000000.0; // Convert kHz to GHz
    double peak_gflops = 2.0 * cuda_cores * gpu_clock_ghz; // 2 ops per cycle with FMA

    printf("Memory clock rate (base estimate): %.1f GHz\n", memory_clock_rate);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("Peak memory bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("CUDA cores (estimate): %d\n", cuda_cores);
    printf("GPU clock: %.3f GHz\n", gpu_clock_ghz);
    printf("Peak performance (FP32): %.2f TFLOP/s\n", peak_gflops / 1000);
    printf("Arithmetic intensity ridge point: %.2f FLOP/byte\n", peak_gflops / peak_bandwidth);
    printf("\n");
}

// Compute reference result in FP64 on GPU using cuBLAS
void compute_C_reference_gpu_fp64(float *h_A, float *h_B, float *h_C_exact, int N) {
    //printf("Computing reference result in FP64 on GPU...\n");

    // Allocate GPU memory for FP64 computation
    size_t size_fp64 = N * N * sizeof(double);

    double *d_A_fp64, *d_B_fp64, *d_C_fp64;
    cudaMalloc(&d_A_fp64, size_fp64);
    cudaMalloc(&d_B_fp64, size_fp64);
    cudaMalloc(&d_C_fp64, size_fp64);

    // Allocate host memory for FP64 data
    double *h_A_fp64 = (double*)malloc(size_fp64);
    double *h_B_fp64 = (double*)malloc(size_fp64);
    double *h_C_fp64 = (double*)malloc(size_fp64);

    // Convert FP32 to FP64 on CPU (fast conversion, GPU GEMM still dominates)
    for (int i = 0; i < N * N; i++) {
        h_A_fp64[i] = (double)h_A[i];
        h_B_fp64[i] = (double)h_B[i];
    }

    // Copy FP64 data to GPU
    cudaMemcpy(d_A_fp64, h_A_fp64, size_fp64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp64, h_B_fp64, size_fp64, cudaMemcpyHostToDevice);

    // Create cuBLAS handle for FP64
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform FP64 GEMM on GPU
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                &alpha, d_B_fp64, N, d_A_fp64, N, &beta, d_C_fp64, N);

    // Copy result back to host
    cudaMemcpy(h_C_fp64, d_C_fp64, size_fp64, cudaMemcpyDeviceToHost);

    // Convert back to FP32 for compatibility
    for (int i = 0; i < N * N; i++) {
        h_C_exact[i] = (float)h_C_fp64[i];
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A_fp64); cudaFree(d_B_fp64); cudaFree(d_C_fp64);
    free(h_A_fp64); free(h_B_fp64); free(h_C_fp64);
}

// Compute reference result in FP64 on CPU
void compute_C_reference(float *A, float *B, float *C_exact, int N) {
    //printf("Computing reference result in FP64 on CPU...\n");

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            // Calculate reference in double precision
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += (double)A[row * N + k] * (double)B[k * N + col];
            }
            // Store result as float
            C_exact[row * N + col] = (float)sum;
        }
    }
}

void verify_result(float *A, float *B, float *C, int N) {
    // Use more appropriate epsilon for float comparison
    float eps = 1e-6;
    float max_rel_error = 0.0f;
    float sum_rel_error = 0.0f;
    int error_count = 0;

    // Allocate memory for CPU FP64 reference
    float *C_exact = (float*)malloc(N * N * sizeof(float));
    compute_C_reference(A, B, C_exact, N);

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            int idx = row * N + col;

            // Compare GPU float result with CPU FP64 reference
            double abs_error = fabs((double)C[idx] - (double)C_exact[idx]);
            double rel_error = abs_error / (fabs((double)C_exact[idx]) > 1e-10 ? fabs((double)C_exact[idx]) : 1e-10);

            // Record statistics
            max_rel_error = fmax(max_rel_error, (float)rel_error);
            sum_rel_error += (float)rel_error;

            // Check error threshold
            if (abs_error > (double)eps) {
                error_count++;
                if (error_count <= 5) { // Limit output to first 5 errors
                    printf("Mismatch at (%d, %d): GPU = %f, CPU_FP64 = %f, Rel Error = %e\n",
                          row, col, C[idx], C_exact[idx], rel_error);
                }
            }
        }
    }

    printf("Max relative error: %e\n", max_rel_error);
    printf("Average relative error: %e\n", sum_rel_error / (N * N));
    printf("Number of elements with error > %e: %d (%.2f%%)\n",
           eps, error_count, 100.0f * error_count / (N * N));

    if (error_count == 0)
        printf("Result verified: correct within epsilon %e.\n", eps);

    // Clean up
    free(C_exact);
}

void printCacheInfo() {
    printf("\n===== DETAILED CACHE ANALYSIS =====\n");

    int value;
    cudaError_t err;

    // L1 Global Cache
    err = cudaDeviceGetAttribute(&value, cudaDevAttrGlobalL1CacheSupported, 0);
    printf("Global L1 Cache Supported: %s\n",
           (err == cudaSuccess) ? (value ? "YES" : "NO") : "UNKNOWN");

    // L1 Local Cache
    err = cudaDeviceGetAttribute(&value, cudaDevAttrLocalL1CacheSupported, 0);
    printf("Local L1 Cache Supported: %s\n",
           (err == cudaSuccess) ? (value ? "YES" : "NO") : "UNKNOWN");

    // L2 Cache Size
    err = cudaDeviceGetAttribute(&value, cudaDevAttrL2CacheSize, 0);
    if (err == cudaSuccess) {
        printf("L2 Cache Size: %d bytes (%.2f MB)\n", value, value / (1024.0 * 1024.0));
    }

    // Cache configuration
    cudaFuncCache cacheConfig;
    err = cudaDeviceGetCacheConfig(&cacheConfig);
    if (err == cudaSuccess) {
        printf("Current cache preference: ");
        switch(cacheConfig) {
            case cudaFuncCachePreferNone: printf("No preference\n"); break;
            case cudaFuncCachePreferShared: printf("Prefer shared memory\n"); break;
            case cudaFuncCachePreferL1: printf("Prefer L1 cache\n"); break;
            case cudaFuncCachePreferEqual: printf("Equal L1/shared\n"); break;
            default: printf("Unknown\n"); break;
        }
    }

    // Memory transaction size (cache line related)
    printf("\nMemory Transaction Analysis:\n");
    printf("Expected cache line size: 128 bytes (32 floats)\n");
    printf("Your warp accesses: 32 consecutive floats\n");
    printf("Theoretical coalescing: PERFECT (1:1 ratio expected)\n");

    printf("=====================================\n\n");
}



// Comparison function for qsort (for percentile calculation)
static int compare_doubles(const void *a, const void *b) {
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;

    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

// Compute comprehensive statistics for an array of doubles
void compute_array_statistics(const double* array, int size, ArrayStats* stats) {
    if (size <= 0 || array == NULL || stats == NULL) {
        return;
    }

    // Copy array for sorting (needed for percentile)
    double* sorted_array = (double*)malloc(size * sizeof(double));
    memcpy(sorted_array, array, size * sizeof(double));

    // Calculate average and find min/max
    double sum = 0.0;
    stats->minimum = array[0];
    stats->maximum = array[0];

    for (int i = 0; i < size; i++) {
        sum += array[i];
        if (array[i] < stats->minimum) stats->minimum = array[i];
        if (array[i] > stats->maximum) stats->maximum = array[i];
    }
    stats->average = sum / size;

    // Calculate standard deviation
    double variance_sum = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = array[i] - stats->average;
        variance_sum += diff * diff;
    }
    stats->std_dev = sqrt(variance_sum / size);

    // Sort array and calculate percentiles
    qsort(sorted_array, size, sizeof(double), compare_doubles);

    // Calculate 10th percentile
    double percentile_index_10 = 0.10 * (size - 1);
    int lower_index_10 = (int)floor(percentile_index_10);
    int upper_index_10 = (int)ceil(percentile_index_10);

    if (lower_index_10 == upper_index_10) {
        stats->p10 = sorted_array[lower_index_10];
    } else {
        // Linear interpolation between the two closest values
        double weight_10 = percentile_index_10 - lower_index_10;
        stats->p10 = sorted_array[lower_index_10] * (1.0 - weight_10) + sorted_array[upper_index_10] * weight_10;
    }

    // Calculate 95th percentile
    double percentile_index = 0.95 * (size - 1);
    int lower_index = (int)floor(percentile_index);
    int upper_index = (int)ceil(percentile_index);

    if (lower_index == upper_index) {
        stats->p95 = sorted_array[lower_index];
    } else {
        // Linear interpolation between the two closest values
        double weight = percentile_index - lower_index;
        stats->p95 = sorted_array[lower_index] * (1.0 - weight) + sorted_array[upper_index] * weight;
    }

    free(sorted_array);
}

// Unified kernel dispatch function that both benchmark and error analysis can use
// Function to map kernel name to KernelType
KernelType getKernelTypeFromName(const char* name) {
    if (strcmp(name, "naive") == 0) return KERNEL_NAIVE;
    if (strcmp(name, "tiled") == 0) return KERNEL_TILED;
    if (strcmp(name, "tiled_opt") == 0) return KERNEL_TILED_OPT;
    if (strcmp(name, "tiled_pairwise") == 0) return KERNEL_TILED_PAIRWISE;
    if (strcmp(name, "tiled_rect") == 0) return KERNEL_TILED_RECT;
    if (strcmp(name, "tiled_mixprec") == 0) return KERNEL_TILED_MIXPREC;
    if (strcmp(name, "tiled_pairwise_mixprec") == 0) return KERNEL_TILED_PAIRWISE_MIXPREC;
    if (strcmp(name, "cublas") == 0) return KERNEL_CUBLAS;
    if (strcmp(name, "cublas_tensor") == 0) return KERNEL_CUBLAS_TENSOR;
    if (strcmp(name, "cutlass") == 0) return KERNEL_CUTLASS;
    if (strcmp(name, "cutlass_tensor") == 0) return KERNEL_CUTLASS_TENSOR;
    if (strcmp(name, "cutlass_splitk_flat") == 0) return KERNEL_CUTLASS_SPLITK_FLAT;
    if (strcmp(name, "cutlass_splitk_pairwise") == 0) return KERNEL_CUTLASS_SPLITK_PAIRWISE;
    return static_cast<KernelType>(-1); // Return invalid value for unknown names
}

// Function to map matrix type name to MatrixType
MatrixType getMatrixTypeFromName(const char* name) {
    if (strcmp(name, "wellcond") == 0) return MATRIX_ODO_WELL_CONDITIONED;
    if (strcmp(name, "illcond") == 0) return MATRIX_ODO_ILL_CONDITIONED;
    if (strcmp(name, "zeromean") == 0) return MATRIX_ZEROMEAN;
    if (strcmp(name, "uniform_positive") == 0) return MATRIX_UNIFORM_POSITIVE;
    if (strcmp(name, "2powers") == 0) return MATRIX_SCALED_2POWERS;
    if (strcmp(name, "rademacher") == 0) return MATRIX_RADEMACHER;
    if (strcmp(name, "sanity") == 0) return MATRIX_SANITY;
    if (strcmp(name, "lognormal") == 0) return MATRIX_LOGNORMAL;
    if (strcmp(name, "file") == 0) return MATRIX_FROM_FILE;
    return static_cast<MatrixType>(-1); // Return invalid value for unknown names
}

// Reverse conversion functions: enum to string
const char* kernelTypeToString(KernelType kernel_type) {
    switch(kernel_type) {
        case KERNEL_NAIVE: return "naive";
        case KERNEL_TILED: return "tiled";
        case KERNEL_TILED_OPT: return "tiled_opt";
        case KERNEL_TILED_PAIRWISE: return "tiled_pairwise";
        case KERNEL_TILED_RECT: return "tiled_rect";
        case KERNEL_TILED_MIXPREC: return "tiled_mixprec";
        case KERNEL_TILED_PAIRWISE_MIXPREC: return "tiled_pairwise_mixprec";  // Add this
        case KERNEL_CUBLAS: return "cublas";
        case KERNEL_CUBLAS_TENSOR: return "cublas_tensor";
        case KERNEL_CUTLASS: return "cutlass";
        case KERNEL_CUTLASS_TENSOR: return "cutlass_tensor";
        case KERNEL_CUTLASS_SPLITK_FLAT: return "cutlass_splitk_flat";
        case KERNEL_CUTLASS_SPLITK_PAIRWISE: return "cutlass_splitk_pairwise";
        default: return "unknown";
    }
}

const char* matrixTypeToString(MatrixType matrix_type) {
    switch(matrix_type) {
        case MATRIX_ODO_WELL_CONDITIONED: return "wellcond";
        case MATRIX_ODO_ILL_CONDITIONED: return "illcond";
        case MATRIX_ZEROMEAN: return "zeromean";
        case MATRIX_UNIFORM_POSITIVE: return "uniform_positive";
        case MATRIX_SCALED_2POWERS: return "2powers";
        case MATRIX_RADEMACHER: return "rademacher";
        case MATRIX_SANITY: return "sanity";
        case MATRIX_LOGNORMAL: return "lognormal";
        case MATRIX_FROM_FILE: return "file";
        default: return "unknown";
    }
}

// Optimized kernel dispatch using function pointer table
typedef void (*KernelFunc)(float*, float*, float*, int, dim3, dim3);

static KernelFunc kernel_function_table[] = {
    launch_naive,                           // KERNEL_NAIVE
    launch_tiled,                          // KERNEL_TILED
    launch_tiled_opt,                      // KERNEL_TILED_OPT
    launch_tiled_pairwise,                 // KERNEL_TILED_PAIRWISE
    launch_tiled_rect,                     // KERNEL_TILED_RECT
    launch_tiled_mixprec,           // KERNEL_TILED_MIXPREC
    launch_tiled_pairwise_mixprec,  // KERNEL_TILED_PAIRWISE_MIXPREC - Add this
    launch_cublas,                         // KERNEL_CUBLAS
    launch_cublas_tensor,                  // KERNEL_CUBLAS_TENSOR
    launch_cutlass,                        // KERNEL_CUTLASS
    launch_cutlass_tensor,                 // KERNEL_CUTLASS_TENSOR
    launch_cutlass_splitk_flat,            // KERNEL_CUTLASS_SPLITK_FLAT
    launch_cutlass_splitk_pairwise         // KERNEL_CUTLASS_SPLITK_PAIRWISE
};



void launch_kernel_by_type(KernelType kernel_type, float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Bounds check for safety
    if (kernel_type < 0 || kernel_type >= sizeof(kernel_function_table)/sizeof(kernel_function_table[0])) {
        printf("ERROR: Invalid kernel type %d\n", (int)kernel_type);
        return;
    }

    // Direct function pointer call - zero overhead dispatch!
    kernel_function_table[kernel_type](d_A, d_B, d_C, n, blocks, threads);
}

// Kernel function pointer lookup utility
void* get_kernel_function_pointer(KernelType kernel_type) {
    switch(kernel_type) {
        case KERNEL_NAIVE:
            return (void*)matmul_naive;
        case KERNEL_TILED:
            return (void*)matmul_tiled;
        case KERNEL_TILED_OPT:
            return (void*)matmul_tiled_opt;
        case KERNEL_TILED_PAIRWISE:
            return (void*)matmul_tiled_pairwise;
        case KERNEL_TILED_RECT:
            return (void*)matmul_tiled_rectangular;

        // Template kernels - get pointer to specific instantiation
        case KERNEL_TILED_MIXPREC:
            // This gets the function pointer for the specific template instantiation
            // that matches your compile-time COMPUTE_TYPE and ACCUMULATE_TYPE
            return (void*)matmul_tiled_mixprec<COMPUTE_TYPE, ACCUMULATE_TYPE>;

        case KERNEL_TILED_PAIRWISE_MIXPREC:
            return (void*)matmul_tiled_pairwise_mixprec<COMPUTE_TYPE, ACCUMULATE_TYPE>;

        // Library kernels - no function pointers available
        case KERNEL_CUBLAS:
        case KERNEL_CUBLAS_TENSOR:
        case KERNEL_CUTLASS:
        case KERNEL_CUTLASS_TENSOR:
        case KERNEL_CUTLASS_SPLITK_FLAT:
        case KERNEL_CUTLASS_SPLITK_PAIRWISE:
            return nullptr;

        default:
            return nullptr;
    }
}

// Template-based kernel dimension computation for compile-time efficiency
template<KernelType kernel_type>
void compute_kernel_dimensions_template(int n, dim3* threadsPerBlock, dim3* numBlocks);

// Template specializations for each kernel type
template<>
void compute_kernel_dimensions_template<KERNEL_NAIVE>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
void compute_kernel_dimensions_template<KERNEL_TILED>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(TILE_SIZE, TILE_SIZE);
    *numBlocks = dim3((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
}

template<>
void compute_kernel_dimensions_template<KERNEL_TILED_RECT>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_N, BLOCK_M);
    *numBlocks = dim3((n + TILE_N - 1) / TILE_N, (n + TILE_M - 1) / TILE_M);
}

template<>
void compute_kernel_dimensions_template<KERNEL_CUBLAS>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
void compute_kernel_dimensions_template<KERNEL_CUBLAS_TENSOR>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
void compute_kernel_dimensions_template<KERNEL_CUTLASS>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
void compute_kernel_dimensions_template<KERNEL_CUTLASS_TENSOR>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
void compute_kernel_dimensions_template<KERNEL_CUTLASS_SPLITK_FLAT>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    // Use standard 2D configuration for split-K flat implementation
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
void compute_kernel_dimensions_template<KERNEL_CUTLASS_SPLITK_PAIRWISE>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    // Use standard 2D configuration for split-K pairwise implementation
    *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

template<>
void compute_kernel_dimensions_template<KERNEL_HELPER_1D>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    // Compute total elements from matrix dimension (like other kernels)
    const int total_elements = n * n;

    *threadsPerBlock = dim3(256);
    *numBlocks = dim3((total_elements + 256 - 1) / 256);
}

// Add template specialization for mixed precision kernel
template<>
void compute_kernel_dimensions_template<KERNEL_TILED_MIXPREC>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(TILE_SIZE, TILE_SIZE);
    *numBlocks = dim3((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
}

template<>
void compute_kernel_dimensions_template<KERNEL_TILED_PAIRWISE_MIXPREC>(int n, dim3* threadsPerBlock, dim3* numBlocks) {
    *threadsPerBlock = dim3(TILE_SIZE, TILE_SIZE);
    *numBlocks = dim3((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
}

// Runtime dispatch function that calls the appropriate template specialization
void compute_kernel_dimensions_dispatch(KernelType kernel_type, int n, dim3* threadsPerBlock, dim3* numBlocks) {
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
        case KERNEL_CUTLASS_SPLITK_FLAT:
            compute_kernel_dimensions_template<KERNEL_CUTLASS_SPLITK_FLAT>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_CUTLASS_SPLITK_PAIRWISE:
            compute_kernel_dimensions_template<KERNEL_CUTLASS_SPLITK_PAIRWISE>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_HELPER_1D:
            compute_kernel_dimensions_template<KERNEL_HELPER_1D>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_TILED_MIXPREC:
            compute_kernel_dimensions_template<KERNEL_TILED_MIXPREC>(n, threadsPerBlock, numBlocks);
            break;
        case KERNEL_TILED_PAIRWISE_MIXPREC:
            compute_kernel_dimensions_template<KERNEL_TILED_PAIRWISE_MIXPREC>(n, threadsPerBlock, numBlocks);
            break;
        default:
            // Default fallback
            *threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
            *numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            break;
    }
}

// 1D kernel dimension dispatch function for helper kernels
void compute_kernel_dimensions_dispatch_1D(int total_elements, int* threadsPerBlock, int* numBlocks) {
    // Standard 1D configuration for element-wise operations
    *threadsPerBlock = 256;  // Optimal block size for most GPUs
    *numBlocks = (total_elements + 256 - 1) / 256;
}
// Efficient string-based wrapper that uses templates internally
void compute_dimensions(const char* kernel_name, int n, dim3* threadsPerBlock, dim3* numBlocks) {
    KernelType kernel_type = getKernelTypeFromName(kernel_name);
    compute_kernel_dimensions_dispatch(kernel_type, n, threadsPerBlock, numBlocks);
}

// Add to utils.cu:
template<typename T>
void fill_matrix_typed(T* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
    }
}

// Explicit instantiations
template void fill_matrix_typed<float>(float* mat, int N);
template void fill_matrix_typed<__half>(__half* mat, int N);
#ifdef __CUDA_BF16_TYPES_EXIST__
template void fill_matrix_typed<__nv_bfloat16>(__nv_bfloat16* mat, int N);
#endif

const char* getComputeTypeString() {
    if constexpr (std::is_same_v<COMPUTE_TYPE, float>) {
        return "FP32";
    } else if constexpr (std::is_same_v<COMPUTE_TYPE, __half>) {
        return "FP16";
    } else if constexpr (std::is_same_v<COMPUTE_TYPE, double>) {
        return "FP64";
    }
#ifdef __CUDA_BF16_TYPES_EXIST__
    else if constexpr (std::is_same_v<COMPUTE_TYPE, __nv_bfloat16>) {
        return "BF16";
    }
#endif
    else if constexpr (sizeof(COMPUTE_TYPE) == 1) {
        return "INT8";  // For int8_t types
    } else if constexpr (sizeof(COMPUTE_TYPE) == 2 && std::is_integral_v<COMPUTE_TYPE>) {
        return "INT16"; // For int16_t types
    } else {
        return "UNKNOWN";
    }
}

// Add this function to utils.cu:
const char* getAccumulateTypeString() {
    if constexpr (std::is_same_v<ACCUMULATE_TYPE, float>) {
        return "FP32";
    } else if constexpr (std::is_same_v<ACCUMULATE_TYPE, double>) {
        return "FP64";
    } else if constexpr (std::is_same_v<ACCUMULATE_TYPE, __half>) {
        return "FP16";
    }
#ifdef __CUDA_BF16_TYPES_EXIST__
    else if constexpr (std::is_same_v<ACCUMULATE_TYPE, __nv_bfloat16>) {
        return "BF16";
    }
#endif
    else {
        return "UNKNOWN";
    }
}

