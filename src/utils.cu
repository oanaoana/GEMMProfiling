#include "../include/utils.cuh"
#include <stdio.h>
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

void verify_result(float *A, float *B, float *C, int N) {
    // Use more appropriate epsilon for float-to-double comparison
    float eps = 1e-6;
    float max_rel_error = 0.0f;
    float sum_rel_error = 0.0f;
    int error_count = 0;

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            // Calculate reference in double precision
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += (double)A[row * N + k] * (double)B[k * N + col];
            }

            // Store CPU result for printing
            float cpu_result = (float)sum;

            // Compare GPU float result with double precision reference
            double abs_error = fabs((double)C[row * N + col] - sum);
            double rel_error = abs_error / (fabs(sum) > 1e-10 ? fabs(sum) : 1e-10);

            // Record statistics in double precision
            max_rel_error = fmax(max_rel_error, (float)rel_error);
            sum_rel_error += (float)rel_error;

            // Convert error threshold to match the precision of abs_error
            if (abs_error > (double)eps) {
                error_count++;
                if (error_count <= 5) { // Limit output to first 5 errors
                    printf("Mismatch at (%d, %d): GPU = %f, CPU = %f, Rel Error = %e\n",
                          row, col, C[row * N + col], cpu_result, rel_error);
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
}


void check_occupancy() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Check occupancy for different configurations
    int maxActiveBlocks16, maxActiveBlocks32;

    // For TILE_SIZE=16 (16×16 = 256 threads per block)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks16,
                                                  matmul_tiled, 256, 2048);

    // For TILE_SIZE=32 (32×32 = 1024 threads per block)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks32,
                                                  matmul_tiled, 1024, 8192);

    printf("=== OCCUPANCY ANALYSIS ===\n");
    printf("GPU: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);

    printf("\nTILE_SIZE=16 (256 threads/block, 2KB shared mem):\n");
    printf("  Max active blocks per SM: %d\n", maxActiveBlocks16);
    printf("  Threads per SM: %d (%.1f%% occupancy)\n",
           maxActiveBlocks16 * 256,
           (maxActiveBlocks16 * 256.0f / prop.maxThreadsPerMultiProcessor) * 100);

    printf("\nTILE_SIZE=32 (1024 threads/block, 8KB shared mem):\n");
    printf("  Max active blocks per SM: %d\n", maxActiveBlocks32);
    printf("  Threads per SM: %d (%.1f%% occupancy)\n",
           maxActiveBlocks32 * 1024,
           (maxActiveBlocks32 * 1024.0f / prop.maxThreadsPerMultiProcessor) * 100);
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