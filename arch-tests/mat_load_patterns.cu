// Memory Load Pattern Analysis for GPU Architecture Testing
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global variables for pitched memory
int g_pitch_A = 0;
int g_pitch_C = 0;

// ===== KERNEL FUNCTIONS =====

// Row-major copy test kernel (Sequential access pattern)
__global__ void matrix_copy_test_rowmajor(float *A, float *C, int N, int pitch_A, int pitch_C) {
    // Calculate global thread ID regardless of 1D/2D/3D block layout
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // For 2D blocks, need different calculation
    if (blockDim.y > 1) {
        tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    }

    if (tid < N * N) {
        // Row-major mapping (independent of thread layout)
        int row = tid / N;
        int col = tid % N;

        // if (tid < 32) {
        //     printf("T%02d: row=%d, col=%d, A_addr=%d\n",
        //            tid, row, col, row * pitch_A + col);
        // }

        float value = __ldg(&A[row * pitch_A + col]);
        C[row * pitch_C + col] = value;
    }
}

// Column-major copy test kernel (Strided access pattern)
__global__ void matrix_copy_test_colmajor(float *A, float *C, int N, int pitch_A, int pitch_C) {
    // Same thread ID calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockDim.y > 1) {
        tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    }

    if (tid < N * N) {
        // Column-major mapping (independent of thread layout)
        int col = tid / N;
        int row = tid % N;

        // if (tid < 32) {
        //     printf("T%02d: row=%d, col=%d, A_addr=%d\n",
        //            tid, row, col, row * pitch_A + col);
        // }

        float value = __ldg(&A[row * pitch_A + col]);
        C[row * pitch_C + col] = value;
    }
}

// Random access pattern kernel (Worst case for coalescing)
__global__ void matrix_copy_test_random(float *A, float *C, int N, int pitch_A, int pitch_C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N * N) {
        // Random access pattern (worst coalescing expected)
        int row = (tid * 17 + 23) % N;  // Pseudo-random
        int col = (tid * 31 + 47) % N;

        // if (tid < 32) {
        //     printf("RND T%02d: row=%d, col=%d, A_addr=%d\n",
        //            tid, row, col, row * pitch_A + col);
        // }

        float value = __ldg(&A[row * pitch_A + col]);

        // Write to sequential location to avoid write coalescing issues
        int out_row = tid / N;
        int out_col = tid % N;
        C[out_row * pitch_C + out_col] = value;
    }
}

// ===== LAUNCH FUNCTIONS =====

void launch_copy_test_rowmajor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    printf("DEBUG: RowMajor - Grid(%d,%d,%d), Block(%d,%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

    matrix_copy_test_rowmajor<<<blocks, threads>>>(d_A, d_C, n, g_pitch_A, g_pitch_C);
    cudaDeviceSynchronize();
}

void launch_copy_test_colmajor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    printf("DEBUG: ColMajor - Grid(%d,%d,%d), Block(%d,%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

    matrix_copy_test_colmajor<<<blocks, threads>>>(d_A, d_C, n, g_pitch_A, g_pitch_C);
    cudaDeviceSynchronize();
}

void launch_copy_test_random(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    printf("DEBUG: Random - Grid(%d,%d,%d), Block(%d,%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

    matrix_copy_test_random<<<blocks, threads>>>(d_A, d_C, n, g_pitch_A, g_pitch_C);
    cudaDeviceSynchronize();
}

// ===== DEVICE INFO FUNCTIONS =====

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

void printDeviceInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n===== DEVICE INFORMATION =====\n");
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Global memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Memory bandwidth calculation
    double memory_clock_rate = prop.memoryClockRate / 1000000.0; // Convert from kHz to GHz
    double bus_width = prop.memoryBusWidth;
    double peak_bandwidth = 2.0 * memory_clock_rate * (bus_width / 8); // GB/s

    printf("Memory clock rate: %.1f GHz\n", memory_clock_rate);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("Peak memory bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("===============================\n\n");
}

// ===== TEST FRAMEWORK =====

void runLoadPatternTest(const char* pattern_name, void (*launch_func)(float*, float*, float*, int, dim3, dim3),
                        float* d_A, float* d_C, int N, dim3 blocks, dim3 threads) {
    printf("=== %s LOAD PATTERN TEST ===\n", pattern_name);
    printf("Grid: (%d,%d,%d), Block: (%d,%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("Total threads: %d\n", blocks.x * blocks.y * threads.x * threads.y);

    launch_func(d_A, nullptr, d_C, N, blocks, threads);
    printf("Test completed.\n\n");
}
