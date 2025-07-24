// Memory Access Pattern Tests for GPU Architecture Analysis
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default tile size if not defined
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

// Global variables for pitched memory
int g_pitch_A = 0;
int g_pitch_C = 0;

// ===== DIRECT MEMORY ACCESS PATTERNS =====

// Row-major pattern - simple copy
__global__ void matrix_copy_test_rowmajor(float *A, float *C, int N, int pitch_A, int pitch_C) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockDim.y > 1) {
        tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    }

    if (tid < N * N) {
        // Row-major mapping
        int row = tid / N;
        int col = tid % N;

        // Load value with texture cache
        float value = __ldg(&A[row * pitch_A + col]);

        // Store
        C[row * pitch_C + col] = value;
    }
}

// Column-major pattern - simple copy
__global__ void matrix_copy_test_colmajor(float *A, float *C, int N, int pitch_A, int pitch_C) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockDim.y > 1) {
        tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    }

    if (tid < N * N) {
        // Column-major mapping
        int col = tid / N;
        int row = tid % N;

        // Load value with texture cache
        float value = __ldg(&A[row * pitch_A + col]);

        // Store
        C[row * pitch_C + col] = value;
    }
}

// Random access pattern - simple copy with random mapping
__global__ void matrix_copy_test_random(float *A, float *C, int N, int pitch_A, int pitch_C) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockDim.y > 1) {
        tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    }

    if (tid < N * N) {
        // Simple hash function for pseudo-random mapping
        // Using prime numbers to create a scatter effect
        int hash = (tid * 1664525 + 1013904223) % (N * N);
        int row = hash / N;
        int col = hash % N;

        // Load value with texture cache
        float value = __ldg(&A[row * pitch_A + col]);

        // Store in sequential order
        int out_row = tid / N;
        int out_col = tid % N;
        C[out_row * pitch_C + out_col] = value;
    }
}

// ===== TILED MEMORY ACCESS PATTERNS =====

// Row-major tiled copy kernel
__global__ void matrix_copy_tiled_rowmajor(float *A, float *C, int N, int pitch_A, int pitch_C) {
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global indices for this thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Load tile using row-major access pattern (coalesced)
    if (row < N && col < N) {
        // Row-major loading (threads in the same warp access consecutive memory)
        tile[ty][tx] = A[row * pitch_A + col];
    }

    // Make sure all threads finished loading
    __syncthreads();

    // Store back to global memory using same pattern
    if (row < N && col < N) {
        C[row * pitch_C + col] = tile[ty][tx];
    }
}

// Column-major tiled copy kernel
__global__ void matrix_copy_tiled_colmajor(float *A, float *C, int N, int pitch_A, int pitch_C) {
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global indices for this thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Load tile using column-major access pattern (non-coalesced)
    if (row < N && col < N) {
        // Column-major loading (threads in the same warp access non-consecutive memory)
        tile[ty][tx] = A[col * pitch_A + row]; // Column-major access
    }

    // Make sure all threads finished loading
    __syncthreads();

    // Store back to global memory in row-major order (to see impact of just the loading)
    if (row < N && col < N) {
        C[row * pitch_C + col] = tile[ty][tx];
    }
}

// ===== LAUNCH FUNCTIONS =====

// Launch non-tiled row-major kernel
void launch_copy_test_rowmajor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    printf("DEBUG: RowMajor - Grid(%d,%d,%d), Block(%d,%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

    matrix_copy_test_rowmajor<<<blocks, threads>>>(d_A, d_C, n, g_pitch_A, g_pitch_C);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Launch non-tiled column-major kernel
void launch_copy_test_colmajor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    printf("DEBUG: ColMajor - Grid(%d,%d,%d), Block(%d,%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

    matrix_copy_test_colmajor<<<blocks, threads>>>(d_A, d_C, n, g_pitch_A, g_pitch_C);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Launch non-tiled random access kernel
void launch_copy_test_random(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    printf("DEBUG: Random - Grid(%d,%d,%d), Block(%d,%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

    matrix_copy_test_random<<<blocks, threads>>>(d_A, d_C, n, g_pitch_A, g_pitch_C);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Launch tiled row-major kernel
void launch_copy_tiled_rowmajor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    printf("DEBUG: Tiled RowMajor - Grid(%d,%d,%d), Block(%d,%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("Using TILE_SIZE=%d\n", TILE_SIZE);

    matrix_copy_tiled_rowmajor<<<blocks, threads>>>(d_A, d_C, n, g_pitch_A, g_pitch_C);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Launch tiled column-major kernel
void launch_copy_tiled_colmajor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    printf("DEBUG: Tiled ColMajor - Grid(%d,%d,%d), Block(%d,%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    printf("Using TILE_SIZE=%d\n", TILE_SIZE);

    matrix_copy_tiled_colmajor<<<blocks, threads>>>(d_A, d_C, n, g_pitch_A, g_pitch_C);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Functions for printing device info and cache configuration
void printDeviceInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", prop.warpSize);
}

void printCacheInfo() {
    int l2_size = 0;
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, 0);

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    printf("L2 cache size: %d KB\n", l2_size / 1024);
    printf("Global memory: %.1f GB (%.1f GB free)\n",
           total / (1024.0 * 1024.0 * 1024.0), free / (1024.0 * 1024.0 * 1024.0));
}
