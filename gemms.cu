// gemms.cu
#include "gemms.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// Naive implementation
__global__ void matmul_naive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch wrapper for naive implementation
void launch_naive(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, n);
}

// Tiled implementation
__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load A tile (this was correct)
        int A_col = t * TILE_SIZE + threadIdx.x;
        tile_A[threadIdx.y][threadIdx.x] = (row < N && A_col < N) ?
                                          A[row * N + A_col] : 0.0f;

        // FIXED: Load B tile correctly
        int B_row = t * TILE_SIZE + threadIdx.y;  // Use threadIdx.y for row
        int B_col = col;  // Use the actual column this thread is computing
        tile_B[threadIdx.y][threadIdx.x] = (B_row < N && B_col < N) ?
                                          B[B_row * N + B_col] : 0.0f;

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Launch wrapper for tiled implementation
void launch_tiled(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
}

// cuBLAS implementation
void launch_cublas(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Setup alpha and beta for sgemm
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Call cuBLAS sgemm
    // Note: cuBLAS uses column-major order while our code uses row-major order
    // So we compute C = B*A as a workaround for row-major C = A*B
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_B, n,  // B matrix
                d_A, n,  // A matrix
                &beta,
                d_C, n); // C matrix

    // Destroy handle
    cublasDestroy(handle);
}

// Add TensorCore implementation
void launch_cublas_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Enable TensorCore math if supported
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_B, n,
                d_A, n,
                &beta,
                d_C, n);

    cublasDestroy(handle);
}