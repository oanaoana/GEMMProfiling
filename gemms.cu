#include "gemms.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>  // Add this include
#include <stdio.h>

__global__ void matmul_naive( float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;}
}

__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < N && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

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