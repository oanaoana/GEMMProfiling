#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matmul_naive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void fill_matrix(float *mat, int N) {
    for (int i = 0; i < N*N; ++i)
        mat[i] = static_cast<float>(i % 100);
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    fill_matrix(h_A, N);
    fill_matrix(h_B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    matmul_naive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Timing run
    cudaEventRecord(start);
    matmul_naive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate GFLOP/s
    // For matrix multiplication: 2*N^3 floating point operations
    double operations = 2.0 * N * N * N;  // 2*N^3 for N×N matrices
    double gigaFlops = (operations / (milliseconds / 1000.0)) / 1e9;

    printf("Matrix size: %d x %d\n", N, N);
    printf("Execution time: %f ms\n", milliseconds);
    printf("Performance: %f GFLOP/s\n", gigaFlops);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Result[0] = %f\n", h_C[0]);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
