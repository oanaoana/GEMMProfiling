#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include "gemms.cuh"
#include "utils.cuh"

#define N 1024

int main() {

    printf("Matrix size: %d x %d\n", N, N);
    printDevicePerformanceInfo();

    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_tiled = (float*)malloc(size);  // For tiled GEMM result
    fill_matrix(h_A, N);
    fill_matrix(h_B, N);

    float *d_A, *d_B, *d_C, *d_C_tiled;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_C_tiled, size);  // For tiled GEMM result
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // =========== NAIVE GEMM ===========
    printf("\n===== NAIVE GEMM =====\n");

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

    printf("Execution time: %f ms\n", milliseconds);
    printf("Performance: %f GFLOP/s\n", gigaFlops);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    verify_result(h_A, h_B, h_C, N);

    // =========== TILED GEMM ===========
    printf("\n===== TILED GEMM =====\n");

    // Calculate grid and block dimensions for tiled kernel
    dim3 tiledThreads(TILE_SIZE, TILE_SIZE);
    dim3 tiledBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up run
    matmul_tiled<<<tiledBlocks, tiledThreads>>>(d_A, d_B, d_C_tiled, N);
    cudaDeviceSynchronize();

    // Timing run
    cudaEventRecord(start);
    matmul_tiled<<<tiledBlocks, tiledThreads>>>(d_A, d_B, d_C_tiled, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float tiled_milliseconds = 0;
    cudaEventElapsedTime(&tiled_milliseconds, start, stop);

    // Calculate GFLOP/s
    double tiled_gigaFlops = (operations / (tiled_milliseconds / 1000.0)) / 1e9;

    printf("Execution time: %f ms\n", tiled_milliseconds);
    printf("Performance: %f GFLOP/s\n", tiled_gigaFlops);
    printf("Speedup over naive: %.2fx\n", milliseconds / tiled_milliseconds);

    cudaMemcpy(h_C_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost);
    verify_result(h_A, h_B, h_C_tiled, N);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_tiled);
    free(h_A); free(h_B); free(h_C); free(h_C_tiled);
    return 0;
}