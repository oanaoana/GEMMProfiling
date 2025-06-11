#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include "gemms.cuh"
#include "utils.cuh"
#include <cublas_v2.h>

// Array of matrix sizes to test
const int SIZES[] = {64, 128, 256, 512, 1024, 2048, 4096};
const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

void runBenchmark(const char* name, int n,
                  void (*kernel)(float*, float*, float*, int, dim3, dim3),
                  float* h_A, float* h_B, float* h_C,
                  float* d_A, float* d_B, float* d_C,
                  FILE* dataFile) {

    size_t size = n * n * sizeof(float);
    size_t mem_access = 3 * size; // Read A, B and write C matrices
    double operations = 2.0 * n * n * n;  // 2*N^3 for N×N matrices
    double arithmetic_intensity = operations / mem_access;

    // Set grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    kernel(d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
    cudaDeviceSynchronize();

    // Timing run
    cudaEventRecord(start);
    kernel(d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double gigaFlops = (operations / (milliseconds / 1000.0)) / 1e9;
    double gb_per_s = mem_access / (milliseconds / 1000.0) / 1e9;

    printf("%s - Size: %d, Time: %.2f ms, GFLOP/s: %.2f, GB/s: %.2f, AI: %.2f\n",
           name, n, milliseconds, gigaFlops, gb_per_s, arithmetic_intensity);

    // Copy results back for verification
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // After calculating performance metrics:
    fprintf(dataFile, "%s,%d,%.2f,%.2f,%.2f,%.2f\n",
            name, n, milliseconds, gigaFlops, gb_per_s, arithmetic_intensity);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Wrapper functions for kernel launches
void launch_naive(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, n);
}

void launch_tiled(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
}

int main() {
    // Print device info
    printDevicePerformanceInfo();

    // Open file for data collection
    FILE* dataFile = fopen("roofline_data.csv", "w");
    fprintf(dataFile, "algorithm,size,time_ms,gflops,bandwidth_gb,arithmetic_intensity\n");

    for (int i = 0; i < NUM_SIZES; i++) {
        int n = SIZES[i];
        printf("\nTesting matrix size: %d x %d\n", n, n);

        size_t size = n * n * sizeof(float);

        // Allocate host memory
        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        float *h_C = (float*)malloc(size);

        // Initialize matrices
        fill_matrix(h_A, n);
        fill_matrix(h_B, n);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        // Copy input matrices to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Calculate operations and memory access for this size
        size_t mem_access = 3 * size; // Read A, B and write C matrices
        double operations = 2.0 * n * n * n;  // 2*N^3 for N×N matrices
        double arithmetic_intensity = operations / mem_access;

        // Run naive implementation
        printf("\n===== NAIVE GEMM =====\n");
        runBenchmark("Naive", n, launch_naive, h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

        // Run tiled implementation
        printf("\n===== TILED GEMM =====\n");
        runBenchmark("Tiled", n, launch_tiled, h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

        // Run cuBLAS implementation
        printf("\n===== cuBLAS GEMM =====\n");
        runBenchmark("cuBLAS", n, launch_cublas, h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

        // Cleanup for this size
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
    }

    //verify_result(h_A, h_B, h_C, n);


    fclose(dataFile);
    return 0;
}