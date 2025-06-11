#include "benchmark.h"
#include "gemms.cuh"
#include "utils.cuh"
#include <stdio.h>
#include <stdlib.h>

// Define available tests (must match NUM_TESTS in header)
TestCase available_tests[NUM_TESTS] = {
    {"naive", launch_naive, true},
    {"tiled", launch_tiled, true},
    {"cublas", launch_cublas, true},
    // Add more tests here and update NUM_TESTS accordingly
};

// Define available sizes
const int SIZES[] = {128, 256, 512, 1024, 2048, 4096};
const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

// Benchmark function
void runBenchmark(const char* name, int n, KernelFunc kernel,
                  float* h_A, float* h_B, float* h_C,
                  float* d_A, float* d_B, float* d_C,
                  FILE* dataFile) {

    size_t size = n * n * sizeof(float);
    size_t mem_access = 3 * size; // Read A, Read B, Write C
    double operations = 2.0 * n * n * n; // 2*N^3 FLOPs for matrix multiplication
    double arithmetic_intensity = operations / mem_access;

    // Set dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup run
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
    double bandwidth_gbps = (mem_access / (milliseconds / 1000.0)) / 1e9;

    printf("%s (N=%d): %.2f ms, %.2f GFLOP/s, %.2f GB/s, AI=%.2f\n",
           name, n, milliseconds, gigaFlops, bandwidth_gbps, arithmetic_intensity);

    // Save to CSV
    fprintf(dataFile, "%s,%d,%.2f,%.2f,%.2f,%.2f\n",
            name, n, milliseconds, gigaFlops, bandwidth_gbps, arithmetic_intensity);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Main benchmark function
void runAllBenchmarks(bool* enabled_tests, bool* enabled_sizes) {
    printDevicePerformanceInfo();

    // Open data file for roofline model
    FILE* dataFile = fopen("roofline_data.csv", "w");
    fprintf(dataFile, "algorithm,size,time_ms,gflops,bandwidth_gb,arithmetic_intensity\n");

    // Test each matrix size
    for (int i = 0; i < NUM_SIZES; i++) {
        if (!enabled_sizes[i]) continue;

        int n = SIZES[i];
        printf("\n--- Testing matrix size %d x %d ---\n", n, n);

        // Allocate host memory
        size_t size = n * n * sizeof(float);
        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        float *h_C = (float*)malloc(size);

        fill_matrix(h_A, n);
        fill_matrix(h_B, n);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        // Copy input data to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Run each enabled test
        for (int j = 0; j < NUM_TESTS; j++) {
            if (!enabled_tests[j]) continue;

            printf("\n===== %s =====\n", available_tests[j].name);
            runBenchmark(available_tests[j].name, n, available_tests[j].kernel,
                         h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

            // Optionally verify results
            // cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
            // verify_result(h_A, h_B, h_C, n);
        }

        // Free memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
    }

    fclose(dataFile);
    printf("\nBenchmark complete. Results saved to roofline_data.csv\n");
}