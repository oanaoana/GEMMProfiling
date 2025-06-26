#include "benchmark.h"
#include "gemms.cuh"
#include "utils.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Define available tests (must match NUM_TESTS in header)
TestCase available_tests[NUM_TESTS] = {
    {"naive", launch_naive, true},
    {"tiled", launch_tiled, true},                    // Square 16×16
    {"tiled_rect", launch_tiled_rect, true},          // Rectangular 16×32
    {"cublas", launch_cublas, true},
    {"cublas_tensor", launch_cublas_tensor, true},
    {"cutlass", launch_cutlass, true}
};

// Define available sizes
const int SIZES[] = {256, 512, 1024, 2048, 4096};
const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

// Benchmark function
void runBenchmark(const char* name, int n, KernelFunc kernel,
                  float* h_A, float* h_B, float* h_C,
                  float* d_A, float* d_B, float* d_C,
                  FILE* dataFile) {

    size_t size = n * n * sizeof(float);
    double operations = 2.0 * n * n * n; // 2*N^3 FLOPs for matrix multiplication
    double mem_access_bytes = 3.0 * n * n * sizeof(float); // 3*N^2 floats read/written
    double arithmetic_intensity = operations / mem_access_bytes;

    // Set dimensions dynamically based on kernel type
    dim3 threadsPerBlock, numBlocks;

    if (strcmp(name, "naive") == 0) {
        // Naive: use BLOCK_SIZE for general purpose blocking
        threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
        numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }
    else if (strcmp(name, "tiled") == 0) {
        // Square tiling: use TILE_SIZE
        threadsPerBlock = dim3(TILE_SIZE, TILE_SIZE);
        numBlocks = dim3((n + TILE_SIZE - 1) / TILE_SIZE,
                        (n + TILE_SIZE - 1) / TILE_SIZE);
    }
    else if (strcmp(name, "tiled_rect") == 0) {
        // Rectangular tiling: use TILE_M, TILE_N
        threadsPerBlock = dim3(BLOCK_N, BLOCK_M);  // Note: N,M order for x,y
        numBlocks = dim3((n + TILE_N - 1) / TILE_N,
                        (n + TILE_M - 1) / TILE_M);
    }
    else {
        // Default for cuBLAS, CUTLASS, etc.: use BLOCK_SIZE
        threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
        numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }

    printf("Debug %s: Grid(%d,%d), Block(%d,%d)\n",
           name, numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup run
    kernel(d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
    cudaDeviceSynchronize();

    // Clear result matrix with a known pattern
    cudaMemset(d_C, 0, size);

    // Add checksum BEFORE kernel
    float checksum_before = 0.0f;
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100; i++) checksum_before += h_C[i];
    printf("Debug: Checksum before %s: %.6f\n", name, checksum_before);

    // WARM-UP RUNS
    printf("  Warming up...");
    for (int i = 0; i < 3; i++) {
        kernel(d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
    }
    cudaDeviceSynchronize();  // Ensure all warm-up runs complete
    printf(" done\n");

    // MULTIPLE TIMED RUNS
    int num_runs = (n < 1024) ? 10 : 1;
    float total_time = 0.0f;
    float run_times[num_runs];  // Declare array here

    for (int run = 0; run < num_runs; run++) {
        cudaEventRecord(start);

        // Multiple kernel calls per timing measurement
        int iterations_per_run = (n < 1024) ? 10 : 1;
        for (int iter = 0; iter < iterations_per_run; iter++) {
            kernel(d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float run_time;
        cudaEventElapsedTime(&run_time, start, stop);
        run_time /= iterations_per_run;  // Average per kernel call

        run_times[run] = run_time;  // Store the time
        total_time += run_time;

        printf("  Run %d: %.2f ms\n", run + 1, run_time);
    }

    // Use AVERAGE time
    float average_time = total_time / num_runs;
    printf("  Average: %.2f ms\n", average_time);

    // Calculate variance (MOVE this to after average_time calculation)
    float variance = 0.0f;
    for (int run = 0; run < num_runs; run++) {
        float diff = run_times[run] - average_time;
        variance += diff * diff;
    }
    variance /= num_runs;
    float std_dev = sqrtf(variance);
    float cv = (std_dev / average_time) * 100.0f;
    printf("  Std dev: %.2f ms (CV: %.1f%%)\n", std_dev, cv);

    // Add checksum AFTER kernel
    float checksum_after = 0.0f;
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100; i++) checksum_after += h_C[i];
    printf("Debug: Checksum after %s: %.6f\n", name, checksum_after);

    // Check for kernel errors
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        printf("CUDA Error after %s kernel: %s\n", name, cudaGetErrorString(kernel_err));
    }

    double gigaFlops = (operations / (average_time / 1000.0)) / 1e9;
    double bandwidth_gbps = (mem_access_bytes / (average_time / 1000.0)) / 1e9;

    printf("%s (N=%d): %.2f ms, %.2f GFLOP/s, %.2f GB/s, AI=%.2f\n",
           name, n, average_time, gigaFlops, bandwidth_gbps, arithmetic_intensity);

    // Save to CSV
    fprintf(dataFile, "%s,%d,%.2f,%.2f,%.2f,%.2f\n",
            name, n, average_time, gigaFlops, bandwidth_gbps, arithmetic_intensity);


    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Main benchmark function
void runAllBenchmarks(bool* enabled_tests, bool* enabled_sizes) {
    printf("=== Starting Benchmarks ===\n");

    // Open data file for roofline model
    FILE* dataFile = fopen("roofline_data.csv", "w");
    if (!dataFile) {
        printf("ERROR: Could not create roofline_data.csv\n");
        return;
    }
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

        if (!h_A || !h_B || !h_C) {
            printf("ERROR: Failed to allocate host memory\n");
            return;
        }

        fill_matrix(h_A, n);
        fill_matrix(h_B, n);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaError_t err;
        err = cudaMalloc(&d_A, size);
        if (err != cudaSuccess) {
            printf("ERROR: cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
            return;
        }
        err = cudaMalloc(&d_B, size);
        if (err != cudaSuccess) {
            printf("ERROR: cudaMalloc d_B failed: %s\n", cudaGetErrorString(err));
            return;
        }
        err = cudaMalloc(&d_C, size);
        if (err != cudaSuccess) {
            printf("ERROR: cudaMalloc d_C failed: %s\n", cudaGetErrorString(err));
            return;
        }

        // Copy input data to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Run each enabled test
        for (int j = 0; j < NUM_TESTS; j++) {
            if (!enabled_tests[j]) continue;

            printf("\n===== %s =====\n", available_tests[j].name);
            runBenchmark(available_tests[j].name, n, available_tests[j].kernel,
                         h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

            // Verify results for first size only
            if (i == 0) {
                cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
                verify_result(h_A, h_B, h_C, n);
            }
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