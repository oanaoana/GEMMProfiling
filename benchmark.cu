#include "benchmark.h"
#include "gemms.cuh"
#include "include/utils.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Add these global variable definitions to benchmark.cu
int g_pitch_A = 0;
int g_pitch_B = 0;
int g_pitch_C = 0;
bool g_use_pitched_memory = false;

// Declare extern flag from main.cu
extern bool g_verify_results;

// Define available tests (must match NUM_TESTS in header)
TestCase available_tests[NUM_TESTS] = {
    {"naive", launch_naive, true},
    {"tiled", launch_tiled, true},
    {"tiled_rect", launch_tiled_rect, true},
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
        // Rectangular tiling: use TILE_M, TILE_N for output
        threadsPerBlock = dim3(BLOCK_N, BLOCK_M);          // (16, 16)
        numBlocks = dim3((n + TILE_N - 1) / TILE_N,        // (n+15)/16
                        (n + TILE_M - 1) / TILE_M);        // (n+15)/16
    }
    else if (strcmp(name, "cublas") == 0 || strcmp(name, "cublas_tensor") == 0 ||
             strcmp(name, "cutlass") == 0 || strcmp(name, "cutlass_tensor") == 0) {
        // For cuBLAS and CUTLASS, use BLOCK_SIZE
        threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
        numBlocks = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
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
            // Force use of pitched version for tiled kernel
            if (strcmp(name, "tiled_pitch") == 0) {
                launch_tiled_pitched(d_A, d_B, d_C, n, numBlocks, threadsPerBlock, g_pitch_A);
            } else {
                kernel(d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
            }
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
    // Reset pitch info for each benchmark run
    g_pitch_A = 0;
    g_pitch_B = 0;
    g_pitch_C = 0;
    g_use_pitched_memory = false;

    printf("=== Starting Benchmarks ===\n");

    FILE* dataFile = fopen("roofline_data.csv", "w");
    if (!dataFile) {
        printf("ERROR: Could not create roofline_data.csv\n");
        return;
    }
    fprintf(dataFile, "algorithm,size,time_ms,gflops,bandwidth_gb,arithmetic_intensity\n");

    for (int i = 0; i < NUM_SIZES; i++) {
        if (!enabled_sizes[i]) continue;

        int n = SIZES[i];
        printf("\n--- Testing matrix size %d x %d ---\n", n, n);

        // Allocate memory
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

        float *d_A, *d_B, *d_C;
        g_use_pitched_memory = true;        // Enable pitched mode

        size_t pitch_A, pitch_B, pitch_C;
        cudaMalloc((void**)&d_A, n * n * sizeof(float));
        cudaMalloc((void**)&d_B, n * n * sizeof(float));
        cudaMalloc((void**)&d_C, n * n * sizeof(float));
        pitch_A = n;  // Row-major pitch
        pitch_B = n;  // Row-major pitch
        pitch_C = n;  // Row-major pitch
        //err = cudaMallocPitch((void**)&d_A, &pitch_A, n * sizeof(float), n);

        g_pitch_A = pitch_A / sizeof(float);
        g_pitch_B = pitch_B / sizeof(float);
        g_pitch_C = pitch_C / sizeof(float);

        //cudaMemcpy2D(d_A, pitch_A, h_A, n * sizeof(float), n * sizeof(float), n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Run each enabled test
        for (int j = 0; j < NUM_TESTS; j++) {
            if (!enabled_tests[j]) continue;

            printf("\n===== %s =====\n", available_tests[j].name);
            runBenchmark(available_tests[j].name, n, available_tests[j].kernel,
                         h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

            // Clean, simple verification based on runtime flag
            if (g_verify_results) {
                printf("\nVerifying results for %s kernel at size %d...\n",
                       available_tests[j].name, n);
                cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
                verify_result(h_A, h_B, h_C, n);
            }
        }

        // Free memory (now in correct scope)
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