#include "../include/benchmark.h"
#include "../include/gemms.cuh"
#include "../include/utils.cuh"
#include "../include/error_analysis.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>


// Define available tests using unified kernel types
const char* available_test_names[NUM_TESTS] = {
    "naive", "tiled", "tiled_opt", "tiled_pairwise",
    "tiled_rect", "cublas", "cublas_tensor", "cutlass", ""
};

// Benchmark function using unified kernel dispatch
void runBenchmark(const char* name, int n, KernelType kernel_type,
                  float* h_A, float* h_B, float* h_C,
                  float* d_A, float* d_B, float* d_C,
                  FILE* dataFile) {

    size_t size = n * n * sizeof(float);
    double operations = 2.0 * n * n * n; // 2*N^3 FLOPs for matrix multiplication
    double mem_access_bytes = 3.0 * n * n * sizeof(float); // 3*N^2 floats read/written
    double arithmetic_intensity = operations / mem_access_bytes;

    // Compute dimensions using efficient template-based config function
    dim3 threadsPerBlock, numBlocks;
    compute_dimensions(name, n, &threadsPerBlock, &numBlocks);

    printf("Debug %s: Grid(%d,%d), Block(%d,%d)\n",
           name, numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup run using unified dispatch
    launch_kernel_by_type(kernel_type, d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
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
        launch_kernel_by_type(kernel_type, d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
    }
    cudaDeviceSynchronize();  // Ensure all warm-up runs complete
    printf(" done\n");

    // MULTIPLE TIMED RUNS
    int num_runs = (n < 1024) ? 10 : 5;  // At least 5 runs for N=1024
    float total_time = 0.0f;
    float run_times[10];  // Fixed array size for safety

    for (int run = 0; run < num_runs; run++) {
        cudaEventRecord(start);

        // Multiple kernel calls per timing measurement
        int iterations_per_run = (n < 1024) ? 10 : 3;  // At least 3 iterations for N=1024
        for (int iter = 0; iter < iterations_per_run; iter++) {
            launch_kernel_by_type(kernel_type, d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
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

    FILE* dataFile = fopen("data/roofline_data.csv", "w");
    if (!dataFile) {
        printf("ERROR: Could not create data/roofline_data.csv\n");
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

        cudaMalloc((void**)&d_A, n * n * sizeof(float));
        cudaMalloc((void**)&d_B, n * n * sizeof(float));
        cudaMalloc((void**)&d_C, n * n * sizeof(float));

        //cudaMemcpy2D(d_A, pitch_A, h_A, n * sizeof(float), n * sizeof(float), n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Run each enabled test
        for (int j = 0; j < NUM_TESTS; j++) {
            if (!enabled_tests[j]) continue;
            if (strlen(available_test_names[j]) == 0) continue; // Skip empty names

            printf("\n===== %s =====\n", available_test_names[j]);

            // Do the lookup ONCE before timing-critical code
            KernelType kernel_type = getKernelTypeFromName(available_test_names[j]);
            if (kernel_type == static_cast<KernelType>(-1)) {
                printf("Error: Unknown kernel type for '%s'\n", available_test_names[j]);
                continue;
            }

            runBenchmark(available_test_names[j], n, kernel_type,
                         h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

            // Verification removed - use --error-analysis for accuracy testing
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
    printf("\nBenchmark complete. Results saved to data/roofline_data.csv\n");
}

// Simple function to run a single benchmark with arbitrary size
void runSingleBenchmark(const char* test_name, int n) {
    printf("=== Single Benchmark Test ===\n");
    printf("Test: %s, Size: %dx%d\n", test_name, n, n);

    // Find the kernel type using unified lookup
    KernelType kernel_type = getKernelTypeFromName(test_name);
    if (kernel_type == static_cast<KernelType>(-1)) {
        printf("Error: Test '%s' not found\n", test_name);
        return;
    }

    // Allocate memory
    size_t size = n * n * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
        h_C[i] = 0.0f;
    }

    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    // Open output file
    FILE* dataFile = fopen("data/roofline_data.csv", "w");
    if (dataFile) {
        fprintf(dataFile, "Test,N,Time(ms),GFLOP/s,Bandwidth(GB/s),ArithmeticIntensity\n");
    }

    // Run the benchmark using unified dispatch
    runBenchmark(test_name, n, kernel_type, h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

    // Clean up
    if (dataFile) fclose(dataFile);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nSingle benchmark complete!\n");
}

// Note: runNumericalAnalysisBenchmarks has been moved to error_tests.cu