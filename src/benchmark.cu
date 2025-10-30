#include "../include/benchmark.h"
#include "../include/gemms.cuh"
#include "../include/utils.cuh"
#include "../include/error_analysis.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <time.h>  // For timestamp

// Generic occupancy checker for any kernel
void check_kernel_occupancy(void* kernel_func, const char* kernel_name,
                           int threads_per_block, size_t shared_mem_bytes) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, kernel_func,
                                                  threads_per_block, shared_mem_bytes);

    printf("=== OCCUPANCY: %s ===\n", kernel_name);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Shared memory: %zu bytes\n", shared_mem_bytes);
    printf("Max active blocks per SM: %d\n", maxActiveBlocks);
    printf("Threads per SM: %d (%.1f%% occupancy)\n",
           maxActiveBlocks * threads_per_block,
           (maxActiveBlocks * threads_per_block * 100.0f) / prop.maxThreadsPerMultiProcessor);
    printf("\n");
}

// Alternative: Kernel-specific occupancy analysis
void check_occupancy_for_kernel(KernelType kernel_type, int n) {

    void* kernel_ptr = get_kernel_function_pointer(kernel_type);

    dim3 threadsPerBlock, numBlocks;
    compute_kernel_dimensions_dispatch(kernel_type, n, &threadsPerBlock, &numBlocks);
    // Use the computed dimensions instead of hardcoded values!
    int total_threads_per_block = threadsPerBlock.x * threadsPerBlock.y;

    const char* kernel_name = kernelTypeToString(kernel_type);

    if (kernel_ptr == nullptr) {
        printf("=== OCCUPANCY: %s ===\n", kernel_name);
        printf("Function pointer not available (library kernel)\n\n");
        return;
    }

    // Get the ACTUAL kernel attributes instead of guessing!
    cudaFuncAttributes attr = {0};
    cudaError_t err = cudaFuncGetAttributes(&attr, kernel_ptr);

    if (err != cudaSuccess) {
        printf("=== OCCUPANCY: %s ===\n", kernel_name);
        printf("Error getting kernel attributes: %s\n", cudaGetErrorString(err));
        return;
    }

    // Use the ACTUAL shared memory size from the kernel
    size_t actual_shared_mem = attr.sharedSizeBytes;

    switch(kernel_type) {
        case KERNEL_NAIVE:
            //shared_mem = 0;
            check_kernel_occupancy(kernel_ptr, kernel_name,
                                  total_threads_per_block, actual_shared_mem);
            break;

        case KERNEL_TILED:
            // shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
            check_kernel_occupancy(kernel_ptr, kernel_name,
                                  total_threads_per_block, actual_shared_mem);
            break;

        case KERNEL_TILED_PAIRWISE:
            //shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
            check_kernel_occupancy(kernel_ptr, kernel_name,
                                  total_threads_per_block, actual_shared_mem);
            break;

        case KERNEL_TILED_MIXPREC:
            //shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(COMPUTE_TYPE);
            printf("Mixed precision kernel using COMPUTE_TYPE=%s (%zu bytes)\n",
                   getComputeTypeString(), sizeof(COMPUTE_TYPE));
            check_kernel_occupancy(kernel_ptr, kernel_name,
                                  total_threads_per_block,
                                  actual_shared_mem);
            break;

        case KERNEL_TILED_PAIRWISE_MIXPREC:
            //shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(COMPUTE_TYPE);
            printf("Mixed precision pairwise kernel using COMPUTE_TYPE=%s (%zu bytes)\n",
                   getComputeTypeString(), sizeof(COMPUTE_TYPE));
            check_kernel_occupancy(kernel_ptr, kernel_name,
                                  total_threads_per_block,
                                  actual_shared_mem);
            break;

        case KERNEL_TILED_RECT:
            //shared_mem = 2 * TILE_M * TILE_N * sizeof(float);
            check_kernel_occupancy(kernel_ptr, kernel_name,
                                  total_threads_per_block,
                                  actual_shared_mem);
            break;

        case KERNEL_CUBLAS:
            printf("cuBLAS occupancy analysis not available (library function)\n");
            break;

        default:
            printf("Occupancy analysis not implemented for this kernel type\n");
            break;
    }
}

// Benchmark function using unified kernel dispatch
void runBenchmark(int n, KernelType kernel_type,
                  float* h_A, float* h_B, float* h_C,
                  float* d_A, float* d_B, float* d_C,
                  FILE* dataFile) {

    const char* name = kernelTypeToString(kernel_type);

    // Add validation check
    if (!validate_benchmark_precision_requirements(kernel_type)) {
        return;
    }

    size_t size = n * n * sizeof(float);
    double operations = 2.0 * n * n * n; // 2*N^3 FLOPs for matrix multiplication
    double mem_access_bytes = 3.0 * n * n * sizeof(float); // 3*N^2 floats read/written
    double arithmetic_intensity = operations / mem_access_bytes;

    // Compute dimensions using efficient template-based config function
    dim3 threadsPerBlock, numBlocks;
    compute_kernel_dimensions_dispatch(kernel_type, n, &threadsPerBlock, &numBlocks);

    printf("Debug %s: Grid(%d,%d), Block(%d,%d)\n",
           name, numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup run - SIMPLIFIED
    launch_basic_kernel_by_type(kernel_type, d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
    cudaDeviceSynchronize();

    // Clear result matrix with a known pattern
    cudaMemset(d_C, 0, size);

    // Add checksum BEFORE kernel
    float checksum_before = 0.0f;
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100; i++) checksum_before += h_C[i];
    printf("Debug: Checksum before %s: %.6f\n", name, checksum_before);

    // WARM-UP RUNS - SIMPLIFIED
    printf("  Warming up...");
    for (int i = 0; i < 3; i++) {
        launch_basic_kernel_by_type(kernel_type, d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
    }
    cudaDeviceSynchronize();
    printf(" done\n");

    // MULTIPLE TIMED RUNS
    // Adaptive sampling until you get stable results
    int num_runs = (n < 1024) ? 15 : 10;
    int max_runs = (n < 1024) ? 25 : 15;

    float total_time = 0.0f;
    float* run_times = (float*)malloc(max_runs * sizeof(float));
    if (!run_times) {
        printf("Error: Could not allocate memory for run times\n");
        return;
    }

    for (int run = 0; run < num_runs; run++) {
        cudaEventRecord(start);

        int iterations_per_run = (n < 1024) ? 10 : 3;
        for (int iter = 0; iter < iterations_per_run; iter++) {
            // SIMPLIFIED - only basic launcher
            launch_basic_kernel_by_type(kernel_type, d_A, d_B, d_C, n, numBlocks, threadsPerBlock);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float run_time;
        cudaEventElapsedTime(&run_time, start, stop);
        run_time /= iterations_per_run;  // Average per kernel call

        run_times[run] = run_time;  // Store the time
        total_time += run_time;

        printf("  Run %d: %.2f ms\n", run + 1, run_time);

        // Check stability after minimum runs
        if (run >= (num_runs - 1)) {
            // Calculate coefficient of variation
            float average_time = total_time / (run + 1);
            float variance = 0.0f;
            for (int i = 0; i <= run; i++) {
                float diff = run_times[i] - average_time;
                variance += diff * diff;
            }
            variance /= (run + 1);
            float cv = (sqrtf(variance) / average_time) * 100.0f;

            // If CV < 2% and we have enough samples, stop
            if (cv < 2.0f && run >= (num_runs - 1)) {
                printf("  Converged after %d runs (CV: %.1f%%)\n", run + 1, cv);
                num_runs = run + 1;
                total_time = 0.0f;
                for (int i = 0; i < num_runs; i++) total_time += run_times[i];
                break;
            }

            // If we haven't converged but reached max runs, continue
            if (run < max_runs - 1 && cv >= 2.0f) {
                num_runs = run + 2;  // Add one more run
            }
        }
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

    // Add these calculations before the fprintf:
    float min_time = run_times[0];
    float max_time = run_times[0];
    for (int run = 0; run < num_runs; run++) {
        if (run_times[run] < min_time) min_time = run_times[run];
        if (run_times[run] > max_time) max_time = run_times[run];
    }

    // Get timestamp
    time_t now = time(0);
    struct tm* timeinfo = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H:%M:%S", timeinfo);

    // Get tile size
    int tile_size = 0;
    switch(kernel_type) {
        case KERNEL_TILED:
        case KERNEL_TILED_PAIRWISE:
        case KERNEL_TILED_MIXPREC:
        case KERNEL_TILED_PAIRWISE_MIXPREC:
            tile_size = TILE_SIZE;
            break;
        case KERNEL_TILED_RECT:
            tile_size = TILE_M; // Or format as "MxN"
            break;
        default:
            tile_size = 0;
            break;
    }

    // Replace the simple fprintf with enhanced version:
    fprintf(dataFile, "%s,%d,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%d,%s\n",
            name, n,                              // algorithm, size
            getComputeTypeString(),               // compute_type
            getAccumulateTypeString(),            // accumulate_type
            num_runs,                             // num_runs
            average_time, std_dev, min_time, max_time,  // timing statistics
            gigaFlops, bandwidth_gbps, arithmetic_intensity, // performance
            tile_size,                            // tile_size
            timestamp);                           // timestamp

    printf("Debug timing: average_time = %.6f, operations = %.2e\n", average_time, operations);
}

void initialize_benchmark_matrices(float* h_A, float* h_B, float* h_C, int n) {
    fill_matrix(h_A, n);          // Random values for A
    fill_matrix(h_B, n);          // Random values for B
    memset(h_C, 0, n * n * sizeof(float));  // Consistent zero init for C

    printf("Debug: Matrix initialization complete\n");
    printf("  A[0] = %.6f, B[0] = %.6f, C[0] = %.6f\n", h_A[0], h_B[0], h_C[0]);
}

// Kernel-based benchmark function - takes KernelType directly
void runKernelPerformance(KernelType kernel_type, int n) {
    const char* kernel_name = kernelTypeToString(kernel_type);

    printf("=== Kernel Performance Test ===\n");
    printf("Kernel: %s, Size: %dx%d\n", kernel_name, n, n);

    // Add validation check at the start
    if (!validate_benchmark_precision_requirements(kernel_type)) {
        return;
    }

    printf("Using COMPUTE_TYPE: %s (%zu bytes per element)\n",
           getComputeTypeString(), sizeof(COMPUTE_TYPE));
    printf("Using ACCUMULATE_TYPE: %s (%zu bytes per element)\n",
           getAccumulateTypeString(), sizeof(ACCUMULATE_TYPE));

    // SIMPLIFIED - always use float for benchmarking
    size_t size = n * n * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *d_A, *d_B, *d_C;

    if (!h_A || !h_B || !h_C) {
        printf("ERROR: Failed to allocate host memory\n");
        return;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // SIMPLIFIED - always use float initialization
    initialize_benchmark_matrices(h_A, h_B, h_C, n);

    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    // Create descriptive filename
    char filename[512];

    // Use global folder
    snprintf(filename, sizeof(filename), "%s/perf_%s_%d_%s.csv",
         g_data_folder,
         kernel_name,
         n,
         getComputeTypeString());

    printf("Writing performance data to: %s\n", filename);

    // Open output file with descriptive name
    FILE* dataFile = fopen(filename, "w");
    if (dataFile) {
        fprintf(dataFile, "algorithm,size,compute_type,accumulate_type,num_runs,mean_time_ms,std_dev_ms,min_time_ms,max_time_ms,gflops,bandwidth_gb,arithmetic_intensity,tile_size,timestamp\n");
    } else {
        printf("Warning: Could not open %s for writing\n", filename);
    }

    // Run the benchmark - need to cast back to float* for runBenchmark compatibility
    // OR make runBenchmark templated to handle COMPUTE_TYPE
    runBenchmark(n, kernel_type, h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

    // Check occupancy for the selected kernel
    assess_kernel_resources(kernel_type, n);

    // Clean up
    if (dataFile) fclose(dataFile);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nKernel benchmark complete!\n");
}


// Kernel resource assessment function
void assess_kernel_resources(KernelType kernel_type, int n) {
    printf("\n=== Kernel Resource Assessment ===\n");
    printf("Kernel: %s, Matrix Size: %dx%d\n", kernelTypeToString(kernel_type), n, n);

    // Get kernel attributes (keep this part)
    void* kernel_ptr = get_kernel_function_pointer(kernel_type);

    if (kernel_ptr == nullptr) {
        printf("  Note: Library kernel (cuBLAS/CUTLASS) - attributes not available\n");
        printf("  Cannot retrieve detailed attributes for library kernels\n");
        return;
    }

    cudaFuncAttributes attr = {0};
    cudaError_t err = cudaFuncGetAttributes(&attr, kernel_ptr);

    if (err == cudaSuccess) {
        printf("  Kernel Resource Details:\n");
        printf("    Registers per thread: %d\n", attr.numRegs);
        printf("    Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
        printf("    Max threads per block: %d\n", attr.maxThreadsPerBlock);
        printf("    Constant memory: %zu bytes\n", attr.constSizeBytes);
        printf("    Local memory per thread: %zu bytes\n", attr.localSizeBytes);
    }

    // Get launch configuration
    dim3 threadsPerBlock, numBlocks;
    compute_kernel_dimensions_dispatch(kernel_type, n, &threadsPerBlock, &numBlocks);
    int total_threads_per_block = threadsPerBlock.x * threadsPerBlock.y;

    printf("  Launch configuration:\n");
    printf("    Threads per block: %d x %d = %d\n",
           threadsPerBlock.x, threadsPerBlock.y, total_threads_per_block);
    printf("    Grid size: %d x %d = %d blocks\n",
           numBlocks.x, numBlocks.y, numBlocks.x * numBlocks.y);
    printf("    Total threads: %d\n",
           (numBlocks.x * numBlocks.y) * total_threads_per_block);

    // REUSE your occupancy function instead of duplicating code
    printf("  Occupancy Analysis:\n");
    if (err == cudaSuccess) {
        check_kernel_occupancy(kernel_ptr, kernelTypeToString(kernel_type),
                              total_threads_per_block, attr.sharedSizeBytes);

        // Add the limiting factors analysis (this is the extra value)
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        int max_blocks_by_threads = prop.maxThreadsPerMultiProcessor / total_threads_per_block;
        int max_blocks_by_sm_limit = prop.maxBlocksPerMultiProcessor;

        printf("    Limiting factors:\n");
        printf("      Max blocks by thread limit: %d\n", max_blocks_by_threads);
        printf("      Max blocks by SM limit: %d\n", max_blocks_by_sm_limit);
        if (attr.sharedSizeBytes > 0) {
            int max_blocks_by_shared_mem = prop.sharedMemPerMultiprocessor / attr.sharedSizeBytes;
            printf("      Max blocks by shared memory: %d\n", max_blocks_by_shared_mem);
        }
    }

    printf("===================================\n\n");
}

// Add this function to the top of benchmark.cu (after the includes):

bool validate_benchmark_precision_requirements(KernelType kernel_type) {
    // For mixed precision kernels, we need FP32 types for benchmark compatibility
    if (is_mixprec_kernel(kernel_type) && !areBothTypesFP32()) {
        printf("ERROR: Benchmark currently only supports FP32 types.\n");
        printf("Mixed precision kernel %s requires COMPUTE_TYPE=float and ACCUMULATE_TYPE=float for benchmarking.\n",
               kernelTypeToString(kernel_type));
        printf("Please recompile with: make COMPUTE_TYPE=float ACCUMULATE_TYPE=float\n");
        return false;
    }

    // For non-mixprec kernels, they should always work with FP32
    if (!is_mixprec_kernel(kernel_type) && !areBothTypesFP32()) {
        printf("ERROR: Non-mixprec kernel %s requires COMPUTE_TYPE=float and ACCUMULATE_TYPE=float.\n",
               kernelTypeToString(kernel_type));
        printf("Please recompile with: make COMPUTE_TYPE=float ACCUMULATE_TYPE=float\n");
        return false;
    }

    return true;
}