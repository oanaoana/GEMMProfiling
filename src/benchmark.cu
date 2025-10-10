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
#ifdef __NVCC__
                   // Simple type name detection for common types
                   (sizeof(COMPUTE_TYPE) == 4) ? "float" :
                   (sizeof(COMPUTE_TYPE) == 2) ? "half/bf16" : "unknown",
#else
                   "configured",
#endif
                   sizeof(COMPUTE_TYPE));
            check_kernel_occupancy(kernel_ptr, kernel_name,
                                  total_threads_per_block,
                                  actual_shared_mem);
            break;

        case KERNEL_TILED_PAIRWISE_MIXPREC:
            //shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(COMPUTE_TYPE);
            printf("Mixed precision pairwise kernel using COMPUTE_TYPE=%s (%zu bytes)\n",
#ifdef __NVCC__
                   (sizeof(COMPUTE_TYPE) == 4) ? "float" :
                   (sizeof(COMPUTE_TYPE) == 2) ? "half/bf16" : "unknown",
#else
                   "configured",
#endif
                   sizeof(COMPUTE_TYPE));
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

    const char* name = kernelTypeToString(kernel_type);  // Get name from enum

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

            runBenchmark(n, kernel_type,
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

void initialize_benchmark_matrices(float* h_A, float* h_B, float* h_C, int n) {
    fill_matrix(h_A, n);          // Random values for A
    fill_matrix(h_B, n);          // Random values for B
    memset(h_C, 0, n * n * sizeof(float));  // Consistent zero init for C

    printf("Debug: Matrix initialization complete\n");
    printf("  A[0] = %.6f, B[0] = %.6f, C[0] = %.6f\n", h_A[0], h_B[0], h_C[0]);
}

// Kernel-based benchmark function - takes KernelType directly
void runKernelBenchmark(KernelType kernel_type, int n) {
    const char* kernel_name = kernelTypeToString(kernel_type);

    printf("=== Kernel Benchmark Test ===\n");
    printf("Kernel: %s, Size: %dx%d\n", kernel_name, n, n);

    // Allocate memory
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

    // Initialize matrices
    initialize_benchmark_matrices(h_A, h_B, h_C, n);

    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    // Open output file
    FILE* dataFile = fopen("data/roofline_data.csv", "w");
    if (dataFile) {
        fprintf(dataFile, "algorithm,size,time_ms,gflops,bandwidth_gb,arithmetic_intensity\n");
    }

    // Run the benchmark using the kernel type directly
    runBenchmark(n, kernel_type, h_A, h_B, h_C, d_A, d_B, d_C, dataFile);

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

    cudaFuncAttributes attr = {0}; // Initialize to zero
    cudaError_t err = cudaErrorInvalidDeviceFunction; // Default to error state

    // Note: For CUTLASS kernels, we can't easily get function pointers since they're template-generated
    // This function will focus on the kernels we can assess directly
    switch(kernel_type) {
        case KERNEL_NAIVE:
            err = cudaFuncGetAttributes(&attr, matmul_naive);
            if (err != cudaSuccess) {
                printf("  Error getting naive kernel attributes: %s\n", cudaGetErrorString(err));
            }
            break;

        case KERNEL_TILED:
            err = cudaFuncGetAttributes(&attr, matmul_tiled);
            if (err != cudaSuccess) {
                printf("  Error getting tiled kernel attributes: %s\n", cudaGetErrorString(err));
            }
            break;

        case KERNEL_TILED_OPT:
            err = cudaFuncGetAttributes(&attr, matmul_tiled_opt);
            if (err != cudaSuccess) {
                printf("  Error getting tiled optimized kernel attributes: %s\n", cudaGetErrorString(err));
            }
            break;

        case KERNEL_TILED_PAIRWISE:
            err = cudaFuncGetAttributes(&attr, matmul_tiled_pairwise);
            if (err != cudaSuccess) {
                printf("  Error getting tiled pairwise kernel attributes: %s\n", cudaGetErrorString(err));
            }
            break;

        case KERNEL_TILED_RECT:
            err = cudaFuncGetAttributes(&attr, matmul_tiled_rectangular);
            if (err != cudaSuccess) {
                printf("  Error getting tiled rectangular kernel attributes: %s\n", cudaGetErrorString(err));
            }
            break;

        case KERNEL_CUTLASS_SPLITK_FLAT:
            printf("  Note: CUTLASS Split-K Flat uses template-generated kernels\n");
            printf("  Resource usage depends on CUTLASS template instantiation\n");
            printf("  Cannot retrieve detailed attributes for template kernels\n");
            break;

        case KERNEL_CUTLASS_SPLITK_PAIRWISE:
            printf("  Note: CUTLASS Split-K Pairwise uses template-generated kernels\n");
            printf("  Resource usage depends on CUTLASS template instantiation\n");
            printf("  Cannot retrieve detailed attributes for template kernels\n");
            break;

        case KERNEL_CUBLAS:
        case KERNEL_CUBLAS_TENSOR:
            printf("  Note: cuBLAS kernels are proprietary and cannot be assessed\n");
            break;

        case KERNEL_CUTLASS:
        case KERNEL_CUTLASS_TENSOR:
            printf("  Note: CUTLASS kernels use template-generated code\n");
            printf("  Resource usage depends on CUTLASS template instantiation\n");
            printf("  Cannot retrieve detailed attributes for template kernels\n");
            break;

        case KERNEL_TILED_MIXPREC:
            printf("  Note: Mixed precision kernel uses compile-time type configuration\n");
            printf("  Current types: COMPUTE_TYPE=%s, ACCUMULATE_TYPE=%s\n",
                   // You might want to add type name macros to your config
                   "configured at build time", "configured at build time");
            // Can't get attributes for template kernel easily
            break;

        default:
            printf("  Unknown kernel type\n");
            break;
    }

    // Only show kernel attributes if we successfully retrieved them
    if (err == cudaSuccess) {
        printf("  Kernel Resource Details:\n");
        printf("    Registers per thread: %d\n", attr.numRegs);
        printf("    Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
        printf("    Max threads per block: %d\n", attr.maxThreadsPerBlock);
        printf("    Constant memory: %zu bytes\n", attr.constSizeBytes);
        printf("    Local memory per thread: %zu bytes\n", attr.localSizeBytes);
    }

    // Calculate theoretical occupancy for this kernel
    dim3 threadsPerBlock, numBlocks;
    compute_kernel_dimensions_dispatch(kernel_type, n, &threadsPerBlock, &numBlocks);

    printf("  Launch configuration:\n");
    printf("    Threads per block: %d x %d = %d\n",
           threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.x * threadsPerBlock.y);
    printf("    Grid size: %d x %d = %d blocks\n",
           numBlocks.x, numBlocks.y, numBlocks.x * numBlocks.y);
    printf("    Total threads: %d\n",
           (numBlocks.x * numBlocks.y) * (threadsPerBlock.x * threadsPerBlock.y));

    // Calculate and report occupancy for native CUDA kernels
    if (err == cudaSuccess) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        int maxActiveBlocks = 0;
        int threadsPerBlockTotal = threadsPerBlock.x * threadsPerBlock.y;
        size_t sharedMemPerBlock = attr.sharedSizeBytes;

        // Calculate occupancy based on the specific kernel
        void* kernel_ptr = nullptr;
        switch(kernel_type) {
            case KERNEL_NAIVE:
                kernel_ptr = (void*)matmul_naive;
                break;
            case KERNEL_TILED:
                kernel_ptr = (void*)matmul_tiled;
                break;
            case KERNEL_TILED_OPT:
                kernel_ptr = (void*)matmul_tiled_opt;
                break;
            case KERNEL_TILED_PAIRWISE:
                kernel_ptr = (void*)matmul_tiled_pairwise;
                break;
            case KERNEL_TILED_RECT:
                kernel_ptr = (void*)matmul_tiled_rectangular;
                break;
            default:
                kernel_ptr = nullptr;
                break;
        }

        if (kernel_ptr != nullptr) {
            cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxActiveBlocks, kernel_ptr, threadsPerBlockTotal, sharedMemPerBlock);

            if (occ_err == cudaSuccess) {
                int activeThreadsPerSM = maxActiveBlocks * threadsPerBlockTotal;
                double occupancy_percent = (activeThreadsPerSM * 100.0) / prop.maxThreadsPerMultiProcessor;

                printf("  Occupancy Analysis:\n");
                printf("    Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
                printf("    Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
                printf("    Active blocks per SM: %d\n", maxActiveBlocks);
                printf("    Active threads per SM: %d\n", activeThreadsPerSM);
                printf("    Theoretical occupancy: %.1f%%\n", occupancy_percent);

                // Calculate limiting factors
                int max_blocks_by_threads = prop.maxThreadsPerMultiProcessor / threadsPerBlockTotal;
                int max_blocks_by_sm_limit = prop.maxBlocksPerMultiProcessor;

                printf("    Limiting factors:\n");
                printf("      Max blocks by thread limit: %d\n", max_blocks_by_threads);
                printf("      Max blocks by SM limit: %d\n", max_blocks_by_sm_limit);
                if (sharedMemPerBlock > 0) {
                    int max_blocks_by_shared_mem = prop.sharedMemPerMultiprocessor / sharedMemPerBlock;
                    printf("      Max blocks by shared memory: %d\n", max_blocks_by_shared_mem);
                }
            } else {
                printf("  Occupancy Analysis: Error calculating occupancy: %s\n", cudaGetErrorString(occ_err));
            }
        }
    }

    printf("===================================\n\n");
}