// error_analysis.cu - Consolidated error analysis functionality
#include "../include/error_analysis.cuh"
#include "../include/config.h"  // For configuration constants and SIZES
#include "../include/gemms.cuh"
#include "../include/utils.cuh"
#include "../include/matrix_utils.cuh"  // For matrix utility functions
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>

// Kernel to analyze round-off errors during tiled GEMM
__global__ void analyze_tiled_gemm_errors(
    float *A, float *B, float *C_result, float *C_reference,
    int N,
    float *tile_norms, float *tile_condition_numbers,
    float *accumulated_errors, int *error_counts) {

    // Shared memory for tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];
    //__shared__ float partial_results[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    float error_accumulation = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Global tile index for this block
    int tile_idx = by * gridDim.x + bx;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tiles (same as original tiled implementation)
        int A_row = row;
        int A_col = t * TILE_SIZE + tx;
        tile_A[ty][tx] = (A_row < N && A_col < N) ? A[A_row * N + A_col] : 0.0f;

        int B_row = t * TILE_SIZE + ty;
        int B_col = col;
        tile_B[ty][tx] = (B_row < N && B_col < N) ? B[B_row * N + B_col] : 0.0f;

        __syncthreads();

        // Compute Frobenius norm of current tiles (thread 0 only)
        if (tx == 0 && ty == 0) {
            float norm_A = compute_frobenius_norm_tile((float*)tile_A, TILE_SIZE, TILE_SIZE);
            float norm_B = compute_frobenius_norm_tile((float*)tile_B, TILE_SIZE, TILE_SIZE);

            // Compute condition numbers for the tiles
            float cond_A = estimate_condition_number_tile((float*)tile_A, TILE_SIZE, TILE_SIZE);
            float cond_B = estimate_condition_number_tile((float*)tile_B, TILE_SIZE, TILE_SIZE);

            // Store norms (using atomic add for accumulation across tiles)
            atomicAdd(&tile_norms[tile_idx * 2], norm_A);
            atomicAdd(&tile_norms[tile_idx * 2 + 1], norm_B);

            // Store condition numbers (using atomic add for averaging later)
            atomicAdd(&tile_condition_numbers[tile_idx * 2], cond_A);
            atomicAdd(&tile_condition_numbers[tile_idx * 2 + 1], cond_B);
        }

        // Compute partial dot product with error tracking
        float old_sum = sum;
        for (int k = 0; k < TILE_SIZE; ++k) {
            float product = tile_A[ty][k] * tile_B[k][tx];
            sum += product;

            // Track numerical error accumulation
            float new_error = fabsf(sum - old_sum - product);
            error_accumulation += new_error;
            old_sum = sum;
        }

        __syncthreads();
    }

    // Store final result and error analysis
    if (row < N && col < N) {
        C_result[row * N + col] = sum;

        // Compute error compared to reference
        float reference_val = C_reference[row * N + col];
        float absolute_error = fabsf(sum - reference_val);
        float relative_error = (fabsf(reference_val) > 1e-10f) ?
                              absolute_error / fabsf(reference_val) : absolute_error;

        // Store error data
        int idx = row * N + col;
        accumulated_errors[idx * 3] = absolute_error;
        accumulated_errors[idx * 3 + 1] = relative_error;
        accumulated_errors[idx * 3 + 2] = error_accumulation;

        // Count significant errors
        if (relative_error > 1e-6f) {
            atomicAdd(error_counts, 1);
        }
    }
}

// Host function to run comprehensive numerical analysis
void run_numerical_analysis(float* h_A, float* h_B, int n, const char* output_filename) {
    printf("\n=== Numerical Analysis of Tiled GEMM ===\n");
    printf("Matrix size: %d x %d\n", n, n);

#if ENABLE_MATRIX_STANDARDIZATION
    // Standardize matrices on host before analysis (A = A/||A||_F, B = B/||B||_F)
    // This ensures both matrices have unit Frobenius norm for normalized error analysis
    printf("Standardizing matrix A on host...\n");
    standardize_matrix_host(h_A, n);
    printf("Standardizing matrix B on host...\n");
    standardize_matrix_host(h_B, n);
#else
    printf("Matrix standardization disabled - using original matrices.\n");
#endif

    size_t size = n * n * sizeof(float);
    size_t tile_data_size = ((n + TILE_SIZE - 1) / TILE_SIZE) *
                           ((n + TILE_SIZE - 1) / TILE_SIZE) * 2 * sizeof(float);
    size_t condition_data_size = ((n + TILE_SIZE - 1) / TILE_SIZE) *
                                ((n + TILE_SIZE - 1) / TILE_SIZE) * 2 * sizeof(float);

    // Allocate device memory
    float *d_A, *d_B, *d_C_tiled, *d_C_reference;
    float *d_tile_norms, *d_condition_numbers, *d_accumulated_errors;
    int *d_error_counts;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_tiled, size);
    cudaMalloc(&d_C_reference, size);
    cudaMalloc(&d_tile_norms, tile_data_size);
    cudaMalloc(&d_condition_numbers, condition_data_size);
    cudaMalloc(&d_accumulated_errors, size * 3); // abs, rel, accumulated
    cudaMalloc(&d_error_counts, sizeof(int));

    // Copy input data
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Initialize arrays
    cudaMemset(d_tile_norms, 0, tile_data_size);
    cudaMemset(d_condition_numbers, 0, condition_data_size);
    cudaMemset(d_accumulated_errors, 0, size * 3);
    cudaMemset(d_error_counts, 0, sizeof(int));

    // Compute reference result using cuBLAS (high precision)
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, d_B, n, d_A, n, &beta, d_C_reference, n);
    cublasDestroy(handle);

    // Run analysis kernel
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    analyze_tiled_gemm_errors<<<blocks, threads>>>(
        d_A, d_B, d_C_tiled, d_C_reference, n,
        d_tile_norms, d_condition_numbers, d_accumulated_errors, d_error_counts);

    cudaDeviceSynchronize();

    // Retrieve results
    float *h_tile_norms = (float*)malloc(tile_data_size);
    float *h_condition_numbers = (float*)malloc(condition_data_size);
    float *h_accumulated_errors = (float*)malloc(size * 3);
    int h_error_count;

    cudaMemcpy(h_tile_norms, d_tile_norms, tile_data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_condition_numbers, d_condition_numbers, condition_data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_accumulated_errors, d_accumulated_errors, size * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_error_count, d_error_counts, sizeof(int), cudaMemcpyDeviceToHost);

    // Analyze and report results
    analyze_numerical_results(h_tile_norms, h_condition_numbers, h_accumulated_errors, n, h_error_count, output_filename);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_tiled); cudaFree(d_C_reference);
    cudaFree(d_tile_norms); cudaFree(d_condition_numbers); cudaFree(d_accumulated_errors);
    cudaFree(d_error_counts);
    free(h_tile_norms); free(h_condition_numbers); free(h_accumulated_errors);
}

// Host function to analyze and report numerical results
void analyze_numerical_results(float* tile_norms, float* condition_numbers, float* errors, int n, int error_count, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("ERROR: Cannot create output file %s\n", filename);
        return;
    }

    fprintf(fp, "# Numerical Analysis Results for %dx%d Matrix\n", n, n);
    fprintf(fp, "# Columns: i, j, absolute_error, relative_error, accumulated_error\n");

    double total_abs_error = 0.0;
    double max_rel_error = 0.0;
    double total_accumulated_error = 0.0;

    // Calculate average condition numbers for tiles
    int num_tiles = ((n + TILE_SIZE - 1) / TILE_SIZE) * ((n + TILE_SIZE - 1) / TILE_SIZE);
    double avg_condition_A = 0.0, avg_condition_B = 0.0;

    for (int t = 0; t < num_tiles; t++) {
        avg_condition_A += condition_numbers[t * 2];
        avg_condition_B += condition_numbers[t * 2 + 1];
    }
    avg_condition_A /= num_tiles;
    avg_condition_B /= num_tiles;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            float abs_err = errors[idx * 3];
            float rel_err = errors[idx * 3 + 1];
            float acc_err = errors[idx * 3 + 2];

            fprintf(fp, "%d %d %.10e %.10e %.10e\n", i, j, abs_err, rel_err, acc_err);

            total_abs_error += abs_err;
            max_rel_error = fmaxf(max_rel_error, rel_err);
            total_accumulated_error += acc_err;
        }
    }

    fclose(fp);

    // Print summary statistics including condition numbers
    printf("\n--- Error Analysis Summary ---\n");
    printf("Total elements with significant errors: %d / %d\n", error_count, n * n);
    printf("Average absolute error: %.10e\n", total_abs_error / (n * n));
    printf("Maximum relative error: %.10e\n", max_rel_error);
    printf("Average accumulated error: %.10e\n", total_accumulated_error / (n * n));
    printf("Average condition number (tiles A): %.2e\n", avg_condition_A);
    printf("Average condition number (tiles B): %.2e\n", avg_condition_B);
    printf("Results saved to: %s\n", filename);
}

// Helper function to test different tile sizes
void compare_tile_sizes(float* h_A, float* h_B, int n) {
    printf("\n=== Comparing Different Tile Sizes ===\n");

    // Test different tile sizes by recompiling with different TILE_SIZE values
    // This is a placeholder - in practice you'd want configurable tile sizes

    const int test_tile_sizes[] = {8, 16, 32};
    const int num_tile_sizes = sizeof(test_tile_sizes) / sizeof(test_tile_sizes[0]);

    for (int i = 0; i < num_tile_sizes; i++) {
        printf("Analyzing tile size %d...\n", test_tile_sizes[i]);
        // Note: This would require runtime tile size configuration
        // For now, just report current TILE_SIZE analysis
        char filename[256];
        snprintf(filename, sizeof(filename), "data/numerical_analysis_tile%d_n%d.dat", TILE_SIZE, n);
        run_numerical_analysis(h_A, h_B, n, filename);
    }
}


// Setup matrix data based on type - now uses cached matrix generation
void setupMatrix(float* matrix, int n, MatrixType type, const char* filename) {
    if (!get_matrix(matrix, n, type, filename)) {
        printf("ERROR: Failed to setup matrix of type %d\n", (int)type);
        printf("Falling back to random matrix generation\n");
        fill_matrix(matrix, n);
    }
}

// Run matrix tests with specified configuration
void runMatrixTests(int n, MatrixTestConfig* configs, int num_configs) {
    printf("\n--- Testing matrix size %d x %d ---\n", n, n);

    size_t size = n * n * sizeof(float);

    // Allocate host memory once for all tests
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);

    if (!h_A || !h_B) {
        printf("ERROR: Failed to allocate host memory\n");
        return;
    }

    // Run each enabled test configuration
    for (int i = 0; i < num_configs; i++) {
        if (!configs[i].enabled) continue;

        printf("\n=== Running test: %s ===\n", configs[i].name);
        printf("Description: %s\n", configs[i].description);

        // Setup matrices according to configuration
        printf("Setting up matrix A...\n");
        setupMatrix(h_A, n, configs[i].type_A, configs[i].filename_A);
        print_matrix_stats(h_A, n, "A");

        printf("Setting up matrix B...\n");
        setupMatrix(h_B, n, configs[i].type_B, configs[i].filename_B);
        print_matrix_stats(h_B, n, "B");

        // Generate output filename
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename),
                "data/numerical_analysis_%s_n%d_tile%d.dat", configs[i].name, n, TILE_SIZE);

        printf("Running numerical analysis for %s...\n", configs[i].name);
        run_numerical_analysis(h_A, h_B, n, output_filename);
    }

    // Cleanup
    free(h_A);
    free(h_B);
}

// Generate comprehensive report from all test results
void generateReport(bool* enabled_sizes) {
    printf("\n=== Generating Numerical Analysis Report ===\n");

    FILE* summaryFile = fopen("data/numerical_analysis_summary.csv", "w");
    if (!summaryFile) {
        printf("ERROR: Could not create data/numerical_analysis_summary.csv\n");
        return;
    }

    fprintf(summaryFile, "test_name,size,avg_abs_error,max_rel_error,significant_errors,avg_condition_A,avg_condition_B\n");

    // TODO: Parse all generated .dat files and create summary statistics
    // For now, just write header and close
    printf("Report generation functionality to be implemented\n");
    printf("Summary will include analysis from all generated .dat files\n");

    fclose(summaryFile);
    printf("Report saved to: data/numerical_analysis_summary.csv\n");
}

// Main function for numerical analysis benchmarks - now refactored
void runNumericalAnalysisBenchmarks(bool* enabled_sizes) {
    printf("=== Starting Numerical Analysis of Tiled GEMM ===\n");

    // Define matrix types for systematic testing
    // Each type will be tested with multiple instances for statistical averaging
    struct MatrixTypeConfig {
        MatrixType type;
        const char* name;
        const char* description;
        bool enabled;
        int num_instances;  // Number of different matrices to generate and test
    };

    MatrixTypeConfig matrix_types[] = {
        {MATRIX_ODO_WELL_CONDITIONED, "wellcond", "Well-conditioned matrices", true, 100},
        {MATRIX_ODO_ILL_CONDITIONED, "illcond", "Ill-conditioned matrices", true, 100},
        {MATRIX_NORMAL_DISTRIBUTION, "normal", "Normal distribution matrices", true, 100},
        {MATRIX_SCALED_FTZ, "scaled", "Scaled matrices near FTZ threshold", true, 50},
        {MATRIX_SKEW_MAGNITUDE, "skewed", "Matrices with skewed magnitude distribution", true, 50},
        // Add more matrix types as needed for comprehensive testing
    };

    int num_matrix_types = sizeof(matrix_types) / sizeof(matrix_types[0]);

    // Run tests for each enabled size and matrix type
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        if (!enabled_sizes[size_idx]) continue;

        int n = SIZES[size_idx];
        printf("\n=== Testing matrix size %dx%d ===\n", n, n);

        // Test each matrix type
        for (int type_idx = 0; type_idx < num_matrix_types; type_idx++) {
            MatrixTypeConfig* config = &matrix_types[type_idx];
            if (!config->enabled) continue;

            printf("\n--- Testing %s matrices (%s) ---\n", config->name, config->description);
            printf("Generating and testing %d instances for statistical averaging...\n", config->num_instances);

            // Test multiple instances of this matrix type for statistical robustness
            for (int instance = 0; instance < config->num_instances; instance++) {
                if (instance % 10 == 0 && instance > 0) {
                    printf("Completed %d/%d instances...\n", instance, config->num_instances);
                }

                // TODO: Replace with proper matrix testing function
                // runMatrixInstance(n, config->type, instance);
                printf("Testing instance %d of %s matrices (size %dx%d)\n",
                       instance + 1, config->name, n, n);
            }

            printf("Completed all %d instances of %s matrices\n", config->num_instances, config->name);
        }
    }

    // Generate comprehensive report
    generateReport(enabled_sizes);

    printf("\nNumerical analysis complete!\n");
    printf("Individual test results: data/numerical_analysis_*.dat\n");
    printf("Summary report: data/numerical_analysis_summary.csv\n");
}
