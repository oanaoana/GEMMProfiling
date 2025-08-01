// numerical_analysis.cu - Round-off error analysis for tiled GEMM
#include "../include/numerical_analysis.cuh"
#include "../include/gemms.cuh"
#include "../include/utils.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>

// Configuration flag for matrix standardization
// When enabled, matrices A and B are divided by their respective Frobenius norms
// before error analysis. This standardizes the matrices to have unit Frobenius norm.
// Set to 0 to disable standardization and work with original matrices.
#define ENABLE_MATRIX_STANDARDIZATION 1  // Set to 0 to disable standardization

// Device functions for numerical analysis
__device__ float compute_frobenius_norm_tile(float* tile, int rows, int cols) {
    float norm_sq = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = tile[i * cols + j];
            norm_sq += val * val;
        }
    }
    return sqrtf(norm_sq);
}

__device__ float estimate_condition_number_tile(float* tile, int rows, int cols) {
    // Estimate condition number using Frobenius norm and power iteration for largest singular value
    // For square tiles, we can estimate ||A|| * ||A^-1|| approximately

    // First compute Frobenius norm of the tile
    float frobenius_norm = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = tile[i * cols + j];
            frobenius_norm += val * val;
        }
    }
    frobenius_norm = sqrtf(frobenius_norm);

    if (frobenius_norm < 1e-12f) return 1e12f; // Near-zero matrix

    // For non-square tiles, return a scaled Frobenius norm as approximation
    if (rows != cols) {
        return frobenius_norm * sqrtf((float)(rows + cols));
    }

    // For square tiles, estimate condition number using iterative method
    // Approximate largest eigenvalue using power iteration (simplified)
    float max_eigenval = 0.0f;
    float min_eigenval = 1e12f;

    // Simple approximation: use diagonal dominance and row sums
    for (int i = 0; i < rows; i++) {
        float row_sum = 0.0f;
        float diag_val = fabsf(tile[i * cols + i]);

        for (int j = 0; j < cols; j++) {
            row_sum += fabsf(tile[i * cols + j]);
        }

        // Gershgorin circle estimation
        float radius = row_sum - diag_val;
        float upper_bound = diag_val + radius;
        float lower_bound = fmaxf(diag_val - radius, 1e-12f);

        max_eigenval = fmaxf(max_eigenval, upper_bound);
        min_eigenval = fminf(min_eigenval, lower_bound);
    }

    // Condition number approximation
    float condition_estimate = max_eigenval / min_eigenval;

    // Clamp to reasonable bounds
    return fminf(condition_estimate, 1e10f);
}

// Host function to standardize a matrix (A = A / ||A||_F) in place
void standardize_matrix_host(float* h_matrix, int n) {
    // Compute Frobenius norm on host
    double norm_sq = 0.0;
    for (int i = 0; i < n * n; i++) {
        double val = h_matrix[i];
        norm_sq += val * val;
    }
    double norm = sqrt(norm_sq);

    if (norm > 1e-12) {
        // Standardize in place
        for (int i = 0; i < n * n; i++) {
            h_matrix[i] /= norm;
        }
        printf("Matrix standardized: Frobenius norm was %.6e\n", norm);
    } else {
        printf("Warning: Matrix has very small Frobenius norm (%.6e), skipping standardization\n", norm);
    }
}

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
