// error_analysis.cu - Consolidated error analysis functionality
#include "../include/error_analysis.cuh"
#include "../include/config.h"  // For configuration constants and SIZES
#include "../include/generate_test_matrix.cuh"  // For get_matrix and print_matrix_stats
#include "../include/gemms.cuh"
#include "../include/utils.cuh"
#include "../include/matrix_utils.cuh"  // For matrix utility functions
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Device function for atomic add with double precision
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// GPU kernels for device-only error computation
__global__ void compute_matrix_abs_kernel(float* matrix, float* abs_matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        abs_matrix[idx] = fabsf(matrix[idx]);
    }
}

__global__ void compute_matrix_abs_fp64_kernel(float* matrix, double* abs_matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        abs_matrix[idx] = fabs((double)matrix[idx]);
    }
}

__global__ void compute_frobenius_error_kernel(float* C_kernel, double* C_reference,
                                             double* abs_AB_product, double* error_results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n;

    // Dynamic shared memory for flexible block sizes
    extern __shared__ double shared_mem[];
    double* shared_error = shared_mem;
    double* shared_norm = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    double local_error = 0.0;
    double local_norm = 0.0;

    // Each thread processes multiple elements if needed
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        double diff = (double)C_kernel[i] - C_reference[i];  // FP64 difference
        local_error += diff * diff;

        // Use the precomputed |A|*|B| product
        double norm_val = (double)abs_AB_product[i];
        local_norm += norm_val * norm_val;
    }

    shared_error[tid] = local_error;
    shared_norm[tid] = local_norm;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_error[tid] += shared_error[tid + stride];
            shared_norm[tid] += shared_norm[tid + stride];
        }
        __syncthreads();
    }

    // Write results
    if (tid == 0) {
        atomicAddDouble(&error_results[0], shared_error[0]);  // Frobenius error squared
        atomicAddDouble(&error_results[1], shared_norm[0]);   // Norm squared
    }
}

__global__ void compute_reference_fp64_device(float* A, float* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += (double)A[row * n + k] * (double)B[k * n + col];
        }
        C[row * n + col] = sum;
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
    printf("Using original matrices without standardization.\n");

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

    // Run tests for each enabled size
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        if (!enabled_sizes[size_idx]) continue;

        int n = SIZES[size_idx];
        printf("\n=== Testing matrix size %dx%d ===\n", n, n);

        // Test each matrix type using the working runMatrixTests function
        MatrixTestConfig configs[] = {
            {MATRIX_ODO_WELL_CONDITIONED, MATRIX_ODO_WELL_CONDITIONED, "wellcond", "Well-conditioned matrices", NULL, NULL, true},
            {MATRIX_ODO_ILL_CONDITIONED, MATRIX_ODO_ILL_CONDITIONED, "illcond", "Ill-conditioned matrices", NULL, NULL, true},
            {MATRIX_ZEROMEAN, MATRIX_ZEROMEAN, "zeromean", "Zero-mean distribution matrices", NULL, NULL, true},
            {MATRIX_UNIFORM_POSITIVE, MATRIX_UNIFORM_POSITIVE, "uniform_positive", "Uniform positive matrices", NULL, NULL, true},
            {MATRIX_RADEMACHER, MATRIX_RADEMACHER, "rademacher", "Rademacher distribution matrices", NULL, NULL, true}
        };

        int num_configs = sizeof(configs) / sizeof(configs[0]);
        runMatrixTests(n, configs, num_configs);
    }

    // Generate comprehensive report
    generateReport(enabled_sizes);

    printf("\nNumerical analysis complete!\n");
    printf("Individual test results: data/numerical_analysis_*.dat\n");
    printf("Summary report: data/numerical_analysis_summary.csv\n");
}


inline double gamma(int n, double u) {
    const double nu = n * u;
    return nu / (1.0 - nu);
}

inline int ceil_log2_int(int x) {
    int p = 0, v = x - 1;
    while (v > 0) { v >>= 1; ++p; }
    return p;
}

// Compute theoretical error bound factor based on kernel type and matrix size
float compute_beta_factor(KernelType kernel_type, bool single_pass, int n) {
    const double u = unit_roundoff_fp32(); // Use precision from config

    if (single_pass) {
        // For naive kernels: single-pass accumulation over n elements
        return (float)gamma(n, u);
    }

    // For tiled kernels: two-stage accumulation
    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    int tile_inner_k = TILE_SIZE; // Inner loop accumulation size

    // Stage 1: Accumulation within each tile (size TILE_SIZE)
    double beta_inner = gamma(tile_inner_k, u);

    // Stage 2: Accumulation across tiles
    bool pairwise = (kernel_type == KERNEL_TILED_PAIRWISE);
    double beta_outer;

    if (pairwise) {
        // Pairwise summation has logarithmic error growth
        beta_outer = gamma(ceil_log2_int(num_tiles), u);
    } else {
        // Standard summation has linear error growth
        beta_outer = gamma(num_tiles, u);
    }

    // Total error bound: inner + outer (conservative, cross-terms are O(u^2))
    return (float)(beta_inner + beta_outer);
}

// Integer ULP distance between two FP32 values.
__device__ __forceinline__ int32_t ord(float x) {
    int32_t i = __float_as_int(x);
    return (i < 0) ? 0x80000000 - i : i + 0x80000000;
}

__device__ __forceinline__ uint32_t ulp_distance(float a, float b) {
    int32_t da = ord(a), db = ord(b);
    int32_t d  = da - db;
    return (d < 0) ? uint32_t(-d) : uint32_t(d);
}

// Size of one ULP at x (FP32)
__device__ __forceinline__ float ulp_of(float x) {
    if (!isfinite(x)) return NAN;
    x = fabsf(x);
    if (x == 0.0f) return ldexpf(1.0f, -149);  // subnormal spacing
    uint32_t u = __float_as_uint(x);
    uint32_t exp = (u >> 23) & 0xFFu;
    if (exp == 0u) return ldexpf(1.0f, -149);
    int e = int(exp) - 127;
    return ldexpf(1.0f, e - 23);               // 2^(e-23)
}

__global__ void ulp_metrics_kernel(const float* __restrict__ Ctest,
                                   const float* __restrict__ Cref,
                                   uint32_t* __restrict__ dULP,   // out: integer ULP distance
                                   float* __restrict__ errULP,    // out: scaled err (may be nullptr)
                                   unsigned long long* __restrict__ invalid_count, // NaN/Inf pairs
                                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ct = Ctest[i];
    float cr = Cref[i];

    if (!isfinite(ct) || !isfinite(cr)) {
        dULP[i] = 0;
        if (errULP) errULP[i] = NAN;
        atomicAdd(invalid_count, 1ull);
        return;
    }

    dULP[i] = ulp_distance(ct, cr);

    if (errULP) {
        float ulp = ulp_of(cr);
        float e = fabsf(ct - cr) / ulp;   // uses spacing at the reference
        errULP[i] = e;
    }
}

// Efficient multi-sample testing for specific matrix type and kernel
void run_multi_sample_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int num_samples, const char* output_prefix) {
    printf("\n=== Multi-Sample Analysis ===\n");
    printf("Matrix Type: %d, Kernel: %d, Size: %dx%d, Samples: %d\n",
           (int)matrix_type, (int)kernel_type, n, n, num_samples);

    // Compute theoretical error bound factor
    // Use single_pass=false for tiled_pairwise kernel, single_pass=true for the rest
    bool single_pass = true;
    if (kernel_type == KERNEL_TILED_PAIRWISE) single_pass=false;
    float beta_factor = compute_beta_factor(kernel_type, single_pass, n);
    // Allocate device memory (reused across all samples)
    size_t size = n * n * sizeof(float);
    float *d_A, *d_B, *d_C_kernel;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_kernel, size);

    // Device memory for optimized error computation (no host transfers needed)
    double *d_C_reference_fp64;  // FP64 reference result
    double *d_A_abs, *d_B_abs;   // Absolute value matrices for norm computation (FP64)
    double *d_abs_AB_product;    // Product |A| * |B| (FP64)
    double *d_error_results;     // Error computation results [frobenius², norm²]

    size_t size_fp64 = n * n * sizeof(double);
    cudaMalloc(&d_C_reference_fp64, size_fp64);
    cudaMalloc(&d_A_abs, size_fp64);
    cudaMalloc(&d_B_abs, size_fp64);
    cudaMalloc(&d_abs_AB_product, size_fp64);  // For |A| * |B| product (FP64)
    cudaMalloc(&d_error_results, 2 * sizeof(double));  // [error², norm²]

    // Create cuBLAS handle for |A| * |B| computation
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Host memory only for final statistics (much smaller)
    double *frobenius_errors = (double*)malloc(num_samples * sizeof(double));
    double *frobenius_M_error = (double*)malloc(num_samples * sizeof(double));
    double *normalized_errors = (double*)malloc(num_samples * sizeof(double));

    // Declare variables that might be accessed after goto
    FILE* fp = NULL;

    // Configure kernel launch parameters
    dim3 threadsPerBlock, numBlocks;
    compute_kernel_dimensions_dispatch(kernel_type, n, &threadsPerBlock, &numBlocks);

    printf("Running %d samples...\n", num_samples);

    // Run multiple samples
    for (int sample = 0; sample < num_samples; sample++) {
        if (sample % 10 == 0 && sample > 0) {
            printf("Completed %d/%d samples...\n", sample, num_samples);
        }

        // Generate new matrices for this sample using the specified matrix type
        // Use different seeds for each sample to ensure truly random matrices
        unsigned long long base_seed = (unsigned long long)time(NULL);
        generate_matrix_device_with_seed(d_A, n, matrix_type, base_seed + sample * 1000);
        generate_matrix_device_with_seed(d_B, n, matrix_type, base_seed + sample * 1000 + 1);

        // Compute FP64 reference directly on device (no host transfers!)
        dim3 block_ref(TILE_SIZE, TILE_SIZE);
        dim3 grid_ref((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
        compute_reference_fp64_device<<<grid_ref, block_ref>>>(d_A, d_B, d_C_reference_fp64, n);

        // Launch the specified kernel using unified dispatch
        launch_kernel_by_type(kernel_type, d_A, d_B, d_C_kernel, n, numBlocks, threadsPerBlock);

        // Compute absolute value matrices on device for norm computation (convert to FP64)
        int total_elements = n * n;
        // Use same block size as main GEMM kernels for consistency
        int block_size = threadsPerBlock.x * threadsPerBlock.y;  // Match main kernel block size
        int grid_size = (total_elements + block_size - 1) / block_size;
        compute_matrix_abs_fp64_kernel<<<grid_size, block_size>>>(d_A, d_A_abs, total_elements);
        compute_matrix_abs_fp64_kernel<<<grid_size, block_size>>>(d_B, d_B_abs, total_elements);

        // Compute |A| * |B| using cuBLAS (double precision)
        const double alpha = 1.0, beta = 0.0;
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                    &alpha, d_B_abs, n, d_A_abs, n, &beta, d_abs_AB_product, n);

        // Reset error accumulation array
        cudaMemset(d_error_results, 0, 2 * sizeof(double));

        // Compute errors entirely on device
        size_t shared_mem_size = 2 * block_size * sizeof(double);  // For error and norm arrays
        compute_frobenius_error_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_C_kernel, d_C_reference_fp64, d_abs_AB_product, d_error_results, n);

        cudaDeviceSynchronize();

        // Copy only the final error results (2 doubles instead of n² floats!)
        double host_error_results[2];
        cudaMemcpy(host_error_results, d_error_results, 2 * sizeof(double), cudaMemcpyDeviceToHost);

        frobenius_errors[sample] = sqrt(host_error_results[0]);
        frobenius_M_error[sample] = sqrt(host_error_results[1]);
        // Compute beta normalized error: empirical_error / (|A||B|)
        double theoretical_bound = frobenius_M_error[sample];
        normalized_errors[sample] = frobenius_errors[sample] / theoretical_bound;

    }

    printf("Completed all %d samples\n", num_samples);

    // Compute comprehensive statistics using utility function
    ArrayStats frob_stats;
    compute_array_statistics(frobenius_errors, num_samples, &frob_stats);

    ArrayStats beta_stats;
    compute_array_statistics(normalized_errors, num_samples, &beta_stats);

    // Print summary
    printf("\n=== Multi-Sample Analysis Results ===\n");
    printf("Matrix Type: %s, Kernel: %s, Size: %dx%d\n", matrixTypeToString(matrix_type), kernelTypeToString(kernel_type), n, n);
    printf("Number of samples: %d\n", num_samples);
    printf("\nFrobenius Error Statistics:\n");
    printf("  Average: %.3e\n", frob_stats.average);
    printf("  Std Dev: %.3e\n", frob_stats.std_dev);
    printf("  95th %%ile: %.3e\n", frob_stats.p95);
    printf("  Max: %.3e\n", frob_stats.maximum);
    printf("\nNormalized Error |C-C_ref|/(|A||B|) Statistics:\n");
    printf("  Average: %.3e\n", beta_stats.average);
    printf("  Std Dev: %.3e\n", beta_stats.std_dev);
    printf("  95th %%ile: %.3e\n", beta_stats.p95);
    printf("  Max: %.3e\n", beta_stats.maximum);
    printf("Theoretical error bound factor (beta): %.6e\n", beta_factor);
    printf("Average Error_beta/beta: %.6e\n", beta_stats.average/beta_factor);
    const double u32 = unit_roundoff_fp32();
    printf("Average Error_beta/u32: %.6e\n", beta_stats.average/u32);

    // Save summary results with metadata to file
    char filename[256];
    snprintf(filename, sizeof(filename), "data/%s_%s_summary_n%d.csv",
             output_prefix, matrixTypeToString(matrix_type), n);
    fp = fopen(filename, "w");
    if (fp) {
        // Write header with all metadata and statistics
        fprintf(fp, "matrix_type,kernel_type,matrix_size,num_samples,");
        fprintf(fp, "frob_avg,frob_std,frob_p95,frob_max,");
        fprintf(fp, "beta_avg,beta_std,beta_p95,beta_max,");
        fprintf(fp, "theoretical_beta,u32,E_{AB}/beta,E_{AB}/u\n");

        // Write single row with all the summary data
        fprintf(fp, "%s,%s,%d,%d,",
                matrixTypeToString(matrix_type),
                kernelTypeToString(kernel_type),
                n,
                num_samples);
        fprintf(fp, "%.10e,%.10e,%.10e,%.10e,",
                frob_stats.average, frob_stats.std_dev, frob_stats.p95, frob_stats.maximum);
        fprintf(fp, "%.10e,%.10e,%.10e,%.10e,",
                beta_stats.average, beta_stats.std_dev, beta_stats.p95, beta_stats.maximum);
        fprintf(fp, "%.10e,%.10e,%.10e,%.10e\n",
                beta_factor, u32, beta_stats.average/beta_factor, beta_stats.average/u32);

        fclose(fp);
        printf("\nSummary results saved to: %s\n", filename);
    }

    // Cleanup device memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_kernel);
    cudaFree(d_C_reference_fp64); cudaFree(d_A_abs); cudaFree(d_B_abs);
    cudaFree(d_abs_AB_product); cudaFree(d_error_results);

    // Cleanup cuBLAS handle
    cublasDestroy(cublas_handle);

    // Cleanup host memory (much less now!)
    free(frobenius_errors); free(frobenius_M_error); free(normalized_errors);
}
