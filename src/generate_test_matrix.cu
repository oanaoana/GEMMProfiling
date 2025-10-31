// generate_test_matrix.cu - Matrix generation and file I/O for numerical analysis
#include "../include/generate_test_matrix.cuh"
#include "../include/utils.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <type_traits>  // for std::is_same_v
#include <cuda_fp16.h>        // for __half
#include <cuda_bf16.h>        // for __nv_bfloat16

// Helper function to check if file exists
bool file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

// Generate filename for matrix based on type and size
void generate_matrix_filename(char* filename, size_t filename_size, MatrixType type, int n) {
    const char* type_names[] = {
        "wellcond", "illcond", "normaldist", "scaledftz", "skewmag", "fromfile"
    };

    int type_index = (int)type;
    if (type_index < 0 || type_index >= sizeof(type_names)/sizeof(type_names[0])) {
        type_index = 0; // fallback to random
    }

    snprintf(filename, filename_size, "data/matrix_%s_%dx%d.bin", type_names[type_index], n, n);
}

// Write matrix to binary file
bool write_matrix_to_file(const char* filename, float* matrix, int n) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("ERROR: Cannot create matrix file %s\n", filename);
        return false;
    }

    // Write header with matrix size for verification
    int header[2] = {n, n};
    if (fwrite(header, sizeof(int), 2, fp) != 2) {
        printf("ERROR: Failed to write matrix header to %s\n", filename);
        fclose(fp);
        return false;
    }

    // Write matrix data
    size_t elements_written = fwrite(matrix, sizeof(float), n * n, fp);
    fclose(fp);

    if (elements_written != n * n) {
        printf("ERROR: Failed to write complete matrix to %s\n", filename);
        return false;
    }

    printf("Matrix saved to: %s\n", filename);
    return true;
}

// Read matrix from binary file
bool read_matrix_from_file(const char* filename, float* matrix, int n) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("ERROR: Cannot open matrix file %s\n", filename);
        return false;
    }

    // Read and verify header
    int header[2];
    if (fread(header, sizeof(int), 2, fp) != 2) {
        printf("ERROR: Failed to read matrix header from %s\n", filename);
        fclose(fp);
        return false;
    }

    if (header[0] != n || header[1] != n) {
        printf("ERROR: Matrix size mismatch in %s. Expected %dx%d, got %dx%d\n",
               filename, n, n, header[0], header[1]);
        fclose(fp);
        return false;
    }

    // Read matrix data
    size_t elements_read = fread(matrix, sizeof(float), n * n, fp);
    fclose(fp);

    if (elements_read != n * n) {
        printf("ERROR: Failed to read complete matrix from %s\n", filename);
        return false;
    }

    printf("Matrix loaded from: %s\n", filename);
    return true;
}
__global__ void scale_matrix_columns_kernel(float* matrix, const float* scale_values, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        // Scale column 'col' by scale_values[col]
        matrix[row * n + col] *= scale_values[col];
    }
}

// SVD-based matrix generation with controlled condition number and custom seed
void generate_matrix_svd_with_seed(float* d_A, int n, float cond_num, unsigned long long seed) {
    // Create cuBLAS and cuSOLVER handles
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    cublasCreate(&cublasH);
    cusolverDnCreate(&cusolverH);

    // Allocate device memory
    float *d_Q1, *d_Q2, *d_Rwork;
    float *d_temp1, *d_temp2;
    cudaMalloc(&d_Q1, n * n * sizeof(float));
    cudaMalloc(&d_Q2, n * n * sizeof(float));
    cudaMalloc(&d_temp1, n * n * sizeof(float));
    cudaMalloc(&d_temp2, n * n * sizeof(float));

    // Step 1: Generate two random matrices
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, d_temp1, n * n);
    curandGenerateUniform(gen, d_temp2, n * n);

    // Step 2: QR decomposition → Q1, Q2
    std::vector<int> devInfo(1);
    int *d_info;
    cudaMalloc(&d_info, sizeof(int));
    int lwork = 0;
    cusolverDnSgeqrf_bufferSize(cusolverH, n, n, d_temp1, n, &lwork);
    cudaMalloc(&d_Rwork, lwork * sizeof(float));
    std::vector<float> tau(n);
    float* d_tau;
    cudaMalloc(&d_tau, n * sizeof(float));

    // QR on d_temp1 → d_Q1
    cusolverDnSgeqrf(cusolverH, n, n, d_temp1, n, d_tau, d_Rwork, lwork, d_info);
    cusolverDnSorgqr(cusolverH, n, n, n, d_temp1, n, d_tau, d_Rwork, lwork, d_info);
    cudaMemcpy(d_Q1, d_temp1, n * n * sizeof(float), cudaMemcpyDeviceToDevice);

    // QR on d_temp2 → d_Q2
    cusolverDnSgeqrf(cusolverH, n, n, d_temp2, n, d_tau, d_Rwork, lwork, d_info);
    cusolverDnSorgqr(cusolverH, n, n, n, d_temp2, n, d_tau, d_Rwork, lwork, d_info);
    cudaMemcpy(d_Q2, d_temp2, n * n * sizeof(float), cudaMemcpyDeviceToDevice);

    // Step 3: Generate random singular values and scale Q1 columns directly
    float* d_sigma_vals;
    cudaMalloc(&d_sigma_vals, n * sizeof(float));

    float sigma_min = 1.0f / cond_num;
    float sigma_max = 1.0f;

    // Generate n random values in [sigma_min, sigma_max] using seed
    generate_matrix_distribution_with_seed(d_sigma_vals, 1, n, DIST_UNIFORM,
                                       sigma_min, sigma_max, seed + 999999);

    // Sort in DESCENDING order (required for SVD)
    thrust::device_ptr<float> dev_ptr(d_sigma_vals);
    thrust::sort(dev_ptr, dev_ptr + n, thrust::greater<float>());

    // Force exact endpoints after sorting to guarantee condition number
    cudaMemcpy(&d_sigma_vals[0], &sigma_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_sigma_vals[n-1], &sigma_min, sizeof(float), cudaMemcpyHostToDevice);

    // Scale each column of Q1 by corresponding singular value (in-place)
    dim3 threads(16, 16);  // 2D block for matrix
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    scale_matrix_columns_kernel<<<blocks, threads>>>(d_Q1, d_sigma_vals, n);
    cudaDeviceSynchronize();

    cudaFree(d_sigma_vals);

    // Step 4: A = Q1_scaled * Q2^T (just ONE matrix multiply!)
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, d_Q1, n, d_Q2, n, &beta, d_A, n);

    // Cleanup
    cudaFree(d_Q1); cudaFree(d_Q2); cudaFree(d_temp1); cudaFree(d_temp2);
    cudaFree(d_tau); cudaFree(d_Rwork); cudaFree(d_info);
    cublasDestroy(cublasH); cusolverDnDestroy(cusolverH); curandDestroyGenerator(gen);
}

// CUDA kernel for scaling matrix elements
__global__ void scale_matrix_kernel(float* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}

// CUDA kernel to convert uniform [0,1) values to Rademacher {-1, +1}
__global__ void rademacher_transform_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Convert uniform [0,1) to Rademacher: -1 if < 0.5, +1 if >= 0.5
        data[idx] = (data[idx] < 0.5f) ? -1.0f : 1.0f;
    }
}

// CUDA kernel to convert integer {0,1} signs directly to Rademacher {-1, +1}
// Unified CUDA kernel to apply random signs from integer {0,1} to {-1,+1}
// If replace_mode=true: result[i] = sign(signs[i])
// If replace_mode=false: result[i] *= sign(signs[i])
__global__ void random_signs_kernel(int* signs, float* result, int size, bool replace_mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Convert integer {0,1} to sign {-1,+1}
        float sign = (signs[idx] == 0) ? -1.0f : 1.0f;

        if (replace_mode) {
            // Replace mode: result = sign
            result[idx] = sign;
        } else {
            // Multiply mode: result *= sign
            result[idx] *= sign;
        }
    }
}

// CUDA kernel to generate jittered Rademacher: sign * (1 + δ) where δ ∈ (-2^(-12), 2^(-12))
__global__ void rademacher_jittered_kernel(int* signs, float* jitter, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Convert integer {0,1} to sign {-1,+1}
        float sign = (signs[idx] == 0) ? -1.0f : 1.0f;

        // Apply jitter: sign * (1 + δ) where δ ∈ (-2^(-12), 2^(-12))
        result[idx] = sign * (1.0f + jitter[idx]);
    }
}

// CUDA kernel to generate 2-powers matrix: s * 2^(-p) where s is ±1 and p is integer [10,30]
__global__ void twopowers_transform_kernel(int* signs, int* exponents, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Use sign directly: convert 0 to -1, 1 to +1
        float s = (signs[idx] == 0) ? -1.0f : 1.0f;

        // Use the integer exponent directly (should be in range [10,30])
        int p = exponents[idx];

        // Compute s * 2^(-p) using ldexpf
        result[idx] = s * ldexpf(1.0f, -p);
    }
}// CUDA kernel for clamping matrix elements to interval (min_val, max_val)
// CUDA kernel to transform values from interval [a,b] to [c,d]
// Formula: y = c + (x - a) * (d - c) / (b - a)
__global__ void transform_interval_kernel(float* data, int n, float a, float b, float c, float d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        // Transform from [a,b] to [c,d]
        float scale = (d - c) / (b - a);
        data[idx] = c + (x - a) * scale;
    }
}

// Simple kernel for integer range transformation
__global__ void transform_int_range_kernel(int* data, int size, int min_val, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int val = ((unsigned int*)data)[idx];
        data[idx] = min_val + (val % range);
    }
}

// Generate integer uniform distribution in range [min_val, max_val] (inclusive)
void generate_integer_uniform_with_seed(int* d_integers, int m, int n, int min_val, int max_val, unsigned long long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    size_t total_elements = m * n;
    curandGenerate(gen, (unsigned int*)d_integers, total_elements);

    // Transform to desired range [min_val, max_val]
    int range = max_val - min_val + 1;
    dim3 threads(256);
    dim3 blocks((total_elements + threads.x - 1) / threads.x);

    // Launch kernel to map to range
    transform_int_range_kernel<<<blocks, threads>>>(d_integers, total_elements, min_val, range);
    cudaDeviceSynchronize();

    curandDestroyGenerator(gen);
}

// Non-seeded version using time(NULL)
void generate_integer_uniform(int* d_integers, int m, int n, int min_val, int max_val) {
    generate_integer_uniform_with_seed(d_integers, m, n, min_val, max_val, time(NULL));
}

float compute_condition_number(float* d_matrix, int n) {
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    // Allocate workspace for SVD
    float* d_S;  // Singular values
    float* d_U;
    float* d_VT;
    int* devInfo;

    cudaMalloc(&d_S, n * sizeof(float));
    cudaMalloc(&d_U, n * n * sizeof(float));
    cudaMalloc(&d_VT, n * n * sizeof(float));
    cudaMalloc(&devInfo, sizeof(int));

    // Query workspace size
    int lwork = 0;
    cusolverDnSgesvd_bufferSize(cusolver_handle, n, n, &lwork);

    float* d_work;
    cudaMalloc(&d_work, lwork * sizeof(float));

    // Compute SVD: A = U * S * V^T
    cusolverDnSgesvd(cusolver_handle, 'N', 'N',  // Don't compute U, VT
                     n, n, d_matrix, n,
                     d_S, d_U, n, d_VT, n,
                     d_work, lwork, nullptr, devInfo);

    // Copy singular values to host
    float* h_S = new float[n];
    cudaMemcpy(h_S, d_S, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Condition number = σ_max / σ_min
    float sigma_max = h_S[0];
    float sigma_min = h_S[n-1];
    float cond = sigma_max / sigma_min;

    printf("  [Verification] Actual singular values: σ_max=%.6e, σ_min=%.6e\n", sigma_max, sigma_min);
    printf("  [Verification] Actual condition number: %.6e\n", cond);

    // Cleanup
    delete[] h_S;
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolver_handle);

    return cond;
}

// Efficient matrix generation for multi-sample analysis
// - Works directly on pre-allocated device memory
// - No file I/O overhead
// - No memory allocation/deallocation
// - Supports all matrix types with custom seeds
void generate_matrix_device_with_seed(float* d_matrix, int n, MatrixType type, unsigned long long seed) {
    switch (type) {
        case MATRIX_ODO_WELL_CONDITIONED:
            // {
            // printf("=== MATRIX_ODO_WELL_CONDITIONED Generation ===\n");
            // printf("  Matrix size: %d x %d\n", n, n);
            // printf("  Seed: %llu\n", seed);
            // printf("  Target condition number: %.2e\n", WELL_COND_NUMBER);
            generate_matrix_svd_with_seed(d_matrix, n, WELL_COND_NUMBER, seed);
            // // Verify actual condition number
            // float actual_cond = compute_condition_number(d_matrix, n);
            // printf("  [Verification] Actual condition number: %.6e\n", actual_cond);
            // printf("=============================================\n");

            // // Sample check - copy first few values back to host
            // std::vector<float> h_check(10);
            // cudaMemcpy(h_check.data(), d_matrix, 10 * sizeof(float), cudaMemcpyDeviceToHost);
            // printf("  First 10 values: ");
            // for (int i = 0; i < 10; i++) {
            //     printf("%.6e ", h_check[i]);
            // }
            // printf("\n");
            // printf("========================================\n");
            // }
            break;

        case MATRIX_ODO_ILL_CONDITIONED:
            // {
            // printf("=== MATRIX_ODO_ILL_CONDITIONED Generation ===\n");
            // printf("  Matrix size: %d x %d\n", n, n);
            // printf("  Seed: %llu\n", seed);
            // printf("  Target condition number: %.2e\n", ILL_COND_NUMBER);

            generate_matrix_svd_with_seed(d_matrix, n, ILL_COND_NUMBER, seed);
            // // Verify actual condition number
            // float actual_cond = compute_condition_number(d_matrix, n);
            // printf("  [Verification] Actual condition number: %.6e\n", actual_cond);
            // printf("=============================================\n");
            // }
            break;

        case MATRIX_ZEROMEAN:
            {
                // Generate normal distribution with zero mean and std = 1/sqrt(n)
                // This gives entries with expected magnitude scaling appropriately with matrix size
                const float mean = 0.0f;
                const float std = 1.0f / sqrtf((float)n);
                generate_matrix_distribution_with_seed(d_matrix, n, n, DIST_NORMAL, mean, std, seed);
            }
            break;

        case MATRIX_UNIFORM_POSITIVE:
            {
                // Generate uniform distribution in (0,1) interval
                const float epsilon = 1e-6f;  // Small margin for open interval
                generate_matrix_distribution_with_seed(d_matrix, n, n, DIST_UNIFORM,
                                                      epsilon, 1.0f - epsilon, seed);
            }
            break;

        case MATRIX_RADEMACHER:
            {
                // Generate jittered Rademacher distribution: sign * (1 + δ) where δ ∈ (-2^(-12), 2^(-12))
                // This adds small perturbations to the ±1 structure for more realistic numerical analysis
                int total_elements = n * n;

                // Allocate temporary memory for signs and jitter
                int *d_signs;
                float *d_jitter;
                cudaMalloc(&d_signs, total_elements * sizeof(int));
                cudaMalloc(&d_jitter, total_elements * sizeof(float));

                // Generate integer signs {0,1}
                generate_integer_uniform_with_seed(d_signs, n, n, 0, 1, seed);

                // Generate jitter δ ∈ (-2^(-12), 2^(-12)) = (-1/4096, 1/4096)
                const float jitter_bound = 1.0f / 4096.0f;  // 2^(-12)
                generate_matrix_distribution_with_seed(d_jitter, n, n, DIST_UNIFORM,
                                                      -jitter_bound, jitter_bound, seed + 54321);

                // Convert to jittered Rademacher: sign * (1 + δ)
                int block_size = 256;
                int grid_size = (total_elements + block_size - 1) / block_size;

                rademacher_jittered_kernel<<<grid_size, block_size>>>(d_signs, d_jitter, d_matrix, total_elements);
                cudaDeviceSynchronize();

                // Cleanup temporary memory
                cudaFree(d_signs);
                cudaFree(d_jitter);
            }
            break;

        case MATRIX_SANITY:
            {
                // Generate original Rademacher distribution: exact ±1 values
                // Perfect for debugging - should always produce zero errors with exact arithmetic
                int total_elements = n * n;

                // Allocate temporary memory for signs
                int *d_signs;
                cudaMalloc(&d_signs, total_elements * sizeof(int));

                // Generate integer signs {0,1}
                generate_integer_uniform_with_seed(d_signs, n, n, 0, 1, seed);

                // Convert directly to exact Rademacher {-1, +1}
                int block_size = 256;
                int grid_size = (total_elements + block_size - 1) / block_size;

                random_signs_kernel<<<grid_size, block_size>>>(d_signs, d_matrix, total_elements, true);
                cudaDeviceSynchronize();

                // Cleanup temporary memory
                cudaFree(d_signs);
            }
            break;

        case MATRIX_SCALED_2POWERS:
            {
                // Generate 2-powers matrix: s * 2^(-p) where s is ±1 and p is integer [10,30]
                printf("=== MATRIX_SCALED_2POWERS Generation ===\n");
                printf("  Matrix size: %d x %d\n", n, n);
                printf("  Seed: %llu\n", seed);
                int total_elements = n * n;

                // Allocate temporary memory for signs and exponents
                int *d_signs, *d_exponents;
                cudaMalloc(&d_signs, total_elements * sizeof(int));
                cudaMalloc(&d_exponents, total_elements * sizeof(int));

                // Generate integer distributions for signs {0,1} and exponents [10,30]
                generate_integer_uniform_with_seed(d_signs, n, n, 0, 1, seed);
                generate_integer_uniform_with_seed(d_exponents, n, n, 10, 30, seed + 12345);

                // Transform to 2-powers matrix
                int block_size = 256;
                int grid_size = (total_elements + block_size - 1) / block_size;

                twopowers_transform_kernel<<<grid_size, block_size>>>(d_signs, d_exponents, d_matrix, total_elements);
                cudaDeviceSynchronize();

                // Sample check - copy first few values back to host
                std::vector<float> h_check(10);
                cudaMemcpy(h_check.data(), d_matrix, 10 * sizeof(float), cudaMemcpyDeviceToHost);
                printf("  First 10 values: ");
                for (int i = 0; i < 10; i++) {
                    printf("%.6e ", h_check[i]);
                }
                printf("\n");
                printf("========================================\n");

                // Cleanup temporary memory
                cudaFree(d_signs);
                cudaFree(d_exponents);
            }
            break;

        case MATRIX_LOGNORMAL:
            {
                // Generate signed log-normal distribution: sign * exp(N(0, σ)) where σ ∈ {1, 2}
                // Random signs ensure zero mean while keeping log-normal magnitude distribution
                int total_elements = n * n;
                const float sigma = (n < 1024) ? 1.0f : 2.0f;
                const float mean = 0.0f;  // Normal distribution mean = 0

                // Allocate temporary memory for signs
                int *d_signs;
                cudaMalloc(&d_signs, total_elements * sizeof(int));

                // Generate positive log-normal values: exp(N(0, σ))
                generate_matrix_distribution_with_seed(d_matrix, n, n, DIST_LOG_NORMAL, mean, sigma, seed);

                // Generate random signs {0,1} → {-1,+1}
                generate_integer_uniform_with_seed(d_signs, n, n, 0, 1, seed + 98765);

                // Apply random signs to get zero-mean signed log-normal
                int block_size = 256;
                int grid_size = (total_elements + block_size - 1) / block_size;

                random_signs_kernel<<<grid_size, block_size>>>(d_signs, d_matrix, total_elements, false);
                cudaDeviceSynchronize();

                // Cleanup temporary memory
                cudaFree(d_signs);
            }
            break;

        case MATRIX_FROM_FILE:
            printf("ERROR: MATRIX_FROM_FILE not supported in generate_matrix_device_with_seed\n");
            printf("Use get_matrix() for file-based matrix loading\n");
            // Fill with zeros as fallback
            cudaMemset(d_matrix, 0, n * n * sizeof(float));
            break;

        default:
            printf("Unknown matrix type %d, using moderate SVD conditioning\n", (int)type);
            generate_matrix_svd_with_seed(d_matrix, n, 10.0f, seed);
            break;
    }
}

// SVD-based matrix generation with controlled condition number
void generate_matrix_svd(float* d_A, int n, float cond_num) {
    // Use current time as seed for better randomization
    unsigned long long seed = (unsigned long long)time(NULL) + (unsigned long long)clock();
    generate_matrix_svd_with_seed(d_A, n, cond_num, seed);
}

// Main function to get matrix (from cache or generate new)
bool get_matrix(float* matrix, int n, MatrixType type, const char* custom_filename) {
    char filename[512];

    // Use custom filename if provided and type is MATRIX_FROM_FILE
    if (type == MATRIX_FROM_FILE && custom_filename) {
        strncpy(filename, custom_filename, sizeof(filename) - 1);
        filename[sizeof(filename) - 1] = '\0';
    } else {
        // Generate standard filename
        generate_matrix_filename(filename, sizeof(filename), type, n);
    }

    // Check if file exists and try to load it
    if (file_exists(filename)) {
        printf("Found existing matrix file: %s\n", filename);
        if (read_matrix_from_file(filename, matrix, n)) {
            return true;
        } else {
            printf("Failed to load matrix from file, regenerating...\n");
        }
    }

    // File doesn't exist or failed to load, generate new matrix
    printf("Generating new matrix (type: %d, size: %dx%d)\n", (int)type, n, n);

    if (type == MATRIX_FROM_FILE && custom_filename) {
        printf("ERROR: Custom file %s not found and cannot generate matrix for MATRIX_FROM_FILE type\n", custom_filename);
        return false;
    }

    // Allocate device memory for matrix generation
    float* d_matrix;
    cudaMalloc(&d_matrix, n * n * sizeof(float));

    // Generate matrix based on type
    switch (type) {
        case MATRIX_ODO_WELL_CONDITIONED:
            printf("Generating well-conditioned matrix using SVD (condition number: %.2e)\n", WELL_COND_NUMBER);
            generate_matrix_svd(d_matrix, n, WELL_COND_NUMBER);
            break;

        case MATRIX_ODO_ILL_CONDITIONED:
            printf("Generating ill-conditioned matrix using SVD (condition number: %.2e)\n", ILL_COND_NUMBER);
            generate_matrix_svd(d_matrix, n, ILL_COND_NUMBER);
            break;

        case MATRIX_ZEROMEAN:
            printf("Generating zero-mean normal distribution matrix N(0, 1/sqrt(n)) with std=%.6f\n", 1.0f/sqrtf((float)n));
            generate_matrix_distribution(d_matrix, n, n, DIST_NORMAL, 0.0f, 1.0f/sqrtf((float)n));
            break;

        case MATRIX_UNIFORM_POSITIVE:
            printf("Generating uniform distribution matrix in (0,1) interval\n");
            {
                const float epsilon = 1e-6f;
                generate_matrix_distribution(d_matrix, n, n, DIST_UNIFORM, epsilon, 1.0f - epsilon);
            }
            break;

        case MATRIX_RADEMACHER:
            printf("Generating jittered Rademacher distribution matrix: sign * (1 + δ), δ ∈ (-2^(-12), 2^(-12))\n");
            {
                // Generate jittered Rademacher for more realistic numerical behavior
                int total_elements = n * n;

                // Allocate temporary memory for signs and jitter
                int *d_signs;
                float *d_jitter;
                cudaMalloc(&d_signs, total_elements * sizeof(int));
                cudaMalloc(&d_jitter, total_elements * sizeof(float));

                // Generate integer signs {0,1}
                generate_integer_uniform(d_signs, n, n, 0, 1);

                // Generate jitter δ ∈ (-2^(-12), 2^(-12)) = (-1/4096, 1/4096)
                const float jitter_bound = 1.0f / 4096.0f;  // 2^(-12)
                generate_matrix_distribution(d_jitter, n, n, DIST_UNIFORM,
                                           -jitter_bound, jitter_bound);

                // Convert to jittered Rademacher: sign * (1 + δ)
                int block_size = 256;
                int grid_size = (total_elements + block_size - 1) / block_size;

                rademacher_jittered_kernel<<<grid_size, block_size>>>(d_signs, d_jitter, d_matrix, total_elements);
                cudaDeviceSynchronize();

                // Cleanup temporary memory
                cudaFree(d_signs);
                cudaFree(d_jitter);
            }
            break;

        case MATRIX_SANITY:
            printf("Generating SANITY matrix: exact Rademacher ±1 (for debugging/verification)\n");
            {
                // Generate original exact Rademacher - perfect for sanity checks
                int total_elements = n * n;

                // Allocate temporary memory for signs
                int *d_signs;
                cudaMalloc(&d_signs, total_elements * sizeof(int));

                // Generate integer signs {0,1}
                generate_integer_uniform(d_signs, n, n, 0, 1);

                // Convert directly to exact Rademacher {-1, +1}
                int block_size = 256;
                int grid_size = (total_elements + block_size - 1) / block_size;

                random_signs_kernel<<<grid_size, block_size>>>(d_signs, d_matrix, total_elements, true);
                cudaDeviceSynchronize();

                // Cleanup temporary memory
                cudaFree(d_signs);
            }
            break;

        case MATRIX_SCALED_2POWERS:
            {
                printf("Generating 2-powers matrix: s * 2^(-p) where s=±1, p∈[10,30]\n");
                int total_elements = n * n;

                // Allocate temporary memory for signs and exponents
                int *d_signs, *d_exponents;
                cudaMalloc(&d_signs, total_elements * sizeof(int));
                cudaMalloc(&d_exponents, total_elements * sizeof(int));

                // Generate integer distributions for signs {0,1} and exponents [10,30]
                generate_integer_uniform(d_signs, n, n, 0, 1);
                generate_integer_uniform(d_exponents, n, n, 10, 30);

                // Transform to 2-powers matrix
                int block_size = 256;
                int grid_size = (total_elements + block_size - 1) / block_size;

                twopowers_transform_kernel<<<grid_size, block_size>>>(d_signs, d_exponents, d_matrix, total_elements);
                cudaDeviceSynchronize();

                // Cleanup temporary memory
                cudaFree(d_signs);
                cudaFree(d_exponents);
                break;
            }

        case MATRIX_LOGNORMAL:
            {
                // Generate signed log-normal distribution: sign * exp(N(0, σ)) where σ ∈ {1, 2}
                // Random signs ensure zero mean while keeping log-normal magnitude distribution
                int total_elements = n * n;
                const float sigma = (n < 1024) ? 1.0f : 2.0f;
                const float mean = 0.0f;  // Normal distribution mean = 0

                printf("Generating signed log-normal distribution matrix: sign * exp(N(0, %.1f))\n", sigma);

                // Allocate temporary memory for signs
                int *d_signs;
                cudaMalloc(&d_signs, total_elements * sizeof(int));

                // Generate positive log-normal values: exp(N(0, σ))
                generate_matrix_distribution(d_matrix, n, n, DIST_LOG_NORMAL, mean, sigma);

                // Generate random signs {0,1} → {-1,+1} (use deterministic seed based on matrix address)
                uintptr_t address_seed = reinterpret_cast<uintptr_t>(d_matrix);
                generate_integer_uniform_with_seed(d_signs, n, n, 0, 1, static_cast<unsigned long long>(address_seed));

                // Apply random signs to get zero-mean signed log-normal
                int block_size = 256;
                int grid_size = (total_elements + block_size - 1) / block_size;

                random_signs_kernel<<<grid_size, block_size>>>(d_signs, d_matrix, total_elements, false);
                cudaDeviceSynchronize();

                // Cleanup temporary memory
                cudaFree(d_signs);
            }
            break;

        case MATRIX_FROM_FILE:
            printf("ERROR: MATRIX_FROM_FILE should not reach generation code\n");
            cudaFree(d_matrix);
            return false;

        default:
            printf("Unknown matrix type %d, using moderate SVD conditioning\n", (int)type);
            generate_matrix_svd(d_matrix, n, 10.0f);
            break;
    }

    // Copy result back to host
    cudaMemcpy(matrix, d_matrix, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup device memory
    cudaFree(d_matrix);

    // Save generated matrix to file for future use
    if (!write_matrix_to_file(filename, matrix, n)) {
        printf("Warning: Failed to save matrix to file, but matrix was generated successfully\n");
    }

    return true;
}

// Utility function to print matrix statistics
void print_matrix_stats(float* matrix, int n, const char* name) {
    double sum = 0.0, sum_sq = 0.0;
    float min_val = matrix[0], max_val = matrix[0];

    for (int i = 0; i < n * n; i++) {
        float val = matrix[i];
        sum += val;
        sum_sq += val * val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    double mean = sum / (n * n);
    double variance = (sum_sq / (n * n)) - (mean * mean);
    double frobenius_norm = sqrt(sum_sq);

    printf("Matrix %s statistics:\n", name);
    printf("  Size: %dx%d\n", n, n);
    printf("  Range: [%.6e, %.6e]\n", min_val, max_val);
    printf("  Mean: %.6e\n", mean);
    printf("  Std dev: %.6e\n", sqrt(variance));
    printf("  Frobenius norm: %.6e\n", frobenius_norm);
}

// Generate matrix with specified distribution (operates on GPU memory)
// Parameters:
//   d_matrix: Device memory pointer (must be pre-allocated)
//   m, n: Matrix dimensions (rows, columns)
//   dist_type: DIST_UNIFORM, DIST_NORMAL, or DIST_LOG_NORMAL
//   param1, param2: Distribution parameters
//     - UNIFORM: param1=min, param2=max
//     - NORMAL: param1=mean, param2=std_dev
//     - LOG_NORMAL: param1=log_mean, param2=log_std_dev

// Seeded version for multi-sample analysis
void generate_matrix_distribution_with_seed(float* d_matrix, int m, int n, DistributionType dist_type,
                                           float param1, float param2, unsigned long long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    size_t total_elements = m * n;

    switch (dist_type) {
        case DIST_UNIFORM:
            curandGenerateUniform(gen, d_matrix, total_elements);
            // Transform from [0,1) to [param1, param2)
            if (param1 != 0.0f || param2 != 1.0f) {
                dim3 threads(256);
                dim3 blocks((total_elements + threads.x - 1) / threads.x);
                transform_interval_kernel<<<blocks, threads>>>(
                    d_matrix, total_elements,
                    0.0f, 1.0f,      // from [0, 1)
                    param1, param2   // to [param1, param2)
                );
                cudaDeviceSynchronize();
            }
            break;

        case DIST_NORMAL:
            curandGenerateNormal(gen, d_matrix, total_elements, param1, param2);
            break;

        case DIST_LOG_NORMAL:
            curandGenerateLogNormal(gen, d_matrix, total_elements, param1, param2);
            break;

        default:
            printf("Unknown distribution type %d, using uniform [0,1)\n", (int)dist_type);
            curandGenerateUniform(gen, d_matrix, total_elements);
            break;
    }

    curandDestroyGenerator(gen);
}

// Original version (uses time(NULL) for seed)
void generate_matrix_distribution(float* d_matrix, int m, int n, DistributionType dist_type,
                                 float param1, float param2) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    size_t total_elements = m * n;

    switch (dist_type) {
        case DIST_UNIFORM:
            curandGenerateUniform(gen, d_matrix, total_elements);
            // Transform from [0,1) to [param1, param2)
            if (param1 != 0.0f || param2 != 1.0f) {
                dim3 threads(256);
                dim3 blocks((total_elements + threads.x - 1) / threads.x);
                transform_interval_kernel<<<blocks, threads>>>(
                    d_matrix, total_elements,
                    0.0f, 1.0f,      // from [0, 1)
                    param1, param2   // to [param1, param2)
                );
                cudaDeviceSynchronize();
            }
            break;

        case DIST_NORMAL:
            curandGenerateNormal(gen, d_matrix, total_elements, param1, param2);
            break;

        case DIST_LOG_NORMAL:
            curandGenerateLogNormal(gen, d_matrix, total_elements, param1, param2);
            break;

        default:
            printf("Unknown distribution type %d, using uniform [0,1)\n", (int)dist_type);
            curandGenerateUniform(gen, d_matrix, total_elements);
            break;
    }

    curandDestroyGenerator(gen);
}

template<typename T>
void generate_matrix_device_with_seed_typed(T* d_matrix, int n, MatrixType type,
                                           unsigned long long seed,
                                           dim3 numBlocks, dim3 threadsPerBlock) {

    bool needs_conversion = !std::is_same<T, float>::value;

    if (needs_conversion) {
        // Generate in float first, then convert in-place on device
        float* d_temp_float;
        cudaMalloc(&d_temp_float, n * n * sizeof(float));

        // Generate using existing float implementation
        generate_matrix_device_with_seed(d_temp_float, n, type, seed);

        // Convert using the SAME kernel dispatch configuration as your GEMM kernels
        convert_matrix_kernel<float, T><<<numBlocks, threadsPerBlock>>>(
            d_temp_float, d_matrix, n
        );
        cudaDeviceSynchronize();

        cudaFree(d_temp_float);
    } else {
        // T is float, use existing implementation directly
        generate_matrix_device_with_seed((float*)d_matrix, n, type, seed);
    }
}

// CUDA kernel for type conversion with kernel-aware dispatch
template<typename SrcType, typename DstType>
__global__ void convert_matrix_kernel(const SrcType* src, DstType* dst, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // // DEBUG: First thread only
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("=== CONVERT_MATRIX_KERNEL ===\n");
    //     printf("  SrcType size: %zu\n", sizeof(SrcType));
    //     printf("  DstType size: %zu\n", sizeof(DstType));
    //     printf("  Matrix size: %d x %d\n", n, n);
    //     printf("  Grid: (%d, %d), Block: (%d, %d)\n",
    //            gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    //     // Test the actual conversion
    //     SrcType test_src = static_cast<SrcType>(0.75f);
    //     DstType test_dst;

    //     if constexpr (std::is_same_v<SrcType, float> && std::is_same_v<DstType, __half>) {
    //         test_dst = __float2half(test_src);
    //         printf("  Conversion test: float(0.75) -> __half = %f\n", __half2float(test_dst));
    //     }

    //     // Check first few values
    //     printf("  First 5 src values: %f %f %f %f %f\n",
    //            (float)src[0], (float)src[1], (float)src[2], (float)src[3], (float)src[4]);
    // }

    if (row < n && col < n) {
        int idx = row * n + col;

        // Your existing conversion code
        if constexpr (std::is_same_v<SrcType, float> && std::is_same_v<DstType, __half>) {
            dst[idx] = __float2half(src[idx]);
        }
        else if constexpr (std::is_same_v<SrcType, float> && std::is_same_v<DstType, __nv_bfloat16>) {
            dst[idx] = __float2bfloat16(src[idx]);
        }
        else if constexpr (std::is_same_v<SrcType, __half> && std::is_same_v<DstType, float>) {
            dst[idx] = __half2float(src[idx]);
        }
        else if constexpr (std::is_same_v<SrcType, __nv_bfloat16> && std::is_same_v<DstType, float>) {
            dst[idx] = __bfloat162float(src[idx]);
        }
        else {
            // For other combinations, use static_cast
            dst[idx] = static_cast<DstType>(src[idx]);
        }
    }

    // DEBUG: Check what was written
    __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("  First 5 dst values: %f %f %f %f %f\n",
    //            (float)dst[0], (float)dst[1], (float)dst[2], (float)dst[3], (float)dst[4]);
    //     printf("==============================\n");
    // }
}

static_assert(!std::is_same_v<__half, float>, "Type traits working: __half != float");
static_assert(std::is_same_v<float, float>, "Type traits working: float == float");

// Fix: Correct template instantiation guards - use build-time checks instead of runtime
template void generate_matrix_device_with_seed_typed<float>(float* d_matrix, int n, MatrixType type, unsigned long long seed, dim3 numBlocks, dim3 threadsPerBlock);
template void generate_matrix_device_with_seed_typed<double>(double* d_matrix, int n, MatrixType type, unsigned long long seed, dim3 numBlocks, dim3 threadsPerBlock);
// Add explicit instantiations for the conversion kernel
template __global__ void convert_matrix_kernel<float, double>(const float* src, double* dst, int n);
template __global__ void convert_matrix_kernel<float, float>(const float* src, float* dst, int n);

template void generate_matrix_device_with_seed_typed<__half>(__half* d_matrix, int n, MatrixType type, unsigned long long seed, dim3 numBlocks, dim3 threadsPerBlock);
template __global__ void convert_matrix_kernel<float, __half>(const float* src, __half* dst, int n);

// #if defined(__CUDA_BF16_TYPES_EXIST__)
// template void generate_matrix_device_with_seed_typed<__nv_bfloat16>(__nv_bfloat16* d_matrix, int n, MatrixType type, unsigned long long seed, dim3 numBlocks, dim3 threadsPerBlock);
// template __global__ void convert_matrix_kernel<float, __nv_bfloat16>(const float* src, __nv_bfloat16* dst, int n);
// #endif


