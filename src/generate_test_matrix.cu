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

// SVD-based matrix generation with controlled condition number
void generate_matrix_svd(float* d_A, int n, float cond_num) {
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
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
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

    // Step 3: Construct Sigma on host and upload
    std::vector<float> h_sigma(n * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float sval = std::pow(cond_num, -((float)i / (n - 1)));  // log-uniform decay
        h_sigma[i * n + i] = sval;
    }
    float* d_sigma;
    cudaMalloc(&d_sigma, n * n * sizeof(float));
    cudaMemcpy(d_sigma, h_sigma.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Step 4: A_illcond = Q1 * Sigma * Q2^T
    float alpha = 1.0f, beta = 0.0f;
    float* d_tmp;
    cudaMalloc(&d_tmp, n * n * sizeof(float));
    // tmp = Q1 * Sigma
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_Q1, n, d_sigma, n, &beta, d_tmp, n);
    // d_A = tmp * Q2^T
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, d_tmp, n, d_Q2, n, &beta, d_A, n);

    // Cleanup
    cudaFree(d_Q1); cudaFree(d_Q2); cudaFree(d_temp1); cudaFree(d_temp2);
    cudaFree(d_sigma); cudaFree(d_tmp); cudaFree(d_tau); cudaFree(d_Rwork); cudaFree(d_info);
    cublasDestroy(cublasH); cusolverDnDestroy(cusolverH); curandDestroyGenerator(gen);
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
            printf("Generating well-conditioned matrix using SVD (condition number: %.2e)\n", 2.0f);
            generate_matrix_svd(d_matrix, n, 2.0f);
            break;

        case MATRIX_ODO_ILL_CONDITIONED:
            printf("Generating ill-conditioned matrix using SVD (condition number: %.2e)\n", 1e6f);
            generate_matrix_svd(d_matrix, n, 1e6f);
            break;

        case MATRIX_NORMAL_DISTRIBUTION:
            printf("Generating normal distribution matrix (mean: 0.0, std: 1.0)\n");
            generate_matrix_distribution(d_matrix, n, n, DIST_NORMAL, 0.0f, 1.0f);
            break;

        case MATRIX_SCALED_FTZ:
            {
                printf("Generating FTZ-scaled matrix (scale: %.2e)\n", 1e-30f);
                // First generate normal, then scale to FTZ range
                generate_matrix_distribution(d_matrix, n, n, DIST_NORMAL, 0.0f, 1.0f);

                float ftz_scale = 1e-30f;
                float* h_matrix = (float*)malloc(n * n * sizeof(float));
                cudaMemcpy(h_matrix, d_matrix, n * n * sizeof(float), cudaMemcpyDeviceToHost);
                for (int i = 0; i < n * n; i++) {
                    h_matrix[i] *= ftz_scale;
                }
                cudaMemcpy(d_matrix, h_matrix, n * n * sizeof(float), cudaMemcpyHostToDevice);
                free(h_matrix);
                break;
            }

        case MATRIX_SKEW_MAGNITUDE:
            printf("Generating skewed magnitude matrix using log-normal distribution\n");
            generate_matrix_distribution(d_matrix, n, n, DIST_LOG_NORMAL, 0.0f, 2.0f);
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

// CUDA kernel to scale uniform values from [0,1) to [min,max)
__global__ void scale_uniform_kernel(float* data, int n, float min_val, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = min_val + data[idx] * scale;
    }
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
void generate_matrix_distribution(float* d_matrix, int m, int n, DistributionType dist_type,
                                 float param1, float param2) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    size_t total_elements = m * n;

    switch (dist_type) {
        case DIST_UNIFORM:
            curandGenerateUniform(gen, d_matrix, total_elements);
            // Scale from [0,1) to [param1, param2)
            if (param1 != 0.0f || param2 != 1.0f) {
                float scale = param2 - param1;
                dim3 threads(256);
                dim3 blocks((total_elements + threads.x - 1) / threads.x);
                scale_uniform_kernel<<<blocks, threads>>>(d_matrix, total_elements, param1, scale);
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
