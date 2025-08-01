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

// Helper function to check if file exists
bool file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

// Generate filename for matrix based on type and size
void generate_matrix_filename(char* filename, size_t filename_size, MatrixType type, int n) {
    const char* type_names[] = {
        "random", "wellcond", "illcond", "identity", "hilbert", "fromfile", "custom"
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

    // Define condition numbers for different matrix types
    float condition_number;
    switch (type) {
        case MATRIX_RANDOM:
            condition_number = 10.0f;  // Moderately conditioned
            break;
        case MATRIX_WELL_CONDITIONED:
            condition_number = 2.0f;   // Well conditioned
            break;
        case MATRIX_ILL_CONDITIONED:
            condition_number = 1e6f;   // Ill conditioned
            break;
        case MATRIX_IDENTITY:
            condition_number = 1.0f;   // Perfect condition
            break;
        case MATRIX_HILBERT:
            condition_number = 1e12f;  // Very ill conditioned
            break;
        case MATRIX_CUSTOM:
            condition_number = 1e3f;   // Moderately ill conditioned
            break;
        default:
            printf("Unknown matrix type %d, using moderate conditioning\n", (int)type);
            condition_number = 10.0f;
            break;
    }

    printf("Generating matrix with condition number %.2e\n", condition_number);

    // Allocate device memory for matrix generation
    float* d_matrix;
    cudaMalloc(&d_matrix, n * n * sizeof(float));

    // Generate matrix using SVD method
    generate_matrix_svd(d_matrix, n, condition_number);

    // Copy result back to host
    cudaMemcpy(matrix, d_matrix, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_matrix);

    // Save generated matrix to file for future use
    if (!write_matrix_to_file(filename, matrix, n)) {
        printf("Warning: Failed to save matrix to file, but matrix was generated successfully\n");
    }

    return true;
}// Utility function to print matrix statistics
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
