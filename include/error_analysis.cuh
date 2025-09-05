// error_analysis.cuh - Header for consolidated error analysis functionality
#pragma once
#include <cuda_runtime.h>
#include "config.h"  // For fundamental types like MatrixType
#include <cstdint>
#include "generate_test_matrix.cuh"  // For generate_matrix_svd function


// Analysis configuration
#define MAX_CONDITION_NUMBER 1e10f
#define SIGNIFICANT_ERROR_THRESHOLD 1e-6f

// Bin labels and representatives
static const char*  BIN_LABELS[9] = {"0","1","2","3-4","5-8","9-16","17-32","33-64","65+"};
static const uint32_t BIN_REP_UPPER[9] = {0,1,2,4,8,16,32,64, UINT32_MAX};

// Error analysis types
typedef struct {
    double total_absolute_error;
    double max_relative_error;
    double average_condition_number_A;
    double average_condition_number_B;
    int significant_error_count;
    double frobenius_norm_A;
    double frobenius_norm_B;
    double error_accumulation_factor;
} NumericalAnalysisResult;


// Structure to hold matrix test configuration
typedef struct {
    MatrixType type_A;
    MatrixType type_B;
    const char* name;
    const char* description;
    const char* filename_A;  // For loading from file
    const char* filename_B;  // For loading from file
    bool enabled;
} MatrixTestConfig;

// Function declarations for numerical analysis
__global__ void analyze_tiled_gemm_errors(
    float *A, float *B, float *C_result, float *C_reference,
    int N,
    float *tile_norms, float *tile_condition_numbers,
    float *accumulated_errors, int *error_counts);

// Host functions
void run_numerical_analysis(float* h_A, float* h_B, int n, const char* output_filename);
void analyze_numerical_results(float* tile_norms, float* condition_numbers, float* errors, int n, int error_count, const char* filename);
void compare_tile_sizes(float* h_A, float* h_B, int n);

// Device utility functions
__device__ float compute_frobenius_norm_tile(float* tile, int rows, int cols);
__device__ float estimate_condition_number_tile(float* tile, int rows, int cols);

// Matrix standardization
void standardize_matrix_host(float* h_matrix, int n);

// Theoretical error bound computation
float compute_beta_factor(KernelType kernel_type, bool single_pass, int n);

// Function declarations for error analysis tests
void runNumericalAnalysisBenchmarks(bool* enabled_sizes);
void setupMatrix(float* matrix, int n, MatrixType type, const char* filename);
void runMatrixTests(int n, MatrixTestConfig* configs, int num_configs);
void generateReport(bool* enabled_sizes);

// Efficient multi-sample testing for specific matrix type and kernel
void run_multi_sample_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int num_samples, const char* output_prefix);

// ULP-based analysis for specific matrix type and kernel
void run_ulp_samples_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int num_samples, const char* output_prefix);
