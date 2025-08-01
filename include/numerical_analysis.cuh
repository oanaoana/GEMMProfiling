// numerical_analysis.cuh - Header for numerical analysis of tiled GEMM
#pragma once
#include <cuda_runtime.h>

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

// Analysis configuration
#define MAX_CONDITION_NUMBER 1e10f
#define SIGNIFICANT_ERROR_THRESHOLD 1e-6f

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
