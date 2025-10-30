// error_analysis.cuh - Header for consolidated error analysis functionality
#pragma once
#include <cuda_runtime.h>
#include "config.h"  // For fundamental types like MatrixType
#include <cstdint>
#include "generate_test_matrix.cuh"  // For generate_matrix_svd function


// Analysis configuration
#define MAX_CONDITION_NUMBER 1e10f
#define SIGNIFICANT_ERROR_THRESHOLD 1e-6f
#define NUM_BINS 9

// Bin labels and representatives
static const char*  BIN_LABELS[NUM_BINS] = {"0","1","2","3-4","5-8","9-16","17-32","33-64","65+"};
static const uint32_t BIN_REP_UPPER[NUM_BINS] = {0,1,2,4,8,16,32,64, UINT32_MAX};

// Theoretical error bound computation
float compute_beta_factor(KernelType kernel_type, bool single_pass, int n);

// Helper function to generate a specific matrix pair from a reproducible sequence
void generate_matrix_pair_from_sequence(float* d_A, float* d_B, int n, MatrixType matrix_type,
                                       unsigned long long base_seed, int sample_index);

// Function to find the sample index closest to median error from analysis results
int find_median_sample_index(const float* error_values, int num_samples);

// Add these template declarations to error_analysis.cuh:

// Template kernel declarations
template<typename AccumulateType>
__global__ void compute_frobenius_error_kernel(AccumulateType* C_kernel, double* C_reference,
                                             double* abs_AB_product, double* error_results, int n);

template<typename ComputeType>
__global__ void compute_reference_and_norm_fp64_device(const ComputeType* A, const ComputeType* B,
                                                       double* C_ref, double* abs_AB_product, int n);

template<typename T>
__global__ void compute_matrix_abs_fp64_kernel_typed(const T* matrix, double* abs_matrix, int size);

// Regular (non-template) kernel declarations
__global__ void compute_matrix_abs_kernel(float* matrix, float* abs_matrix, int size);
__global__ void convert_fp64_to_fp32_kernel(double* d_input, float* d_output, int size);
__global__ void compute_matrix_abs_fp64_kernel(float* matrix, double* abs_matrix, int size);
__global__ void compute_reference_fp64_device(float* A, float* B, double* C, int n);
__global__ void compute_EAB_entrywise(const float* __restrict__ C_kernel,
                                     const double* __restrict__ C_ref64,
                                     const double* __restrict__ absAB64,
                                     double* __restrict__ EAB,
                                     int n, double denom_floor);

// ULP analysis kernels
__global__ void ulp_metrics_kernel(const float* __restrict__ Ctest,
                                   const float* __restrict__ Cref,
                                   uint32_t* __restrict__ dULP,
                                   float* __restrict__ errULP,
                                   unsigned long long* __restrict__ invalid_count,
                                   int n);

__global__ void ulp_stream_hist_kernel(const float* __restrict__ Ctest,
                                       const float* __restrict__ Cref,
                                       unsigned long long* __restrict__ gBins,
                                       double* __restrict__ gErrSum,
                                       double* __restrict__ gErrSumSq,
                                       unsigned long long* __restrict__ gCount,
                                       unsigned long long* __restrict__ gInvalid,
                                       unsigned long long* __restrict__ gRefZeroOrSub,
                                       int n);

// Host function declarations
void run_ulp_samples_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int num_samples, bool reproducible);
void run_multi_sample_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int num_samples, bool reproducible);
void run_per_tile_reference_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int sample_index, bool reproducible);

// Utility functions
void generate_seed_array(unsigned long long* seeds, int num_samples, unsigned long long base_seed);
void generate_matrix_pair_from_sequence(float* d_A, float* d_B, int n, MatrixType matrix_type, unsigned long long base_seed, int sample_index);
int find_median_sample_index(const float* error_values, int num_samples);
float compute_beta_factor(KernelType kernel_type, int K);
double compute_log_c_hat_median(const double* frobenius_errors, int num_samples, double beta_factor, double u_compute);

// Device helper functions (if needed by other files)
__device__ double atomicAddDouble(double* address, double val);
