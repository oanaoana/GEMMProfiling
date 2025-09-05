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

// Efficient multi-sample testing for specific matrix type and kernel
void run_multi_sample_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int num_samples, const char* output_prefix);

// ULP-based analysis for specific matrix type and kernel
void run_ulp_samples_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int num_samples, const char* output_prefix);
