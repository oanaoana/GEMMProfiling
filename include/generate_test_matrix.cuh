// generate_test_matrix.cuh - Header for matrix generation and file I/O
#pragma once
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>        // for __half
#include <cuda_bf16.h>        // for __nv_bfloat16
#include "config.h"  // For MatrixType and DistributionType

// File I/O functions
bool file_exists(const char* filename);
void generate_matrix_filename(char* filename, size_t filename_size, MatrixType type, int n);
bool write_matrix_to_file(const char* filename, float* matrix, int n);
bool read_matrix_from_file(const char* filename, float* matrix, int n);

// Matrix generation functions
void generate_matrix_svd(float* d_A, int n, float cond_num);
void generate_matrix_svd_with_seed(float* d_A, int n, float cond_num, unsigned long long seed);
void generate_matrix_distribution(float* d_matrix, int m, int n, DistributionType dist_type,
                                 float param1, float param2);
void generate_matrix_distribution_with_seed(float* d_matrix, int m, int n, DistributionType dist_type,
                                           float param1, float param2, unsigned long long seed);

// Keep existing float-based function for backward compatibility
void generate_matrix_device_with_seed(float* d_matrix, int n, MatrixType type, unsigned long long seed);

// Template function declaration (NO specializations in header - just the general template)
template<typename T>
void generate_matrix_device_with_seed_typed(T* d_matrix, int n, MatrixType type, unsigned long long seed,
                                           dim3 numBlocks, dim3 threadsPerBlock);

// Template kernel declaration
template<typename SrcType, typename DstType>
__global__ void convert_matrix_kernel(const SrcType* src, DstType* dst, int n);

// Utility functions
void print_matrix_stats(float* matrix, int n, const char* name);

// Check if in a file or generate
bool get_matrix(float* matrix, int n, MatrixType type, const char* custom_filename);