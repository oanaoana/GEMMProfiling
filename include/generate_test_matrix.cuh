// generate_test_matrix.cuh - Header for matrix generation and file I/O
#pragma once
#include <stdbool.h>
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

// Efficient matrix generation for multi-sample analysis (no file I/O, no memory allocation)
void generate_matrix_device_with_seed(float* d_matrix, int n, MatrixType type, unsigned long long seed);

// Utility functions
void print_matrix_stats(float* matrix, int n, const char* name);

// Check if in a file or generate
bool get_matrix(float* matrix, int n, MatrixType type, const char* custom_filename);