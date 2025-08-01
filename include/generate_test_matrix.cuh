// generate_test_matrix.cuh - Header for matrix generation and file I/O
#pragma once
#include "../include/error_tests.cuh"
#include <stdbool.h>

// File I/O functions
bool file_exists(const char* filename);
void generate_matrix_filename(char* filename, size_t filename_size, MatrixType type, int n);
bool write_matrix_to_file(const char* filename, float* matrix, int n);
bool read_matrix_from_file(const char* filename, float* matrix, int n);

// Matrix generation functions
void generate_matrix_svd(float* d_A, int n, float cond_num);
bool get_matrix(float* matrix, int n, MatrixType type, const char* custom_filename);

// Utility functions
void print_matrix_stats(float* matrix, int n, const char* name);
