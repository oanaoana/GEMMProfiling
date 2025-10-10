#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "config.h"  // For SIZES and NUM_SIZES

#define NUM_TESTS 9

extern const char* available_test_names[NUM_TESTS];

// Public API functions

void initialize_benchmark_matrices(float* h_A, float* h_B, float* h_C, int n);

void runAllBenchmarks(bool* enabled_tests, bool* enabled_sizes);
void runKernelBenchmark(KernelType kernel_type, int matrix_size);
void assess_kernel_resources(KernelType kernel_type, int n);

// Implementation detail functions (used internally)
void runBenchmark(int n, KernelType kernel_type,
                  float* h_A, float* h_B, float* h_C,
                  float* d_A, float* d_B, float* d_C,
                  FILE* dataFile);

// Generic occupancy checker for any kernel
void check_kernel_occupancy(void* kernel_func, const char* kernel_name,
                           int threads_per_block, size_t shared_mem_bytes);

// Kernel-specific occupancy analysis
void check_occupancy_for_kernel(KernelType kernel_type, int n);

