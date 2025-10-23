#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "config.h"  // For SIZES and NUM_SIZES

// Public API functions

void initialize_benchmark_matrices(float* h_A, float* h_B, float* h_C, int n);

void runKernelPerformance(KernelType kernel_type, int matrix_size);
void assess_kernel_resources(KernelType kernel_type, int n);

bool validate_benchmark_precision_requirements(KernelType kernel_type);

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


