#ifndef UTILS_CUH
#define UTILS_CUH
#include <cuda_runtime.h>
#include "config.h"  // For KernelType enum

void fill_matrix(float *mat, int N);
void verify_result(float *A, float *B, float *C, int N);
void compute_C_reference(float *A, float *B, float *C_exact, int N);
void compute_C_reference_gpu_fp64(float *h_A, float *h_B, float *h_C_exact, int N);

void printDevicePerformanceInfo();
void printCacheInfo();
void check_occupancy();

// Statistics computation for arrays
typedef struct {
    double average;
    double std_dev;
    double minimum;
    double maximum;
    double p95;  // 95th percentile
} ArrayStats;

void compute_array_statistics(const double* array, int size, ArrayStats* stats);

// Unified kernel dispatch function
void launch_kernel_by_type(KernelType kernel_type, float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);

// Kernel name to type mapping
KernelType getKernelTypeFromName(const char* name);

#endif // UTILS_CUH