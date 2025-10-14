#ifndef UTILS_CUH
#define UTILS_CUH
#include <cuda_runtime.h>
#include "config.h"  // For KernelType enum

void fill_matrix(float *mat, int N);
template<typename T>
void fill_matrix_typed(T* mat, int N);
void verify_result(float *A, float *B, float *C, int N);
void compute_C_reference(float *A, float *B, float *C_exact, int N);
void compute_C_reference_gpu_fp64(float *h_A, float *h_B, float *h_C_exact, int N);

void printDevicePerformanceInfo();
void printCacheInfo();

// Statistics computation for arrays
typedef struct {
    double average;
    double std_dev;
    double minimum;
    double maximum;
    double p10;  // 10th percentile
    double p95;  // 95th percentile
} ArrayStats;

void compute_array_statistics(const double* array, int size, ArrayStats* stats);

// Unified kernel dispatch function
void launch_kernel_by_type(KernelType kernel_type, float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);

// Kernel name to type mapping
KernelType getKernelTypeFromName(const char* name);

// Matrix type name to enum mapping
MatrixType getMatrixTypeFromName(const char* name);

void* get_kernel_function_pointer(KernelType kernel_type);

// Reverse conversion: enum to string
const char* kernelTypeToString(KernelType kernel_type);
const char* matrixTypeToString(MatrixType matrix_type);

// For kernel launch and dimension computation
template<KernelType kernel_type>
void compute_kernel_dimensions_template(int n, dim3* threadsPerBlock, dim3* numBlocks);

void compute_kernel_dimensions_dispatch(KernelType kernel_type, int n, dim3* threadsPerBlock, dim3* numBlocks);
void compute_kernel_dimensions_dispatch_1D(int n, int* threadsPerBlock, int* numBlocks);

void compute_dimensions(const char* kernel_name, int n, dim3* threadsPerBlock, dim3* numBlocks);

// Add this function declaration:
const char* getComputeTypeString();
const char* getAccumulateTypeString();

#endif // UTILS_CUH