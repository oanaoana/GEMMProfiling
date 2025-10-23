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
template<typename ComputeType, typename AccumulateType>
void launch_mixprec_kernel_by_type(KernelType kernel_type, ComputeType* d_A, ComputeType* d_B, AccumulateType* d_C, int n, dim3 blocks, dim3 threads);

void launch_basic_kernel_by_type(KernelType kernel_type, float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);

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

// Type information structure
struct TypeInfo {
    const char* name;        // "FP32", "FP16", "FP64"
    const char* cuda_type;   // "float", "__half", "double"
    size_t size_bytes;       // 4, 2, 8
    bool supports_mixed;     // Can this be used in mixed precision?
};

// Get type info for current compilation
TypeInfo getComputeTypeInfo();
TypeInfo getAccumulateTypeInfo();

// Keep existing functions for backward compatibility
const char* getComputeTypeString();
const char* getAccumulateTypeString();

// New helper functions
const char* getTypeNameFromSize(size_t bytes);
bool areBothTypesFP32();  // Helper for folder naming logic
bool is_mixprec_kernel(KernelType kernel_type);
void validate_precision_settings(KernelType kernel_type);

#endif // UTILS_CUH