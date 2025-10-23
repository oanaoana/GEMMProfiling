#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "config.h"  // Include for all configuration constants and types

// CUDA kernel function declarations
__global__ void matmul_naive(float *A, float *B, float *C, int N);
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N);
__global__ void matmul_tiled_opt(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N);
__global__ void matmul_tiled_pairwise(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N);
__global__ void matmul_tiled_rectangular(float *A, float *B, float *C, int N);
__global__ void matmul_tiled_square(float *A, float *B, float *C, int N, int tile_size);
template <typename ComputeType, typename AccumulateType>
__global__ void matmul_tiled_mixprec(
    const ComputeType* __restrict__ A,       // CORRECT - inputs in ComputeType
    const ComputeType* __restrict__ B,       // CORRECT - inputs in ComputeType
    AccumulateType* __restrict__ C,          // CORRECT - output in AccumulateType
    int N);

template <typename ComputeType, typename AccumulateType>
__global__ void matmul_tiled_pairwise_mixprec(
    const ComputeType* __restrict__ A,       // CORRECT - inputs in ComputeType
    const ComputeType* __restrict__ B,       // CORRECT - inputs in ComputeType
    AccumulateType* __restrict__ C,          // CORRECT - output in AccumulateType
    int N);

// Kernel launch wrappers - all with consistent signature for benchmark
void launch_naive(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled_opt(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled_pairwise(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled_rect(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled_rectangular(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
// Mixed precision kernel launches
template<typename ComputeType, typename AccumulateType>
void launch_tiled_mixprec(ComputeType* d_A, ComputeType* d_B, AccumulateType* d_C, int n, dim3 blocks, dim3 threads);

template<typename ComputeType, typename AccumulateType>
void launch_tiled_pairwise_mixprec(ComputeType* d_A, ComputeType* d_B, AccumulateType* d_C, int n, dim3 blocks, dim3 threads);

// cuBLAS wrappers - need to be adapted to consistent signature
void launch_cublas(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cublas_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);

// CUTLASS wrappers
void launch_cutlass(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cutlass_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cutlass_splitk_flat(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cutlass_splitk_pairwise(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);

// Template version of cutlass_splitk_pairwise for compile-time slice counts
template<int SLICES>
void cutlass_splitk_pairwise_template(int M, int N, int K,
                                      const float* dA, int lda,
                                      const float* dB, int ldb,
                                      float* dC, int ldc);

void cutlass_splitk_pairwise(int M, int N, int K,
                            const float* dA, int lda,
                            const float* dB, int ldb,
                            float* dC, int ldc);

// Original cuBLAS wrapper that takes handle
void launch_cublas_handle(cublasHandle_t handle, float* d_A, float* d_B, float* d_C, int n);

// Utility functions
void printDeviceInfo();
void checkCudaError(cudaError_t error, const char* function);
