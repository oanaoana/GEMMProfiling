#pragma once
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 16

// Kernel declarations
__global__ void matmul_naive(float *A, float *B, float *C, int N);
__global__ void matmul_tiled(float *A, float *B, float *C, int N);

// Launch wrappers
void launch_naive(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cublas(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cublas_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
