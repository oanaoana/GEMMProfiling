#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "config.h"  // Include for all configuration constants and types

// CUDA kernel function declarations
__global__ void matmul_naive(float *A, float *B, float *C, int N);
__global__ void matmul_tiled(float *A, float *B, float *C, int N);
__global__ void matmul_tiled_rectangular(float *A, float *B, float *C, int N);
__global__ void matmul_tiled_square(float *A, float *B, float *C, int N, int tile_size);

// Kernel launch wrappers - all with consistent signature for benchmark
void launch_naive(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled_rect(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled_rectangular(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);

// Configuration-aware wrapper
void launch_tiled_config(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads, TileConfig* tiles);

// cuBLAS wrappers - need to be adapted to consistent signature
void launch_cublas(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cublas_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);

// CUTLASS wrappers
void launch_cutlass(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cutlass_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);

// Original cuBLAS wrapper that takes handle
void launch_cublas_handle(cublasHandle_t handle, float* d_A, float* d_B, float* d_C, int n);

// Utility functions
void printDeviceInfo();
void checkCudaError(cudaError_t error, const char* function);
