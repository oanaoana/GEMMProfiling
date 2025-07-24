#pragma once
#include <cuda_runtime.h>

// Square tiling configuration (original)
#define TILE_SIZE 16     // For square tiles: 16×16
#define BLOCK_SIZE 16     // Thread block size for square

// Rectangular tiling configuration
#define TILE_M 16      // Output tile height
#define TILE_N 16      // Output tile width
#define TILE_K 32      // Shared dimension (2x bigger!)

#define BLOCK_M 16     // Thread block height
#define BLOCK_N 16     // Thread block width

// Kernel declarations
__global__ void matmul_naive(float *A, float *B, float *C, int N);
__global__ void matmul_tiled(float *A, float *B, float *C, int N);               // Square version
__global__ void matmul_tiled_rectangular(float *A, float *B, float *C, int N);   // Rectangular version

// Launch wrappers
void launch_naive(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_tiled(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);           // Square
void launch_tiled_rect(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);      // Rectangular
void launch_cublas(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cublas_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cutlass(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
void launch_cutlass_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);

// Pitched kernel
void launch_tiled_pitched(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads, int pitch_A);
void launch_tiled_pitched_wrapper(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads);
