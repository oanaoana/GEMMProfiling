#pragma once
#include <cuda_runtime.h>

// Tile and block sizes for regular tiled implementation
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Tile sizes for rectangular tiled implementation
#ifndef TILE_M
#define TILE_M 16
#endif

#ifndef TILE_N
#define TILE_N 16
#endif

#ifndef TILE_K
#define TILE_K 32
#endif

#ifndef BLOCK_M
#define BLOCK_M 16
#endif

#ifndef BLOCK_N
#define BLOCK_N 16
#endif

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
