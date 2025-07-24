// gemms.cu
#include "gemms.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "cutlass/gemm/device/gemm.h"

extern int g_pitch_A;  // Declare extern to access from benchmark.cu

// Naive implementation
__global__ void matmul_naive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch wrapper for naive implementation
void launch_naive(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, n);
}

// Tiled implementation
__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    // Shared memory with bank conflict avoidance (+1 padding)
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

    // Thread indices
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // Global indices for this thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Accumulator - keep in register
    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Main tiling loop
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile A - coalesced memory access
        int A_row = row;
        int A_col = t * TILE_SIZE + tx;
        tile_A[ty][tx] = (A_row < N && A_col < N) ? A[A_row * N + A_col] : 0.0f;

        // Load tile B - coalesced memory access
        int B_row = t * TILE_SIZE + ty;
        int B_col = col;
        tile_B[ty][tx] = (B_row < N && B_col < N) ? B[B_row * N + B_col] : 0.0f;

        // Wait for all threads to finish loading
        __syncthreads();

        // Compute partial dot product with optimizations
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            // Use fused multiply-add for better performance
            sum = __fmaf_rn(tile_A[ty][k], tile_B[k][tx], sum);
        }

        // Wait before loading next tiles
        __syncthreads();
    }

    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Launch wrapper for tiled implementation
void launch_tiled(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // For now, assume regular allocation (no pitch)
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// CORRECTED rectangular tiled implementation
__global__ void matmul_tiled_rectangular(float *A, float *B, float *C, int N) {

    __shared__ float tile_A[TILE_M][TILE_K + 1];    // 16×32
    __shared__ float tile_B[TILE_K][TILE_N + 1];    // 32×16

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;

    float sum = 0.0f;
    int num_tiles = (N + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // FIXED: Load A tile with proper indexing
        // Each thread loads consecutive elements
        int A_col_base = t * TILE_K;

        // Load first half: threads 0-15 load columns 0-15
        if (tx < TILE_K && row < N && (A_col_base + tx) < N) {
            tile_A[ty][tx] = A[row * N + (A_col_base + tx)];
        } else if (tx < TILE_K) {
            tile_A[ty][tx] = 0.0f;
        }

        // Load second half: threads 0-15 load columns 16-31
        if ((tx + BLOCK_N) < TILE_K && row < N && (A_col_base + tx + BLOCK_N) < N) {
            tile_A[ty][tx + BLOCK_N] = A[row * N + (A_col_base + tx + BLOCK_N)];
        } else if ((tx + BLOCK_N) < TILE_K) {
            tile_A[ty][tx + BLOCK_N] = 0.0f;
        }

        // FIXED: Load B tile with proper indexing
        int B_row_base = t * TILE_K;

        // Load first half: threads row 0-15 load rows 0-15
        if (ty < TILE_K && (B_row_base + ty) < N && col < N) {
            tile_B[ty][tx] = B[(B_row_base + ty) * N + col];
        } else if (ty < TILE_K) {
            tile_B[ty][tx] = 0.0f;
        }

        // Load second half: threads row 0-15 load rows 16-31
        if ((ty + BLOCK_M) < TILE_K && (B_row_base + ty + BLOCK_M) < N && col < N) {
            tile_B[ty + BLOCK_M][tx] = B[(B_row_base + ty + BLOCK_M) * N + col];
        } else if ((ty + BLOCK_M) < TILE_K) {
            tile_B[ty + BLOCK_M][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum = __fmaf_rn(tile_A[ty][k], tile_B[k][tx], sum);
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Launch wrapper for rectangular tiled implementation
void launch_tiled_rect(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Use TILE_M, TILE_N for block dimensions
    dim3 rect_threads(BLOCK_N, BLOCK_M);
    dim3 rect_blocks((n + TILE_N - 1) / TILE_N, (n + TILE_M - 1) / TILE_M);

    matmul_tiled_rectangular<<<rect_blocks, rect_threads>>>(d_A, d_B, d_C, n);
}

// cuBLAS implementation
void launch_cublas(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Setup alpha and beta for sgemm
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Call cuBLAS sgemm
    // Note: cuBLAS uses column-major order while our code uses row-major order
    // So we compute C = B*A as a workaround for row-major C = A*B
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_B, n,  // B matrix
                d_A, n,  // A matrix
                &beta,
                d_C, n); // C matrix

    // Destroy handle
    cublasDestroy(handle);
}

// Add TensorCore implementation
void launch_cublas_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Enable TensorCore math if supported
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_B, n,
                d_A, n,
                &beta,
                d_C, n);

    cublasDestroy(handle);
}

// CUTLASS implementation
void launch_cutlass(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Define CUTLASS GEMM kernel
    using Gemm = cutlass::gemm::device::Gemm<
        float,                           // Data type of A matrix
        cutlass::layout::RowMajor,       // Layout of A matrix
        float,                           // Data type of B matrix
        cutlass::layout::RowMajor,       // Layout of B matrix
        float,                           // Data type of C matrix
        cutlass::layout::RowMajor,       // Layout of C matrix
        float                            // Data type for internal accumulation
    >;

    // CUTLASS GEMM arguments
    Gemm::Arguments arguments{
        {n, n, n},          // Problem size (M, N, K)
        {d_A, n},           // Tensor A (ptr, leading dimension)
        {d_B, n},           // Tensor B (ptr, leading dimension)
        {d_C, n},           // Tensor C (ptr, leading dimension)
        {d_C, n},           // Tensor D (ptr, leading dimension) - output
        {1.0f, 0.0f}        // Scalars alpha, beta
    };

    // Initialize CUTLASS GEMM
    Gemm gemm_op;

    // Check if arguments are valid
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM cannot implement these arguments\n");
        return;
    }

    // Initialize the GEMM operator
    status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("Failed to initialize CUTLASS GEMM\n");
        return;
    }

    // Launch the GEMM kernel
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM kernel launch failed\n");
        return;
    }

    // Synchronize
    cudaDeviceSynchronize();
}

// Optional: Advanced CUTLASS with Tensor Cores
void launch_cutlass_tensor(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Using mixed precision for Tensor Cores (requires input conversion)
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,                 // Data type of A matrix (FP16)
        cutlass::layout::RowMajor,       // Layout of A matrix
        cutlass::half_t,                 // Data type of B matrix (FP16)
        cutlass::layout::RowMajor,       // Layout of B matrix
        float,                           // Data type of C matrix (FP32)
        cutlass::layout::RowMajor,       // Layout of C matrix
        float,                           // Data type for internal accumulation
        cutlass::arch::OpClassTensorOp,  // Use Tensor Cores
        cutlass::arch::Sm80              // Target SM 8.0+ for RTX 4080
    >;

    // Note: This requires converting FP32 inputs to FP16
    // For simplicity, using regular CUTLASS for now
    // In production, you'd convert d_A and d_B to half precision

    printf("CUTLASS Tensor Core version requires FP16 input conversion\n");
    // Fall back to regular CUTLASS
    launch_cutlass(d_A, d_B, d_C, n, blocks, threads);
}

// Add pitched kernel and launch function
__global__ void matmul_tiled_pitch(float *A, float *B, float *C, int N, int pitch_A) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    //printf("Debug: Pitched kernel launched with pitch_A = %d\n", pitch_A);
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_row = row;
        int A_col = t * TILE_SIZE + tx;
        int B_row = t * TILE_SIZE + ty;
        int B_col = col;

        // Load A tile with pitch
        if (A_row < N && A_col < N) {
            tile_A[ty][tx] = A[A_row * pitch_A + A_col];  // Use pitch
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        // Load B tile (regular)
        if (B_row < N && B_col < N) {
            tile_B[ty][tx] = B[B_row * N + B_col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum = __fmaf_rn(tile_A[ty][k], tile_B[k][tx], sum);
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_tiled_pitched(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads, int pitch_A) {
    matmul_tiled_pitch<<<blocks, threads>>>(d_A, d_B, d_C, n, pitch_A);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// Add this wrapper function to gemms.cu
void launch_tiled_pitched_wrapper(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Use global pitch value set in benchmark
    launch_tiled_pitched(d_A, d_B, d_C, n, blocks, threads, g_pitch_A);
}
