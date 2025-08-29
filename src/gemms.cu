// gemms.cu
#include "../include/gemms.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "cutlass/gemm/device/gemm.h"
#include <vector_types.h>  // For float4 definition

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
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
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

// Launch wrapper for tiled implementation with consistent signature
void launch_tiled(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Use compile-time constants for fixed tiling
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
}

// Optimized tiled implementation with transposed B for better memory access
__global__ void matmul_tiled_opt(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int N) {
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
        // Vectorized load for tile A - only threads with tx % 4 == 0 do the wide load
        int A_row = row;
        int A_col = t * TILE_SIZE + tx;

        if ((tx & 3) == 0) {
            float4 vec = {0,0,0,0};
            if (A_row < N && A_col + 3 < N) {
                // 16B-aligned if A_col % 4 == 0 and base from cudaMalloc
                vec = *reinterpret_cast<const float4*>(&A[A_row * N + A_col]);
            } else {
                // edge-safe scalar gather (rare path)
                vec.x = (A_row < N && A_col + 0 < N) ? A[A_row*N + A_col + 0] : 0.f;
                vec.y = (A_row < N && A_col + 1 < N) ? A[A_row*N + A_col + 1] : 0.f;
                vec.z = (A_row < N && A_col + 2 < N) ? A[A_row*N + A_col + 2] : 0.f;
                vec.w = (A_row < N && A_col + 3 < N) ? A[A_row*N + A_col + 3] : 0.f;
            }
            // write into shared as scalars (avoid 16B alignment issues with +1 padding)
            tile_A[ty][tx + 0] = vec.x;
            tile_A[ty][tx + 1] = vec.y;
            tile_A[ty][tx + 2] = vec.z;
            tile_A[ty][tx + 3] = vec.w;
        }

        // Do the same for B (note B_col varies with tx, so same guard works)
        int B_row = t * TILE_SIZE + ty;
        int B_col = col;
        if ((tx & 3) == 0) {
            float4 vec = {0,0,0,0};
            if (B_row < N && B_col + 3 < N) {
                vec = *reinterpret_cast<const float4*>(&B[B_row * N + B_col]);
            } else {
                vec.x = (B_row < N && B_col + 0 < N) ? B[B_row*N + B_col + 0] : 0.f;
                vec.y = (B_row < N && B_col + 1 < N) ? B[B_row*N + B_col + 1] : 0.f;
                vec.z = (B_row < N && B_col + 2 < N) ? B[B_row*N + B_col + 2] : 0.f;
                vec.w = (B_row < N && B_col + 3 < N) ? B[B_row*N + B_col + 3] : 0.f;
            }
            tile_B[ty][tx + 0] = vec.x;
            tile_B[ty][tx + 1] = vec.y;
            tile_B[ty][tx + 2] = vec.z;
            tile_B[ty][tx + 3] = vec.w;
        }

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

// Launch wrapper for optimized tiled implementation
void launch_tiled_opt(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    matmul_tiled_opt<<<blocks, threads>>>(d_A, d_B, d_C, n);
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

void launch_tiled_pairwise(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Use compile-time constants for fixed tiling
    matmul_tiled_pairwise<<<blocks, threads>>>(d_A, d_B, d_C, n);
}

// Launch wrapper for rectangular tiled implementation
void launch_tiled_rect(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Use TILE_M, TILE_N for block dimensions
    dim3 rect_threads(BLOCK_N, BLOCK_M);
    dim3 rect_blocks((n + TILE_N - 1) / TILE_N, (n + TILE_M - 1) / TILE_M);

    matmul_tiled_rectangular<<<rect_blocks, rect_threads>>>(d_A, d_B, d_C, n);
}

__global__ void matmul_tiled_pairwise(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // Shared memory with bank-conflict padding
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

    // Thread / block indices
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;

    // Global output indices for this thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Number of K-tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Compute exact number of levels needed for pairwise summation
    // This is ceil(log2(num_tiles)) using efficient bit manipulation
    //int needed_levels = (num_tiles <= 1) ? 1 : (32 - __clz(num_tiles - 1));

    // --- ONLINE PAIRWISE STACK (per thread) ---
    // Each level holds one partial sum for this (row,col).
    float level_acc[MAX_LEVELS];
    #pragma unroll
    for (int l = 0; l < MAX_LEVELS; ++l) level_acc[l] = 0.0f;
    unsigned int occ_mask = 0u; // bit l set ⇒ level l occupied

    // Main tiling loop over K in chunks of TILE_SIZE
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A (row-major)
        int A_row = row;
        int A_col0 = t * TILE_SIZE + tx;
        tile_A[ty][tx] = (A_row < N && A_col0 < N) ? A[A_row * N + A_col0] : 0.0f;

        // Load tile of B (row-major)
        int B_row0 = t * TILE_SIZE + ty;
        int B_col  = col;
        tile_B[ty][tx] = (B_row0 < N && B_col < N) ? B[B_row0 * N + B_col] : 0.0f;

        __syncthreads();

        // --- compute partial for this K-tile (depth = TILE_SIZE) ---
        float p = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            p = __fmaf_rn(tile_A[ty][k], tile_B[k][tx], p);
        }

        __syncthreads(); // safe to reuse shared tiles next iteration

        // --- ONLINE PAIRWISE INSERT of partial p ---
        int l = 0;
        // Carry-add up the occupied levels like binary addition
        while ((occ_mask & (1u << l)) != 0u) {
            p += level_acc[l];
            occ_mask &= ~(1u << l);
            ++l;
            // Guard against overflow of MAX_LEVELS in extreme num_tiles
            if (l >= MAX_LEVELS - 1) break;
        }
        level_acc[l] = p;
        occ_mask |= (1u << l);
    }

    // Final fold: sum the remaining occupied levels
    float result = 0.0f;
    #pragma unroll
    for (int l = 0; l < MAX_LEVELS; ++l) {
        if (occ_mask & (1u << l)) result += level_acc[l];
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}
// cuBLAS implementation
void launch_cublas(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set math mode - use pedantic for true FP32 (disable TF32)
    // Comment/uncomment the next line to toggle between pedantic FP32 and TF32 modes
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);  // Forces true FP32, no TF32

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

// Square tiled implementation with dynamic shared memory
__global__ void matmul_tiled_square(float *A, float *B, float *C, int N, int tile_size) {
    // Use dynamic shared memory or template parameter
    extern __shared__ float shared_mem[];
    float* tile_A = shared_mem;                           // [tile_size][tile_size]
    float* tile_B = &shared_mem[tile_size * tile_size];   // [tile_size][tile_size]

    // Thread indices
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // Global indices for this thread
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    // Accumulator - keep in register
    float sum = 0.0f;
    int num_tiles = (N + tile_size - 1) / tile_size;

    // Main tiling loop
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile A - coalesced memory access
        int A_row = row;
        int A_col = t * tile_size + tx;
        tile_A[ty * tile_size + tx] = (A_row < N && A_col < N) ? A[A_row * N + A_col] : 0.0f;

        // Load tile B - coalesced memory access
        int B_row = t * tile_size + ty;
        int B_col = col;
        tile_B[ty * tile_size + tx] = (B_row < N && B_col < N) ? B[B_row * N + B_col] : 0.0f;

        // Wait for all threads to finish loading
        __syncthreads();

        // Compute partial dot product with optimizations
        #pragma unroll
        for (int k = 0; k < tile_size; ++k) {
            // Use fused multiply-add for better performance
            sum = __fmaf_rn(tile_A[ty * tile_size + k], tile_B[k * tile_size + tx], sum);
        }

        // Wait before loading next tiles
        __syncthreads();
    }

    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
