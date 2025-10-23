// gemms.cu
#include "../include/gemms.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "cutlass/gemm/device/gemm.h"
#include <vector_types.h>
#include <vector>

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

void launch_tiled_pairwise(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    // Use compile-time constants for fixed tiling
    matmul_tiled_pairwise<<<blocks, threads>>>(d_A, d_B, d_C, n);
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

#define CUDA_CHECK_CUTLASS(stmt) do {                               \
    cudaError_t err = (stmt);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                cudaGetErrorString(err), __FILE__, __LINE__);       \
        std::exit(EXIT_FAILURE);                                    \
    }                                                               \
} while(0)

#define CUTLASS_CHECK(status) do {                                  \
    cutlass::Status s = (status);                                   \
    if (s != cutlass::Status::kSuccess) {                           \
        fprintf(stderr, "CUTLASS error at %s:%d: %d\n",             \
                __FILE__, __LINE__, int(s));                        \
        std::exit(EXIT_FAILURE);                                    \
    }                                                               \
} while(0)

// Kernel for pairwise reduction of partial results
__global__ void add_inplace_rn(float* __restrict__ L,
                               const float* __restrict__ R,
                               int M, int N, int ldc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    for (int t = tid; t < total; t += blockDim.x * gridDim.x) {
        int i = t / N;               // row
        int j = t - i * N;           // col
        int idx = i * ldc + j;
        L[idx] = __fadd_rn(L[idx], R[idx]);
    }
}

// Flat split-K GEMM implementation (sequential accumulation)
void cutlass_splitk_flat(int M, int N, int K,
                         const float* dA, int lda,
                         const float* dB, int ldb,
                         float* dC, int ldc,
                         int split_k_slices) {

    // Pre-allocate all temporary memory to avoid allocation overhead
    std::vector<float*> A_slices(split_k_slices);
    std::vector<float*> B_slices(split_k_slices);
    std::vector<cudaStream_t> streams(split_k_slices);

    // Calculate slice sizes
    std::vector<int> k_starts(split_k_slices);
    std::vector<int> k_sizes(split_k_slices);

    for (int s = 0; s < split_k_slices; s++) {
        k_starts[s] = (s * K) / split_k_slices;
        int k_end = ((s + 1) * K) / split_k_slices;
        k_sizes[s] = k_end - k_starts[s];

        if (k_sizes[s] > 0) {
            cudaMalloc(&A_slices[s], M * k_sizes[s] * sizeof(float));
            cudaMalloc(&B_slices[s], k_sizes[s] * N * sizeof(float));
            cudaStreamCreate(&streams[s]);
        }
    }

    // Launch all slice preparations in parallel using streams
    for (int s = 0; s < split_k_slices; s++) {
        if (k_sizes[s] <= 0) continue;

        int k0 = k_starts[s];
        int Ks = k_sizes[s];

        // Use 2D memcpy for better performance
        cudaMemcpy2DAsync(A_slices[s], Ks * sizeof(float),
                         dA + k0, lda * sizeof(float),
                         Ks * sizeof(float), M,
                         cudaMemcpyDeviceToDevice, streams[s]);

        cudaMemcpy2DAsync(B_slices[s], N * sizeof(float),
                         dB + k0 * ldb, ldb * sizeof(float),
                         N * sizeof(float), Ks,
                         cudaMemcpyDeviceToDevice, streams[s]);
    }

    // Execute GEMM operations sequentially with accumulation
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float>;

    for (int s = 0; s < split_k_slices; s++) {
        if (k_sizes[s] <= 0) continue;

        // Wait for this slice's data to be ready
        cudaStreamSynchronize(streams[s]);

        cutlass::gemm::GemmCoord problem{M, N, k_sizes[s]};
        float beta = (s == 0) ? 0.0f : 1.0f;  // First slice overwrites, subsequent accumulate

        typename Gemm::Arguments arguments(
            problem,
            {A_slices[s], k_sizes[s]},
            {B_slices[s], N},
            {dC, ldc},         // Input for accumulation (when beta=1)
            {dC, ldc},         // Output
            {1.0f, beta}       // alpha=1, beta=0 (first) or beta=1 (accumulate)
        );

        Gemm gemm;
        auto status = gemm.initialize(arguments);
        if (status == cutlass::Status::kSuccess) {
            gemm();
        } else {
            printf("CUTLASS error in slice %d\n", s);
        }
    }

    // Cleanup
    for (int s = 0; s < split_k_slices; s++) {
        if (k_sizes[s] > 0) {
            cudaFree(A_slices[s]);
            cudaFree(B_slices[s]);
            cudaStreamDestroy(streams[s]);
        }
    }

    cudaDeviceSynchronize();
}

void launch_cutlass_splitk_flat(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    int split_k_slices = SPLIT_K_SLICES;
    //printf("Running CUTLASS Split-K (flat) with %d slices for %dx%d matrix...\n", split_k_slices, n, n);
    cutlass_splitk_flat(n, n, n, d_A, n, d_B, n, d_C, n, split_k_slices);
    cudaDeviceSynchronize();
}

// Pairwise split-K GEMM implementation (tree reduction)
void cutlass_splitk_pairwise(int M, int N, int K,
                             const float* dA, int lda,
                             const float* dB, int ldb,
                             float* dC, int ldc,
                             int split_k_slices) {

    // Pre-allocate all temporary memory to avoid allocation overhead
    std::vector<float*> A_slices(split_k_slices);
    std::vector<float*> B_slices(split_k_slices);
    std::vector<float*> C_workspaces(split_k_slices);  // Separate workspaces for each slice
    std::vector<cudaStream_t> streams(split_k_slices);

    // Calculate slice sizes
    std::vector<int> k_starts(split_k_slices);
    std::vector<int> k_sizes(split_k_slices);

    size_t bytesC = size_t(ldc) * size_t(M) * sizeof(float);

    for (int s = 0; s < split_k_slices; s++) {
        k_starts[s] = (s * K) / split_k_slices;
        int k_end = ((s + 1) * K) / split_k_slices;
        k_sizes[s] = k_end - k_starts[s];

        if (k_sizes[s] > 0) {
            cudaMalloc(&A_slices[s], M * k_sizes[s] * sizeof(float));
            cudaMalloc(&B_slices[s], k_sizes[s] * N * sizeof(float));
            cudaMalloc(&C_workspaces[s], bytesC);  // Workspace for this slice
            cudaMemset(C_workspaces[s], 0, bytesC);  // Zero initialize
            cudaStreamCreate(&streams[s]);
        }
    }

    // Launch all slice preparations in parallel using streams
    for (int s = 0; s < split_k_slices; s++) {
        if (k_sizes[s] <= 0) continue;

        int k0 = k_starts[s];
        int Ks = k_sizes[s];

        // Use 2D memcpy for better performance
        cudaMemcpy2DAsync(A_slices[s], Ks * sizeof(float),
                         dA + k0, lda * sizeof(float),
                         Ks * sizeof(float), M,
                         cudaMemcpyDeviceToDevice, streams[s]);

        cudaMemcpy2DAsync(B_slices[s], N * sizeof(float),
                         dB + k0 * ldb, ldb * sizeof(float),
                         N * sizeof(float), Ks,
                         cudaMemcpyDeviceToDevice, streams[s]);
    }

    // Execute GEMM operations into separate workspaces in parallel!
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float>;

    std::vector<Gemm> gemm_ops(split_k_slices);

    for (int s = 0; s < split_k_slices; s++) {
        if (k_sizes[s] <= 0) continue;

        // Wait for this slice's data to be ready
        cudaStreamSynchronize(streams[s]);

        cutlass::gemm::GemmCoord problem{M, N, k_sizes[s]};

        typename Gemm::Arguments arguments(
            problem,
            {A_slices[s], k_sizes[s]},
            {B_slices[s], N},
            {C_workspaces[s], ldc},  // Output to workspace (not accumulating)
            {C_workspaces[s], ldc},
            {1.0f, 0.0f}             // alpha=1, beta=0 (no accumulation, fresh workspace)
        );

        auto status = gemm_ops[s].initialize(arguments);
        if (status == cutlass::Status::kSuccess) {
            // Launch on the slice's stream for parallelism
            cudaStream_t stream = streams[s];
            gemm_ops[s](stream);
        } else {
            printf("CUTLASS error in slice %d\n", s);
        }
    }

    cudaDeviceSynchronize();

    // Now perform pairwise reduction of workspaces
    std::vector<float*> work = C_workspaces;  // Copy workspace pointers for reduction
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;

    while (work.size() > 1) {
        std::vector<float*> next;
        for (size_t i = 0; i + 1 < work.size(); i += 2) {
            // Add work[i+1] into work[i] in-place
            add_inplace_rn<<<blocks, threads>>>(work[i], work[i+1], M, N, ldc);
            cudaDeviceSynchronize();
            next.push_back(work[i]);  // Left workspace holds the sum
        }
        if (work.size() & 1) {
            next.push_back(work.back());  // Carry odd one up
        }
        work.swap(next);
    }

    // Copy final result to output
    cudaMemcpy2D(dC, sizeof(float) * ldc,
                 work[0], sizeof(float) * ldc,
                 sizeof(float) * N, M, cudaMemcpyDeviceToDevice);

    // Cleanup
    for (int s = 0; s < split_k_slices; s++) {
        if (k_sizes[s] > 0) {
            cudaFree(A_slices[s]);
            cudaFree(B_slices[s]);
            cudaFree(C_workspaces[s]);
            cudaStreamDestroy(streams[s]);
        }
    }

    cudaDeviceSynchronize();
}

// Template version of cutlass_splitk_pairwise for compile-time slice counts
template<int SLICES>
void cutlass_splitk_pairwise_template(int M, int N, int K,
                                      const float* dA, int lda,
                                      const float* dB, int ldb,
                                      float* dC, int ldc) {
    const int split_k_slices = SLICES;

    // Pre-allocate all temporary memory to avoid allocation overhead
    std::vector<float*> A_slices(split_k_slices);
    std::vector<float*> B_slices(split_k_slices);
    std::vector<float*> C_workspaces(split_k_slices);  // Separate workspaces for each slice
    std::vector<cudaStream_t> streams(split_k_slices);

    // Calculate slice sizes
    std::vector<int> k_starts(split_k_slices);
    std::vector<int> k_sizes(split_k_slices);

    size_t bytesC = size_t(ldc) * size_t(M) * sizeof(float);

    for (int s = 0; s < split_k_slices; s++) {
        k_starts[s] = (s * K) / split_k_slices;
        int k_end = ((s + 1) * K) / split_k_slices;
        k_sizes[s] = k_end - k_starts[s];

        if (k_sizes[s] > 0) {
            cudaMalloc(&A_slices[s], M * k_sizes[s] * sizeof(float));
            cudaMalloc(&B_slices[s], k_sizes[s] * N * sizeof(float));
            cudaMalloc(&C_workspaces[s], bytesC);  // Workspace for this slice
            cudaMemset(C_workspaces[s], 0, bytesC);  // Zero initialize
            cudaStreamCreate(&streams[s]);
        }
    }

    // Launch all slice preparations in parallel using streams
    for (int s = 0; s < split_k_slices; s++) {
        if (k_sizes[s] <= 0) continue;

        int k0 = k_starts[s];
        int Ks = k_sizes[s];

        // Use 2D memcpy for better performance
        cudaMemcpy2DAsync(A_slices[s], Ks * sizeof(float),
                         dA + k0, lda * sizeof(float),
                         Ks * sizeof(float), M,
                         cudaMemcpyDeviceToDevice, streams[s]);

        cudaMemcpy2DAsync(B_slices[s], N * sizeof(float),
                         dB + k0 * ldb, ldb * sizeof(float),
                         N * sizeof(float), Ks,
                         cudaMemcpyDeviceToDevice, streams[s]);
    }

    // Execute GEMM operations into separate workspaces in parallel!
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float>;

    std::vector<Gemm> gemm_ops(split_k_slices);

    for (int s = 0; s < split_k_slices; s++) {
        if (k_sizes[s] <= 0) continue;

        // Wait for this slice's data to be ready
        cudaStreamSynchronize(streams[s]);

        cutlass::gemm::GemmCoord problem{M, N, k_sizes[s]};

        typename Gemm::Arguments arguments(
            problem,
            {A_slices[s], k_sizes[s]},
            {B_slices[s], N},
            {C_workspaces[s], ldc},  // Output to workspace (not accumulating)
            {C_workspaces[s], ldc},
            {1.0f, 0.0f}             // alpha=1, beta=0 (no accumulation, fresh workspace)
        );

        auto status = gemm_ops[s].initialize(arguments);
        if (status == cutlass::Status::kSuccess) {
            // Launch on the slice's stream for parallelism
            cudaStream_t stream = streams[s];
            gemm_ops[s](stream);
        } else {
            printf("CUTLASS error in slice %d\n", s);
        }
    }

    cudaDeviceSynchronize();

    // Now perform pairwise reduction of workspaces
    std::vector<float*> work = C_workspaces;  // Copy workspace pointers for reduction
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;

    while (work.size() > 1) {
        std::vector<float*> next;
        for (size_t i = 0; i + 1 < work.size(); i += 2) {
            // Add work[i+1] into work[i] in-place
            add_inplace_rn<<<blocks, threads>>>(work[i], work[i+1], M, N, ldc);
            cudaDeviceSynchronize();
            next.push_back(work[i]);  // Left workspace holds the sum
        }
        if (work.size() & 1) {
            next.push_back(work.back());  // Carry odd one up
        }
        work.swap(next);
    }

    // Copy final result to output
    cudaMemcpy2D(dC, sizeof(float) * ldc,
                 work[0], sizeof(float) * ldc,
                 sizeof(float) * N, M, cudaMemcpyDeviceToDevice);

    // Cleanup
    for (int s = 0; s < split_k_slices; s++) {
        if (k_sizes[s] > 0) {
            cudaFree(A_slices[s]);
            cudaFree(B_slices[s]);
            cudaFree(C_workspaces[s]);
            cudaStreamDestroy(streams[s]);
        }
    }

    cudaDeviceSynchronize();
}

// Launch wrapper for CUTLASS Split-K pairwise (tree reduction) kernel
void launch_cutlass_splitk_pairwise(float* d_A, float* d_B, float* d_C, int n, dim3 blocks, dim3 threads) {
    int split_k_slices = SPLIT_K_SLICES;
    //printf("Running CUTLASS Split-K (pairwise) with %d slices for %dx%d matrix...\n", split_k_slices, n, n);
    cutlass_splitk_pairwise(n, n, n, d_A, n, d_B, n, d_C, n, split_k_slices);
    cudaDeviceSynchronize();
}

// Revised mixed-precision FMA supporting only the 4 required cases
template<typename ComputeType, typename AccumType>
__device__ __forceinline__ AccumType mixprec_fma(ComputeType a, ComputeType b, AccumType c) {
    // Case 1: Baseline (Standard) - FP32 compute, FP32 accumulate
    if constexpr (std::is_same_v<ComputeType, float> && std::is_same_v<AccumType, float>) {
        return __fmaf_rn(a, b, c);
    }
    // Case 2: Mixed (Tensor Core Emulation) - FP16 compute, FP32 accumulate
    else if constexpr (std::is_same_v<ComputeType, __half> && std::is_same_v<AccumType, float>) {
        return __fmaf_rn(__half2float(a), __half2float(b), c);
    }
    // Case 3: Mixed (LLM Focus) - BF16 compute, FP32 accumulate
    else if constexpr (std::is_same_v<ComputeType, __nv_bfloat16> && std::is_same_v<AccumType, float>) {
        return __fmaf_rn(__bfloat162float(a), __bfloat162float(b), c);
    }
    // Case 4: High Accuracy (Double) - FP32 compute, FP64 accumulate
    else if constexpr (std::is_same_v<ComputeType, float> && std::is_same_v<AccumType, double>) {
        return __fma_rn(static_cast<double>(a), static_cast<double>(b), c);
    }
    else {
        // Static assertion to catch unsupported combinations at compile time
        static_assert(
            (std::is_same_v<ComputeType, float> && std::is_same_v<AccumType, float>) ||
            (std::is_same_v<ComputeType, __half> && std::is_same_v<AccumType, float>) ||
            (std::is_same_v<ComputeType, __nv_bfloat16> && std::is_same_v<AccumType, float>) ||
            (std::is_same_v<ComputeType, float> && std::is_same_v<AccumType, double>),
            "Unsupported precision combination for mixprec_fma. Only fp32/fp32, fp16/fp32, bf16/fp32, fp32/fp64 are supported."
        );

        // This should never be reached due to static_assert, but provide fallback
        return c + static_cast<AccumType>(a) * static_cast<AccumType>(b);
    }
}

template <typename ComputeType, typename AccumulateType>
__global__ void matmul_tiled_mixprec(
    const ComputeType* __restrict__ A,
    const ComputeType* __restrict__ B,
    AccumulateType* __restrict__ C,
    int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ ComputeType tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ ComputeType tile_B[TILE_SIZE][TILE_SIZE];

    AccumulateType sum = static_cast<AccumulateType>(0.0);

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile A - convert from AccumulateType to ComputeType
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (a_row < N && a_col < N) {
            tile_A[threadIdx.y][threadIdx.x] = static_cast<ComputeType>(A[a_row * N + a_col]);
        } else {
            tile_A[threadIdx.y][threadIdx.x] = static_cast<ComputeType>(0.0);
        }

        // Load tile B - convert from AccumulateType to ComputeType
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < N && b_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = static_cast<ComputeType>(B[b_row * N + b_col]);
        } else {
            tile_B[threadIdx.y][threadIdx.x] = static_cast<ComputeType>(0.0);
        }

        __syncthreads();

        // Compute partial sum using the revised mixprec_fma (supports only 4 cases)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum = mixprec_fma(tile_A[threadIdx.y][k], tile_B[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Then your launch function
template <typename ComputeType, typename AccumulateType>
void launch_tiled_mixprec(ComputeType* d_A, ComputeType* d_B, AccumulateType* d_C, int n, dim3 blocks, dim3 threads) {
    matmul_tiled_mixprec<ComputeType, AccumulateType><<<blocks, threads>>>(d_A, d_B, d_C, n);
}

// Zero literal helpers for both float/double (and others via cast)
template <typename T>
__device__ __forceinline__ T zlit() { return static_cast<T>(0.0); }

// ---------- Utility ----------

__device__ __forceinline__ int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

// Compute ceil(log2(x)) for x>=1; returns 0 for x==0 (never used here).
__device__ __forceinline__ int ceil_log2_int_gpu(int x) {
    if (x <= 1) return 0;
    return 32 - __clz(x - 1);
}

template <typename ComputeType, typename AccumulateType>
__global__ void matmul_tiled_pairwise_mixprec(
    const ComputeType* __restrict__ A,
    const ComputeType* __restrict__ B,
    AccumulateType* __restrict__ C,
    int N)
{
    // Thread / block indices
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x,  by = blockIdx.y;

    // Global output indices for this thread
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    // Shared memory tiles with bank-conflict padding (+1 on the inner dimension)
    __shared__ ComputeType tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ ComputeType tile_B[TILE_SIZE][TILE_SIZE + 1];

    const int num_tiles = ceil_div_int(N, TILE_SIZE);

    // --- ONLINE PAIRWISE STACK (per thread) ---
    AccumulateType level_acc[MAX_LEVELS];
    #pragma unroll
    for (int l = 0; l < MAX_LEVELS; ++l) level_acc[l] = zlit<AccumulateType>();
    unsigned int occ_mask = 0u; // bit l set ⇒ level l occupied

    // Main loop over K-tiles
    for (int t = 0; t < num_tiles; ++t) {
        // Load A tile (row-major), converting from AccumulateType to ComputeType
        const int A_row = row;
        const int A_col = t * TILE_SIZE + tx;
        tile_A[ty][tx] =
            (A_row < N && A_col < N)
            ? static_cast<ComputeType>(A[A_row * N + A_col])
            : zlit<ComputeType>();

        // Load B tile (row-major), converting to ComputeType
        const int B_row = t * TILE_SIZE + ty;
        const int B_col = col;
        tile_B[ty][tx] =
            (B_row < N && B_col < N)
            ? static_cast<ComputeType>(B[B_row * N + B_col])
            : zlit<ComputeType>();

        __syncthreads();

        // --- compute the partial for this K-tile (depth = TILE_SIZE) in ComputeType, accumulated in AccumulateType ---
        AccumulateType p = zlit<AccumulateType>();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            p = mixprec_fma(tile_A[ty][k], tile_B[k][tx], p);
        }

        __syncthreads(); // safe to reuse shared tiles next iteration

        // --- ONLINE PAIRWISE INSERT of partial p (binary counter) ---
        int l = 0;
        // Optional: early bound to prevent overflow (num_tiles may exceed MAX_LEVELS coverage)
        // Needed levels ≤ ceil_log2(num_tiles) + 1.
        // We keep the original "carry" pattern and cap at MAX_LEVELS-1.
        while ((occ_mask & (1u << l)) != 0u) {
            p = static_cast<AccumulateType>(p + level_acc[l]);
            occ_mask &= ~(1u << l);
            ++l;
            if (l >= MAX_LEVELS - 1) break; // cap carry chain
        }
        level_acc[l] = p;
        occ_mask |= (1u << l);
    }

    // Final fold of occupied levels
    AccumulateType result = zlit<AccumulateType>();
    #pragma unroll
    for (int l = 0; l < MAX_LEVELS; ++l) {
        if (occ_mask & (1u << l)) result = static_cast<AccumulateType>(result + level_acc[l]);
    }

    // Write-back
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}

// Launch wrapper for tiled pairwise mixed precision
template <typename ComputeType, typename AccumulateType>
void launch_tiled_pairwise_mixprec(ComputeType* d_A, ComputeType* d_B, AccumulateType* d_C, int n, dim3 blocks, dim3 threads) {
    // Direct template instantiation using the configured types
    matmul_tiled_pairwise_mixprec<ComputeType, AccumulateType><<<blocks, threads>>>(d_A, d_B, d_C, n);
}

// Explicit instantiations for the kernels
template __global__ void matmul_tiled_mixprec<COMPUTE_TYPE, ACCUMULATE_TYPE>(
    const COMPUTE_TYPE* A, const COMPUTE_TYPE* B, ACCUMULATE_TYPE* C, int N);

template __global__ void matmul_tiled_pairwise_mixprec<COMPUTE_TYPE, ACCUMULATE_TYPE>(
    const COMPUTE_TYPE* A, const COMPUTE_TYPE* B, ACCUMULATE_TYPE* C, int N);

// Explicit instantiations for the launch functions
template void launch_tiled_mixprec<COMPUTE_TYPE, ACCUMULATE_TYPE>(
    COMPUTE_TYPE* d_A, COMPUTE_TYPE* d_B, ACCUMULATE_TYPE* d_C, int n, dim3 blocks, dim3 threads);

template void launch_tiled_pairwise_mixprec<COMPUTE_TYPE, ACCUMULATE_TYPE>(
    COMPUTE_TYPE* d_A, COMPUTE_TYPE* d_B, ACCUMULATE_TYPE* d_C, int n, dim3 blocks, dim3 threads);
