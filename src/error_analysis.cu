// error_analysis.cu - Consolidated error analysis functionality
#include "../include/error_analysis.cuh"
#include "../include/config.h"  // For configuration constants and SIZES
#include "../include/generate_test_matrix.cuh"  // For get_matrix and print_matrix_stats
#include "../include/gemms.cuh"
#include "../include/utils.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>  // for std::max and std::sort
#include <string>     // for std::to_string
#include <vector>     // for std::vector
#include <sys/stat.h>
#include <sys/types.h>
#include <typeinfo>
#include <type_traits>  // for std::is_same_v

// Device function for atomic add with double precision
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Helper function to generate reproducible seed arrays
void generate_seed_array(unsigned long long* seeds, int num_samples, unsigned long long base_seed = 0) {
    if (base_seed == 0) {
        // Use time-based seed for truly random behavior
        base_seed = (unsigned long long)time(NULL);
        printf("Using time-based random seed: %llu\n", base_seed);
    } else {
        printf("Using reproducible base seed: %llu\n", base_seed);
    }

    // Generate deterministic but well-distributed seeds
    for (int i = 0; i < num_samples; i++) {
        seeds[i] = base_seed ^ (0x9E3779B97F4A7C15ull + (unsigned long long)i * 0xBF58476D1CE4E5B9ull);
    }
}

// Helper function to generate a specific matrix pair from a reproducible sequence
void generate_matrix_pair_from_sequence(float* d_A, float* d_B, int n, MatrixType matrix_type,
                                       unsigned long long base_seed, int sample_index) {
    // Generate the same seed that would be used for this sample in the sequence
    unsigned long long seedA = base_seed ^ (0x9E3779B97F4A7C15ull + (unsigned long long)sample_index * 0xBF58476D1CE4E5B9ull);
    unsigned long long seedB = seedA ^ 0x94D049BB133111EBull;

    printf("Generating matrix pair from sequence: base_seed=%llu, sample_index=%d\n", base_seed, sample_index);
    printf("  seedA=%llu, seedB=%llu\n", seedA, seedB);

    generate_matrix_device_with_seed(d_A, n, matrix_type, seedA);
    generate_matrix_device_with_seed(d_B, n, matrix_type, seedB);
}

// Function to find the sample index closest to median error from analysis results
// This requires that you've run analysis with reproducible seeds and saved per-sample results
int find_median_sample_index(const float* error_values, int num_samples) {
    // Create a copy of error values with indices for sorting
    std::vector<std::pair<float, int>> indexed_errors;
    for (int i = 0; i < num_samples; i++) {
        indexed_errors.push_back({error_values[i], i});
    }

    // Sort by error value
    std::sort(indexed_errors.begin(), indexed_errors.end());

    // Find median
    int median_index = num_samples / 2;
    int closest_sample_index = indexed_errors[median_index].second;
    float median_error = indexed_errors[median_index].first;

    printf("Median error: %.6e, found at sample index: %d\n", median_error, closest_sample_index);
    return closest_sample_index;
}

// GPU kernels for device-only error computation
__global__ void compute_matrix_abs_kernel(float* matrix, float* abs_matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        abs_matrix[idx] = fabsf(matrix[idx]);
    }
}

// Conversion kernel from double to float
__global__ void convert_fp64_to_fp32_kernel(double* d_input, float* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = (float)d_input[idx];
    }
}
// Template version that works with any input type
template<typename T>
__global__ void compute_matrix_abs_fp64_kernel_typed(const T* matrix, double* abs_matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        abs_matrix[idx] = fabs((double)matrix[idx]);
    }
}

__global__ void compute_matrix_abs_fp64_kernel(float* matrix, double* abs_matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        abs_matrix[idx] = fabs((double)matrix[idx]);
    }
}

template<typename AccumulateType>
__global__ void compute_frobenius_error_kernel(AccumulateType* C_kernel, double* C_reference,
                                             double* abs_AB_product, double* error_results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n;

    // Dynamic shared memory for flexible block sizes
    extern __shared__ double shared_mem[];
    double* shared_error = shared_mem;
    double* shared_norm = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    double local_error = 0.0;
    double local_norm = 0.0;

    // Each thread processes multiple elements if needed
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        double diff = (double)C_kernel[i] - C_reference[i];  // Cast to double for precision
        local_error += diff * diff;

        // Use the precomputed |A|*|B| product
        double norm_val = (double)abs_AB_product[i];
        local_norm += norm_val * norm_val;
    }

    shared_error[tid] = local_error;
    shared_norm[tid] = local_norm;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_error[tid] += shared_error[tid + stride];
            shared_norm[tid] += shared_norm[tid + stride];
        }
        __syncthreads();
    }

    // Write results
    if (tid == 0) {
        atomicAddDouble(&error_results[0], shared_error[0]);  // Frobenius error squared
        atomicAddDouble(&error_results[1], shared_norm[0]);   // Norm squared
    }
}

__global__ void compute_reference_fp64_device(float* A, float* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += (double)A[row * n + k] * (double)B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Modified reference kernel that also computes |A| * |B| on-the-fly
template<typename ComputeType>
__global__ void compute_reference_and_norm_fp64_device(const ComputeType* A, const ComputeType* B,
                                                       double* C_ref, double* abs_AB_product, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double c_sum = 0.0;
        double abs_sum = 0.0;

        for (int k = 0; k < n; k++) {
            double a_val = (double)A[row * n + k];
            double b_val = (double)B[k * n + col];

            // Compute reference: C = A * B
            c_sum += a_val * b_val;

            // Compute normalization: |A| * |B|
            abs_sum += fabs(a_val) * fabs(b_val);
        }

        int idx = row * n + col;
        C_ref[idx] = c_sum;
        abs_AB_product[idx] = abs_sum;
    }
}

inline double gamma(int n, double u) {
    const double nu = n * u;
    return nu / (1.0 - nu);
}

__host__ __device__ inline int ceil_log2_int(int x) {
    return (x <= 1) ? 0 : 32 - __builtin_clz(x - 1);
}

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// γ_k(u) helper with guard (valid if k*u < 1)
__host__ __device__ inline double gamma_bound(int k_eff, double u) {
    if (k_eff <= 0) return 0.0;
    // guard against ku >= 1 (use a soft clamp to stay within model domain)
    const double ku = k_eff * u;
    const double eps = 1e-12;
    if (ku >= 1.0 - 1e-9) {
        const double ku_clamped = fmin(ku, 1.0 - 1e-9);
        return ku_clamped / fmax(1.0 - ku_clamped, eps);
    }
    return ku / (1.0 - ku);
}

// Depth helpers. We treat "k_eff" as the reduction depth (≈ #adds) like d(k) in the paper.
// For flat we keep your existing convention (tile_inner_k, L); for pairwise we use ceil(log2).
__host__ __device__ inline int depth_flat(int count) { return max(count, 1); }
__host__ __device__ inline int depth_pairwise(int count) {
    return (count <= 1) ? 1 : ceil_log2_int(count);
}

// New signature: take compute precision (u_c) and accumulate precision (u_a).
// Optionally include the cross-term; if false you'll get the first-order sum.
// If uc==ua and you want the condensed form, set collapse_same_u=true.
float compute_beta_factor(KernelType kernel_type, int K)
{
    // Get unit roundoffs for compute and accumulate types
    const double u_c = unit_roundoff_from_type();
    const double u_a = unit_roundoff_accumulate_type();

    // Use configuration flags from config.h
    const bool include_cross_term = INCLUDE_CROSS_TERMS;
    const bool collapse_same_u = COLLAPSE_SAME_PRECISION;

    // Defaults to avoid UB
    int  L            = 1;      // number of inter (tiles/slices) to combine
    int  tile_inner_k = 1;      // intra depth (K inside a tile/slice)
    bool inter_pairwise = false;
    bool intra_pairwise = false; // keep hook if you later add pairwise *within* tiles

    switch (kernel_type) {
        case KERNEL_TILED: {
            L            = ceil_div(K, TILE_SIZE);
            tile_inner_k = TILE_SIZE;
            inter_pairwise = false;
            break;
        }
        case KERNEL_TILED_PAIRWISE: {
            L            = ceil_div(K, TILE_SIZE);
            tile_inner_k = TILE_SIZE;
            inter_pairwise = true;   // pairwise across tiles
            break;
        }
        case KERNEL_CUTLASS_SPLITK_FLAT: {
            const int S  = SPLIT_K_SLICES;
            L            = S;                        // inter = slices
            tile_inner_k = ceil_div(K, S);          // intra depth per slice
            inter_pairwise = false;
            break;
        }
        case KERNEL_CUTLASS_SPLITK_PAIRWISE: {
            const int S  = SPLIT_K_SLICES;
            L            = S;
            tile_inner_k = ceil_div(K, S);
            inter_pairwise = true;                  // pairwise across slices
            break;
        }
        case KERNEL_CUBLAS: {
            const int TILE_K_BLAS = TILE_SIZE;      // or a dedicated constant
            L            = ceil_div(K, TILE_K_BLAS);
            tile_inner_k = TILE_K_BLAS;
            inter_pairwise = false;
            break;
        }
         case KERNEL_TILED_MIXPREC: {                // Mixed precision tiled kernel
            L            = ceil_div(K, TILE_SIZE);
            tile_inner_k = TILE_SIZE;
            inter_pairwise     = false;
            break;
        }
        default: {
            L            = ceil_div(K, TILE_SIZE);
            tile_inner_k = TILE_SIZE;
            inter_pairwise = false;
            break;
        }
    }

    // Map to reduction depths d(b_k) and d(s)
    const int d_intra = intra_pairwise
                        ? depth_pairwise(tile_inner_k)
                        : depth_flat(tile_inner_k);

    const int d_inter = inter_pairwise
                        ? depth_pairwise(L)
                        : depth_flat(L);

    // Stage 1 (within tile/slice) uses u_c; Stage 2 (across tiles/slices) uses u_a
    const double beta_inner = gamma_bound(d_intra, u_c);
    const double beta_outer = gamma_bound(d_inter, u_a);

    if (collapse_same_u && fabs(u_c - u_a) < 1e-20) {
        // Use composition (1+γ_a)(1+γ_b)-1 <= γ_{a+b} when uc == ua
        const int d_total = d_intra + d_inter;
        return (float)gamma_bound(d_total, u_c);
    }

    // Theorem 3.5 form: intra + inter + cross (cross is O(u^2), optional)
    const double beta = include_cross_term
                        ? (beta_inner + beta_outer + beta_inner * beta_outer)
                        : (beta_inner + beta_outer);

    return (float)beta;
}

// Integer ULP distance between two FP32 values.
__device__ __forceinline__ int32_t ord(float x) {
    int32_t i = __float_as_int(x);
    return (i < 0) ? 0x80000000 - i : i + 0x80000000;
}

__device__ __forceinline__ uint32_t ulp_distance(float a, float b) {
    int32_t da = ord(a), db = ord(b);
    int32_t d  = da - db;
    return (d < 0) ? uint32_t(-d) : uint32_t(d);
}

// Size of one ULP at x (FP32)
__device__ __forceinline__ float ulp_of(float x) {
    if (!isfinite(x)) return NAN;
    x = fabsf(x);
    if (x == 0.0f) return ldexpf(1.0f, -149);  // subnormal spacing
    uint32_t u = __float_as_uint(x);
    uint32_t exp = (u >> 23) & 0xFFu;
    if (exp == 0u) return ldexpf(1.0f, -149);
    int e = int(exp) - 127;
    return ldexpf(1.0f, e - 23);               // 2^(e-23)
}

__global__ void ulp_metrics_kernel(const float* __restrict__ Ctest,
                                   const float* __restrict__ Cref,
                                   uint32_t* __restrict__ dULP,   // out: integer ULP distance
                                   float* __restrict__ errULP,    // out: scaled err (may be nullptr)
                                   unsigned long long* __restrict__ invalid_count, // NaN/Inf pairs
                                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ct = Ctest[i];
    float cr = Cref[i];

    if (!isfinite(ct) || !isfinite(cr)) {
        dULP[i] = 0;
        if (errULP) errULP[i] = NAN;
        atomicAdd(invalid_count, 1ull);
        return;
    }

    dULP[i] = ulp_distance(ct, cr);

    if (errULP) {
        float ulp = ulp_of(cr);
        float e = fabsf(ct - cr) / ulp;   // uses spacing at the reference
        errULP[i] = e;
    }
}

// ULP histogram bins and representative values (narrow then doubling)
__device__ __forceinline__ int ulp_bin(uint32_t d){
    if (d == 0u) return 0;
    if (d == 1u) return 1;
    if (d == 2u) return 2;
    if (d <= 4u)  return 3;
    if (d <= 8u)  return 4;
    if (d <= 16u) return 5;
    if (d <= 32u) return 6;
    if (d <= 64u) return 7;
    return 8;
}

__device__ __forceinline__ void atomicAdd_f64(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(addr, val);
#else
    atomicAddDouble(addr, val);
#endif
}

// ULP computed in double to avoid FTZ/underflow
__device__ __forceinline__ double ulp_of_double(float x) {
    // work with bits; no branches on x==0 needed, the exp=0 path covers zero too
    uint32_t u = __float_as_uint(fabsf(x));
    uint32_t exp = (u >> 23) & 0xFFu;
    if (exp == 0u) {
        // subnormals or zero: spacing is 2^-149 (representable & stable in double)
        return ldexp(1.0, -149);
    }
    int e_unbiased = int(exp) - 127;          // FP32 unbiased exponent
    return ldexp(1.0, e_unbiased - 23);       // 2^(e-23), in double
}

// ----- fused streaming kernel -----
__global__ void ulp_stream_hist_kernel(const float* __restrict__ Ctest,
                                       const float* __restrict__ Cref,
                                       unsigned long long* __restrict__ gBins,   // [NUM_BINS]
                                       double* __restrict__ gErrSum,             // scalar
                                       double* __restrict__ gErrSumSq,           // scalar
                                       unsigned long long* __restrict__ gCount,  // scalar
                                       unsigned long long* __restrict__ gInvalid,// scalar
                                       unsigned long long* __restrict__ gRefZeroOrSub,  // count of ref entries with exp==0
                                       int n)
{
    __shared__ unsigned long long sBins[NUM_BINS];
    if (threadIdx.x < NUM_BINS) sBins[threadIdx.x] = 0ull;
    __syncthreads();

    // Local accumulators to reduce atomic pressure
    double locErrSum = 0.0, locErrSumSq = 0.0;
    unsigned long long locCount = 0, locInvalid = 0, locRefZeroOrSub = 0;

    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float ct = Ctest[i], cr = Cref[i];
        if (!isfinite(ct) || !isfinite(cr)) { ++locInvalid; continue; }

        uint32_t d = ulp_distance(ct, cr);
        int b = ulp_bin(d);
        atomicAdd(&sBins[b], 1ull);

        // scaled error: |Δ| / ULP(ref)
        uint32_t ur = __float_as_uint(cr);
        bool is_sub_or_zero = ((ur & 0x7F800000u) == 0u);  // exp==0 (zero or subnormal)

        double ulp = ulp_of_double(cr);
        double delta = fabs((double)ct - (double)cr);  // do the division in double
        double e = delta / ulp;

        if (is_sub_or_zero) ++locRefZeroOrSub;

        locErrSum += (double)e;
        locErrSumSq += (double)e * (double)e;
        ++locCount;
    }
    __syncthreads();

    // Merge block-local histogram into global
    if (threadIdx.x < NUM_BINS) atomicAdd(&gBins[threadIdx.x], sBins[threadIdx.x]);

    // Flush per-thread locals (ALL threads do this)
    atomicAdd_f64(gErrSum,   locErrSum);
    atomicAdd_f64(gErrSumSq, locErrSumSq);
    atomicAdd(gCount,   locCount);
    atomicAdd(gInvalid, locInvalid);
    atomicAdd(gRefZeroOrSub, locRefZeroOrSub);
}

// Wilson CI for a pooled proportion
static inline std::pair<double,double> wilson_ci(unsigned long long s, unsigned long long n, double conf=0.95){
    if (n==0) return {NAN,NAN};
    const double z = 1.959963984540054; // 95%
    const double phat = double(s)/double(n);
    const double z2 = z*z, denom = 1.0 + z2/double(n);
    const double center = (phat + z2/(2.0*double(n))) / denom;
    const double half   = (z * std::sqrt( (phat*(1.0-phat))/double(n) + z2/(4.0*double(n)*double(n)) )) / denom;
    return {std::max(0.0, center - half), std::min(1.0, center + half)};
}

// Percentile from 9-bin pooled histogram (returns an upper-bound representative)
static inline uint32_t percentile_ulps(const unsigned long long B[NUM_BINS], double p){
    unsigned long long tot=0; for(int i=0;i<NUM_BINS;++i) tot += B[i];
    if (!tot) return 0u;
    unsigned long long thr = (unsigned long long)std::ceil(p * double(tot));
    unsigned long long acc=0;
    for (int i=0;i<NUM_BINS;++i){ acc += B[i]; if (acc >= thr) return BIN_REP_UPPER[i]; }
    return BIN_REP_UPPER[NUM_BINS-1];
}

// Sum counts for all bins whose upper edge <= K (e.g., K=1 -> bins 0 and 1)
static inline unsigned long long successes_le_k(const unsigned long long B[NUM_BINS], uint32_t K){
    unsigned long long s = 0;
    for (int i=0;i<NUM_BINS;++i) if (BIN_REP_UPPER[i] <= K) s += B[i];
    return s;
}

static inline unsigned long long sum_bins(const unsigned long long B[NUM_BINS]){
    unsigned long long s=0; for (int i=0;i<NUM_BINS;++i) s+=B[i]; return s;
}

void run_ulp_samples_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int num_samples, const char* output_prefix, bool reproducible) {
    printf("\n=== ULP Analysis ===\n");
    printf("Matrix Type: %d, Kernel: %d, Size: %dx%d, Samples: %d\n",
           (int)matrix_type, (int)kernel_type, n, n, num_samples);

    size_t size = n * n * sizeof(float);
    size_t size_fp64 = n * n * sizeof(double);
    float *d_A, *d_B, *d_C_kernel;
    double *d_C_reference_fp64;  // FP64 reference result
    float *d_C_reference_fp32;   // FP32 reference for ULP comparison

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_kernel, size);
    cudaMalloc(&d_C_reference_fp64, size_fp64);
    cudaMalloc(&d_C_reference_fp32, size);

    // Create cuBLAS handle for reference computation
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // ULP histogram and statistics accumulators
    unsigned long long *dBins, *dCount, *dInvalid, *dRefZeroOrSub;
    double *dSum, *dSumSq;

    cudaMalloc(&dBins, NUM_BINS * sizeof(unsigned long long));
    cudaMalloc(&dCount, sizeof(unsigned long long));
    cudaMalloc(&dInvalid, sizeof(unsigned long long));
    cudaMalloc(&dRefZeroOrSub, sizeof(unsigned long long));
    cudaMalloc(&dSum, sizeof(double));
    cudaMalloc(&dSumSq, sizeof(double));

    // Initialize accumulators (these accumulate across all samples)
    cudaMemset(dBins, 0, NUM_BINS * sizeof(unsigned long long));
    cudaMemset(dCount, 0, sizeof(unsigned long long));
    cudaMemset(dInvalid, 0, sizeof(unsigned long long));
    cudaMemset(dRefZeroOrSub, 0, sizeof(unsigned long long));
    cudaMemset(dSum, 0, sizeof(double));
    cudaMemset(dSumSq, 0, sizeof(double));

    // Configure kernel launch parameters
    dim3 threadsPerBlock, numBlocks;
    compute_kernel_dimensions_dispatch(kernel_type, n, &threadsPerBlock, &numBlocks);

    // Configure 1D parameters for ULP analysis kernel
    int total_elements = n * n;
    int block_size_1d, grid_size_1d;
    compute_kernel_dimensions_dispatch_1D(total_elements, &block_size_1d, &grid_size_1d);

    printf("Running %d samples...\n", num_samples);

    // Generate seed array for reproducibility
    unsigned long long* seeds = new unsigned long long[num_samples];
    if (reproducible) {
        generate_seed_array(seeds, num_samples, ERROR_SEED);
    } else {
        // For non-reproducible mode, use time-based base seed
        unsigned long long time_seed = (unsigned long long)time(NULL);
        generate_seed_array(seeds, num_samples, time_seed);
    }

    // For each matrix sample s
    for (int s = 0; s < num_samples; ++s) {
        if (s % 10 == 0 && s > 0) {
            printf("Completed %d/%d samples...\n", s, num_samples);
        }

        // Generate new matrices for this sample using the specified matrix type
        // Use different seeds for each sample from our reproducible array
        auto seedA = seeds[s];
        auto seedB = seedA ^ 0x94D049BB133111EBull;

        generate_matrix_device_with_seed(d_A, n, matrix_type, seedA);
        generate_matrix_device_with_seed(d_B, n, matrix_type, seedB);

        // Launch the specified kernel using unified dispatch
        if (is_mixprec_kernel(kernel_type)) {
            launch_mixprec_kernel_by_type<COMPUTE_TYPE, ACCUMULATE_TYPE>(
                kernel_type, (COMPUTE_TYPE*)d_A, (COMPUTE_TYPE*)d_B, (ACCUMULATE_TYPE*)d_C_kernel, n, numBlocks, threadsPerBlock);
        } else if (areBothTypesFP32()) {
            launch_basic_kernel_by_type(kernel_type, d_A, d_B, d_C_kernel, n, numBlocks, threadsPerBlock);
        } else {
            printf("ERROR: Non-mixprec kernels require FP32 types\n");
            return;
        }

        // Compute FP64 reference directly on device using same configuration as test kernel
        // This ensures consistency with the cuBLAS kernel configuration from gemms.cu
        compute_reference_fp64_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C_reference_fp64, n);

        // Convert FP64 reference to FP32 for ULP comparison
        convert_fp64_to_fp32_kernel<<<grid_size_1d, block_size_1d>>>(d_C_reference_fp64, d_C_reference_fp32, total_elements);

        // Stream this matrix's entries into the SAME accumulators (no reset)
        ulp_stream_hist_kernel<<<grid_size_1d, block_size_1d>>>(
            d_C_kernel, d_C_reference_fp32, dBins, dSum, dSumSq, dCount, dInvalid, dRefZeroOrSub, total_elements);
    }

    printf("Completed all %d samples\n", num_samples);

    // Copy results back to host
    unsigned long long hBins[NUM_BINS];
    unsigned long long hCount, hInvalid, hRefZeroOrSub;
    double hSum, hSumSq;

    cudaMemcpy(hBins, dBins, NUM_BINS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hCount, dCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hInvalid, dInvalid, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hRefZeroOrSub, dRefZeroOrSub, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hSum, dSum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hSumSq, dSumSq, sizeof(double), cudaMemcpyDeviceToHost);

    // Histogram (counts)
    const unsigned long long total_from_bins = sum_bins(hBins);   // authoritative
    const unsigned long long total_from_counter = hCount;         // sanity check
    if (total_from_bins != total_from_counter) {
        fprintf(stderr, "[warn] sum(hBins)=%llu != hCount=%llu — using sum(hBins) for normalization\n",
                total_from_bins, total_from_counter);
    }
    const unsigned long long TOTAL = total_from_bins ? total_from_bins : total_from_counter;

    printf("Histogram (counts):\n");
    for (int i = 0; i < NUM_BINS; ++i) {
        double pct = TOTAL ? 100.0 * (double)hBins[i] / (double)TOTAL : 0.0;
        printf("  %-5s : %10llu  (%6.2f%%)\n", BIN_LABELS[i], hBins[i], pct);
    }

    // Proportions and CI
    const unsigned long long s_le1 = successes_le_k(hBins, 1);     // bins 0 and 1 with your scheme
    const double frac0   = TOTAL ? (double)hBins[0] / (double)TOTAL : 0.0;
    const double fracle1 = TOTAL ? (double)s_le1   / (double)TOTAL : 0.0;
    auto [ci_lo, ci_hi]  = wilson_ci(s_le1, TOTAL);

    // Percentiles (integer ULP)
    uint32_t p95 = percentile_ulps(hBins, 0.95);
    uint32_t p99 = percentile_ulps(hBins, 0.99);

    // Scaled error stats (|Δ| / ULP(ref)) from your dSum/dSumSq
    double mean_scaled = TOTAL ? hSum / (double)TOTAL : NAN;
    double var_scaled  = TOTAL ? std::max(0.0, (hSumSq / (double)TOTAL) - mean_scaled*mean_scaled) : NAN;
    double std_scaled  = TOTAL ? std::sqrt(var_scaled) : NAN;

    // Invalids
    double frac_invalid = (hCount + hInvalid) ? (double)hInvalid / (double)(hCount + hInvalid) : 0.0;

    // Headline
    printf("frac(ULP=0)          : %.6f\n", frac0);
    printf("frac(ULP<=1)         : %.6f  (Wilson 95%% CI: [%.6f, %.6f])\n", fracle1, ci_lo, ci_hi);
    printf("p95 ULP (≤)          : %s\n", (p95==UINT32_MAX? ">=65" : std::to_string(p95).c_str()));
    printf("p99 ULP (≤)          : %s\n", (p99==UINT32_MAX? ">=65" : std::to_string(p99).c_str()));
    printf("scaled |Δ|/ULP(ref)  : mean=%.6e, std=%.6e\n", mean_scaled, std_scaled);
    printf("invalid fraction     : %.6f\n", frac_invalid);
    printf("reference sub/zero   : %llu / %llu (%.6f%% of total)\n", hRefZeroOrSub, TOTAL, TOTAL ? 100.0 * (double)hRefZeroOrSub / (double)TOTAL : 0.0);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_kernel);
    cudaFree(d_C_reference_fp64);
    cudaFree(d_C_reference_fp32);
    cudaFree(dBins);
    cudaFree(dCount);
    cudaFree(dInvalid);
    cudaFree(dSum);
    cudaFree(dSumSq);
    cublasDestroy(cublas_handle);

    // Cleanup seed array
    delete[] seeds;
}

// Given measured E/u and E/beta, compute "effective depth"
inline double effective_depth(double E_over_u, double E_over_beta) {
  return E_over_u / std::max(E_over_beta, 1e-300); // ~ beta/u
}

// Compute log c_hat median using the formula: median(log(E/u) - log(β/u))
// Returns the median of log values, not exp(median)
double compute_log_c_hat_median(const double* frobenius_errors, int num_samples,
                               double beta_factor, double u_compute) {
    // Allocate temporary array for log calculations
    double* log_values = (double*)malloc(num_samples * sizeof(double));

    // Compute log(E/u) - log(β/u) for each sample
    for (int i = 0; i < num_samples; i++) {
        double E_over_u = frobenius_errors[i] / u_compute;
        double beta_over_u = beta_factor / u_compute;

        // Handle edge cases to avoid log(0) or log(negative)
        if (E_over_u <= 0.0 || beta_over_u <= 0.0) {
            log_values[i] = NAN;
            continue;
        }

        log_values[i] = log(E_over_u) - log(beta_over_u);
    }

    // Sort the log values to find median (excluding NaN values)
    std::vector<double> valid_logs;
    for (int i = 0; i < num_samples; i++) {
        if (isfinite(log_values[i])) {
            valid_logs.push_back(log_values[i]);
        }
    }

    double median_log = NAN;
    if (!valid_logs.empty()) {
        std::sort(valid_logs.begin(), valid_logs.end());
        size_t n = valid_logs.size();
        if (n % 2 == 0) {
            median_log = (valid_logs[n/2 - 1] + valid_logs[n/2]) / 2.0;
        } else {
            median_log = valid_logs[n/2];
        }
    }

    free(log_values);
    return median_log;
}

// Per-entry normalized ERROR computation on device
__global__ void compute_per_entry_normalized_error(float* C_kernel,
                                                           double* C_reference_fp64,
                                                           double* abs_AB_product,
                                                           float* normalized_output,
                                                           int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // Compute the actual error: |C_kernel - C_reference|
        float c_kernel = C_kernel[idx];
        double c_ref = C_reference_fp64[idx];
        double error = fabs((double)c_kernel - c_ref);  // Error in FP64

        double abs_ab = abs_AB_product[idx];

        double normalized_val;
        if (abs_ab > 0.0) {
            normalized_val = error / abs_ab;  // |C - C_ref| / (|A| * |B|) in FP64
        } else {
            normalized_val = 0.0;  // Handle division by zero
        }

        normalized_output[idx] = (float)normalized_val;  // Downcast to FP32 for storage
    }
}

__global__ void compute_EAB_entrywise(
    const float*  __restrict__ C_kernel,
    const double* __restrict__ C_ref64,
    const double* __restrict__ absAB64,
    double*       __restrict__ EAB,          // <- FP64 output
    int n, double denom_floor)               // e.g., 1e-300 or a data-driven floor
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double err = fabs((double)C_kernel[i] - C_ref64[i]);
        double den = fmax(absAB64[i], denom_floor);   // avoid 0/near-0 blowups
        EAB[i] = err / den;
    }
}

template<typename T>
void validate_matrix_on_device(const T* d_matrix, int n, const char* label, bool verbose = false) {
    // DEBUG: Check pointer and type info
    printf("\n[DEBUG validate_matrix_on_device]\n");
    printf("  Pointer: %p\n", d_matrix);
    printf("  Type T: %s (size=%zu)\n", typeid(T).name(), sizeof(T));
    printf("  Copying %d elements = %zu bytes\n", n*n, n*n*sizeof(T));

    // Copy JUST the first element to verify pointer is valid
    T first_val;
    cudaError_t err = cudaMemcpy(&first_val, d_matrix, sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("  ERROR: Failed to copy first element: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("  First element before full copy: %f\n", static_cast<float>(first_val));

    // Copy to host
    std::vector<T> h_matrix(n * n);
    err = cudaMemcpy(h_matrix.data(), d_matrix, n * n * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("  ERROR: cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("  First element after full copy: %f\n", static_cast<float>(h_matrix[0]));

    // Statistics
    int nan_count = 0;
    int inf_count = 0;
    int zero_count = 0;
    int negative_count = 0;
    double sum = 0.0;
    double min_val = 1e30;
    double max_val = -1e30;

    // Collect sample values for inspection
    std::vector<float> samples;
    int sample_stride = std::max(1, (n * n) / 100);  // Sample ~100 values

    for (int i = 0; i < n * n; i++) {
        float val = static_cast<float>(h_matrix[i]);

        if (std::isnan(val)) {
            nan_count++;
            if (verbose && nan_count <= 5) {
                printf("  NaN at index %d (row=%d, col=%d)\n", i, i/n, i%n);
            }
        } else if (std::isinf(val)) {
            inf_count++;
            if (verbose && inf_count <= 5) {
                printf("  Inf at index %d (row=%d, col=%d): %f\n", i, i/n, i%n, val);
            }
        } else {
            // Valid value
            sum += val;
            if (val == 0.0f) zero_count++;
            if (val < 0.0f) negative_count++;
            min_val = std::min(min_val, (double)val);
            max_val = std::max(max_val, (double)val);

            if (i % sample_stride == 0) {
                samples.push_back(val);
            }
        }
    }

    // Print summary
    printf("\n=== Matrix Validation: %s ===\n", label);
    printf("  Size: %d x %d (%d elements)\n", n, n, n*n);
    printf("  Type: %s\n", typeid(T).name());
    printf("  Type size: %zu bytes\n", sizeof(T));

    if (nan_count > 0 || inf_count > 0) {
        printf("  ❌ INVALID VALUES DETECTED!\n");
        printf("  NaN count: %d (%.2f%%)\n", nan_count, 100.0 * nan_count / (n*n));
        printf("  Inf count: %d (%.2f%%)\n", inf_count, 100.0 * inf_count / (n*n));
    } else {
        printf("  ✓ No NaN or Inf values\n");
    }

    int valid_count = n*n - nan_count - inf_count;
    if (valid_count > 0) {
        printf("  Valid values: %d (%.2f%%)\n", valid_count, 100.0 * valid_count / (n*n));
        printf("  Zero count: %d (%.2f%%)\n", zero_count, 100.0 * zero_count / (n*n));
        printf("  Negative count: %d (%.2f%%)\n", negative_count, 100.0 * negative_count / (n*n));
        printf("  Min value: %.6e\n", min_val);
        printf("  Max value: %.6e\n", max_val);
        printf("  Mean value: %.6e\n", sum / valid_count);

        if (verbose && samples.size() > 0) {
            printf("  Sample values (first 10):\n");
            for (int i = 0; i < std::min(10, (int)samples.size()); i++) {
                printf("    [%d]: %.6e\n", i, samples[i]);
            }
        }
    }
    printf("=========================\n\n");
}

// Specialized validation for comparing two matrices
template<typename T1, typename T2>
void validate_matrix_difference(const T1* d_matrix1, const T2* d_matrix2, int n,
                                const char* label1, const char* label2, bool verbose = false) {
    // Copy both matrices to host
    std::vector<T1> h_matrix1(n * n);
    std::vector<T2> h_matrix2(n * n);
    cudaMemcpy(h_matrix1.data(), d_matrix1, n * n * sizeof(T1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_matrix2.data(), d_matrix2, n * n * sizeof(T2), cudaMemcpyDeviceToHost);

    printf("\n=== Comparing %s vs %s ===\n", label1, label2);
    printf("  Size: %d x %d\n", n, n);
    printf("  Type 1: %s (%zu bytes)\n", typeid(T1).name(), sizeof(T1));
    printf("  Type 2: %s (%zu bytes)\n", typeid(T2).name(), sizeof(T2));

    int mismatch_count = 0;
    int both_invalid = 0;
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    double sum_rel_diff = 0.0;
    int valid_comparisons = 0;

    for (int i = 0; i < n * n; i++) {
        float val1 = static_cast<float>(h_matrix1[i]);
        float val2 = static_cast<float>(h_matrix2[i]);

        bool invalid1 = std::isnan(val1) || std::isinf(val1);
        bool invalid2 = std::isnan(val2) || std::isinf(val2);

        if (invalid1 && invalid2) {
            both_invalid++;
            continue;
        }

        if (invalid1 || invalid2) {
            mismatch_count++;
            if (verbose && mismatch_count <= 10) {
                printf("  Mismatch at [%d,%d]: %s=%.6e, %s=%.6e\n",
                       i/n, i%n, label1, val1, label2, val2);
            }
            continue;
        }

        // Both valid - compute difference
        double abs_diff = std::abs(val1 - val2);
        double rel_diff = (val2 != 0.0f) ? abs_diff / std::abs(val2) : abs_diff;

        sum_abs_diff += abs_diff;
        sum_rel_diff += rel_diff;
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        valid_comparisons++;

        if (verbose && abs_diff > 1e-3 && mismatch_count < 10) {
            printf("  Large diff at [%d,%d]: %s=%.6e, %s=%.6e, diff=%.6e (rel=%.6e)\n",
                   i/n, i%n, label1, val1, label2, val2, abs_diff, rel_diff);
            mismatch_count++;
        }
    }

    printf("  Both invalid: %d\n", both_invalid);
    printf("  One invalid: %d\n", mismatch_count);
    printf("  Valid comparisons: %d\n", valid_comparisons);

    if (valid_comparisons > 0) {
        printf("  Max absolute difference: %.6e\n", max_abs_diff);
        printf("  Mean absolute difference: %.6e\n", sum_abs_diff / valid_comparisons);
        printf("  Mean relative difference: %.6e\n", sum_rel_diff / valid_comparisons);
    }
    printf("================================\n\n");
}

// Efficient multi-sample testing for specific matrix type and kernel
// Uses consistent kernel configurations for fair error analysis:
// - Test kernel: Uses kernel-specific optimized configuration
// - Reference: Uses standard 2D tiled configuration (kernel-independent)
// - Helper kernels: Use standard 1D configurations (efficient for element-wise ops)
void run_multi_sample_analysis(MatrixType matrix_type, KernelType kernel_type, int n, int num_samples, const char* output_prefix, bool reproducible) {
    TypeInfo compute_info = getComputeTypeInfo();
    TypeInfo accumulate_info = getAccumulateTypeInfo();

    printf("=== Type Configuration ===\n");
    printf("Compute Type: %s (%zu bytes)\n", compute_info.name, compute_info.size_bytes);
    printf("Accumulate Type: %s (%zu bytes)\n", accumulate_info.name, accumulate_info.size_bytes);
    printf("Reference Type: FP64 (8 bytes)\n");
    printf("Matrix Type: %s\n", matrixTypeToString(matrix_type));
    printf("==========================\n\n");

    // Allocate matrices using COMPUTE_TYPE
    size_t size_compute = n * n * sizeof(COMPUTE_TYPE);
    size_t size_accumulate = n * n * sizeof(ACCUMULATE_TYPE);
    COMPUTE_TYPE *d_A, *d_B;
    ACCUMULATE_TYPE*d_C_kernel;

    cudaMalloc(&d_A, size_compute);
    cudaMalloc(&d_B, size_compute);
    cudaMalloc(&d_C_kernel, size_accumulate);

    // Allocate reference matrices in FP64
    size_t size_fp64 = n * n * sizeof(double);
    double *d_A_ref, *d_B_ref, *d_C_reference_fp64;

    cudaMalloc(&d_A_ref, size_fp64);
    cudaMalloc(&d_B_ref, size_fp64);
    cudaMalloc(&d_C_reference_fp64, size_fp64);

    // Device memory for optimized error computation (no host transfers needed)
    double *d_abs_AB_product;    // Product |A| * |B| (FP64)
    double *d_error_results;     // Error computation results [frobenius², norm²]

    cudaMalloc(&d_abs_AB_product, size_fp64);  // For |A| * |B| product (FP64)
    cudaMalloc(&d_error_results, 2 * sizeof(double));  // [error², norm²]

    // Host memory only for final statistics (much smaller)
    double *frobenius_errors = (double*)malloc(num_samples * sizeof(double));
    double *frobenius_M_error = (double*)malloc(num_samples * sizeof(double));
    double *normalized_errors = (double*)malloc(num_samples * sizeof(double));

    // Declare variables that might be accessed after goto
    FILE* fp = NULL;

    // Configure kernel launch parameters for the test kernel
    // NOTE: Different kernels use different optimal configurations:
    // - Reference: Uses standard 2D tiled configuration for consistency
    // - Helper kernels: Use 1D linear configurations for efficiency
    dim3 threadsPerBlock, numBlocks;
    compute_kernel_dimensions_dispatch(kernel_type, n, &threadsPerBlock, &numBlocks);

    // Configure 1D parameters for helper kernels
    int total_elements = n * n;
    int block_size_1d, grid_size_1d;
    compute_kernel_dimensions_dispatch_1D(total_elements, &block_size_1d, &grid_size_1d);
    size_t shared_mem_size = 2 * block_size_1d * sizeof(double);  // For error and norm arrays

    printf("Running %d samples...\n", num_samples);

    // Generate seed array for reproducibility
    unsigned long long* seeds = new unsigned long long[num_samples];
    if (reproducible) {
        generate_seed_array(seeds, num_samples, ERROR_SEED);
    } else {
        // For non-reproducible mode, use time-based base seed
        unsigned long long time_seed = (unsigned long long)time(NULL);
        generate_seed_array(seeds, num_samples, time_seed);
    }

    // Compute theoretical error bound factor
    float beta_factor = compute_beta_factor(kernel_type, n);

    // Run multiple samples
    for (int sample = 0; sample < num_samples; sample++) {
        if (sample % 10 == 0 && sample > 0) {
            printf("Completed %d/%d samples...\n", sample, num_samples);
        }

        // Generate new matrices for this sample using the specified matrix type
        // Use different seeds for each sample from our reproducible array
        auto seedA = seeds[sample];
        auto seedB = seedA ^ 0x94D049BB133111EBull;
        generate_matrix_device_with_seed_typed<COMPUTE_TYPE>(d_A, n, matrix_type, seedA, numBlocks, threadsPerBlock);
        generate_matrix_device_with_seed_typed<COMPUTE_TYPE>(d_B, n, matrix_type, seedB, numBlocks, threadsPerBlock);

        // // After both A and B are generated:
        // std::vector<COMPUTE_TYPE> h_A_sample(10);
        // std::vector<COMPUTE_TYPE> h_B_sample(10);
        // cudaMemcpy(h_A_sample.data(), d_A, 10 * sizeof(COMPUTE_TYPE), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_B_sample.data(), d_B, 10 * sizeof(COMPUTE_TYPE), cudaMemcpyDeviceToHost);

        // printf("\n=== INPUT MATRICES (first 10 elements) ===\n");
        // printf("COMPUTE_TYPE size: %zu bytes\n", sizeof(COMPUTE_TYPE));
        // printf("A: ");
        // for (int i = 0; i < 10; i++) {
        //     printf("%.6f ", static_cast<float>(h_A_sample[i]));
        // }
        // printf("\nB: ");
        // for (int i = 0; i < 10; i++) {
        //     printf("%.6f ", static_cast<float>(h_B_sample[i]));
        // }
        // printf("\n==========================================\n\n");

        // // Now check the BYTES to see if they're truly different types
        // printf("=== RAW MEMORY (first 10 elements as hex) ===\n");
        // unsigned char* bytes_A = reinterpret_cast<unsigned char*>(h_A_sample.data());
        // unsigned char* bytes_B = reinterpret_cast<unsigned char*>(h_B_sample.data());
        // printf("A bytes: ");
        // for (int i = 0; i < 10 * sizeof(COMPUTE_TYPE); i++) {
        //     printf("%02x ", bytes_A[i]);
        // }
        // printf("\n=============================================\n\n");
        // // // ✅ VALIDATE INPUT MATRICES
        // // validate_matrix_on_device(d_A, n, "Input Matrix A (after generation)", sample == 0);  // verbose on first sample
        // // validate_matrix_on_device(d_B, n, "Input Matrix B (after generation)", sample == 0);

        //         // After generate_matrix_device_with_seed_typed:
        // COMPUTE_TYPE test_val;
        // cudaMemcpy(&test_val, d_A, sizeof(COMPUTE_TYPE), cudaMemcpyDeviceToHost);
        // printf("First A element as COMPUTE_TYPE: raw=%f\n", static_cast<float>(test_val));

        // float test_val_float;
        // cudaMemcpy(&test_val_float, d_A, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("First A element as float: raw=%f\n", test_val_float);

        // if (sizeof(COMPUTE_TYPE) == 2) {
        //     // Check if it's actually half precision
        //     unsigned short bits;
        //     cudaMemcpy(&bits, d_A, sizeof(unsigned short), cudaMemcpyDeviceToHost);
        //     printf("First A element raw bits (hex): 0x%04x\n", bits);
        // }

        // // Right before launch_mixprec_kernel_by_type:
        // printf("\n=== BEFORE KERNEL LAUNCH ===\n");
        // printf("Pointers being passed:\n");
        // printf("  d_A: %p (COMPUTE_TYPE*, size=%zu)\n", d_A, sizeof(COMPUTE_TYPE));
        // printf("  d_B: %p (COMPUTE_TYPE*, size=%zu)\n", d_B, sizeof(COMPUTE_TYPE));
        // printf("  d_C_kernel: %p (ACCUMULATE_TYPE*, size=%zu)\n", d_C_kernel, sizeof(ACCUMULATE_TYPE));

        // // Verify the pointers contain the expected data
        // std::vector<COMPUTE_TYPE> verify_A(5);
        // cudaMemcpy(verify_A.data(), d_A, 5 * sizeof(COMPUTE_TYPE), cudaMemcpyDeviceToHost);
        // printf("  First 5 A values before kernel: %f %f %f %f %f\n",
        //        (float)verify_A[0], (float)verify_A[1], (float)verify_A[2],
        //        (float)verify_A[3], (float)verify_A[4]);
        // printf("============================\n\n");

        launch_mixprec_kernel_by_type<COMPUTE_TYPE, ACCUMULATE_TYPE>(
            kernel_type, d_A, d_B, d_C_kernel, n, numBlocks, threadsPerBlock);
        // Launch the specified kernel using unified dispatch
        if (is_mixprec_kernel(kernel_type)) {
            launch_mixprec_kernel_by_type<COMPUTE_TYPE, ACCUMULATE_TYPE>(
                kernel_type, (COMPUTE_TYPE*)d_A, (COMPUTE_TYPE*)d_B, (ACCUMULATE_TYPE*)d_C_kernel, n, numBlocks, threadsPerBlock);
        } else if (areBothTypesFP32()) {
            launch_basic_kernel_by_type(kernel_type, (float*)d_A, (float*)d_B, (float*)d_C_kernel, n, numBlocks, threadsPerBlock);
        } else {
            printf("ERROR: Non-mixprec kernels require FP32 types\n");
            return;
        }

        // cudaError_t err = cudaGetLastError();
        // printf("Kernel launch status: %s\n", cudaGetErrorString(err));

        // cudaDeviceSynchronize();
        // err = cudaGetLastError();
        // printf("Kernel execution status: %s\n", cudaGetErrorString(err));

        // // VALIDATE IMMEDIATELY - Don't do anything else first!
        // printf("After kernel: d_C_kernel pointer = %p\n", d_C_kernel);

        // // Copy just the first element to check
        // ACCUMULATE_TYPE first_element;
        // cudaMemcpy(&first_element, d_C_kernel, sizeof(ACCUMULATE_TYPE), cudaMemcpyDeviceToHost);
        // printf("First element from d_C_kernel: %f\n", (float)first_element);

        // // ✅ VALIDATE KERNEL OUTPUT
        // validate_matrix_on_device(d_C_kernel, n, "Kernel Output C", sample == 0);

         // Compute FP64 reference directly on device using same configuration as test kernel
        // This ensures consistency with the cuBLAS kernel configuration from gemms.cu
        compute_reference_and_norm_fp64_device<COMPUTE_TYPE><<<numBlocks, threadsPerBlock>>>(
            d_A, d_B, d_C_reference_fp64, d_abs_AB_product, n);

        // // ✅ VALIDATE REFERENCE
        // validate_matrix_on_device(d_C_reference_fp64, n, "FP64 Reference C", sample == 0);

        // // ✅ COMPARE KERNEL VS REFERENCE
        // validate_matrix_difference(d_C_kernel, d_C_reference_fp64, n,
        //                            "Kernel Output", "FP64 Reference", sample == 0);

        // Reset error accumulation array
        cudaMemset(d_error_results, 0, 2 * sizeof(double));

        // Compute errors entirely on device

        compute_frobenius_error_kernel<ACCUMULATE_TYPE><<<grid_size_1d, block_size_1d, shared_mem_size>>>(
            d_C_kernel, d_C_reference_fp64, d_abs_AB_product, d_error_results, n);

        cudaDeviceSynchronize();

        // Copy only the final error results (2 doubles instead of n² floats!)
        double host_error_results[2];
        cudaMemcpy(host_error_results, d_error_results, 2 * sizeof(double), cudaMemcpyDeviceToHost);

        frobenius_errors[sample] = sqrt(host_error_results[0]);
        frobenius_M_error[sample] = sqrt(host_error_results[1]);
        // Compute beta normalized error: empirical_error / (|A||B|)
        double theoretical_bound = frobenius_M_error[sample];
        normalized_errors[sample] = frobenius_errors[sample] / theoretical_bound;

    }

    printf("Completed all %d samples\n", num_samples);

    // Compute comprehensive statistics using utility function
    ArrayStats frob_stats;
    compute_array_statistics(frobenius_errors, num_samples, &frob_stats);

    ArrayStats beta_stats;
    compute_array_statistics(normalized_errors, num_samples, &beta_stats);

    // Compute log c_hat median
    const double u_compute = unit_roundoff_from_type();  // Uses actual COMPUTE_TYPE
    double log_c_hat_median = compute_log_c_hat_median(normalized_errors, num_samples, beta_factor, u_compute);

    // Print summary
    printf("\n=== Multi-Sample Analysis Results ===\n");
    printf("Matrix Type: %s, Kernel: %s, Size: %dx%d\n", matrixTypeToString(matrix_type), kernelTypeToString(kernel_type), n, n);
    printf("Number of samples: %d\n", num_samples);
    printf("\nFrobenius Error Statistics:\n");
    printf("  Average: %.3e\n", frob_stats.average);
    printf("  Std Dev: %.3e\n", frob_stats.std_dev);
    printf("  10th %%ile: %.3e\n", frob_stats.p10);
    printf("  95th %%ile: %.3e\n", frob_stats.p95);
    printf("  Max: %.3e\n", frob_stats.maximum);
    printf("\nNormalized Error |C-C_ref|/(|A||B|) Statistics:\n");
    printf("  Average: %.3e\n", beta_stats.average);
    printf("  Std Dev: %.3e\n", beta_stats.std_dev);
    printf("  10th %%ile: %.3e\n", beta_stats.p10);
    printf("  95th %%ile: %.3e\n", beta_stats.p95);
    printf("  Max: %.3e\n", beta_stats.maximum);
    printf("Theoretical error bound factor (beta): %.6e\n", beta_factor);
    printf("Average Error_beta/beta: %.6e\n", beta_stats.average/beta_factor);
    printf("Average Error_beta/u: %.6e\n", beta_stats.average/u_compute);
    printf("Log c_hat median: %.6e\n", log_c_hat_median);

    char folder[64];
    char filename[512];

    // Determine folder based on kernel type, not data types
    if (is_mixprec_kernel(kernel_type)) {
        // Mixed precision kernels - include type info in folder name
        snprintf(folder, sizeof(folder), "data/UC_%s_UA_%s/",
             getComputeTypeString(), getAccumulateTypeString());
    } else {
        // Classical kernels - use generic folder (always FP32)
        snprintf(folder, sizeof(folder), "data/UC_UA_FP32/");
    }

    mkdir(folder, static_cast<mode_t>(0777));

    // Construct filename (same for both modes)
    snprintf(filename, sizeof(filename), "%s%s_%s_%s_n%d.csv",
         folder, output_prefix, kernelTypeToString(kernel_type),
         matrixTypeToString(matrix_type), n);

    fp = fopen(filename, "w");
    if (fp) {
        // Write header with all metadata and statistics
        fprintf(fp, "matrix_type,kernel_type,matrix_size,num_samples,split_k_slices,");
        fprintf(fp, "|C-C_ref|_avg,|C-C_ref|_std,|C-C_ref|_p10,|C-C_ref|_p95,|C-C_ref|_max,");
        fprintf(fp, "|C-C_ref|/(|A||B|)_avg,|C-C_ref|/(|A||B|)_std,|C-C_ref|/(|A||B|)_p10,|C-C_ref|/(|A||B|)_p95,|C-C_ref|/(|A||B|)_max,");
        fprintf(fp, "theoretical_beta,UC,UA,E_{AB}/beta,E_{AB}/u_c,log_c_hat_median\n");

        // Write single row with all the summary data
        fprintf(fp, "%s,%s,%d,%d,%d,",
                matrixTypeToString(matrix_type),
                kernelTypeToString(kernel_type),
                n,
                num_samples,
                SPLIT_K_SLICES);
        fprintf(fp, "%.16e,%.16e,%.16e,%.16e,%.16e,",
                frob_stats.average, frob_stats.std_dev, frob_stats.p10, frob_stats.p95, frob_stats.maximum);
        fprintf(fp, "%.16e,%.16e,%.16e,%.16e,%.16e,",
                beta_stats.average, beta_stats.std_dev, beta_stats.p10, beta_stats.p95, beta_stats.maximum);
        fprintf(fp, "%.16e,%s,%s,%.16e,%.16e,%.16e\n",
                beta_factor,                     // theoretical_beta
                getComputeTypeString(),          // UC
                getAccumulateTypeString(),       // UA
                beta_stats.average/beta_factor,  // E_{AB}/beta
                beta_stats.average/u_compute,    // E_{AB}/u_c
                log_c_hat_median);               // log_c_hat_median

        fclose(fp);
        printf("\nSummary results saved to: %s\n", filename);
    }

    // Cleanup device memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_kernel);
    cudaFree(d_C_reference_fp64);
    cudaFree(d_abs_AB_product); cudaFree(d_error_results);

    // Cleanup host memory (much less now!)
    free(frobenius_errors); free(frobenius_M_error); free(normalized_errors);

    // Cleanup seed array
    delete[] seeds;
}

// New function to compute per-element normalized ERROR values and analyze per-tile statistics
// This generates matrices with the same seed as run_multi_sample_analysis
// and computes |C_kernel - C_reference| / (|A| * |B|) for each matrix element,
// then analyzes statistics within each TILE_SIZE x TILE_SIZE tile
void run_per_tile_reference_analysis(MatrixType matrix_type, KernelType kernel_type,
                                    int n, int sample_index, const char* output_prefix,
                                    bool reproducible) {
    printf("\n=== Per-Tile Error Analysis ===\n");
    printf("Matrix Type: %d, Kernel: %d, Size: %dx%d, Sample: %d\n",
           (int)matrix_type, (int)kernel_type, n, n, sample_index);
    printf("Using TILE_SIZE: %d (from config)\n", TILE_SIZE);

    // Allocate device memory
    size_t size = n * n * sizeof(float);
    size_t size_fp64 = n * n * sizeof(double);

    float *d_A, *d_B, *d_C_kernel;
    double *d_C_reference_fp64;
    double *d_A_abs, *d_B_abs;
    double *d_abs_AB_product;
    float *d_normalized;  // Result of division (FP32 for storage)

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_kernel, size);
    cudaMalloc(&d_C_reference_fp64, size_fp64);
    cudaMalloc(&d_A_abs, size_fp64);
    cudaMalloc(&d_B_abs, size_fp64);
    cudaMalloc(&d_abs_AB_product, size_fp64);
    cudaMalloc(&d_normalized, size);

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Generate the same seed as in run_multi_sample_analysis
    unsigned long long* seeds = new unsigned long long[sample_index + 1];
    if (reproducible) {
        generate_seed_array(seeds, sample_index + 1, ERROR_SEED);
    } else {
        unsigned long long time_seed = (unsigned long long)time(NULL);
        generate_seed_array(seeds, sample_index + 1, time_seed);
    }

    // Use the same seed generation logic as run_multi_sample_analysis
    auto seedA = seeds[sample_index];
    auto seedB = seedA ^ 0x94D049BB133111EBull;

    printf("Using seedA=%llu, seedB=%llu (same as sample %d in multi-sample analysis)\n",
           seedA, seedB, sample_index);

    // Generate matrices with the same seeds
    generate_matrix_device_with_seed(d_A, n, matrix_type, seedA);
    generate_matrix_device_with_seed(d_B, n, matrix_type, seedB);

    // CRITICAL: Use the EXACT SAME kernel configuration as run_multi_sample_analysis
    dim3 threadsPerBlock, numBlocks;
    compute_kernel_dimensions_dispatch(kernel_type, n, &threadsPerBlock, &numBlocks);

    int total_elements = n * n;
    int block_size_1d, grid_size_1d;
    compute_kernel_dimensions_dispatch_1D(total_elements, &block_size_1d, &grid_size_1d);

    // Compute FP64 reference using SAME configuration as run_multi_sample_analysis
    compute_reference_fp64_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C_reference_fp64, n);

    // Launch the specified kernel using SAME dispatch as run_multi_sample_analysis
    if (is_mixprec_kernel(kernel_type)) {
        launch_mixprec_kernel_by_type<COMPUTE_TYPE, ACCUMULATE_TYPE>(
            kernel_type, (COMPUTE_TYPE*)d_A, (COMPUTE_TYPE*)d_B, (ACCUMULATE_TYPE*)d_C_kernel, n, numBlocks, threadsPerBlock);
    } else if (areBothTypesFP32()) {
        launch_basic_kernel_by_type(kernel_type, d_A, d_B, d_C_kernel, n, numBlocks, threadsPerBlock);
    } else {
        printf("ERROR: Non-mixprec kernels require FP32 types\n");
        return;
    }

    // Compute absolute value matrices (convert to FP64)
    compute_matrix_abs_fp64_kernel<<<grid_size_1d, block_size_1d>>>(d_A, d_A_abs, total_elements);
    compute_matrix_abs_fp64_kernel<<<grid_size_1d, block_size_1d>>>(d_B, d_B_abs, total_elements);

    // Compute |A| * |B| using cuBLAS (double precision)
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, d_B_abs, n, d_A_abs, n, &beta, d_abs_AB_product, n);

    // Compute per-entry normalized ERROR values on device: |C_kernel - C_reference| / (|A| * |B|)
    // Use compute_EAB_entrywise instead of compute_per_entry_normalized_error
    const double denom_floor = 1e-300;  // Floor to avoid division by near-zero values

    // Keep normalized results in FP64 throughout
    double *d_normalized_fp64;  // Change from float to double
    cudaMalloc(&d_normalized_fp64, n * n * sizeof(double));  // Allocate as double

    compute_EAB_entrywise<<<grid_size_1d, block_size_1d>>>(
        d_C_kernel, d_C_reference_fp64, d_abs_AB_product, d_normalized_fp64, total_elements, denom_floor);

    // NO conversion step needed anymore - keep in FP64
    cudaDeviceSynchronize();

    // Copy FP64 normalized errors to host
    double *h_normalized = (double*)malloc(n * n * sizeof(double));
    cudaMemcpy(h_normalized, d_normalized_fp64, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate tile information (but don't compute statistics)
    int num_tiles_per_dim = (n + TILE_SIZE - 1) / TILE_SIZE;  // Ceiling division
    int total_tiles = num_tiles_per_dim * num_tiles_per_dim;

    printf("Matrix divided into %dx%d = %d tiles of size %dx%d\n",
           num_tiles_per_dim, num_tiles_per_dim, total_tiles, TILE_SIZE, TILE_SIZE);

    printf("Per-element |C-C_ref|/(|A|*|B|) computed for full matrix.\n");
    printf("Tile statistics will be computed in plotting script.\n");

    // Save error matrix + tile information to binary file
    char filename[256];
    snprintf(filename, sizeof(filename), "data/%s_%s_per_tile_n%d_sample%d.bin",
             output_prefix, matrixTypeToString(matrix_type), n, sample_index);

    FILE* fp = fopen(filename, "wb");
    if (fp) {
        // Write header (6 integers: n, sample_index, matrix_type, kernel_type, tile_size, num_tiles_per_dim)
        fwrite(&n, sizeof(int), 1, fp);
        fwrite(&sample_index, sizeof(int), 1, fp);
        fwrite(&matrix_type, sizeof(MatrixType), 1, fp);
        fwrite(&kernel_type, sizeof(KernelType), 1, fp);
        int tile_size_val = TILE_SIZE;
        fwrite(&tile_size_val, sizeof(int), 1, fp);
        fwrite(&num_tiles_per_dim, sizeof(int), 1, fp);

        // Write the per-element error matrix (FP64)
        fwrite(h_normalized, sizeof(double), n * n, fp);

        fclose(fp);
        printf("Error matrix saved to: %s\n", filename);
        printf("File size: %.2f MB\n", (double)(6*sizeof(int) + n*n*sizeof(double)) / (1024*1024));
    } else {
        printf("Error: Could not open file for writing\n");
    }

    // Simplified CSV with just basic info (no pre-computed tile stats)
    snprintf(filename, sizeof(filename), "data/%s_%s_per_tile_n%d_sample%d_info.csv",
             output_prefix, matrixTypeToString(matrix_type), n, sample_index);

    fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "matrix_type,kernel_type,matrix_size,sample_index,seedA,seedB,tile_size,num_tiles_per_dim,total_tiles\n");
        fprintf(fp, "%s,%s,%d,%d,%llu,%llu,%d,%d,%d\n",
                matrixTypeToString(matrix_type), kernelTypeToString(kernel_type),
                n, sample_index, seedA, seedB, TILE_SIZE, num_tiles_per_dim, total_tiles);
        fclose(fp);
        printf("Metadata saved to: %s\n", filename);
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_kernel);
    cudaFree(d_C_reference_fp64);
    cudaFree(d_A_abs); cudaFree(d_B_abs); cudaFree(d_abs_AB_product);
    cudaFree(d_normalized_fp64);  // Updated variable name
    cublasDestroy(cublas_handle);
    free(h_normalized);
    delete[] seeds;

    printf("Per-tile error analysis completed.\n");
}


template __global__ void compute_reference_and_norm_fp64_device<float>(const float* A, const float* B, double* C_ref, double* abs_AB_product, int n);
template __global__ void compute_reference_and_norm_fp64_device<double>(const double* A, const double* B, double* C_ref, double* abs_AB_product, int n);

#ifdef __CUDA_FP16_TYPES_EXIST__
template __global__ void compute_reference_and_norm_fp64_device<__half>(const __half* A, const __half* B, double* C_ref, double* abs_AB_product, int n);
#endif

#ifdef __CUDA_BF16_TYPES_EXIST__
template __global__ void compute_reference_and_norm_fp64_device<__nv_bfloat16>(const __nv_bfloat16* A, const __nv_bfloat16* B, double* C_ref, double* abs_AB_product, int n);
#endif

// Current build configuration
template __global__ void compute_frobenius_error_kernel<ACCUMULATE_TYPE>(ACCUMULATE_TYPE*, double*, double*, double*, int);