#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// # L1
// ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
//              l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./test_coalescence

// # L2
// ncu --metrics lts__t_requests_pipe_lsu_mem_global_op_ld.sum,\
//              lts__t_sectors_pipe_lsu_mem_global_op_ld.sum ./test_coalescence l2


__global__ void coalesced_read_kernel(float* A, float* sink, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N * N) {
        float val = A[tid];
        sink[tid] = val;  // Write to ensure read isn't optimized out
    }
}

__global__ void coalesced_l2_read_kernel(float* A, float* sink, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N * N) {
        float val = __ldg(&A[tid]);  // Try to force L2 path
        sink[tid] = val;
    }
}

void runKernel(const char* tag, void (*kernel)(float*, float*, int), float* d_A, float* d_sink, int N) {
    printf("\n=== Running kernel: %s ===\n", tag);
    dim3 threads(256);
    dim3 blocks((N * N + threads.x - 1) / threads.x);
    kernel<<<blocks, threads>>>(d_A, d_sink, N);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    const int N = 512;
    const size_t size = N * N * sizeof(float);

    float *d_A, *d_sink;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_sink, size);

    float* h_data = (float*)malloc(size);
    for (int i = 0; i < N * N; ++i) h_data[i] = static_cast<float>(i);

    cudaMemcpy(d_A, h_data, size, cudaMemcpyHostToDevice);

    if (argc > 1 && strcmp(argv[1], "l2") == 0) {
        runKernel("L2 Read Kernel", coalesced_l2_read_kernel, d_A, d_sink, N);
    } else {
        runKernel("L1 Read Kernel", coalesced_read_kernel, d_A, d_sink, N);
    }

    cudaFree(d_A);
    cudaFree(d_sink);
    free(h_data);

    return 0;
}
