#pragma once
#include <cuda_runtime.h>

// FIX: Update NUM_TESTS to match your implementations
#define NUM_TESTS 5  // Was 3, now 5 for: naive, tiled, cublas, cublas_tensor, cutlass

extern const int NUM_SIZES;

typedef void (*KernelFunc)(float*, float*, float*, int, dim3, dim3);

typedef struct {
    const char* name;
    KernelFunc kernel;
    bool enabled;
} TestCase;

extern TestCase available_tests[NUM_TESTS];
extern const int SIZES[];

void runAllBenchmarks(bool* enabled_tests, bool* enabled_sizes);