#pragma once
#include <cuda_runtime.h>

// Define kernel function pointer type
typedef void (*KernelFunc)(float*, float*, float*, int, dim3, dim3);

// Define test case structure
typedef struct {
    const char* name;
    KernelFunc kernel;
    bool enabled;
} TestCase;

// Define the number of available tests
#define NUM_TESTS 3  // Adjust this based on how many tests you have

// Exported functions
void runAllBenchmarks(bool* enabled_tests, bool* enabled_sizes);

// External variables that will be defined in benchmark.cu
extern TestCase available_tests[NUM_TESTS];
extern const int SIZES[];
extern const int NUM_SIZES;