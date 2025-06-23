#pragma once
#include <cuda_runtime.h>

#define NUM_TESTS 3
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