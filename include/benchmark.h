#pragma once
#include <cuda_runtime.h>
#include "config.h"  // For SIZES and NUM_SIZES

// FIX: Update NUM_TESTS to match your implementations
#define NUM_TESTS 8 // Increase from 6 to 8

typedef void (*KernelFunc)(float*, float*, float*, int, dim3, dim3);
typedef void (*KernelFuncPitched)(float*, float*, float*, int, dim3, dim3, int);

// Keep the original TestCase structure simple
typedef struct {
    const char* name;
    KernelFunc kernel;
    bool enabled;
} TestCase;

extern TestCase available_tests[NUM_TESTS];

void runAllBenchmarks(bool* enabled_tests, bool* enabled_sizes);
void runNumericalAnalysisBenchmarks(bool* enabled_sizes);

extern bool g_enable_verification;

extern int g_pitch_A;
extern int g_pitch_B;
extern int g_pitch_C;
extern bool g_use_pitched_memory;