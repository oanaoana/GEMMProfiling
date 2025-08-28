#pragma once
#include <cuda_runtime.h>
#include "config.h"  // For SIZES and NUM_SIZES

// FIX: Update NUM_TESTS to match your implementations
#define NUM_TESTS 9 // Increase to include tiled_opt

// Unified kernel dispatch - no longer need function pointers
extern const char* available_test_names[NUM_TESTS];

void runAllBenchmarks(bool* enabled_tests, bool* enabled_sizes);
void runSingleBenchmark(const char* test_name, int matrix_size);

extern bool g_enable_verification;

extern int g_pitch_A;
extern int g_pitch_B;
extern int g_pitch_C;
extern bool g_use_pitched_memory;