#pragma once
#include <cuda_runtime.h>

// Enumeration for different matrix types
typedef enum {
    MATRIX_RANDOM,
    MATRIX_WELL_CONDITIONED,
    MATRIX_ILL_CONDITIONED,
    MATRIX_IDENTITY,
    MATRIX_HILBERT,
    MATRIX_FROM_FILE,
    MATRIX_CUSTOM
} MatrixType;

// Structure to hold matrix test configuration
typedef struct {
    MatrixType type_A;
    MatrixType type_B;
    const char* name;
    const char* description;
    const char* filename_A;  // For loading from file
    const char* filename_B;  // For loading from file
    bool enabled;
} MatrixTestConfig;

// Function declarations for error analysis tests
void runNumericalAnalysisBenchmarks(bool* enabled_sizes);
void setupMatrix(float* matrix, int n, MatrixType type, const char* filename);
void runMatrixTests(int n, MatrixTestConfig* configs, int num_configs);
void generateReport(bool* enabled_sizes);
