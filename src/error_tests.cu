// error_tests.cu - Numerical error analysis tests for GEMM implementations
#include "../include/error_tests.cuh"
#include "../include/benchmark.h"
#include "../include/gemms.cuh"
#include "../include/utils.cuh"
#include "../include/numerical_analysis.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

// Setup matrix data based on type
void setupMatrix(float* matrix, int n, MatrixType type, const char* filename) {
    switch (type) {
        case MATRIX_RANDOM:
            fill_matrix(matrix, n);
            break;

        case MATRIX_WELL_CONDITIONED:
            // Identity-like matrix
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix[i * n + j] = (i == j) ? 1.0f : 0.01f;
                }
            }
            break;

        case MATRIX_ILL_CONDITIONED:
            // Hilbert-like matrix
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix[i * n + j] = 1.0f / (1.0f + abs(i - j));
                }
            }
            break;

        case MATRIX_IDENTITY:
            // Pure identity matrix
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix[i * n + j] = (i == j) ? 1.0f : 0.0f;
                }
            }
            break;

        case MATRIX_HILBERT:
            // True Hilbert matrix
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix[i * n + j] = 1.0f / (float)(i + j + 1);
                }
            }
            break;

        case MATRIX_FROM_FILE:
            // TODO: Implement matrix loading from file
            printf("Loading matrix from file: %s\n", filename ? filename : "unknown");
            // For now, fall back to random
            fill_matrix(matrix, n);
            break;

        case MATRIX_CUSTOM:
            // Placeholder for custom matrix generation
            // For now, fall back to random
            fill_matrix(matrix, n);
            break;

        default:
            printf("Unknown matrix type, using random\n");
            fill_matrix(matrix, n);
            break;
    }
}

// Run matrix tests with specified configuration
void runMatrixTests(int n, MatrixTestConfig* configs, int num_configs) {
    printf("\n--- Testing matrix size %d x %d ---\n", n, n);

    size_t size = n * n * sizeof(float);

    // Allocate host memory once for all tests
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);

    if (!h_A || !h_B) {
        printf("ERROR: Failed to allocate host memory\n");
        return;
    }

    // Run each enabled test configuration
    for (int i = 0; i < num_configs; i++) {
        if (!configs[i].enabled) continue;

        printf("\n=== Running test: %s ===\n", configs[i].name);
        printf("Description: %s\n", configs[i].description);

        // Setup matrices according to configuration
        setupMatrix(h_A, n, configs[i].type_A, configs[i].filename_A);
        setupMatrix(h_B, n, configs[i].type_B, configs[i].filename_B);

        // Generate output filename
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename),
                "data/numerical_analysis_%s_n%d_tile%d.dat", configs[i].name, n, TILE_SIZE);

        printf("Running numerical analysis for %s...\n", configs[i].name);
        run_numerical_analysis(h_A, h_B, n, output_filename);
    }

    // Cleanup
    free(h_A);
    free(h_B);
}

// Generate comprehensive report from all test results
void generateReport(bool* enabled_sizes) {
    printf("\n=== Generating Numerical Analysis Report ===\n");

    FILE* summaryFile = fopen("data/numerical_analysis_summary.csv", "w");
    if (!summaryFile) {
        printf("ERROR: Could not create data/numerical_analysis_summary.csv\n");
        return;
    }

    fprintf(summaryFile, "test_name,size,avg_abs_error,max_rel_error,significant_errors,avg_condition_A,avg_condition_B\n");

    // TODO: Parse all generated .dat files and create summary statistics
    // For now, just write header and close
    printf("Report generation functionality to be implemented\n");
    printf("Summary will include analysis from all generated .dat files\n");

    fclose(summaryFile);
    printf("Report saved to: data/numerical_analysis_summary.csv\n");
}

// Main function for numerical analysis benchmarks - now refactored
void runNumericalAnalysisBenchmarks(bool* enabled_sizes) {
    printf("=== Starting Numerical Analysis of Tiled GEMM ===\n");

    // Define test configurations
    MatrixTestConfig test_configs[] = {
        {MATRIX_RANDOM, MATRIX_RANDOM, "random", "Random matrices with uniform distribution", NULL, NULL, true},
        {MATRIX_WELL_CONDITIONED, MATRIX_WELL_CONDITIONED, "wellcond", "Well-conditioned identity-like matrices", NULL, NULL, true},
        {MATRIX_ILL_CONDITIONED, MATRIX_ILL_CONDITIONED, "illcond", "Ill-conditioned Hilbert-like matrices", NULL, NULL, true},
        {MATRIX_IDENTITY, MATRIX_IDENTITY, "identity", "Pure identity matrices", NULL, NULL, false},
        {MATRIX_HILBERT, MATRIX_HILBERT, "hilbert", "True Hilbert matrices", NULL, NULL, false},
        // Add more configurations as needed
    };

    int num_configs = sizeof(test_configs) / sizeof(test_configs[0]);

    // Run tests for each enabled size
    for (int i = 0; i < NUM_SIZES; i++) {
        if (!enabled_sizes[i]) continue;

        int n = SIZES[i];
        runMatrixTests(n, test_configs, num_configs);
    }

    // Generate comprehensive report
    generateReport(enabled_sizes);

    printf("\nNumerical analysis complete!\n");
    printf("Individual test results: data/numerical_analysis_*.dat\n");
    printf("Summary report: data/numerical_analysis_summary.csv\n");
}
