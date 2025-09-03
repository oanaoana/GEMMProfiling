// main.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstring>

#include "../include/utils.cuh"
#include "../include/gemms.cuh"
#include "../include/benchmark.h"
#include "../include/error_analysis.cuh"
#include "../include/config.h"

// Simple mode enumeration
typedef enum {
    MODE_NONE = 0,
    MODE_PERFORMANCE,
    MODE_ERROR_ANALYSIS,
    MODE_COMPLETE_ANALYSIS,
    MODE_ALL_BENCHMARKS
} RunMode;

void printUsage() {
    printf("Usage: ./main <mode> [options]\n\n");
    printf("Modes:\n");
    printf("  --all                 Run all benchmark tests and sizes\n");
    printf("  --performance         Run performance test for specific kernel/size\n");
    printf("  --error-analysis      Run error analysis for specific kernel/size\n");
    printf("  --complete            Run both error analysis AND performance test for kernel/size\n");
    printf("\nOptions:\n");
    printf("  --test=NAME           Specify kernel (required for --performance, --error-analysis, --complete)\n");
    printf("  --size=N              Specify matrix size (required for --performance, --error-analysis, --complete)\n");
    printf("  --matrix-type=TYPE    Specify matrix type for error analysis (optional, default: wellcond)\n");
    printf("  --help                Show this help\n");
    printf("\nAvailable kernels: naive, tiled, tiled_pairwise, tiled_rect, cublas, cublas_tensor, cutlass\n");
    printf("Available matrix types: wellcond, illcond, zeromean, uniform_positive, 2powers, rademacher, sanity, lognormal, file\n");
    printf("Available sizes for --all: ");
    for (int i = 0; i < NUM_SIZES; i++) {
        printf("%d ", SIZES[i]);
    }
    printf("(other modes support any size)\n");
    printf("\nExamples:\n");
    printf("  ./main --all\n");
    printf("  ./main --performance --test=tiled --size=1024\n");
    printf("  ./main --error-analysis --test=tiled_pairwise --size=768\n");
    printf("  ./main --error-analysis --test=tiled_pairwise --size=512 --matrix-type=illcond\n");
    printf("  ./main --complete --test=tiled_pairwise --size=1024 --matrix-type=normal\n");
    printf("  ./main --complete --test=tiled_pairwise --size=1024\n");
}

int main(int argc, char **argv) {
    // Handle special debug mode
    if (argc > 1 && strcmp(argv[1], "--debug") == 0) {
        printf("=== Debug Mode ===\n");
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        printf("✓ CUDA works, found %d device(s)\n", device_count);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("✓ GPU: %s\n", prop.name);
        printf("✓ Debug test passed\n");
        return 0;
    }

    // Parse arguments
    RunMode mode = MODE_NONE;
    char test_name[64] = "";
    int matrix_size = 0;
    MatrixType matrix_type = MATRIX_ODO_WELL_CONDITIONED; // Default to well-conditioned

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage();
            return 0;
        } else if (strcmp(argv[i], "--all") == 0) {
            mode = MODE_ALL_BENCHMARKS;
        } else if (strcmp(argv[i], "--performance") == 0) {
            mode = MODE_PERFORMANCE;
        } else if (strcmp(argv[i], "--error-analysis") == 0) {
            mode = MODE_ERROR_ANALYSIS;
        } else if (strcmp(argv[i], "--complete") == 0) {
            mode = MODE_COMPLETE_ANALYSIS;
        } else if (strncmp(argv[i], "--test=", 7) == 0) {
            strncpy(test_name, argv[i] + 7, sizeof(test_name) - 1);
            test_name[sizeof(test_name) - 1] = '\0';
        } else if (strncmp(argv[i], "--size=", 7) == 0) {
            matrix_size = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--matrix-type=", 14) == 0) {
            MatrixType parsed_type = getMatrixTypeFromName(argv[i] + 14);
            if (parsed_type == static_cast<MatrixType>(-1)) {
                printf("Error: Unknown matrix type '%s'\n", argv[i] + 14);
                printf("Available types: wellcond, illcond, normal, scaled, skewed, file\n");
                return 1;
            }
            matrix_type = parsed_type;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage();
            return 1;
        }
    }

    // Validate arguments
    if (mode == MODE_NONE) {
        printf("Error: Must specify a mode (--all, --performance, or --error-analysis)\n");
        printUsage();
        return 1;
    }

    if ((mode == MODE_PERFORMANCE || mode == MODE_ERROR_ANALYSIS || mode == MODE_COMPLETE_ANALYSIS) && strlen(test_name) == 0) {
        printf("Error: --test=NAME is required for performance, error analysis, and complete modes\n");
        printUsage();
        return 1;
    }

    if ((mode == MODE_PERFORMANCE || mode == MODE_ERROR_ANALYSIS || mode == MODE_COMPLETE_ANALYSIS) && matrix_size <= 0) {
        printf("Error: --size=N is required for performance, error analysis, and complete modes\n");
        printUsage();
        return 1;
    }

    // Initialize CUDA and print info
    printf("Loading configuration...\n");
    printf("Configuration: TILE_SIZE=%d, TILE_M=%d, TILE_N=%d, TILE_K=%d\n",
           TILE_SIZE, TILE_M, TILE_N, TILE_K);

    printf("\nGEMM Performance Profiling\n");
    printf("==========================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    cudaProfilerStart();

    // Execute based on mode
    switch (mode) {
        case MODE_ALL_BENCHMARKS: {
            printf("\nRunning all benchmarks...\n");
            bool enabled_tests[NUM_TESTS];
            bool enabled_sizes[NUM_SIZES];
            for (int i = 0; i < NUM_TESTS; i++) enabled_tests[i] = true;
            for (int i = 0; i < NUM_SIZES; i++) enabled_sizes[i] = true;
            runAllBenchmarks(enabled_tests, enabled_sizes);
            break;
        }

        case MODE_PERFORMANCE: {
            printf("\nRunning performance test: %s at size %d\n", test_name, matrix_size);
            runSingleBenchmark(test_name, matrix_size);
            break;
        }

        case MODE_ERROR_ANALYSIS: {
            printf("\nRunning error analysis: %s at size %d\n", test_name, matrix_size);

            KernelType kernel_type = getKernelTypeFromName(test_name);
            char output_name[128];
            snprintf(output_name, sizeof(output_name), "error_analysis_%s", test_name);

            run_multi_sample_analysis(matrix_type, kernel_type, matrix_size, DEFAULT_NUM_SAMPLES, output_name);
            break;
        }

        case MODE_COMPLETE_ANALYSIS: {
            printf("\n=== Complete Analysis: %s at size %d ===\n", test_name, matrix_size);

            // First run error analysis
            printf("\n[1/2] Running Error Analysis...\n");
            KernelType kernel_type = getKernelTypeFromName(test_name);
            char output_name[128];
            snprintf(output_name, sizeof(output_name), "complete_analysis_%s", test_name);

            run_multi_sample_analysis(matrix_type, kernel_type, matrix_size, DEFAULT_NUM_SAMPLES, output_name);

            // Then run performance test
            printf("\n[2/2] Running Performance Test...\n");
            runSingleBenchmark(test_name, matrix_size);

            printf("\n=== Complete Analysis Finished ===\n");
            break;
        }

        default:
            printf("Error: Invalid mode\n");
            return 1;
    }

    cudaProfilerStop();
    printf("\nComplete!\n");
    return 0;
}