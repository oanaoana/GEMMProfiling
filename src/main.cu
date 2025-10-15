// main.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "../include/benchmark.h"
#include "../include/error_analysis.cuh"
#include "../include/utils.cuh"
#include "../include/config.h"

typedef enum {
    MODE_NONE = 0,
    MODE_PERFORMANCE,
    MODE_ERROR_ANALYSIS,
    MODE_ULP_ANALYSIS,
    MODE_TRADEOFF,
    MODE_ASSESS_RESOURCES,
    MODE_PER_TILE_ANALYSIS  // Add new mode
} RunMode;

void printUsage() {
    printf("Usage: ./main <mode> [options]\n\n");
    printf("Modes:\n");
    printf("  --performance         Run performance test for specific kernel/size\n");
    printf("  --error-analysis      Run error analysis for specific kernel/size\n");
    printf("  --ulp-analysis        Run ULP analysis for specific kernel/size\n");
    printf("  --tradeoff            Run trade-off analysis for specific kernel/size\n");
    printf("  --assess-resources    Assess kernel resource usage for specific kernel/size\n");
    printf("  --per-tile            Run per-tile error analysis for a single sample (requires --sample=N)\n");  // Updated
    printf("\nOptions:\n");
    printf("  --test=NAME           Specify kernel (required for --performance, --error-analysis, --ulp-analysis, --tradeoff, --assess-resources, --per-tile)\n");
    printf("  --size=N              Specify matrix size (required for --performance, --error-analysis, --ulp-analysis, --tradeoff, --assess-resources, --per-tile)\n");
    printf("  --matrix-type=TYPE    Specify matrix type for error analysis (optional, default: wellcond)\n");
    printf("  --sample=N            Specify sample index for per-tile analysis (required for --per-tile, optional for others, default: 0)\n");  // Updated
    printf("  --help                Show this help\n");
    printf("\nAvailable kernels:\n");
    printf("  naive, tiled, tiled_opt, tiled_pairwise, tiled_rect\n");
    printf("  tiled_mixprec         Mixed precision kernel (uses compile-time COMPUTE_TYPE/ACCUMULATE_TYPE)\n");
    printf("  cublas, cutlass, etc.\n");
    printf("\nNotes:\n");
    printf("  --per-tile analyzes error distribution within individual tiles of a single matrix sample\n");
    printf("  --error-analysis runs multi-sample statistical analysis across many random matrices\n");
}

int main(int argc, char **argv) {

    // Parse arguments
    RunMode mode = MODE_NONE;
    char test_name[64] = "";
    int matrix_size = 0;
    int sample_index = 0;  // Add sample index variable
    MatrixType matrix_type = MATRIX_ODO_WELL_CONDITIONED; // Default to well-conditioned

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage();
            return 0;
        } else if (strcmp(argv[i], "--performance") == 0) {
            mode = MODE_PERFORMANCE;
        } else if (strcmp(argv[i], "--error-analysis") == 0) {
            mode = MODE_ERROR_ANALYSIS;
        } else if (strcmp(argv[i], "--ulp-analysis") == 0) {
            mode = MODE_ULP_ANALYSIS;
        } else if (strcmp(argv[i], "--assess-resources") == 0) {
            mode = MODE_ASSESS_RESOURCES;
        } else if (strcmp(argv[i], "--tradeoff") == 0) {
            mode = MODE_TRADEOFF;
        } else if (strcmp(argv[i], "--per-tile") == 0) {  // Add per-tile mode parsing
            mode = MODE_PER_TILE_ANALYSIS;
        } else if (strncmp(argv[i], "--test=", 7) == 0) {
            strncpy(test_name, argv[i] + 7, sizeof(test_name) - 1);
            test_name[sizeof(test_name) - 1] = '\0';
        } else if (strncmp(argv[i], "--size=", 7) == 0) {
            matrix_size = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--sample=", 9) == 0) {  // Add sample parsing
            sample_index = atoi(argv[i] + 9);
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
        printf("Error: Must specify a mode (--all, --performance, --error-analysis, --ulp-analysis, --complete, --assess-resources, or --per-tile)\n");
        printUsage();
        return 1;
    }

    if ((mode == MODE_PERFORMANCE || mode == MODE_ERROR_ANALYSIS || mode == MODE_ULP_ANALYSIS || mode == MODE_TRADEOFF || mode == MODE_ASSESS_RESOURCES || mode == MODE_PER_TILE_ANALYSIS) && strlen(test_name) == 0) {
        printf("Error: --test=NAME is required for performance, error analysis, ULP analysis, tradeoff, assess-resources, and per-tile modes\n");
        printUsage();
        return 1;
    }

    if ((mode == MODE_PERFORMANCE || mode == MODE_ERROR_ANALYSIS || mode == MODE_ULP_ANALYSIS || mode == MODE_TRADEOFF || mode == MODE_ASSESS_RESOURCES || mode == MODE_PER_TILE_ANALYSIS) && matrix_size <= 0) {
        printf("Error: --size=N is required for performance, error analysis, ULP analysis, tradeoff, assess-resources, and per-tile modes\n");
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

        case MODE_PERFORMANCE: {
            KernelType kernel_type = getKernelTypeFromName(test_name);
            printf("\nRunning performance test: %s at size %d\n", test_name, matrix_size);
            runKernelPerformance(kernel_type, matrix_size);
            break;
        }

        case MODE_ERROR_ANALYSIS: {
            printf("\nRunning error analysis: %s at size %d\n", test_name, matrix_size);

            KernelType kernel_type = getKernelTypeFromName(test_name);

            // Add this validation check:
            if (kernel_type == (KernelType)-1) {
                printf("Error: Test '%s' not found\n", test_name);
                printf("Available tests: naive, tiled, tiled_opt, tiled_pairwise, tiled_rect, tiled_mixprec, cublas, cutlass\n");
                return 1;
            }

            char output_name[128];
            snprintf(output_name, sizeof(output_name), "error_analysis_%s", test_name);

            run_multi_sample_analysis(matrix_type, kernel_type, matrix_size, DEFAULT_NUM_SAMPLES, output_name);
            break;
        }

        case MODE_ULP_ANALYSIS: {
            printf("\nRunning ULP analysis: %s at size %d\n", test_name, matrix_size);

            KernelType kernel_type = getKernelTypeFromName(test_name);
            char output_name[128];
            snprintf(output_name, sizeof(output_name), "ulp_analysis_%s", test_name);

            run_ulp_samples_analysis(matrix_type, kernel_type, matrix_size, DEFAULT_NUM_SAMPLES, output_name);
            break;
        }

        case MODE_TRADEOFF: {
            printf("\n=== Complete Analysis: %s at size %d ===\n", test_name, matrix_size);

            // First run error analysis
            printf("\n[1/2] Running Error Analysis...\n");
            KernelType kernel_type = getKernelTypeFromName(test_name);
            char output_name[128];
            snprintf(output_name, sizeof(output_name), "complete_analysis_%s", test_name);

            run_multi_sample_analysis(matrix_type, kernel_type, matrix_size, DEFAULT_NUM_SAMPLES, output_name);

            // Then run performance test
            printf("\n[2/2] Running Performance Test...\n");
            runKernelPerformance(kernel_type, matrix_size);

            printf("\n=== Complete Analysis Finished ===\n");
            break;
        }

        case MODE_PER_TILE_ANALYSIS: {  // Add new case
            printf("\nRunning per-tile reference analysis: %s at size %d, sample %d\n",
                   test_name, matrix_size, sample_index);

            KernelType kernel_type = getKernelTypeFromName(test_name);
            if (kernel_type == (KernelType)-1) {
                printf("Error: Test '%s' not found\n", test_name);
                return 1;
            }

            char output_name[128];
            snprintf(output_name, sizeof(output_name), "per_tile_%s", test_name);

            run_per_tile_reference_analysis(matrix_type, kernel_type, matrix_size,
                                           sample_index, output_name, true);
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
