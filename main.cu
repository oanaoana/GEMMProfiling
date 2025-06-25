// main.cu
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdlib.h>
#include "benchmark.h"

void printUsage() {
    printf("Usage: ./main [options]\n\n");
    printf("Options:\n");
    printf("  --help           Show this help\n");
    printf("  --test=NAME      Run only specified test (naive, tiled, cublas)\n");
    printf("  --size=N         Run only specified matrix size\n");
    printf("  --all            Run all tests and sizes\n");
}

int main(int argc, char **argv) {
    // Add debug mode first
    if (argc > 1 && strcmp(argv[1], "--debug") == 0) {
        printf("=== Debug Mode ===\n");

        // Test basic CUDA
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        printf("✓ CUDA works, found %d device(s)\n", device_count);

        // Test device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("✓ GPU: %s\n", prop.name);

        printf("✓ Debug test passed\n");
        return 0;
    }

    // Default: enable all tests and sizes
    bool enabled_tests[NUM_TESTS] = {true, true, true, true, true}; // All enabled by default
    bool enabled_sizes[NUM_SIZES] = {true, true, true, true, true}; // 5 elements: 256,512,1024,2048,4096

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            printUsage();
            return 0;
        } else if (strncmp(argv[i], "--test=", 7) == 0) {
            // Disable all tests first
            for (int j = 0; j < NUM_TESTS; j++) {
                enabled_tests[j] = false;
            }

            // Enable specific test
            const char* test_name = argv[i] + 7;
            bool found = false;

            for (int j = 0; j < NUM_TESTS; j++) {
                if (strcmp(available_tests[j].name, test_name) == 0) {
                    enabled_tests[j] = true;
                    found = true;
                    break;
                }
            }

            if (!found) {
                printf("Unknown test: %s\n", test_name);
                printf("Available tests: ");
                for (int j = 0; j < NUM_TESTS; j++) {
                    printf("%s ", available_tests[j].name);
                }
                printf("\n");
                return 1;
            }
        } else if (strncmp(argv[i], "--size=", 7) == 0) {
            // Disable all sizes first
            for (int j = 0; j < NUM_SIZES; j++) {
                enabled_sizes[j] = false;
            }

            // Enable specific size
            int target_size = atoi(argv[i] + 7);
            bool found = false;

            for (int j = 0; j < NUM_SIZES; j++) {
                if (SIZES[j] == target_size) {
                    enabled_sizes[j] = true;
                    found = true;
                    break;
                }
            }

            if (!found) {
                printf("Size %d not supported\n", target_size);
                printf("Available sizes: ");
                for (int j = 0; j < NUM_SIZES; j++) {
                    printf("%d ", SIZES[j]);
                }
                printf("\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--all") == 0) {
            // Already enabled all by default
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage();
            return 1;
        }
    }

    printf("GEMM Performance Profiling\n");
    printf("==========================\n");

    // Initialize CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Show which tests will run
    printf("\nEnabled tests: ");
    for (int i = 0; i < NUM_TESTS; i++) {
        if (enabled_tests[i]) {
            printf("%s ", available_tests[i].name);
        }
    }
    printf("\n");

    printf("Enabled sizes: ");
    for (int i = 0; i < NUM_SIZES; i++) {
        if (enabled_sizes[i]) {
            printf("%d ", SIZES[i]);
        }
    }
    printf("\n\n");

    // TEMPORARILY COMMENT OUT THE BENCHMARK CALL
    printf("About to call runAllBenchmarks...\n");
    fflush(stdout);

    // Start profiling if using CUDA profiler
    cudaProfilerStart();

    // Run benchmarks - NOW PROPERLY IMPLEMENTED
    runAllBenchmarks(enabled_tests, enabled_sizes);

    // Stop profiling
    cudaProfilerStop();

    printf("\nBenchmarking complete!\n");
    return 0;
}