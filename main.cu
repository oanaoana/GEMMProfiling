// main.cu
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <stdlib.h>  // Add this for atoi
#include "benchmark.h"

void printUsage() {
    printf("Usage: ./main [options]\n\n");
    printf("Options:\n");
    printf("  --help           Show this help\n");
    printf("  --test=NAME      Run only specified test (naive, tiled, cublas, etc)\n");
    printf("  --size=N         Run only specified matrix size\n");
    printf("  --all            Run all tests and sizes\n");
}

int main(int argc, char **argv) {
    // Default: enable all tests and sizes
    bool enabled_tests[NUM_TESTS];
    bool enabled_sizes[10]; // Assume max 10 sizes

    // Initialize to true (enable all by default)
    for (int i = 0; i < NUM_TESTS; i++) {
        enabled_tests[i] = true;
    }
    for (int i = 0; i < NUM_SIZES; i++) {
        enabled_sizes[i] = true;
    }

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

            // Enable specified test
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
                return 1;
            }
        } else if (strncmp(argv[i], "--size=", 7) == 0) {
            // Disable all sizes first
            for (int j = 0; j < NUM_SIZES; j++) {
                enabled_sizes[j] = false;
            }

            // Enable specified size
            int size = atoi(argv[i] + 7);
            bool found = false;
            for (int j = 0; j < NUM_SIZES; j++) {
                if (SIZES[j] == size) {
                    enabled_sizes[j] = true;
                    found = true;
                    break;
                }
            }

            if (!found) {
                printf("Unsupported matrix size: %d\n", size);
                return 1;
            }
        } else if (strcmp(argv[i], "--all") == 0) {
            // Already enabled by default
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage();
            return 1;
        }
    }

    // Run benchmarks
    runAllBenchmarks(enabled_tests, enabled_sizes);

    // Generate roofline plot
    printf("To generate roofline plot, run: python plot_roofline.py\n");

    return 0;
}