// Simple Memory Coalescing Test Runner with block/tile size definitions
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define default block and tile sizes if not already defined in mat_load_patterns.cu
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

// Include the load pattern functions
#include "mat_load_patterns.cu"  // This brings in all kernels and functions

// Runtime variables to override defaults
int g_block_size = BLOCK_SIZE;
int g_tile_size = TILE_SIZE;

void printUsage() {
    printf("Memory Coalescing Test Runner\n");
    printf("Usage: ./memory_test [options]\n\n");
    printf("Required:\n");
    printf("  --memory-test=N       Run memory coalescing test for NxN matrix\n");
    printf("\nOptional:\n");
    printf("  --help                Show this help\n");
    printf("  --pattern=PATTERN     Access pattern: rowmajor, colmajor, random (default: rowmajor)\n");
    printf("  --config=CONFIG       Thread config: 1d-tile, 2d-tile, 1d-block, 2d-block (default: 1d-tile)\n");
    printf("  --block-size=SIZE     Override BLOCK_SIZE for block configs (default: %d)\n", BLOCK_SIZE);
    printf("  --tile-size=SIZE      Override TILE_SIZE for tile configs (default: %d)\n", TILE_SIZE);
    printf("\nThread Configurations:\n");
    printf("  1d-block              1D layout with BLOCK_SIZE*BLOCK_SIZE threads per block\n");
    printf("  1d-tile               1D layout with TILE_SIZE*TILE_SIZE threads per block\n");
    printf("  2d-block              2D layout with (BLOCK_SIZE,BLOCK_SIZE) threads per block\n");
    printf("  2d-tile               2D layout with (TILE_SIZE,TILE_SIZE) threads per block\n");
    printf("\nAccess Patterns:\n");
    printf("  rowmajor              Sequential memory access (good coalescing expected)\n");
    printf("  colmajor              Strided memory access (poor coalescing expected)\n");
    printf("  random                Random memory access (worst coalescing expected)\n");
    printf("\nExamples:\n");
    printf("  ./memory_test --memory-test=512\n");
    printf("  ./memory_test --memory-test=512 --pattern=rowmajor --config=1d-tile\n");
    printf("  ./memory_test --memory-test=512 --pattern=colmajor --config=2d-tile\n");
    printf("  ./memory_test --memory-test=512 --pattern=rowmajor --config=1d-block --block-size=8\n");
    printf("  ./memory_test --memory-test=512 --pattern=colmajor --config=2d-tile --tile-size=32\n");
    printf("\nProfiling Commands:\n");
    printf("  ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \\\n");
    printf("      ./memory_test --memory-test=512 --pattern=rowmajor --config=1d-tile\n");
    printf("\nNotes:\n");
    printf("  - Default BLOCK_SIZE=%d and TILE_SIZE=%d can be overridden at runtime\n", BLOCK_SIZE, TILE_SIZE);
    printf("  - Use --block-size/--tile-size to override these values\n");
    printf("  - Total threads per block must not exceed 1024\n");
    printf("  - 1D configs use SIZE*SIZE threads, 2D configs use (SIZE,SIZE) layout\n");
}

void runSingleMemoryTest(int size, const char* pattern, const char* config) {
    printf("Running %s memory test on %dx%d matrix with %s configuration\n",
           pattern, size, size, config);

    // Allocate and initialize
    float *h_A, *h_C;
    float *d_A, *d_C;
    size_t matrix_size = size * size * sizeof(float);

    h_A = (float*)malloc(matrix_size);
    h_C = (float*)malloc(matrix_size);

    for (int i = 0; i < size * size; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
    }

    // Use pitched memory allocation (like original)
    size_t pitch_A, pitch_C;
    cudaMallocPitch((void**)&d_A, &pitch_A, size * sizeof(float), size);
    cudaMallocPitch((void**)&d_C, &pitch_C, size * sizeof(float), size);

    cudaMemcpy2D(d_A, pitch_A, h_A, size * sizeof(float),
                 size * sizeof(float), size, cudaMemcpyHostToDevice);

    g_pitch_A = pitch_A / sizeof(float);
    g_pitch_C = pitch_C / sizeof(float);

    printf("Pitch A: %d floats, Pitch C: %d floats, Regular stride: %d\n",
           g_pitch_A, g_pitch_C, size);

    // Configure thread layout based on config parameter
    dim3 chosen_threads, chosen_blocks;

    if (strcmp(config, "1d-block") == 0) {
        chosen_threads = dim3(g_block_size * g_block_size, 1, 1);
        chosen_blocks = dim3((size * size + (g_block_size * g_block_size) - 1) / (g_block_size * g_block_size), 1, 1);
        printf("Using 1D with BLOCK_SIZE: %d threads per block\n", g_block_size * g_block_size);
    } else if (strcmp(config, "1d-tile") == 0) {
        chosen_threads = dim3(g_tile_size * g_tile_size, 1, 1);
        chosen_blocks = dim3((size * size + (g_tile_size * g_tile_size) - 1) / (g_tile_size * g_tile_size), 1, 1);
        printf("Using 1D with TILE_SIZE: %d threads per block\n", g_tile_size * g_tile_size);
    } else if (strcmp(config, "2d-block") == 0) {
        chosen_threads = dim3(g_block_size, g_block_size);
        chosen_blocks = dim3((size + g_block_size - 1) / g_block_size, (size + g_block_size - 1) / g_block_size);
        printf("Using 2D with BLOCK_SIZE: (%d,%d) threads per block\n", g_block_size, g_block_size);
    } else if (strcmp(config, "2d-tile") == 0) {
        chosen_threads = dim3(g_tile_size, g_tile_size);
        chosen_blocks = dim3((size + g_tile_size - 1) / g_tile_size, (size + g_tile_size - 1) / g_tile_size);
        printf("Using 2D with TILE_SIZE: (%d,%d) threads per block\n", g_tile_size, g_tile_size);
    } else {
        printf("Unknown config: %s. Using default 1d-tile\n", config);
        chosen_threads = dim3(g_tile_size * g_tile_size, 1, 1);
        chosen_blocks = dim3((size * size + (g_tile_size * g_tile_size) - 1) / (g_tile_size * g_tile_size), 1, 1);
    }

    printf("Grid: (%d,%d,%d), Block: (%d,%d,%d)\n",
           chosen_blocks.x, chosen_blocks.y, chosen_blocks.z,
           chosen_threads.x, chosen_threads.y, chosen_threads.z);
    printf("Total threads: %d\n",
           chosen_blocks.x * chosen_blocks.y * chosen_threads.x * chosen_threads.y);

    // Launch the appropriate test based on pattern
    if (strcmp(pattern, "rowmajor") == 0) {
        launch_copy_test_rowmajor(d_A, nullptr, d_C, size, chosen_blocks, chosen_threads);
    } else if (strcmp(pattern, "colmajor") == 0) {
        launch_copy_test_colmajor(d_A, nullptr, d_C, size, chosen_blocks, chosen_threads);
    } else if (strcmp(pattern, "random") == 0) {
        launch_copy_test_random(d_A, nullptr, d_C, size, chosen_blocks, chosen_threads);
    } else {
        printf("Unknown pattern: %s\n", pattern);
        goto cleanup;
    }

    printf("Test completed. Check ncu output for coalescing metrics.\n\n");

cleanup:
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
}

int main(int argc, char** argv) {
    int size = 0;
    const char* pattern = "rowmajor";  // Default
    const char* config = "1d-tile";    // Default

    // Parse command line arguments - EXACTLY like original main.cu
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            printUsage();
            return 0;
        } else if (strncmp(argv[i], "--memory-test=", 14) == 0) {
            size = atoi(argv[i] + 14);
        } else if (strncmp(argv[i], "--pattern=", 10) == 0) {
            pattern = argv[i] + 10;
        } else if (strncmp(argv[i], "--config=", 9) == 0) {
            config = argv[i] + 9;
        } else if (strncmp(argv[i], "--block-size=", 13) == 0) {
            g_block_size = atoi(argv[i] + 13);
            if (g_block_size <= 0 || g_block_size > 32) {
                printf("Error: block-size must be between 1 and 32\n");
                return 1;
            }
        } else if (strncmp(argv[i], "--tile-size=", 12) == 0) {
            g_tile_size = atoi(argv[i] + 12);
            if (g_tile_size <= 0 || g_tile_size > 32) {
                printf("Error: tile-size must be between 1 and 32\n");
                return 1;
            }
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage();
            return 1;
        }
    }

    if (size == 0) {
        printf("Error: Must specify --memory-test=SIZE\n");
        printUsage();
        return 1;
    }

    // Validate thread sizes
    if (g_block_size * g_block_size > 1024) {
        printf("Error: block-size %d creates %d threads per block (max 1024)\n",
               g_block_size, g_block_size * g_block_size);
        return 1;
    }

    if (g_tile_size * g_tile_size > 1024) {
        printf("Error: tile-size %d creates %d threads per block (max 1024)\n",
               g_tile_size, g_tile_size * g_tile_size);
        return 1;
    }

    printf("Using: block_size=%d, tile_size=%d\n", g_block_size, g_tile_size);

    // // Print device info once at the start
    // printDeviceInfo();
    // printCacheInfo();

    // Run the single memory test - exactly like original
    runSingleMemoryTest(size, pattern, config);

    return 0;
}