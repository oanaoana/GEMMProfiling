// config.cu - Configuration assignment and defaults
#include "../include/config.h"
#include <string.h>  // for strcmp
#include <cuda_runtime.h>  // for dim3

// Set routines that take user params or defaults
void set_matrix_config(MatrixTypeConfig* config, MatrixType matrix_type, const char* name,
                       bool enabled, float param1, float param2) {
    config->matrix_type = matrix_type;
    config->name = name;
    config->enabled = enabled;
    config->param1 = param1;
    config->param2 = param2;
}

void set_kernel_config(KernelTypeConfig* config, KernelType kernel_type, const char* name, bool enabled) {
    config->kernel_type = kernel_type;
    config->name = name;
    config->enabled = enabled;
}

void set_allocation(KernelAllocation* alloc, int block_x, int block_y,
                    int grid_x, int grid_y, bool use_dynamic_grid) {
    alloc->block_x = block_x;
    alloc->block_y = block_y;
    alloc->grid_x = grid_x;
    alloc->grid_y = grid_y;
    alloc->use_dynamic_grid = use_dynamic_grid;
}

void set_tiles(TileConfig* tiles, int tile_size, int tile_m, int tile_n, int tile_k) {
    tiles->tile_size = tile_size;
    tiles->tile_m = tile_m;
    tiles->tile_n = tile_n;
    tiles->tile_k = tile_k;
}

// ============================================================================
// GLOBAL CONFIGURATION VARIABLES
// ============================================================================

// Matrix sizes for benchmarking and testing
const int SIZES[] = DEFAULT_SIZES;
const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
