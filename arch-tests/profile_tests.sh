#!/bin/bash
# filepath: /home/oana/Projects/GEMMProfiling/arch-tests/profile_tests.sh

# Comprehensive Memory Access Pattern Comparison
# Tests direct and tiled memory access patterns

MEMORY_TEST="./memory_test"
SIZE=1024
# Fix metrics initialization - use = instead of += to initialize
#NCU_METRICS="l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS+="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed"

# Set block and tile sizes
BLOCK_SIZE=16
TILE_SIZE=16

echo "=== Comprehensive Memory Access Pattern Comparison ==="
echo "Block size: $BLOCK_SIZE, Tile size: $TILE_SIZE, Matrix size: ${SIZE}x${SIZE}"
echo ""

# Part 1: Direct (non-tiled) Memory Access Patterns
echo "==== DIRECT MEMORY ACCESS PATTERNS ===="
echo ""

echo "--- Testing Direct Row-Major Pattern ---"
ncu --metrics "$NCU_METRICS" $MEMORY_TEST --memory-test=$SIZE --pattern=rowmajor --config=2d-tile \
    --tile-size=$TILE_SIZE --block-size=$BLOCK_SIZE
echo ""

echo "--- Testing Direct Column-Major Pattern ---"
ncu --metrics "$NCU_METRICS" $MEMORY_TEST --memory-test=$SIZE --pattern=colmajor --config=2d-tile \
    --tile-size=$TILE_SIZE --block-size=$BLOCK_SIZE
echo ""

echo "--- Testing Direct Random Pattern ---"
ncu --metrics "$NCU_METRICS" $MEMORY_TEST --memory-test=$SIZE --pattern=random --config=2d-tile \
    --tile-size=$TILE_SIZE --block-size=$BLOCK_SIZE
echo ""

# Part 2: Shared Memory Tiled Access Patterns
echo "==== TILED MEMORY ACCESS PATTERNS ===="
echo ""

echo "--- Testing Tiled Row-Major Pattern ---"
ncu --metrics "$NCU_METRICS" $MEMORY_TEST --memory-test=$SIZE --pattern=tiled-row --config=2d-tile \
    --tile-size=$TILE_SIZE --block-size=$BLOCK_SIZE
echo ""

echo "--- Testing Tiled Column-Major Pattern ---"
ncu --metrics "$NCU_METRICS" $MEMORY_TEST --memory-test=$SIZE --pattern=tiled-col --config=2d-tile \
    --tile-size=$TILE_SIZE --block-size=$BLOCK_SIZE
echo ""

echo "=== Testing Complete ==="
echo "Now you can compare the memory access efficiency metrics."