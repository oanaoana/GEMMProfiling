#!/bin/bash
# filepath: /home/oana/Projects/GEMMProfiling/arch-tests/compare_rowmajor_patterns.sh

# Compare different row-major access patterns with and without pitched memory

MEMORY_TEST="./memory_test"
SIZE=1024
NCU_METRICS="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
TILE_SIZE=16
BLOCK_SIZE=16

echo "=== Comparing Row-Major Memory Access Patterns ==="
echo "Matrix size: ${SIZE}x${SIZE}, Tile size: $TILE_SIZE, Block size: $BLOCK_SIZE"
echo ""

# Test row-major direct access
echo "--- Testing Direct Row-Major (1D) with PITCHED memory ---"
$MEMORY_TEST --memory-test=$SIZE --pattern=rowmajor --config=1d-tile --tile-size=$TILE_SIZE
echo ""

echo "--- Testing Direct Row-Major (1D) with REGULAR memory ---"
$MEMORY_TEST --memory-test=$SIZE --pattern=rowmajor --config=1d-tile --tile-size=$TILE_SIZE --no-pitch
echo ""
echo "----------------------------------------------------------------"
echo ""

# Test row-major 2D thread organization
echo "--- Testing Direct Row-Major (2D) with PITCHED memory ---"
$MEMORY_TEST --memory-test=$SIZE --pattern=rowmajor --config=2d-tile --tile-size=$TILE_SIZE
echo ""

echo "--- Testing Direct Row-Major (2D) with REGULAR memory ---"
$MEMORY_TEST --memory-test=$SIZE --pattern=rowmajor --config=2d-tile --tile-size=$TILE_SIZE --no-pitch
echo ""
echo "----------------------------------------------------------------"
echo ""

# Test tiled row-major
echo "--- Testing Tiled Row-Major with PITCHED memory ---"
$MEMORY_TEST --memory-test=$SIZE --pattern=tiled-row --config=2d-tile --tile-size=$TILE_SIZE
echo ""

echo "--- Testing Tiled Row-Major with REGULAR memory ---"
$MEMORY_TEST --memory-test=$SIZE --pattern=tiled-row --config=2d-tile --tile-size=$TILE_SIZE --no-pitch
echo ""
echo "----------------------------------------------------------------"
echo ""

# Measure with ncu for the most promising configurations
echo "=== Performance Profiling with NCU ==="
echo ""

# Direct 2D row-major with pitched memory
echo "--- Profiling Direct 2D Row-Major (PITCHED) ---"
ncu --metrics "$NCU_METRICS" $MEMORY_TEST --memory-test=$SIZE --pattern=rowmajor --config=2d-tile --tile-size=$TILE_SIZE
echo ""

# Direct 2D row-major with regular memory
echo "--- Profiling Direct 2D Row-Major (REGULAR) ---"
ncu --metrics "$NCU_METRICS" $MEMORY_TEST --memory-test=$SIZE --pattern=rowmajor --config=2d-tile --tile-size=$TILE_SIZE --no-pitch
echo ""

# Tiled row-major with pitched memory
echo "--- Profiling Tiled Row-Major (PITCHED) ---"
ncu --metrics "$NCU_METRICS" $MEMORY_TEST --memory-test=$SIZE --pattern=tiled-row --config=2d-tile --tile-size=$TILE_SIZE
echo ""

# Tiled row-major with regular memory
echo "--- Profiling Tiled Row-Major (REGULAR) ---"
ncu --metrics "$NCU_METRICS" $MEMORY_TEST --memory-test=$SIZE --pattern=tiled-row --config=2d-tile --tile-size=$TILE_SIZE --no-pitch
echo ""

echo "=== Comparison Complete ==="