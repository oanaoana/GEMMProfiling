#!/bin/bash

# Example comprehensive test script
# Run with: ./run_tests.sh

MEMORY_TEST="./memory_test"
SIZE=512

echo "=== GPU Memory Coalescing Analysis ==="
echo "Matrix Size: ${SIZE}x${SIZE}"
echo ""

# Test different 1D block sizes
echo "### Testing 1D Layout with Different Block Sizes ###"
for block_size in 64 128 256 512 1024; do
    echo "--- Block Size: $block_size ---"
    echo "Row-Major (Sequential):"
    $MEMORY_TEST --memory-test=$SIZE --pattern=rowmajor --layout=1d --block-size=$block_size
    echo ""
    echo "Column-Major (Strided):"
    $MEMORY_TEST --memory-test=$SIZE --pattern=colmajor --layout=1d --block-size=$block_size
    echo ""
done

# Test different 2D tile sizes
echo "### Testing 2D Layout with Different Tile Sizes ###"
for tile_size in 8 16 32; do
    echo "--- Tile Size: ${tile_size}x${tile_size} ---"
    echo "Row-Major (Sequential):"
    $MEMORY_TEST --memory-test=$SIZE --pattern=rowmajor --layout=2d --tile-size=$tile_size
    echo ""
    echo "Column-Major (Strided):"
    $MEMORY_TEST --memory-test=$SIZE --pattern=colmajor --layout=2d --tile-size=$tile_size
    echo ""
done

echo "=== All Tests Completed ==="