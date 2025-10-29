#!/bin/bash

# Systematic Performance Analysis Script
# ======================================
# Run performance tests for all kernel/size combinations from systematic_config.sh

echo "=== Systematic Performance Analysis ==="

# Source the systematic configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/systematic_config.sh"

echo "Kernels: ${KERNELS[@]}"
echo "Sizes: ${SIZES[@]}"
echo "Total combinations: $((${#KERNELS[@]} * ${#SIZES[@]}))"
echo ""

# Check if main executable exists
if [ ! -f "./main" ]; then
    echo "Error: ./main executable not found!"
    exit 1
fi

# Run performance tests for each combination
for kernel in "${KERNELS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "Running: $kernel, size=${size}x${size}"
        ./main --performance --test="$kernel" --size="$size"
        echo ""
    done
done

echo "Performance analysis complete!"
echo "Generated files in data/perf_*.csv"