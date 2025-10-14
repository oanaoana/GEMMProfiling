#!/bin/bash

# Systematic Error Analysis Script
# ================================
# This script runs error analysis for all combinations defined in systematic_config.sh

# Source the configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/systematic_config.sh"

echo "Starting systematic error analysis..."
total_tests=$((${#KERNELS[@]} * ${#MATRIX_TYPES[@]} * ${#SIZES[@]}))
echo "Total configurations: ${#KERNELS[@]} kernels × ${#MATRIX_TYPES[@]} matrix types × ${#SIZES[@]} sizes = $total_tests tests"
echo "Estimated time: ~30-45 minutes"
echo ""

# Create data directory if it doesn't exist
mkdir -p data

echo "Configuration:"
echo "  Kernels: ${KERNELS[*]}"
echo "  Matrix Types: ${MATRIX_TYPES[*]}"
echo "  Sizes: ${SIZES[*]}"
echo ""

# Counter for progress tracking
current_test=0

# Start timestamp
start_time=$(date +%s)

# Run all combinations
for kernel in "${KERNELS[@]}"; do
    echo "=== Testing kernel: $kernel ==="

    for matrix_type in "${MATRIX_TYPES[@]}"; do
        echo "  Matrix type: $matrix_type"

        for size in "${SIZES[@]}"; do
            current_test=$((current_test + 1))

            echo "    [$current_test/$total_tests] Size: $size"

            # Run the error analysis
            ./main --error-analysis --test="$kernel" --size="$size" --matrix-type="$matrix_type"

            # Check if the command succeeded
            if [ $? -ne 0 ]; then
                echo "    ERROR: Failed test - kernel:$kernel, matrix_type:$matrix_type, size:$size"
                echo "    Continuing with next test..."
            fi
        done
        echo ""
    done
    echo ""
done

# End timestamp and duration calculation
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "=== Error Analysis Complete ==="
echo "Total time: ${minutes}m ${seconds}s"
echo ""

# Count generated files
summary_files=$(find data -name "*_summary_n*.csv" | wc -l)
echo "Generated $summary_files summary files in data/ directory"

# List some example files
echo ""
echo "Example output files:"
find data -name "*_summary_n*.csv" | head -5
