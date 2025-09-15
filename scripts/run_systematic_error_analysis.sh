#!/bin/bash

# Systematic Error Analysis Script
# ================================
# This script runs error analysis for all combinations of:
# - Kernels: tiled, tiled_pairwise, cublas
# - Matrix Types: uniform_positive, wellcond, illcond, zeromean, 2powers
# - Sizes: 256, 512, 1024, 2048

echo "Starting systematic error analysis..."
echo "Total configurations: 3 kernels × 5 matrix types × 4 sizes = 60 tests"
echo "Estimated time: ~30-45 minutes"
echo ""

# Create data directory if it doesn't exist
mkdir -p data

# Define test parameters
#KERNELS=("tiled" "tiled_pairwise" "cublas" "cutlass_splitk_flat" "cutlass_splitk_pairwise")
KERNELS=("cutlass_splitk_flat" "cutlass_splitk_pairwise")
#KERNELS=("cublas")
#MATRIX_TYPES=("uniform_positive" "wellcond" "illcond" "zeromean" "2powers")
MATRIX_TYPES=("uniform_positive")
# "wellcond" "illcond" "zeromean" "2powers")
SIZES=(256 384 512 1024 1280 1536 1792 2048 3072 4096)
#SIZES=(256 384 512 768 1024 1280 1536 1792 2048 3072 4096)

# Counter for progress tracking
total_tests=$((${#KERNELS[@]} * ${#MATRIX_TYPES[@]} * ${#SIZES[@]}))
current_test=0

# Start timestamp
start_time=$(date +%s)

echo "Configuration:"
echo "  Kernels: ${KERNELS[*]}"
echo "  Matrix Types: ${MATRIX_TYPES[*]}"
echo "  Sizes: ${SIZES[*]}"
echo ""

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

            # Brief pause between tests
            sleep 1
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

echo ""
echo "Run 'python scripts/analyze_systematic_results.py' to analyze the results!"
