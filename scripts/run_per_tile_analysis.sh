#!/bin/bash

# Per-Tile Analysis Runner Script
# ===============================
# Run per-tile analysis for specified kernels, matrix type, and sample numbers
# Edit the variables below to configure your analysis

# CONFIGURATION - Edit these variables as needed
KERNELS=("tiled" "tiled_pairwise")                    # Kernels to test
SAMPLES=(0) #(0 1 2 3 4)                                   # Sample numbers to generate
MATRIX_TYPE="uniform_positive"                        # Matrix type to test
SIZE=256                                              # Matrix size (NxN)

# Available options for reference:
# KERNELS: tiled, tiled_pairwise, cublas, cutlass_splitk_flat, cutlass_splitk_pairwise
# MATRIX_TYPES: uniform_positive, wellcond, illcond, 2powers, zeromean
# SIZES: 256, 512, 1024, 2048, 4096, etc.

echo "=== Per-Tile Analysis Configuration ==="
echo "Kernels: ${KERNELS[@]}"
echo "Samples: ${SAMPLES[@]}"
echo "Matrix type: $MATRIX_TYPE"
echo "Matrix size: ${SIZE}x${SIZE}"
echo "Total runs: $((${#KERNELS[@]} * ${#SAMPLES[@]}))"
echo ""

# Check if main executable exists
if [ ! -f "./main" ]; then
    echo "Error: ./main executable not found!"
    echo "Please compile first with: make"
    exit 1
fi

# Start timing
start_time=$(date +%s)
total_runs=0
successful_runs=0
failed_runs=0

# Run analysis for each combination
for kernel in "${KERNELS[@]}"; do
    echo "--- Processing kernel: $kernel ---"

    for sample in "${SAMPLES[@]}"; do
        echo "Running: $kernel, sample=$sample, type=$MATRIX_TYPE, size=$SIZE"

        # Run the per-tile analysis
        if ./main --per-tile --test="$kernel" --size="$SIZE" --matrix-type="$MATRIX_TYPE" --sample="$sample"; then
            echo "  ✓ Success"
            ((successful_runs++))
        else
            echo "  ✗ Failed"
            ((failed_runs++))
        fi

        ((total_runs++))
        echo "  Progress: $successful_runs/$total_runs completed"
        echo ""
    done

    echo "Completed kernel: $kernel"
    echo ""
done

# End timing
end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo "=== Per-Tile Analysis Complete ==="
echo "Total runs: $total_runs"
echo "Successful: $successful_runs"
echo "Failed: $failed_runs"
echo "Time taken: ${minutes}m ${seconds}s"
echo ""

# Show generated files
echo "Generated files:"
find data -name "per_tile_*_${MATRIX_TYPE}_*_n${SIZE}_sample*.bin" -o -name "per_tile_*_${MATRIX_TYPE}_*_n${SIZE}_sample*_info.csv" | sort
echo ""

echo "Next steps:"
echo "  python scripts/plot_heatmap.py  # Create heatmap visualizations"