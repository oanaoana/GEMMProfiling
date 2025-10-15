#!/bin/bash

# Systematic Error Analysis Script
# ================================
# This script runs error analysis for all combinations defined in systematic_config.sh

echo "=== Systematic Error Analysis (Full Run) ==="
echo ""

# SCREEN SETUP - ADD THIS SECTION
echo "IMPORTANT: Long-running analysis detected!"
echo ""
if [ -z "$STY" ]; then
    echo "⚠️  You're not in a screen session. This analysis may take hours."
    echo ""
    echo "RECOMMENDED: Run in screen session to avoid interruption:"
    echo "  1. Start screen:    screen -S gemm_analysis"
    echo "  2. Run this script: ./scripts/run_systematic_error_analysis.sh"
    echo "  3. Detach safely:   Ctrl+A then D (keeps running in background)"
    echo "  4. Reattach later:  screen -r gemm_analysis"
    echo ""
    echo "SCREEN COMMANDS:"
    echo "  screen -ls                    # List all screen sessions"
    echo "  screen -r gemm_analysis       # Reattach to this session"
    echo "  screen -S gemm_analysis       # Start new session named 'gemm_analysis'"
    echo "  # Inside screen: Ctrl+A then D to detach (keeps running)"
    echo ""
    read -p "Continue without screen? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "To run with screen protection:"
        echo "  screen -S gemm_analysis"
        echo "  ./scripts/run_systematic_error_analysis.sh"
        exit 1
    fi
    echo "⚠️  Running without screen protection..."
else
    echo "✅ Running in screen session: $STY"
fi
echo ""

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
summary_files=$(find data -name "*_n*.csv" | wc -l)
echo "Generated $summary_files summary files in data/ directory"

echo ""
echo "Next steps:"
echo "  python scripts/plot_beta_ratios.py"
echo ""

# Only show screen info if we're actually in screen
if [ -n "$STY" ]; then
    echo "Screen session info:"
    echo "  Current session: $STY"
    echo "  Detach: Ctrl+A then D"
    echo "  Reattach later: screen -r gemm_analysis"
fi