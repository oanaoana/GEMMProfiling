#!/bin/bash

# Systematic Error Analysis Script
# ================================
# This script runs error analysis for all combinations defined in systematic.config

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
source "$SCRIPT_DIR/systematic.config"

echo "Configuration:"
echo "  Kernels: ${KERNELS[*]}"
echo "  Matrix Types: ${MATRIX_TYPES[*]}"
echo "  Sizes: ${SIZES[*]}"
echo "  Data folder: $DATA_FOLDER"
echo ""

# Create data folder
mkdir -p "$DATA_FOLDER"

# Check if main executable exists
if [ ! -f "./main" ]; then
    echo "Building main executable..."
    make clean
    make COMPUTE_TYPE="$COMPUTE_TYPE" ACCUMULATE_TYPE="$ACCUMULATE_TYPE"
    if [ $? -ne 0 ]; then
        echo "ERROR: Build failed. Exiting."
        exit 1
    fi
fi

echo "Starting systematic error analysis..."
total_tests=$((${#KERNELS[@]} * ${#MATRIX_TYPES[@]} * ${#SIZES[@]}))
echo "Total configurations: ${#KERNELS[@]} kernels × ${#MATRIX_TYPES[@]} matrix types × ${#SIZES[@]} sizes = $total_tests tests"
echo ""

# Count total configurations
total_tests=$((${#KERNELS[@]} * ${#MATRIX_TYPES[@]} * ${#SIZES[@]}))
echo "Total configurations: $total_tests tests"
echo ""

# Counters
generated=0
existed=0
failed=0
start_time=$(date +%s)

# Run all combinations
for kernel in "${KERNELS[@]}"; do
    echo "=== Kernel: $kernel ==="

    for matrix_type in "${MATRIX_TYPES[@]}"; do
        for size in "${SIZES[@]}"; do
            expected_file="$DATA_FOLDER/error_analysis_${kernel}_${matrix_type}_n${size}.csv"

            if [ ! -f "$expected_file" ]; then
                echo "  Generating: $kernel, matrix=$matrix_type, size=${size}x${size}"
                ./main --error-analysis --test="$kernel" --size="$size" --matrix-type="$matrix_type"

                if [ $? -eq 0 ]; then
                    ((generated++))
                else
                    echo "  ERROR: Failed - kernel:$kernel, matrix:$matrix_type, size:$size"
                    ((failed++))
                fi
            else
                ((existed++))
            fi
        done
    done
    echo ""
done

# End timestamp and duration
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "=== Error Analysis Complete ==="
echo "Time: ${minutes}m ${seconds}s"
echo "Generated: $generated"
echo "Already existed: $existed"
echo "Failed: $failed"
echo "Total: $((generated + existed + failed))/$total_tests"
echo ""
echo "Results saved to: $DATA_FOLDER"
echo ""
echo "Next steps:"
echo "  python scripts/plot_beta_ratios.py"

# Only show screen info if we're actually in screen
if [ -n "$STY" ]; then
    echo "Screen session info:"
    echo "  Current session: $STY"
    echo "  Detach: Ctrl+A then D"
    echo "  Reattach later: screen -r gemm_analysis"
fi