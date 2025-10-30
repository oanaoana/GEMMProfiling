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
    echo "  2. Run this script: ./scripts/run_systematic_error_analysis_mixprec.sh"
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
        echo "  ./scripts/run_systematic_error_analysis_mixprec.sh"
        exit 1
    fi
    echo "⚠️  Running without screen protection..."
else
    echo "✅ Running in screen session: $STY"
fi
echo ""

# Source the configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/mixprec.config"

# Helper function to convert precision names to C++ types
get_cpp_type() {
    case "$1" in
        "FP16") echo "__half" ;;
        "FP32") echo "float" ;;
        "FP64") echo "double" ;;
        *) echo "float" ;;  # default
    esac
}

echo "Running systematic error analysis (Mixed Precision)..."

for precision in "${PRECISION_CONFIGS[@]}"; do
    IFS=':' read -r COMPUTE_NAME ACCUMULATE_NAME <<< "$precision"

    # Convert to C++ types
    COMPUTE_TYPE=$(get_cpp_type "$COMPUTE_NAME")
    ACCUMULATE_TYPE=$(get_cpp_type "$ACCUMULATE_NAME")

    DATA_FOLDER="data/UC_${COMPUTE_NAME}_UA_${ACCUMULATE_NAME}"

    echo ""
    echo "=========================================="
    echo "Configuration: UC=$COMPUTE_TYPE, UA=$ACCUMULATE_TYPE"
    echo "Data folder: $DATA_FOLDER"
    echo "=========================================="

    # Create data folder
    mkdir -p "$DATA_FOLDER"

    # Build with specified precision
    make clean
    make COMPUTE_TYPE="$COMPUTE_TYPE" ACCUMULATE_TYPE="$ACCUMULATE_TYPE"

    # Run analysis for all kernels/matrices/sizes
    for kernel in "${KERNELS[@]}"; do
        for matrix_type in "${MATRIX_TYPES[@]}"; do
            for size in "${SIZES[@]}"; do
                echo "Running: kernel=$kernel, matrix=$matrix_type, size=$size"
                ./main --error-analysis \
                       --test="$kernel" \
                       --matrix-type="$matrix_type" \
                       --size="$size"
            done
        done
    done

    echo "Results for UC=$COMPUTE_TYPE, UA=$ACCUMULATE_TYPE saved to: $DATA_FOLDER"
done

echo ""
echo "All mixed precision configurations completed!"