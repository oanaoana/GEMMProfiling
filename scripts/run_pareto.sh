#!/bin/bash

# Pareto Analysis Script
# ======================
# Generate error and performance data if missing, then create Pareto plots
# Reads configuration from systematic_config.sh

echo "=== Pareto Analysis Setup ==="

# SCREEN SETUP
echo "IMPORTANT: Long-running analysis detected!"
echo ""
if [ -z "$STY" ]; then
    echo "⚠️  You're not in a screen session. This analysis may take hours."
    echo ""
    echo "RECOMMENDED: Run in screen session to avoid interruption:"
    echo "  1. Start screen:    screen -S pareto_analysis"
    echo "  2. Run this script: ./scripts/run_pareto.sh"
    echo "  3. Detach safely:   Ctrl+A then D (keeps running in background)"
    echo "  4. Reattach later:  screen -r pareto_analysis"
    echo ""
    echo "SCREEN COMMANDS:"
    echo "  screen -ls                    # List all screen sessions"
    echo "  screen -r pareto_analysis     # Reattach to this session"
    echo "  screen -S pareto_analysis     # Start new session named 'pareto_analysis'"
    echo "  # Inside screen: Ctrl+A then D to detach (keeps running)"
    echo ""
    read -p "Continue without screen? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "To run with screen protection:"
        echo "  screen -S pareto_analysis"
        echo "  ./scripts/run_pareto.sh"
        exit 1
    fi
    echo "⚠️  Running without screen protection..."
else
    echo "✅ Running in screen session: $STY"
fi
echo ""

# Source the systematic configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/systematic.config"

echo "Kernels: ${KERNELS[@]}"
echo "Sizes: ${SIZES[@]}"
echo "Matrix types: ${MATRIX_TYPES[@]}"
echo ""

# Check if main executable exists
if [ ! -f "./main" ]; then
    echo "Error: ./main executable not found!"
    echo "Please compile first with: make"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p "$DATA_FOLDER"

echo "=== Generating Error Analysis Data ==="
error_generated=0
error_existed=0

for kernel in "${KERNELS[@]}"; do
    for size in "${SIZES[@]}"; do
        for matrix_type in "${MATRIX_TYPES[@]}"; do
            expected_file="$DATA_FOLDER/error_analysis_${kernel}_${matrix_type}_n${size}.csv"

            if [ ! -f "$expected_file" ]; then
                echo "Generating: $kernel, size=${size}x${size}, type=$matrix_type"
                ./main --error-analysis --test="$kernel" --size="$size" --matrix-type="$matrix_type"
                ((error_generated++))
            else
                ((error_existed++))
            fi
        done
    done
done

echo "✓ Error analysis complete: $error_generated generated, $error_existed already existed"

echo ""
echo "=== Generating Performance Data ==="
perf_generated=0
perf_existed=0

for kernel in "${KERNELS[@]}"; do
    for size in "${SIZES[@]}"; do
        expected_file="$DATA_FOLDER/perf_${kernel}_${size}_FP32.csv"

        if [ ! -f "$expected_file" ]; then
            echo "Generating: $kernel, size=${size}x${size}"
            ./main --performance --test="$kernel" --size="$size"
            ((perf_generated++))
        else
            ((perf_existed++))
        fi
    done
done

echo "✓ Performance analysis complete: $perf_generated generated, $perf_existed already existed"

echo ""
echo "=== Data Generation Summary ==="
echo "Error analysis: $error_generated generated, $error_existed existed"
echo "Performance: $perf_generated generated, $perf_existed existed"
echo ""
echo "Next steps:"
echo "  python scripts/plot_pareto.py      # Create Pareto frontier plots"