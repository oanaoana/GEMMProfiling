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
source "$SCRIPT_DIR/systematic_config.sh"

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
mkdir -p data

echo "=== Checking Error Analysis Data ==="
missing_error_files=0
total_error_files=0

# Check for missing error analysis files
for kernel in "${KERNELS[@]}"; do
    for size in "${SIZES[@]}"; do
        for matrix_type in "${MATRIX_TYPES[@]}"; do
            expected_file="data/error_analysis_${matrix_type}_n${size}.csv"
            ((total_error_files++))

            if [ ! -f "$expected_file" ]; then
                echo "Missing: $expected_file"
                ((missing_error_files++))
            fi
        done
    done
done

echo "Error files: $((total_error_files - missing_error_files))/$total_error_files exist"

# Generate missing error analysis files
if [ $missing_error_files -gt 0 ]; then
    echo ""
    echo "Generating missing error analysis files..."

    for kernel in "${KERNELS[@]}"; do
        for size in "${SIZES[@]}"; do
            for matrix_type in "${MATRIX_TYPES[@]}"; do
                expected_file="data/error_analysis_${matrix_type}_n${size}.csv"

                if [ ! -f "$expected_file" ]; then
                    echo "Generating: $kernel, size=${size}x${size}, type=$matrix_type"
                    ./main --error-analysis --test="$kernel" --size="$size" --matrix-type="$matrix_type"
                fi
            done
        done
    done

    echo "✓ Error analysis generation complete"
else
    echo "✓ All error analysis files exist"
fi

echo ""
echo "=== Checking Performance Data ==="
missing_perf_files=0
total_perf_files=0

# Check for missing performance files
for kernel in "${KERNELS[@]}"; do
    for size in "${SIZES[@]}"; do
        expected_file="data/perf_${kernel}_${size}_FP32.csv"
        ((total_perf_files++))

        if [ ! -f "$expected_file" ]; then
            echo "Missing: $expected_file"
            ((missing_perf_files++))
        fi
    done
done

echo "Performance files: $((total_perf_files - missing_perf_files))/$total_perf_files exist"

# Generate missing performance files
if [ $missing_perf_files -gt 0 ]; then
    echo ""
    echo "Generating missing performance files..."

    for kernel in "${KERNELS[@]}"; do
        for size in "${SIZES[@]}"; do
            expected_file="data/perf_${kernel}_${size}_FP32.csv"

            if [ ! -f "$expected_file" ]; then
                echo "Generating: $kernel, size=${size}x${size}"
                ./main --performance --test="$kernel" --size="$size"
            fi
        done
    done

    echo "✓ Performance generation complete"
else
    echo "✓ All performance files exist"
fi

echo ""
echo "=== Data Generation Summary ==="
echo "Error analysis files: $total_error_files"
echo "Performance files: $total_perf_files"
echo "Total data files available for Pareto analysis"

echo ""
echo "Next steps:"
echo "  python scripts/plot_pareto.py      # Create Pareto frontier plots"