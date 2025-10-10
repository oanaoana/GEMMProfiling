#!/bin/bash

# Systematic Error Analysis Script (Resume Mode)
# ==============================================
# This script runs error analysis for missing combinations only.
# It checks existing CSV files and skips completed tests.
# Configuration is loaded from systematic_config.sh
#
# Usage with screen (recommended for remote machines):
#   screen -S gemm_analysis
#   ./scripts/run_systematic_error_analysis_resume.sh
#   # Press Ctrl+A then D to detach
#   # screen -r gemm_analysis to reattach later

# Source the configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/systematic_config.sh"

echo "Starting systematic error analysis (resume mode)..."
echo "This script will check existing files and only run missing tests."
echo ""

# Screen usage reminder
if [ -z "$STY" ]; then
    echo "TIP: You're not in a screen session. To run safely on remote machine:"
    echo "  screen -S gemm_analysis"
    echo "  ./scripts/run_systematic_error_analysis_resume.sh"
    echo "  # Press Ctrl+A then D to detach"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create data directory if it doesn't exist
mkdir -p data

# Function to check if output file exists
check_file_exists() {
    local kernel=$1
    local matrix_type=$2
    local size=$3
    local filename="data/error_analysis_${kernel}_${matrix_type}_summary_n${size}.csv"
    [ -f "$filename" ]
}

# Count total and missing tests
total_tests=$((${#KERNELS[@]} * ${#MATRIX_TYPES[@]} * ${#SIZES[@]}))
missing_tests=0
completed_tests=0

echo "Scanning existing files..."

# Create list of missing tests
missing_configs=()

for kernel in "${KERNELS[@]}"; do
    for matrix_type in "${MATRIX_TYPES[@]}"; do
        for size in "${SIZES[@]}"; do
            if check_file_exists "$kernel" "$matrix_type" "$size"; then
                completed_tests=$((completed_tests + 1))
            else
                missing_tests=$((missing_tests + 1))
                missing_configs+=("$kernel|$matrix_type|$size")
            fi
        done
    done
done

echo "Status:"
echo "  Total configurations: $total_tests"
echo "  Completed tests: $completed_tests"
echo "  Missing tests: $missing_tests"
echo ""

if [ $missing_tests -eq 0 ]; then
    echo "All tests are complete! No missing files found."
    exit 0
fi

echo "Missing configurations:"
for config in "${missing_configs[@]}"; do
    IFS='|' read -r kernel matrix_type size <<< "$config"
    echo "  $kernel - $matrix_type - size $size"
done
echo ""

read -p "Run $missing_tests missing tests? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Start timestamp
start_time=$(date +%s)
current_test=0

echo "Starting missing tests..."
echo ""

# Run missing tests only
for config in "${missing_configs[@]}"; do
    IFS='|' read -r kernel matrix_type size <<< "$config"
    current_test=$((current_test + 1))

    echo "[$current_test/$missing_tests] Running: $kernel - $matrix_type - size $size"

    # Run the error analysis
    ./main --error-analysis --test="$kernel" --size="$size" --matrix-type="$matrix_type"

    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "  ERROR: Failed test - kernel:$kernel, matrix_type:$matrix_type, size:$size"
        echo "  Continuing with next test..."
    else
        echo "  ✓ Completed successfully"
    fi
done

# End timestamp and duration calculation
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "=== Resume Analysis Complete ==="
echo "Missing tests completed: $missing_tests"
echo "Time taken: ${minutes}m ${seconds}s"
echo ""

# Count total files now
summary_files=$(find data -name "*_summary_n*.csv" | wc -l)
echo "Total summary files in data/: $summary_files"

echo ""
echo "You can now run: python scripts/plot_beta_ratios.py"
echo ""
echo "Screen session commands:"
echo "  Detach: Ctrl+A then D"
echo "  Reattach: screen -r gemm_analysis"
echo "  List sessions: screen -ls"
