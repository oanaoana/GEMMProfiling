#!/usr/bin/env python3
"""
Systematic Error Analysis Results
=================================

This script analyzes the results from the systematic error analysis sweep.
It reads all the CSV summary files and provides comprehensive analysis and visualization.

Expected data structure from run_systematic_error_analysis.sh:
- Kernels: tiled, tiled_pairwise, cublas
- Matrix Types: uniform_positive, wellcond, illcond, zeromean, 2powers  
- Sizes: 256, 512, 1024, 2048
- Total: 60 test configurations

Usage:
    python scripts/analyze_systematic_results.py
"""

import csv
import glob
import os
from collections import defaultdict
import statistics

def load_systematic_data(data_dir="data"):
    """Load all error analysis CSV summary files."""
    
    # Find all summary CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*_summary_n*.csv"))
    
    if not csv_files:
        print(f"No summary CSV files found in {data_dir}")
        print("Run './scripts/run_systematic_error_analysis.sh' first!")
        return []
    
    print(f"Found {len(csv_files)} CSV summary files")
    
    # Read all CSV files
    all_data = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                numeric_fields = ['matrix_size', 'num_samples', 'frob_avg', 'frob_std', 
                                'frob_p95', 'frob_max', 'beta_avg', 'beta_std', 
                                'beta_p95', 'beta_max', 'theoretical_beta', 'u32',
                                'beta_over_theoretical', 'beta_over_u32']
                for field in numeric_fields:
                    if field in row and row[field]:
                        try:
                            row[field] = float(row[field])
                        except ValueError:
                            print(f"Warning: Could not convert {field}={row[field]} to float")
                
                all_data.append(row)
    
    print(f"Loaded {len(all_data)} test results")
    return all_data

def validate_systematic_data(data):
    """Check if we have the expected systematic test configuration."""
    
    expected_kernels = {'tiled', 'tiled_pairwise', 'cublas'}
    expected_matrix_types = {'uniform_positive', 'wellcond', 'illcond', 'zeromean', '2powers'}
    expected_sizes = {256, 512, 1024, 2048}
    
    actual_kernels = set(row['kernel_type'] for row in data)
    actual_matrix_types = set(row['matrix_type'] for row in data)
    actual_sizes = set(int(row['matrix_size']) for row in data)
    
    print("\n" + "="*60)
    print("SYSTEMATIC TEST VALIDATION")
    print("="*60)
    
    print(f"Expected kernels: {sorted(expected_kernels)}")
    print(f"Actual kernels:   {sorted(actual_kernels)}")
    missing_kernels = expected_kernels - actual_kernels
    if missing_kernels:
        print(f"MISSING KERNELS: {sorted(missing_kernels)}")
    
    print(f"\nExpected matrix types: {sorted(expected_matrix_types)}")
    print(f"Actual matrix types:   {sorted(actual_matrix_types)}")
    missing_matrix_types = expected_matrix_types - actual_matrix_types
    if missing_matrix_types:
        print(f"MISSING MATRIX TYPES: {sorted(missing_matrix_types)}")
    
    print(f"\nExpected sizes: {sorted(expected_sizes)}")
    print(f"Actual sizes:   {sorted(actual_sizes)}")
    missing_sizes = expected_sizes - actual_sizes
    if missing_sizes:
        print(f"MISSING SIZES: {sorted(missing_sizes)}")
    
    expected_total = len(expected_kernels) * len(expected_matrix_types) * len(expected_sizes)
    print(f"\nExpected total configurations: {expected_total}")
    print(f"Actual total configurations:   {len(data)}")
    
    if len(data) == expected_total and not (missing_kernels or missing_matrix_types or missing_sizes):
        print("✓ COMPLETE: All expected configurations found!")
    else:
        print("⚠ INCOMPLETE: Some configurations are missing")
    
    return len(data) == expected_total

def analyze_by_kernel(data):
    """Analyze results grouped by kernel type."""
    
    print("\n" + "="*60)
    print("ANALYSIS BY KERNEL")
    print("="*60)
    
    kernel_stats = defaultdict(list)
    
    for row in data:
        kernel_stats[row['kernel_type']].append(row['beta_avg'])
    
    print(f"{'Kernel':<15} {'Count':<6} {'Avg Error':<12} {'Std Dev':<12} {'Min Error':<12} {'Max Error':<12}")
    print("-" * 75)
    
    kernel_results = []
    for kernel in sorted(kernel_stats.keys()):
        errors = kernel_stats[kernel]
        avg_error = statistics.mean(errors)
        std_error = statistics.stdev(errors) if len(errors) > 1 else 0.0
        min_error = min(errors)
        max_error = max(errors)
        
        print(f"{kernel:<15} {len(errors):<6} {avg_error:<12.3e} {std_error:<12.3e} {min_error:<12.3e} {max_error:<12.3e}")
        kernel_results.append((kernel, avg_error, std_error, min_error, max_error))
    
    # Rank kernels by average error
    print(f"\nKernel Ranking (by average error):")
    ranked = sorted(kernel_results, key=lambda x: x[1])
    for i, (kernel, avg_error, _, _, _) in enumerate(ranked, 1):
        print(f"  {i}. {kernel}: {avg_error:.3e}")

def analyze_by_matrix_type(data):
    """Analyze results grouped by matrix type."""
    
    print("\n" + "="*60)
    print("ANALYSIS BY MATRIX TYPE")
    print("="*60)
    
    matrix_stats = defaultdict(list)
    
    for row in data:
        matrix_stats[row['matrix_type']].append(row['beta_avg'])
    
    print(f"{'Matrix Type':<15} {'Count':<6} {'Avg Error':<12} {'Std Dev':<12} {'Min Error':<12} {'Max Error':<12}")
    print("-" * 75)
    
    matrix_results = []
    for matrix_type in sorted(matrix_stats.keys()):
        errors = matrix_stats[matrix_type]
        avg_error = statistics.mean(errors)
        std_error = statistics.stdev(errors) if len(errors) > 1 else 0.0
        min_error = min(errors)
        max_error = max(errors)
        
        print(f"{matrix_type:<15} {len(errors):<6} {avg_error:<12.3e} {std_error:<12.3e} {min_error:<12.3e} {max_error:<12.3e}")
        matrix_results.append((matrix_type, avg_error, std_error, min_error, max_error))
    
    # Rank matrix types by average error
    print(f"\nMatrix Type Ranking (by average error):")
    ranked = sorted(matrix_results, key=lambda x: x[1])
    for i, (matrix_type, avg_error, _, _, _) in enumerate(ranked, 1):
        print(f"  {i}. {matrix_type}: {avg_error:.3e}")

def analyze_scaling_with_size(data):
    """Analyze how error scales with matrix size."""
    
    print("\n" + "="*60)
    print("ERROR SCALING WITH MATRIX SIZE")
    print("="*60)
    
    # Group by kernel and matrix type, then look at size scaling
    groups = defaultdict(list)
    
    for row in data:
        key = (row['kernel_type'], row['matrix_type'])
        groups[key].append((int(row['matrix_size']), row['beta_avg']))
    
    print("Analyzing error growth patterns...")
    
    size_analysis = []
    for (kernel, matrix_type), size_error_pairs in groups.items():
        # Sort by size
        size_error_pairs.sort()
        sizes = [pair[0] for pair in size_error_pairs]
        errors = [pair[1] for pair in size_error_pairs]
        
        if len(sizes) >= 3:  # Need at least 3 points for trend analysis
            # Simple growth factor analysis
            growth_factors = []
            for i in range(1, len(errors)):
                if errors[i-1] > 0:
                    growth_factor = errors[i] / errors[i-1]
                    size_ratio = sizes[i] / sizes[i-1]
                    growth_factors.append(growth_factor / size_ratio)  # Normalize by size ratio
            
            avg_growth = statistics.mean(growth_factors) if growth_factors else 1.0
            size_analysis.append((kernel, matrix_type, avg_growth, sizes, errors))
    
    # Show top cases with fastest error growth
    print(f"\nFastest Error Growth (normalized growth factor per size doubling):")
    size_analysis.sort(key=lambda x: x[2], reverse=True)
    
    for i, (kernel, matrix_type, growth, sizes, errors) in enumerate(size_analysis[:10]):
        print(f"  {i+1:2}. {kernel} + {matrix_type}: {growth:.3f}x")
        print(f"      Sizes: {sizes}")
        print(f"      Errors: {[f'{e:.2e}' for e in errors]}")

def find_best_worst_cases(data):
    """Find the best and worst performing configurations."""
    
    print("\n" + "="*60)
    print("BEST AND WORST CASES")
    print("="*60)
    
    # Sort by beta_avg
    sorted_data = sorted(data, key=lambda x: x['beta_avg'])
    
    print("🏆 TOP 5 BEST (Lowest Error):")
    for i, row in enumerate(sorted_data[:5], 1):
        print(f"  {i}. {row['kernel_type']} + {row['matrix_type']} @ n={int(row['matrix_size'])}: {row['beta_avg']:.3e}")
        print(f"     Ratio to theoretical: {row['beta_over_theoretical']:.3f}")
    
    print("\n💥 TOP 5 WORST (Highest Error):")
    for i, row in enumerate(sorted_data[-5:], 1):
        print(f"  {i}. {row['kernel_type']} + {row['matrix_type']} @ n={int(row['matrix_size'])}: {row['beta_avg']:.3e}")
        print(f"     Ratio to theoretical: {row['beta_over_theoretical']:.3f}")

def generate_summary_report(data, filename="data/systematic_analysis_report.txt"):
    """Generate a comprehensive text report."""
    
    with open(filename, 'w') as f:
        f.write("SYSTEMATIC ERROR ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic stats
        f.write(f"Total Configurations Tested: {len(data)}\n")
        f.write(f"Kernels: {sorted(set(row['kernel_type'] for row in data))}\n")
        f.write(f"Matrix Types: {sorted(set(row['matrix_type'] for row in data))}\n")
        f.write(f"Sizes: {sorted(set(int(row['matrix_size']) for row in data))}\n\n")
        
        # Overall statistics
        all_errors = [row['beta_avg'] for row in data]
        f.write(f"Overall Error Statistics:\n")
        f.write(f"  Mean: {statistics.mean(all_errors):.3e}\n")
        f.write(f"  Median: {statistics.median(all_errors):.3e}\n")
        f.write(f"  Std Dev: {statistics.stdev(all_errors):.3e}\n")
        f.write(f"  Min: {min(all_errors):.3e}\n")
        f.write(f"  Max: {max(all_errors):.3e}\n\n")
        
        # Detailed data
        f.write("DETAILED RESULTS:\n")
        f.write("kernel,matrix_type,size,beta_avg,beta_std,beta_over_theoretical\n")
        for row in sorted(data, key=lambda x: (x['kernel_type'], x['matrix_type'], x['matrix_size'])):
            f.write(f"{row['kernel_type']},{row['matrix_type']},{int(row['matrix_size'])},"
                   f"{row['beta_avg']:.6e},{row['beta_std']:.6e},{row['beta_over_theoretical']:.3f}\n")
    
    print(f"\nDetailed report saved to: {filename}")

def main():
    print("Systematic Error Analysis Results")
    print("=" * 50)
    
    # Load data
    data = load_systematic_data()
    if not data:
        return
    
    # Validate completeness
    is_complete = validate_systematic_data(data)
    
    # Run analyses
    analyze_by_kernel(data)
    analyze_by_matrix_type(data)
    analyze_scaling_with_size(data)
    find_best_worst_cases(data)
    
    # Generate summary report
    generate_summary_report(data)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analyzed {len(data)} configurations")
    print("Check 'data/systematic_analysis_report.txt' for detailed results")
    
    if not is_complete:
        print("\n⚠️  WARNING: Not all expected configurations were found.")
        print("   Consider re-running './scripts/run_systematic_error_analysis.sh'")

if __name__ == "__main__":
    main()
