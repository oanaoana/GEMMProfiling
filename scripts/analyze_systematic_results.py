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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

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
                numeric_fields = ['matrix_size', 'num_samples', '|C-C_ref|_avg', '|C-C_ref|_std',
                                '|C-C_ref|_p95', '|C-C_ref|_max', '|C-C_ref|/(|A||B|)_avg', '|C-C_ref|/(|A||B|)_std',
                                '|C-C_ref|/(|A||B|)_p95', '|C-C_ref|/(|A||B|)_max', 'theoretical_beta', 'u32',
                                'E_{AB}/beta', 'E_{AB}/u']
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
        kernel_stats[row['kernel_type']].append(row['|C-C_ref|/(|A||B|)_avg'])

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
        matrix_stats[row['matrix_type']].append(row['|C-C_ref|/(|A||B|)_avg'])

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
        groups[key].append((int(row['matrix_size']), row['|C-C_ref|/(|A||B|)_avg']))

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

    # Sort by |C-C_ref|/(|A||B|)_avg
    sorted_data = sorted(data, key=lambda x: x['|C-C_ref|/(|A||B|)_avg'])

    print("🏆 TOP 5 BEST (Lowest Error):")
    for i, row in enumerate(sorted_data[:5], 1):
        print(f"  {i}. {row['kernel_type']} + {row['matrix_type']} @ n={int(row['matrix_size'])}: {row['|C-C_ref|/(|A||B|)_avg']:.3e}")
        print(f"     E_AB/beta: {row['E_{AB}/beta']:.3f}")

    print("\n💥 TOP 5 WORST (Highest Error):")
    for i, row in enumerate(sorted_data[-5:], 1):
        print(f"  {i}. {row['kernel_type']} + {row['matrix_type']} @ n={int(row['matrix_size'])}: {row['|C-C_ref|/(|A||B|)_avg']:.3e}")
        print(f"     E_AB/beta: {row['E_{AB}/beta']:.3f}")

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
        all_errors = [row['|C-C_ref|/(|A||B|)_avg'] for row in data]
        f.write(f"Overall Error Statistics:\n")
        f.write(f"  Mean: {statistics.mean(all_errors):.3e}\n")
        f.write(f"  Median: {statistics.median(all_errors):.3e}\n")
        f.write(f"  Std Dev: {statistics.stdev(all_errors):.3e}\n")
        f.write(f"  Min: {min(all_errors):.3e}\n")
        f.write(f"  Max: {max(all_errors):.3e}\n\n")

        # Detailed data
        f.write("DETAILED RESULTS:\n")
        f.write("kernel,matrix_type,size,|C-C_ref|/(|A||B|)_avg,|C-C_ref|/(|A||B|)_std,E_AB/beta\n")
        for row in sorted(data, key=lambda x: (x['kernel_type'], x['matrix_type'], x['matrix_size'])):
            f.write(f"{row['kernel_type']},{row['matrix_type']},{int(row['matrix_size'])},"
                   f"{row['|C-C_ref|/(|A||B|)_avg']:.6e},{row['|C-C_ref|/(|A||B|)_std']:.6e},{row['E_{AB}/beta']:.3f}\n")

    print(f"\nDetailed report saved to: {filename}")

def create_plots(data, output_dir="plots"):
    """Create comprehensive visualization plots."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert data to pandas DataFrame for easier plotting
    df = pd.DataFrame(data)

    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    print(f"\nGenerating plots in {output_dir}/...")

    # 1. Kernel Comparison Bar Plot
    create_kernel_comparison_plot(df, output_dir)

    # 2. Matrix Type Comparison
    create_matrix_type_comparison_plot(df, output_dir)

    # 3. Error Scaling with Size
    create_error_scaling_plots(df, output_dir)

    # 4. Heatmap of All Configurations
    create_configuration_heatmap(df, output_dir)

    # 5. Error Distribution Histograms
    create_error_distribution_plots(df, output_dir)

    # 6. Box Plots for Statistical Analysis
    create_statistical_comparison_plots(df, output_dir)

    # 7. Ratio to Theoretical Bounds
    create_theoretical_ratio_plots(df, output_dir)

    print(f"✓ All plots saved to {output_dir}/")

def create_kernel_comparison_plot(df, output_dir):
    """Create bar plot comparing kernel performance."""

    plt.figure(figsize=(12, 8))

    # Calculate statistics by kernel
    kernel_stats = df.groupby('kernel_type')['|C-C_ref|/(|A||B|)_avg'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    kernel_stats = kernel_stats.sort_values('mean')

    # Create bar plot with error bars
    bars = plt.bar(kernel_stats['kernel_type'], kernel_stats['mean'],
                   yerr=kernel_stats['std'], capsize=5, alpha=0.7,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    plt.yscale('log')
    plt.xlabel('Kernel Type', fontsize=12)
    plt.ylabel('Average Normalized Error', fontsize=12)
    plt.title('Kernel Performance Comparison\n(Average ± Std Dev across all configurations)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val, count in zip(bars, kernel_stats['mean'], kernel_stats['count']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height*1.1,
                f'{mean_val:.2e}\n({count} tests)',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kernel_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_matrix_type_comparison_plot(df, output_dir):
    """Create comparison plot for matrix types."""

    plt.figure(figsize=(14, 8))

    # Calculate statistics by matrix type
    matrix_stats = df.groupby('matrix_type')['|C-C_ref|/(|A||B|)_avg'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    matrix_stats = matrix_stats.sort_values('mean')

    # Create bar plot
    bars = plt.bar(matrix_stats['matrix_type'], matrix_stats['mean'],
                   yerr=matrix_stats['std'], capsize=5, alpha=0.7)

    plt.yscale('log')
    plt.xlabel('Matrix Type', fontsize=12)
    plt.ylabel('Average Normalized Error', fontsize=12)
    plt.title('Matrix Type Performance Comparison\n(Average ± Std Dev across all kernels and sizes)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, mean_val in zip(bars, matrix_stats['mean']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height*1.1,
                f'{mean_val:.2e}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matrix_type_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_error_scaling_plots(df, output_dir):
    """Create plots showing how error scales with matrix size."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Error Scaling with Matrix Size', fontsize=16)

    kernels = df['kernel_type'].unique()
    matrix_types = df['matrix_type'].unique()

    # Plot 1: All kernels, average across matrix types
    ax = axes[0, 0]
    for kernel in kernels:
        kernel_data = df[df['kernel_type'] == kernel]
        size_avg = kernel_data.groupby('matrix_size')['|C-C_ref|/(|A||B|)_avg'].mean()
        ax.loglog(size_avg.index, size_avg.values, 'o-', label=kernel, linewidth=2, markersize=6)

    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Average Normalized Error')
    ax.set_title('Average Error vs Size (by Kernel)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: All matrix types, average across kernels
    ax = axes[0, 1]
    colors = plt.cm.Set1(np.linspace(0, 1, len(matrix_types)))
    for i, matrix_type in enumerate(matrix_types):
        matrix_data = df[df['matrix_type'] == matrix_type]
        size_avg = matrix_data.groupby('matrix_size')['|C-C_ref|/(|A||B|)_avg'].mean()
        ax.loglog(size_avg.index, size_avg.values, 'o-', label=matrix_type,
                 color=colors[i], linewidth=2, markersize=6)

    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Average Normalized Error')
    ax.set_title('Average Error vs Size (by Matrix Type)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 3: Individual kernel trends
    ax = axes[1, 0]
    for kernel in kernels:
        kernel_data = df[df['kernel_type'] == kernel]
        sizes = kernel_data['matrix_size'].values
        errors = kernel_data['|C-C_ref|/(|A||B|)_avg'].values
        ax.loglog(sizes, errors, 'o', alpha=0.6, label=kernel)

    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Normalized Error')
    ax.set_title('All Individual Data Points')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Error growth rates
    ax = axes[1, 1]
    growth_data = []
    for kernel in kernels:
        for matrix_type in matrix_types:
            subset = df[(df['kernel_type'] == kernel) & (df['matrix_type'] == matrix_type)]
            if len(subset) >= 2:
                subset_sorted = subset.sort_values('matrix_size')
                sizes = subset_sorted['matrix_size'].values
                errors = subset_sorted['|C-C_ref|/(|A||B|)_avg'].values

                # Calculate growth rate (slope in log-log plot)
                if len(sizes) >= 2 and all(e > 0 for e in errors) and all(s > 0 for s in sizes):
                    log_sizes = np.log(sizes)
                    log_errors = np.log(errors)
                    slope = np.polyfit(log_sizes, log_errors, 1)[0]
                    growth_data.append({'kernel': kernel, 'matrix_type': matrix_type, 'slope': slope})

    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        kernel_slopes = growth_df.groupby('kernel')['slope'].mean()
        bars = ax.bar(kernel_slopes.index, kernel_slopes.values, alpha=0.7)
        ax.set_xlabel('Kernel Type')
        ax.set_ylabel('Average Error Growth Rate (log-log slope)')
        ax.set_title('Error Scaling Rate by Kernel')
        ax.grid(True, alpha=0.3)

        for bar, slope in zip(bars, kernel_slopes.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{slope:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_scaling_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_configuration_heatmap(df, output_dir):
    """Create heatmap showing error for all kernel-matrix type combinations."""

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Configuration Heatmaps: Error by Kernel × Matrix Type', fontsize=16)

    sizes = sorted(df['matrix_size'].unique())

    for i, size in enumerate(sizes):
        ax = axes[i//2, i%2]

        # Filter data for this size
        size_data = df[df['matrix_size'] == size]

        # Create pivot table for heatmap
        heatmap_data = size_data.pivot_table(values='|C-C_ref|/(|A||B|)_avg',
                                            index='kernel_type',
                                            columns='matrix_type',
                                            aggfunc='mean')

        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2e', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Normalized Error'})
        ax.set_title(f'Matrix Size: {size}×{size}')
        ax.set_xlabel('Matrix Type')
        ax.set_ylabel('Kernel Type')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'configuration_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_error_distribution_plots(df, output_dir):
    """Create histograms and box plots showing error distributions."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Error Distribution Analysis', fontsize=16)

    # Plot 1: Overall error distribution
    ax = axes[0, 0]
    ax.hist(np.log10(df['|C-C_ref|/(|A||B|)_avg']), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('log₁₀(Normalized Error)')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Error Distribution')
    ax.grid(True, alpha=0.3)

    # Plot 2: Error distribution by kernel
    ax = axes[0, 1]
    kernels = df['kernel_type'].unique()
    for kernel in kernels:
        kernel_errors = df[df['kernel_type'] == kernel]['|C-C_ref|/(|A||B|)_avg']
        ax.hist(np.log10(kernel_errors), bins=20, alpha=0.6, label=kernel)
    ax.set_xlabel('log₁₀(Normalized Error)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution by Kernel')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Box plot by kernel
    ax = axes[1, 0]
    kernel_data = [df[df['kernel_type'] == kernel]['|C-C_ref|/(|A||B|)_avg'].values for kernel in kernels]
    box_plot = ax.boxplot(kernel_data, labels=kernels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax.set_yscale('log')
    ax.set_ylabel('Normalized Error')
    ax.set_title('Error Distribution by Kernel (Box Plot)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Box plot by matrix type
    ax = axes[1, 1]
    matrix_types = df['matrix_type'].unique()
    matrix_data = [df[df['matrix_type'] == mt]['|C-C_ref|/(|A||B|)_avg'].values for mt in matrix_types]
    ax.boxplot(matrix_data, labels=matrix_types)
    ax.set_yscale('log')
    ax.set_ylabel('Normalized Error')
    ax.set_title('Error Distribution by Matrix Type')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_comparison_plots(df, output_dir):
    """Create detailed statistical comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Performance Analysis', fontsize=16)

    # Plot 1: Violin plot by kernel
    ax = axes[0, 0]
    sns.violinplot(data=df, x='kernel_type', y='|C-C_ref|/(|A||B|)_avg', ax=ax)
    ax.set_yscale('log')
    ax.set_ylabel('Normalized Error')
    ax.set_title('Error Distribution Shape by Kernel')
    ax.grid(True, alpha=0.3)

    # Plot 2: Strip plot showing all points
    ax = axes[0, 1]
    sns.stripplot(data=df, x='kernel_type', y='|C-C_ref|/(|A||B|)_avg', ax=ax, size=8, alpha=0.7)
    ax.set_yscale('log')
    ax.set_ylabel('Normalized Error')
    ax.set_title('All Data Points by Kernel')
    ax.grid(True, alpha=0.3)

    # Plot 3: Mean ± std by size
    ax = axes[1, 0]
    sizes = sorted(df['matrix_size'].unique())
    for kernel in df['kernel_type'].unique():
        kernel_data = df[df['kernel_type'] == kernel]
        means = []
        stds = []
        for size in sizes:
            size_data = kernel_data[kernel_data['matrix_size'] == size]['|C-C_ref|/(|A||B|)_avg']
            means.append(size_data.mean())
            stds.append(size_data.std())

        ax.errorbar(sizes, means, yerr=stds, marker='o', label=kernel,
                   capsize=5, linewidth=2, markersize=6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Mean Error ± Std Dev')
    ax.set_title('Error vs Size with Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Correlation matrix
    ax = axes[1, 1]
    # Create numeric matrix for correlation
    numeric_df = df.copy()
    numeric_df['kernel_num'] = pd.Categorical(df['kernel_type']).codes
    numeric_df['matrix_num'] = pd.Categorical(df['matrix_type']).codes

    corr_data = numeric_df[['matrix_size', 'kernel_num', 'matrix_num', '|C-C_ref|/(|A||B|)_avg', 'E_{AB}/beta']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_theoretical_ratio_plots(df, output_dir):
    """Create plots comparing actual errors to theoretical bounds."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparison to Theoretical Error Bounds', fontsize=16)

    # Plot 1: Ratio to theoretical bound by kernel
    ax = axes[0, 0]
    kernel_ratios = df.groupby('kernel_type')['E_{AB}/beta'].agg(['mean', 'std']).reset_index()
    bars = ax.bar(kernel_ratios['kernel_type'], kernel_ratios['mean'],
                  yerr=kernel_ratios['std'], capsize=5, alpha=0.7)
    ax.set_ylabel('Error / Theoretical Bound')
    ax.set_title('Performance vs Theoretical Bounds')
    ax.grid(True, alpha=0.3)

    # Add horizontal line at ratio = 1
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Theoretical Bound')
    ax.legend()

    for bar, mean_val in zip(bars, kernel_ratios['mean']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + kernel_ratios['std'].max()*0.1,
               f'{mean_val:.1f}×', ha='center', va='bottom', fontsize=11)

    # Plot 2: Ratio distribution
    ax = axes[0, 1]
    for kernel in df['kernel_type'].unique():
        kernel_ratios = df[df['kernel_type'] == kernel]['E_{AB}/beta']
        ax.hist(kernel_ratios, bins=15, alpha=0.6, label=kernel)
    ax.set_xlabel('Error / Theoretical Bound')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Theoretical Bound Ratios')
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Ratio vs matrix size
    ax = axes[1, 0]
    for kernel in df['kernel_type'].unique():
        kernel_data = df[df['kernel_type'] == kernel]
        sizes = kernel_data['matrix_size']
        ratios = kernel_data['E_{AB}/beta']
        ax.semilogx(sizes, ratios, 'o', alpha=0.7, label=kernel)

    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Error / Theoretical Bound')
    ax.set_title('Theoretical Bound Ratio vs Size')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Best/worst cases
    ax = axes[1, 1]
    best_cases = df.nsmallest(5, 'E_{AB}/beta')
    worst_cases = df.nlargest(5, 'E_{AB}/beta')

    # Create labels for best/worst cases
    best_labels = [f"{row['kernel_type']}\n{row['matrix_type']}\nn={int(row['matrix_size'])}"
                   for _, row in best_cases.iterrows()]
    worst_labels = [f"{row['kernel_type']}\n{row['matrix_type']}\nn={int(row['matrix_size'])}"
                    for _, row in worst_cases.iterrows()]

    x_pos = np.arange(5)
    bars1 = ax.bar(x_pos - 0.2, best_cases['E_{AB}/beta'], 0.4,
                   label='Best Cases', alpha=0.7, color='green')
    bars2 = ax.bar(x_pos + 0.2, worst_cases['E_{AB}/beta'], 0.4,
                   label='Worst Cases', alpha=0.7, color='red')

    ax.set_ylabel('Error / Theoretical Bound')
    ax.set_title('Best vs Worst Cases')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Case {i+1}' for i in range(5)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'theoretical_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

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

    # Create comprehensive plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*60)

    try:
        create_plots(data)
        print("✓ All plots generated successfully!")
    except ImportError as e:
        print(f"⚠️  Plotting dependencies missing: {e}")
        print("Install with: pip install matplotlib seaborn pandas numpy")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")

    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analyzed {len(data)} configurations")
    print("Check 'data/systematic_analysis_report.txt' for detailed results")
    print("Check 'plots/' directory for visualization plots")

    if not is_complete:
        print("\n⚠️  WARNING: Not all expected configurations were found.")
        print("   Consider re-running './scripts/run_systematic_error_analysis.sh'")

if __name__ == "__main__":
    main()
