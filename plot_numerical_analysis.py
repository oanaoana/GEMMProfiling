#!/usr/bin/env python3
"""
Visualization script for GEMM tiling numerical analysis results.
This script creates plots showing error distributions, tile conditioning, and error accumulation patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse
from matplotlib.colors import LogNorm
import time
import os

def timeit(func):
    """Decorator to measure execution time of functions"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@timeit
def load_analysis_data(filename):
    """Load numerical analysis data from file - optimized for performance."""
    print(f"Loading data from {filename}")

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return np.array([])

    try:
        # Use numpy's fast loader instead of line-by-line processing
        # Skip comment lines starting with #
        data = np.loadtxt(filename, comments='#')

        # Convert first two columns to integers if we have data
        if len(data) > 0 and data.shape[1] >= 5:
            data[:, 0] = data[:, 0].astype(int)
            data[:, 1] = data[:, 1].astype(int)

        print(f"Loaded {len(data)} data points")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return np.array([])

@timeit
def plot_error_heatmap(data, matrix_size, save_path, subsample=False):
    """Create heatmap of errors across the matrix - optimized for performance."""
    print(f"Creating error heatmap for {matrix_size}x{matrix_size} matrix")

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Reshape data for heatmaps - use vectorized operations for speed
    abs_errors = np.zeros((matrix_size, matrix_size))
    rel_errors = np.zeros((matrix_size, matrix_size))
    acc_errors = np.zeros((matrix_size, matrix_size))

    # Fast vectorized approach when we have enough data
    if len(data) > 0 and data.shape[1] >= 5:
        i = data[:, 0].astype(int)
        j = data[:, 1].astype(int)

        # Filter valid indices
        valid = (i >= 0) & (i < matrix_size) & (j >= 0) & (j < matrix_size)
        i_valid, j_valid = i[valid], j[valid]

        if len(i_valid) > 0:
            abs_errors[i_valid, j_valid] = data[valid, 2]
            rel_errors[i_valid, j_valid] = data[valid, 3]
            acc_errors[i_valid, j_valid] = data[valid, 4]
    else:
        # Fall back to original code if data format is unexpected
        for row in data:
            if len(row) >= 5:
                i, j = int(row[0]), int(row[1])
                if 0 <= i < matrix_size and 0 <= j < matrix_size:
                    abs_errors[i, j] = row[2]
                    rel_errors[i, j] = row[3]
                    acc_errors[i, j] = row[4]

    # For large matrices, subsample for display if requested
    if subsample and matrix_size > 1000:
        subsample_factor = 4
        abs_errors_display = abs_errors[::subsample_factor, ::subsample_factor]
        rel_errors_display = rel_errors[::subsample_factor, ::subsample_factor]
        acc_errors_display = acc_errors[::subsample_factor, ::subsample_factor]
        print(f"Subsampling large matrix from {matrix_size}x{matrix_size} to {len(abs_errors_display)}x{len(abs_errors_display)}")
    else:
        abs_errors_display = abs_errors
        rel_errors_display = rel_errors
        acc_errors_display = acc_errors

    # Add a small epsilon to handle zero values with LogNorm
    epsilon = np.finfo(float).eps

    try:
        # Absolute errors
        im1 = axes[0].imshow(np.maximum(abs_errors_display, epsilon), cmap='viridis', norm=LogNorm())
        axes[0].set_title('Absolute Errors')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        plt.colorbar(im1, ax=axes[0])

        # Relative errors
        im2 = axes[1].imshow(np.maximum(rel_errors_display, epsilon), cmap='plasma', norm=LogNorm())
        axes[1].set_title('Relative Errors')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Row')
        plt.colorbar(im2, ax=axes[1])

        # Accumulated errors
        im3 = axes[2].imshow(np.maximum(acc_errors_display, epsilon), cmap='inferno', norm=LogNorm())
        axes[2].set_title('Accumulated Errors')
        axes[2].set_xlabel('Column')
        axes[2].set_ylabel('Row')
        plt.colorbar(im3, ax=axes[2])
    except Exception as e:
        print(f"Warning: Error with log scale: {e}, using linear scale")

        # Fall back to linear scale if LogNorm fails
        im1 = axes[0].imshow(abs_errors_display, cmap='viridis')
        axes[0].set_title('Absolute Errors')
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(rel_errors_display, cmap='plasma')
        axes[1].set_title('Relative Errors')
        plt.colorbar(im2, ax=axes[1])

        im3 = axes[2].imshow(acc_errors_display, cmap='inferno')
        axes[2].set_title('Accumulated Errors')
        plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Plot saved to {save_path}")

@timeit
def plot_tile_boundary_effects(data, matrix_size, tile_size, save_path):
    """Analyze error patterns at tile boundaries - optimized for performance."""
    print(f"Creating tile boundary analysis plots")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Create tile boundary maps - vectorized approach
    tile_boundary_x = np.zeros(matrix_size)
    tile_boundary_y = np.zeros(matrix_size)

    # Mark tile boundaries
    for i in range(tile_size, matrix_size, tile_size):
        if i > 0:  # Skip first boundary
            tile_boundary_x[i-1:i+1] = 1
            tile_boundary_y[i-1:i+1] = 1

    # Extract errors by position type - vectorized approach
    if len(data) > 0:
        i = data[:, 0].astype(int)
        j = data[:, 1].astype(int)
        rel_error = data[:, 3]

        # Filter valid indices
        valid = (i >= 0) & (i < matrix_size) & (j >= 0) & (j < matrix_size)
        i_valid = i[valid]
        j_valid = j[valid]
        rel_error_valid = rel_error[valid]

        # Create mask for boundary points
        is_boundary = (np.take(tile_boundary_x, i_valid) > 0) | (np.take(tile_boundary_y, j_valid) > 0)

        boundary_errors = rel_error_valid[is_boundary]
        internal_errors = rel_error_valid[~is_boundary]
    else:
        boundary_errors = []
        internal_errors = []

    # Plot distributions
    if len(boundary_errors) > 0 and len(internal_errors) > 0:
        axes[0, 0].hist(boundary_errors, bins=50, alpha=0.7, label='Tile Boundaries', density=True)
        axes[0, 0].hist(internal_errors, bins=50, alpha=0.7, label='Tile Interiors', density=True)
        axes[0, 0].set_xlabel('Relative Error')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Error Distribution: Boundaries vs Interiors')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
    else:
        axes[0, 0].text(0.5, 0.5, "Insufficient data", ha='center', va='center')

    # Plot error vs distance from tile boundary - optimized
    if len(data) > 0:
        # Calculate distances to nearest tile boundary
        i_mod = i_valid % tile_size
        j_mod = j_valid % tile_size

        dist_x = np.minimum(i_mod, tile_size - i_mod)
        dist_y = np.minimum(j_mod, tile_size - j_mod)
        distances = np.minimum(dist_x, dist_y)

        # Bin by distance and plot mean error
        max_dist = tile_size // 2
        dist_bins = list(range(max_dist + 1))
        mean_errors = []
        std_errors = []

        for d in dist_bins:
            errors_at_dist = rel_error_valid[distances == d]
            if len(errors_at_dist) > 0:
                mean_errors.append(np.mean(errors_at_dist))
                std_errors.append(np.std(errors_at_dist))
            else:
                mean_errors.append(0)
                std_errors.append(0)

        axes[0, 1].errorbar(dist_bins, mean_errors, yerr=std_errors, marker='o')
        axes[0, 1].set_xlabel('Distance from Tile Boundary')
        axes[0, 1].set_ylabel('Mean Relative Error')
        axes[0, 1].set_title('Error vs Distance from Tile Boundary')
    else:
        axes[0, 1].text(0.5, 0.5, "Insufficient data", ha='center', va='center')

    # Create tile grid visualization - faster approach
    if matrix_size % tile_size == 0:  # Only if matrix size is divisible by tile size
        tile_avg_errors = np.zeros((matrix_size // tile_size, matrix_size // tile_size))

        if len(data) > 0:
            # Calculate which tile each point belongs to
            tile_i = i_valid // tile_size
            tile_j = j_valid // tile_size

            # Filter valid tile indices
            valid_tiles = (tile_i >= 0) & (tile_i < matrix_size // tile_size) & \
                        (tile_j >= 0) & (tile_j < matrix_size // tile_size)

            # For each tile, calculate average error
            for ti in range(matrix_size // tile_size):
                for tj in range(matrix_size // tile_size):
                    mask = (tile_i == ti) & (tile_j == tj)
                    if np.any(mask):
                        tile_avg_errors[ti, tj] = np.mean(rel_error_valid[mask])

        im = axes[1, 0].imshow(tile_avg_errors, cmap='viridis')
        axes[1, 0].set_title('Average Error per Tile')
        axes[1, 0].set_xlabel('Tile Column')
        axes[1, 0].set_ylabel('Tile Row')
        plt.colorbar(im, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, "Matrix size not divisible by tile size", ha='center', va='center')

    # Error accumulation pattern
    if len(data) > 0:
        # Find near-diagonal points
        diag_mask = np.abs(i_valid - j_valid) <= matrix_size // 10
        diag_errors = data[valid, 4][diag_mask]  # Accumulated error
        diag_positions = i_valid[diag_mask]

        if len(diag_positions) > 0:
            # If too many points, sample for better visualization
            if len(diag_positions) > 10000:
                sample_idx = np.random.choice(len(diag_positions), 10000, replace=False)
                diag_positions = diag_positions[sample_idx]
                diag_errors = diag_errors[sample_idx]

            axes[1, 1].scatter(diag_positions, diag_errors, alpha=0.6, s=1)
            axes[1, 1].set_xlabel('Position along Diagonal')
            axes[1, 1].set_ylabel('Accumulated Error')
            axes[1, 1].set_title('Error Accumulation along Diagonal')
        else:
            axes[1, 1].text(0.5, 0.5, "No diagonal data points found", ha='center', va='center')
    else:
        axes[1, 1].text(0.5, 0.5, "Insufficient data", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Tile analysis plot saved to {save_path}")

@timeit
def create_summary_plots(analysis_files, output_dir):
    """Create summary plots comparing different conditions - with optimizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Collect data from all files
    conditions = ['random', 'wellcond', 'illcond']
    condition_data = {}

    for condition in conditions:
        condition_files = [f for f in analysis_files if condition in str(f)]
        if condition_files:
            # Use the largest matrix size for this condition
            try:
                largest_file = max(condition_files, key=lambda x: int(str(x).split('_n')[1].split('_')[0]))
                print(f"Loading {condition} condition data from: {largest_file}")
                condition_data[condition] = load_analysis_data(largest_file)
            except (IndexError, ValueError):
                print(f"Could not determine matrix size for {condition} files")
                # Just use the first file
                condition_data[condition] = load_analysis_data(condition_files[0])

    if not condition_data:
        print("No data files found for summary plots")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error distribution comparison
    for condition, data in condition_data.items():
        if len(data) > 0:
            rel_errors = data[:, 3]  # Relative errors
            positive_errors = rel_errors[rel_errors > 0]
            if len(positive_errors) > 0:
                axes[0, 0].hist(positive_errors, bins=50, alpha=0.7,
                            label=condition.title(), density=True)

    axes[0, 0].set_xlabel('Relative Error')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Error Distribution by Matrix Condition')
    axes[0, 0].legend()
    # Use try/except for log scales in case of zero values
    try:
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
    except:
        pass

    # Statistics comparison
    conditions_list = list(condition_data.keys())
    mean_errors = []
    max_errors = []
    std_errors = []

    for condition in conditions_list:
        data = condition_data[condition]
        if len(data) > 0:
            rel_errors = data[:, 3]
            mean_errors.append(np.mean(rel_errors))
            max_errors.append(np.max(rel_errors))
            std_errors.append(np.std(rel_errors))
        else:
            mean_errors.append(0)
            max_errors.append(0)
            std_errors.append(0)

    x = np.arange(len(conditions_list))

    axes[0, 1].bar(x, mean_errors, alpha=0.7)
    axes[0, 1].set_xlabel('Matrix Condition')
    axes[0, 1].set_ylabel('Mean Relative Error')
    axes[0, 1].set_title('Mean Error by Condition')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([c.title() for c in conditions_list])
    try:
        axes[0, 1].set_yscale('log')
    except:
        pass

    axes[1, 0].bar(x, max_errors, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Matrix Condition')
    axes[1, 0].set_ylabel('Maximum Relative Error')
    axes[1, 0].set_title('Maximum Error by Condition')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([c.title() for c in conditions_list])
    try:
        axes[1, 0].set_yscale('log')
    except:
        pass

    axes[1, 1].bar(x, std_errors, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Matrix Condition')
    axes[1, 1].set_ylabel('Standard Deviation of Errors')
    axes[1, 1].set_title('Error Variance by Condition')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([c.title() for c in conditions_list])
    try:
        axes[1, 1].set_yscale('log')
    except:
        pass

    plt.tight_layout()
    plt.savefig(output_dir / 'numerical_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Summary plots saved to {output_dir}")

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Visualize GEMM tiling numerical analysis')
    parser.add_argument('--data-dir', default='.', help='Directory containing analysis data files')
    parser.add_argument('--output-dir', default='./plots', help='Output directory for plots')
    parser.add_argument('--tile-size', type=int, default=16, help='Tile size used in analysis')
    parser.add_argument('--quick', action='store_true', help='Quick mode: subsample large matrices')
    parser.add_argument('--file', help='Process a single specific file')

    args = parser.parse_args()

    # Find all analysis data files
    data_dir = Path(args.data_dir)

    if args.file:
        analysis_files = [Path(args.file)]
    else:
        analysis_files = list(data_dir.glob('numerical_analysis_*.dat'))

    if not analysis_files:
        print(f"No analysis data files found in {data_dir}")
        return

    print(f"Found {len(analysis_files)} analysis files")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Process each file
    for file_path in analysis_files:
        print(f"Processing {file_path}")

        # Extract matrix size from filename
        try:
            matrix_size = int(str(file_path).split('_n')[1].split('_')[0])
        except (IndexError, ValueError):
            print(f"Could not extract matrix size from {file_path}, assuming 1024")
            matrix_size = 1024

        # Load and process data
        data = load_analysis_data(file_path)
        if len(data) == 0:
            print(f"No data found in {file_path}")
            continue

        # Create plots
        base_name = file_path.stem

        # Error heatmaps
        heatmap_path = output_dir / f'{base_name}_heatmaps.png'
        plot_error_heatmap(data, matrix_size, heatmap_path, subsample=args.quick)

        # Tile boundary analysis
        boundary_path = output_dir / f'{base_name}_tile_analysis.png'
        plot_tile_boundary_effects(data, matrix_size, args.tile_size, boundary_path)

        print(f"  Plots saved: {heatmap_path}, {boundary_path}")

    # Create summary comparison plots if we have multiple files
    if len(analysis_files) > 1:
        create_summary_plots(analysis_files, output_dir)

    end_time = time.time()
    print(f"\nAll plots saved to {output_dir}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
