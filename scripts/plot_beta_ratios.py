#!/usr/bin/env python3
"""
Simple Beta Ratio Plots
=======================

Creates 15 clean plots:
- 5 plots for E_{AB}/u (one per matrix type, 3 kernel lines, matrix size on x-axis)
- 5 plots for E_{AB}/beta (one per matrix type, 3 kernel lines, matrix size on x-axis)
- 5 plots for E_AB normalized (one per matrix type, 3 kernel lines, matrix size on x-axis)

Usage:
    python scripts/plot_beta_ratios.py
"""

import csv
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

# Configure data folder here
DATA_FOLDER = "data/data_prelim"  # Change this to "data" for current data

def load_data():
    """Load all CSV files and return combined dataframe."""
    csv_files = glob.glob(f"{DATA_FOLDER}/error_analysis_*_*_summary_n*.csv")

    if not csv_files:
        print(f"No CSV files found in {DATA_FOLDER}/!")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} CSV files in {DATA_FOLDER}/")

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} test results")
    return combined_df

def create_beta_plots(df):
    """Create the 15 simple plots as requested."""

    os.makedirs("plots", exist_ok=True)

    # Get unique values
    matrix_types = sorted(df['matrix_type'].unique())
    kernels = sorted(df['kernel_type'].unique())

    print(f"Matrix types: {matrix_types}")
    print(f"Kernels: {kernels}")

    # Colors for all kernels (expandable color palette)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    kernel_colors = dict(zip(kernels, colors[:len(kernels)]))

    print(f"Kernel color mapping: {kernel_colors}")

    # 1. Create 5 plots for E_{AB}/u
    for matrix_type in matrix_types:
        plt.figure(figsize=(10, 6))

        # Collect all y-values for this matrix type to determine bounds
        all_y_values = []

        for kernel in kernels:
            # Get data for this kernel and matrix type
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) > 0:
                # Sort by matrix size
                subset = subset.sort_values('matrix_size')
                y_values = subset['E_{AB}/u']
                all_y_values.extend(y_values)
                plt.plot(subset['matrix_size'], y_values,
                        'o-', label=kernel, color=kernel_colors[kernel],
                        linewidth=2, markersize=6)

        plt.xlabel('Matrix Size', fontsize=12)
        plt.ylabel('E_{AB} / u', fontsize=12)
        plt.title(f'E_{{AB}}/u - Matrix Type: {matrix_type}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Set y-axis bounds proportional to data
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:  # Only adjust if there's significant range
                margin = 0.2  # 20% margin
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        filename = f"plots/E_AB_over_u_{matrix_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    # 2. Create 5 plots for E_{AB}/beta
    for matrix_type in matrix_types:
        plt.figure(figsize=(10, 6))

        # Collect all y-values for this matrix type to determine bounds
        all_y_values = []

        for kernel in kernels:
            # Get data for this kernel and matrix type
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) > 0:
                # Sort by matrix size
                subset = subset.sort_values('matrix_size')
                y_values = subset['E_{AB}/beta']
                all_y_values.extend(y_values)
                plt.plot(subset['matrix_size'], y_values,
                        'o-', label=kernel, color=kernel_colors[kernel],
                        linewidth=2, markersize=6)

        plt.xlabel('Matrix Size', fontsize=12)
        plt.ylabel('E_{AB} / β', fontsize=12)
        plt.title(f'E_{{AB}}/β - Matrix Type: {matrix_type}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Add horizontal line at y=1 (theoretical bound)
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5,
                   label='Theoretical Bound')

        # Set y-axis bounds proportional to data (log scale)
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            # Include y=1 line in bounds calculation
            y_min = min(y_min, 1.0)
            y_max = max(y_max, 1.0)

            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:  # Only adjust if there's significant range
                margin = 0.2  # 20% margin for log scale
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        filename = f"plots/E_AB_over_beta_{matrix_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    # 3. Create 5 plots for E_AB normalized
    for matrix_type in matrix_types:
        plt.figure(figsize=(10, 6))

        # Collect all y-values for this matrix type to determine bounds
        all_y_values = []

        for kernel in kernels:
            # Get data for this kernel and matrix type
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) > 0:
                # Sort by matrix size
                subset = subset.sort_values('matrix_size')
                y_values = subset['|C-C_ref|/(|A||B|)_avg']
                all_y_values.extend(y_values)
                plt.plot(subset['matrix_size'], y_values,
                        'o-', label=kernel, color=kernel_colors[kernel],
                        linewidth=2, markersize=6)

        plt.xlabel('Matrix Size', fontsize=12)
        plt.ylabel('E_AB', fontsize=12)
        plt.title(f'E_AB - Matrix Type: {matrix_type}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Set y-axis bounds proportional to data (log scale)
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:  # Only adjust if there's significant range
                margin = 0.2  # 20% margin for log scale
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        filename = f"plots/E_AB_{matrix_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

def main():
    print("Simple Beta Ratio Plotting")
    print("=" * 30)

    # Load data
    df = load_data()
    if df.empty:
        return

    # Check required columns
    required_cols = ['E_{AB}/beta', 'E_{AB}/u', '|C-C_ref|/(|A||B|)_avg', 'kernel_type', 'matrix_type', 'matrix_size']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return

    # Create plots
    create_beta_plots(df)

    print("\n✓ All plots created successfully!")
    print("Check plots/ directory for:")
    print("  - E_AB_over_u_*.png (5 files)")
    print("  - E_AB_over_beta_*.png (5 files)")
    print("  - E_AB_*.png (5 files)")

if __name__ == "__main__":
    main()
