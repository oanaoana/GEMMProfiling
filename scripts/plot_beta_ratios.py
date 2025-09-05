#!/usr/bin/env python3
"""
Simple Beta Ratio Plots
=======================

Creates 10 clean plots:
- 5 plots for E_{AB}/u (one per matrix type, 3 kernel lines, matrix size on x-axis)
- 5 plots for E_{AB}/beta (one per matrix type, 3 kernel lines, matrix size on x-axis)

Usage:
    python scripts/plot_beta_ratios.py
"""

import csv
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    """Load all CSV files and return combined dataframe."""
    csv_files = glob.glob("data/error_analysis_*_*_summary_n*.csv")

    if not csv_files:
        print("No CSV files found!")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} CSV files")

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} test results")
    return combined_df

def create_beta_plots(df):
    """Create the 10 simple plots as requested."""

    os.makedirs("plots", exist_ok=True)

    # Get unique values
    matrix_types = sorted(df['matrix_type'].unique())
    kernels = sorted(df['kernel_type'].unique())

    print(f"Matrix types: {matrix_types}")
    print(f"Kernels: {kernels}")

    # Colors for the 3 kernels
    colors = ['blue', 'red', 'green']
    kernel_colors = dict(zip(kernels, colors))

    # 1. Create 5 plots for E_{AB}/u
    for matrix_type in matrix_types:
        plt.figure(figsize=(10, 6))

        for kernel in kernels:
            # Get data for this kernel and matrix type
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) > 0:
                # Sort by matrix size
                subset = subset.sort_values('matrix_size')
                plt.plot(subset['matrix_size'], subset['E_{AB}/u'],
                        'o-', label=kernel, color=kernel_colors[kernel],
                        linewidth=2, markersize=6)

        plt.xlabel('Matrix Size', fontsize=12)
        plt.ylabel('E_{AB} / u', fontsize=12)
        plt.title(f'E_{{AB}}/u - Matrix Type: {matrix_type}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Save plot
        filename = f"plots/E_AB_over_u_{matrix_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    # 2. Create 5 plots for E_{AB}/beta
    for matrix_type in matrix_types:
        plt.figure(figsize=(10, 6))

        for kernel in kernels:
            # Get data for this kernel and matrix type
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) > 0:
                # Sort by matrix size
                subset = subset.sort_values('matrix_size')
                plt.plot(subset['matrix_size'], subset['E_{AB}/beta'],
                        'o-', label=kernel, color=kernel_colors[kernel],
                        linewidth=2, markersize=6)

        plt.xlabel('Matrix Size', fontsize=12)
        plt.ylabel('E_{AB} / β', fontsize=12)
        plt.title(f'E_{{AB}}/β - Matrix Type: {matrix_type}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add horizontal line at y=1 (theoretical bound)
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5,
                   label='Theoretical Bound')

        # Save plot
        filename = f"plots/E_AB_over_beta_{matrix_type}.png"
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
    required_cols = ['E_{AB}/beta', 'E_{AB}/u', 'kernel_type', 'matrix_type', 'matrix_size']
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

if __name__ == "__main__":
    main()
