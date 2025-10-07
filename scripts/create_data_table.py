#!/usr/bin/env python3
"""
Create Data Table Summary
========================

Creates a comprehensive data table with frobenius, normalized frobenius,
E_{AB}/u, E_{AB}/beta for every matrix type, kernel, and size.

Usage:
    python scripts/create_data_table.py
"""

import csv
import glob
import pandas as pd
import os

# Configure data folder here
DATA_FOLDER = "data/data9_17"  # Change this to "data" for current data

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

def create_summary_tables(df):
    """Create comprehensive summary tables."""

    # Create output directory
    os.makedirs("data/summary_tables", exist_ok=True)

    # 1. Complete data table with all measurements
    print("Creating complete data table...")

    # Select and rename columns for clarity
    table_df = df[['kernel_type', 'matrix_type', 'matrix_size',
                   '|C-C_ref|_avg', '|C-C_ref|/(|A||B|)_avg', 'theoretical_beta', 'E_{AB}/u', 'E_{AB}/beta']].copy()

    # Rename columns for clarity
    table_df.columns = ['Kernel', 'Matrix_Type', 'Size',
                       'Frobenius_Error', 'Normalized_Frobenius', 'Theoretical_Beta',
                       'E_{AB}/u', 'E_{AB}/beta']

    # Sort by matrix type, kernel, size for better readability
    table_df = table_df.sort_values(['Matrix_Type', 'Kernel', 'Size'])

    # Save complete table
    table_df.to_csv("data/summary_tables/complete_error_summary.csv", index=False)
    print("✓ Saved: data/summary_tables/complete_error_summary.csv")

    # 2. Summary by matrix type (averaged across kernels and sizes)
    print("Creating matrix type summary...")

    matrix_summary = df.groupby('matrix_type').agg({
        '|C-C_ref|_avg': ['mean', 'std', 'min', 'max'],
        '|C-C_ref|/(|A||B|)_avg': ['mean', 'std', 'min', 'max'],
        'theoretical_beta': ['mean', 'std', 'min', 'max'],
        'E_{AB}/u': ['mean', 'std', 'min', 'max'],
        'E_{AB}/beta': ['mean', 'std', 'min', 'max']
    }).round(6)

    # Flatten column names
    matrix_summary.columns = ['_'.join(col).strip() for col in matrix_summary.columns]
    matrix_summary.reset_index(inplace=True)

    matrix_summary.to_csv("data/summary_tables/matrix_type_summary.csv", index=False)
    print("✓ Saved: data/summary_tables/matrix_type_summary.csv")

    # 3. Summary by kernel type (averaged across matrix types and sizes)
    print("Creating kernel type summary...")

    kernel_summary = df.groupby('kernel_type').agg({
        '|C-C_ref|_avg': ['mean', 'std', 'min', 'max'],
        '|C-C_ref|/(|A||B|)_avg': ['mean', 'std', 'min', 'max'],
        'theoretical_beta': ['mean', 'std', 'min', 'max'],
        'E_{AB}/u': ['mean', 'std', 'min', 'max'],
        'E_{AB}/beta': ['mean', 'std', 'min', 'max']
    }).round(6)

    # Flatten column names
    kernel_summary.columns = ['_'.join(col).strip() for col in kernel_summary.columns]
    kernel_summary.reset_index(inplace=True)

    kernel_summary.to_csv("data/summary_tables/kernel_type_summary.csv", index=False)
    print("✓ Saved: data/summary_tables/kernel_type_summary.csv")

    # 4. Pivot table: Matrix Type vs Kernel (averaged across sizes)
    print("Creating matrix-kernel pivot tables...")

    # Beta over theoretical
    pivot_theoretical = df.groupby(['matrix_type', 'kernel_type'])['E_{AB}/beta'].mean().unstack()
    pivot_theoretical.to_csv("data/summary_tables/matrix_kernel_beta_theoretical.csv")
    print("✓ Saved: data/summary_tables/matrix_kernel_beta_theoretical.csv")

    # Beta over u32
    pivot_u32 = df.groupby(['matrix_type', 'kernel_type'])['E_{AB}/u'].mean().unstack()
    pivot_u32.to_csv("data/summary_tables/matrix_kernel_beta_u32.csv")
    print("✓ Saved: data/summary_tables/matrix_kernel_beta_u32.csv")

    # Frobenius errors
    pivot_frob = df.groupby(['matrix_type', 'kernel_type'])['|C-C_ref|_avg'].mean().unstack()
    pivot_frob.to_csv("data/summary_tables/matrix_kernel_frobenius.csv")
    print("✓ Saved: data/summary_tables/matrix_kernel_frobenius.csv")

    # Normalized frobenius
    pivot_norm = df.groupby(['matrix_type', 'kernel_type'])['|C-C_ref|/(|A||B|)_avg'].mean().unstack()
    pivot_norm.to_csv("data/summary_tables/matrix_kernel_normalized_frobenius.csv")
    print("✓ Saved: data/summary_tables/matrix_kernel_normalized_frobenius.csv")

    # 5. Create a human-readable formatted table
    print("Creating formatted summary table...")

    # Create a nicely formatted table for each matrix type
    with open("data/summary_tables/formatted_summary.txt", "w") as f:
        f.write("GEMM ERROR ANALYSIS SUMMARY\n")
        f.write("="*105 + "\n\n")

        for matrix_type in sorted(df['matrix_type'].unique()):
            f.write(f"Matrix Type: {matrix_type.upper()}\n")
            f.write("-" * 55 + "\n")

            matrix_data = df[df['matrix_type'] == matrix_type]

            # Header with proper spacing
            f.write(f"{'Kernel':<23} {'Size':<6} {'|C-C_ref|':<13} {'|C-C_ref|/|A||B|':<16} {'Theo_Beta':<13} {'E_AB/u':<12} {'E_AB/beta':<12}\n")
            f.write("-" * 105 + "\n")

            for _, row in matrix_data.sort_values(['kernel_type', 'matrix_size']).iterrows():
                f.write(f"{row['kernel_type']:<23} {int(row['matrix_size']):<6} "
                       f"{row['|C-C_ref|_avg']:<13.3e} {row['|C-C_ref|/(|A||B|)_avg']:<16.3e} "
                       f"{row['theoretical_beta']:<13.3e} {row['E_{AB}/u']:<12.3e} {row['E_{AB}/beta']:<12.3e}\n")

            f.write("\n")

    print("✓ Saved: data/summary_tables/formatted_summary.txt")

def print_quick_summary(df):
    """Print a quick summary to console."""

    print("\n" + "="*60)
    print("QUICK DATA SUMMARY")
    print("="*60)

    print(f"Total configurations: {len(df)}")
    print(f"Matrix types: {sorted(df['matrix_type'].unique())}")
    print(f"Kernels: {sorted(df['kernel_type'].unique())}")
    print(f"Sizes: {sorted(df['matrix_size'].unique())}")

    print(f"\nOverall Statistics:")
    print(f"  Frobenius Error - Mean: {df['|C-C_ref|_avg'].mean():.3e}, Std: {df['|C-C_ref|_avg'].std():.3e}")
    print(f"  Normalized Error - Mean: {df['|C-C_ref|/(|A||B|)_avg'].mean():.3e}, Std: {df['|C-C_ref|/(|A||B|)_avg'].std():.3e}")
    print(f"  E_AB/u - Mean: {df['E_{AB}/u'].mean():.3e}, Std: {df['E_{AB}/u'].std():.3e}")
    print(f"  E_AB/beta - Mean: {df['E_{AB}/beta'].mean():.3e}, Std: {df['E_{AB}/beta'].std():.3e}")

    print(f"\nBest performing configurations (lowest E_AB/beta):")
    best = df.nsmallest(3, 'E_{AB}/beta')
    for i, (_, row) in enumerate(best.iterrows(), 1):
        print(f"  {i}. {row['kernel_type']} + {row['matrix_type']} @ n={int(row['matrix_size'])}: {row['E_{AB}/beta']:.3e}×")

def main():
    print("Creating Data Table Summary")
    print("=" * 40)

    # Load data
    df = load_data()
    if df.empty:
        return

    # Check required columns
    required_cols = ['kernel_type', 'matrix_type', 'matrix_size', '|C-C_ref|_avg', '|C-C_ref|/(|A||B|)_avg', 'theoretical_beta', 'E_{AB}/u', 'E_{AB}/beta']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Create summary tables
    create_summary_tables(df)

    # Print quick summary
    print_quick_summary(df)

    print("\n" + "="*60)
    print("DATA TABLE CREATION COMPLETE")
    print("="*60)
    print("Check data/summary_tables/ directory for:")
    print("  - complete_error_summary.csv (all data)")
    print("  - matrix_type_summary.csv (by matrix type)")
    print("  - kernel_type_summary.csv (by kernel type)")
    print("  - matrix_kernel_*.csv (pivot tables)")
    print("  - formatted_summary.txt (human readable)")

if __name__ == "__main__":
    main()
