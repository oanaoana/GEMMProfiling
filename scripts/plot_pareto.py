#!/usr/bin/env python3

"""
Pareto Analysis Plotting Script
===============================
Create a single plot showing performance vs accuracy trade-offs for selected kernels.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration copied from plot_beta_ratios.py
DATA_FOLDER = "data"
PLOTS_FOLDER = "plots"
OUTPUT_FORMAT = "png"  # "eps", "png", or "both"

# Select which matrix types to process (None = all available)
MATRIX_TYPES = ['uniform_positive', 'wellcond']  # Set to None for all
KERNELS = None  # Set to None for all, or list like ['tiled', 'cublas']
SIZES = None    # Set to None for all, or list like [256, 512, 1024]
ERROR_METRIC = '|C-C_ref|/(|A||B|)_avg'  # Choose: '|C-C_ref|/(|A||B|)_avg' or 'E_{AB}/u_c'

# Color scheme: 3 colors for 3 kernel families
KERNEL_COLORS = {
    'cublas': 'blue',                    # cuBLAS - blue
    'cutlass_splitk_flat': 'red',        # CUTLASS family - red
    'cutlass_splitk_pairwise': 'red',    # CUTLASS family - red
    'tiled': 'green',                    # Tiled family - green
    'tiled_pairwise': 'green'            # Tiled family - green
}

# Marker scheme: squares for pairwise, circles for others
KERNEL_MARKERS = {
    'cublas': 'o',                       # Circle
    'cutlass_splitk_flat': 'o',          # Circle
    'cutlass_splitk_pairwise': 's',      # Square (pairwise)
    'tiled': 'o',                        # Circle
    'tiled_pairwise': 's'                # Square (pairwise)
}

# Kernel labels for legends
KERNEL_LABELS = {
    'cublas': 'cuBLAS',
    'cutlass_splitk_flat': 'CUTLASS Split-K Flat',
    'cutlass_splitk_pairwise': 'CUTLASS Split-K Pairwise',
    'tiled': 'Tiled (Ours)',
    'tiled_pairwise': 'Tiled Pairwise (Ours)'
}

# Styling configuration
LINE_WIDTH = 1.5
MARKER_SIZE = 9
AXIS_LABEL_FONTSIZE = 18
TICK_LABELSIZE = 16
LEGEND_FONTSIZE = 14
TITLE_FONTSIZE = 18
AXIS_LABEL_BOLD = False
TITLE_BOLD = False
LEGEND_BOLD = False
GRID_ALPHA = 0.3
GRID_LINEWIDTH = 1
PLOT_DPI = 300
BBOX_INCHES = 'tight'

def get_kernel_color(kernel):
    """Get color for kernel, with fallback."""
    return KERNEL_COLORS.get(kernel, 'black')

def get_kernel_marker(kernel):
    """Get marker for kernel, with fallback."""
    return KERNEL_MARKERS.get(kernel, 'o')

def get_kernel_label(kernel):
    """Get label for kernel, with fallback."""
    return KERNEL_LABELS.get(kernel, kernel)

def save_plot(base_filename, format='png', dpi=300, bbox_inches='tight'):
    """Save the current plot in specified format(s)."""
    if format in ['eps', 'both']:
        eps_filename = f"{base_filename}.eps"
        plt.savefig(eps_filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {eps_filename}")

    if format in ['png', 'both']:
        png_filename = f"{base_filename}.png"
        plt.savefig(png_filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {png_filename}")

def load_and_merge_data():
    """Load and merge error and performance data."""

    # Load error data
    error_files = glob.glob(f"{DATA_FOLDER}/error_analysis_*_n*.csv")
    if not error_files:
        print(f"No error analysis files found in {DATA_FOLDER}/")
        return pd.DataFrame()

    print(f"Found {len(error_files)} error analysis files")

    error_dfs = []
    for csv_file in error_files:
        try:
            df = pd.read_csv(csv_file)
            required_columns = ['kernel_type', 'matrix_type', 'matrix_size',
                              '|C-C_ref|/(|A||B|)_avg', 'E_{AB}/u_c']
            if all(col in df.columns for col in required_columns):
                error_dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not error_dfs:
        print("No valid error analysis files found!")
        return pd.DataFrame()

    error_df = pd.concat(error_dfs, ignore_index=True)
    print(f"Loaded {len(error_df)} error analysis data points")

    # Load performance data
    perf_files = glob.glob(f"{DATA_FOLDER}/perf_*.csv")
    if not perf_files:
        perf_files = glob.glob(f"{DATA_FOLDER}/perf_*.dat")
        print(f"Found {len(perf_files)} performance .dat files")
    else:
        print(f"Found {len(perf_files)} performance .csv files")

    if not perf_files:
        print(f"No performance files found in {DATA_FOLDER}/")
        return pd.DataFrame()

    perf_dfs = []
    for csv_file in perf_files:
        try:
            df = pd.read_csv(csv_file)
            if all(col in df.columns for col in ['algorithm', 'size', 'gflops']):
                df = df.rename(columns={'algorithm': 'kernel_type', 'size': 'matrix_size'})
                perf_dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not perf_dfs:
        print("No valid performance files found!")
        return pd.DataFrame()

    perf_df = pd.concat(perf_dfs, ignore_index=True)
    print(f"Loaded {len(perf_df)} performance data points")

    # Merge data
    merged_df = pd.merge(error_df, perf_df, on=['kernel_type', 'matrix_size'], how='inner')
    print(f"Successfully merged {len(merged_df)} data points")

    return merged_df

def format_matrix_size_labels_readable(sizes):
    """Format matrix size labels in readable ×10^n format."""
    labels = []
    tick_positions = []

    for size in sorted(set(sizes)):
        if size >= 1000:
            # Convert to thousands format
            thousands = size / 1000
            if thousands == int(thousands):
                # For whole numbers
                thousands = int(thousands)
                if thousands == 1:
                    labels.append(r'$1 \times 10^3$')
                else:
                    labels.append(fr'${thousands} \times 10^3$')
            else:
                # For decimals, show with one decimal place
                labels.append(fr'${thousands:.1f} \times 10^3$')
        else:
            labels.append(str(size))

        tick_positions.append(size)

    return tick_positions, labels

def format_gflops_labels(gflops_values):
    """Format GFLOPS labels to show values like 5×10³ instead of 5000."""
    labels = []
    tick_positions = []

    for gflops in sorted(set(gflops_values)):
        if gflops >= 1000:
            # Convert to thousands format
            thousands = gflops / 1000
            if thousands == int(thousands):
                # For whole numbers
                thousands = int(thousands)
                if thousands == 1:
                    labels.append(r'$1 \times 10^3$')
                else:
                    labels.append(fr'${thousands} \times 10^3$')
            else:
                # For decimals, show with one decimal place
                labels.append(fr'${thousands:.1f} \times 10^3$')
        else:
            labels.append(str(int(gflops)))

        tick_positions.append(gflops)

    return tick_positions, labels

def format_matrix_size_powers_of_2(sizes):
    """Format matrix size labels using power-of-2 notation only for actual powers of 2."""
    labels = []
    tick_positions = []
    for size in sizes:
        if size & (size - 1) == 0:  # Check if power of 2
            power = int(np.log2(size))
            labels.append(f'$2^{{{power}}}$')
            tick_positions.append(size)
    return tick_positions, labels

def create_pareto_plots_per_matrix_type(df):
    """Create one performance vs accuracy plot per matrix type."""

    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    matrix_types = sorted(df['matrix_type'].unique())

    for matrix_type in matrix_types:
        matrix_df = df[df['matrix_type'] == matrix_type]

        if matrix_df.empty:
            continue

        print(f"\nCreating plot for matrix type: {matrix_type}")

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Plot each kernel
        kernels = sorted(matrix_df['kernel_type'].unique())

        for kernel in kernels:
            kernel_df = matrix_df[matrix_df['kernel_type'] == kernel]

            if kernel_df.empty:
                continue

            # Plot performance (x) vs accuracy (y)
            x_data = kernel_df['gflops']
            y_data = kernel_df[ERROR_METRIC]

            plt.scatter(
                x_data, y_data,
                color=get_kernel_color(kernel),
                marker=get_kernel_marker(kernel),
                s=100,
                alpha=0.8,
                label=get_kernel_label(kernel),
                edgecolors='black',
                linewidth=0.5
            )

            # Add size annotations for each point
            for _, row in kernel_df.iterrows():
                plt.annotate(
                    f"{int(row['matrix_size'])}",
                    (row['gflops'], row[ERROR_METRIC]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )

        # Formatting
        plt.xlabel('Performance (GFLOPS)',
                   fontsize=AXIS_LABEL_FONTSIZE,
                   weight='bold' if AXIS_LABEL_BOLD else 'normal')

        # Format y-axis label based on error metric
        if ERROR_METRIC == '|C-C_ref|/(|A||B|)_avg':
            y_label = r'$\mathbf{E_{AB}}$' if AXIS_LABEL_BOLD else r'$E_{AB}$'
        else:  # E_{AB}/u_c
            y_label = r'$\mathbf{E_{AB}/u_c}$' if AXIS_LABEL_BOLD else r'$E_{AB}/u_c$'

        plt.ylabel(y_label,
                   fontsize=AXIS_LABEL_FONTSIZE,
                   weight='bold' if AXIS_LABEL_BOLD else 'normal')

        # Use log scale for error (y-axis)
        plt.yscale('log')

        # Title
        title_text = f'Performance vs Accuracy - {matrix_type.replace("_", " ").title()}'
        plt.title(title_text,
                  fontsize=TITLE_FONTSIZE,
                  weight='bold' if TITLE_BOLD else 'normal')

        # Legend
        plt.legend(fontsize=LEGEND_FONTSIZE,
                   prop={'weight': 'bold' if LEGEND_BOLD else 'normal'})

        # Grid
        plt.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

        # Tick formatting
        plt.tick_params(labelsize=TICK_LABELSIZE)

        # Format x-axis ticks to be more readable
        if SIZES is not None and all(size in SIZES for size in matrix_df['matrix_size']):
            tick_positions, tick_labels = format_matrix_size_labels_readable(matrix_df['matrix_size'])
            plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=TICK_LABELSIZE)
        else:
            plt.xticks(fontsize=TICK_LABELSIZE)

        plt.tight_layout()

        # Save plot - CHANGED BACK TO "pareto"
        error_name = ERROR_METRIC.replace('|', '').replace('/', '_').replace('{', '').replace('}', '')
        filename = f"pareto_{matrix_type}_{error_name}"  # ← FIXED: Back to "pareto"
        filepath = os.path.join(PLOTS_FOLDER, filename)

        save_plot(filepath, format=OUTPUT_FORMAT, dpi=PLOT_DPI, bbox_inches=BBOX_INCHES)
        plt.close()

        print(f"  ✓ Saved: {filename}.{OUTPUT_FORMAT}")

def create_roofline_plots_per_matrix_type(df):
    """Create one performance vs matrix size plot per matrix type."""

    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    matrix_types = sorted(df['matrix_type'].unique())

    for matrix_type in matrix_types:
        matrix_df = df[df['matrix_type'] == matrix_type]

        if matrix_df.empty:
            continue

        print(f"\nCreating roofline plot for matrix type: {matrix_type}")

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Plot each kernel
        kernels = sorted(matrix_df['kernel_type'].unique())

        for kernel in kernels:
            kernel_df = matrix_df[matrix_df['kernel_type'] == kernel]

            if kernel_df.empty:
                continue

            # Sort by matrix size for connected lines
            kernel_df = kernel_df.sort_values('matrix_size')

            # Plot performance (y) vs matrix size (x)
            x_data = kernel_df['matrix_size']
            y_data = kernel_df['gflops']

            plt.plot(
                x_data, y_data,
                color=get_kernel_color(kernel),
                marker=get_kernel_marker(kernel),
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=get_kernel_label(kernel),
                markeredgecolor='black',
                markeredgewidth=0.5
            )

        # Formatting
        plt.xlabel('Matrix Size (N)',
                   fontsize=AXIS_LABEL_FONTSIZE,
                   weight='bold' if AXIS_LABEL_BOLD else 'normal')

        plt.ylabel('Performance (GFLOPS)',
                   fontsize=AXIS_LABEL_FONTSIZE,
                   weight='bold' if AXIS_LABEL_BOLD else 'normal')

        # Title
        title_text = f'Performance vs Matrix Size - {matrix_type.replace("_", " ").title()}'
        plt.title(title_text,
                  fontsize=TITLE_FONTSIZE,
                  weight='bold' if TITLE_BOLD else 'normal')

        # Legend
        plt.legend(fontsize=LEGEND_FONTSIZE,
                   prop={'weight': 'bold' if LEGEND_BOLD else 'normal'})

        # Grid
        plt.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

        # Tick formatting
        plt.tick_params(labelsize=TICK_LABELSIZE)

        # Format x-axis with the new scientific notation
        unique_sizes = sorted(matrix_df['matrix_size'].unique())
        tick_positions, tick_labels = format_matrix_size_labels_readable(unique_sizes)
        plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=TICK_LABELSIZE)

        plt.tight_layout()

        # Save plot
        filename = f"roofline_{matrix_type}"
        filepath = os.path.join(PLOTS_FOLDER, filename)

        save_plot(filepath, format=OUTPUT_FORMAT, dpi=PLOT_DPI, bbox_inches=BBOX_INCHES)
        plt.close()

        print(f"  ✓ Saved: {filename}.{OUTPUT_FORMAT}")

def main():
    print("Performance vs Accuracy Plotting")
    print("=" * 40)

    # Load and merge data
    print("\n=== Loading Data ===")
    df = load_and_merge_data()

    if df.empty:
        print("Error: No data found!")
        return

    # Filter data if specified
    if KERNELS is not None:
        df = df[df['kernel_type'].isin(KERNELS)]
        print(f"Filtered to kernels: {KERNELS}")

    if MATRIX_TYPES is not None:
        df = df[df['matrix_type'].isin(MATRIX_TYPES)]
        print(f"Filtered to matrix types: {MATRIX_TYPES}")

    if SIZES is not None:
        df = df[df['matrix_size'].isin(SIZES)]
        print(f"Filtered to sizes: {SIZES}")

    if df.empty:
        print("No data remaining after filtering!")
        return

    print(f"\n=== Creating Plots ===")
    print(f"Final dataset: {len(df)} data points")
    print(f"Using error metric: {ERROR_METRIC}")

    # Show what we're plotting
    combinations = df[['kernel_type', 'matrix_size', 'matrix_type']].drop_duplicates()
    print("Plotting combinations:")
    for _, row in combinations.iterrows():
        print(f"  {row['kernel_type']} @ {row['matrix_size']}x{row['matrix_size']} ({row['matrix_type']})")

    # Create both types of plots
    create_pareto_plots_per_matrix_type(df)
    create_roofline_plots_per_matrix_type(df)

    print(f"\n✓ Plots saved to {PLOTS_FOLDER}/")
    print("Generated files:")
    plot_files = glob.glob(f"{PLOTS_FOLDER}/*_{OUTPUT_FORMAT}")
    for pfile in sorted(plot_files):
        print(f"  {os.path.basename(pfile)}")

if __name__ == "__main__":
    main()