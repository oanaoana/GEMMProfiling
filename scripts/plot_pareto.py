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
from matplotlib.ticker import LogLocator
import warnings
warnings.filterwarnings('ignore')

# Configuration copied from plot_beta_ratios.py
DATA_FOLDER = "data"
PLOTS_FOLDER = "plots"
OUTPUT_FORMAT = "both"  # "eps", "png", or "both"

# Select which matrix types to process (None = all available)
MATRIX_TYPES = ['uniform_positive']  # Set to None for all
KERNELS = None  # Set to None for all, or list like ['tiled', 'cublas']
SIZES = None    # Set to None for all, or list like [256, 512, 1024]
# ERROR_METRIC = '|C-C_ref|/(|A||B|)_avg'  # Choose: '|C-C_ref|/(|A||B|)_avg' or 'E_{AB}/u_c'
ERROR_METRIC = 'E_{AB}/u_c'

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
    """Format GFLOPS labels normally without scientific notation."""
    labels = []
    tick_positions = []

    for gflops in sorted(set(gflops_values)):
        # Just show the number as-is
        if gflops == int(gflops):
            labels.append(str(int(gflops)))
        else:
            labels.append(f'{gflops:.1f}')

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

        # Calculate marker sizes based on matrix size
        all_sizes = matrix_df['matrix_size'].unique()
        min_size = all_sizes.min()
        max_size = all_sizes.max()

        # Scale marker sizes between 20 and 200 (adjust these values as needed)
        min_marker_size = 20
        max_marker_size = 300

        # Plot each kernel
        kernels = sorted(matrix_df['kernel_type'].unique())

        for kernel in kernels:
            kernel_df = matrix_df[matrix_df['kernel_type'] == kernel]

            if kernel_df.empty:
                continue

            # Plot performance (x) vs accuracy (y)
            x_data = kernel_df['gflops']
            y_data = kernel_df[ERROR_METRIC]

            # Calculate marker sizes proportional to matrix size
            marker_sizes = []
            for size in kernel_df['matrix_size']:
                # Linear scaling between min and max marker sizes
                normalized = (size - min_size) / (max_size - min_size)
                marker_size = min_marker_size + normalized * (max_marker_size - min_marker_size)
                marker_sizes.append(marker_size)

            plt.scatter(
                x_data, y_data,
                color=get_kernel_color(kernel),
                marker=get_kernel_marker(kernel),
                s=marker_sizes,  # Use calculated marker sizes
                alpha=0.8,
                label=get_kernel_label(kernel),
                edgecolors='black',
                linewidth=0.5
            )

        # Use log scale for error (y-axis)
        ax = plt.gca()
        ax.set_yscale('log')

        # Set major ticks at each power of ten
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        # Get actual y-axis data range
        y_min = matrix_df[ERROR_METRIC].min()
        y_max = matrix_df[ERROR_METRIC].max()

        y_min = 10**np.floor(np.log10(y_min))
        y_max = 10**np.ceil(np.log10(y_max))
        ax.set_ylim(y_min, y_max)

        # Grid
        plt.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

        # Let matplotlib choose x-axis ticks automatically
        plt.tick_params(labelsize=TICK_LABELSIZE)
        # If x_data is in TFLOPS
        tick_positions = ax.get_xticks()
        tick_positions = [x for x in tick_positions if x >= 0]
        tick_labels = [f"{int(round(x/1000))}" for x in tick_positions]  # Convert GFLOPS to TFLOPS

        plt.xticks(tick_positions, tick_labels)

        # Formatting
        plt.xlabel('Performance (TFLOPS)',
                   fontsize=AXIS_LABEL_FONTSIZE,
                   weight='bold' if AXIS_LABEL_BOLD else 'normal')

        # Format y-axis label based on error metric
        if ERROR_METRIC == '|C-C_ref|/(|A||B|)_avg':
            y_label = r'$E_{AB}$'
        elif ERROR_METRIC == 'E_{AB}/u_c':
            y_label = r'$E_{AB}/u_c$'
        else:
            # For any other metric, use the column name directly
            y_label = ERROR_METRIC

        plt.ylabel(y_label,
                   fontsize=AXIS_LABEL_FONTSIZE,
                   weight='bold' if AXIS_LABEL_BOLD else 'normal')

         # Title
        title_text = f'Performance vs Accuracy - {matrix_type.replace("_", " ").title()}'
        plt.title(title_text,
                  fontsize=TITLE_FONTSIZE,
                  weight='bold' if TITLE_BOLD else 'normal')

        # # Legend
        # plt.legend(fontsize=LEGEND_FONTSIZE,
        #            prop={'weight': 'bold' if LEGEND_BOLD else 'normal'})

        plt.tight_layout()

        # Save plot
        error_name = ERROR_METRIC.replace('|', '').replace('/', '_').replace('{', '').replace('}', '')
        filename = f"pareto_{matrix_type}_{error_name}"
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

            # Plot performance (y) vs matrix size (x) - MARKERS ONLY
            x_data = kernel_df['matrix_size']
            y_data = kernel_df['gflops']

            plt.scatter(  # Changed from plt.plot to plt.scatter
                x_data, y_data,
                color=get_kernel_color(kernel),
                marker=get_kernel_marker(kernel),
                s=MARKER_SIZE * 10,  # Scale up marker size since no lines
                alpha=0.8,
                label=get_kernel_label(kernel),
                edgecolors='black',
                linewidth=0.5
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

        # Format x-axis with powers of 2 notation
        unique_sizes = sorted(matrix_df['matrix_size'].unique())
        tick_positions, tick_labels = format_matrix_size_powers_of_2(unique_sizes)
        plt.xticks(tick_positions, tick_labels, fontsize=TICK_LABELSIZE)

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