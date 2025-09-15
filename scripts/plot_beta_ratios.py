#!/usr/bin/env python3
"""
Simple Beta Ratio Plots
=======================

Creates 15 clean plots:
- 5 plots for E_{AB}/u (one per matrix type, 5 kernel lines, K dimension on x-axis)
- 5 plots for E_{AB}/beta (one per matrix type, 5 kernel lines, K dimension on x-axis)
- 5 plots for E_AB normalized (one per matrix type, 5 kernel lines, K dimension on x-axis)

Usage:
    python scripts/plot_beta_ratios.py
"""

import csv
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO


# Configure data folder here
DATA_FOLDER = "data"  # Change this to "data" for current data
OUTPUT_FORMAT = "png"  # "eps", "png", or "both"

# Select which matrix types to process (None = all available)
# Available types: '2powers', 'illcond', 'uniform_positive', 'wellcond', 'zeromean'
MATRIX_TYPES = ['uniform_positive']  # Set to None for all, or list like ['2powers', 'wellcond']

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

# Line and marker styling
LINE_WIDTH = 1.5          # Width of plot lines
MARKER_SIZE = 9           # Size of markers
THEORETICAL_LINE_WIDTH = 2  # Width of theoretical/reference lines

KERNEL_LABELS = {
    'cublas': 'cuBLAS',
    'cutlass_splitk_flat': 'CUTLASS Split-K Flat',
    'cutlass_splitk_pairwise': 'CUTLASS Split-K Pairwise',
    'tiled': 'Tiled (Ours)',
    'tiled_pairwise': 'Tiled Pairwise (Ours)'
}

# Enable better mathematical notation using matplotlib's mathtext (no LaTeX required)
plt.rcParams['font.size'] = 12

def save_plot(base_filename, format='both', dpi=300, bbox_inches='tight'):
    """Save the current plot in specified format(s).

    Args:
        base_filename: Base filename without extension
        format: 'eps', 'png', or 'both'
        dpi: Resolution for saving
        bbox_inches: Bounding box setting
    """
    if format in ['eps', 'both']:
        eps_filename = f"{base_filename}.eps"
        plt.savefig(eps_filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {eps_filename}")

    if format in ['png', 'both']:
        png_filename = f"{base_filename}.png"
        plt.savefig(png_filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {png_filename}")

def format_matrix_size_labels(sizes):
    """Format matrix size labels, using power-of-2 notation only for actual powers of 2."""
    labels = []
    tick_positions = []
    for size in sizes:
        if size & (size - 1) == 0:  # Check if power of 2
            power = int(np.log2(size))
            labels.append(f'$2^{{{power}}}$')
            tick_positions.append(size)
    return tick_positions, labels

def load_data():
    """Load all CSV files and return combined dataframe."""
    csv_files = glob.glob(f"{DATA_FOLDER}/error_analysis_*_*_summary_n*.csv")

    if not csv_files:
        print(f"No CSV files found in {DATA_FOLDER}/!")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} CSV files in {DATA_FOLDER}/")

    all_data = []
    for csv_file in csv_files:
        try:
            # First try normal reading
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except pd.errors.EmptyDataError:
            # If that fails, read the file manually and fix the header
            print(f"Fixing header for {csv_file}")
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            # The header is split across first two lines, combine them
            if len(lines) >= 2 and lines[1].startswith(','):
                header_line = lines[0].strip() + lines[1].strip()
                # Remove the data lines that start with the matrix type
                data_lines = [line for line in lines[2:] if not line.strip().startswith(',')]

                # Create a temporary fixed content
                fixed_content = header_line + '\n' + ''.join(data_lines)

                # Read from the fixed content
                from io import StringIO
                df = pd.read_csv(StringIO(fixed_content))
                all_data.append(df)
            else:
                print(f"Cannot fix header for {csv_file}, skipping...")
                continue

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} test results")
    return combined_df

def create_beta_plots(df, actual_matrix_sizes):
    """Create the 15 simple plots as requested."""

    os.makedirs("plots", exist_ok=True)

    # Get unique values
    matrix_types = sorted(df['matrix_type'].unique())
    kernels = sorted(df['kernel_type'].unique())

    print(f"Matrix types: {matrix_types}")
    print(f"Kernels: {kernels}")

    print(f"Using kernel color mapping: {KERNEL_COLORS}")
    print(f"Using kernel marker mapping: {KERNEL_MARKERS}")

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
                color = KERNEL_COLORS[kernel]
                marker = KERNEL_MARKERS[kernel]
                plt.plot(subset['matrix_size'], y_values,
                        f'{marker}-', color=color,
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, fillstyle='none')

        # Overlays removed as requested

        plt.xlabel('K - inner matrix dimension (K=N)', fontsize=14, weight='bold')
        plt.ylabel(r'$\mathbf{E_{AB} / u}$', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Set x-axis to show only powers of 2
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        plt.xticks(tick_positions, tick_labels)

        # Set y-axis bounds proportional to data
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:  # Only adjust if there's significant range
                margin = 0.2  # 20% margin
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        base_filename = f"plots/E_AB_over_u_{matrix_type}"
        save_plot(base_filename, format=OUTPUT_FORMAT)
        plt.close()

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
                color = KERNEL_COLORS[kernel]
                marker = KERNEL_MARKERS[kernel]
                plt.plot(subset['matrix_size'], y_values,
                        f'{marker}-', color=color,
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, fillstyle='none')

        plt.xlabel('K - inner matrix dimension (K=N)', fontsize=14, weight='bold')
        plt.ylabel(r'$\mathbf{E_{AB} / \beta}$', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Set x-axis to show only powers of 2
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        plt.xticks(tick_positions, tick_labels)

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
        base_filename = f"plots/E_AB_over_beta_{matrix_type}"
        save_plot(base_filename, format=OUTPUT_FORMAT)
        plt.close()

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
                color = KERNEL_COLORS[kernel]
                marker = KERNEL_MARKERS[kernel]
                plt.plot(subset['matrix_size'], y_values,
                        f'{marker}-', color=color,
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, fillstyle='none')

        plt.xlabel('K - inner matrix dimension (K=N)', fontsize=14, weight='bold')
        plt.ylabel(r'$\mathbf{E_{AB}}$', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Set x-axis to show only powers of 2
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        plt.xticks(tick_positions, tick_labels)

        # Set y-axis bounds proportional to data (log scale)
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:  # Only adjust if there's significant range
                margin = 0.2  # 20% margin for log scale
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        base_filename = f"plots/E_AB_{matrix_type}"
        save_plot(base_filename, format=OUTPUT_FORMAT)
        plt.close()

U32 = 2.0**-24

def gamma_of(k, u=U32):
    """
    Higham-style gamma: k*u / (1 - k*u).
    Works with scalars or NumPy arrays. Clips near 1/u to avoid blow-ups.
    """
    k = np.asarray(k, dtype=np.float64)
    ku = k * u
    ku = np.clip(ku, 0.0, 0.999999)       # safe guard
    return ku / (1.0 - ku)

def effective_depth(E_over_u, E_over_beta):
    """
    Given measured E/u and E/beta, compute "effective depth".
    Returns E_over_u / max(E_over_beta, 1e-300) which approximates beta/u.
    """
    E_over_u = np.asarray(E_over_u, dtype=np.float64)
    E_over_beta = np.asarray(E_over_beta, dtype=np.float64)
    return E_over_u / np.maximum(E_over_beta, 1e-300)

def ceil_div(a, b):
    return (a + b - 1) // b

def ceil_log2(x):
    x = np.asarray(x, dtype=np.int64)
    x = np.maximum(x, 1)
    return np.ceil(np.log2(x)).astype(np.int64)

def beta_over_u_tiled_flat(K, TILE_K=32, u=U32):
    K = np.asarray(K, dtype=np.int64)
    L = ceil_div(K, TILE_K)
    beta = gamma_of(TILE_K, u) + gamma_of(L, u)
    return beta / u

def beta_over_u_tiled_pairwise(K, TILE_K=32, u=U32):
    K = np.asarray(K, dtype=np.int64)
    L = ceil_div(K, TILE_K)
    beta = gamma_of(TILE_K, u) + gamma_of(ceil_log2(L), u)
    return beta / u

def beta_over_u_splitk_flat(K, S=16, u=U32):
    K = np.asarray(K, dtype=np.int64)
    Ks = ceil_div(K, S)
    beta = gamma_of(Ks, u) + gamma_of(S, u)
    return beta / u

def beta_over_u_splitk_pairwise(K, S=16, u=U32):
    K = np.asarray(K, dtype=np.int64)
    Ks = ceil_div(K, S)
    beta = gamma_of(Ks, u) + gamma_of(ceil_log2(S), u)
    return beta / u

def calibrate_c_hat(E_over_beta_values):
    """
    Input: array of E/β values for a single kernel across all matrix sizes
    Returns: scalar c_hat as median over all values
    """
    arr = np.asarray(E_over_beta_values, dtype=np.float64)
    return float(np.median(arr))

def predict_E_over_u(beta_over_u, c_hat):
    """Elementwise predicted E/u = c_hat * (beta/u)."""
    return c_hat * np.asarray(beta_over_u, dtype=np.float64)

def create_deff_plot(df, actual_matrix_sizes, kernel_filter, plot_name, theoretical_func=None, theoretical_params=None, theoretical_label=None, theoretical_color='darkblue', multiple_theories=None):
    """
    Create effective depth plot overlayed with beta/u for specified kernels.

    Args:
        df: DataFrame with the data
        actual_matrix_sizes: List of matrix sizes from data
        kernel_filter: Function that takes kernel name and returns True if it should be included
        plot_name: Name for the plot files (e.g., "cutlass_splitk_pairwise", "tiled_pairwise", "all_kernels")
        theoretical_func: Function to compute theoretical beta/u (optional)
        theoretical_params: Dict of parameters for theoretical function (optional)
        theoretical_label: Label for theoretical line (optional)
        theoretical_color: Color for theoretical line
        multiple_theories: List of dicts for multiple theoretical lines (optional)
                          Each dict should have: {'func': func, 'params': {}, 'label': str, 'color': str}
    """

    os.makedirs("plots", exist_ok=True)

    # Filter kernels based on the provided filter function
    matching_kernels = [k for k in df['kernel_type'].unique() if kernel_filter(k)]

    if not matching_kernels:
        print(f"No kernels found matching filter for {plot_name}")
        return

    print(f"Found kernels for {plot_name}: {matching_kernels}")

    # Get unique matrix types
    matrix_types = sorted(df['matrix_type'].unique())

    # Create one plot per matrix type
    for matrix_type in matrix_types:
        plt.figure(figsize=(12, 8))

        all_y_values = []

        # Plot experimental effective depth for each matching kernel
        for i, kernel in enumerate(matching_kernels):
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) > 0:
                subset = subset.sort_values('matrix_size')

                # Compute effective depth from experimental data
                E_over_u = subset['E_{AB}/u'].values
                E_over_beta = subset['E_{AB}/beta'].values
                deff_experimental = effective_depth(E_over_u, E_over_beta)

                all_y_values.extend(deff_experimental)

                color = KERNEL_COLORS[kernel]
                marker = KERNEL_MARKERS[kernel]
                plt.plot(subset['matrix_size'], deff_experimental,
                        f'{marker}-', color=color,
                        markersize=MARKER_SIZE, linewidth=LINE_WIDTH, fillstyle='none')

        # Add single theoretical β/u line if provided (TEMPORARILY DISABLED)
        # if theoretical_func is not None:
        #     K_grid = np.array(actual_matrix_sizes, dtype=np.int64)
        #     if theoretical_params:
        #         theoretical_values = theoretical_func(K_grid, **theoretical_params)
        #     else:
        #         theoretical_values = theoretical_func(K_grid)

        #     plt.plot(K_grid, theoretical_values, '--',
        #             color=theoretical_color, alpha=0.9, linewidth=THEORETICAL_LINE_WIDTH)

        #     all_y_values.extend(theoretical_values)

        # Add multiple theoretical lines if provided
        if multiple_theories is not None:
            K_grid = np.array(actual_matrix_sizes, dtype=np.int64)
            for theory in multiple_theories:
                if theory['params']:
                    theoretical_values = theory['func'](K_grid, **theory['params'])
                else:
                    theoretical_values = theory['func'](K_grid)

                linestyle = theory.get('linestyle', '--')
                plt.plot(K_grid, theoretical_values, linestyle,
                        color=theory['color'], alpha=0.8, linewidth=THEORETICAL_LINE_WIDTH)

                all_y_values.extend(theoretical_values)

        plt.xlabel('K - inner matrix dimension (K=N)', fontsize=14, weight='bold')
        plt.ylabel(r'$\mathbf{D_{eff}}$', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Set x-axis to show only powers of 2
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        plt.xticks(tick_positions, tick_labels)

        # Set y-axis bounds proportional to data
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:  # Only adjust if there's significant range
                margin = 0.2  # 20% margin
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        base_filename = f"plots/Deff_{plot_name}_{matrix_type}"
        save_plot(base_filename, format=OUTPUT_FORMAT)
        plt.close()

def create_uncertainty_band_plots(df, actual_matrix_sizes):
    """Create plots with mean + p10-p90 uncertainty bands for E/u ratios."""

    os.makedirs("plots", exist_ok=True)

    # Get unique values
    matrix_types = sorted(df['matrix_type'].unique())
    kernels = sorted(df['kernel_type'].unique())

    print(f"Creating uncertainty band plots for kernels: {kernels}")

    # Create one plot per matrix type
    for matrix_type in matrix_types:
        plt.figure(figsize=(10, 6))

        # Colors for kernels
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        kernel_colors = dict(zip(kernels, colors[:len(kernels)]))

        all_y_values = []

        for kernel in kernels:
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) > 0:
                subset = subset.sort_values('matrix_size')

                # Get mean, p10, and p95 for E/u ratios
                mean_values = subset['E_{AB}/u'].values

                # The percentiles are already in the same units as the averages
                # E_{AB}/u = normalized_error_avg / u32
                # So p10/p95 bands should be: percentile / u32
                u32 = 2.0**-24  # unit roundoff
                p10_eu = subset['|C-C_ref|/(|A||B|)_p10'].values / u32
                p95_eu = subset['|C-C_ref|/(|A||B|)_p95'].values / u32

                # Debug: print values to see what we're getting
                print(f"Debug {kernel} - {matrix_type}:")
                print(f"  Mean E/u: {mean_values}")
                print(f"  P10 E/u: {p10_eu}")
                print(f"  P95 E/u: {p95_eu}")
                print(f"  Band width: {p95_eu - p10_eu}")
                print(f"  Relative band width: {(p95_eu - p10_eu) / mean_values * 100}%")
                print(f"  Matrix sizes: {subset['matrix_size'].values}")

                # Check if bands are too narrow to see
                band_width = p95_eu - p10_eu
                relative_width = band_width / mean_values * 100
                if np.any(relative_width < 5):  # Less than 5% variation
                    print(f"  WARNING: Band very narrow for {kernel}, consider different visualization")

                all_y_values.extend(mean_values)
                all_y_values.extend(p10_eu)
                all_y_values.extend(p95_eu)

                color = KERNEL_COLORS[kernel]
                marker = KERNEL_MARKERS[kernel]

                # Plot mean line
                plt.plot(subset['matrix_size'], mean_values, f'{marker}-',
                        label=f'{KERNEL_LABELS[kernel]} (mean)', color=color,
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, fillstyle='none')

                # Plot uncertainty band (p10-p95)
                print(f"  Plotting band for {kernel}: x={subset['matrix_size'].values}, y1={p10_eu}, y2={p95_eu}")
                plt.fill_between(subset['matrix_size'], p10_eu, p95_eu,
                               alpha=0.5, color=color,
                               label=f'{KERNEL_LABELS[kernel]} p10-p95')

        plt.xlabel('K - inner matrix dimension (K=N)', fontsize=14, weight='bold')
        plt.ylabel(r'$\mathbf{E_{AB} / u}$', fontsize=14, weight='bold')
        plt.title(f'E_{{AB}}/u with Uncertainty Bands - {matrix_type}', fontsize=16, weight='bold')
        plt.legend(frameon=False, ncol=2, fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Set x-axis to show only powers of 2
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        plt.xticks(tick_positions, tick_labels)

        # Set y-axis bounds proportional to data
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:  # Only adjust if there's significant range
                margin = 0.2  # 20% margin
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

                # Save plot
        base_filename = f"plots/Deff_{plot_name}_{matrix_type}"
        save_plot(base_filename, format=OUTPUT_FORMAT)
        plt.close()

def create_relative_spread_plots(df, actual_matrix_sizes):
    """Create companion plots showing relative deviation from mean as percentages."""

    os.makedirs("plots", exist_ok=True)

    # Get unique values
    matrix_types = sorted(df['matrix_type'].unique())
    kernels = sorted(df['kernel_type'].unique())

    print(f"Creating relative-spread plots for kernels: {kernels}")

    # Create one plot per matrix type
    for matrix_type in matrix_types:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                       gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.3})

        # Colors for kernels
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        kernel_colors = dict(zip(kernels, colors[:len(kernels)]))

        all_y_values = []
        all_relative_values = []

        for kernel in kernels:
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) > 0:
                subset = subset.sort_values('matrix_size')

                # Get mean, p10, and p95 for E/u ratios
                mean_values = subset['E_{AB}/u'].values

                u32 = 2.0**-24  # unit roundoff
                p10_eu = subset['|C-C_ref|/(|A||B|)_p10'].values / u32
                p95_eu = subset['|C-C_ref|/(|A||B|)_p95'].values / u32

                # Compute relative deviations as percentages
                p10_relative = (p10_eu / mean_values - 1) * 100  # (p10/mean - 1) × 100%
                p95_relative = (p95_eu / mean_values - 1) * 100  # (p95/mean - 1) × 100%

                all_y_values.extend(mean_values)
                all_relative_values.extend(p10_relative)
                all_relative_values.extend(p95_relative)

                color = KERNEL_COLORS[kernel]
                marker = KERNEL_MARKERS[kernel]

                # Top panel: Main plot with mean lines (original scale)
                ax1.plot(subset['matrix_size'], mean_values, f'{marker}-',
                        color=color,
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, fillstyle='none')

                # Bottom panel: Relative spread (companion panel)
                ax2.fill_between(subset['matrix_size'], p10_relative, p95_relative,
                               alpha=0.7, color=color)

                # print(f"Relative spread for {kernel} - {matrix_type}:")
                # print(f"  P10 deviation: {p10_relative}")
                # print(f"  P95 deviation: {p95_relative}")
                # print(f"  Spread range: {p95_relative - p10_relative}%")

        # Format top panel (main plot)
        ax1.set_ylabel(r'$\mathbf{\||C-C_{ref}\||_\infty / (u \||A\|| \||B\||)}$', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # Set x-axis for top panel
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)
        ax1.set_xticklabels([])  # Remove x-labels from top panel

        # Format bottom panel (relative spread)
        ax2.set_xlabel('K - inner matrix dimension (K=N)', fontsize=14, weight='bold')
        ax2.set_ylabel('Deviation from mean (%) - P10 to P95 spread', fontsize=14, weight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=12)        # Set x-axis for bottom panel
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels)

        # Set y-axis bounds for relative spread (symmetric around 0)
        if all_relative_values:
            max_abs_rel = max(abs(min(all_relative_values)), abs(max(all_relative_values)))
            margin = max_abs_rel * 0.1  # 10% margin
            ax2.set_ylim(-max_abs_rel - margin, max_abs_rel + margin)

        # Set y-axis bounds for main plot
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:
                margin = 0.2  # 20% margin
                ax1.set_ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        base_filename = f"plots/E_AB_over_u_relative_spread_{matrix_type}"
        save_plot(base_filename, format=OUTPUT_FORMAT)
        plt.close()

def create_calibration_comparison_plots(df, actual_matrix_sizes, target_kernels, plot_name="calibration_comparison"):
    """Create plots comparing experimental E/u with c_hat * β/u predictions."""

    os.makedirs("plots", exist_ok=True)

    # Theoretical functions for different algorithms
    theoretical_functions = {
        'cublas': beta_over_u_tiled_flat,  # cuBLAS follows tiled flat pattern
        'tiled': beta_over_u_tiled_flat,
        'cutlass_splitk_flat': beta_over_u_splitk_flat,
        'cutlass_splitk_pairwise': beta_over_u_splitk_pairwise,
        'tiled_pairwise': beta_over_u_tiled_pairwise
    }

    # Parameters for theoretical functions
    theoretical_params = {
        'cublas': {'TILE_K': 32},
        'tiled': {'TILE_K': 32},
        'cutlass_splitk_flat': {'S': 16},
        'cutlass_splitk_pairwise': {'S': 16},
        'tiled_pairwise': {'TILE_K': 32}
    }

    # Get unique matrix types
    matrix_types = sorted(df['matrix_type'].unique())

    # Create one plot per matrix type
    for matrix_type in matrix_types:
        plt.figure(figsize=(12, 8))

        all_y_values = []
        K_grid = np.array(actual_matrix_sizes, dtype=np.int64)

        for kernel in target_kernels:
            # Get experimental data for this kernel and matrix type
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) == 0:
                print(f"No data found for {kernel} - {matrix_type}")
                continue

            subset = subset.sort_values('matrix_size')

            # Get experimental E/u values
            experimental_E_over_u = subset['E_{AB}/u'].values
            experimental_E_over_beta = subset['E_{AB}/beta'].values

            # Calibrate c_hat using E/β values for this kernel
            c_hat = calibrate_c_hat(experimental_E_over_beta)
            print(f"Calibrated c_hat for {kernel} ({matrix_type}): {c_hat:.4f}")

            # Get theoretical β/u values
            if kernel in theoretical_functions:
                theoretical_func = theoretical_functions[kernel]
                params = theoretical_params[kernel]
                theoretical_beta_over_u = theoretical_func(K_grid, **params)

                # Predict E/u using calibrated c_hat
                predicted_E_over_u = predict_E_over_u(theoretical_beta_over_u, c_hat)

                # Only use values at actual matrix sizes tested
                mask = np.isin(K_grid, subset['matrix_size'].values)
                predicted_E_over_u_filtered = predicted_E_over_u[mask]
                actual_sizes_filtered = K_grid[mask]

            else:
                print(f"No theoretical function for {kernel}")
                continue

            color = KERNEL_COLORS[kernel]
            marker = KERNEL_MARKERS[kernel]

            # Plot experimental E/u (solid line)
            plt.plot(subset['matrix_size'], experimental_E_over_u,
                    f'{marker}-', color=color, linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE, fillstyle='none',
                    label=f'{kernel} E/u (experimental)')

            # Plot predicted E/u (dashed line with different marker)
            plt.plot(actual_sizes_filtered, predicted_E_over_u_filtered,
                    'x--', color=color, linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE+1, alpha=0.8,
                    label=f'{kernel} c_hat×β/u (c={c_hat:.3f})')

            all_y_values.extend(experimental_E_over_u)
            all_y_values.extend(predicted_E_over_u_filtered)

        plt.xlabel('K - inner matrix dimension (K=N)', fontsize=14, weight='bold')
        plt.ylabel(r'$\mathbf{E_{AB} / u}$', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(frameon=False, fontsize=10, loc='best')

        # Set x-axis to show only powers of 2
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        plt.xticks(tick_positions, tick_labels)

        # Set y-axis bounds proportional to data
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:  # Only adjust if there's significant range
                margin = 0.2  # 20% margin
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        base_filename = f"plots/{plot_name}_{matrix_type}"
        save_plot(base_filename, format=OUTPUT_FORMAT)
        plt.close()

def main():
    print("Simple Beta Ratio Plotting")
    print("=" * 30)

    # Load data
    df = load_data()
    if df.empty:
        return

    # Check required columns (including new p10 percentiles)
    required_cols = ['E_{AB}/beta', 'E_{AB}/u', '|C-C_ref|/(|A||B|)_avg', '|C-C_ref|/(|A||B|)_p10', '|C-C_ref|/(|A||B|)_p95', 'kernel_type', 'matrix_type', 'matrix_size']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return

    # Get actual matrix sizes from the data (used by both plot functions)
    actual_matrix_sizes = sorted(df['matrix_size'].unique())
    print(f"Actual matrix sizes tested: {actual_matrix_sizes}")

    # Filter matrix types if specified
    if MATRIX_TYPES is not None:
        available_types = sorted(df['matrix_type'].unique())
        print(f"Available matrix types: {available_types}")
        print(f"Filtering to selected matrix types: {MATRIX_TYPES}")
        df = df[df['matrix_type'].isin(MATRIX_TYPES)]
        if df.empty:
            print("No data found for selected matrix types!")
            return
        filtered_types = sorted(df['matrix_type'].unique())
        print(f"Processing matrix types: {filtered_types}")
    else:
        print(f"Processing all matrix types: {sorted(df['matrix_type'].unique())}")

    # Create plots
    create_beta_plots(df, actual_matrix_sizes)

    # Create relative-spread companion plots
    create_relative_spread_plots(df, actual_matrix_sizes)

    # Create effective depth plot for all kernels together (with multiple theoretical lines)
    all_theories = [
        {'func': beta_over_u_splitk_pairwise, 'params': {'S': 16}, 'label': 'β/u SplitK Pairwise (theory)', 'color': 'red', 'linestyle': '--'},
        {'func': beta_over_u_tiled_pairwise, 'params': {'TILE_K': 32}, 'label': 'β/u Tiled Pairwise (theory)', 'color': 'green', 'linestyle': ':'},
        {'func': beta_over_u_splitk_flat, 'params': {'S': 16}, 'label': 'β/u SplitK Flat (theory)', 'color': 'red', 'linestyle': '-.'},
        {'func': beta_over_u_tiled_flat, 'params': {'TILE_K': 32}, 'label': 'β/u Tiled Flat (theory)', 'color': 'blue', 'linestyle': '-'}  # Blue because cuBLAS and tiled match this
    ]

    create_deff_plot(df, actual_matrix_sizes,
                    lambda k: True,  # Include all kernels
                    "all_kernels",
                    multiple_theories=all_theories)

    # Create calibration comparison plots (experimental vs predicted E/u)
    # Flat algorithms: cuBLAS and tiled (both use tiled flat pattern)
    flat_kernels = ['cublas', 'tiled']
    create_calibration_comparison_plots(df, actual_matrix_sizes, flat_kernels, "calibration_flat")

    # Pairwise algorithms: cutlass_splitk_pairwise and tiled_pairwise
    pairwise_kernels = ['cutlass_splitk_pairwise', 'tiled_pairwise']
    create_calibration_comparison_plots(df, actual_matrix_sizes, pairwise_kernels, "calibration_pairwise")

    print("\n✓ All plots created successfully!")
    print("Check plots/ directory for:")
    print("  - E_AB_over_u_*.eps/.png (5 files each format)")
    print("  - E_AB_over_u_relative_spread_*.eps/.png (5 files each format - relative-spread analysis)")
    print("  - E_AB_over_beta_*.eps/.png (5 files each format)")
    print("  - E_AB_*.eps/.png (5 files each format)")
    print("  - Deff_all_kernels_*.eps/.png (effective depth plots - all kernels combined)")
    print("  - calibration_flat_*.eps/.png (experimental vs predicted E/u for flat algorithms)")
    print("  - calibration_pairwise_*.eps/.png (experimental vs predicted E/u for pairwise algorithms)")

if __name__ == "__main__":
    main()
