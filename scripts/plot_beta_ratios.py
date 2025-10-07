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
DATA_FOLDER = "data/data9_17"  # Change this to "data" for current data
OUTPUT_FORMAT = "eps"  # "eps", "png", or "both"

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
THEORETICAL_LINE_WIDTH = 1.5  # Width of theoretical/reference lines

# Font size configuration - adjust these for your report
AXIS_LABEL_FONTSIZE = 18   # Size for axis labels (xlabel, ylabel)
TICK_LABEL_FONTSIZE = 16   # Size for tick labels (numbers on axes)
LEGEND_FONTSIZE = 14       # Size for legend text
TITLE_FONTSIZE = 18        # Size for plot titles
DEFAULT_FONTSIZE = 14      # Default matplotlib font size

# Font weight configuration - set to True for bold, False for normal
AXIS_LABEL_BOLD = False     # Make axis labels bold
TICK_LABEL_BOLD = False    # Make tick labels bold

# Calibration configuration - specific matrix sizes to use for c_hat calibration
CUTLASS_CALIBRATION_SIZES = [4096]  # Only use these sizes for CUTLASS kernel calibration

# Deff plot configuration - select ONE kernel to plot
DEFF_SELECT_KERNEL = 'cublas'  # Which kernel to plot: 'cublas', 'tiled', 'cutlass_splitk_flat', 'cutlass_splitk_pairwise', 'tiled_pairwise'
DEFF_PLOT_ALL_KERNELS = True # Set to True to generate Deff plots for ALL kernels in addition to the selected one

KERNEL_LABELS = {
    'cublas': 'cuBLAS',
    'cutlass_splitk_flat': 'CUTLASS Split-K Flat',
    'cutlass_splitk_pairwise': 'CUTLASS Split-K Pairwise',
    'tiled': 'Tiled (Ours)',
    'tiled_pairwise': 'Tiled Pairwise (Ours)'
}

# Enable better mathematical notation using matplotlib's mathtext (no LaTeX required)
plt.rcParams['font.size'] = DEFAULT_FONTSIZE

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

        axis_weight = 'bold' if AXIS_LABEL_BOLD else 'normal'
        # Create conditional mathtext labels based on bold setting
        y_label_text = r'$\mathbf{E_{AB} / u}$' if AXIS_LABEL_BOLD else r'$E_{AB} / u$'

        plt.xlabel('N - inner matrix dimension', fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.ylabel(y_label_text, fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Configure tick labels
        plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        # Apply bold formatting to tick labels if requested
        if TICK_LABEL_BOLD:
            ax = plt.gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_weight('bold')

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

        axis_weight = 'bold' if AXIS_LABEL_BOLD else 'normal'
        # Create conditional mathtext labels based on bold setting
        y_label_text = r'$\mathbf{E_{AB} / \beta}$' if AXIS_LABEL_BOLD else r'$E_{AB} / \beta$'

        plt.xlabel('N - inner matrix dimension', fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.ylabel(y_label_text, fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Configure tick labels
        plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        # Apply bold formatting to tick labels if requested
        if TICK_LABEL_BOLD:
            ax = plt.gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_weight('bold')

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

        axis_weight = 'bold' if AXIS_LABEL_BOLD else 'normal'
        # Create conditional mathtext labels based on bold setting
        y_label_text = r'$\mathbf{E_{AB}}$' if AXIS_LABEL_BOLD else r'$E_{AB}$'

        plt.xlabel('N - inner matrix dimension', fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.ylabel(y_label_text, fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Configure tick labels
        plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        # Apply bold formatting to tick labels if requested
        if TICK_LABEL_BOLD:
            ax = plt.gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_weight('bold')

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

def calibrate_c_hat(log_c_hat_median_values, matrix_sizes=None, target_sizes=None):
    """
    Input: array of log_c_hat_median values for a single kernel across all matrix sizes
           matrix_sizes: corresponding matrix sizes (optional)
           target_sizes: specific sizes to use for calibration (optional)
    Returns: scalar c_hat as exp(median(log_c_hat_median_values))
    """
    arr = np.asarray(log_c_hat_median_values, dtype=np.float64)

    # If target sizes are specified and matrix_sizes are provided, filter the data
    if target_sizes is not None and matrix_sizes is not None:
        matrix_sizes = np.asarray(matrix_sizes)
        mask = np.isin(matrix_sizes, target_sizes)
        if np.any(mask):
            arr = arr[mask]
            print(f"  Using calibration sizes {target_sizes}, found {np.sum(mask)} matching data points")
        else:
            print(f"  Warning: No data found for target sizes {target_sizes}, using all data")

    median_log_c_hat = float(np.median(arr))
    return float(np.exp(median_log_c_hat))

def predict_E_over_u(beta_over_u, c_hat):
    """Elementwise predicted E/u = c_hat * (beta/u)."""
    print(f"Predicting E/u with c_hat: {c_hat}")
    return c_hat * np.asarray(beta_over_u, dtype=np.float64)

def create_single_kernel_deff_plot(df, actual_matrix_sizes, kernel_name):
    """
    Create effective depth plot for ONE specific kernel with appropriate theoretical line.

    Args:
        df: DataFrame with the data
        actual_matrix_sizes: List of matrix sizes from data
        kernel_name: Name of the single kernel to plot
    """

    os.makedirs("plots", exist_ok=True)

    # Check if kernel exists in data
    available_kernels = sorted(df['kernel_type'].unique())
    if kernel_name not in available_kernels:
        print(f"Kernel '{kernel_name}' not found in data. Available: {available_kernels}")
        return

    print(f"Creating Deff plot for single kernel: {kernel_name}")

    # Get theoretical function for this kernel
    theoretical_functions = {
        'cublas': beta_over_u_tiled_flat,
        'tiled': beta_over_u_tiled_flat,
        'cutlass_splitk_flat': beta_over_u_splitk_flat,
        'cutlass_splitk_pairwise': beta_over_u_splitk_pairwise,
        'tiled_pairwise': beta_over_u_tiled_pairwise
    }

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

        # Get data for this kernel and matrix type
        subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel_name)]
        if len(subset) == 0:
            print(f"No data found for {kernel_name} - {matrix_type}")
            plt.close()
            continue

        subset = subset.sort_values('matrix_size')

        # Compute effective depth from experimental data
        E_over_u = subset['E_{AB}/u'].values
        E_over_beta = subset['E_{AB}/beta'].values
        deff_experimental = effective_depth(E_over_u, E_over_beta)

        # Plot experimental effective depth
        color = KERNEL_COLORS[kernel_name]
        marker = KERNEL_MARKERS[kernel_name]
        plt.plot(subset['matrix_size'], deff_experimental,
                f'{marker}', color=color,
                markersize=MARKER_SIZE, fillstyle='none')

        all_y_values = list(deff_experimental)

        # Add theoretical line if available
        if kernel_name in theoretical_functions:
            K_grid = np.array(actual_matrix_sizes, dtype=np.int64)
            theoretical_func = theoretical_functions[kernel_name]
            params = theoretical_params[kernel_name]
            theoretical_values = theoretical_func(K_grid, **params)

            # Determine theory line style and color
            if 'pairwise' in kernel_name:
                theory_label = 'β/u pairwise (theory)'
            else:
                theory_label = 'β/u flat (theory)'

            plt.plot(K_grid, theoretical_values, '--',
                    color=color, alpha=0.8, linewidth=THEORETICAL_LINE_WIDTH)

            all_y_values.extend(theoretical_values)

        # Format the plot
        axis_weight = 'bold' if AXIS_LABEL_BOLD else 'normal'
        y_label_text = r'$\mathbf{D_{eff}}$' if AXIS_LABEL_BOLD else r'$D_{eff}$'

        plt.xlabel('N - inner matrix dimension', fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.ylabel(y_label_text, fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Configure tick labels
        plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        if TICK_LABEL_BOLD:
            ax = plt.gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_weight('bold')

        # Set x-axis to show only powers of 2
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        plt.xticks(tick_positions, tick_labels)

        # Set y-axis bounds proportional to data
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:
                margin = 0.2
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        base_filename = f"plots/Deff_{kernel_name}_{matrix_type}"
        save_plot(base_filename, format=OUTPUT_FORMAT)
        plt.close()

        print(f"Created: Deff_{kernel_name}_{matrix_type}.{OUTPUT_FORMAT}")

def deff_all_kernels(df, actual_matrix_sizes):
    """
    Create effective depth plots with ALL available kernels plotted together in one plot.

    Args:
        df: DataFrame with the data
        actual_matrix_sizes: List of matrix sizes from data
    """
    available_kernels = sorted(df['kernel_type'].unique())
    print(f"Creating combined Deff plot for all kernels: {available_kernels}")

    os.makedirs("plots", exist_ok=True)

    # Get theoretical functions and parameters
    theoretical_functions = {
        'cublas': beta_over_u_tiled_flat,
        'tiled': beta_over_u_tiled_flat,
        'cutlass_splitk_flat': beta_over_u_splitk_flat,
        'cutlass_splitk_pairwise': beta_over_u_splitk_pairwise,
        'tiled_pairwise': beta_over_u_tiled_pairwise
    }

    theoretical_params = {
        'cublas': {'TILE_K': 32},
        'tiled': {'TILE_K': 32},
        'cutlass_splitk_flat': {'S': 16},
        'cutlass_splitk_pairwise': {'S': 16},
        'tiled_pairwise': {'TILE_K': 32}
    }

    # Get unique matrix types
    matrix_types = sorted(df['matrix_type'].unique())

    # Create one combined plot per matrix type
    for matrix_type in matrix_types:
        plt.figure(figsize=(12, 8))

        all_y_values = []

        # Plot each kernel
        for kernel_name in available_kernels:
            # Get data for this kernel and matrix type
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel_name)]
            if len(subset) == 0:
                print(f"No data found for {kernel_name} - {matrix_type}")
                continue

            subset = subset.sort_values('matrix_size')

            # Compute effective depth from experimental data
            E_over_u = subset['E_{AB}/u'].values
            E_over_beta = subset['E_{AB}/beta'].values
            deff_experimental = effective_depth(E_over_u, E_over_beta)

            # Plot experimental effective depth
            color = KERNEL_COLORS[kernel_name]
            marker = KERNEL_MARKERS[kernel_name]
            plt.plot(subset['matrix_size'], deff_experimental,
                    f'{marker}', color=color,
                    markersize=MARKER_SIZE, fillstyle='none')

            all_y_values.extend(deff_experimental)

            # Add theoretical line if available
            if kernel_name in theoretical_functions:
                K_grid = np.array(actual_matrix_sizes, dtype=np.int64)
                theoretical_func = theoretical_functions[kernel_name]
                params = theoretical_params[kernel_name]
                theoretical_values = theoretical_func(K_grid, **params)

                plt.plot(K_grid, theoretical_values, '--',
                        color=color, alpha=0.8, linewidth=THEORETICAL_LINE_WIDTH)

                all_y_values.extend(theoretical_values)

        # Format the plot
        axis_weight = 'bold' if AXIS_LABEL_BOLD else 'normal'
        y_label_text = r'$\mathbf{D_{eff}}$' if AXIS_LABEL_BOLD else r'$D_{eff}$'

        plt.xlabel('N - inner matrix dimension', fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.ylabel(y_label_text, fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Configure tick labels
        plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        if TICK_LABEL_BOLD:
            ax = plt.gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_weight('bold')

        # Set x-axis to show only powers of 2
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        plt.xticks(tick_positions, tick_labels)

        # Set y-axis bounds proportional to data
        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max / y_min if y_min > 0 else y_max - y_min
            if y_range > 1:
                margin = 0.2
                plt.ylim(y_min * (1 - margin), y_max * (1 + margin))

        # Save plot
        base_filename = f"plots/Deff_all_kernels_{matrix_type}"
        save_plot(base_filename, format=OUTPUT_FORMAT)
        plt.close()

        print(f"Created: Deff_all_kernels_{matrix_type}.{OUTPUT_FORMAT}")

    print(f"\n✓ Completed combined Deff plot for {len(available_kernels)} kernels")

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

        plt.xlabel('N - inner matrix dimension', fontsize=AXIS_LABEL_FONTSIZE, weight='bold')
        plt.ylabel(r'$\mathbf{E_{AB} / u}$', fontsize=AXIS_LABEL_FONTSIZE, weight='bold')
        plt.title(f'E_{{AB}}/u with Uncertainty Bands - {matrix_type}', fontsize=TITLE_FONTSIZE, weight='bold')
        plt.legend(frameon=False, ncol=2, fontsize=LEGEND_FONTSIZE, loc='best')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

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

def create_spread_cloud_plots(df, actual_matrix_sizes):
    """Create standalone spread cloud plots showing relative deviation from mean as percentages."""

    os.makedirs("plots", exist_ok=True)

    # Get unique values
    matrix_types = sorted(df['matrix_type'].unique())
    kernels = sorted(df['kernel_type'].unique())

    print(f"Creating standalone spread cloud plots for kernels: {kernels}")

    # Create one plot per matrix type
    for matrix_type in matrix_types:
        plt.figure(figsize=(10, 6))

        all_relative_values = []

        for kernel in kernels:
            subset = df[(df['matrix_type'] == matrix_type) & (df['kernel_type'] == kernel)]
            if len(subset) > 0:
                subset = subset.sort_values('matrix_size')

                # Get mean values for E/u ratios
                mean_values = subset['E_{AB}/u'].values

                u32 = 2.0**-24  # unit roundoff
                p10_eu = subset['|C-C_ref|/(|A||B|)_p10'].values / u32
                p95_eu = subset['|C-C_ref|/(|A||B|)_p95'].values / u32

                # Compute relative deviations as percentages
                p10_relative = (p10_eu / mean_values - 1) * 100  # (p10/mean - 1) × 100%
                p95_relative = (p95_eu / mean_values - 1) * 100  # (p95/mean - 1) × 100%

                all_relative_values.extend(p10_relative)
                all_relative_values.extend(p95_relative)

                color = KERNEL_COLORS[kernel]

                # Create spread cloud plot
                plt.fill_between(subset['matrix_size'], p10_relative, p95_relative,
                               alpha=0.7, color=color)

        # Format the plot
        axis_weight = 'bold' if AXIS_LABEL_BOLD else 'normal'
        plt.xlabel('N - inner matrix dimension', fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.ylabel('Deviation from mean (%) - P10 to P95 spread', fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        plt.grid(True, alpha=0.3)

        # Configure tick labels
        plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        # Apply bold formatting to tick labels if requested
        if TICK_LABEL_BOLD:
            ax = plt.gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_weight('bold')

        # Set x-axis to show only powers of 2
        tick_positions, tick_labels = format_matrix_size_labels(actual_matrix_sizes)
        plt.xticks(tick_positions, tick_labels)

        # Set y-axis bounds (symmetric around 0)
        if all_relative_values:
            max_abs_rel = max(abs(min(all_relative_values)), abs(max(all_relative_values)))
            margin = max_abs_rel * 0.1  # 10% margin
            plt.ylim(-max_abs_rel - margin, max_abs_rel + margin)

        # Save plot
        base_filename = f"plots/spread_cloud_{matrix_type}"
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
            log_c_hat_median_values = subset['log_c_hat_median'].values

            # Calibrate c_hat using log_c_hat_median values for this kernel
            # Use specific sizes for CUTLASS kernels
            if 'cutlass' in kernel.lower():
                target_sizes = CUTLASS_CALIBRATION_SIZES
                print(f"Using specific calibration sizes for {kernel}: {target_sizes}")
            else:
                target_sizes = None

            c_hat = calibrate_c_hat(log_c_hat_median_values,
                                  matrix_sizes=subset['matrix_size'].values,
                                  target_sizes=target_sizes)
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

        axis_weight = 'bold' if AXIS_LABEL_BOLD else 'normal'
        # Create conditional mathtext labels based on bold setting
        y_label_text = r'$\mathbf{E_{AB} / u}$' if AXIS_LABEL_BOLD else r'$E_{AB} / u$'

        plt.xlabel('N - inner matrix dimension', fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.ylabel(y_label_text, fontsize=AXIS_LABEL_FONTSIZE, weight=axis_weight)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # Configure tick labels
        plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        # Apply bold formatting to tick labels if requested
        if TICK_LABEL_BOLD:
            ax = plt.gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_weight('bold')

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

    # Create standalone spread cloud plots
    create_spread_cloud_plots(df, actual_matrix_sizes)

    # Create effective depth plot for selected kernel
    available_kernels = sorted(df['kernel_type'].unique())
    print(f"Available kernels: {available_kernels}")
    print(f"Creating Deff plot for selected kernel: {DEFF_SELECT_KERNEL}")

    create_single_kernel_deff_plot(df, actual_matrix_sizes, DEFF_SELECT_KERNEL)

    # Optionally create Deff plots for all kernels
    if DEFF_PLOT_ALL_KERNELS:
        print(f"DEFF_PLOT_ALL_KERNELS is True - creating plots for all kernels")
        deff_all_kernels(df, actual_matrix_sizes)
    else:
        print(f"DEFF_PLOT_ALL_KERNELS is False - only plotting selected kernel")

    # Create calibration comparison plots (experimental vs predicted E/u)
    # Flat algorithms: cuBLAS, tiled, and CUTLASS flat
    flat_kernels = ['cublas', 'tiled', 'cutlass_splitk_flat']
    create_calibration_comparison_plots(df, actual_matrix_sizes, flat_kernels, "calibration_flat")

    # Pairwise algorithms: cutlass_splitk_pairwise and tiled_pairwise
    pairwise_kernels = ['cutlass_splitk_pairwise', 'tiled_pairwise']
    create_calibration_comparison_plots(df, actual_matrix_sizes, pairwise_kernels, "calibration_pairwise")

    print("\n✓ All plots created successfully!")
    print("Check plots/ directory for:")
    print("  - E_AB_over_u_*.eps/.png (5 files each format)")
    print("  - spread_cloud_*.eps/.png (5 files each format - standalone spread cloud plots)")
    print("  - E_AB_over_beta_*.eps/.png (5 files each format)")
    print("  - E_AB_*.eps/.png (5 files each format)")
    print("  - Deff_all_kernels_*.eps/.png (effective depth plots - all kernels combined)")
    print("  - calibration_flat_*.eps/.png (experimental vs predicted E/u for flat algorithms)")
    print("  - calibration_pairwise_*.eps/.png (experimental vs predicted E/u for pairwise algorithms)")

if __name__ == "__main__":
    main()
