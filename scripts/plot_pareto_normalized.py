#!/usr/bin/env python3
# filepath: /home/oana/Projects/GEMMProfiling/scripts/plot_pareto_normalized.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_config import *  # Import all configuration

# Configuration
# =============================================================================

# Data folder and kernels to analyze
DATA_FOLDER = "data/UC_UA_FP32"
DEVICE_PROPS_FILE = "data/deviceprops.csv"
KERNELS = ["tiled", "tiled_pairwise"]
MATRIX_SIZES = [256, 512, 1024, 1536, 2048, 3072, 4096]
SELECT_MATRIX = "uniform_positive"

# Marker and color mapping
PARETO_MARKER_MAP = {
    "tiled": "o",
    "tiled_pairwise": "s",
}
PARETO_COLOR_MAP = {
    "tiled": "blue",
    "tiled_pairwise": "red",
}

# =============================================================================
# Load Device Properties
# =============================================================================

def load_device_properties():
    """Load device properties from CSV file."""
    if not os.path.exists(DEVICE_PROPS_FILE):
        print(f"Warning: Device properties file not found: {DEVICE_PROPS_FILE}")
        print("Run: ./main --assess-resources --test=tiled --size=1024")
        return None

    # Read the CSV with property,value format
    df = pd.read_csv(DEVICE_PROPS_FILE)

    # Convert to dictionary for easy access
    props = {}
    for _, row in df.iterrows():
        props[row['property']] = row['value']

    print(f"Device: {props.get('device_name', 'Unknown')}")
    print(f"Peak Bandwidth: {props.get('peak_bandwidth_gb_s', 'N/A')} GB/s")
    print(f"Peak GFLOPS (FP32): {props.get('peak_gflops_fp32', 'N/A')}")
    print(f"Arithmetic Intensity Ridge: {props.get('arithmetic_intensity_ridge', 'N/A')} FLOP/byte")
    print()

    return props

def compute_roofline_metrics(df, device_props, bytes_per_elem=4):
    """
    Compute arithmetic intensity, roofline limit, and efficiency for each kernel run.

    """
    if device_props is None:
        print("Warning: No device properties available for roofline analysis")
        df['AI'] = None
        df['roofline_limit'] = None
        df['efficiency'] = None
        return df

    peak_bandwidth = float(device_props.get('peak_bandwidth_gb_s', 0))
    #peak_gflops = float(device_props.get('peak_gflops_fp32', 0))
    peak_gflops=33000.0  # Temporary override for testing

    # === Compute AI (FLOPs per byte) ===
    # For square GEMM: C = A * B where all are n×n
    # FLOPs = 2 * n^3 (multiply-add for each of n^2 output elements, each needing n ops)
    # Bytes transferred = 3 * n^2 * bytes_per_elem (read A, B, write C)
    # AI = (2 * n^3) / (3 * n^2 * bytes_per_elem) = (2 * n) / (3 * bytes_per_elem)

    df['AI'] = (2.0 * df['size']) / (3.0 * bytes_per_elem)

    # === Roofline limit ===
    # Performance is limited by either:
    # 1. Compute bound: peak_gflops
    # 2. Memory bound: peak_bandwidth * AI
    # The roofline is the minimum of these two
    df['roofline_limit'] = df['AI'].apply(
        lambda ai: min(peak_gflops, peak_bandwidth * ai)
    )

    # === Efficiency ===
    # How close are we to the roofline limit?
    df['efficiency'] = df['gflops'] / df['roofline_limit']

    return df

# =============================================================================
# Data Collection
# =============================================================================

# Load device properties
device_props = load_device_properties()

data = {}
for kernel in KERNELS:
    kernel_data = []
    for size in MATRIX_SIZES:
        # Read error data
        error_file = f"{DATA_FOLDER}/error_analysis_{kernel}_{SELECT_MATRIX}_n{size}.csv"
        # Read performance data
        perf_file = f"{DATA_FOLDER}/perf_{kernel}_{size}_FP32.csv"

        if os.path.exists(error_file) and os.path.exists(perf_file):
            error_df = pd.read_csv(error_file)
            perf_df = pd.read_csv(perf_file)

            kernel_data.append({
                "size": size,
                "EAB": error_df['|C-C_ref|/(|A||B|)_avg'].iloc[0],
                "EAB_uc": error_df['E_{AB}/u_c'].iloc[0],
                "gflops": perf_df['gflops'].iloc[0]
            })

    df = pd.DataFrame(kernel_data)  # ← FIX: Create df here

    # Compute roofline metrics
    if not df.empty:
        df = compute_roofline_metrics(df, device_props, bytes_per_elem=4)  # FP32 = 4 bytes
        print(f"\n{kernel} - Roofline Analysis:")
        print(df[['size', 'gflops', 'AI', 'roofline_limit', 'efficiency']].to_string(index=False))

    data[kernel] = df

# ===========================================================
# Plotting: EAB_uc vs Efficiency (Pareto Frontier)
# ===========================================================

plt.figure(figsize=FIGURE_SIZE)

# Find the reference peak (best observed GFLOPS across all kernels)
ref_peak = 0.0
for kernel, df in data.items():
    if not df.empty:
        ref_peak = max(ref_peak, df['gflops'].max())

print(f"Reference peak GFLOPS (best observed): {ref_peak:.2f}")
print()

# Add efficiency column and sort by it
for kernel, df in data.items():
    if not df.empty:
        df['rel_eff'] = df['gflops'] / ref_peak
        df.sort_values('rel_eff', inplace=True)  # Sort by relative efficiency

        print(f"{kernel} - Performance Analysis:")
        print(df[['size', 'gflops', 'rel_eff', 'EAB_uc']].to_string(index=False))
        print()

for kernel, df in data.items():
    if not df.empty:
        plt.scatter(
            df['rel_eff'], df['EAB_uc'],  # ← Changed from gflops, error
            marker=PARETO_MARKER_MAP.get(kernel, 'o'),
            color=PARETO_COLOR_MAP.get(kernel, 'black'),
            s=MARKER_SIZE**2,
            label=kernel.replace('_', ' ').title(),
            alpha=0.7
        )

        # Connect points with lines
        plt.plot(
            df['rel_eff'], df['EAB_uc'],  # ← Changed from gflops, error
            color=PARETO_COLOR_MAP.get(kernel, 'black'),
            linewidth=LINE_WIDTH,
            alpha=0.5
        )

# Add 100% efficiency reference line
plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5,
            label='100% Efficiency')

ax = plt.gca()

ax.set_xlabel("Efficiency (Relative to Peak GFLOPS)",
           fontsize=AXIS_LABEL_FONTSIZE,
           weight='bold' if AXIS_LABEL_BOLD else 'normal')
ax.set_ylabel(r"Normalized Error: $E_{AB}/u_c$",
           fontsize=AXIS_LABEL_FONTSIZE,
           weight='bold' if AXIS_LABEL_BOLD else 'normal')

ax.set_yscale('log')
ax.set_xlim(0.5, 1.05)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
apply_tick_label_bold(ax)

ax.set_title("Pareto Frontier: Error vs Efficiency", fontsize=TITLE_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE)
ax.grid(True, which='both', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)

# Save the figure
os.makedirs("plots", exist_ok=True)
save_plot("plots/pareto_error_vs_efficiency")

print(f"\n✓ Pareto plot saved")
print("\nData summary:")
for kernel, df in data.items():
    if not df.empty:
        print(f"\n{kernel}:")
        print(f"  Sizes: {df['size'].tolist()}")
        print(f"  Efficiency range: {df['efficiency'].min():.3f} - {df['efficiency'].max():.3f}")
        print(f"  EAB/uc range: {df['EAB_uc'].min():.2e} - {df['EAB_uc'].max():.2e}")

# Add this section after the data summary and before plt.show():

print("\n" + "="*70)
print("Performance Comparison: Tiled vs Tiled Pairwise")
print("="*70)

# Create comparison table
if 'tiled' in data and 'tiled_pairwise' in data:
    df_tiled = data['tiled']
    df_pairwise = data['tiled_pairwise']

    if not df_tiled.empty and not df_pairwise.empty:
        # Merge on size
        comparison = pd.merge(
            df_tiled[['size', 'gflops']],
            df_pairwise[['size', 'gflops']],
            on='size',
            suffixes=('_tiled', '_pairwise')
        )

        # Compute ratio and slowdown percentage
        comparison['ratio'] = comparison['gflops_pairwise'] / comparison['gflops_tiled']
        comparison['slower_%'] = (1 - comparison['ratio']) * 100

        # Rename columns for clarity
        comparison.rename(columns={
            'gflops_tiled': 'flat_gflops',
            'gflops_pairwise': 'pairwise_gflops'
        }, inplace=True)

        # Print formatted table
        print("\n{:<10} {:<15} {:<18} {:<10} {:<12}".format(
            "Size", "Flat (GFLOPS)", "Pairwise (GFLOPS)", "Ratio", "Slower (%)"
        ))
        print("-" * 70)

        for _, row in comparison.iterrows():
            print("{:<10} {:<15.2f} {:<18.2f} {:<10.3f} {:<12.1f}".format(
                row['size'],
                row['flat_gflops'],
                row['pairwise_gflops'],
                row['ratio'],
                row['slower_%']
            ))

        print("-" * 70)
        print(f"Average ratio: {comparison['ratio'].mean():.3f}")
        print(f"Average slowdown: {comparison['slower_%'].mean():.1f}%")
        print()
    else:
        print("Warning: One or both kernel datasets are empty")
else:
    print("Warning: Could not find both 'tiled' and 'tiled_pairwise' in data")

plt.show()