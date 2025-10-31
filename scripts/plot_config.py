"""
Shared plotting configuration for all GEMMProfiling plotting scripts.
This ensures consistent styling across all plots.
"""

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
OUTPUT_FORMAT = "png"  # "eps", "png", or "both"

# ============================================================================
# LINE AND MARKER STYLING
# ============================================================================
LINE_WIDTH = 1.5          # Width of plot lines
MARKER_SIZE = 10          # Size of markers
THEORETICAL_LINE_WIDTH = 1.5  # Width of theoretical/reference lines

# ============================================================================
# FONT SIZE CONFIGURATION
# ============================================================================
AXIS_LABEL_FONTSIZE = 18   # Size for axis labels (xlabel, ylabel)
TICK_LABEL_FONTSIZE = 18   # Size for tick labels (numbers on axes)
LEGEND_FONTSIZE = 14       # Size for legend text
TITLE_FONTSIZE = 18        # Size for plot titles
DEFAULT_FONTSIZE = 14      # Default matplotlib font size

# ============================================================================
# FONT WEIGHT CONFIGURATION
# ============================================================================
AXIS_LABEL_BOLD = False    # Make axis labels bold
TICK_LABEL_BOLD = False    # Make tick labels bold

# ============================================================================
# FIGURE SIZE
# ============================================================================
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 6
FIGURE_SIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)

# ============================================================================
# GRID CONFIGURATION
# ============================================================================
GRID_ALPHA = 0.3           # Grid transparency
GRID_LINESTYLE = '--'      # Grid line style

# ============================================================================
# SAVE CONFIGURATION
# ============================================================================
SAVE_DPI = 300             # DPI for saved figures
SAVE_BBOX = 'tight'        # Bounding box for saved figures

# ============================================================================
# COLOR SCHEMES - Beta Ratios (Single Precision)
# ============================================================================
# Color scheme: 3 colors for 3 kernel families
KERNEL_COLORS = {
    'cublas': 'blue',
    'cutlass_splitk_flat': 'red',
    'cutlass_splitk_pairwise': 'red',
    'tiled': 'green',
    'tiled_pairwise': 'green'
}

# Marker scheme: squares for pairwise, circles for others
KERNEL_MARKERS = {
    'cublas': 'o',
    'cutlass_splitk_flat': 'o',
    'cutlass_splitk_pairwise': 's',
    'tiled': 'o',
    'tiled_pairwise': 's'
}

# Kernel labels for legends
KERNEL_LABELS = {
    'cublas': 'cuBLAS',
    'cutlass_splitk_flat': 'CUTLASS Split-K Flat',
    'cutlass_splitk_pairwise': 'CUTLASS Split-K Pairwise',
    'tiled': 'Tiled (Ours)',
    'tiled_pairwise': 'Tiled Pairwise (Ours)'
}

# ============================================================================
# COLOR SCHEMES - Mixed Precision
# ============================================================================
MIXPREC_KERNEL_MARKERS = {
    "tiled_mixprec": "o",
    "tiled_pairwise_mixprec": "s",
}

MIXPREC_COLOR_MAP = {
    "UC_FP32_UA_FP32": "green",
    "UC_FP16_UA_FP32": "red",
    "UC_FP16_UA_FP16": "blue",
}

MIXPREC_LINESTYLE_MAP = {
    "UC_FP32_UA_FP32": "-",      # Solid line
    "UC_FP16_UA_FP32": "--",     # Dashed line
    "UC_FP16_UA_FP16": "-.",     # Dash-dot line
}

# ============================================================================
# Y-AXIS LIMITS CONFIGURATION
# ============================================================================
# Y-axis limits for E_AB/u_c plots per matrix type
Y_LIMITS_E_AB_OVER_U = {
    'uniform_positive': (1e-1, 1e+2),
    'wellcond': (1e-2, 1e+1),
    '2powers': (1e-2, 1e+1),
    'illcond': (1e-2, 1e+1),
    'zeromean': (1e-2, 1e+1)
}

# ============================================================================
# CALIBRATION CONFIGURATION
# ============================================================================
# Calibration configuration - specific matrix sizes to use for c_hat calibration
CUTLASS_CALIBRATION_SIZES = [4096]  # Only use these sizes for CUTLASS kernel calibration

# ============================================================================
# DEFF PLOT CONFIGURATION
# ============================================================================
# Deff plot configuration - select ONE kernel to plot
DEFF_SELECT_KERNEL = 'cublas'  # Which kernel to plot: 'cublas', 'tiled', 'cutlass_splitk_flat', 'cutlass_splitk_pairwise', 'tiled_pairwise'
DEFF_PLOT_ALL_KERNELS = True   # Set to True to generate Deff plots for ALL kernels in addition to the selected one

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
import numpy as np

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

def get_axis_label_weight():
    """Get axis label font weight as string."""
    return 'bold' if AXIS_LABEL_BOLD else 'normal'

def apply_tick_label_bold(ax):
    """Apply bold formatting to tick labels if enabled."""
    if TICK_LABEL_BOLD:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_weight('bold')

def save_plot(base_filename, format=None):
    """
    Save plot in specified format(s).

    Args:
        base_filename: Base filename without extension (e.g., "plots/myplot")
        format: Output format - "png", "eps", or "both" (uses OUTPUT_FORMAT if None)
    """
    import matplotlib.pyplot as plt

    fmt = format if format is not None else OUTPUT_FORMAT

    if fmt == "both":
        plt.savefig(f"{base_filename}.png", dpi=SAVE_DPI, bbox_inches=SAVE_BBOX)
        plt.savefig(f"{base_filename}.eps", dpi=SAVE_DPI, bbox_inches=SAVE_BBOX)
    else:
        plt.savefig(f"{base_filename}.{fmt}", dpi=SAVE_DPI, bbox_inches=SAVE_BBOX)