import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the precision folders you want to compare
precision_folders = [
    #"data/UC_FP16_UA_FP16",
    "data/UC_FP32_UA_FP32",
    "data/UC_UA_FP32"

]

kernels = ["tiled_mixprec", "tiled_pairwise_mixprec"]
matrix_sizes = [256, 512, 1024, 1536, 2048, 3072, 4096]
select_matrix = "uniform_positive"

OUTPUT_FORMAT = "png"  # "eps", "png", or "both"
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

# Marker and color mapping (adjust as needed)
MARKER_MAP = {
    "tiled_mixprec": "o",
    "tiled_pairwise_mixprec": "s",
}
COLOR_MAP = {
    "UC_UA_FP32": "blue",
    "UC_FP32_UA_FP32": "black",
    "UC_FP16_UA_FP32": "red",
    "UC_FP16_UA_FP16": "green",
}

LINESTYLE_MAP = {
    "UC_FP32_UA_FP32": "-",      # Solid line
    "UC_UA_FP32": "-",           # Solid line
    "UC_FP16_UA_FP32": "--",     # Dashed line
    "UC_FP16_UA_FP16": "-.",     # Dash-dot line
}

# Collect data
data = {}
for folder in precision_folders:
    label = os.path.basename(folder)
    data[label] = {}
    for kernel in kernels:
        kernel_data = []
        for size in matrix_sizes:
            csv_path = f"{folder}/error_analysis_{kernel}_{select_matrix}_n{size}.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Assume your CSV has columns: 'gflops', 'error_metric'
                kernel_data.append({
                    "size": size,
                    "error": df['|C-C_ref|/(|A||B|)_avg'].iloc[0]  # or your error metric
                })
        data[label][kernel] = pd.DataFrame(kernel_data)

plt.figure(figsize=(10, 7))
for label, kernels_data in data.items():
    for kernel, df in kernels_data.items():
        if not df.empty:
            plt.plot(
                df['size'], df['error'],
                marker=MARKER_MAP.get(kernel, 'o'),
                color=COLOR_MAP.get(label, 'black'),
                linestyle=LINESTYLE_MAP.get(label, '-'),
                label=f"{kernel.replace('_', ' ').title()} ({label.replace('_', ' ')})",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
            )

plt.xlabel("Matrix Size (N)",
           fontsize=AXIS_LABEL_FONTSIZE,
           weight='bold' if AXIS_LABEL_BOLD else 'normal')
plt.ylabel(r"$|C - C_{ref}|/(|A||B|)_{avg}$",
           fontsize=AXIS_LABEL_FONTSIZE,
           weight='bold' if AXIS_LABEL_BOLD else 'normal')
plt.yscale('log')
plt.xticks(matrix_sizes, [str(s) for s in matrix_sizes], fontsize=TICK_LABEL_FONTSIZE)
plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.title("Multiprecision Error Analysis", fontsize=TITLE_FONTSIZE)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the figure
output_path = f"plots/multiprec_error_analysis.{OUTPUT_FORMAT}"
if OUTPUT_FORMAT == "both":
    plt.savefig("plots/multiprec_error.png", dpi=300, bbox_inches='tight')
    plt.savefig("plots/multiprec_error.eps", dpi=300, bbox_inches='tight')
else:
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()