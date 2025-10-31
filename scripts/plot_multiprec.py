import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_config import *  # Import all configuration

# Define the precision folders you want to compare
precision_folders = [
    "data/UC_FP16_UA_FP16",
    "data/UC_FP16_UA_FP32",
    "data/UC_FP32_UA_FP32"
]

kernels = ["tiled_mixprec", "tiled_pairwise_mixprec"]
matrix_sizes = [256, 512, 1024, 1536, 2048, 3072, 4096]
select_matrix = "uniform_positive"

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
                kernel_data.append({
                    "size": size,
                    "error": df['|C-C_ref|/(|A||B|)_avg'].iloc[0],
                    "E_{AB}/u_c": df['E_{AB}/u_c'].iloc[0],
                    "E_{AB}/beta": df['E_{AB}/beta'].iloc[0]
                })
        data[label][kernel] = pd.DataFrame(kernel_data)

# First plot: normalized error
plt.figure(figsize=FIGURE_SIZE)
for label, kernels_data in data.items():
    for kernel, df in kernels_data.items():
        if not df.empty:
            plt.plot(
                df['size'], df['error'],
                marker=MIXPREC_KERNEL_MARKERS.get(kernel, 'o'),
                color=MIXPREC_COLOR_MAP.get(label, 'black'),
                linestyle=MIXPREC_LINESTYLE_MAP.get(label, '-'),
                label=f"{kernel.replace('_', ' ').title()} ({label.replace('_', ' ')})",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
            )

ax = plt.gca()
ax.set_xlabel("k - inner matrix dimension",
              fontsize=AXIS_LABEL_FONTSIZE,
              weight=get_axis_label_weight())
ax.set_ylabel(r"$E_{AB}$",
              fontsize=AXIS_LABEL_FONTSIZE,
              weight=get_axis_label_weight())
ax.set_yscale('log')

tick_positions, tick_labels = format_matrix_size_labels(matrix_sizes)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
apply_tick_label_bold(ax)

ax.legend(fontsize=LEGEND_FONTSIZE)
ax.set_title("Multiprecision Error Analysis", fontsize=TITLE_FONTSIZE)
ax.grid(True, which='both', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)

save_plot("plots/multiprec_E_AB")
plt.close()

# Second plot: E_{AB}/u_c vs matrix size
plt.figure(figsize=FIGURE_SIZE)
for label, kernels_data in data.items():
    for kernel, df in kernels_data.items():
        if not df.empty and 'E_{AB}/u_c' in df.columns:
            plt.plot(
                df['size'], df['E_{AB}/u_c'],
                marker=MIXPREC_KERNEL_MARKERS.get(kernel, 'o'),
                color=MIXPREC_COLOR_MAP.get(label, 'purple'),
                linestyle=MIXPREC_LINESTYLE_MAP.get(label, '-'),
                label=f"{kernel.replace('_', ' ').title()} ({label.replace('_', ' ')})",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
            )

ax = plt.gca()
ax.set_xlabel("k - inner matrix dimension",
              fontsize=AXIS_LABEL_FONTSIZE,
              weight=get_axis_label_weight())
ax.set_ylabel(r"$E_{AB}/u_c$",
              fontsize=AXIS_LABEL_FONTSIZE,
              weight=get_axis_label_weight())
ax.set_yscale('log')

tick_positions, tick_labels = format_matrix_size_labels(matrix_sizes)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
apply_tick_label_bold(ax)

ax.legend(fontsize=LEGEND_FONTSIZE)
ax.set_title("Computational Error normalized by Compute Unit Precision", fontsize=TITLE_FONTSIZE)
ax.grid(True, which='both', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)

save_plot("plots/multiprec_E_AB_over_uc")
plt.close()

# Third plot: E_{AB}/beta vs matrix size
plt.figure(figsize=FIGURE_SIZE)
for label, kernels_data in data.items():
    for kernel, df in kernels_data.items():
        if not df.empty and 'E_{AB}/beta' in df.columns:
            plt.plot(
                df['size'], df['E_{AB}/beta'],
                marker=MIXPREC_KERNEL_MARKERS.get(kernel, 'o'),
                color=MIXPREC_COLOR_MAP.get(label, 'purple'),
                linestyle=MIXPREC_LINESTYLE_MAP.get(label, '-'),
                label=f"{kernel.replace('_', ' ').title()} ({label.replace('_', ' ')})",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
            )

ax = plt.gca()
ax.set_xlabel("k - inner matrix dimension",
              fontsize=AXIS_LABEL_FONTSIZE,
              weight=get_axis_label_weight())
ax.set_ylabel(r"$E_{AB}/\beta$",
              fontsize=AXIS_LABEL_FONTSIZE,
              weight=get_axis_label_weight())
ax.set_yscale('log')

tick_positions, tick_labels = format_matrix_size_labels(matrix_sizes)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
apply_tick_label_bold(ax)

ax.legend(fontsize=LEGEND_FONTSIZE)
ax.set_title("Computational Error normalized by Beta", fontsize=TITLE_FONTSIZE)
ax.grid(True, which='both', linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)

save_plot("plots/multiprec_E_AB_over_beta")
plt.close()

print("\nAll plots generated successfully!")