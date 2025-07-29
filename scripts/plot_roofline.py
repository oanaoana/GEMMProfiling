# plot_roofline.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the data
if not os.path.exists('roofline_data.csv'):
    print("Error: roofline_data.csv not found!")
    print("Run ./main first to generate the data.")
    exit(1)

df = pd.read_csv('roofline_data.csv')

# Get peak values from device info (adjust based on your GPU)
peak_compute = 49.03  # TFLOP/s
peak_bandwidth = 716.86  # GB/s
ridge_point = peak_compute / peak_bandwidth

# Convert GFLOP/s to TFLOP/s for plotting
df['TFLOP/s'] = df['gflops'] / 1000.0

# Calculate efficiency percentages
df['compute_efficiency_%'] = (df['TFLOP/s'] / peak_compute) * 100
df['bandwidth_efficiency_%'] = (df['bandwidth_gb'] / peak_bandwidth) * 100

# Create the plot
plt.figure(figsize=(12, 9))

# Generate the roofline curve
x = np.logspace(-2, 4, 1000)
y_mem = np.minimum(x * peak_bandwidth, peak_compute)
plt.loglog(x, y_mem, 'k-', linewidth=2, label='Roofline')

# Plot peak compute and ridge point
plt.axhline(y=peak_compute, color='k', linestyle='--',
            label=f'Peak Compute: {peak_compute:.1f} TFLOP/s')
plt.axvline(x=ridge_point, color='k', linestyle=':',
            label=f'Ridge Point: {ridge_point:.1f} FLOP/byte')

# Plot data points by algorithm
algorithms = df['algorithm'].unique()
markers = ['o', 's', '^', 'd', '*', 'x', 'p']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

for idx, algo in enumerate(algorithms):
    data = df[df['algorithm'] == algo]
    plt.scatter(data['arithmetic_intensity'], data['TFLOP/s'],
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                s=100, alpha=0.8, label=algo)

    # Annotate points with matrix size and efficiency
    for _, row in data.iterrows():
        plt.annotate(f"{int(row['size'])} ({row['compute_efficiency_%']:.0f}%)",
                     (row['arithmetic_intensity'], row['TFLOP/s']),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha='left',
                     fontsize=8)

# Format plot
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
plt.ylabel('Performance (TFLOP/s)', fontsize=12)
plt.title('Roofline Model: GEMM Performance Analysis', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()

# Add a table with summary information
plt.figtext(0.5, 0.01, 'Summary of Maximum Performance:',
            ha='center', fontsize=10, fontweight='bold')

# Save and show
plt.savefig('roofline_model.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
print("\nPerformance Summary:")
print("=" * 80)
print(f"{'Algorithm':<15} {'Size':<8} {'Time (ms)':<10} {'TFLOP/s':<10} {'GB/s':<10} {'Compute Eff.':<12} {'Bandwidth Eff.':<15}")
print("-" * 80)

for algo in algorithms:
    # Get the row with maximum FLOP/s for each algorithm
    best = df[df['algorithm'] == algo].loc[df[df['algorithm'] == algo]['gflops'].idxmax()]
    print(f"{algo:<15} {int(best['size']):<8} {best['time_ms']:<10.2f} {best['TFLOP/s']:<10.2f} "
          f"{best['bandwidth_gb']:<10.2f} {best['compute_efficiency_%']:<12.2f}% "
          f"{best['bandwidth_efficiency_%']:<15.2f}%")

print("=" * 80)