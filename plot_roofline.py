import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('roofline_data.csv')

# Extract peak performance values (these should match what your CUDA code reports)
# Update these with the values from your GPU
peak_compute = 49.03  # TFLOP/s - from your device info
peak_bandwidth = 716.86  # GB/s - from your device info
ridge_point = peak_compute / peak_bandwidth  # FLOP/byte

# Compute performance and efficiency
df['TFLOP/s'] = df['gflops'] / 1000
df['compute_efficiency_%'] = (df['TFLOP/s'] / peak_compute) * 100


# Create the roofline plot
plt.figure(figsize=(10, 8))

# Generate roofline model line
x = np.logspace(-2, 4, 1000)  # Arithmetic intensity range
y_mem = np.minimum(x * peak_bandwidth, peak_compute)  # Memory-bound or compute-bound

# Plot the roofline
plt.loglog(x, y_mem, 'k-', linewidth=2, label='Roofline')
plt.axhline(y=peak_compute, color='k', linestyle='--', label='Peak Compute: %.1f TFLOP/s' % peak_compute)
plt.axvline(x=ridge_point, color='k', linestyle=':', label='Ridge Point: %.1f FLOP/byte' % ridge_point)

# Plot the data points
for algo in df['algorithm'].unique():
    data = df[df['algorithm'] == algo]
    plt.scatter(data['arithmetic_intensity'], data['gflops']/1000, label=f"{algo}", s=100, alpha=0.7)

    # Annotate with matrix sizes
    for i, row in data.iterrows():
        label = f"{int(row['size'])} ({row['compute_efficiency_%']:.0f}%)"
        plt.annotate(label,
                    (row['arithmetic_intensity'], row['TFLOP/s']),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha='left',
                    fontsize=8)


# Formatting
plt.ylim(1, peak_compute * 1.2)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
#plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlabel('Arithmetic Intensity (FLOP/byte)')
plt.ylabel('Performance (TFLOP/s)')
plt.title('Roofline Model: GEMM Performance Analysis')
plt.legend()

# Save the figure
plt.savefig('roofline1.png', dpi=300, bbox_inches='tight')
plt.show()