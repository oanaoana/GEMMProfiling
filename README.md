# GEMM Profiling: Backward Error Analysis of Matrix Multiplication Tiling Strategies

## Project Overview

This project performs a comprehensive backward error analysis of different tiling strategies in General Matrix Multiplication (GEMM) implementations on CUDA GPUs. Unlike traditional performance profiling, this work focuses on the numerical accuracy implications of various tiling approaches, examining how tiling affects floating-point error propagation and accumulation.

## Key Research Questions

- How do different tiling strategies affect the backward error of GEMM operations?
- What is the relationship between tile size and numerical stability?
- How do different matrix properties (condition number, sparsity) impact error behavior in tiled implementations?
- What are the accuracy vs. performance tradeoffs for various GEMM implementations?

## Implemented GEMM Variants

This repository contains multiple GEMM implementations, each with different tiling strategies:

- **Naive Implementation**: Basic matrix multiplication without tiling
- **Square Tiling**: Classic square tiles (TILE_SIZE × TILE_SIZE)
- **Rectangular Tiling**: Non-square tiles (TILE_M × TILE_N × TILE_K)
- **cuBLAS**: NVIDIA's optimized BLAS implementation
- **cuBLAS Tensor Cores**: Mixed-precision acceleration using Tensor Cores
- **CUTLASS**: CUDA Templates for Linear Algebra Subroutines

## Backward Error Analysis

The numerical analysis in this project focuses on backward error analysis, which quantifies how much the input matrices would need to be perturbed to make the computed result exact. This approach provides insights into:

1. **Tile Boundary Effects**: Error accumulation at tile boundaries vs. tile interiors
2. **Error Distribution**: Statistical properties of error across the result matrix
3. **Condition Number Impact**: How matrix conditioning affects numerical stability under different tiling strategies
4. **Accumulated Error Patterns**: Visualization of error propagation and accumulation patterns

## Tools and Visualization

This repository includes several tools for error analysis:

- **Roofline Analysis**: Performance visualization showing arithmetic intensity vs. computational throughput
- **Numerical Error Visualization**: Heatmaps showing error distribution across the result matrix
- **Tile Boundary Analysis**: Specialized analysis of error behavior at tile boundaries
- **Statistical Error Analysis**: Comparison of error distributions across different implementations

## Building and Running

### Prerequisites
- CUDA Toolkit 11.0 or higher
- Python 3.7+ with NumPy, Matplotlib, and Seaborn for visualization
- (Optional) CUTLASS library for additional GEMM implementations

### Compilation
```bash
make clean && make
```

### Running Benchmarks
```bash
./main --size=1024
```

### Running Numerical Analysis
```bash
# Generate numerical analysis data
./main --size=1024 --test=tiled --numerical-analysis

# Visualize the results
python plot_numerical_analysis.py
```

## Command Line Options

- `--size=N`: Set matrix size (N×N)
- `--test=TYPE`: Run specific implementation (naive, tiled, tiled_rect, cublas, cutlass)
- `--verify`: Enable result verification
- `--numerical-analysis`: Perform backward error analysis and save results
- `--quick`: (for visualization) Use subsampling for large matrices

## Results and Findings

The key findings of this backward error analysis include:

1. Tile size has a direct impact on numerical stability, with larger tiles generally providing better numerical properties but potentially worse performance
2. Error accumulation patterns differ significantly between implementations, with certain tiling strategies showing elevated errors at tile boundaries
3. Matrix condition number amplifies the differences between tiling strategies, with ill-conditioned matrices showing much larger variations in backward error
4. There are observable tradeoffs between computational performance and numerical accuracy across the different implementations

## Future Work

- Analysis of mixed-precision implementations
- Impact of block-cyclic distribution on error patterns
- Extension to other linear algebra operations
- Automatic tile size selection based on numerical stability requirements

## License

[Add your license information here]

## Acknowledgements

[Add acknowledgements as needed]