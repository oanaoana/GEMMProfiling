# GEMM Profiling: Backward Error Analysis of Matrix Multiplication Tiling Strategies

## Project Overview

This project performs a comprehensive backward error analysis of different tiling strategies in General Matrix Multiplication (GEMM) implementations on CUDA GPUs. Unlike traditional performance profiling, this work focuses on the numerical accuracy implications of various tiling approaches, examining how tiling affects floating-point error propagation and accumulation.

## Project Structure

```
GEMMProfiling/
├── src/                         # Source code files
│   ├── main.cu                  # Main application entry point
│   ├── benchmark.cu             # Performance benchmarking routines
│   ├── gemms.cu                 # GEMM implementations (naive, tiled, cuBLAS, etc.)
│   ├── error_analysis.cu        # ULP analysis and statistical error analysis
│   ├── generate_test_matrix.cu  # Matrix generation utilities
│   ├── config.cu                # Configuration and constants
│   └── utils.cu                 # Utility functions and memory management
├── include/                     # Header files
│   ├── benchmark.h              # Benchmark function declarations
│   ├── gemms.cuh                # GEMM implementation headers
│   ├── error_analysis.cuh       # ULP analysis and statistical functions
│   ├── generate_test_matrix.cuh # Matrix generation utilities
│   ├── config.h                 # Configuration constants
│   └── utils.cuh                # Utility function headers
├── scripts/                     # Python analysis and plotting scripts
│   ├── plot_numerical_analysis.py  # Visualize numerical analysis results
│   ├── plot_roofline.py         # Generate roofline performance plots
│   ├── analysis/                # Analysis utility scripts
│   ├── plotting/                # Additional plotting utilities
│   └── utils/                   # Script utilities
├── data/                        # Generated data files (created at runtime)
│   ├── numerical_analysis_*.dat # Raw numerical analysis data
│   ├── numerical_analysis_summary.csv  # Summary statistics
│   └── roofline_data.csv        # Performance benchmarking data
├── plots/                       # Generated visualization outputs
├── build/                       # Compiled object files
├── bin/                         # Additional executables
├── arch-tests/                  # Architecture-specific tests
├── old_stuff/                   # Legacy code and scripts
└── output/                      # Additional output files
```

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

## Numerical Analysis Framework

### ULP (Units in the Last Place) Analysis

The project now focuses on **ULP-based error analysis**, providing statistically rigorous numerical accuracy assessment:

1. **ULP Distance Computation**:
   - Computes exact ULP distance between kernel results and FP64 reference
   - Uses FP64 computation with single rounding to FP32 for highest precision reference
   - Handles special cases (infinities, NaN, sign differences)

2. **Statistical Analysis**:
   - 9-bin histogram analysis of ULP distribution: [0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65+]
   - Wilson confidence intervals for robust proportion estimation
   - Percentile calculations (50th, 90th, 95th, 99th percentiles)

3. **Multi-Sample Analysis**:
   - Tests multiple random matrix pairs for statistical robustness
   - Independent seed generation using cryptographic-quality hash functions
   - Aggregated statistics across all samples

### Analysis Types

#### ULP Analysis
```bash
# Run ULP analysis with statistical measures
./main --ulp-analysis --test=cublas --size=256
```

**Output includes:**
- ULP distance histogram with 9 bins
- Wilson confidence intervals for each bin
- Key percentiles (50th, 90th, 95th, 99th)
- Statistical summary of numerical accuracy

#### Multi-Sample Error Analysis
```bash
# Run comprehensive multi-sample analysis
./main --error-analysis --test=tiled --size=512
```

**Features:**
- Multiple independent matrix samples
- Frobenius norm error computation
- Statistical aggregation across samples
- Robust error characterization

### Matrix Generation

The framework supports various matrix types for comprehensive testing:

- **MATRIX_RANDOM**: Standard uniform random distribution [0,1]
- **MATRIX_ODO_WELL_CONDITIONED**: Near-identity matrices with low condition numbers
- **MATRIX_ODO_ILL_CONDITIONED**: Matrices with high condition numbers for stress testing
- **MATRIX_ZEROMEAN**: Zero-mean uniform distribution [-0.5, 0.5]
- **MATRIX_UNIFORM_POSITIVE**: Positive uniform distribution [0.1, 1.0]
- **MATRIX_RADEMACHER**: Random ±1 values (Rademacher distribution)
- **MATRIX_LOAD_FROM_FILE**: Load custom matrices from binary files

## Unified Kernel Launch Architecture

The project uses a **unified kernel dispatch system** for maintainability and performance:

```cpp
// Centralized kernel dispatch
KernelType kernel_type = getKernelTypeFromName("tiled_pairwise");
launch_kernel_by_type(kernel_type, d_A, d_B, d_C, n, blocks, threads);
```

Benefits:
- ✅ **Zero Runtime Overhead**: Direct function pointer calls during timing
- ✅ **Single Source of Truth**: All dispatch logic centralized in `src/utils.cu`
- ✅ **Type Safe**: Compile-time validation with enum-based kernel types
- ✅ **Easy Extension**: Add new kernels by extending enum and function table## Tools and Visualization

This repository includes analysis tools for:

- **ULP Analysis**: Statistical analysis of Units in Last Place error distributions
- **Multi-Sample Analysis**: Robust error characterization across multiple test cases
- **Performance Benchmarking**: Roofline analysis and throughput measurement
- **Statistical Visualization**: Error distribution plots with confidence intervals

## Building and Running

### Prerequisites
- CUDA Toolkit 11.0 or higher
- Python 3.7+ with NumPy, Matplotlib, and Seaborn for visualization
- (Optional) CUTLASS library for additional GEMM implementations
- Make sure you have sufficient GPU memory for large matrix operations

### Compilation

The project uses a Makefile that automatically handles the new source structure:

```bash
# Clean and build everything
make clean && make

# Build with verbose output
make clean && make VERBOSE=1

# Check for CUTLASS support (optional)
# Set CUTLASS_PATH if you have CUTLASS installed
export CUTLASS_PATH=/path/to/cutlass
make clean && make
```

The build system will:
- Compile all source files from `src/` directory
- Place object files in `build/` directory
- Generate the main executable in the project root
- Automatically detect CUTLASS support if available

### Running the Application

#### Basic Usage

```bash
# Show all available options
./main --help

# Run all benchmarks with default matrix sizes
./main --all

# Run specific GEMM implementation performance test
./main --performance --test=tiled --size=512

# Run ULP analysis
./main --ulp-analysis --test=cublas --size=256
```

#### Performance Benchmarking

```bash
# Run comprehensive performance analysis
./main --all

# Run specific implementation with performance testing
./main --performance --test=tiled --size=512

# Quick benchmark of all implementations
./main --all
```

#### ULP Analysis

The ULP analysis provides the most precise numerical accuracy assessment:

```bash
# Run ULP analysis on specific implementation
./main --ulp-analysis --test=cublas --size=512

# Run ULP analysis with specific matrix type
./main --ulp-analysis --test=tiled --size=1024 --matrix-type=illcond
```

This generates:
- ULP distance histogram with Wilson confidence intervals
- Percentile analysis of ULP distribution
- Statistical summary of numerical errors
- Comparison against FP64 reference computation

#### Multi-Sample Error Analysis

```bash
# Run comprehensive error analysis
./main --error-analysis --test=tiled --size=1024

# Run with specific matrix types
./main --error-analysis --test=cublas --size=512 --matrix-type=wellcond
```

This will generate:
- Frobenius norm error analysis across multiple samples
- Statistical aggregation of error metrics
- Robust characterization of implementation accuracy

#### Available Test Types

- `naive`: Basic O(n³) matrix multiplication
- `tiled`: Square tiling implementation
- `tiled_rect`: Rectangular tiling (if implemented)
- `cublas`: NVIDIA cuBLAS library
- `cutlass`: CUTLASS template library (if available)
- `all`: Run all available implementations

### Data Analysis and Visualization

After running numerical analysis, use the Python scripts to visualize results:

```bash
# Create numerical analysis plots
cd scripts/
python plot_numerical_analysis.py

# Generate roofline performance plots
python plot_roofline.py

# For custom analysis, data files are in ../data/
```

The visualization scripts will:
- Read data from the `data/` directory
- Generate plots in the `plots/` directory
- Create heatmaps, error distribution plots, and performance analysis

### Advanced Usage

#### Testing Different Matrix Types

The error analysis automatically tests multiple matrix types:

```bash
# ULP analysis tests all supported matrix types automatically
./main --ulp-analysis --test=cublas --size=1024

# Error analysis uses random matrices by default
./main --error-analysis --test=tiled --size=512
```

#### Custom Tile Size Analysis

To test different tile sizes, modify the `TILE_SIZE` definition in the source code:

```bash
# Edit include/config.h to change TILE_SIZE
# Then recompile and run
make clean && make
./main --ulp-analysis --test=tiled --size=1024
```

#### Architecture-Specific Tests

```bash
# Run memory access pattern tests
cd arch-tests/
make
./run_tests.sh
```

## Command Line Options

### Core Options
- `--help`, `-h`: Show detailed usage information
- `--size=N`: Set matrix size (N×N), default is 1024
- `--all`: Run all tests and available matrix sizes

### Test Selection
- `--test=TYPE`: Run specific implementation
  - `naive`: Basic matrix multiplication
  - `tiled`: Square tiling implementation
  - `cublas`: NVIDIA cuBLAS library
  - `cutlass`: CUTLASS library (if available)
  - `all`: Run all available implementations

### Verification and Analysis
- `--ulp-analysis`: Perform ULP (Units in Last Place) analysis with statistical measures
- `--error-analysis`: Run multi-sample error analysis with Frobenius norm computation
- `--performance`: Run performance benchmarking only
- `--complete`: Run both error analysis and performance testing
- `--matrix-type=TYPE`: Specify matrix type for analysis (optional)

### Example Commands

```bash
# Performance benchmarking
./main --performance --test=tiled --size=512
./main --all

# ULP analysis
./main --ulp-analysis --test=cublas --size=1024
./main --ulp-analysis --test=tiled --size=512 --matrix-type=illcond

# Multi-sample error analysis
./main --error-analysis --test=tiled --size=1024
./main --error-analysis --test=cublas --size=512 --matrix-type=wellcond

# Complete analysis (both error and performance)
./main --complete --test=tiled --size=1024

# Quick testing
./main --performance --test=cublas --size=256
```

## Output Files and Data Organization

### Generated Data Files

All generated data is automatically saved to the `data/` directory:

**ULP Analysis Data:**
- ULP histogram data with statistical measures
- Wilson confidence intervals for robust proportion estimation
- Percentile analysis (50th, 90th, 95th, 99th percentiles)

**Error Analysis Data:**
- Multi-sample Frobenius norm error data
- Statistical aggregation across samples
- Comprehensive error characterization

**Performance Data:**
- `roofline_data.csv`: Performance benchmarking results for roofline analysis

### Visualization Outputs

Generated plots are saved to the `plots/` directory:
- ULP analysis visualizations showing error distribution
- Statistical plots with confidence intervals and percentiles
- Multi-sample error analysis plots
- Performance roofline plots

## Workflow Examples

### Complete Analysis Workflow

```bash
# 1. Build the project
make clean && make

# 2. Run ULP analysis for precise error measurement
./main --ulp-analysis --test=cublas --size=1024

# 3. Run multi-sample error analysis
./main --error-analysis --test=tiled --size=1024

# 4. Generate visualizations (if visualization scripts available)
cd scripts/
python plot_error_analysis.py
python plot_roofline.py

# 5. View results
ls ../plots/                    # Check generated plots
cat ../data/error_analysis_summary.txt  # View summary statistics
```

### Performance Comparison Workflow

```bash
# 1. Compare different implementations
./main --performance --test=naive --size=512
./main --performance --test=tiled --size=512
./main --performance --test=cublas --size=512

# 2. Run full benchmark suite
./main --all

# 3. Analyze performance characteristics (if scripts available)
cd scripts/
python plot_roofline.py
```

### Development and Testing Workflow

```bash
# 1. Test with small matrices during development
./main --ulp-analysis --test=tiled --size=256

# 2. Verify correctness with performance test
./main --performance --test=tiled --size=512

# 3. Comprehensive error analysis
./main --error-analysis --test=tiled --size=1024

# 4. Check architecture-specific behavior
cd arch-tests/
make && ./run_tests.sh
```

## Troubleshooting

### Common Issues

**Build Errors:**
- Ensure CUDA Toolkit is properly installed and in PATH
- Check that nvcc compiler version is compatible
- Verify CUTLASS path if using CUTLASS features

**Runtime Errors:**
- Check GPU memory availability for large matrices
- Ensure data/ directory is writable
- Verify cuBLAS library is available

**Missing Output:**
- Data files are generated in `data/` directory
- Plots are created in `plots/` directory
- Check file permissions if files aren't created

### Memory Requirements

Approximate GPU memory usage for different matrix sizes:
- 256×256: ~1 MB
- 512×512: ~4 MB
- 1024×1024: ~16 MB
- 2048×2048: ~64 MB
- 4096×4096: ~256 MB

Additional memory is required for intermediate results and analysis data.

## Results and Key Findings

The ULP-based error analysis has revealed several important insights:

### ULP Distribution Patterns
- Most accurate implementations show concentrated ULP distribution in lower bins (0-2 ULP)
- Different GEMM implementations exhibit distinct ULP distribution signatures
- Wilson confidence intervals provide robust statistical characterization of implementation accuracy

### Statistical Measures
- 50th percentile ULP distances characterize typical accuracy
- 95th and 99th percentiles reveal worst-case error behavior
- Confidence intervals enable rigorous comparison between implementations

### Implementation Comparison
- cuBLAS typically provides best overall accuracy with tight ULP distributions
- Custom tiling strategies show varying accuracy depending on tile size and algorithm
- FP64 reference computation ensures highest precision baseline for comparison

### Matrix Properties Impact
- Matrix conditioning affects ULP distribution spread
- Random matrices provide good baseline for general accuracy assessment
- Well-conditioned vs ill-conditioned matrices reveal implementation robustness

## Future Work

### Planned Enhancements
- Analysis of mixed-precision implementations (FP16, INT8)
- Impact of block-cyclic distribution on error patterns
- Extension to other linear algebra operations (LU, QR, SVD)
- Automatic tile size selection based on numerical stability requirements
- Integration with tensor operation libraries

### Research Directions
- Error analysis for sparse matrix operations
- Fault tolerance and error correction in tiled computations
- Machine learning applications with controlled numerical precision
- Real-time error monitoring during computation

## License

MIT License

Copyright (c) 2025 Oana Marin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

