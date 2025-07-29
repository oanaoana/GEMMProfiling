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
│   ├── numerical_analysis.cu    # Backward error analysis kernels and functions
│   ├── error_tests.cu           # Modular error testing framework
│   └── utils.cu                 # Utility functions and memory management
├── include/                     # Header files
│   ├── benchmark.h              # Benchmark function declarations
│   ├── gemms.cuh                # GEMM implementation headers
│   ├── numerical_analysis.cuh   # Numerical analysis headers
│   ├── error_tests.cuh          # Error testing framework headers
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

### Error Testing Modules

The project includes a modular error testing framework (`src/error_tests.cu`) with three main components:

1. **setupMatrix()**: Configures different matrix types
   - Random matrices (uniform distribution)
   - Well-conditioned matrices (near-identity structure)
   - Ill-conditioned matrices (Hilbert-like)
   - Support for loading matrices from files

2. **runMatrixTests()**: Executes numerical analysis on configured matrix pairs
   - Performs tiled GEMM with error tracking
   - Computes condition numbers for each tile
   - Analyzes error propagation patterns

3. **generateReport()**: Creates summary statistics and reports
   - Aggregates results across all test configurations
   - Generates CSV files for further analysis
   - Provides statistical summaries

### Error Analysis Details

The backward error analysis focuses on:

1. **Tile-Level Condition Numbers**: Full condition number estimation for each tile using:
   - Gershgorin circle theorem for eigenvalue bounds
   - Frobenius norm calculations
   - Power iteration approximations for large tiles

2. **Error Accumulation Tracking**: Monitors how errors propagate during computation:
   - Absolute errors compared to cuBLAS reference
   - Relative errors normalized by magnitude
   - Accumulated floating-point errors during tile operations

3. **Statistical Analysis**: Comprehensive error characterization:
   - Error distribution across the result matrix
   - Boundary effects at tile edges
   - Correlation between condition numbers and error magnitude

### Matrix Test Types

The framework automatically tests multiple matrix configurations:

- **Random**: Uniformly distributed elements, moderate condition numbers
- **Well-conditioned**: Near-identity structure, low condition numbers
- **Ill-conditioned**: Hilbert-like matrices, very high condition numbers
- **Custom**: Support for loading problem-specific matrices

## Tools and Visualization

This repository includes several analysis tools:

- **Roofline Analysis**: Performance visualization showing arithmetic intensity vs. computational throughput
- **Numerical Error Visualization**: Heatmaps showing error distribution across the result matrix
- **Tile Boundary Analysis**: Specialized analysis of error behavior at tile boundaries
- **Statistical Error Analysis**: Comparison of error distributions across different implementations
- **Condition Number Analysis**: Visualization of how matrix conditioning affects numerical stability

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

# Run all benchmarks with default matrix size (1024x1024)
./main

# Run specific GEMM implementation
./main --test=tiled --size=512

# Run with result verification enabled
./main --test=cublas --size=256 --verify
```

#### Performance Benchmarking

```bash
# Run comprehensive performance analysis
./main --all --size=1024

# Run specific implementation with verification
./main --test=tiled --size=512 --verify

# Quick benchmark of all implementations
./main --test=all --size=256
```

#### Numerical Analysis

The numerical analysis module provides comprehensive backward error analysis:

```bash
# Run numerical analysis with default settings
./main --numerical-analysis --size=1024

# Run analysis on specific matrix size
./main --numerical-analysis --size=512

# Run with verification enabled
./main --numerical-analysis --size=1024 --verify
```

This will generate:
- Raw error data: `data/numerical_analysis_*.dat`
- Summary statistics: `data/numerical_analysis_summary.csv`
- Performance data: `data/roofline_data.csv`

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

The error testing framework supports different matrix types:

```bash
# The numerical analysis automatically tests:
# - Random matrices
# - Well-conditioned matrices
# - Ill-conditioned matrices
# - Hilbert matrices (highly ill-conditioned)
./main --numerical-analysis --size=1024
```

#### Custom Tile Size Analysis

To test different tile sizes, modify the `TILE_SIZE` definition in the source code:

```bash
# Edit src/gemms.cu or relevant headers to change TILE_SIZE
# Then recompile and run
make clean && make
./main --numerical-analysis --size=1024
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
- `--verify`: Enable result verification against cuBLAS
- `--no-verify`: Disable result verification (default)
- `--verify=true/false`: Explicitly set verification mode
- `--numerical-analysis`: Perform comprehensive backward error analysis

### Example Commands

```bash
# Performance benchmarking
./main --test=tiled --size=512 --verify
./main --all --size=1024

# Numerical error analysis
./main --numerical-analysis --size=1024
./main --numerical-analysis --size=512 --verify

# Quick testing
./main --test=cublas --size=256
```

## Output Files and Data Organization

### Generated Data Files

All generated data is automatically saved to the `data/` directory:

**Numerical Analysis:**
- `numerical_analysis_random_n{size}_tile{TILE_SIZE}.dat`: Random matrix analysis
- `numerical_analysis_wellcond_n{size}_tile{TILE_SIZE}.dat`: Well-conditioned matrix analysis
- `numerical_analysis_illcond_n{size}_tile{TILE_SIZE}.dat`: Ill-conditioned matrix analysis
- `numerical_analysis_summary.csv`: Summary statistics across all tests

**Performance Data:**
- `roofline_data.csv`: Performance benchmarking results for roofline analysis

### Visualization Outputs

Generated plots are saved to the `plots/` directory:
- `numerical_analysis_*_heatmaps.png`: Error distribution heatmaps
- `numerical_analysis_*_tile_analysis.png`: Tile-level error analysis
- `numerical_analysis_summary.png`: Overview of all test results
- `roofline_model.png`: Performance roofline plots

## Workflow Examples

### Complete Analysis Workflow

```bash
# 1. Build the project
make clean && make

# 2. Run comprehensive numerical analysis
./main --numerical-analysis --size=1024

# 3. Generate visualizations
cd scripts/
python plot_numerical_analysis.py
python plot_roofline.py

# 4. View results
ls ../plots/                    # Check generated plots
cat ../data/numerical_analysis_summary.csv  # View summary statistics
```

### Performance Comparison Workflow

```bash
# 1. Compare different implementations
./main --test=naive --size=512 --verify
./main --test=tiled --size=512 --verify
./main --test=cublas --size=512 --verify

# 2. Run full benchmark suite
./main --all --size=1024

# 3. Analyze performance characteristics
cd scripts/
python plot_roofline.py
```

### Development and Testing Workflow

```bash
# 1. Test with small matrices during development
./main --numerical-analysis --size=256

# 2. Verify correctness
./main --test=tiled --size=512 --verify

# 3. Profile specific scenarios
./main --test=tiled --size=1024 --numerical-analysis

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

The backward error analysis has revealed several important insights:

### Tile Size Impact
- Larger tiles generally provide better numerical properties due to reduced boundary effects
- Smaller tiles show increased error accumulation at tile boundaries
- Optimal tile size depends on matrix conditioning and hardware characteristics

### Error Patterns
- Error accumulation patterns differ significantly between implementations
- Certain tiling strategies show elevated errors at tile boundaries vs. tile interiors
- Hierarchical tiling can reduce boundary-related error accumulation

### Matrix Conditioning Effects
- Matrix condition number amplifies differences between tiling strategies
- Ill-conditioned matrices show much larger variations in backward error
- Well-conditioned matrices demonstrate more consistent error behavior across implementations

### Performance vs. Accuracy Tradeoffs
- Observable tradeoffs between computational performance and numerical accuracy
- cuBLAS provides best balance of performance and accuracy for most cases
- Custom tiling allows fine-tuning for specific accuracy requirements

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

[Add your license information here]

## Acknowledgements

[Add acknowledgements as needed]