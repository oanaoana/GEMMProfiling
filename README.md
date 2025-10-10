# GEMM Numerical Analysis and Performance Profiling

A comprehensive CUDA-based matrix multiplication (GEMM) **numerical analysis tool** that focuses on accuracy assessment across different kernel implementations and precision types, with performance profiling as a secondary feature.

## Features

- **Comprehensive Error Analysis**: Multi-sample statistical error analysis, ULP analysis, per-tile validation
- **Mixed Precision Research**: Configurable compute and accumulate types with accuracy assessment
- **Multiple GEMM Kernels**: Naive, tiled variants, mixed-precision implementations, cuBLAS, and CUTLASS
- **Performance Profiling**: Timing analysis with roofline model metrics (secondary feature)
- **Research-Focused**: Designed for numerical accuracy research in GPU computing

## Available Kernels

| Kernel Name | Description |
|-------------|-------------|
| `naive` | Basic per-thread implementation |
| `tiled` | Shared memory tiling |
| `tiled_opt` | Optimized tiled implementation |
| `tiled_pairwise` | Tiled with pairwise summation |
| `tiled_pairwise_mixprec` | Mixed-precision tiled with pairwise summation |
| `tiled_rect` | Rectangular tiling (non-square tiles) |
| `tiled_mixprec` | Mixed-precision tiling (uses `COMPUTE_TYPE`/`ACCUMULATE_TYPE`) |
| `cublas` | NVIDIA cuBLAS library |
| `cutlass` | NVIDIA CUTLASS library |

## Core Usage - Error Analysis

### Multi-Sample Error Analysis
```bash
# Primary use case - comprehensive error analysis
./main --error-analysis --test=tiled_mixprec --size=1024 --matrix-type=wellcond

# Test different precision combinations
./main --error-analysis --test=tiled_pairwise_mixprec --size=2048 --matrix-type=illcond

# Compare against reference implementations
./main --error-analysis --test=cublas --size=1024 --matrix-type=normal
```

### ULP (Units in Last Place) Analysis
```bash
# Detailed precision analysis
./main --ulp-analysis --test=tiled_mixprec --size=1024 --matrix-type=wellcond

# Compare mixed precision variants
./main --ulp-analysis --test=tiled_pairwise_mixprec --size=1024
```

### Per-Tile Reference Analysis
```bash
# Detailed tile-level validation
./main --per-tile --test=tiled_mixprec --size=1024 --sample=0

# Analyze specific problematic samples
./main --per-tile --test=tiled_pairwise_mixprec --size=2048 --sample=5
```

### Combined Analysis (Tradeoff Study)
```bash
# Run both accuracy and performance analysis
./main --tradeoff --test=tiled_mixprec --size=1024 --matrix-type=wellcond
```

## Secondary Usage - Performance Analysis

### Performance Testing
```bash
# Performance benchmarking (secondary feature)
./main --performance --test=tiled_mixprec --size=1024

# Resource utilization assessment
./main --assess-resources --test=tiled_pairwise_mixprec --size=2048
```

## Mixed Precision Configuration

The tool's primary focus is mixed-precision accuracy research. Configure in `config.h`:

```cpp
#define COMPUTE_TYPE __half        // Input/computation precision
#define ACCUMULATE_TYPE float      // Accumulation precision
```

Or override at compile time:
```bash
make COMPUTE_TYPE=__half ACCUMULATE_TYPE=float
```

### Mixed Precision Research Workflow
```bash
# 1. Compile for specific precision combination
make COMPUTE_TYPE=__half ACCUMULATE_TYPE=float

# 2. Run comprehensive error analysis
./main --error-analysis --test=tiled_mixprec --size=1024 --matrix-type=wellcond

# 3. Analyze ULP errors for precision insights
./main --ulp-analysis --test=tiled_mixprec --size=1024

# 4. Investigate specific tiles if needed
./main --per-tile --test=tiled_mixprec --size=1024 --sample=0

# 5. Optional: Check performance impact
./main --performance --test=tiled_mixprec --size=1024
```

## Matrix Types for Numerical Analysis

| Type | Description | Use Case |
|------|-------------|----------|
| `wellcond` | Well-conditioned matrices | Baseline accuracy testing |
| `illcond` | Ill-conditioned matrices | Stress testing numerical stability |
| `normal` | Normal distribution | General-purpose testing |
| `scaled` | Scaled matrices | Range sensitivity analysis |
| `skewed` | Skewed distributions | Distribution robustness |
| `file` | Load from file | Custom test cases |

## Analysis Output Files

- **Error Analysis**: `data/error_analysis_<kernel>.csv` - Statistical error metrics
- **ULP Analysis**: `data/ulp_analysis_<kernel>.csv` - Precision error analysis
- **Per-tile Analysis**: `data/per_tile_<kernel>.csv` - Detailed tile validation
- **Performance Data**: `data/roofline_data.csv` - Performance metrics (secondary)

## Research-Focused Examples

### Mixed Precision Accuracy Study
```bash
#!/bin/bash
# Study different precision combinations

precisions=("float,float" "__half,float" "__half,__half")
kernels=("tiled_mixprec" "tiled_pairwise_mixprec")
conditions=("wellcond" "illcond")

for precision in "${precisions[@]}"; do
    IFS=',' read compute accumulate <<< "$precision"
    make clean && make COMPUTE_TYPE=$compute ACCUMULATE_TYPE=$accumulate

    for kernel in "${kernels[@]}"; do
        for condition in "${conditions[@]}"; do
            echo "Testing $kernel with $compute/$accumulate on $condition matrices"
            ./main --error-analysis --test=$kernel --size=1024 --matrix-type=$condition
            ./main --ulp-analysis --test=$kernel --size=1024 --matrix-type=$condition
        done
    done
done
```

### Accuracy vs. Performance Tradeoff
```bash
# Comprehensive analysis for research paper
./main --tradeoff --test=tiled_pairwise_mixprec --size=2048 --matrix-type=wellcond
```

## Build System

```bash
# Build with default configuration (float/float)
make

# Build for half precision research
make COMPUTE_TYPE=__half ACCUMULATE_TYPE=float

# Build for extreme precision testing
make COMPUTE_TYPE=__half ACCUMULATE_TYPE=double

# Clean build
make clean
```

## Key Research Features

1. **Statistical Error Analysis**: Multi-sample error statistics with confidence intervals
2. **ULP-Level Precision**: Units in Last Place error measurement for IEEE 754 compliance
3. **Condition Number Sensitivity**: Testing across well/ill-conditioned matrices
4. **Mixed Precision Variants**: Multiple algorithmic approaches to mixed precision
5. **Tile-Level Validation**: Detailed analysis of computational blocks
6. **Reference Comparison**: High-precision validation against cuBLAS/CPU references

## Use Cases

- **Numerical Analysis Research**: Study accuracy of mixed-precision GEMM implementations
- **Algorithm Development**: Validate new mixed-precision algorithms
- **Hardware Evaluation**: Assess numerical behavior on different GPU architectures
- **Precision Studies**: Compare accuracy across different precision combinations
- **Performance vs Accuracy**: Quantify tradeoffs in mixed-precision computing

## Dependencies

- **CUDA Toolkit** (11.0+) with mixed precision support
- **cuBLAS** - Reference implementation
- **CUTLASS** - NVIDIA's template library
- **C++17** - For template metaprogramming

---

**Primary Focus**: This tool is designed for **numerical analysis research** in GPU matrix multiplication, with particular emphasis on mixed-precision arithmetic accuracy assessment.

