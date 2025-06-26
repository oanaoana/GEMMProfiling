# GEMM Profiling

A CUDA project implementing and benchmarking multiple General Matrix Multiplication (GEMM) algorithms with performance analysis and roofline modeling.

## Project Structure

```
GEMMProfiling/
├── include/
│   ├── gemms.cuh         # GEMM kernel declarations
│   ├── benchmark.h       # Benchmark framework declarations
│   └── utils.cuh         # Utility functions
├── gemms.cu              # GEMM implementations (5 variants)
├── benchmark.cu          # Benchmarking and timing framework
├── utils.cu              # Matrix utilities and verification
├── main.cu               # Command-line interface
├── plot_roofline.py      # Roofline model visualization
├── commands.txt          # Build and profiling commands
└── Makefile              # Build system with CUTLASS support
```

## Current Implementations

### ✅ **Naive GEMM**
- **Status**: Complete and working
- **Algorithm**: Basic triple-nested loop
- **Performance**: ~1-2 GFLOP/s (baseline reference)
- **Purpose**: Establishes performance baseline

### ✅ **Tiled GEMM**
- **Status**: Optimized with FMA and loop unrolling
- **Algorithm**: Shared memory tiling with bank conflict avoidance
- **Optimizations**:
  - `__fmaf_rn()` fused multiply-add
  - `#pragma unroll` compiler hints
  - Padded shared memory (`TILE_SIZE + 1`)
  - Coalesced memory access patterns
- **Performance**: ~4-10 GFLOP/s (2-5x improvement over naive)
- **Tile Size**: 16x16 (configurable in `gemms.cuh`)

### ✅ **cuBLAS Standard**
- **Status**: Complete with proper row/column major handling
- **Algorithm**: NVIDIA's optimized BLAS library
- **Performance**: ~40-70 GFLOP/s (production baseline)
- **Notes**: Handles row-major to column-major conversion

### ✅ **cuBLAS with Tensor Core Hints**
- **Status**: Complete with `CUBLAS_TENSOR_OP_MATH` enabled
- **Algorithm**: cuBLAS with Tensor Core acceleration hints
- **Performance**: ~45-75 GFLOP/s (slight improvement over standard)
- **Hardware**: Optimized for RTX 4080 Tensor Cores

### ✅ **CUTLASS**
- **Status**: Complete integration with external library
- **Algorithm**: NVIDIA's CUTLASS template library
- **Performance**: ~15-35 GFLOP/s (between tiled and cuBLAS)
- **Purpose**: Research-quality optimized kernels with readable source

## Hardware Target

**NVIDIA GeForce RTX 4080**
- **Compute Capability**: 8.9 (Ada Lovelace)
- **Global Memory**: 16,375 MB
- **Peak Compute**: ~83 TFLOP/s (FP32)
- **Peak Memory Bandwidth**: ~717 GB/s
- **Multiprocessors**: 76 SMs
- **Shared Memory**: 49,152 bytes per block
- **Tensor Cores**: 4th Gen (Ada Lovelace)

## Key Optimizations Implemented

### Tiled GEMM Optimizations
- ✅ **Shared memory tiling** with 16x16 tiles
- ✅ **Bank conflict avoidance** with `+1` padding
- ✅ **Fused multiply-add** (`__fmaf_rn()`)
- ✅ **Loop unrolling** (`#pragma unroll`)
- ✅ **Coalesced memory access** patterns
- ✅ **Register accumulation** for reduced memory traffic

### Build System Features
- ✅ **CUTLASS integration** with automatic detection
- ✅ **C++17 support** for modern CUDA features
- ✅ **Optimization flags** (`-O3`, `--use_fast_math`)
- ✅ **Conditional compilation** (works with/without CUTLASS)

## Performance Results

**Current Performance Status:**

Performance Summary:
================================================================================
Algorithm       Size     Time (ms)  TFLOP/s    GB/s       Compute Eff. Bandwidth Eff.
--------------------------------------------------------------------------------
naive           4096     5.65       3.04       8.90       6.20        % 1.24           %
tiled           4096     5.20       3.01       8.10       5.91        % 1.22           %
cublas          4096     4.94       27.80      40.72      56.70       % 5.68           %
cublas_tensor   4096     2.70       50.89      74.55      103.80      % 10.40          %
cutlass         4096     4.07       33.80      49.51      68.94       % 6.91           %
================================================================================

!!! Need better tiling, moving to rectangular


## Dependencies

### Required
- **CUDA Toolkit** 11.0+ (tested with 12.0+)
- **cuBLAS** (included with CUDA)
- **C++17 compiler** (nvcc with GCC 9+)

### Optional
- **CUTLASS** 3.4.1+ (for CUTLASS implementation)
```bash
cd ~
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && git checkout v3.4.1
export CUTLASS_PATH=~/cutlass
```

### Python (for analysis)
- **matplotlib** (roofline plotting)
- **pandas** (data analysis)
- **numpy** (numerical operations)

## Build and Run

### Quick Start
```bash
# Build all implementations
make clean && make

# Run all benchmarks
./main

# Test specific implementation
./main --test=tiled --size=1024

# Generate roofline plot
python plot_roofline.py
```

### Build Options
```bash
# Standard build
make

# Build with CUTLASS (if installed)
export CUTLASS_PATH=~/cutlass
make clean && make

# Debug build
make debug

# Manual build with all optimizations
nvcc -O3 --use_fast_math -arch=sm_89 -std=c++17 \
     -I./include -I$CUTLASS_PATH/include \
     main.cu benchmark.cu gemms.cu utils.cu -o main -lcublas
```

## Profiling and Analysis

### Performance Profiling
```bash
# Profile with NVIDIA Nsight Compute
ncu --set basic ./main --test=tiled --size=1024
ncu --set full ./main --test=cutlass --size=2048

# Profile all implementations
ncu --set basic ./main > profile_results.txt
```

### Roofline Analysis
```bash
# Generate performance data
./main --size=512 --size=1024 --size=2048

# Create roofline plot
python plot_roofline.py

# Output: roofline_plot.png with all implementations plotted
```

### Test tiling sizes
```bash
./test_tile_size
```

## Development Status

### ✅ **Completed Features**
- All 5 GEMM implementations working
- Comprehensive benchmarking framework
- Matrix size scaling (256 to 4096)
- Correctness verification
- Performance measurement and CSV export
- CUTLASS library integration
- Optimized tiled implementation

### 🚧 **In Progress**
- Advanced register blocking (4x4 per thread)
- Vectorized memory access (float4 loads)
- Double buffering for compute/memory overlap
- Roofline plotting improvements
- Add rectangular tiling

### 📋 **Future Enhancements**
- Mixed-precision implementations (FP16/FP32)
- Batched GEMM operations
- Multi-GPU distribution
- Auto-tuning framework
- Energy efficiency analysis

## Research Applications

This project demonstrates:
- **Memory hierarchy optimization** through tiling strategies
- **Compute optimization** via FMA and instruction-level parallelism
- **Library integration** with external high-performance libraries
- **Performance analysis** using roofline modeling
- **Algorithm comparison** across implementation complexity levels

Perfect for **CUDA learning**, **HPC research**, and **GPU optimization studies**.

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [cuBLAS Library](https://docs.nvidia.com/cuda/cublas/)
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)