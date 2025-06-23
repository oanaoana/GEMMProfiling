# GEMM Profiling

A project to develop and profile General Matrix Multiplication (GEMM) implementations on CUDA.

```
GEMMProfiling/
├── include/
│   ├── gemms.cuh         # All kernel declarations
│   ├── benchmark.h       # Benchmark framework declarations
│   └── utils.cuh         # Utility functions
├── gemms.cu              # GEMM implementations
├── benchmark.cu          # Main benchmark framework
├── utils.cu              # Utility functions implementation
├── main.cu               # Command-line interface
├── plot_roofline.py      # Roofline analysis script
├── commands.txt          # Compilation and profiling commands
└── Makefile
```

## Hardware Specifications

**Target Device: NVIDIA GeForce RTX 4080**
- Global memory: 16,375 MB
- Shared memory per block: 49,152 bytes
- Registers per block: 65,536
- Warp size: 32
- Max threads per block: 1,024
- Max threads dim: [1024, 1024, 64]
- Max grid size: [2147483647, 65535, 65535]
- Clock rate: 2,520 MHz
- Multiprocessor count: 76
- Compute capability: 8.9
- Memory Bus Width: 256 bits
- L2 Cache Size: 64 MB
- Peak Compute Performance: ~83 TFLOP/s (FP32)
- Peak Memory Bandwidth: ~717 GB/s

## Project Overview

This project implements and compares three different GEMM (General Matrix Multiplication) implementations:

### 1. **Naive GEMM**
A straightforward implementation without optimizations:
- Direct computation: `C[i][j] = Σ(A[i][k] * B[k][j])`
- No memory access optimizations
- Serves as baseline for comparison

### 2. **Tiled GEMM**
Optimized implementation using shared memory tiling:
- **Shared memory tiling**: Loads data into 16x16 shared memory tiles
- **Memory coalescing**: Optimized memory access patterns
- **Bank conflict avoidance**: Padded shared memory arrays
- **Reduced global memory traffic**: Each element loaded once per tile

### 3. **cuBLAS GEMM**
NVIDIA's highly optimized library implementation:
- Industry-standard reference for peak performance
- Uses Tensor Cores when available (RTX 4080 supports them)
- Heavily optimized assembly kernels
- Multi-level memory hierarchy optimization

## Optimizations Implemented

The tiled implementation incorporates several key optimizations:

- **Shared Memory Tiling**: Reduces global memory accesses by ~32x for 16x16 tiles
- **Memory Coalescing**: Ensures consecutive threads access consecutive memory addresses
- **Bank Conflict Avoidance**: Uses `[TILE_SIZE][TILE_SIZE + 1]` padding
- **Boundary Checking**: Handles non-multiple matrix sizes correctly
- **Register Optimization**: Accumulates results in registers before writing back

## Benchmarking Framework

Performance metrics collected:
- **Execution Time** (milliseconds)
- **Throughput** (GFLOP/s)
- **Memory Bandwidth** (GB/s)
- **Arithmetic Intensity** (FLOPs/byte)
- **Compute Efficiency** (% of peak performance)
- **Memory Efficiency** (% of peak bandwidth)
- **Correctness Verification** (automated result checking)

## Usage

### Basic Compilation and Execution
```bash
# Build the project
make

# Run all tests on all sizes
./main

# Run specific algorithm
./main --test=naive --size=1024
./main --test=tiled --size=1024
./main --test=cublas --size=1024

# Run specific matrix size
./main --size=512

# Show help
./main --help
```

### Profiling with NVIDIA Tools
```bash
# Profile with Nsight Compute (detailed kernel analysis)
ncu --set basic ./main --test=tiled --size=1024

# Profile with Nsight Systems (timeline analysis)
nsys profile --trace=cuda ./main --test=tiled --size=1024

# Compare implementations
ncu --set basic ./main --test=naive --size=1024 -o naive_profile
ncu --set basic ./main --test=tiled --size=1024 -o tiled_profile
ncu --set basic ./main --test=cublas --size=1024 -o cublas_profile
```

### Performance Analysis
```bash
# Generate roofline model data
./main > performance_results.txt

# Create roofline plot (requires Python with matplotlib)
python plot_roofline.py

# View detailed comparison
cat roofline_data.csv
```

## Current Performance Results (Work in Progress)

**Current Performance Status (RTX 4080, 1024x1024) - UNOPTIMIZED:**

| Implementation | Time (ms) | GFLOP/s | Efficiency vs cuBLAS | Status |
|----------------|-----------|---------|----------------------|---------|
| Naive          | ~250-500  | ~0.5-1.0| ~1-2%               | ⚠️ Baseline implementation |
| Tiled          | ~150-300  | ~0.7-1.4| ~1.5-3%             | 🚧 Under optimization |
| cuBLAS         | ~3-5      | ~40-70  | 100% (reference)     | ✅ Production library |

**⚠️ Current Implementation Status:**
- **Performance is significantly suboptimal** - This is active development, not final results
- **Tiled kernel shows modest improvement** - Basic shared memory utilization working
- **Major optimizations still pending** - See roadmap below for planned improvements

## Known Performance Issues

### Current Bottlenecks
- **Naive Implementation:**
  - Severely memory bandwidth limited
  - Uncoalesced global memory access patterns
  - Poor cache utilization across all levels

- **Tiled Implementation:**
  - Basic tiling working but not optimized
  - Single-element-per-thread computation
  - No register blocking or double buffering
  - Simple boundary handling with performance penalty

### Performance Gap Analysis
The 10-50x performance gap vs cuBLAS is due to missing optimizations:

**Memory Optimizations (Not Yet Implemented):**
- ❌ Vectorized memory access (float4, texture memory)
- ❌ Double buffering for compute/memory overlap
- ❌ Advanced prefetching strategies
- ❌ Multi-level cache blocking

**Compute Optimizations (Not Yet Implemented):**
- ❌ Tensor Core utilization (WMMA/mma instructions)
- ❌ Register blocking (computing multiple elements per thread)
- ❌ Loop unrolling and instruction pipelining
- ❌ Warp-level cooperative algorithms

**Algorithmic Optimizations (Planned):**
- ❌ Hierarchical blocking strategies
- ❌ Mixed-precision computation
- ❌ Batched operations
- ❌ Auto-tuning for different problem sizes

## Development Roadmap

### Phase 1: Memory Optimization (In Progress)
- [ ] **Vectorized loads/stores** - Use float4 for improved bandwidth
- [ ] **Coalescing optimization** - Ensure all memory accesses are coalesced
- [ ] **Shared memory bank conflict elimination** - Optimize padding strategies
- [ ] **Prefetching** - Overlap tile loading with computation

### Phase 2: Compute Optimization (Planned)
- [ ] **Register blocking** - Compute 4x4 or 8x8 tiles per thread
- [ ] **Double buffering** - Pipeline memory and compute operations
- [ ] **Loop unrolling** - Reduce instruction overhead
- [ ] **Occupancy optimization** - Balance shared memory and register usage

### Phase 3: Advanced Optimizations (Future)
- [ ] **Tensor Core integration** - Use WMMA API for mixed precision
- [ ] **Multi-level blocking** - Implement hierarchical tiling
- [ ] **Auto-tuning framework** - Runtime parameter optimization
- [ ] **Specialized kernels** - Different variants for different problem sizes

### Phase 4: Production Readiness (Long-term)
- [ ] **Error handling** - Robust input validation
- [ ] **Multiple data types** - FP16, INT8, complex numbers
- [ ] **Batched operations** - Multiple small matrices
- [ ] **Multi-GPU support** - Distributed computation

## Current vs Target Performance

**Immediate Goals (Next Implementation Cycle):**
- **Target**: 2-4x improvement over current tiled implementation
- **Method**: Register blocking + vectorized memory access
- **Expected**: ~3-7 GFLOP/s (still far from cuBLAS, but measurable progress)

**Medium-term Goals:**
- **Target**: 10-20% of cuBLAS performance
- **Method**: Advanced memory hierarchy optimization + compute optimization
- **Expected**: ~8-15 GFLOP/s

**Why This Gap Exists:**
This is a research/development project implementing algorithms from scratch. Production libraries like cuBLAS represent decades of optimization by teams of experts with access to internal GPU architecture details not available to external developers.

## Profiling and Analysis Tools

**Current Analysis Capabilities:**
- ✅ Basic timing and throughput measurement
- ✅ Memory bandwidth utilization analysis
- ✅ Correctness verification
- 🚧 ncu profiling integration (kernel-level metrics)
- ❌ Detailed instruction analysis (planned)
- ❌ Energy consumption measurement (planned)

## Contributing to Development

**Areas needing improvement:**
1. **Memory access pattern optimization**
2. **Register utilization efficiency**
3. **Thread block and grid dimension tuning**
4. **Kernel fusion opportunities**
5. **Problem size specialization**

This is an active development project - performance numbers reflect current implementation state, not final capabilities.

## References

- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass)
- [Roofline Performance Model](https://crd.lbl.gov/departments/computer-science/par/research/roofline/)
- [GEMM Optimization Techniques](https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-wavefront-path-tracing/)