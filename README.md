# GEMM Profiling

A project to develop and profile General Matrix Multiplication (GEMM) implementations on CUDA.

GEMMProfiling/
├── include/
│   ├── gemms.cuh         # All kernel declarations
│   └── utils.cuh         # Utility functions
├── gemms.cu              # GEMM implementations
├── benchmark.cu          # Main benchmark framework
├── main.cu               # Simple command-line interface
├── plot_roofline.py      # Analysis script
└── Makefile

## Hardware Specifications

Device 0: NVIDIA GeForce RTX 4080
  Global memory: 16375 MB
  Shared memory per block: 49152
  Registers per block: 65536
  Warp size: 32
  Max threads per block: 1024
  Max threads dim: [1024, 1024, 64]
  Max grid size: [2147483647, 65535, 65535]
  Clock rate: 2520000 kHz
  Multiprocessor count: 76
  Compute capability: 8.9
  Memory Bus Width: 256 bits
  L2 Cache Size: 67108864

## Project Overview

This project implements and compares different GEMM (General Matrix Multiplication) implementations:

1. **Naive GEMM**: A simple implementation without optimizations
2. **Tiled GEMM**: Using shared memory tiling to improve memory access patterns and computation efficiency

### Optimizations

The tiled implementation incorporates several optimizations:
- **Shared memory tiling**: Reduces global memory accesses by loading data into shared memory tiles
- **Memory coalescing**: Ensures threads access memory in patterns that maximize bandwidth
- **Loop unrolling**: Reduces instruction overhead in critical loops
- **Register blocking**: Keeps frequently accessed values in registers

## Benchmarking

Performance is measured in GFLOP/s across different matrix sizes and compared against vendor libraries.

## Usage

[Usage instructions to be added]

## Results

[Performance comparison results to be added]

## Future Improvements

- Implement additional optimizations (e.g., double buffering, vectorization)
- Explore different tile sizes for optimal performance
- Compare with other GEMM implementations (cuBLAS, etc.)