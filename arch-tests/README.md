# Diagnostic Tests for GEMM Kernels

This folder contains small standalone kernels that isolate performance or correctness issues encountered during GEMM development.

Each test is self-contained and can be compiled/profiled independently.

## Goals

- Validate memory coalescence patterns (read/write)
- Diagnose shared memory bank conflicts
- Compare pitched vs. regular memory allocation performance
- Evaluate different memory access patterns (row-major, column-major, tiled)
- Spot warp divergence, occupancy, and ILP bottlenecks
- Explore numerical traps (e.g. NaNs, non-determinism)

## Structure

| File                     | What it tests                                    |
|--------------------------|--------------------------------------------------|
| `memory_test.cu`         | Memory access patterns with pitched/regular memory|
| `mat_load_patterns.cu`   | Row-major, column-major, and tiled load patterns |
| `profile_tests.sh`       | Shell script to profile all memory tests         |
| `coalesced_read_test.cu` | Global memory read access patterns               |
| `shared_conflict_test.cu`| Shared memory bank conflicts                     |

## Memory Test Scripts

The repository includes several shell scripts to automate testing:

| Script                          | Purpose                                         |
|---------------------------------|-------------------------------------------------|
| `compare_rowmajor_patterns.sh`  | Compare row-major access with pitched/regular   |
| `compare_pitch_vs_regular.sh`   | Compare all patterns with pitched/regular memory|
| `test_2d_rowmajor.sh`           | Test 2D row-major with different tile sizes     |

## Usage Examples

### Basic Memory Test

```bash
./memory_test --memory-test=1024 --pattern=rowmajor --config=2d-tile --tile-size=16
```

### Pitched vs. Regular Memory

```bash
./memory_test --memory-test=1024 --pattern=tiled-row --config=2d-tile --tile-size=16 --no-pitch
```

### Profiling with NCU

```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed ./memory_test --memory-test=1024 --pattern=tiled-row --config=2d-tile
```

## Key Findings

- For 2D matrices with power-of-2 dimensions, regular memory allocation performs similarly to pitched memory
- Tiled access patterns typically show better throughput due to improved cache utilization
- Row-major access patterns perform better than column-major on row-major stored matrices
- Thread block dimensions significantly impact memory coalescing and overall throughput

More will be added as performance debugging uncovers new failure modes.

Eventually, this suite may move into its own standalone repository.
