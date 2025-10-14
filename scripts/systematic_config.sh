#!/bin/bash

# Systematic Error Analysis Configuration
# =======================================
# Simple configuration for systematic error analysis

# Kernels to test
KERNELS=("tiled" "tiled_pairwise" "cublas" "cutlass_splitk_flat" "cutlass_splitk_pairwise")

# Matrix types to test
#MATRIX_TYPES=("uniform_positive" "wellcond" "illcond" "zeromean" "2powers")
MATRIX_TYPES=("uniform_positive")

# Matrix sizes to test
SIZES=(256 384 512 768 1024 1280 1536 1792 2048 3072 4096)
