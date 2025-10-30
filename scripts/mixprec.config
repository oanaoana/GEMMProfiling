#!/bin/bash

# Mixed Precision Configuration
KERNELS=("tiled_mixprec" "tiled_pairwise_mixprec")
#MATRIX_TYPES=("uniform_positive" "wellcond" "illcond" "zeromean" "2powers")
MATRIX_TYPES=("uniform_positive")
SIZES=(256 512 1024 1536 2048 3072 4096)

# Precision configurations to test
PRECISION_CONFIGS=(
    "FP32:FP32"
)

# Note: DATA_FOLDER will be computed per precision config