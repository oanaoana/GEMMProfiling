// matrix_utils.cuh - Matrix utilities for scaling, normalization, and transformations
#pragma once
#include <cuda_runtime.h>

// ============================================================================
// MATRIX NORM CALCULATIONS
// ============================================================================

// Device function to compute Frobenius norm of a tile (inline implementation)
__device__ inline float compute_frobenius_norm_tile(float* tile, int rows, int cols) {
    float norm_sq = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = tile[i * cols + j];
            norm_sq += val * val;
        }
    }
    return sqrtf(norm_sq);
}

// Device function to standardize a matrix tile in place (inline implementation)
__device__ inline void standardize_matrix_tile(float* tile, int rows, int cols) {
    float norm = compute_frobenius_norm_tile(tile, rows, cols);

    if (norm > 1e-12f) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                tile[i * cols + j] /= norm;
            }
        }
    }
}

// Device function to scale matrix tile by a constant (inline implementation)
__device__ inline void scale_matrix_tile(float* tile, int rows, int cols, float scale_factor) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            tile[i * cols + j] *= scale_factor;
        }
    }
}

// Device function to estimate condition number of a tile (inline implementation)
__device__ inline float estimate_condition_number_tile(float* tile, int rows, int cols) {
    // Estimate condition number using Frobenius norm and power iteration for largest singular value
    // For square tiles, we can estimate ||A|| * ||A^-1|| approximately

    // First compute Frobenius norm of the tile
    float frobenius_norm = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = tile[i * cols + j];
            frobenius_norm += val * val;
        }
    }
    frobenius_norm = sqrtf(frobenius_norm);

    if (frobenius_norm < 1e-12f) return 1e12f; // Near-zero matrix

    // For non-square tiles, return a scaled Frobenius norm as approximation
    if (rows != cols) {
        return frobenius_norm * sqrtf((float)(rows + cols));
    }

    // For square tiles, estimate condition number using iterative method
    // Approximate largest eigenvalue using power iteration (simplified)
    float max_eigenval = 0.0f;
    float min_eigenval = 1e12f;

    // Simple approximation: use diagonal dominance and row sums
    for (int i = 0; i < rows; i++) {
        float row_sum = 0.0f;
        float diag_val = fabsf(tile[i * cols + i]);

        for (int j = 0; j < cols; j++) {
            row_sum += fabsf(tile[i * cols + j]);
        }

        // Gershgorin circle estimation
        float radius = row_sum - diag_val;
        float upper_bound = diag_val + radius;
        float lower_bound = fmaxf(diag_val - radius, 1e-12f);

        max_eigenval = fmaxf(max_eigenval, upper_bound);
        min_eigenval = fminf(min_eigenval, lower_bound);
    }

    // Condition number approximation
    float condition_estimate = max_eigenval / min_eigenval;

    // Clamp to reasonable bounds
    return fminf(condition_estimate, 1e10f);
}

// ============================================================================
// HOST FUNCTION DECLARATIONS
// ============================================================================

// Host function to compute Frobenius norm of a full matrix
float compute_frobenius_norm_host(float* h_matrix, int n);

// Host function to standardize a matrix (A = A / ||A||_F) in place
void standardize_matrix_host(float* h_matrix, int n);

// Host function to scale matrix by a constant
void scale_matrix_host(float* h_matrix, int n, float scale_factor);

// Host function to estimate condition number of a full matrix
float estimate_condition_number_host(float* h_matrix, int n);

// ============================================================================
// CONFIGURATION
// ============================================================================

// Configuration flag for matrix standardization
#ifndef ENABLE_MATRIX_STANDARDIZATION
#define ENABLE_MATRIX_STANDARDIZATION 1  // Set to 0 to disable standardization
#endif
