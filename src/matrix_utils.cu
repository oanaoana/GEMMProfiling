// matrix_utils.cu - Implementation of matrix utilities
#include "../include/matrix_utils.cuh"
#include <stdio.h>
#include <math.h>

// ============================================================================
// HOST FUNCTION IMPLEMENTATIONS
// ============================================================================

// Host function to compute Frobenius norm of a full matrix
float compute_frobenius_norm_host(float* h_matrix, int n) {
    double norm_sq = 0.0;
    for (int i = 0; i < n * n; i++) {
        double val = h_matrix[i];
        norm_sq += val * val;
    }
    return (float)sqrt(norm_sq);
}

// Host function to standardize a matrix (A = A / ||A||_F) in place
void standardize_matrix_host(float* h_matrix, int n) {
    // Compute Frobenius norm on host
    double norm_sq = 0.0;
    for (int i = 0; i < n * n; i++) {
        double val = h_matrix[i];
        norm_sq += val * val;
    }
    double norm = sqrt(norm_sq);

    if (norm > 1e-12) {
        // Standardize in place
        for (int i = 0; i < n * n; i++) {
            h_matrix[i] /= norm;
        }
        printf("Matrix standardized: Frobenius norm was %.6e\n", norm);
    } else {
        printf("Warning: Matrix has very small Frobenius norm (%.6e), skipping standardization\n", norm);
    }
}

// Host function to scale matrix by a constant
void scale_matrix_host(float* h_matrix, int n, float scale_factor) {
    for (int i = 0; i < n * n; i++) {
        h_matrix[i] *= scale_factor;
    }
}

// Host function to estimate condition number of a full matrix
float estimate_condition_number_host(float* h_matrix, int n) {
    // Simple implementation using Frobenius norm approximation
    float norm = compute_frobenius_norm_host(h_matrix, n);

    if (norm < 1e-12f) return 1e12f;

    // Very rough estimate - more sophisticated methods would use SVD
    float det_approx = 1.0f;
    for (int i = 0; i < n; i++) {
        det_approx *= fabs(h_matrix[i * n + i]); // Diagonal product approximation
    }

    if (det_approx < 1e-12f) return 1e12f;

    return norm * norm / det_approx;
}