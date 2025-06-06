#include "utils.cuh"
#include <stdio.h>

void fill_matrix(float *mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void verify_result(float *A, float *B, float *C, int N) {
    // Use more appropriate epsilon for float-to-double comparison
    float eps = 1e-6;
    float max_rel_error = 0.0f;
    float sum_rel_error = 0.0f;
    int error_count = 0;

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            // Calculate reference in double precision
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += (double)A[row * N + k] * (double)B[k * N + col];
            }

            // Store CPU result for printing
            float cpu_result = (float)sum;

            // Compare GPU float result with double precision reference
            double abs_error = fabs((double)C[row * N + col] - sum);
            double rel_error = abs_error / (fabs(sum) > 1e-10 ? fabs(sum) : 1e-10);

            // Record statistics in double precision
            max_rel_error = fmax(max_rel_error, (float)rel_error);
            sum_rel_error += (float)rel_error;

            // Convert error threshold to match the precision of abs_error
            if (abs_error > (double)eps) {
                error_count++;
                if (error_count <= 5) { // Limit output to first 5 errors
                    printf("Mismatch at (%d, %d): GPU = %f, CPU = %f, Rel Error = %e\n",
                          row, col, C[row * N + col], cpu_result, rel_error);
                }
            }
        }
    }

    printf("Max relative error: %e\n", max_rel_error);
    printf("Average relative error: %e\n", sum_rel_error / (N * N));
    printf("Number of elements with error > %e: %d (%.2f%%)\n",
           eps, error_count, 100.0f * error_count / (N * N));

    if (error_count == 0)
        printf("Result verified: correct within epsilon %e.\n", eps);
}