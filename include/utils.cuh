#include <cuda_runtime.h>

void fill_matrix(float *mat, int N);
void verify_result(float *A, float *B, float *C, int N);