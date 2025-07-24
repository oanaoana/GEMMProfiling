#ifndef UTILS_CUH
#define UTILS_CUH
#include <cuda_runtime.h>

void fill_matrix(float *mat, int N);
void verify_result(float *A, float *B, float *C, int N);

void printDevicePerformanceInfo();
void printCacheInfo();
void check_occupancy();

// Global pitch variables (declared elsewhere, used here)
extern int g_pitch_A, g_pitch_C;
#endif // UTILS_CUH