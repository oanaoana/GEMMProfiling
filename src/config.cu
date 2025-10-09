// config.cu - Configuration assignment and defaults
#include "../include/config.h"
#include <string.h>  // for strcmp
#include <cuda_runtime.h>  // for dim3
#include <type_traits>  // for template metaprogramming

// ============================================================================
// GLOBAL CONFIGURATION VARIABLES - DEFINITIONS
// ============================================================================

// Matrix sizes for benchmarking and testing
const int SIZES[] = DEFAULT_SIZES;
const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

// ============================================================================
// PRECISION AND UNIT ROUNDOFF FUNCTIONS - DEFINITIONS
// ============================================================================

// Unit roundoff values for different floating-point precisions
double unit_roundoff_fp32() { return ldexp(1.0, -24); }
double unit_roundoff_fp64() { return ldexp(1.0, -53); }
double unit_roundoff_tf32() { return ldexp(1.0, -10); }
double unit_roundoff_bf16() { return ldexp(1.0, -8); }
double unit_roundoff_fp16() { return ldexp(1.0, -10); }

// Template-based unit roundoff selection
template<typename T>
double get_unit_roundoff() {
    if constexpr (std::is_same_v<T, float>) {
        return unit_roundoff_fp32();
    } else if constexpr (std::is_same_v<T, double>) {
        return unit_roundoff_fp64();
    } else if constexpr (std::is_same_v<T, __half>) {
        return unit_roundoff_fp16();
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return unit_roundoff_bf16();
    } else {
        return unit_roundoff_fp32(); // Default fallback
    }
}

// Explicit template instantiations for the types you actually use
template double get_unit_roundoff<float>();
template double get_unit_roundoff<double>();
template double get_unit_roundoff<__half>();
template double get_unit_roundoff<__nv_bfloat16>();

// Helper functions that use the compile-time type configuration
double unit_roundoff_from_type() {
    if constexpr (std::is_same_v<COMPUTE_TYPE, float>) {
        return unit_roundoff_fp32();
    } else if constexpr (std::is_same_v<COMPUTE_TYPE, __half>) {
        return unit_roundoff_fp16();
    } else if constexpr (std::is_same_v<COMPUTE_TYPE, __nv_bfloat16>) {
        return unit_roundoff_bf16();
    } else {
        return unit_roundoff_fp32(); // fallback
    }
}

double unit_roundoff_accumulate_type() {
    if constexpr (std::is_same_v<ACCUMULATE_TYPE, float>) {
        return unit_roundoff_fp32();
    } else if constexpr (std::is_same_v<ACCUMULATE_TYPE, double>) {
        return unit_roundoff_fp64();
    } else {
        return unit_roundoff_fp32(); // fallback
    }
}
