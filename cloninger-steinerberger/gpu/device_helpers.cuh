#pragma once
/*
 * Device helper functions for GPU kernels.
 *
 * Contents:
 *   - CUDA_CHECK error macro
 *   - atomicMinDouble (CAS idiom for positive FP64)
 *   - binary_search_le (sorted array lookup)
 *   - shfl_down_double (warp shuffle for double)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

/* ================================================================
 * Error checking macro
 * ================================================================ */
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* ================================================================
 * Device helpers
 * ================================================================ */

/* AtomicMin for double using CAS idiom.
 * For positive FP64 values, integer bit ordering matches FP64 ordering,
 * so compare-and-swap on the bit pattern works correctly. */
__device__ double atomicMinDouble(double* addr, double val) {
    unsigned long long int* addr_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val)
            return __longlong_as_double(assumed);
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* Binary search: find largest i such that arr[i] <= val.
 * arr must be sorted ascending, arr[0] = 0. */
__device__ int binary_search_le(const long long* arr, int n, long long val) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (arr[mid] <= val) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

/* Block size for fused kernels */
#define FUSED_BLOCK_SIZE 256

/* Warp shuffle for double (split into two 32-bit halves) */
__device__ __forceinline__ double shfl_down_double(
    unsigned int mask, double val, int offset)
{
    int lo = __double2loint(val);
    int hi = __double2hiint(val);
    lo = __shfl_down_sync(mask, lo, offset);
    hi = __shfl_down_sync(mask, hi, offset);
    return __hiloint2double(hi, lo);
}
