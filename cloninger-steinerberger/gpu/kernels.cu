/*
 * GPU kernels for Cloninger-Steinerberger branch-and-prune algorithm.
 *
 * Fused single-pass architecture:
 *   Each thread generates compositions from a flat index, applies FP32 pruning,
 *   computes INT32/INT64 autoconvolution, and reduces — all in registers.
 *   Zero intermediate DRAM traffic between generation and evaluation.
 *
 * Templated on D (number of bins). Instantiated for D=4, 6.
 *
 * Module layout:
 *   device_helpers.cuh   - Error checking, atomic ops, warp shuffles
 *   phase1_kernels.cuh   - Fused kernels: find_min + prove_target (templated on D)
 *   phase2_kernels.cuh   - (empty — kernels fused into phase1_kernels.cuh)
 *   host_find_min.cuh    - Host pipeline for find_best_bound_direct
 *   host_prove.cuh       - Host pipeline for run_single_level
 *   dispatch.cuh         - extern "C" dispatch functions for Python ctypes
 */

#include "device_helpers.cuh"
#include "phase1_kernels.cuh"
#include "phase2_kernels.cuh"
#include "host_find_min.cuh"
#include "host_prove.cuh"
#include "refinement_kernel.cuh"
#include "host_refine.cuh"
#include "dispatch.cuh"
