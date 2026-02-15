#pragma once
/*
 * Phase 2 kernels have been fused into phase1_kernels.cuh.
 *
 * The fused single-pass architecture eliminates the separate Phase 2 step:
 * composition generation, pruning, and autoconvolution all happen in registers
 * within a single kernel, with zero intermediate DRAM traffic.
 */
