# Current Bottlenecks — Sidon Autocorrelation GPU Prover

## The Problem

We are proving lower bounds on the autoconvolution constant *c* via the Cloninger–Steinerberger exhaustive branch-and-prune algorithm. The goal is to show c ≥ 1.20 by eliminating every discrete mass distribution in the lattice B_{n,m}.

## Our Setup

- **Hardware**: NVIDIA A100-SXM4-80GB on RunPod ($1.64/hr, $10/session budget)
- **Parameters**: Level 0 at n=3, m=50 → 664 billion grid points (d=6 bins)
- **Current throughput**: ~93 billion configs/sec at Level 0 (~7 seconds total)
- **Pipeline**: Level 0 enumerates and prunes → survivors feed into Level 1 refinement (n=6, d=12) → Level 2 (n=12, d=24) → Level 3 (n=24, d=48)

## Bottleneck 1: Survivor Explosion at Level 0

This is the dominant blocker. At target c=1.20, Level 0 produces **1.55 billion survivors** (0.23% survival rate). Each survivor becomes a parent at Level 1, where it spawns up to ∏(2·b_i + 1) refinement children — potentially astronomical numbers. Even at 93B configs/sec, the total Level 1 workload is computationally infeasible at this survivor count.

**Root cause**: The FP32 pre-pruning pipeline (asymmetry check, pair-sum bounds, max-element test) is not aggressive enough to kill more configs before the expensive INT64 autoconvolution step. Stronger cheap tests (sum-of-squares bound, sliding partial-sum checks) could prune 10–20% more configs early.

## Bottleneck 2: Register Pressure in Refinement Kernels (D=24, D=48)

The autoconvolution array `conv[2D-1]` lives in registers. For D=24 this is 47 `long long` values = **94 registers just for convolution**, leaving almost nothing for loop variables, configs, and thresholds. This crushes occupancy to an estimated 20–30% on A100, severely underutilizing the hardware.

For D=48 (Level 3) the situation is even worse — the kernel may not even be viable without restructuring.

**Impact**: Low occupancy means the GPU cannot hide memory latency, leaving SMs idle during stalls.

## Bottleneck 3: Suboptimal Window Scan Order

The window scan currently checks from ℓ=D down to ℓ=2. But ℓ=2 (single-element max squared) is the cheapest test and prunes a large fraction of configs. Checking it first would enable earlier exits, saving the cost of computing larger windows that are never needed.

## Bottleneck 4: Fixed Block Size (256)

All kernels use `FUSED_BLOCK_SIZE=256`. The A100 supports up to 1024 threads/block. The optimal block size depends on register usage and shared memory per kernel — no profiling has been done to find the best trade-off. The D=6 kernel with its inner c4 loop may have divergence patterns that favor different block sizes than the D=4 kernel.

## Bottleneck 5: No Compute/Transfer Overlap

Level 0 runs as a monolithic kernel launch. There is no use of CUDA streams to overlap computation on one chunk with result extraction from another. This leaves the PCIe bus idle during compute and the GPU idle during transfers.

## Summary

| Bottleneck | Severity | Effect |
|---|---|---|
| Survivor explosion (1.55B at target 1.20) | **Critical** | Makes Level 1 infeasible |
| Register pressure (D≥24 kernels) | **High** | 20–30% occupancy, wasted SMs |
| Window scan order (large-to-small) | Medium | Missed early-exit opportunities |
| Fixed block size (no tuning) | Medium | Possibly suboptimal occupancy |
| No stream overlap | Low | Minor latency, Level 0 already fast |
