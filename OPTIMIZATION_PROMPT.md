# GPU Optimization Task: Sidon Autocorrelation Branch-and-Prune

You are an expert CUDA/GPU optimization engineer. Your task is to maximize the throughput of a mathematically-rigorous branch-and-prune algorithm running on an NVIDIA A100-SXM4-80GB GPU (80 GB HBM2e, 108 SMs, compute capability 8.0, 2039 GB/s bandwidth, 19.5 TFLOPS FP64).

## The Goal

We are trying to prove that the autoconvolution constant c >= 1.20 using the Cloninger-Steinerberger exhaustive verification algorithm (arXiv:1403.7988). The production entry point is `run_proof.py`. The algorithm works hierarchically:

1. **Level 0** (n=3, d=6 bins, m=50): Enumerate all ~664 billion lattice points in B_{3,50}. Prune via asymmetry + canonical filtering + windowed autoconvolution test. Extract survivors.
2. **Level 1** (n=6, d=12 bins): For each Level 0 survivor parent, generate all child refinements via Cartesian product splits. Each parent component b_i splits into (c_{2i}, c_{2i+1}) with c_{2i}+c_{2i+1}=2*b_i, giving prod(2*b_i+1) children per parent. Prune children. Extract survivors.
3. **Level 2** (n=12, d=24 bins): Same refinement on Level 1 survivors.
4. **Level 3** (n=24, d=48 bins): Same refinement on Level 2 survivors.

If all configurations are eliminated at any level, the proof is complete: c >= 1.20 is formally proven.

### Current Performance (A100, target=1.20, m=50)

- Level 0: ~664B points, completes in ~7s at ~70B configs/s
- Level 0 produces **1.55 billion survivors** (far too many for efficient Level 1 refinement)
- Level 1 refinement of 1.55B parents generates an astronomical number of children
- The algorithm is currently **not feasible** at target=1.20 due to the massive survivor count at Level 0

## Your Two-Phase Assignment

### PHASE 1: Create a Benchmark Test File

Create `benchmark_gpu.py` in the project root. This script must:

1. **Run on the A100 GPU via RunPod** (Linux, no TDR restrictions)
2. **Test ALL steps of the algorithm** so we can measure each bottleneck:
   - `gpu_find_best_bound_direct` (D=4 and D=6) at multiple m values
   - `gpu_run_single_level` (counting mode, no extraction) at multiple (n, m, c_target) combos
   - `gpu_run_single_level` with extraction (measure extraction overhead)
   - `refine_parents` with synthetic parents (measure refinement throughput)
   - Memory usage tracking at each step
3. **Measure and report**:
   - Throughput (configs/second) for each kernel type
   - Pruning effectiveness (% pruned at each stage: asymmetry, canonical, test-value)
   - Time breakdown (kernel time vs host overhead vs memcpy)
   - Register pressure / occupancy (via reported grid/block sizes)
   - Survivor counts at various target thresholds (1.28, 1.25, 1.22, 1.20, 1.15, 1.10)
4. **Be quick** — total runtime should be under 3 minutes on A100 (use small m values for expensive tests)
5. **Output structured results** as JSON for easy comparison before/after optimization

### PHASE 2: Implement Optimizations

After analyzing the benchmark results, implement speedups. Here are the specific bottlenecks and optimization opportunities to investigate:

#### A. Kernel-Level Optimizations (CUDA .cuh files)

**1. Block size tuning** (`device_helpers.cuh` line 63):
- `FUSED_BLOCK_SIZE` is currently 256 for all kernels. A100 supports up to 1024 threads/block.
- Profile whether 128, 256, or 512 gives best occupancy. For the D=6 kernel with its inner c4 loop, divergence patterns may favor smaller blocks.

**2. Grid size tuning** (`host_find_min.cuh` line 72, `host_prove.cuh` line 92):
- Currently capped at `108 * 32 = 3456` blocks. For 664B work items with grid-stride loop, each thread processes ~750K items.
- Consider whether more blocks (larger grid) with fewer iterations per thread improves latency hiding.

**3. Reduce register pressure in D=6 fused kernels** (`phase1_kernels.cuh`):
- The D=6 path declares `conv_t conv[11]` (11 registers), `int cc[6]`, `int my_cfg[8]`, plus loop variables.
- Consider: Can the convolution be computed incrementally as c4 varies (since only c4 and c5 change in the inner loop)? The autoconvolution terms involving c4/c5 are a small subset that can be added/subtracted from a precomputed base.

**4. Incremental convolution in D=6 inner loop** (`phase1_kernels.cuh` lines 293-356):
- Currently recomputes full 6-element autoconvolution for every (c4, c5) pair
- The terms involving only c0,c1,c2,c3 are constant across the c4 loop
- Precompute base_conv[11] from (c0,c1,c2,c3), then for each c4: add contributions from c4 and c5=r3-c4
- This reduces inner loop work from O(36) multiplies to O(11) adds

**5. Refinement kernel register pressure** (`refinement_kernel.cuh`):
- For D_CHILD=12: `conv[23]` (23 long long = 46 registers), `c[12]`, `local_thresh[11]`, `inv_norm_arr[11]` — massive register pressure
- For D_CHILD=24: `conv[47]` — 94 registers just for convolution!
- Consider: use shared memory for the conv[] array instead of registers for large D_CHILD
- Or: tile the autoconvolution computation to reduce peak register usage

**6. Early exit optimization in window scan** (`phase1_kernels.cuh`, `refinement_kernel.cuh`):
- Currently scans windows from ell=D down to ell=2
- The ell=2 window (single-element max squared) is cheapest and prunes many configs
- Consider reversing: check ell=2 FIRST (just find max element), then ell=3, etc.
- Since we already do an FP32 ell=2 pre-check, the benefit may be small, but for the refinement kernel at D=12/24, it could help

**7. Shared memory for parent configs in batched refinement** (`refinement_kernel.cuh` lines 361-490):
- The batched kernel reads parent configs from global memory via `const int* pB = all_parents + pidx * D_PARENT`
- Adjacent threads likely share the same parent, so L2 cache should help
- But explicit shared memory caching could reduce latency for the divmod decomposition

#### B. Algorithmic / Architectural Improvements

**8. Stronger FP32 pre-pruning to reduce full autoconvolution work**:
- The current FP32 checks (asymmetry, pair-sum, max-element) catch some configs cheaply
- Additional FP32 bounds:
  - Sum-of-squares bound: sum(c_i^2) provides a lower bound on the ell=2 window max
  - Sliding triple-sum: for D=6, check partial sums of 3 consecutive bins
  - These are cheap (FP32 multiply-add) and could prune configs before the expensive INT64 autoconvolution

**9. Chunked Level 0 with streaming**:
- Currently Level 0 is a single monolithic kernel launch
- Consider overlapping: launch kernel on chunk A, while memcpy results from previous chunk B
- Use CUDA streams to overlap compute and memory transfers

**10. Survivor compaction**:
- Currently survivors are written via `atomicAdd` which creates contention when survivor counts are high (1.55B at target=1.20)
- Consider: warp-level vote + warp-level atomic to reduce contention
- Or: per-block survivor buffer with final compaction pass


---

## Repository Layout

```
sidon-autocorrelation/
├── run_proof.py                         # Main proof script (hierarchical, what we're optimizing FOR)
├── benchmark_gpu.py                     # YOUR NEW FILE: benchmark all algorithm steps
├── cloninger-steinerberger/
│   ├── core.py                          # Re-exports
│   ├── test_values.py                   # CPU autoconvolution (Numba)
│   └── gpu/
│       ├── kernels.cu                   # CUDA entry point (includes all .cuh)
│       ├── device_helpers.cuh           # CUDA_CHECK, atomicMinDouble, binary_search_le, shfl_down_double, FUSED_BLOCK_SIZE=256
│       ├── phase1_kernels.cuh           # fused_find_min<D>, fused_prove_target<D> (D=4,6)
│       ├── phase2_kernels.cuh           # (empty)
│       ├── host_find_min.cuh            # find_best_bound_direct_d4/d6 host orchestration
│       ├── host_prove.cuh               # run_single_level_d4/d6[_extract] host orchestration
│       ├── refinement_kernel.cuh        # refine_prove_target<D_CHILD>, refine_prove_target_batched<D_CHILD>
│       ├── host_refine.cuh              # refine_parents_impl<D_CHILD>, batched + chunked approaches
│       ├── dispatch.cuh                 # extern "C" wrappers for ctypes
│       ├── wrapper.py                   # Python ctypes interface
│       ├── solvers.py                   # GPU solver API (gpu_find_best_bound_direct, gpu_run_single_level)
│       ├── build.py                     # nvcc build script
│       └── __init__.py                  # Package init
├── gpupod/                              # RunPod A100 pod management
│   ├── cli.py                           # start/sync/build/run/teardown commands
│   ├── config.py                        # GPU_TYPE, DOCKER_IMAGE, COST_PER_HOUR=$1.49
│   ├── pod_manager.py                   # create/terminate pods
│   ├── remote.py                        # SSH execution
│   ├── session.py                       # Session state persistence
│   └── sync.py                          # Code sync via tar over SSH
└── data/                                # Checkpoints, logs, results
```

## Build & Run Instructions

From local Windows machine:
```bash
python -m gpupod start      # Create A100 pod, sync, build
python -m gpupod sync       # Re-sync after edits
python -m gpupod build      # Recompile CUDA
python -m gpupod run benchmark_gpu.py   # Run benchmark
python -m gpupod teardown   # MANDATORY: ALWAYS destroy pod when done
```

## Key Algorithm Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| m | 50 | Grid resolution (paper used 50) |
| n_start | 3 | Starting n (d=2n=6 bins) |
| c_target | 1.20 | Target lower bound to prove |
| correction | 2/m + 1/m² = 0.0404 | Discretization error |
| T (threshold) | 1.2404 | c_target + correction |
| S | 4*n*m = 600 | Total mass (level 0) |
| Grid size | C(605,5) ≈ 664B | Level 0 lattice points |

## Key Data Types

- Compositions: `int32` arrays (bin values, each 0..S)
- Convolutions: `int64` (when S*S > 2B) or `int32`
- Test values: `float64` (FP64 for mathematical rigor)
- FP32 used only for cheap pre-pruning (safe because FP32 bounds are conservative)

## Critical Constraints

1. **Mathematical correctness is paramount.** Every pruned configuration must be genuinely provable. The integer threshold trick (`int_thresh[ell-2] = floor(thresh * m² * 4 * n * ell)`) is the safe way to avoid FP64 rounding errors in the inner loop.

2. **Memory: A100 has 80 GB HBM2e.** Survivor extraction at target=1.20 with 1.55B survivors × 6 ints × 4 bytes = ~37 GB. This is feasible but tight. At Level 1, D_CHILD=12, so 12 ints × 4 bytes = 48 bytes per survivor.

3. **Always terminate RunPod pods after testing.**

4. **Do NOT add yourself as a co-contributor in commits.**

## What to Deliver

1. `benchmark_gpu.py` — comprehensive GPU benchmark that tests all algorithm steps
2. Modified CUDA kernels (`.cuh` files) with optimizations
3. Any necessary Python wrapper changes
4. Before/after benchmark results showing speedup
5. Summary of what changed and why

## Prioritization

Focus on optimizations that will make the **actual `run_proof.py --target 1.20 --m 50` run complete faster**. The bottleneck is:
1. Level 0: reducing survivor count (stronger pruning) or faster extraction
2. Level 1 refinement: throughput for processing billions of parent→child refinements
3. Higher levels: register pressure and autoconvolution cost for D=24, D=48

