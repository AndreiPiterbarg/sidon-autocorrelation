
  ---
  STEP 3: Bottleneck-Specific Optimizations

  B1. Eliminate binary search [DONE]

  - What it changes: Replace find_best_bound (which calls run_single_level ~6x with different targets) with a single pass that computes the minimum effective bound across all compositions, this should be as efficiently written as possible, I want this method to be as quick as it can.
  - Why faster: For each composition, compute effective = max(test_val, asymmetry_bound). Global bound = min(effective) - correction(m). One full enumeration instead of ~6.
  - Difficulty: Trivial (30 lines).
  - Expected speedup: ~6x on binary search runs.
  - ACTUAL speedup: 8.6-10.5x (see below).

  Implementation: find_best_bound_direct() in core.py. Combines:
    1. Seeded running min from uniform composition (enables early pruning)
    2. Symmetry filter first (halves batch: canonical mask b <= rev(b))
    3. Asymmetry filter second on smaller batch (partial sum + compare)
    4. Precomputed window matrix W for vectorized test-value via matmul
    5. batch_size=50000 (optimal for running-min update frequency)

  Methods tested (12 variants benchmarked at production scale):
  Benchmark cases: n=2,m=100 (86M configs, d=4) and n=3,m=7 (41.5M configs, d=6).

                                                                     n=2,m=100  n=3,m=7   total
    v1_pure:          min(test_vals) - correction, no asymmetry        17.42s    36.35s    53.77s
    v2_asym_running:  asymmetry skip, running min starts at inf        15.18s    27.13s    42.31s
    v3_asym_seeded:   asymmetry skip, seeded from uniform              14.62s    26.99s    41.62s
    v4_fft_seeded:    FFT-based convolution + seeded asymmetry         26.64s    31.15s    57.79s  (FFT overhead bad for small d)
    v5_matmul_seeded: conv + matmul window max + seeded asymmetry      14.34s    24.44s    38.78s
    v6_precompW:      precomputed W matrix outside loop + seeded       14.37s    24.61s    38.99s
    v7_quadform:      outer product + quadratic form matrix            21.72s    37.65s    59.36s
    v8_einsum:        numpy einsum quadratic form                      32.36s    60.03s    92.39s  (einsum overhead)
    v9_sym+matmul:    symmetry first + v5                              12.00s    21.20s    33.20s  [WINNER]
    v10_sym+preW:     symmetry first + v6                              12.09s    21.21s    33.31s
    v11_asym1st+preW: asymmetry first, then symmetry + precomputed W   14.70s    21.25s    35.96s  [old winner]
    v12_asym1st+matm: asymmetry first, then symmetry + v5 matmul      12.61s    40.01s    52.62s

  Winner: v9 (symmetry first, asymmetry, matmul, seeded, batch_size=50k)
  Key insight at production scale: symmetry-first halves the batch before
  the more expensive conv+matmul step. At d=4 with 86M configs, this is
  18% faster than asymmetry-first (v11). At d=6, tied.

  Bonus: direct method always returns a bound (binary search returns None
  when the optimal bound falls outside its [lo, hi] search range).

  B2. Numba JIT for compute_test_values_batch

  - What it changes: Replace the Python for i in range(d): for j in range(d): loop and the window loop with a @numba.njit compiled function that iterates over cells and loop indices in compiled code.
  - Why faster: Eliminates Python dispatch overhead (~4x at d=4), numpy temporary array allocation, and enables CPU SIMD vectorization. For d=48, eliminates 5,688 Python loop iterations.
  - Concrete estimate: d=4: 2-3x speedup on test phase. d=6: 3-5x. d=48: 50-100x.
  - Risk: None (identical math, numba already in requirements.txt).
  - Difficulty: Moderate (rewrite function as element-wise loop with @njit(parallel=True)).
  - Expected speedup: 3-5x on test phase (d=4-6 range).

  B3. Streaming composition generation

  - What it changes: Replace the materialize-all-then-batch approach with a true generator that fills a fixed-size buffer and yields it.
  - Why faster: Peak memory drops from O(N_total) to O(batch_size). For n=2,m=100: from 2.6 GB to 6 MB. Enables running n=2,m=200 (11 GB currently impossible). Also eliminates the huge np.vstack(buf) at the end.
  - Concrete estimate: Memory improvement: 400x. Speed improvement: ~1.5-2x (avoids vstack + enables cache-friendly processing).
  - Risk: None (same compositions in same order, just not all in memory).
  - Difficulty: Moderate (rewrite _outer to yield incrementally).
  - Expected speedup: 1.5-2x on enum phase + unlocks larger params.

  B4. Numba JIT for composition generation

  - What it changes: Replace the Python recursion + numpy glue in _outer and _gen_compositions_inner3 with a compiled nested loop that fills a pre-allocated int32 array.
  - Why faster: Current d=6 generation: 39,711 Python->numpy roundtrips, each allocating small arrays. Numba: single compiled nested loop writing to contiguous memory. Throughput goes from 1-2 Mrows/s to 50-200 Mrows/s.    
  - Concrete estimate: d=6 enum drops from 19s to ~0.2-1s.
  - Risk: None.
  - Difficulty: Moderate (nested loops in numba, combined with streaming buffer).
  - Expected speedup: 10-50x on enum phase for d>=6.

  B5. Reversal symmetry

  - What it changes: For each composition a, only check the lexicographically smaller of {a, reverse(a)}.
  - Why faster: Exactly halves the number of compositions to enumerate and test. The test value is symmetric because conv(a)[k] = conv(reverse(a))[conv_len-1-k], and the window max is taken over a symmetric set of windows. 
  - Concrete estimate: Cuts all phases (enum, asym, test) in half.
  - Risk: Must correctly handle palindromes (a = reverse(a)) — check once, not zero times. Needs a unit test.
  - Difficulty: Moderate (filter during enumeration or post-filter each batch).
  - Expected speedup: ~2x across the board.

  B6. Early window termination

  - What it changes: In compute_test_values_batch, once a config's running max exceeds prune_target, stop checking more windows for that config.
  - Why faster: Most configs are pruned by easy windows (e.g. ell=2 or ell=d). With numba (B2), this is a simple if inside the per-cell loop. Without numba, it's hard to exploit in vectorized code.
  - Concrete estimate: For c_target=1.0 at n=2,m=50, ~59% of cells are test-pruned. If half are pruned within 5 window iterations (out of 18), the average iteration count drops from 18 to ~11. Saves ~40% of window work.    
  - Risk: None (produces identical results).
  - Difficulty: Trivial with numba (B2), hard without.
  - Expected speedup: ~1.5x on test phase (synergizes with B2).

  B7. Remove unnecessary .copy() on line 168

  - What it changes: ws = cumconv[:, s_hi].copy() → ws = cumconv[:, s_hi] - cumconv[:, s_lo - 1] with conditional, or restructure to avoid the copy.
  - Why faster: Eliminates one array allocation per window iteration. For 50M cells x 18 iterations = 900M copy ops.
  - Concrete estimate: ~5-10% speedup on window sub-phase.
  - Risk: None.
  - Difficulty: Trivial (1-line change).
  - Expected speedup: ~1.05x on test phase.

  ---
  STEP 4: Mathematical Structure Opportunities



  M2. Convolution structure — reorganize by output index

  The autoconvolution conv[k] = sum_{i+j=k} a[i]*a[j] currently uses d^2 iterations (over all (i,j) pairs). Reorganizing by output index k uses only (2d-1) iterations, where iteration k sums over min(k+1, d, 2d-1-k) pairs. 
  For d=4: saves from 16 to 7 iterations with an average of 2.3 multiply-adds per iteration. For d=6: 11 iterations instead of 36. This is a 2-3x reduction in iteration count for the convolution sub-phase, applicable even  
  without numba.

  M3. Window checking — precompute optimal window ordering

  The test value max_{ell,k} window_sum/(4*n*ell) can be computed more efficiently if we order windows by their "pruning power." Windows with small ell (especially ell=2, which reduces to just examining individual conv     
  values) tend to produce the largest test values. Checking windows in decreasing-power order enables early termination (B6).

  Specifically, the best window for near-uniform distributions is ell=d (full width), and the best for concentrated distributions is ell=2 (single conv value). Checking ell=2 first (d-1 windows to check) handles
  concentrated configs quickly; then ell=d handles near-uniform configs. This pairs with B6.

  M4. Cells that are provably non-binding

  The compositions furthest from uniform — those where most mass is in one bin — always have very high test values (concentrated mass → huge conv peak). The asymmetry argument already handles extreme left/right imbalance,  
  but even for balanced-but-concentrated configs (e.g. most mass in the two center bins), test values are large.

  The hardest cell to prune is near the uniform distribution, where the test value is minimized. For the uniform config with d bins, the test value is exactly sum_{i+j in window} 4 / (4n*ell). At n=2 (d=4), the uniform test   value is 1.25. This means c_target up to 1.25 - correction(m) can always be proven at n=2 regardless of m, just from the uniform cell. The near-uniform cells (small perturbations) have test values close to 1.25, so they 
  are the binding constraint.

  This suggests a neighborhood search: instead of enumerating all C(S+d-1, d-1) compositions, enumerate only those within some L1 distance of the uniform distribution. Compositions far from uniform are provably non-binding.   The number of compositions within L1 distance r of uniform is O(r^{d-1}), much smaller than the full simplex. This requires proving a lower bound on test values for compositions outside the ball.

  M5. Closed-form test value near uniform

  For the uniform distribution a_i = 2 (for all i), the test value at window (ell, k) is:
  (1/(4n*ell)) * sum_{k <= i+j <= k+ell-2} a_i*a_j = (1/(4n*ell)) * 4 * |{(i,j): 0<=i,j<d, k<=i+j<=k+ell-2}|

  This count has a closed form: it's the number of lattice points in a trapezoid. For small perturbations a = 2 + epsilon, the test value changes as a computable quadratic form in epsilon. This could give an analytical     
  bound on the minimum test value over the near-uniform neighborhood, potentially eliminating the need to enumerate those cells entirely.

  M6. Simplex parameterization alternative

  The current lattice B_{n,m} (compositions of S into d parts with granularity 1) has C(S+d-1, d-1) points. An alternative parameterization using a coarser lattice near the boundary (where test values are large and pruning 
  is easy) and finer near the center (where pruning is hard) could reduce the total grid size while maintaining the correction bound. This is essentially the multi-scale approach mentioned in CLAUDE.md.

  ---
  STEP 5: Computational Infrastructure Opportunities

  I1. Vectorization across cells

  Both the convolution and window-checking loops operate independently per cell. The current code vectorizes across cells (batch dimension) for each loop iteration. This is good for numpy but suboptimal: each Python loop   
  iteration dispatches a numpy op over the full batch.

  With numba, we can vectorize across loop iterations instead: process each cell completely (all 34 loop iterations) before moving to the next. This is more cache-friendly (the d-sized arrays fit in L1 cache) and eliminates   Python loop overhead.

  With @njit(parallel=True) and prange, numba can also parallelize across cells on multiple cores.

  I2. Parallelism structure

  - Embarrassingly parallel: different compositions are independent. Batch processing naturally parallelizes.
  - Sequential dependency: the binary search target depends on previous iteration results. But with B1 (single-pass), this dependency is eliminated.
  - Reduction: the global min test value is a reduction over all cells. Can use parallel reduction.
  - Recommendation: numba.prange over the batch dimension for test value computation. For composition generation, use multiple threads filling different segments of the output buffer.

  I3. Solver choices

  There is no iterative solver in this code — all operations are direct computation (convolution + maximum). No LP/SDP/Newton needed. The computation is purely arithmetic, which is ideal for compiled code (numba, C, or     
  GPU).

  For GPU (future): the test value computation for a batch of B cells is a perfect GPU kernel — each thread handles one cell, computing conv and window max independently. At 86M cells with d=4, a modern GPU could process   
  all cells in <1 second.

  I4. Numba vs C vs numpy broadcasting

  ┌──────────────────────────┬──────────┬──────────┬────────────────┐
  │         Approach         │ Enum d=6 │ Test d=4 │ Implementation │
  ├──────────────────────────┼──────────┼──────────┼────────────────┤
  │ Current (Python+numpy)   │ 19.2s    │ 15.6s    │ baseline       │
  ├──────────────────────────┼──────────┼──────────┼────────────────┤
  │ Numpy broadcasting (M2)  │ ~15s     │ ~10s     │ trivial        │
  ├──────────────────────────┼──────────┼──────────┼────────────────┤
  │ Numba JIT                │ ~0.5s    │ ~3s      │ moderate       │
  ├──────────────────────────┼──────────┼──────────┼────────────────┤
  │ Numba parallel (4 cores) │ ~0.15s   │ ~1s      │ moderate       │
  ├──────────────────────────┼──────────┼──────────┼────────────────┤
  │ C extension              │ ~0.3s    │ ~2s      │ hard           │
  ├──────────────────────────┼──────────┼──────────┼────────────────┤
  │ GPU (CUDA)               │ ~0.01s   │ ~0.1s    │ hard           │
  └──────────────────────────┴──────────┴──────────┴────────────────┘

  Numba is the clear winner for effort-to-speedup ratio since it's already a dependency.

  ---
  STEP 6: Ranked Optimization Plan

  Ranked by (expected speedup / implementation effort):

  ┌──────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────┬─────────┬─────────────────┬──────────────────────┐ 
  │ Rank │                                                 Optimization                                                 │                     Speedup                     │  Time   │  Dependencies   │   Changes output?    │ 
  ├──────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────┼─────────────────┼──────────────────────┤ 
  │ 1    │ B1: Eliminate binary search — single-pass min over effective bounds                                          │ 6x on search runs                               │ 1 hour  │ None            │ No                   │ 
  ├──────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────┼─────────────────┼──────────────────────┤ 
  │ 2    │ B2+B6: Numba JIT for test values + early termination — compile conv+window loops with per-cell short-circuit │ 3-5x on test phase                              │ 3 hours │ None            │ No                   │ 
  ├──────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────┼─────────────────┼──────────────────────┤ 
  │ 3    │ B4+B3: Numba streaming composition generation — compiled nested loops, fixed-size buffer, no full            │ 10-50x on enum phase (d>=6), 400x memory        │ 4 hours │ None            │ No                   │ 
  │      │ materialization                                                                                              │ reduction                                       │         │                 │                      │ 
  ├──────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────┼─────────────────┼──────────────────────┤ 
  │ 4    │ B5: Reversal symmetry — canonical representatives only                                                       │ 2x across all phases                            │ 3 hours │ None            │ No                   │ 
  ├──────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────┼─────────────────┼──────────────────────┤ 
  │ 5    │ M2: Reorganize conv by output index — (2d-1) grouped ops instead of d^2                                      │ 1.5x on conv sub-phase                          │ 1 hour  │ None            │ No                   │ 
  ├──────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────┼─────────────────┼──────────────────────┤ 
  │ 6    │ I2: Numba parallel (prange) — multicore test value computation                                               │ Nx (N=core count) on test phase                 │ 1 hour  │ B2              │ No                   │ 
  ├──────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────┼─────────────────┼──────────────────────┤ 
  │ 7    │ B7: Remove .copy()                                                                                           │ 1.05x on window sub-phase                       │ 5 min   │ None            │ No                   │ 
  ├──────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────┼─────────────────┼──────────────────────┤ 
  │ 8    │ M4: Neighborhood enumeration — only check cells near uniform                                                 │ 10-100x cell reduction (needs proof)            │ 1-2     │ Needs math      │ No (if proof         │ 
  │      │                                                                                                              │                                                 │ days    │ proof           │ correct)             │ 
  └──────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────┴─────────┴─────────────────┴──────────────────────┘ 

  Projected speedups

  Top 3 items (B1 + B2/B6 + B4/B3):

  Current heavy mode: 385s
  - Binary search elimination (B1): 385s → ~57s (only 1 iteration each)
  - Numba test values (B2+B6): test phase 15.6s → ~4s (n=2,m=100), 4.3s → ~1s (n=3,m=5)
  - Numba streaming enum (B4+B3): enum phase 14s → ~1s (n=2,m=100), 19s → ~0.5s (n=3,m=5)

  Projected heavy mode with top 3: n=2,m=100 single run ~7s + n=3,m=5 single run ~2s = ~9s (from 385s = ~43x speedup).

  In the benchmark: heavy mode should drop from 385s to ~9s. Light mode should drop from 102s to ~5s.

  All items (1-7):

  Adding symmetry (2x), output-index conv (1.5x on sub-phase), prange (4x on test), copy removal (1.05x):

  Projected heavy mode: ~9s → ~2-3s (~130-190x total speedup).

  More importantly, the memory reduction from B3 unlocks larger parameters: n=2,m=500 (currently 11 GB, infeasible) becomes feasible at 6 MB. n=3,m=20 (currently impossible) becomes tractable. These larger parameters       
  produce tighter bounds.

  Where it shows in the benchmark:
  - Light mode: from 102s to ~3s. All configs complete in under 1s each.
  - Heavy mode: from 385s to ~3s. Can upgrade heavy mode to include n=2,m=500 and n=3,m=20 in the same time budget, pushing the bound significantly closer to 1.28.