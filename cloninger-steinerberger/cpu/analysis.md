
  ---
  STEP 2: Profile Analysis

  Phase breakdown (from instrumented benchmark runs)

  ┌────────────────────────────┬───────┬───────┬───────┬────────┬───────┬────────┬───────┬──────────┐
  │           Config           │ Total │ Enum  │ Enum% │  Asym  │ Asym% │  Test  │ Test% │ Mcells/s │
  ├────────────────────────────┼───────┼───────┼───────┼────────┼───────┼────────┼───────┼──────────┤
  │ n=2,m=10  (92K cells, d=4) │ 0.08s │ 0.07s │ 85%   │ 0.003s │ 4%    │ 0.009s │ 12%   │ 6.4      │
  ├────────────────────────────┼───────┼───────┼───────┼────────┼───────┼────────┼───────┼──────────┤
  │ n=2,m=50  (10.8M, d=4)     │ 3.59s │ 1.77s │ 49%   │ 0.31s  │ 9%    │ 1.51s  │ 42%   │ 5.0      │
  ├────────────────────────────┼───────┼───────┼───────┼────────┼───────┼────────┼───────┼──────────┤
  │ n=2,m=100 (86M, d=4)       │ 33.1s │ 14.0s │ 42%   │ 3.5s   │ 10%   │ 15.6s  │ 47%   │ 3.9      │
  ├────────────────────────────┼───────┼───────┼───────┼────────┼───────┼────────┼───────┼──────────┤
  │ n=3,m=3  (749K, d=6)       │ 2.50s │ 2.15s │ 86%   │ 0.02s  │ 1%    │ 0.33s  │ 13%   │ 1.8      │
  ├────────────────────────────┼───────┼───────┼───────┼────────┼───────┼────────┼───────┼──────────┤
  │ n=3,m=5  (8.3M, d=6)       │ 23.8s │ 19.2s │ 81%   │ 0.34s  │ 1%    │ 4.31s  │ 18%   │ 1.4      │
  └────────────────────────────┴───────┴───────┴───────┴────────┴───────┴────────┴───────┴──────────┘

  Sub-phase breakdown within compute_test_values_batch

  ┌──────────────────────┬─────┬─────┬──────────────────┬──────────────────┐
  │      Sub-phase       │ d=4 │ d=6 │ Loop iters (d=4) │ Loop iters (d=6) │
  ├──────────────────────┼─────┼─────┼──────────────────┼──────────────────┤
  │ dtype convert        │ 10% │ 5%  │ -                │ -                │
  ├──────────────────────┼─────┼─────┼──────────────────┼──────────────────┤
  │ Autoconvolution loop │ 31% │ 35% │ 16 (d^2)         │ 36               │
  ├──────────────────────┼─────┼─────┼──────────────────┼──────────────────┤
  │ Prefix sums          │ 13% │ 9%  │ -                │ -                │
  ├──────────────────────┼─────┼─────┼──────────────────┼──────────────────┤
  │ Window max loop      │ 47% │ 51% │ 18               │ 45               │
  └──────────────────────┴─────┴─────┴──────────────────┴──────────────────┘

  Key observations

  1. Composition enumeration is the #1 bottleneck (42-86% of runtime). For d>=6, it consumes >80% because the Python recursion over outer dimensions makes ~40K calls to _gen_compositions_inner3, each doing numpy array
  allocation + column_stack + tile + hstack. Throughput drops from 16-24 Mrows/s (d=4) to 1-2 Mrows/s (d=6).
  2. Test value computation is #2 (12-47% of runtime). The two Python-level for loops (conv: d^2 iters, windows: ~d^2/2 iters) each do vectorized numpy ops but pay Python dispatch overhead per iteration. For d=48 (the
  paper's scale), these loops would have 5,688 iterations — pure Python overhead would dominate.
  3. All compositions are materialized in memory before processing. At n=2,m=100: 2.6 GB just for the composition array (86M x 4 x int32). This caps the maximum problem size we can run.
  4. Binary search multiplies cost by ~6x — each iteration regenerates all compositions from scratch. The n=2,m=100 binary search does 6 iterations x 33s = 199s, all doing identical enumeration work.
  5. The .copy() call on line 168 allocates a fresh array on every window iteration. With 18 window iterations over 50M cells, that's 18 x 50M x 8 bytes = 7.2 GB of throwaway allocation.
  6. Asymmetry pruning is cheap (1-10%) — it's a simple sum + comparison. Not a bottleneck.

  ---
  STEP 3: Numba JIT for compute_test_values_batch (B2 optimization)

  Implementation: Replaced Python for-loops (conv accumulation + window max) with
  @numba.njit(parallel=True, cache=True) fused function using prange over batch rows.

  Methods tested:
  ┌─────────────────────────────┬──────────────────┬──────────────────┬───────────────┐
  │         Approach            │ d=4 B=50K (ms)   │ d=6 B=50K (ms)   │   Notes       │
  ├─────────────────────────────┼──────────────────┼──────────────────┼───────────────┤
  │ Baseline (Python loops)     │ 14.8             │ 32.9             │ Original      │
  ├─────────────────────────────┼──────────────────┼──────────────────┼───────────────┤
  │ A: JIT conv only (serial)   │  5.7             │ 14.2             │ Conv loop JIT │
  ├─────────────────────────────┼──────────────────┼──────────────────┼───────────────┤
  │ B: Fused JIT (serial)       │  4.4             │  6.9             │ Conv+window   │
  ├─────────────────────────────┼──────────────────┼──────────────────┼───────────────┤
  │ C: Fused JIT (parallel) *** │  0.4             │  0.7             │ WINNER        │
  ├─────────────────────────────┼──────────────────┼──────────────────┼───────────────┤
  │ D: JIT conv only (parallel) │  6.7             │ 16.3             │ Numpy windows │
  └─────────────────────────────┴──────────────────┴──────────────────┴───────────────┘

  Batch computation speedup: 37x (d=4), 47x (d=6)

  Full pipeline impact (find_best_bound_direct, controlled back-to-back):
  ┌──────────────────┬──────────┬────────────┬─────────┐
  │     Config       │ Baseline │ Numba JIT  │ Speedup │
  ├──────────────────┼──────────┼────────────┼─────────┤
  │ n=2,m=100 (d=4)  │ 18.82s   │ 15.18s     │  1.24x  │
  ├──────────────────┼──────────┼────────────┼─────────┤
  │ n=3,m=7  (d=6)   │ 37.68s   │ 26.83s     │  1.40x  │
  └──────────────────┴──────────┴────────────┴─────────┘

  Pipeline-level improvement is capped because test-value computation is now
  only 2.6-10.5% of total time. Remaining bottleneck is composition enumeration
  (35-86%) and symmetry filtering (7-35%).

  Updated profile (after Numba JIT):
  ┌────────────────────────────┬───────┬────────┬───────┬─────────┬───────┬────────┬───────┐
  │           Config           │ Total │  Enum  │ Enum% │  Sym    │ Sym%  │  Test  │ Test% │
  ├────────────────────────────┼───────┼────────┼───────┼─────────┼───────┼────────┼───────┤
  │ n=2,m=100 (86M, d=4)       │ 12.3s │ 4.38s  │ 35.7% │ 4.28s   │ 34.8% │ 1.28s  │ 10.5% │
  ├────────────────────────────┼───────┼────────┼───────┼─────────┼───────┼────────┼───────┤
  │ n=3,m=7  (41.5M, d=6)      │ 30.4s │ 26.08s │ 85.8% │ 2.26s   │  7.4% │ 0.78s  │  2.6% │
  └────────────────────────────┴───────┴────────┴───────┴─────────┴───────┴────────┴───────┘

  Hotspot functions (cProfile)

  ┌────────────────────────────────────────┬────────────────┬────────────────┐
  │                Function                │ Cumtime% (d=4) │ Cumtime% (d=6) │
  ├────────────────────────────────────────┼────────────────┼────────────────┤
  │ compute_test_values_batch              │ 50%            │ 22%            │
  ├────────────────────────────────────────┼────────────────┼────────────────┤
  │ generate_compositions_batched / _outer │ 36%            │ 75%            │
  ├────────────────────────────────────────┼────────────────┼────────────────┤
  │ _gen_compositions_inner3               │ 31%            │ 64%            │
  ├────────────────────────────────────────┼────────────────┼────────────────┤
  │ numpy.column_stack                     │ 14%            │ 26%            │
  ├────────────────────────────────────────┼────────────────┼────────────────┤
  │ asymmetry_prune_mask                   │ 6%             │ <1%            │
  └────────────────────────────────────────┴────────────────┴────────────────┘
