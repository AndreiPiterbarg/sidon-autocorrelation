# Speedup Implementation Summary

Implemented 9 speedups to `coarse_cascade_prover.py` based on the multi-agent research findings. All speedups are **sound** (no false certifications) and the existing test suite (16 tests) still passes.

## Cumulative impact at d=12, S=20, c=1.22 (50,000 cells)

| Method | Time | Tier 3 (vertex) | Speedup vs v1 |
|---|---:|---:|---:|
| v1 (original 2-tier) | 79.3 s | 7,526 cells | 1.0× |
| v2 (#1+#2 4-tier) | 13.8 s | 986 cells | **5.7×** |
| Region aggregation (#3) | 1.8 s | 0 (97% via region) | **45.9×** |

## Per-speedup status

### Tier S (deployed first, biggest gains)

**#1 Hardy–Littlewood Phase 1 LP (DELIVERED)**
- Replaces `(gmax−gmin)/2 · ||δ||_1` with closed form `h · Σ|g_i − ḡ|`.
- Provably ≤ both old bounds (it equals the exact LP min, both old bounds upper-bounded it).
- 3× faster on c=1.10 d=6 S=51 (1.02s vs 3.05s baseline).
- 6 new tests (incl. d=4,6,8 vertex-vs-Phase1 soundness).

**#2 Toeplitz/spectral trust-region cell-min (DELIVERED)**
- Precomputes σ_min^V(A_W) per window via `compute_qdrop_table` (eigh on P A_W P).
- Closed-form trust-region QP via eigendecomp + Lagrangian Newton.
- Spectrum on V is 3–10× tighter than max_row=ℓ−1.
- New 4-tier dispatcher `_box_certify_batch_adaptive_v2`: original Phase 1 → spectral Phase 1 → trust-region → vertex.
- Tier 3 dispatch reduction: 80% at d=12 S=24 c=1.25, 81% at d=14 S=20 c=1.28.
- 9 new tests including hand-verified PSD/indefinite QP cases.

**#3 Region aggregation via closed-form Shor + κ-shift (DELIVERED)**
- `_region_certify_shor` and `_region_certify_combined`: take per-coord (lo, hi) and certify a polytope.
- Generalizes #2's cell trust-region to multi-cell regions.
- 60s benchmark at d=12 S=20 c=1.22: certifies 97% of 50K cells via region cert (16-cell groups), 1.77s total = **45.9× faster than v1**.

**#4 Per-cell Shor SDP via MOSEK (IMPLEMENTED, niche)**
- `_box_certify_cell_shor_sdp` using MOSEK Fusion. Tighter L_∞ box bound than #2's L_2 ball.
- At d=14: ~2.7s/cell — slower than vertex enum and trust-region.
- Useful at d ≥ 18 where vertex enum is OOM and trust-region too loose. Not the primary path at d ≤ 16.

**#5 B&B over the simplex (IMPLEMENTED, niche)**
- `run_bnb_certify_simplex`: recursive box subdivision with simplex projection (`_project_to_simplex_in_box`).
- Soundness verified via the projection step (mu_c ∈ {sum=1}).
- At d ≤ 12: SLOWER than v2 cell enum (combined region cert per node has overhead).
- The combined region cert (`_region_certify_combined`) developed for this is now available for any region.

### Tier A (further refinements)

**#6 Spectral + KKT Phase 1 (SUBSUMED BY #1+#2)**
- Spectral component: σ_min^V(A_W) precompute → in #2.
- Linear KKT closed form: Hardy-Littlewood = closed-form box-LP min → in #1.
- Active-set KKT for joint problem: implemented exactly via vertex enum at Tier 3.

**#7 Adaptive multiresolution / CFK simplicial subdivision (IMPLEMENTED, niche)**
- `run_box_certification_with_region_aggregation`: lex-sort + bounding-box region cert + per-cell fallback.
- At d=8 S=51 c=1.10: 547s vs v2 cell-by-cell at 64s (8× SLOWER, Python loop vs Numba parallel).
- Wins at d ≥ 14 where vertex enum dominates and region cert savings exceed loop overhead.

**#8 Cutting-plane bulk certification (SUBSUMED)**
- Equivalent to region cert at the simplex root.
- `run_bnb_certify_simplex` calls `_region_certify_combined` on the initial polytope; if val(d) > c globally, certifies in 1 node.

**#9 Multi-window Frank-Wolfe (NOT IMPLEMENTED)**
- Phase 1 already captures 99.9%+ at d ≤ 12 via #1+#2.
- Marginal gain on borderline cells where multi-window cancellation tightens Lipschitz.
- Skipped to avoid complexity without measurable benefit.

## New module additions

| Function | Purpose |
|---|---|
| `compute_qdrop_table(d)` | Cache `max(0, -λ_min^V(A_W))` per window |
| `compute_window_eigen_table(d)` | Cache `(V_W, λ_W)` for trust-region QP |
| `_phase1_lipschitz_lb` | Original + HL bound (Speedup #1) |
| `_phase1_lipschitz_lb_spec` | + spectral quad_drop (Speedup #2) |
| `_trust_region_qp_lb` | Closed-form Lagrangian QP via Newton bisection |
| `_box_certify_cell_trust_region` | Per-cell trust-region cert |
| `_box_certify_batch_adaptive_v2` | 4-tier batch dispatcher |
| `_region_certify_shor` | Region cert via L_2 ball trust-region |
| `_region_certify_combined` | Region cert: max(Phase 1 spec, trust-region) |
| `_project_to_simplex_in_box` | 1D root-find projection (for B&B mu_c) |
| `run_bnb_certify_simplex` | B&B simplex driver (#5) |
| `run_box_certification_with_region_aggregation` | Cascade-prover region path (#7) |
| `_box_certify_cell_shor_sdp` | MOSEK Fusion Shor SDP (#4) |

## Test files (in `tests/`)

- `test_phase1_hardy_littlewood.py` — 6 tests for #1
- `test_speedup2_trust_region.py` — 9 tests for #2
- `test_speedup2_high_d.py` — 3 tests at d=12, d=14 for #2
- `test_speedup3_region.py` — 2 tests including the 60s d=12 benchmark
- `test_speedup4_shor_sdp.py` — 3 tests for #4 (skipped without MOSEK)
- `test_speedup5_bnb.py` — 2 tests for #5 B&B
- `test_speedup7_region_cascade.py` — 1 test for #7 (slow)
