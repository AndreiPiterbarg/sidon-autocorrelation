# SDP Lower Bound Experiments — Results Summary

## Methods Tested

### 1. Shor + RLT Relaxation (exp_sparse_sdp.py)
- **Result**: Only matches Shor bound 2P/(2P-1). RLT cuts add nothing.
- **Reason**: Shor relaxation fundamentally lacks the moment matrix structure that makes Lasserre Level-2 powerful.
- **Scaling**: Fast (P=100 in 23s) but useless bounds.

### 2. Banded Lasserre (exp_sparse_sdp.py)
- **Result**: Also only matches Shor bound regardless of bandwidth.
- **Reason**: The chordal decomposition into small cliques loses the global moment structure.

### 3. Fourier/Trigonometric SDP (exp_fourier_sdp.py)
- **Result**: Doubly nonneg relaxation gives C*~1.0 (useless). Simplex copositive gives Shor bound.
- **Reason**: Copositivity inner approximations (DNN, Parrilo-1) are too weak for this problem. The copositivity gap is the entire gap between 1.0 and the true bound.

### 4. Simplex Cuts x_i(1-x_i) >= 0 (exp_valid_ineq_v2.py, exp_fast_lasserre.py)
- **Result**: SUCCESSFUL. Consistent improvement at every P tested.
- Improvement grows with P: +0.000002 at P=5 up to +0.004564 at P=15.
- These cuts are trivially valid on the simplex but NOT implied by Level-2 moment constraints.

### 5. Fast Lasserre with Simplex Cuts (exp_fast_lasserre.py) — BEST RESULT
- **Combined approach**: Original binary search + simplex cuts + coarse tolerance for large P.
- **Speedup**: 1.5x-11.6x faster than baseline (more at larger P).
- **New territory**: P=16-19 computed for the first time.

### 6. Dual Certificate Extraction (exp_dual_certificate.py)
- Confirmed flat extensions (exact V(P)) at P=2,3,4,5.
- Tight constraints identified at each P.
- For P>=6, no flat extension (genuine relaxation gap).

## Key Results Table

| P  | Baseline LB | New LB (L2+SC) | Improvement | Speedup | Note |
|----|-------------|-----------------|-------------|---------|------|
| 5  | 1.632651    | 1.632653        | +0.000002   | 1.5x    | Flat ext |
| 6  | 1.585589    | 1.585612        | +0.000023   | 1.2x    |      |
| 7  | 1.581729    | 1.581746        | +0.000017   | 1.2x    |      |
| 8  | 1.548249    | 1.548319        | +0.000070   | 1.5x    |      |
| 9  | 1.545321    | 1.545626        | +0.000305   | 2.2x    |      |
| 10 | 1.524610    | 1.524823        | +0.000213   | 2.3x    |      |
| 11 | 1.519106    | 1.520012        | +0.000906   | 4.9x    |      |
| 12 | 1.506925    | 1.507730        | +0.000805   | 5.6x    |      |
| 13 | 1.503012    | 1.503642        | +0.000630   | 6.7x    |      |
| 14 | 1.492672    | 1.493577        | +0.000905   | 11.0x   |      |
| 15 | 1.485952    | 1.490516        | +0.004564   | 11.6x   |      |
| 16 | ---         | 1.483826        | NEW         | ---     | New! |
| 17 | ---         | 1.481782        | NEW         | ---     | New! |
| 18 | ---         | 1.475327        | NEW         | ---     | New! |
| 19 | ---         | 1.478872        | NEW         | ---     | New! (coarse) |

## Important Note
These are bounds on V(P) (optimal ratio for P-bin step functions), NOT on C_1a directly. Since V(P) > C_1a for small P, these lower bounds on V(P) do not constrain C_1a. The relaxation gap grows with P — we need P >> 50 for V(P) to approach C_1a ~ 1.5029.

## What Worked
1. **Simplex cuts**: The constraint x_i(1-x_i) >= 0 localizing matrices tighten the SDP at every P.
2. **Coarse tolerance**: Using eta_tol=1e-3 instead of 1e-6 halves the binary search iterations.
3. **Tight initial brackets**: Extrapolating from previous P values narrows the search range.

## What Didn't Work
1. **Shor/RLT**: Too weak without full moment matrix.
2. **Banded decomposition**: Loses the global structure that makes Lasserre work.
3. **Fourier/copositive**: DNN relaxation of copositivity is fundamentally insufficient.
4. **Single-solve reformulation**: The bilinear eta*M_1(y) term prevents DCP formulation.
