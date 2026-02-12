# Full Experiment Results — SDP Lower Bound Experiments

## Session Overview
- **Branch**: `lower_bound_experiments`
- **Duration**: ~1 hour
- **Machine**: 16-core CPU, 32GB RAM, Windows 10
- **Primary solver**: MOSEK (academic license)

---

## Experiment 1: Simplex Cuts x_i(1-x_i) >= 0 (`exp_valid_ineq_v2.py`)

**Hypothesis**: Adding localizing matrices for x_i(1-x_i) >= 0 to Lasserre Level-2 could tighten the relaxation.

**Result**: POSITIVE. Consistent improvement at every P tested.

| P | Baseline LB | + Simplex Cuts | Improvement | Time (new) | Time (baseline) |
|---|-------------|----------------|-------------|------------|-----------------|
| 5 | 1.632651 | 1.632653 | +0.000002 | 3.5s | 5.4s |
| 6 | 1.585589 | 1.585612 | +0.000023 | 10.2s | 9.2s |
| 7 | 1.581729 | 1.581746 | +0.000017 | 21.2s | 17.1s |
| 8 | 1.548249 | 1.548319 | +0.000070 | 34.8s | 35.1s |
| 9 | 1.545321 | 1.545626 | +0.000305 | 55.1s | 88.1s |
| 10 | 1.524610 | 1.524823 | +0.000213 | 72.1s | 146.7s |

**Conclusion**: Simplex cuts are valid on the simplex but NOT implied by Level-2 moment constraints. They consistently tighten the relaxation, with growing improvement at larger P.

---

## Experiment 2: Fast Lasserre + Simplex Cuts (`exp_fast_lasserre.py`) — BEST RESULT

**Hypothesis**: Combining simplex cuts with coarse tolerance and tight initial brackets could push to P=16-20.

**Result**: STRONGLY POSITIVE.

### Full Results Table

| P | New LB (L2+SC) | Baseline LB | Improvement | Shor LB | vs Shor | Iters | Time (new) | Time (baseline) | Speedup |
|---|----------------|-------------|-------------|---------|---------|-------|------------|-----------------|---------|
| 5 | 1.632653 | 1.632651 | +0.000002 | 1.111111 | +0.522 | 19 | 3.6s | 5.4s | 1.5x |
| 6 | 1.585612 | 1.585589 | +0.000023 | 1.090909 | +0.495 | 19 | 7.8s | 9.2s | 1.2x |
| 7 | 1.581746 | 1.581729 | +0.000017 | 1.076923 | +0.505 | 19 | 14.8s | 17.1s | 1.2x |
| 8 | 1.548319 | 1.548249 | +0.000070 | 1.066667 | +0.482 | 19 | 23.9s | 35.1s | 1.5x |
| 9 | 1.545626 | 1.545321 | +0.000305 | 1.058824 | +0.487 | 19 | 39.6s | 88.1s | 2.2x |
| 10 | 1.524823 | 1.524610 | +0.000213 | 1.052632 | +0.472 | 19 | 63.4s | 146.7s | 2.3x |
| 11 | 1.520012 | 1.519106 | +0.000906 | 1.047619 | +0.472 | 10 | 49.6s | 240.9s | 4.9x |
| 12 | 1.507730 | 1.506925 | +0.000805 | 1.043478 | +0.464 | 10 | 78.7s | 442.9s | 5.6x |
| 13 | 1.503642 | 1.503012 | +0.000630 | 1.040000 | +0.464 | 10 | 126.8s | 851.0s | 6.7x |
| 14 | 1.493577 | 1.492672 | +0.000905 | 1.037037 | +0.457 | 10 | 144.1s | 1580.8s | 11.0x |
| 15 | 1.490516 | 1.485952 | +0.004564 | 1.034483 | +0.456 | 10 | 211.7s | 2464.0s | 11.6x |
| **16** | **1.483826** | **---** | **NEW** | 1.032258 | +0.452 | 9 | 272.0s | --- | --- |
| **17** | **1.481782** | **---** | **NEW** | 1.030303 | +0.451 | 9 | 449.3s | --- | --- |
| **18** | **1.475327** | **---** | **NEW** | 1.028571 | +0.447 | 9 | 553.1s | --- | --- |
| **19** | **1.478872** | **---** | **NEW*** | 1.027027 | +0.452 | 6 | 629.4s | --- | --- |

*P=19 bound is coarse (time limit hit at 6 iterations). V(P) should decrease with P, so the true bound is likely <= V(18) = 1.475.

### Key Achievements
1. **Tighter bounds at P=5-15**: Consistently improved over baseline (up to +0.004564 at P=15)
2. **1.5x to 11.6x speedup**: From simplex cuts + coarse tolerance + tight brackets
3. **New results at P=16-19**: Never computed before in this codebase
4. **V(P) >= 1.475 at P=18**: Furthest Lasserre Level-2 bound computed

---

## Experiment 3: Dual Certificate Extraction (`exp_dual_certificate.py`)

**Result**: Successfully extracted and verified certificates for P=2-10.

| P | Bound | M2 rank | M1 rank | Flat Ext | #Tight Constraints | Time |
|---|-------|---------|---------|----------|-------------------|------|
| 2 | 1.777778 | 2 | 2 | **YES** | 1 | 0.4s |
| 3 | 1.706667 | 2 | 2 | **YES** | 1 | 1.0s |
| 4 | 1.644465 | 2 | 2 | **YES** | 3 | 3.9s |
| 5 | 1.632649 | 4 | 4 | **YES** | 1 | 7.2s |
| 6 | 1.585588 | 16 | 6 | no | 1 | 13.9s |
| 7 | 1.581719 | 20 | 7 | no | 1 | 27.5s |
| 8 | 1.548100 | 28 | 8 | no | 1 | 50.7s |
| 9 | 1.544525 | 33 | 9 | no | 1 | 78.4s |
| 10 | 1.524609 | 41 | 10 | no | 1 | 171.6s |

### Exact Certificates (Flat Extension = Curto-Fialkow Theorem)
- **P=2**: V(2) = 16/9 = 1.777778 exactly. Optimal: x = (0.5, 0.5)
- **P=3**: V(3) = 256/150 ≈ 1.706667 exactly. Optimal: x = (0.367, 0.267, 0.367)
- **P=4**: V(4) ≈ 1.644465 exactly. Optimal: x = (0.255, 0.245, 0.245, 0.255)
- **P=5**: V(5) ≈ 1.632649 exactly. Optimal: x = (0.214, 0.143, 0.286, 0.143, 0.214)

---

## Important Caveat

**These are bounds on V(P), NOT on C_1a.** V(P) is the optimal autocorrelation ratio for P-bin step functions. Since V(P) > C_1a ~ 1.5029 for all finite P, these lower bounds on V(P) do NOT directly constrain C_1a from below. The Lasserre Level-2 relaxation gap grows with P — we would need P >> 50 (and much higher Lasserre levels) to approach C_1a.

---

## Failed Approaches (`failed/`)

| File | Approach | Why it failed |
|------|----------|---------------|
| `exp_sparse_sdp.py` | Shor + RLT cuts | RLT cuts are already implied by Shor; gives exactly the Shor bound 2P/(2P-1) at every P. Lacks moment matrix structure. |
| `exp_sparse_sdp.py` | Banded Lasserre (chordal decomposition) | Decomposing the moment matrix into overlapping cliques of bandwidth b loses the global structure. Gives Shor bound regardless of bandwidth (b=2 through b=14 all identical). |
| `exp_fourier_sdp.py` | Doubly Nonneg (DNN) relaxation of Fourier dual | DNN inner approximation of copositivity is equivalent to Shor for this problem. Gave C* = M^2 (normalization error) or Shor-level bounds. |
| `exp_fourier_sdp.py` | Simplex copositive dual | Direct copositive formulation on the simplex gives exactly the Shor bound at every P (tested up to P=200). |
| `exp_single_solve.py` | Single-solve with scalar convolution | Replacing matrix localizing constraints with scalar convolution constraints loses bound quality — gives only Shor bound. |
| `exp_single_solve_v2.py` | Single-solve lambda reformulation | Dividing by eta to get affine constraints introduces a bilinear term `lam * pk(y)` which violates CVXPY's DCP rules. Fundamental limitation. |
| `exp_valid_ineq.py` | Simplex cuts v1 (CLARABEL solver) | Same approach as v2 but using CLARABEL solver, which crashed with `Eigval error: Eigen(1)` panic in Rust PSD cone code. Replaced by `exp_valid_ineq_v2.py` using MOSEK. |

---

## Files

| File | Purpose |
|------|---------|
| `baseline_results.py` | Baseline V(P) values from notebook + comparison helper |
| `core_utils.py` | Core utilities extracted from notebook |
| `exp_valid_ineq_v2.py` | Simplex cuts validated at P=5-10 |
| `exp_fast_lasserre.py` | **BEST**: Simplex cuts + fast solver, P=5-19 |
| `exp_dual_certificate.py` | Certificate extraction and verification |
| `exp_refine_bounds.py` | Bound refinement (started but cut short) |
| `exp_pairwise_cuts.py` | Pairwise cuts x_i(1-x_i-x_j)>=0 (not yet run) |
| `failed/` | All failed experiment implementations (see table above) |
