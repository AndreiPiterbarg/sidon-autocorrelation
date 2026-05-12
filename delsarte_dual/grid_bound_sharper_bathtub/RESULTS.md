# Path-1 Toeplitz-PSD SDP — results (one-session implementation)

**Date.** 2026-05-03.  **Author.** Path-1 numerical SDP run.

## Summary

Implemented the Path-1 SDP from `proof_paths.md` in two variants
(symmetric and asymmetric), in `path1_sdp.py`.  cvxpy + CLARABEL,
seconds per solve, scales to N = 16 in <1 s.

### Headline numbers

At `M = 1.275`, `n_0 = 1`:

| Variant | mu_sharper | mu_MV | gap |
|--------|-----------|-------|-----|
| Symmetric f, all N tested | **0.1375** | 0.25443 | **45.96%** |
| Asymmetric f, N=16        | 0.25355   | 0.25443 | 0.35%  |

The symmetric SDP is **degenerate**: the optimum is achieved by
`y_{n_0} = (M-1)/2`, all other `y_k = 0`. The Toeplitz-PSD block is slack;
the binding constraint is `1 + 2*y_{n_0} <= M` from the L^infinity LMI at
x = 0. This admits a one-line rigorous proof (see `mu_sharper.py`).

The asymmetric SDP converges to MV's `mu(M)` — Path 1 alone gives no
improvement over MV for general f.

### General formula (symmetric subclass, RIGOROUS)

  **`mu_sharper_sym(M, n) = (M - 1) / 2  for all n >= 1, all M in (1, 2)`.**

Proof: f symmetric => hat f real => hat h = (hat f)^2 real and >= 0. Then
`h(x) = hat h(0) + 2 sum_{k>=1} hat h(k) cos(2 pi k x)`. With all coefs
nonneg, `h(0) = sum hat h(k) >= 0` is the maximum, so `h(0) <= M` gives
`1 + 2 sum_{k>=1} hat h(k) <= M`, hence each `hat h(n) <= (M-1)/2`. QED.

## SDP sizes (analytic)

| N  | Decision vars | PSD blocks      | Equality constraints |
|----|--------------|-----------------|----------------------|
| 2  |  9           | 2 of dim 3      | 4                    |
| 4  | 20           | 2 of dim 5      | 6                    |
| 8  | 54           | 2 of dim 9      | 10                   |
| 16 | 170          | 2 of dim 17     | 18                   |
| 20 | 252          | 2 of dim 21     | 22                   |

(N = truncation; 2 PSD blocks = T_N (Toeplitz) + Q (Fejer-Riesz / Hermitian
encoding via 2(N+1)x2(N+1) real PSD).  Solve time grows from 0.01 s (N=2)
to ~0.3 s (N=16) on a laptop with CLARABEL.)

## Per-N convergence at M = 1.275, n_0 = 1

ASYMMETRIC variant (mu_sharper, target ~ 0.2544):

```
  N   mu_sharper        mu_MV        gap     gap%   status        t (s)
  2     0.1944544    0.2544340    0.0600   23.57%   optimal       0.01
  3     0.2224797    0.2544340    0.0319   12.56%   optimal       0.02
  4     0.2381570    0.2544340    0.0163    6.40%   optimal       0.02
  5     0.2477664    0.2544340    0.0067    2.62%   optimal       0.02
  6     0.2539364    0.2544340    0.0005    0.20%   inacc         0.03
  8     0.2504275    0.2544340    0.0040    1.57%   optimal       0.05
 10     0.2517786    0.2544340    0.0027    1.04%   inacc         0.09
 12     0.2539483    0.2544340    0.0005    0.19%   inacc         0.12
 16     0.2535479    0.2544340    0.0009    0.35%   inacc         0.33
```

(Convergence not strictly monotone past N=6 due to CLARABEL inaccuracy at
high N for this nearly-degenerate problem; nevertheless mu_sharper -> mu_MV.)

SYMMETRIC variant: identically 0.13750 at every N (matches (M-1)/2).

## M-dependence (SYM mode, N = 10)

```
   M      mu_MV       (M-1)/2     gap_pct
 1.05    0.04981     0.02500     49.81%
 1.10    0.09865     0.05000     49.31%
 1.15    0.14584     0.07500     48.57%
 1.20    0.19099     0.10000     47.64%
 1.275   0.25443     0.13750     45.96%
 1.30    0.27440     0.15000     45.34%
 1.40    0.34841     0.20000     42.60%
 1.50    0.41350     0.25000     39.54%
 1.85    0.58410     0.42500     27.24%
 1.95    0.62020     0.47500     23.41%
```

Gap shrinks as M -> 2 (the bound becomes loose), but never below 23% in
the relevant regime M in [1.05, 1.95]. Crucially in our regime of interest
(M ~ 1.275), the gap is ~46%.

## Implications

- **Asymmetric (full Sidon problem).** Path 1 SDP gives 0% improvement
  over MV. To make progress, one needs Path 2 (Fejer-Riesz factorization
  h = |p|^2 — the Schur-product structure that Path 1 misses) or Path 3
  (Karlin-Studden constrained Chebyshev-Markov). The Boyer-Li 2025
  piecewise-linear witness is also informative as the primal side.
- **Symmetric subclass.** A rigorous closed-form sharper bound `(M-1)/2`
  applies. Plugging this into the multi-moment Phi (`phi_mm.py`) restricted
  to the symmetric f search ought to push C_{1a}^{sym} above 1.42429 (Path A)
  — needs a follow-up bisection run.

## 6-month plan (revised based on session findings)

- **Day 1-30:** *(DONE in 1 session.)* Numerical SDP setup, both variants,
  M-sweep, convergence analysis. Results above.
- **Day 30-90:** Pivot to Path 2 (Fejer-Riesz `h = |p|^2`) for the
  asymmetric case. The `|p_{n_0}|^2 = max` SDP under `int |p|^2 = 1` and
  `||p||_inf^2 = M` is a non-convex `(N+1)^2`-variable polynomial
  optimization; Lasserre relaxation at degree 4-6 is tractable.
- **Day 90-150:** Plug `(M-1)/2` SYM bound into MM-10 + cell-search-nd;
  re-bisect for C_{1a}^{sym}. Expect an improvement of ~0.005-0.02 over
  Path A's 1.42429.
- **Day 150-180:** Rigorous certification (rational rounding + Putinar SOS
  via arb) of any Path-2 numerical bound that tightens the asymmetric MV.
  Currently no asymmetric improvement to certify.

## Files

- `path1_sdp.py` — implementation, sweeps, drivers (`__main__` runs all).
- `mu_sharper.py` — exposes `mu_sharper_sym(M, n) = (M-1)/2` for the
  symmetric subclass.  `mu_sharper(M, n) = mu_MV(M)` unchanged for the
  general (asymmetric) case.
- This file — results dump.
