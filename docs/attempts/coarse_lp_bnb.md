# Coarse LP / Pólya Branch-and-Bound (DEAD)

Pólya/Handelman LP relaxation sweep of `val(d)` plus a direct Parseval+tail enclosure of `K_2` for the multi-scale arcsine kernel. The LP sweep did not lift the rigorous LB above MV's 1.27481 on its own; the K_2 finding was reused by the multi-scale arcsine certification.

## Pólya LP sweep

For nonneg quadratic on the simplex Delta_d, the Pólya/Handelman LP relaxation at degree R has gap O(C_d / R) with `C_d ~ ||M_W||_infty ~ d`. Fitted models per d (M1: C/R, M2: C/R^2, M4: C/R^a) over 68 successful records, d in {4, 6, 8, 10, 12, 14, 16, 20}. Best cross-d fit: C(d) = 0.113 * d^{0.764}.

Projected feasibility for `alpha >= 1.281`:

| d | val(d) | gap_to_target | R_needed | LP size at R_needed | feasible on laptop? |
|---|--------|---------------|----------|----------------------|----------------------|
| 14 | 1.284 | +0.003 | ~282 | ~6e13 vars, ~5.5e14 nnz | NO |
| 16 | 1.319 | +0.038 | ~25 | ~24M vars, ~244M nnz | NO |
| 20 | 1.328 | +0.047 | ~24 | ~224M vars, ~2.7B nnz | NO |
| 32 | 1.336 | +0.055 | ~29 | ~10^12 vars | NO |
| 64 | 1.384 | +0.103 | ~26 | ~10^16 vars | NO |

**Verdict:** no (d, R) combination on the 8 GB laptop is projected to clear 1.281. Cloud H100 required at minimum; structural rigor-parity barrier (MEMORY: project_rigor_parity_barrier) prevents unlocking without a dual certificate or exact rational LP.

## K_2 rigorous enclosure — reused downstream

Direct Parseval + tail bound for the multi-scale arcsine kernel `K_hat(xi) = lambda_1 J_0(pi delta_1 xi)^2 + lambda_2 J_0(pi delta_2 xi)^2`.

- **Bulk**: rigorous arb-ball via `flint.acb.integral` (adaptive Gauss-Legendre) on `[0, Xi]`.
- **Tail**: `J_0(z)^2 <= 2/(pi z)` (Krasikov 2003) + Jensen on `t -> t^2`. Closed-form tail bound `tail <= (4 / (pi^2 Xi)) * sum_i lambda_i / delta_i^2`.

At K26 best point (delta_1 = 0.138, delta_2 = 0.055, lambda_1 = 0.9312), Xi = 10^5, prec = 128 bits:
- K_2 in [4.3586, 4.3592] (rigorous, width 5.2e-4)
- k_1 = 0.9144970268, S_1 = 54.6739711 (rigorous, exact MV rational coeffs + arb J_0)
- Rigorous M_cert > 1.27987 (vs MV pure-arcsine 1.27481, +0.005)

This enclosure is the direct precursor to the multi-scale rigorous certification ([multiscale_arcsine.md](multiscale_arcsine.md), C_{1a} >= 1651/1280 = 1.28984 at the 2-scale point and 1.293 at the 3-scale point). Sonine derivation in [cascade_estimator.md](cascade_estimator.md) tightens the cross-Bessel term via the same `J_0(z)^2 <= 2/(pi z)` lemma.

## References

- [../proof_outline.md](../proof_outline.md), [../formalization.md](../formalization.md)
- [lasserre.md](lasserre.md) — Pólya/Handelman is a degree-relaxed special case of Lasserre.
