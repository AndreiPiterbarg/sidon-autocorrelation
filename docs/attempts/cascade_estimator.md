# Coarse Cascade Estimator (DEAD)

Branch-and-prune cascade prover (`coarse_cascade_prover.py`) plus two analytic side-quests. Validated correct but did not deliver a publishable bound: the cascade-style route never beat MV's 1.27481, and the engineering ceiling sat well below the multi-scale arcsine result.

## Scope

| Strand | Role |
|--------|------|
| Cascade prover | Cheap pre-cascade estimator for Cloninger-Steinerberger-style discrete enumeration; feeds B&B |
| Route C white transfer | Test White (2022) L^{4/3} chain as a route to bound C_{1a} |
| Sonine derivation | Rigorous bound on multi-scale cross-Bessel integral I(alpha, beta) |

## Cascade prover findings

Two audit passes (initial fixes + 12-agent audit). Original code sampled 2000 random cells while printing "PROOF" — soundness-fatal. After fixes (exhaustive enumeration, vertex-enum box-cert, sub-tree LB via `min_contrib`, S-shift to dodge integer-lattice margin=0, Phase 1 Lipschitz LB capturing 99.9% of cells), the prover became sound and tight.

Best rigorous run (exhaustively certified): C_{1a} >= 1.18 at d=8, S=76 in ~19 min (1.46B cells, every cell verified). Forbidden push: at c >= 1.19 the cell count and the val(d) structural ceiling (val(8) = 1.205, val(10) = 1.241, val(12) = 1.271) put 1.28 out of reach without d >= 14 — infeasible cell counts (~10^14 at d=12, S=101).

Audit's "4 big findings":

| # | Finding | Status |
|---|---------|--------|
| 1 | Refinement monotonicity proved (one-line algebra; was open) | THEOREM |
| 2 | Symmetric extremizer reduction C_{1a} = C_{1a}^sym | REFUTED — Phi is positively 2-homogeneous, Jensen-on-Phi convexity premise false |
| 3 | val(8) = 1.205 is structural ceiling | CONSTRAINT — Lasserre@d=8 cannot bypass |
| 4 | GPU box-cert at d=12 feasible in seconds | FEASIBILITY (not implemented; see [gpu_scs.md](gpu_scs.md)) |

**Verdict:** algorithmically blocked, not mathematically. Beyond 1.18-1.20 would need GPU acceleration of d=14+ box-cert. Superseded by the multi-scale arcsine route (rigorous 1.292 — see [multiscale_arcsine.md](multiscale_arcsine.md)).

## Route C — White (2022) L^{4/3} transfer

Tested whether White's `mu_2^2 in [0.5746, 0.5747]` bound (sign-allowed L^2) transfers to C_{1a} (nonneg L^infty). Result: NEGATIVE with posterior <5%.

- White's near-optimizer `f_c(x) = alpha_c (1/4 - x^2)^{-c}, c = 0.4942` has `||f_c * f_c||_infty = 3.32` on `[-1/2, 1/2]` (6.64 rescaled — 4x the known UB 1.5029); it lives in the *opposite* regime from MV's centred extremizers.
- The Hölder chain `||F*F||_infty >= ||F*F||_{4/3}` trivialises to C_{1a} >= 1 on rescaled supports.
- No Parseval analogue for `||·||_{4/3}`; reverse-Hölder direction is what Hyp_R needs.

Three structural obstructions: sign-vs-nonneg, L^infty-vs-L^2, Hölder direction. Cross-link: [path_a_holder.md](path_a_holder.md) (Hyp_R conditional).

## Sonine derivation — KEPT as math note

Rigorous flint.arb bound on the multi-scale arcsine cross term

$$I(\alpha, \beta) = \int_0^\infty J_0(\pi\alpha\xi)^2 J_0(\pi\beta\xi)^2 \, d\xi.$$

Method: bulk via `acb.integral` adaptive Gauss-Legendre on [0, T=10^5], tail via Sonin-Polya envelope `J_0(z)^2 <= 2/(pi z)` (Krasikov 2003, Landau 1934) giving `tail <= 4/(pi^4 alpha beta T)`. At (alpha, beta) = (0.138, 0.055): width ~6e-5; rigorous K_2 enclosure [4.3586, 4.3592].

This is the analytic backbone of the multi-scale rigorous lift; see [multiscale_arcsine.md](multiscale_arcsine.md) for how it feeds the certified C_{1a} >= 1651/1280.

## References

- [../proof_outline.md](../proof_outline.md), [../formalization.md](../formalization.md)
- Krasikov, "Uniform bounds for Bessel functions", J. Approx. Theory 121 (2003).
- White (2022), arXiv:2210.16437.
