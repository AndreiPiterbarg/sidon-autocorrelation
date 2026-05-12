# Interval Branch-and-Bound (DEAD)

Rigor-parity barrier closed this route: Phase B T1/T2/T3 cuts went net-negative without a dual certificate or exact rational LP (MEMORY: project_rigor_parity_barrier).

## Formal claim

For every integer d >= 2,

$$C_{1a} \;\ge\; \operatorname{val}(d) \;:=\; \min_{\mu \in \Delta_d} \max_{W \in \mathcal{W}_d} \mu^\top M_W \mu,$$

where the window matrices come from `lasserre.core.build_window_matrices` with the correct prefactor `2d / ell` (derived as `1/|I_W|` with `|I_W| = ell / (2d)`; the naive guess `d / (ell - 1)` double-counts overlaps of consecutive pair-sum supports).

The derivation uses only:

1. **Averaging:** `max_t (f*f)(t) >= (1/|I_W|) integral_{I_W} (f*f)`.
2. **Bin-mass integral bound (Tonelli):** the integral lower-bounds `sum_{i+j in K_W} mu_i mu_j` where `mu_i = integral_{B_i} f`.

Symmetry reduction: `sigma(i) = d-1-i` is an involution of `W_d` and the objective, so restricting to `H_d = { mu : mu_0 <= mu_{d-1} }` preserves val(d). Implementation uses the looser cut `mu_0 <= 1/2` (a superset of `H_d`); soundness preserved.

## Why the route closed

- The interval BnB driver produces sound LP-relaxation lower bounds on `val(d)` per cell, but at d >= 12 the rigor-parity barrier kicks in: cuts T1, T2, T3 net-negative against c_target = 1.281, with no available exact rational LP nor a dual Farkas certificate to recover the gap.
- Lasserre's plateau ([lasserre.md](lasserre.md)) caps numerical val(d) at 1.205 / 1.241 / 1.271 / 1.284 for d = 8 / 10 / 12 / 14 — even unblocking the interval BnB would only marginally exceed MV's 1.27481.

## What would unlock

A dual-cert (Farkas-style) construction for the LP relaxation, or an exact rational LP at d >= 14. Neither is on the user's permitted research path post-2026-05-10. Superseded by the multi-scale arcsine route ([multiscale_arcsine.md](multiscale_arcsine.md)).

## References

- [../proof_outline.md](../proof_outline.md), [../formalization.md](../formalization.md), [lasserre.md](lasserre.md)
