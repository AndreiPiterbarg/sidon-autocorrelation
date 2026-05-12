# SDP Attempts

Archived SDP-based attempts at lower-bounding the Sidon autocorrelation
constant `C_{1a}`. None of these produced the working bound (the
rigorous bound currently in the Lean proof comes from the Cohn-Elkies /
multi-scale arcsine route, not from these SDP families).

Already organised by sub-direction; each subdir has its own README with
the math, status, and honest difficulty assessment.

## Subdirectories

| Subdir | Approach | README |
|---|---|---|
| `bochner_sos/` | Continuum Bochner-SOS dual: certify `M(g) <= C_{1a}` for nonneg `g` on `[-1/2, 1/2]` via copositive / Parrilo SDP. Outer SDP that optimizes `g` was not built. | [bochner_sos/README.md](bochner_sos/README.md) |
| `simplex_window_dual/` | Polynomial nonneg window-multiplier certificates for the discrete window problem `val(d) = min_mu max_W mu^T M_W mu`. LP at fixed alpha; reflection symmetry + sweep runner implemented. | [simplex_window_dual/README.md](simplex_window_dual/README.md) |
| `chebyshev_dual/` | Chebyshev-basis Lasserre + Fejer-Riesz periodic dual: well-conditioned `c_k = int T_k(4x) f(x) dx` primal, PSD Toeplitz dual, single combined SDP, exact-rational verification via flint/Arb. See file docstrings (no README). |

## Status

All three sub-directions are dormant. Current rigorous bound
`C_{1a} >= 1651/1280 = 1.28984` comes from
`delsarte_dual/grid_bound_alt_kernel/` (multi-scale arcsine kernel),
not from this folder.
