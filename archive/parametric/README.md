# archive/parametric/

Dead-end exploration of a parametric dual-bound SDP for $C_{1a}$.

## Pipeline

The idea was a two-level SDP:
- **Inner** (`primal_qp.py`): bivariate moment SDP for $\lambda_N(\mu) = \inf_{f \in \mathcal A} \int p(t)\,(f*f)(t)\,dt$.
- **Outer** (`outer_sdp.py`): SOS-relaxation joint SDP combining the dualised inner problem so that $C_{1a} \ge \sup_\mu \lambda_N(\mu)$.

## Files

- `__init__.py`         — package shim re-exporting submodules.
- `chebyshev_duality.py` — algebraic backbone: $K^{(\ell)}$ table, Lukacs / Hausdorff moment bases.
- `primal_qp.py`        — inner moment SDP for $\lambda_N(\mu)$ at fixed $m$.
- `outer_sdp.py`        — joint SDP combining primal + dualised inner via SOS.
- `certify.py`          — rational-arithmetic verification of SDP solutions.
- `sweep.py`            — meta-script scanning $(L, N)$ and extrapolating in $L$.

## Status

Abandoned: the parametric construction did not improve on MV's 1.2748;
extrapolation in $L$ saturated below the target. Kept for reference.
