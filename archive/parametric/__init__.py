"""Parametric dual-bound optimization for C_{1a}.

Pipeline:
    lambda_N(mu) = inf_{f in A} int p(t) (f*f)(t) dt   (bivariate moment SDP)
    C_{1a} >= sup_mu lambda_N(mu)                      (outer SDP via SOS)

Modules
-------
chebyshev_duality:  algebraic backbone (K^(l) table, Lukacs, Hausdorff).
primal_qp:          inner moment SDP for lambda_N(mu) given m.
outer_sdp:          joint SDP combining primal + dualized inner via SOS.
certify:            rational-arithmetic verification of SDP solutions.
sweep:              meta-script to scan (L, N) and extrapolate.
"""

from . import chebyshev_duality  # noqa: F401
from . import primal_qp  # noqa: F401
from . import outer_sdp  # noqa: F401
from . import certify  # noqa: F401
from . import sweep  # noqa: F401

__all__ = [
    "chebyshev_duality",
    "primal_qp",
    "outer_sdp",
    "certify",
    "sweep",
]
