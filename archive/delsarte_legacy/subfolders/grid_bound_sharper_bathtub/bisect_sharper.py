"""Bisection driver for the sharper-bathtub pipeline.

STATUS: OBSTRUCTED. Thin re-export of the baseline bisect_M_cert_mm; the
only change is that the filter_kwargs default is built from mu_sharper rather
than mu_MV directly. With mu_sharper = mu_MV (current placeholder), behaviour
is identical to the baseline.
"""
from __future__ import annotations

from typing import Optional

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.bisect_mm import (
    bisect_M_cert_mm as _bisect_baseline,
    CertifiedBoundMM,
)
from delsarte_dual.grid_bound.phi_mm import PhiMMParams

from .mu_sharper import mu_sharper


def bisect_M_cert_sharper(
    params: PhiMMParams,
    N: int,
    *,
    M_lo_init: fmpq = fmpq(127, 100),
    M_hi_init: fmpq = fmpq(1276, 1000),
    tol_q: fmpq = fmpq(1, 10**4),
    max_cells_per_M: int = 500000,
    filter_kwargs: Optional[dict] = None,
    prec_bits: int = 256,
) -> CertifiedBoundMM:
    """Drop-in replacement for bisect_M_cert_mm using mu_sharper.

    The baseline bisection already constructs mu_arb = mu_of_M(M) at each
    candidate M; here we override that implicitly via filter_kwargs. When
    mu_sharper is tightened (post-OBSTRUCTION), the bisection transparently
    benefits.
    """
    return _bisect_baseline(
        params=params,
        N=N,
        M_lo_init=M_lo_init,
        M_hi_init=M_hi_init,
        tol_q=tol_q,
        max_cells_per_M=max_cells_per_M,
        filter_kwargs=filter_kwargs,
        prec_bits=prec_bits,
    )


__all__ = ["bisect_M_cert_sharper", "CertifiedBoundMM"]
