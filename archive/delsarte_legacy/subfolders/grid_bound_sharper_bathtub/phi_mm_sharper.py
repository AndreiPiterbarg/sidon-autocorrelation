"""Phi_MM for the sharper-bathtub pipeline.

STATUS: OBSTRUCTED.  No tightening of Phi_MM results from the placeholder
mu_sharper = mu_MV.  This module is a thin re-export of the baseline Phi_MM
so the sharper-pipeline surface is self-contained.
"""
from __future__ import annotations

from delsarte_dual.grid_bound.phi_mm import (
    PhiMMParams,
    phi_mm,
    kn_period_one,
    mu_of_M,
    _safe_sqrt,
    arb_sqr,
)


__all__ = [
    "PhiMMParams",
    "phi_mm",
    "kn_period_one",
    "mu_of_M",
    "_safe_sqrt",
    "arb_sqr",
]
