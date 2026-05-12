"""Empirical convergence-rate fitter and projection.

Given a sequence (R_i, alpha_i) for fixed d, fit gap(R) = (val_d - alpha)
to a power law gap = C * R^{-a}, and project to find the R needed to
achieve a target alpha.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np


@dataclass
class FitResult:
    a: float                  # exponent in gap ~ C * R^{-a}
    C: float                  # constant
    log_residual: float       # least-squares residual in log-log fit
    n_points_used: int


def fit_convergence(
    R_list: Sequence[int],
    alpha_list: Sequence[float],
    val_d: float,
    skip_first_n: int = 1,
) -> Optional[FitResult]:
    """Fit gap = val_d - alpha to a power law in R (using last few points).

    skip_first_n: drop the first n_skip points (they often don't fit the
    asymptotic rate). Fitter requires >= 3 points after dropping.
    """
    R = np.asarray(R_list, dtype=float)
    a_arr = np.asarray(alpha_list, dtype=float)
    gap = val_d - a_arr

    # Filter: positive gap and finite alpha
    mask = (gap > 1e-12) & np.isfinite(a_arr)
    R = R[mask]
    gap = gap[mask]
    if len(R) <= skip_first_n + 1:
        return None
    R = R[skip_first_n:]
    gap = gap[skip_first_n:]

    # Linear fit in log-log
    log_R = np.log(R)
    log_gap = np.log(gap)
    A = np.vstack([log_R, np.ones_like(log_R)]).T
    coeffs, residuals, _, _ = np.linalg.lstsq(A, log_gap, rcond=None)
    slope, intercept = coeffs
    a = -slope
    C = float(np.exp(intercept))
    res = float(residuals[0]) if len(residuals) else 0.0
    return FitResult(a=a, C=C, log_residual=res, n_points_used=len(R))


def project_R_for_target(
    fit: FitResult,
    val_d: float,
    target_alpha: float,
) -> float:
    """Given a fit, project the R needed so that val_d - C*R^{-a} >= target_alpha.

    Solves C * R^{-a} <= val_d - target_alpha = epsilon
        =>  R >= (C / epsilon)^(1/a).
    Returns +inf if epsilon <= 0.
    """
    eps = val_d - target_alpha
    if eps <= 0:
        return float("inf")
    return (fit.C / eps) ** (1.0 / fit.a)
