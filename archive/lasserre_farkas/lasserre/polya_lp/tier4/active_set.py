"""Active-set extraction from a coarse LP solution.

Given a coarse-tolerance optimum (x*, y*) at KKT ~ 1e-5, classify each
variable as ACTIVE or INACTIVE so we can build a much smaller LP for
MOSEK polish at 1e-9.

Variable types in the Polya/Handelman LP (see lasserre/polya_lp/build.py):
  alpha     : 1 free variable, KEEP always (it's the objective).
  lambda_W  : n_W >= 0 vars (window weights). Only a small subset >0
              at the optimum -> these are the ACTIVE windows.
  q_K       : n_q free vars (q-multiplier). Always KEEP at this stage;
              they have no sign so a "zero q" is not informative.
  c_beta    : n_le_R >= 0 vars (slacks).  Many are 0 at the optimum.

Active-set theory (Applegate et al. 2023, arXiv:2307.03664):
  At KKT-tol epsilon, lambda_W with reduced cost > 2*epsilon are
  predicted ZERO; reduced cost ~= 0 means tight (could be either).
  Symmetrically, c_beta with reduced cost > 2*epsilon are zero.

Practical rule used here:
  ACTIVE_lambda = { W : x_W > tol_active }
  ACTIVE_cbeta  = { beta : x_{c_beta} > tol_active }
  tol_active is calibrated against the coarse KKT level, default
  10 * coarse_kkt with floor 1e-7 and ceiling 1e-3.

Outputs an ActiveSet record consumed by tier4.polish.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import numpy as np

from lasserre.polya_lp.build import BuildResult


@dataclass
class ActiveSet:
    """Predicted-active variables.

    n_lambda_total       : how many lambda_W variables existed.
    active_lambda_idx    : indices into the lambda block of variables predicted >0.
    n_cbeta_total        : total c-slack variables (0 if eliminate_c_slacks=True).
    active_cbeta_idx     : c-slacks predicted strictly positive (kept active).
    tol_active           : the threshold used for activity detection.
    coarse_alpha         : the coarse alpha used.
    coarse_kkt           : KKT level at which the active set was extracted.
    """
    n_lambda_total: int
    active_lambda_idx: List[int]
    n_cbeta_total: int
    active_cbeta_idx: List[int]
    tol_active: float
    coarse_alpha: Optional[float]
    coarse_kkt: float
    notes: str = ""


def extract_active_set(
    build: BuildResult,
    coarse,            # CoarseResult from tier4.coarse_solve
    tol_active: Optional[float] = None,
    min_active_lambda: int = 1,
) -> ActiveSet:
    """Classify variables based on a coarse solution.

    tol_active: if None, calibrated as max(1e-7, min(1e-3, 10*coarse.kkt)).

    min_active_lambda: keep at least this many lambda_W active. If the
    coarse KKT is poor and zero lambdas pass the threshold, falls back
    to keeping the top-magnitude windows.
    """
    x = coarse.x
    if x is None:
        raise ValueError("coarse.x is None; coarse solve did not return a solution")

    if tol_active is None:
        kkt = max(1e-12, coarse.kkt)
        tol_active = float(min(max(10.0 * kkt, 1e-7), 1e-3))

    lambda_idx = build.lambda_idx
    c_idx = build.c_idx
    n_lambda_total = lambda_idx.stop - lambda_idx.start
    n_c_total = c_idx.stop - c_idx.start

    # --- lambda_W -------------------------------------------------------
    if n_lambda_total > 0:
        lam = x[lambda_idx.start: lambda_idx.stop]
        active_lambda = [int(i) for i, v in enumerate(lam) if v > tol_active]
        if len(active_lambda) < min_active_lambda:
            # fallback: top-magnitude windows
            order = np.argsort(-lam)
            active_lambda = [int(i) for i in order[:min_active_lambda]]
    else:
        active_lambda = []

    # --- c_beta ---------------------------------------------------------
    if n_c_total > 0:
        cb = x[c_idx.start: c_idx.stop]
        active_cbeta = [int(i) for i, v in enumerate(cb) if v > tol_active]
    else:
        active_cbeta = []

    return ActiveSet(
        n_lambda_total=n_lambda_total,
        active_lambda_idx=active_lambda,
        n_cbeta_total=n_c_total,
        active_cbeta_idx=active_cbeta,
        tol_active=tol_active,
        coarse_alpha=coarse.alpha,
        coarse_kkt=coarse.kkt,
        notes=(f"|lam_act|={len(active_lambda)}/{n_lambda_total} "
               f"|c_act|={len(active_cbeta)}/{n_c_total}"),
    )


def summarize_active_set(act: ActiveSet) -> str:
    pct_lam = (100.0 * len(act.active_lambda_idx) / max(1, act.n_lambda_total))
    if act.n_cbeta_total > 0:
        pct_c = 100.0 * len(act.active_cbeta_idx) / act.n_cbeta_total
        c_str = f", c-slack {len(act.active_cbeta_idx)}/{act.n_cbeta_total} ({pct_c:.1f}%)"
    else:
        c_str = ""
    return (f"active: lambda {len(act.active_lambda_idx)}/{act.n_lambda_total} "
            f"({pct_lam:.1f}%){c_str}, tol_active={act.tol_active:.1e}")
