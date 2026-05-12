"""Joint multi-window cell certification.

The current per-window QP (qp_bound.py) computes, for each pruning window W:
    bound_W = max_{δ∈Cell} [- grad_W·δ - (2d/ell) δ^T A_W δ]
and certifies the cell if margin_W > bound_W for SOME W.

This module implements a STRONGER bound: for each composition c, find the
worst δ that simultaneously minimises max_W TV_W(c+δ). The cell is certified
iff this min-max is >= c_target.

Mathematically:
    cert_joint(c) = min_{δ∈Cell} max_W TV_W(c+δ)
                  = min_{δ∈Cell} max_W [TV_W(c) + grad_W·δ + (2d/ell) δ^T A_W δ]
                  = min_{δ∈Cell} max_W [margin_W + grad_W·δ + (2d/ell) δ^T A_W δ + c_target]

The cell is certified iff cert_joint(c) >= c_target, equivalently iff:
    min_{δ∈Cell} max_W [margin_W + grad_W·δ + (2d/ell) δ^T A_W δ] >= 0.

Compared to the per-window bound:
  - per-window: cert iff EXISTS W with margin_W > QP_W(δ for that W alone)
  - joint:      cert iff (min_δ max_W) >= 0, which is a saddle point

The joint bound is provably TIGHTER (a single δ that minimises one window
generally hurts another, so the worst-case simultaneous TV across windows
is higher than any single-window worst).

Implementation: for each pruning window's δ, evaluate ALL windows' TV.
Take the per-δ max. Then min over δ candidates. We use vertex enumeration
of Cell vertices (same as qp_bound.py).

Complexity: O(2^(d-1) * d * |W|^2) — for d ≤ 8 and |W| up to d^2, manageable.
"""
from __future__ import annotations

import numpy as np
import numba
from numba import njit


@njit(cache=True)
def joint_bound_vertex(c_int, S, d, c_target, pruning_windows_ell,
                        pruning_windows_s, n_pruning):
    """Joint multi-window cell cert via vertex enumeration.

    For each vertex δ of Cell:
      - Compute (mu* + δ) where mu* = c/S
      - For each pruning window W = (ell, s): compute TV_W(mu*+δ)
      - Take max over W
    Take min over vertices.

    Returns the joint min-max value (TV at the worst-case δ).
    Cell is certified iff this >= c_target.

    For each composition c, pruning_windows_(ell, s, n) lists the windows
    where TV(c) >= c_target (i.e., the windows that prune c at the grid pt).
    Only these windows can certify the cell — others have margin < 0.

    Args:
      c_int: integer mass coords, shape (d,)
      S: grid resolution
      d: number of bins
      c_target: target lower bound
      pruning_windows_ell: ell values, shape (n_pruning,)
      pruning_windows_s:   s values, shape (n_pruning,)
      n_pruning: number of pruning windows

    Returns:
      Joint min-max TV.
    """
    h = 1.0 / (2.0 * S)
    tol = h * 1e-9
    n_pat = 1 << (d - 1)

    best_min = 1e30  # min over δ of (max over W of TV)

    # Continuous mu* = c_int / S
    mu_star = np.empty(d, dtype=np.float64)
    for i in range(d):
        mu_star[i] = c_int[i] / S

    delta = np.zeros(d, dtype=np.float64)

    for free_idx in range(d):
        for mask in range(n_pat):
            sum_others = 0.0
            bit_pos = 0
            for i in range(d):
                if i == free_idx:
                    continue
                bit = (mask >> bit_pos) & 1
                if bit == 0:
                    delta[i] = -h
                else:
                    delta[i] = h
                sum_others += delta[i]
                bit_pos += 1
            free_val = -sum_others
            if free_val < -h - tol or free_val > h + tol:
                continue
            if free_val > h:
                free_val = h
            elif free_val < -h:
                free_val = -h
            delta[free_idx] = free_val

            # mu = mu* + δ; verify mu >= 0
            ok = True
            for i in range(d):
                if mu_star[i] + delta[i] < -1e-12:
                    ok = False
                    break
            if not ok:
                continue

            # For this δ, compute TV at all pruning windows
            max_tv = -1e30
            for w in range(n_pruning):
                ell = pruning_windows_ell[w]
                s_lo = pruning_windows_s[w]
                s_hi = s_lo + ell - 2
                # TV_W = (2d/ell) * sum over (i,j) in window of (mu_i+δ_i)(mu_j+δ_j)
                tv = 0.0
                for i in range(d):
                    mi = mu_star[i] + delta[i]
                    if mi == 0.0:
                        continue
                    j_lo = max(0, s_lo - i)
                    j_hi = min(d - 1, s_hi - i)
                    for j in range(j_lo, j_hi + 1):
                        mj = mu_star[j] + delta[j]
                        tv += mi * mj
                tv = (2.0 * d / ell) * tv
                if tv > max_tv:
                    max_tv = tv

            if max_tv < best_min:
                best_min = max_tv

    return best_min


@njit(cache=True)
def joint_cell_cert_for_composition(c_int, S, d, c_target):
    """Top-level: compute joint cert for one composition c_int.

    First identifies pruning windows (where TV(c) >= c_target at grid pt),
    then computes joint min-max via vertex enumeration.

    Returns:
      cert_value: min over δ of max over W of TV_W(mu* + δ)
                  Cell is certified iff cert_value >= c_target.
      n_pruning_windows: number of windows that prune c at grid pt
    """
    # Step 1: find pruning windows and compute conv
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        ci = c_int[i]
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = c_int[j]
                if cj != 0:
                    conv[i + j] += 2 * ci * cj

    # Find pruning windows
    max_ell = 2 * d
    S_sq = float(S) * float(S)
    d_d = float(d)
    eps = 1e-9

    # First pass: count
    n_pruning = 0
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        ws = 0
        for k in range(n_cv):
            ws += conv[k]
        ell_f = float(ell)
        thr = c_target * ell_f * S_sq / (2.0 * d_d) - eps
        for s_lo in range(n_windows):
            if s_lo > 0:
                ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
            if ws > thr:
                n_pruning += 1

    if n_pruning == 0:
        return -1.0e30, 0  # no pruning windows -- cell not pruned

    # Second pass: collect
    pruning_ell = np.empty(n_pruning, dtype=np.int32)
    pruning_s = np.empty(n_pruning, dtype=np.int32)
    idx = 0
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        ws = 0
        for k in range(n_cv):
            ws += conv[k]
        ell_f = float(ell)
        thr = c_target * ell_f * S_sq / (2.0 * d_d) - eps
        for s_lo in range(n_windows):
            if s_lo > 0:
                ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
            if ws > thr:
                pruning_ell[idx] = ell
                pruning_s[idx] = s_lo
                idx += 1

    # Compute joint cert via vertex enumeration
    cert = joint_bound_vertex(c_int, S, d, c_target,
                               pruning_ell, pruning_s, n_pruning)
    return cert, n_pruning
