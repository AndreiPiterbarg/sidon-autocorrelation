"""AGENT C probe: bound-tightening audit for d=4 uniform open cell.

Goal: identify which W binds, dump SDP optimum (eps, X), and assess which
constraints are active. Out-of-tree script — does NOT edit production sources.

Cells probed:
  d=4, S=80, c=(20,20,20,20)  (most uniform)
  d=4, S=80, c=(19,21,21,19)  (palindromic perturbation)
  d=4, S=80, c=(15,15,25,25)  (asymmetric balanced)
  d=6, S=30, c=(6,4,5,5,4,6)  (palindromic d=6)
  d=8, S=16, c=(4,2,1,2,0,1,1,5)  (asymmetric d=8, zero entry)
"""
from __future__ import annotations
import os, sys, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v6 as v6

C_TARGET = 1.281


def fW_at(W, mu):
    return W.Q_coef * float(mu @ W.A @ mu)


def probe_cell(c, S, d, label):
    print(f"\n========== {label} d={d} S={S} c={list(c)} ==========")
    cell = v3.Cell.from_integer_composition(np.asarray(c, float), S)
    cache = v3.CellCache.build(cell)
    mu_star = cache.mu_star
    print(f"mu_star = {mu_star}")
    print(f"lo_eps = {cache.lo_eps}")
    print(f"hi_eps = {cache.hi_eps}")
    windows = v3.build_all_windows(d)
    bundle = v6.get_bundle(windows)

    # f_W(mu_star) for each W
    fvals = np.array([fW_at(W, mu_star) for W in windows])
    margins = fvals - C_TARGET
    order = np.argsort(-margins)
    print(f"\nTop 8 W's by f_W(mu*) - c_target:")
    for r in range(min(8, len(windows))):
        i = int(order[r])
        W = windows[i]
        print(f"  rank {r}: w_idx={i} ell={W.ell} s_lo={W.s_lo} "
              f"f_W(mu*)={fvals[i]:.5f} margin={margins[i]:+.5f}")

    # Solve L_single for top 3 binding W's; dump eps, X
    print(f"\nL_single dual analysis:")
    template = v6.get_sdp_template_v6(d)
    for r in range(min(3, len(windows))):
        i = int(order[r])
        W = windows[i]
        m_W = W.Q_coef * float(mu_star @ W.A @ mu_star) - C_TARGET
        grad_W = W.grad_coef * (W.A @ mu_star)
        Q_W = W.Q_coef * W.A
        lb, info = template.solve(cache.lo_eps, cache.hi_eps, mu_star,
                                   Q_W, grad_W, m_W)
        eps_v = info.get('eps')
        X_v = info.get('X')
        if eps_v is None or X_v is None:
            print(f"  w_idx={i} lb={lb}, no eps/X dump")
            continue
        print(f"  w_idx={i} ell={W.ell} lb={lb:+.5f}")
        print(f"    eps*       = {np.round(eps_v, 5)}")
        print(f"    diag(X*)   = {np.round(np.diag(X_v), 6)}")
        # Compare with active bounds
        diag_ub = np.maximum(cache.lo_eps**2, cache.hi_eps**2)
        # "intrinsic" lower bound on eps_i^2 when 0 not in box
        intrinsic_lb = np.where(
            cache.lo_eps * cache.hi_eps > 0,
            np.minimum(cache.lo_eps**2, cache.hi_eps**2),
            0.0,
        )
        print(f"    diag UB   = {np.round(diag_ub, 6)}")
        print(f"    diag intrinsic LB (missing!) = {np.round(intrinsic_lb, 6)}")
        # Active set check
        eps_lo_slack = eps_v - cache.lo_eps
        eps_hi_slack = cache.hi_eps - eps_v
        print(f"    eps lo-slack = {np.round(eps_lo_slack, 5)}")
        print(f"    eps hi-slack = {np.round(eps_hi_slack, 5)}")
        # PSD lift slackness: 1 - eps^T (X^-1 eps) ?
        try:
            # Build [[1, eps^T], [eps, X]] and check min eigval
            Y = np.zeros((d+1, d+1))
            Y[0,0] = 1.0
            Y[0,1:] = eps_v
            Y[1:,0] = eps_v
            Y[1:,1:] = X_v
            evs = np.linalg.eigvalsh(Y)
            print(f"    Shor lift eigvals = {np.round(evs, 6)}")
        except Exception as e:
            print(f"    Shor eig err: {e}")
        # McCormick tightness check (off-diagonal)
        print(f"    X*[off-diag] sample = {np.round(X_v - np.diag(np.diag(X_v)), 5)}")
        # Does PSD bind, or is the bottleneck linear/McCormick?
        # Check trace(Q X*) vs lb_quad_alone
        q_eps = float(grad_W @ eps_v)
        q_X = float(np.sum(Q_W * X_v))
        print(f"    margin={m_W:+.5f} grad.eps={q_eps:+.5f} tr(QX)={q_X:+.5f} sum={m_W+q_eps+q_X:+.5f}")

    # Check what the diag intrinsic LB adds in the F bound: re-evaluate F
    # with a stronger eps_i^2 LB (=intrinsic, not just McCormick min).
    # This is the candidate cut we propose.
    print(f"\nDiag-LB cut potential:")
    # F's M_lb diagonal already uses McCormick min — for diagonal:
    # min(lo^2, lo*hi, hi^2). When 0 not in [lo,hi], lo*hi has same sign
    # as min(lo^2, hi^2), but could be smaller. Check.
    M_lb_diag = np.diag(cache.M_lb)
    intrinsic_lb_diag = np.where(
        cache.lo_eps * cache.hi_eps > 0,
        np.minimum(cache.lo_eps**2, cache.hi_eps**2),
        0.0,
    )
    print(f"  F's diag M_lb (McCormick min on i=j): {np.round(M_lb_diag, 6)}")
    print(f"  Intrinsic eps_i^2 LB:                  {np.round(intrinsic_lb_diag, 6)}")
    delta = intrinsic_lb_diag - M_lb_diag
    print(f"  Delta (intrinsic - McCormick):         {np.round(delta, 6)}")
    if np.any(delta > 1e-12):
        # Hypothetical F LB improvement for the top window
        i_top = int(order[0])
        W_top = windows[i_top]
        # F bound uses Q_coef * sum(A_W * M_lb).  Replace diag M_lb with intrinsic
        # for sum-over-diag part. ALSO check for non-zero intrinsic.
        diag_contrib_old = float(np.sum(np.diag(W_top.A) * M_lb_diag))
        diag_contrib_new = float(np.sum(np.diag(W_top.A) * intrinsic_lb_diag))
        delta_F = W_top.Q_coef * (diag_contrib_new - diag_contrib_old)
        print(f"  Top W F bound boost from diag-LB cut: +{delta_F:.5f}")

    # Symmetry check
    rev = list(reversed(c))
    if list(c) == rev:
        print(f"\nSYMMETRY: c is palindromic. Could add eps_i = eps_{{d-1-i}}.")
    else:
        print(f"\nSYMMETRY: c is NOT palindromic; rev = {rev}")


if __name__ == '__main__':
    probe_cell([20,20,20,20], 80, 4, "d4 uniform")
    probe_cell([19,21,21,19], 80, 4, "d4 palindromic")
    probe_cell([15,15,25,25], 80, 4, "d4 asym balanced")
    probe_cell([6,4,5,5,4,6], 30, 6, "d6 palindromic")
    probe_cell([4,2,1,2,0,1,1,5], 16, 8, "d8 asym w/ zero")
