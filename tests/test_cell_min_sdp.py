#!/usr/bin/env python
r"""Moment-SDP relaxation of the cell-min problem.

Goal
====
Compare three rigorous lower bounds on min_{mu in Cell(mu*)} TV_W(mu):
  (a) Vertex enumeration       -- exact, O(d * 2^(d-1)) vertices.
  (b) McCormick LP             -- looser, polynomial in d.
  (c) Moment-SDP (this file)   -- tighter than McCormick, polynomial in d.

Cell(mu*) = { mu in R^d : lo_i <= mu_i <= hi_i, sum mu_i = 1 }
TV_W(mu)  = (2d/ell) * sum_{(i,j): s_lo <= i+j <= s_lo+ell-2} mu_i mu_j
          = (2d/ell) * mu^T A_W mu,   A_W indicator of the band.

Lasserre order-1 (Shor) relaxation
==================================
Lift to (mu, X) with X = mu mu^T psd via [1 mu^T; mu X] >> 0.
Localizing constraints derived from box bounds:

  (mu_i - lo_i)(mu_j - lo_j) >= 0   ==>  X_ij - lo_j mu_i - lo_i mu_j + lo_i lo_j >= 0
  (hi_i - mu_i)(hi_j - mu_j) >= 0   ==>  X_ij - hi_j mu_i - hi_i mu_j + hi_i hi_j >= 0
  (mu_i - lo_i)(hi_j - mu_j) >= 0   ==>  -X_ij + hi_j mu_i + lo_i mu_j - lo_i hi_j >= 0
  (hi_i - mu_i)(mu_j - lo_j) >= 0   ==>  -X_ij + lo_j mu_i + hi_i mu_j - hi_i lo_j >= 0

These are exactly the McCormick envelopes -- so order-1 Lasserre WITHOUT the
PSD lift is McCormick. The PSD constraint X - mu mu^T >> 0 is what makes
moment-SDP strictly tighter.

Equality sum mu = 1 enters as a moment equation: sum X_{i,j} (over j) = mu_i
(localizing with 1 - sum mu = 0), and sum mu = 1.

Objective: min (2d/ell) * sum_{(i,j) in W_pairs} c_ij X_ij,
where c_ii = 1, c_ij = 2 for i<j (since A_W is symmetric and we sum both halves).
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))

import cvxpy as cp

# Reuse helpers from the existing test
from test_joint_qp_box_cert import (
    compute_tv_w, mccormick_lp_bound, find_all_killing_windows
)


# ---------------------------------------------------------------------------
# (a) Vertex enumeration: exact min over the cell polytope
# ---------------------------------------------------------------------------
def vertex_enum_min(c_int, d, S, ell, s_lo, max_d=14):
    """Exact min via vertex enumeration of the cell polytope.

    Polytope: { lo_i <= mu_i <= hi_i, sum mu_i = 1 }.
    Vertices have d-1 of the box bounds tight; the remaining coord is fixed
    by sum=1. Enumerate all 2^(d-1) sign patterns of "which coords are at lo
    vs hi" choosing one free index, and pick those that lie in the box.
    """
    if d > max_d:
        return None  # too expensive
    mu_star = c_int.astype(np.float64) / S
    r = 1.0 / (2.0 * S)
    lo = np.maximum(mu_star - r, 0.0)
    hi = mu_star + r
    scale = 2.0 * d / ell

    A_W = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_lo + ell - 2:
                A_W[i, j] = 1.0

    best = np.inf
    n_vert = 0
    # For each free coord f, enumerate 2^{d-1} sign patterns over the others
    for f in range(d):
        others = [k for k in range(d) if k != f]
        for mask in range(1 << (d - 1)):
            mu = np.empty(d)
            s_other = 0.0
            for b, k in enumerate(others):
                if (mask >> b) & 1:
                    mu[k] = hi[k]
                else:
                    mu[k] = lo[k]
                s_other += mu[k]
            mu[f] = 1.0 - s_other
            if lo[f] - 1e-12 <= mu[f] <= hi[f] + 1e-12:
                n_vert += 1
                v = scale * mu @ A_W @ mu
                if v < best:
                    best = v
    return best, n_vert


# ---------------------------------------------------------------------------
# (c) Moment-SDP relaxation (Shor / Lasserre order 1)
# ---------------------------------------------------------------------------
def moment_sdp_bound(c_int, d, S, ell, s_lo, solver=cp.MOSEK):
    mu_star = c_int.astype(np.float64) / S
    r = 1.0 / (2.0 * S)
    lo = np.maximum(mu_star - r, 0.0)
    hi = mu_star + r
    scale = 2.0 * d / ell

    # Window pairs (unordered, i<=j)
    W_pairs = []
    for i in range(d):
        for j in range(i, d):
            if s_lo <= i + j <= s_lo + ell - 2:
                W_pairs.append((i, j))
    if not W_pairs:
        return 0.0, "no_pairs", 0.0

    mu = cp.Variable(d, nonneg=True)
    X  = cp.Variable((d, d), symmetric=True)

    cons = []
    # sum mu = 1
    cons.append(cp.sum(mu) == 1.0)
    # box bounds
    cons.append(mu >= lo)
    cons.append(mu <= hi)

    # PSD lift: M = [[1, mu^T],[mu, X]] >> 0
    one = np.array([[1.0]])
    M = cp.bmat([[one, cp.reshape(mu, (1, d), order='C')],
                 [cp.reshape(mu, (d, 1), order='C'), X]])
    cons.append(M >> 0)

    # McCormick / RLT localizing constraints from box (full d x d)
    # X_ij >= lo_i mu_j + lo_j mu_i - lo_i lo_j
    # X_ij >= hi_i mu_j + hi_j mu_i - hi_i hi_j
    # X_ij <= hi_i mu_j + lo_j mu_i - hi_i lo_j
    # X_ij <= lo_i mu_j + hi_j mu_i - lo_i hi_j
    LO = np.outer(lo, lo); HI = np.outer(hi, hi)
    LH = np.outer(lo, hi); HL = np.outer(hi, lo)
    mu_col = cp.reshape(mu, (d, 1), order='C')
    mu_row = cp.reshape(mu, (1, d), order='C')
    lo_row = lo.reshape(1, d); lo_col = lo.reshape(d, 1)
    hi_row = hi.reshape(1, d); hi_col = hi.reshape(d, 1)
    cons.append(X >= mu_col @ lo_row + lo_col @ mu_row - LO)
    cons.append(X >= mu_col @ hi_row + hi_col @ mu_row - HI)
    cons.append(X <= mu_col @ lo_row + hi_col @ mu_row - HL)
    cons.append(X <= mu_col @ hi_row + lo_col @ mu_row - LH)

    # Localizing equality from sum mu = 1: X @ 1 = mu (since X = mu mu^T => X 1 = mu)
    cons.append(X @ np.ones(d) == mu)
    # And the sum^2 = 1 second moment: 1^T X 1 = 1
    cons.append(cp.sum(X) == 1.0)

    # Objective
    obj_expr = 0
    for (i, j) in W_pairs:
        if i == j:
            obj_expr = obj_expr + X[i, i]
        else:
            obj_expr = obj_expr + 2.0 * X[i, j]
    obj = cp.Minimize(scale * obj_expr)

    prob = cp.Problem(obj, cons)
    t0 = time.time()
    try:
        prob.solve(solver=solver, verbose=False)
    except Exception as e:
        return -np.inf, f"solver_error:{e}", time.time() - t0
    dt = time.time() - t0
    return float(prob.value), prob.status, dt


# ---------------------------------------------------------------------------
# Driver: compare three bounds on a representative cell
# ---------------------------------------------------------------------------
def main():
    # Use d=12, S=12 (so cell radius r = 1/24) as the test bed.
    # Pick a "hard" canonical near-uniform composition.
    d = 12
    S = 24  # finer grid -> strictly interior cells, smaller r
    c_target = 1.28
    # Two test compositions: (A) near-uniform, (B) skewed/peaked.
    cases = [
        ("A_near_uniform", np.array([2,2,3,2,2,2,2,2,2,1,2,2], dtype=np.int64)),
        ("B_peaked",       np.array([1,1,2,3,4,4,3,2,1,1,1,1], dtype=np.int64)),
    ]
    # Pick first by default; loop runs both via for.
    case_name, c_int = cases[0]
    assert c_int.sum() == S

    sdp_times = []
    for case_name, c_int in cases:
        if c_int.sum() != S:
            continue
        print(f"=== Case {case_name}: c_int = {c_int.tolist()} ===")
        _run_one_case(c_int, d, S, c_target, sdp_times)
        print()

    avg_sdp = float(np.mean(sdp_times))
    print(f"Per-cell SDP wall-time avg: {avg_sdp:.3f} s")
    print(f"For 14e6 cells (single-thread): {14e6 * avg_sdp / 86400:.1f} days")
    print(f"For 14e6 cells on 64 cores:     {14e6 * avg_sdp / 86400 / 64:.2f} days")


def _run_one_case(c_int, d, S, c_target, sdp_times):
    wins = find_all_killing_windows(c_int, d, S, c_target=0.5)
    # Diversify: pick small ell, mid ell, large ell (=full convolution range)
    by_ell = {}
    for w in wins:
        by_ell.setdefault(w[0], w)
    chosen = []
    for target_ell in [4, 6, 10, 14, 22]:
        if target_ell in by_ell:
            chosen.append(by_ell[target_ell])
    # Always also include the original first
    if wins and wins[0] not in chosen:
        chosen.insert(0, wins[0])

    print(f"d={d}, S={S}, cell radius r={1/(2*S):.4f}")
    print()
    print(f"{'ell':>4} {'s_lo':>5} {'TV(mu*)':>10} {'vertex':>10} "
          f"{'McCorm':>10} {'mom-SDP':>10} {'LP gap':>10} {'SDP gap':>10} "
          f"{'tSDP':>7}")

    for (ell, s_lo, tv, marg) in chosen[:5]:
        vex_res = vertex_enum_min(c_int, d, S, ell, s_lo)
        if vex_res is None:
            continue
        vex, n_vert = vex_res
        lp_lb, lp_st = mccormick_lp_bound(c_int, d, S, c_target, ell, s_lo)
        sdp_lb, sdp_st, t_sdp = moment_sdp_bound(c_int, d, S, ell, s_lo)
        sdp_times.append(t_sdp)
        print(f"{ell:>4} {s_lo:>5} {tv:>10.5f} {vex:>10.5f} "
              f"{lp_lb:>10.5f} {sdp_lb:>10.5f} "
              f"{vex - lp_lb:>10.2e} {vex - sdp_lb:>10.2e} "
              f"{t_sdp:>7.3f}")


if __name__ == "__main__":
    main()
