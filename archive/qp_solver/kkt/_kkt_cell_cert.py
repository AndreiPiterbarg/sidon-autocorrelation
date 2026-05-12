"""Per-cell EXACT QCQP certificate via KKT-augmented face enumeration
across ALL windows W.

Mathematical setup
==================
For a coarse-grid composition c with sum(c)=S the cell is
    Cell = { delta in R^d : |delta_i| <= h, sum delta_i = 0 },  h = 1/(2S).
For each window W = (ell, s_lo) we have the EXACT quadratic identity
    TV_W(c/S + delta) = TV_W(c/S) + g_W . delta + s_W * delta^T A_W delta
where s_W = 2d/ell and A_W is the symmetric 0/1 window indicator.

Certification target
--------------------
We want to lower-bound  cert_box(c) = min_{delta in Cell} max_W TV_W(c/S + delta).

By weak duality (max-min <= min-max):
    cert_box(c) >= max_W min_{delta in Cell} TV_W(c/S + delta)
              = max_W [ TV_W(c/S) + min_{delta in Cell} ( g_W . delta + s_W . delta^T A_W delta ) ].

We compute the inner min EXACTLY for each W via KKT-face enumeration.
A_W is in general INDEFINITE so the QP is non-convex; however every local
minimum of a quadratic over a polytope is a critical point of the quadratic
restricted to some face F of the polytope (Karush-John). Hence enumerating
all 3^d faces and computing the unconstrained KKT critical point on each
gives the exact min (and, if numerically singular, vertex enumeration on
sub-faces is automatic via the iteration).

Per-cell exact QCQP cert
========================
Total enumeration: 3^d faces x |W| windows. At d=4 with up to 27 windows:
81 * 27 = 2187 KKT solves per cell.

Soundness (per the existing proof at top of `_kkt_exact_qp.py` + the J-bench
weak-duality identity):
- Per-window cell-min via KKT face enum is EXACT (no relaxation).
- max_W of exact per-window mins is a SOUND LB on the cell-cert by weak
  duality (max-min <= min-max).
- We never claim more than max_W min_delta; we never relax the constraint.
"""
from __future__ import annotations

import os
import sys
import time
import json
import math
import itertools
from typing import List, Sequence, Tuple

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from qp_bound import build_window_matrix, grad_for_window  # noqa: E402


# =====================================================================
# Per-face KKT MIN solve (mirror of _face_critical_points but minimising)
# =====================================================================

def _face_min_quadratic(grad, A_W, scale, h, d, fixed_idxs, fixed_signs):
    """Solve KKT system on a face for MINIMISATION of
        Q(delta) = grad . delta + scale * delta^T A_W delta
    over { delta in face : sum delta = 0, |delta_i| <= h } .

    Returns (feasible, val) where val is the critical-point value of Q
    on the face if it lies in the box; -inf flag (`feasible=False`) when
    the critical point is infeasible (sub-faces will pick it up) OR the
    KKT system is singular.

    Parameters mirror `_face_critical_points` in `_kkt_exact_qp.py`.
    """
    fixed = set(fixed_idxs)
    free_idxs = np.array([i for i in range(d) if i not in fixed],
                          dtype=np.int64)
    k = len(free_idxs)

    # 0-dim face: all coords fixed at +/-h. Feasible iff sum=0 (i.e.
    # equal number of + and -).
    if k == 0:
        s = sum(fixed_signs)
        if s != 0:
            return False, np.inf
        delta = np.zeros(d)
        for idx, sg in zip(fixed_idxs, fixed_signs):
            delta[idx] = sg * h
        val = float(grad @ delta + scale * delta @ A_W @ delta)
        return True, val
    if k == 1:
        free = int(free_idxs[0])
        fixed_sum_h = sum(fixed_signs) * h
        free_val = -fixed_sum_h
        if abs(free_val) > h + 1e-12:
            return False, np.inf
        free_val = max(-h, min(h, free_val))
        delta = np.zeros(d)
        delta[free] = free_val
        for idx, sg in zip(fixed_idxs, fixed_signs):
            delta[idx] = sg * h
        val = float(grad @ delta + scale * delta @ A_W @ delta)
        return True, val

    # General face k >= 2. Lagrangian:
    #   L(delta_J, mu) = grad_J . delta_J + scale * delta_J^T A^{JJ} delta_J
    #                    + cross-terms-with-fixed + mu (sum_J delta_J + fixed_sum_h)
    # Stationary on free coords:
    #   grad_J + 2*scale*A^{JJ} delta_J + 2*scale*A^{JI} delta_I + mu * 1 = 0
    # Constraint: 1^T delta_J = -fixed_sum_h.
    #
    # H_min := 2*scale*A^{JJ}                (Hessian of the quadratic part)
    # b_J   := grad_J + 2*scale*A^{JI} delta_I_fixed
    # KKT system:
    #     [ H_min   1 ] [ delta_J ]   [ -b_J          ]
    #     [ 1^T     0 ] [   mu    ] = [ -fixed_sum_h  ]
    fixed_sum_h = sum(fixed_signs) * h
    delta_I = np.zeros(d, dtype=np.float64)
    for idx, sg in zip(fixed_idxs, fixed_signs):
        delta_I[idx] = sg * h
    g_J = grad[free_idxs]
    A_JJ = A_W[np.ix_(free_idxs, free_idxs)]
    A_JI_delta = A_W[free_idxs, :] @ delta_I  # shape (k,) — coupling to fixed coords
    H_min = 2.0 * scale * A_JJ
    b_J = g_J + 2.0 * scale * A_JI_delta

    M = np.zeros((k + 1, k + 1), dtype=np.float64)
    M[:k, :k] = H_min
    M[:k, k] = 1.0
    M[k, :k] = 1.0
    rhs = np.zeros(k + 1, dtype=np.float64)
    rhs[:k] = -b_J
    rhs[k] = -fixed_sum_h

    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        # Singular Hessian-with-Lagrangian => critical-set has positive
        # dimension along the face. Sub-faces (one more coord fixed at
        # +/-h) cover the boundary; we conservatively skip the interior.
        return False, np.inf

    delta_J = sol[:k]
    tol = 1e-12 + 1e-10 * h
    if np.any(np.abs(delta_J) > h + tol):
        return False, np.inf
    delta_J = np.clip(delta_J, -h, h)

    delta = delta_I.copy()
    delta[free_idxs] = delta_J
    val = float(grad @ delta + scale * delta @ A_W @ delta)
    return True, val


# =====================================================================
# Per-window EXACT cell-min: enumerate all 3^d faces
# =====================================================================

def cell_min_kkt(grad, A_W, scale, h, d):
    """EXACT min over Cell of  Q(delta) = grad . delta + scale * delta^T A_W delta.

    Iterate over the 3^d faces of Cell (subsets I assigned to {+h, -h} and
    free coords J subject to sum-zero affine + interior box). On each face
    compute the unconstrained KKT critical point; collect those whose
    delta_J lies in [-h, h] and evaluate Q. The global min is attained at
    one of these critical points (Karush-John). delta=0 is always feasible
    so val_min <= 0.
    """
    grad = np.ascontiguousarray(grad, dtype=np.float64)
    A_W = np.ascontiguousarray(A_W, dtype=np.float64)
    best = 0.0  # Q(0) = 0
    indices = list(range(d))
    for t in range(0, d + 1):
        for fixed_idxs in itertools.combinations(indices, t):
            for sign_mask in range(1 << t):
                fixed_signs = tuple(
                    +1 if (sign_mask >> j) & 1 else -1 for j in range(t)
                )
                feas, val = _face_min_quadratic(
                    grad, A_W, scale, h, d, fixed_idxs, fixed_signs
                )
                if feas and val < best:
                    best = val
    return best


# =====================================================================
# Top-level cell certification: max over W of EXACT per-window min
# =====================================================================

def cell_cert_kkt(c_int, S: int, d: int, c_target: float,
                  windows: Sequence[Tuple[int, int]] = None,
                  return_witness: bool = False):
    """SOUND lower bound on cert_box(c) via per-window EXACT KKT min.

    cert_LB := max_W [ TV_W(c/S) + min_{delta in Cell} (g_W . delta + s_W * delta^T A_W delta) ]

    By weak duality cert_LB <= cert_box(c) := min_delta max_W TV_W(c+delta),
    so cert_LB >= c_target implies the cell is certified.

    Args
    ----
    c_int     : integer composition (length d, sum=S).
    S, d      : grid params.
    c_target  : the threshold to certify against.
    windows   : list of (ell, s_lo). If None, enumerates all valid windows
                where TV_W(c/S) > c_target (the only ones that could
                certify; the rest contribute trivially smaller LBs).

    Returns
    -------
    (cert_LB, best_W) when return_witness else cert_LB.
    """
    c_f = np.asarray(c_int, dtype=np.float64)
    h = 1.0 / (2.0 * S)
    if windows is None:
        windows = _candidate_windows(c_f, S, d, c_target)
    if not windows:
        if return_witness:
            return -1e30, (-1, -1)
        return -1e30

    best_LB = -1e30
    best_W = (-1, -1)
    for (ell, s_lo) in windows:
        A_W = build_window_matrix(d, ell, s_lo)
        scale = 2.0 * d / ell
        # tv0 = (2d/ell) * mu*^T A_W mu*
        mu = c_f / float(S)
        tv0 = float(scale * (mu @ A_W @ mu))
        grad = grad_for_window(c_f, A_W, S, d, ell)
        m_min = cell_min_kkt(grad, A_W, scale, h, d)
        LB = tv0 + m_min
        if LB > best_LB:
            best_LB = LB
            best_W = (ell, s_lo)
    if return_witness:
        return best_LB, best_W
    return best_LB


def _candidate_windows(c_f, S, d, c_target):
    """All windows with TV_W(c/S) > c_target  (the only ones that can
    yield a useful per-window LB; others give cert_LB < c_target trivially)."""
    out = []
    conv_len = 2 * d - 1
    mu = c_f / float(S)
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            A = build_window_matrix(d, ell, s_lo)
            tv0 = (2.0 * d / ell) * float(mu @ A @ mu)
            if tv0 > c_target:
                out.append((ell, s_lo))
    return out


# =====================================================================
# Bench: load real uncert cells from cache and compare KKT vs Shor
# =====================================================================

def _load_hardcells_cache(d: int, S: int, c_target: float,
                          path='_coarse_L2_hardcells_cache.json'):
    with open(path) as f:
        data = json.load(f)
    key = f'{d}_{S}_{c_target}'
    if key not in data:
        # Try without the .0 suffix
        for k in data.keys():
            if k.startswith(f'{d}_{S}_'):
                key = k
                break
        else:
            raise KeyError(f'No cache entry for d={d}, S={S}, c={c_target}; '
                            f'available: {list(data.keys())}')
    entry = data[key]
    return entry


def _filter_shor_failing(rows, d, S, c_target, max_screen=200,
                         solver='MOSEK'):
    """Of the provided hard cells (triangle-failing), keep only those
    where Shor SDP also fails — these are the truly uncert cells the new
    KKT cert is supposed to attack."""
    from _coarse_L_bench import cell_cert_shor
    out = []
    rows_sorted = sorted(rows, key=lambda r: r['tri_net'])
    for k, row in enumerate(rows_sorted[:max_screen]):
        c = np.asarray(row['c'], dtype=np.int32)
        Wstar = tuple(row['tri_W'])
        lb, _ = cell_cert_shor(c, S, d, c_target, Wstar, solver=solver)
        cert = lb >= c_target - 1e-9
        if not cert:
            out.append({**row, 'shor_lb_bestW': float(lb)})
    return out


def bench_against_shor(d: int, S: int, c_target: float,
                        max_cells: int = 50, solver: str = 'MOSEK',
                        use_shor_failing_subset: bool = True,
                        verbose: bool = True):
    """Run KKT cert on the {Shor-failing OR triangle-failing} cells from
    the cache and compare cert rate, time, and tightness vs Shor SDP.

    Args
    ----
    use_shor_failing_subset : if True, screen via Shor (best-W) first; only
        run KKT on cells that Shor failed to certify. If False, take the
        most-negative-tri-net cells directly.
    """
    if verbose:
        print(f"\n=== KKT cell-cert vs Shor SDP @ d={d} S={S} c={c_target} ===")

    cache = _load_hardcells_cache(d, S, c_target)
    rows = cache['rows']
    n_grid_pass = cache['n_grid_pass']
    n_tri_cert = cache['n_tri_cert']
    if verbose:
        print(f"    grid passers: {n_grid_pass:,}  tri_cert: {n_tri_cert:,}  "
              f"hard cells: {len(rows):,}")

    if use_shor_failing_subset:
        if verbose:
            print(f"    screening with Shor (best-W) to find Shor-failing...")
        truly_hard = _filter_shor_failing(rows, d, S, c_target,
                                         max_screen=min(200, len(rows)),
                                         solver=solver)
        if verbose:
            print(f"    Shor-failing (best-W): {len(truly_hard)}")
        cells = truly_hard[:max_cells]
        mode = 'shor_failing'
    else:
        cells = sorted(rows, key=lambda r: r['tri_net'])[:max_cells]
        mode = 'top_hard'

    if not cells:
        print("    No cells to test.")
        return None

    # Run KKT, full multi-window Shor, and triangle baseline side by side.
    from _coarse_L_bench import cell_cert_shor, cell_cert_shor_max, all_windows, tv_at
    n_kkt_cert = 0
    n_shor_cert = 0
    n_kkt_strict = 0  # KKT certifies but Shor (best-W) does not
    n_kkt_strict_over_full = 0  # KKT certifies but multi-W Shor does not
    times_kkt = []
    times_shor = []
    rows_out = []

    for k, row in enumerate(cells):
        c = np.asarray(row['c'], dtype=np.int32)
        # Use the same candidate windows for both
        all_W = [(ell, s) for (ell, s) in all_windows(d)
                 if tv_at(c, S, d, ell, s) > c_target]

        t0 = time.time()
        kkt_lb, kkt_W = cell_cert_kkt(c, S, d, c_target, windows=all_W,
                                       return_witness=True)
        dt_kkt = time.time() - t0
        times_kkt.append(dt_kkt)

        t0 = time.time()
        shor_lb, shor_W = cell_cert_shor_max(c, S, d, c_target, all_W,
                                              solver=solver)
        dt_shor = time.time() - t0
        times_shor.append(dt_shor)

        kkt_cert = kkt_lb >= c_target - 1e-9
        shor_cert = shor_lb >= c_target - 1e-9
        if kkt_cert: n_kkt_cert += 1
        if shor_cert: n_shor_cert += 1
        if kkt_cert and not shor_cert:
            n_kkt_strict_over_full += 1

        rows_out.append({
            'k': k, 'c': c.tolist(),
            'tri_net': float(row['tri_net']),
            'kkt_lb': float(kkt_lb),
            'shor_lb': float(shor_lb),
            'kkt_cert': bool(kkt_cert),
            'shor_cert': bool(shor_cert),
            'kkt_W': list(kkt_W),
            'shor_W': list(shor_W),
            'time_kkt_ms': float(dt_kkt * 1000),
            'time_shor_ms': float(dt_shor * 1000),
            'n_W': len(all_W),
        })

        if verbose and k < 8:
            print(f"    [{k:3d}] c={c.tolist()}  tri_net={row['tri_net']:+.5f}  "
                  f"shor_lb={shor_lb:.5f}{' CERT' if shor_cert else '     '}  "
                  f"kkt_lb={kkt_lb:.5f}{' CERT' if kkt_cert else '     '}  "
                  f"|W|={len(all_W)}  t_kkt={dt_kkt*1000:.0f}ms")

    times_kkt = np.asarray(times_kkt)
    times_shor = np.asarray(times_shor)
    if verbose:
        n = len(cells)
        print(f"\n    --- Summary ({mode}, {n} cells) ---")
        print(f"    Shor multi-W cert: {n_shor_cert:,} / {n} "
              f"({100.0*n_shor_cert/n:.1f}%)")
        print(f"    KKT cell-cert    : {n_kkt_cert:,} / {n} "
              f"({100.0*n_kkt_cert/n:.1f}%)")
        print(f"    KKT strict over Shor multi-W: {n_kkt_strict_over_full:,} / {n}")
        print(f"    Time/cell (ms): KKT med={1000*np.median(times_kkt):.0f} "
              f"max={1000*np.max(times_kkt):.0f} | "
              f"Shor med={1000*np.median(times_shor):.0f} "
              f"max={1000*np.max(times_shor):.0f}")

    return {
        'd': d, 'S': S, 'c_target': c_target, 'mode': mode,
        'n_cells': len(cells),
        'n_kkt_cert': n_kkt_cert,
        'n_shor_cert': n_shor_cert,
        'n_kkt_strict_over_shor_multiW': n_kkt_strict_over_full,
        'time_kkt_med_ms': float(1000 * np.median(times_kkt)),
        'time_kkt_max_ms': float(1000 * np.max(times_kkt)),
        'time_shor_med_ms': float(1000 * np.median(times_shor)),
        'time_shor_max_ms': float(1000 * np.max(times_shor)),
        'rows': rows_out,
    }


# =====================================================================
# Soundness self-check via fine-grid sampling
# =====================================================================

def soundness_check(d: int, S: int, c_target: float,
                    n_cells: int = 10, n_grid: int = 9,
                    seed: int = 0, verbose: bool = True):
    """Verify KKT cert_LB <= sup over the fine grid of (min over delta_grid
    of max over W of TV_W). i.e., fine-grid empirical cert_box (an upper
    bound on cert_box from finite sampling) >= our LB.

    Any violation > 1e-8 indicates an unsoundness bug.
    """
    cache = _load_hardcells_cache(d, S, c_target)
    rows = cache['rows']
    rng = np.random.default_rng(seed)
    if len(rows) > n_cells:
        idx = rng.choice(len(rows), n_cells, replace=False)
        cells = [rows[i] for i in idx]
    else:
        cells = rows[:n_cells]
    h = 1.0 / (2.0 * S)
    grid = np.linspace(-h, h, n_grid)
    n_viol = 0
    max_excess = 0.0
    for row in cells:
        c = np.asarray(row['c'], dtype=np.int32)
        from _coarse_L_bench import all_windows, tv_at
        Ws = [(ell, s) for (ell, s) in all_windows(d)
              if tv_at(c, S, d, ell, s) > c_target]
        if not Ws:
            continue
        kkt_lb, _ = cell_cert_kkt(c, S, d, c_target, windows=Ws,
                                    return_witness=True)
        # Fine-grid evaluation
        A_list = [build_window_matrix(d, ell, s) for ell, s in Ws]
        s_list = [2.0 * d / ell for ell, _ in Ws]
        mu_star = c.astype(np.float64) / float(S)
        best_max = math.inf
        for tup in itertools.product(grid, repeat=d - 1):
            last = -sum(tup)
            if abs(last) > h + 1e-12:
                continue
            delta = np.array(list(tup) + [last])
            mu = mu_star + delta
            if (mu < -1e-12).any():
                continue
            max_tv = -math.inf
            for A, sw in zip(A_list, s_list):
                tv = sw * float(mu @ A @ mu)
                if tv > max_tv:
                    max_tv = tv
            if max_tv < best_max:
                best_max = max_tv
        # Sound: kkt_lb <= best_max (cert_box upper-bound from finite grid).
        excess = kkt_lb - best_max
        if excess > 1e-8:
            n_viol += 1
            if excess > max_excess:
                max_excess = excess
            if verbose:
                print(f"  VIOL  c={c.tolist()}  KKT={kkt_lb:.6f}  "
                      f"grid_minmax={best_max:.6f}  excess={excess:.2e}")
    if verbose:
        print(f"  soundness: {n_viol} violations / {len(cells)} cells, "
              f"max excess={max_excess:.2e}")
    return n_viol, max_excess


# =====================================================================
# Driver
# =====================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--S', type=int, default=20)
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--max_cells', type=int, default=50)
    ap.add_argument('--solver', default='MOSEK')
    ap.add_argument('--mode', choices=['shor_failing', 'top_hard'],
                     default='shor_failing')
    ap.add_argument('--soundness', action='store_true')
    ap.add_argument('--out', default='_kkt_cell_cert_results.json')
    args = ap.parse_args()

    if args.soundness:
        print(f"\n[soundness] d={args.d} S={args.S} c={args.c_target}")
        soundness_check(args.d, args.S, args.c_target, n_cells=10, n_grid=9)

    res = bench_against_shor(
        args.d, args.S, args.c_target, max_cells=args.max_cells,
        solver=args.solver,
        use_shor_failing_subset=(args.mode == 'shor_failing'),
    )
    if res is not None:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
