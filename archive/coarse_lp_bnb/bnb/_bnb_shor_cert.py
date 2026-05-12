"""Per-cell branch-and-bound certification with Shor SDP at the leaves.

================================================================
WHY DEPTH-k BnB-WITH-LEAVES-AT-SHOR PUSHES PAST DEPTH-k LINEAR-BnB
================================================================
The earlier `_coarse_BnB_bench.py` implementation uses *linear-correction
bounds* at each leaf cell.  Those bounds shrink only along axes that have
been split — so to halve the bound globally you need depth ~ d.  This
explains the earlier observation:  depth-4 BnB cert ~40% at d=8 (loose
than Shor's 98%).

By replacing the leaf bound with a **single-cell Shor SDP**, each leaf is
already as tight as Shor on the parent cell — but applied to a half-cell.
Subdividing the box and re-running Shor on the sub-cell uses the tighter
RLT/McCormick constraints (lo/hi of x_i = c_i + S*delta_i shrink), which
typically certify many sub-cells that Shor on the parent could not.

================================================================
SOUNDNESS
================================================================
Cell(c) = { mu = c/S + delta : -h <= delta_i <= h, sum delta = 0, mu_i >= 0 }

Subdividing along axis i*: split [lo[i*], hi[i*]] at the midpoint into
[lo[i*], mid] and [mid, hi[i*]].  Each sub-cell is a (closed) subset of
Cell(c); their union is Cell(c).  AND-aggregate at parent: parent
certified iff BOTH sub-cells certified.  This preserves soundness.

At each leaf, Shor SDP feasibility:
   minimise  0
   s.t.  Y = [[1, x'], [x, X]] >> 0
         x_i in [lo_x[i], hi_x[i]]  (in x-units; x_i = c_i + S*delta_i)
         sum x = S
         McCormick: lo*lo <= X[i,i] <= hi*hi  (and standard envelopes)
                    RLT for off-diagonals
         per-window: Tr(A_W X) <= c_target * ell * S^2 / (2d) + eps   (forall W)

If MOSEK reports `infeasible`, then NO point (x, X) in the lifted feasible
region satisfies all window constraints — in particular, NO genuine x in
the cell satisfies max_W TV_W(x/S) <= c_target.  The sub-cell is therefore
SOUNDLY certified: every mu in the sub-cell has max_W TV_W(mu) > c_target.

Status `optimal` (or anything other than `infeasible`) is a soft fail:
either the SDP relaxation has a feasible point (possibly spurious vis-a-vis
the QP) or the solver couldn't conclude.  We treat this as "uncertified"
and let BnB recurse.

Recursion:
- If parent's Shor returns infeasible → certified, no recursion needed.
- If parent's Shor returns optimal/etc and depth < max_depth →
  split along i* (axis of largest box width × largest grad-magnitude),
  recurse on both halves.  AND-aggregate.
- If parent's Shor returns optimal and depth == max_depth →
  return False (cell uncertified by this driver — sound: just means
  "we couldn't certify within the depth budget").

================================================================
BRANCHING HEURISTIC
================================================================
Split axis i* = argmax_i (width_i * abs(grad_i)) where
    width_i = hi_x[i] - lo_x[i]
    grad_i = sum_W lambda_W * (4d/(ell*S)) * (A_W * mu_anchor)[i]
We use a uniform lambda for simplicity.  This focuses splits on the axes
with the steepest TV-direction-weighted reach.

================================================================
USAGE
================================================================
    cell_cert_bnb_shor(c_int, S, d, c_target, max_depth=8) -> bool
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


# =====================================================================
# Window enumeration (same as _coarse_BnB_bench.py / _coarse_L_bench.py)
# =====================================================================

def _enumerate_windows(d):
    out = []
    conv_len = 2 * d - 1
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            out.append((ell, s_lo))
    return out


def _build_A(W, d):
    ell, s_lo = W
    s_hi = s_lo + ell - 2
    A = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                A[i, j] = 1.0
    return A


# =====================================================================
# Shor SDP feasibility on a sub-cell
# =====================================================================

def _shor_cell_feasible(c_int, S, d, c_target, lo_delta, hi_delta,
                          A_list, ell_list, tol=1e-9, eps_margin=1e-9):
    """Shor SDP feasibility test on a sub-cell.

    Sub-cell (in delta-units): delta_i in [lo_delta[i], hi_delta[i]],
    sum delta = 0.  In x = c + S*delta units this is
        x_i in [c_i + S*lo_delta[i], c_i + S*hi_delta[i]]    (clamped >= 0)
        sum x = S.

    Returns (infeasible, status):
        infeasible = True  iff cell is certified (no x in cell with
                              all TV_W(mu) <= c_target)
        status = MOSEK status string
    """
    if not _HAS_CVXPY:
        return False, 'NO_CVXPY'

    c_arr = np.asarray(c_int, dtype=np.float64)
    lo_x = np.maximum(0.0, c_arr + S * lo_delta)
    hi_x = c_arr + S * hi_delta
    # Clamp tiny floating-point inversions
    if np.any(lo_x > hi_x + 1e-12):
        # sub-cell empty after mu>=0 clamp -> trivially "infeasible" / certified
        return True, 'cell_empty_after_mu_ge_0'

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)
    ones11 = np.ones((1, 1))
    Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                  [cp.reshape(x, (d, 1), order='C'), X]])
    cons = [Y >> 0]
    cons += [x >= lo_x, x <= hi_x]
    cons += [cp.sum(x) == float(S)]

    # McCormick envelopes for each x_i^2 = X[i,i]
    for i in range(d):
        li, ui = lo_x[i], hi_x[i]
        cons += [X[i, i] >= li * li, X[i, i] <= ui * ui]
        cons += [X[i, i] >= 2.0 * li * x[i] - li * li]
        cons += [X[i, i] >= 2.0 * ui * x[i] - ui * ui]
        cons += [X[i, i] <= (li + ui) * x[i] - li * ui]

    # RLT for off-diagonals
    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo_x[i], lo_x[j]
            ui, uj = hi_x[i], hi_x[j]
            # X[i,j] >= li*x_j + lj*x_i - li*lj
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]

    # Window threshold: TV_W <= c_target ⇒ x^T A_W x <= c_target * ell * S^2 / (2d)
    thr_eps = eps_margin * S * S
    for w_idx, A_W in enumerate(A_list):
        ell = ell_list[w_idx]
        thr = c_target * ell * S * S / (2.0 * d) + thr_eps
        cons += [cp.trace(A_W @ X) <= thr]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver='MOSEK', verbose=False,
                    mosek_params={
                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                    })
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'
    return prob.status == 'infeasible', str(prob.status)


# =====================================================================
# Branching heuristic: pick the axis to split
# =====================================================================

def _pick_split_axis(c_int, S, d, lo_delta, hi_delta, A_list, ell_list):
    """Choose i* = argmax_i ( width_i * |grad_uniform_i| ).
    width_i = hi_delta[i] - lo_delta[i].
    grad_uniform_i = sum_W (4d/(ell*S)) * (A_W mu_anchor)[i]   uniform lambda.
    Skip axes with width<=0 (already pinned).
    """
    width = hi_delta - lo_delta
    mu_star = c_int.astype(np.float64) / S
    c_box = (lo_delta + hi_delta) / 2.0
    c_prime = c_box - float(np.mean(c_box))
    mu_anchor = mu_star + c_prime

    grad_total = np.zeros(d, dtype=np.float64)
    for w_idx in range(len(A_list)):
        A_W = A_list[w_idx]
        ell = ell_list[w_idx]
        grad_total += (4.0 * d / (ell * S)) * (A_W @ mu_anchor)
    grad_total /= max(1, len(A_list))

    score = width * np.abs(grad_total)
    score[width <= 1e-15] = -np.inf
    if np.all(score == -np.inf):
        return -1
    return int(np.argmax(score))


# =====================================================================
# Main BnB driver: Shor at every node, AND-aggregate over splits
# =====================================================================

def cell_cert_bnb_shor(c_int, S, d, c_target, max_depth=8,
                         A_list=None, ell_list=None,
                         tol=1e-9, eps_margin=1e-9,
                         collect_stats=False):
    """Sound BnB cert with Shor SDP at every node.

    Returns True iff the cell is certified within depth budget.
    If `collect_stats=True`, returns (ok, stats) with per-depth counters.

    Algorithm:
      1) Run Shor on the parent.
         If infeasible → certified.
         Else: pick split axis, recurse on two halves, AND-aggregate.
      2) At depth == max_depth and Shor still feasible → return False.
    """
    if A_list is None:
        windows = _enumerate_windows(d)
        A_list = [_build_A(W, d) for W in windows]
        ell_list = [W[0] for W in windows]

    h = 1.0 / (2.0 * S)
    mu_star = c_int.astype(np.float64) / S
    lo0 = np.maximum(-h, -mu_star)
    hi0 = np.full(d, h)

    stats = {
        'shor_calls': 0,
        'splits': 0,
        'depth_required': 0,
        'shor_infeasible': 0,
        'shor_optimal': 0,
        'shor_other': 0,
    }

    def recurse(lo_delta, hi_delta, depth):
        # Try Shor on this sub-cell.
        infeas, status = _shor_cell_feasible(
            c_int, S, d, c_target, lo_delta, hi_delta,
            A_list, ell_list, tol=tol, eps_margin=eps_margin)
        stats['shor_calls'] += 1
        if infeas:
            stats['shor_infeasible'] += 1
            if depth > stats['depth_required']:
                stats['depth_required'] = depth
            return True
        if status == 'optimal' or status == 'optimal_inaccurate':
            stats['shor_optimal'] += 1
        else:
            stats['shor_other'] += 1
        if depth >= max_depth:
            return False
        # Split.
        i_star = _pick_split_axis(c_int, S, d, lo_delta, hi_delta,
                                   A_list, ell_list)
        if i_star < 0:
            return False
        mid = (lo_delta[i_star] + hi_delta[i_star]) / 2.0
        if abs(hi_delta[i_star] - lo_delta[i_star]) < 1e-15:
            return False
        stats['splits'] += 1
        # Sub-cell 1: delta_i* in [lo_i*, mid]
        lo1 = lo_delta.copy()
        hi1 = hi_delta.copy()
        hi1[i_star] = mid
        # Sub-cell 2: delta_i* in [mid, hi_i*]
        lo2 = lo_delta.copy()
        hi2 = hi_delta.copy()
        lo2[i_star] = mid
        # AND-aggregate; short-circuit on first failure to save SDPs.
        if not recurse(lo1, hi1, depth + 1):
            return False
        if not recurse(lo2, hi2, depth + 1):
            return False
        return True

    ok = recurse(lo0, hi0, 0)
    if collect_stats:
        return ok, stats
    return ok


# =====================================================================
# Bench driver: cert rate at depth 1..8, time per cell
# =====================================================================

def _gather_uncert_cells(d, S, c_target, n_target=30, verbose=True):
    """Reuse _coarse_BnB_bench's pipeline to find uncert cells.

    Specifically: run baseline + N+O pruners to get hard cells, and within
    those, find the ones where Shor's parent SDP also fails (i.e., the cells
    where depth-1+ BnB is needed)."""
    from _coarse_BnB_bench import (precompute_op_rest, prune_coarse_NO,
                                    prune_coarse_baseline, enum_compositions,
                                    _shor_cert as _shor_parent)
    op_rest, _ = precompute_op_rest(d, 2 * d)
    op_rest_d = op_rest * d
    batch = np.array(list(enum_compositions(d, S)), dtype=np.int32)
    if verbose:
        print(f"  total comps: {len(batch):,}", flush=True)

    # Warmup
    warm = batch[:1]
    _ = prune_coarse_baseline(warm, d, S, c_target)
    _ = prune_coarse_NO(warm, d, S, c_target, op_rest_d)

    t0 = time.time()
    s_NO = prune_coarse_NO(batch, d, S, c_target, op_rest_d)
    t_NO = time.time() - t0
    if verbose:
        print(f"  NO survivors: {int(s_NO.sum()):,}  ({t_NO:.3f}s)", flush=True)

    survivor_idx = np.where(s_NO)[0]
    if len(survivor_idx) == 0:
        return [], batch

    # Build A and ell lists (need the parent Shor)
    windows = _enumerate_windows(d)
    A_list = [_build_A(W, d) for W in windows]
    ell_list = [W[0] for W in windows]

    # Find uncert cells: parent Shor returns optimal (= cell may still
    # contain feasible distributions with TV<=c_target).  Iterate until
    # n_target found.  BnB at depth 0 == parent Shor cert; we want cells
    # where parent SDP fails so that BnB at depth>=1 makes a difference.
    uncert = []
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(survivor_idx))
    if verbose:
        print(f"  scanning for {n_target} uncert cells...", flush=True)
    for k in perm:
        ci = int(survivor_idx[k])
        cell = batch[ci]
        infeas, status = _shor_parent(cell, S, d, c_target, A_list, ell_list)
        if not infeas:
            uncert.append((ci, cell.copy()))
            if len(uncert) >= n_target:
                break
    if verbose:
        print(f"  collected {len(uncert)} uncert cells", flush=True)
    return uncert, batch


def bench(d=4, S=200, c_target=1.281, n_cells=30, max_depth=8,
            output_path=None, verbose=True):
    """Run BnB-Shor at depths 1..max_depth on n_cells uncert cells.

    Reports cert rate and time per cell."""
    if not _HAS_CVXPY:
        print("ERROR: cvxpy not available")
        return None

    if verbose:
        print(f"\n=== _bnb_shor_cert.bench: d={d}, S={S}, c={c_target}, "
              f"n_cells={n_cells}, max_depth={max_depth} ===", flush=True)

    # Build window structures once
    windows = _enumerate_windows(d)
    A_list = [_build_A(W, d) for W in windows]
    ell_list = [W[0] for W in windows]
    if verbose:
        print(f"  n_windows = {len(windows)}", flush=True)

    # Gather uncert cells (cells where parent Shor fails).
    uncert, _batch = _gather_uncert_cells(d, S, c_target,
                                            n_target=n_cells,
                                            verbose=verbose)
    n_actual = len(uncert)
    if n_actual == 0:
        print("  no uncert cells found", flush=True)
        return {'n_cells': 0}

    # Run at each max_depth from 1..max_depth (per-cell, capture all stats)
    results_by_depth = {}
    for k in range(1, max_depth + 1):
        depth_results = []
        t0 = time.time()
        for ci, c in uncert:
            t1 = time.time()
            ok, stats = cell_cert_bnb_shor(
                c, S, d, c_target, max_depth=k,
                A_list=A_list, ell_list=ell_list,
                collect_stats=True,
            )
            dt = time.time() - t1
            depth_results.append({
                'ci': ci,
                'ok': ok,
                'time_s': dt,
                'shor_calls': stats['shor_calls'],
                'splits': stats['splits'],
                'depth_required': stats['depth_required'],
            })
        elapsed = time.time() - t0
        n_cert = sum(1 for r in depth_results if r['ok'])
        cert_rate = n_cert / n_actual
        avg_time = np.mean([r['time_s'] for r in depth_results])
        med_time = np.median([r['time_s'] for r in depth_results])
        max_time = max(r['time_s'] for r in depth_results)
        avg_shor = np.mean([r['shor_calls'] for r in depth_results])
        results_by_depth[k] = {
            'n_cert': n_cert,
            'cert_rate': float(cert_rate),
            'avg_time_s': float(avg_time),
            'med_time_s': float(med_time),
            'max_time_s': float(max_time),
            'avg_shor_calls': float(avg_shor),
            'elapsed_s': float(elapsed),
            'rows': depth_results,
        }
        if verbose:
            print(f"  depth={k}: cert {n_cert}/{n_actual} "
                  f"({100*cert_rate:.1f}%); "
                  f"avg time {1000*avg_time:.1f}ms, "
                  f"med {1000*med_time:.1f}ms, "
                  f"max {1000*max_time:.1f}ms; "
                  f"avg shor calls {avg_shor:.1f}; "
                  f"elapsed {elapsed:.1f}s", flush=True)

    out = {
        'd': d, 'S': S, 'c_target': c_target,
        'n_cells_target': n_cells,
        'n_cells_actual': n_actual,
        'max_depth': max_depth,
        'by_depth': {str(k): {kk: vv for kk, vv in v.items() if kk != 'rows'}
                       for k, v in results_by_depth.items()},
    }
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(out, f, indent=2, default=lambda o: int(o)
                      if isinstance(o, np.integer) else float(o))
        if verbose:
            print(f"  Wrote {output_path}", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--S', type=int, default=200)
    ap.add_argument('--c', type=float, default=1.281)
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--max_depth', type=int, default=8)
    ap.add_argument('--out', default='_bnb_shor_cert_results.json')
    args = ap.parse_args()
    bench(d=args.d, S=args.S, c_target=args.c, n_cells=args.n,
            max_depth=args.max_depth, output_path=args.out)


if __name__ == '__main__':
    main()
