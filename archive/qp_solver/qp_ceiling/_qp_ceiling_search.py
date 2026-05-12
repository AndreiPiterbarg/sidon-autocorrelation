"""Binary-search the c_target ceiling per method (triangle / qp-single / qp-joint).

For (d, S), find the largest c_target in [c_lo, c_hi] s.t. EVERY pruned
composition is certified by the method. We early-stop on first uncertified.
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from qp_bound import (build_window_matrix, grad_for_window,
                      qp_bound_for_composition)
from qp_bound_joint import joint_cell_cert_for_composition

# Reuse helpers from the bench file
sys.path.insert(0, _dir)
from _qp_joint_bench import (find_pruning_windows, cell_var_triangle,
                              enumerate_compositions)


def all_pruned_cert(d, S, c_target, method):
    """Return (ok, n_total, n_pruned, n_certified, first_failure).

    method in {'tri', 'qp', 'joint'}. Stops on first uncertified.
    ok = True iff every pruned composition is certified.
    """
    n_total = 0
    n_pruned = 0
    n_certified = 0
    first_fail = None
    S_sq = float(S) * float(S)

    for c in enumerate_compositions(d, S):
        n_total += 1
        c_arr = np.array(c, dtype=np.int32)
        pwins = find_pruning_windows(c_arr, S, d, c_target)
        if not pwins:
            continue
        n_pruned += 1

        ok = False
        if method == 'joint':
            cert, _ = joint_cell_cert_for_composition(
                c_arr, int(S), int(d), float(c_target))
            ok = cert >= c_target
        elif method == 'tri':
            for (ell, s_lo, ws) in pwins:
                tv = float(ws) * 2.0 * d / (S_sq * ell)
                margin = tv - c_target
                B_tri = cell_var_triangle(c_arr, S, d, ell, s_lo)
                if margin > B_tri:
                    ok = True
                    break
        elif method == 'qp':
            c_f = c_arr.astype(np.float64)
            for (ell, s_lo, ws) in pwins:
                tv = float(ws) * 2.0 * d / (S_sq * ell)
                margin = tv - c_target
                B_qp = qp_bound_for_composition(c_f, S, d, ell, s_lo)
                if margin > B_qp:
                    ok = True
                    break
        else:
            raise ValueError(method)

        if ok:
            n_certified += 1
        else:
            first_fail = list(c)
            return False, n_total, n_pruned, n_certified, first_fail

    return True, n_total, n_pruned, n_certified, None


def binary_search(d, S, method, c_lo=1.20, c_hi=1.50, tol=0.001, verbose=True):
    """Find max c_target where EVERY pruned cell certifies."""
    t0 = time.time()
    if verbose:
        print(f"  [{method}] d={d} S={S}: searching in [{c_lo}, {c_hi}]")
    # First check feasibility at c_lo
    ok_lo, *_ = all_pruned_cert(d, S, c_lo, method)
    if not ok_lo:
        if verbose:
            print(f"  [{method}] d={d} S={S}: NOT feasible even at c_lo={c_lo}")
        return c_lo - tol, time.time() - t0
    # And infeasibility at c_hi
    ok_hi, *_ = all_pruned_cert(d, S, c_hi, method)
    if ok_hi:
        if verbose:
            print(f"  [{method}] d={d} S={S}: feasible at c_hi={c_hi} (raise hi)")
        return c_hi, time.time() - t0
    # Binary search
    iters = 0
    while c_hi - c_lo > tol:
        c_mid = 0.5 * (c_lo + c_hi)
        ok, n_total, n_pruned, n_cert, fail = all_pruned_cert(d, S, c_mid, method)
        iters += 1
        if verbose:
            tag = 'OK' if ok else f'FAIL@{fail}'
            print(f"    iter {iters}: c={c_mid:.4f} pruned={n_pruned} cert={n_cert} {tag}")
        if ok:
            c_lo = c_mid
        else:
            c_hi = c_mid
    return c_lo, time.time() - t0


def run_config(d, S, c_lo=1.20, c_hi=1.50, tol=0.001, verbose=False):
    print(f"\n=== d={d}, S={S} ===")
    res = {}
    for method in ('tri', 'qp', 'joint'):
        ceiling, sec = binary_search(d, S, method, c_lo, c_hi, tol, verbose=verbose)
        res[method] = ceiling
        res[method + '_sec'] = sec
        print(f"  -> {method}: ceiling = {ceiling:.4f}  ({sec:.1f}s)")
    return res


if __name__ == '__main__':
    # Warm up numba JIT for each d we'll touch
    for d_warm, S_warm in [(4, 20), (6, 12), (8, 10)]:
        c0 = np.array([S_warm // d_warm] * d_warm, dtype=np.int32)
        c0[0] += S_warm - int(c0.sum())
        _ = joint_cell_cert_for_composition(c0, S_warm, d_warm, 1.20)
        _ = qp_bound_for_composition(c0.astype(np.float64), S_warm, d_warm, 4, 2)
        _ = build_window_matrix(d_warm, 4, 2)

    # Pre-determined reasonable bounds for binary search.
    # Triangle ceiling typically <1.30 at small (d,S); joint can hit higher.
    configs = [
        (4, 20),
        (4, 30),
        (6, 12),
        (6, 18),
        (8, 10),
    ]
    all_results = {}
    for d, S in configs:
        all_results[f'd{d}_S{S}'] = run_config(d, S, c_lo=0.50, c_hi=1.55, tol=0.001)

    print("\n" + "="*78)
    print(f"{'d':>3} {'S':>4} {'tri':>10} {'qp':>10} {'joint':>10} {'joint-tri':>12}")
    print("-"*78)
    for k, v in all_results.items():
        d_s = k.replace('d','').replace('S','').split('_')
        d, S = int(d_s[0]), int(d_s[1])
        print(f"{d:>3} {S:>4} {v['tri']:>10.4f} {v['qp']:>10.4f} "
              f"{v['joint']:>10.4f} {v['joint']-v['tri']:>+12.4f}")

    out = os.path.join(_dir, '_qp_ceiling_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
