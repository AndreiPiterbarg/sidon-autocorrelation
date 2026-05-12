"""Bench: joint multi-window cert with ALL windows vs only PRUNING windows.

Mathematical opportunity (over qp_bound_joint.py's joint_cell_cert_for_composition):

    cert_joint_PW(c)  = min_δ max_{W ∈ pruning_W} TV_W(c+δ)
    cert_joint_ALL(c) = min_δ max_{W ∈ all_W}      TV_W(c+δ)

Since pruning_W ⊆ all_W, we have cert_joint_ALL ≥ cert_joint_PW pointwise (in δ
the max is over a larger set, then we min). Hence the cell is more often
certified by joint-all than by joint-pruning-only:

    {cells certified by joint_PW}  ⊆  {cells certified by joint_ALL}

This bench:
  1. Defines `joint_cell_cert_all_windows(c_int, S, d, c_target)`, structurally
     identical to qp_bound_joint.joint_cell_cert_for_composition but iterating
     over ALL (ell, s_lo) windows in the inner-max.
  2. Reuses joint_bound_vertex by passing the all-windows array.
  3. Compares per-window QP, joint-PW, and joint-ALL on configs from the task.
  4. Sanity-checks: any cell certified by joint-PW must also be certified by
     joint-ALL (strict inclusion). A failure is a soundness bug.
"""
from __future__ import annotations

import os, sys, time, json
import numpy as np
import numba
from numba import njit

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from qp_bound import qp_bound_for_composition  # noqa: E402
from qp_bound_joint import (  # noqa: E402
    joint_bound_vertex,
    joint_cell_cert_for_composition,
)


# -----------------------------------------------------------------------------
# Core: joint cert with ALL windows. The ONLY change vs joint_pruning is that
# we enumerate every (ell, s_lo) with ell ≥ 2 and s_lo ∈ [0, conv_len-ell+2)
# instead of filtering to pruning windows.
# -----------------------------------------------------------------------------

@njit(cache=True)
def _all_windows_arrays(d):
    """Return (ell_arr, s_arr, n_windows) covering every (ell, s_lo) with
    ell ≥ 2 and s_lo ∈ [0, conv_len - ell + 2).

    For ell, n_cv = ell-1, so s_lo ranges over [0, conv_len - n_cv + 1)
    = [0, conv_len - ell + 2).  ell goes 2..max_ell where max_ell = 2*d.
    """
    conv_len = 2 * d - 1
    max_ell = 2 * d

    # First pass: count
    n_total = 0
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows > 0:
            n_total += n_windows

    ell_arr = np.empty(n_total, dtype=np.int32)
    s_arr = np.empty(n_total, dtype=np.int32)
    idx = 0
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            ell_arr[idx] = ell
            s_arr[idx] = s_lo
            idx += 1
    return ell_arr, s_arr, n_total


@njit(cache=True)
def joint_cell_cert_all_windows(c_int, S, d, c_target):
    """Joint min-max cell cert using ALL (ell, s_lo) windows.

    Mirrors qp_bound_joint.joint_cell_cert_for_composition exactly EXCEPT:
    the windows array passed to joint_bound_vertex enumerates every legal
    (ell, s_lo) pair instead of only the pruning windows.

    Returns:
      cert_value: min over δ of max over ALL W of TV_W(mu* + δ).
                  Cell is certified iff cert_value >= c_target.
      n_all_windows: total number of windows enumerated (purely informational).
    """
    ell_arr, s_arr, n_all = _all_windows_arrays(d)
    cert = joint_bound_vertex(c_int, S, d, c_target,
                              ell_arr, s_arr, n_all)
    return cert, n_all


# -----------------------------------------------------------------------------
# Bench machinery — reuses pieces from _qp_joint_bench.py at the file-level.
# -----------------------------------------------------------------------------

def find_pruning_windows_py(c_int, S, d, c_target):
    """Pure-Python: list of (ell, s_lo, ws) where TV_W(c) > c_target at grid pt."""
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        ci = int(c_int[i])
        if ci != 0:
            conv[2*i] += ci * ci
            for j in range(i+1, d):
                cj = int(c_int[j])
                if cj != 0:
                    conv[i+j] += 2 * ci * cj
    out = []
    eps = 1e-9
    max_ell = 2 * d
    S_sq = float(S) * float(S)
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        thr = c_target * float(ell) * S_sq / (2.0 * d) - eps
        for s_lo in range(n_windows):
            ws = sum(int(conv[k]) for k in range(s_lo, s_lo + n_cv))
            if ws > thr:
                out.append((ell, s_lo, ws))
    return out


def enumerate_compositions(d, S):
    if d == 1:
        yield [S]
        return
    for v in range(S + 1):
        for rest in enumerate_compositions(d - 1, S - v):
            yield [v] + rest


def test_one(c_arr, S, d, c_target):
    """Per-composition: per-window QP cert?, joint-PW cert?, joint-ALL cert?

    Returns dict or None if no pruning windows (cell trivially fine at grid pt).
    """
    pwins = find_pruning_windows_py(c_arr, S, d, c_target)
    if not pwins:
        return None

    S_sq = float(S) * float(S)
    qp_cert = False
    best_net_qp = -np.inf
    for (ell, s_lo, ws) in pwins:
        tv = float(ws) * 2.0 * d / (S_sq * ell)
        margin = tv - c_target
        B_qp = qp_bound_for_composition(np.asarray(c_arr, dtype=np.float64),
                                        S, d, ell, s_lo)
        net = margin - B_qp
        if net > best_net_qp:
            best_net_qp = net
        if net > 0:
            qp_cert = True

    cert_pw, _ = joint_cell_cert_for_composition(
        np.asarray(c_arr, dtype=np.int32), int(S), int(d), float(c_target))
    pw_cert = cert_pw >= c_target

    cert_all, n_all = joint_cell_cert_all_windows(
        np.asarray(c_arr, dtype=np.int32), int(S), int(d), float(c_target))
    all_cert = cert_all >= c_target

    return {
        'qp_cert': qp_cert,
        'pw_cert': pw_cert,
        'all_cert': all_cert,
        'cert_pw': cert_pw,
        'cert_all': cert_all,
        'best_net_qp': best_net_qp,
        'n_pruning': len(pwins),
        'n_all': n_all,
    }


def run_bench(d, S, c_target, max_compositions=None):
    print(f"\n=== d={d}, S={S}, c_target={c_target} ===")
    n_total = 0
    n_pruned = 0
    n_qp_cert = 0
    n_pw_cert = 0
    n_all_cert = 0
    n_pw_NOT_all = 0      # joint-PW certs but joint-ALL doesn't (SOUNDNESS BUG)
    n_all_NOT_pw = 0      # joint-ALL certs but joint-PW doesn't (THE WIN)
    n_qp_NOT_pw = 0
    n_qp_NOT_all = 0
    bug_examples = []

    t_pw_total = 0.0
    t_all_total = 0.0

    t0 = time.time()
    for c in enumerate_compositions(d, S):
        n_total += 1
        if max_compositions is not None and n_total > max_compositions:
            break
        c_arr = np.array(c, dtype=np.int32)

        # Time joint-PW vs joint-ALL specifically
        pwins = find_pruning_windows_py(c_arr, S, d, c_target)
        if not pwins:
            continue
        n_pruned += 1

        # Per-window QP
        qp_cert = False
        S_sq = float(S) * float(S)
        for (ell, s_lo, ws) in pwins:
            tv = float(ws) * 2.0 * d / (S_sq * ell)
            margin = tv - c_target
            B_qp = qp_bound_for_composition(np.asarray(c_arr, dtype=np.float64),
                                            S, d, ell, s_lo)
            if margin - B_qp > 0:
                qp_cert = True
                break  # one window suffices for per-window QP

        # Joint-PW
        t1 = time.perf_counter()
        cert_pw, _ = joint_cell_cert_for_composition(
            np.asarray(c_arr, dtype=np.int32), int(S), int(d), float(c_target))
        t_pw_total += time.perf_counter() - t1
        pw_cert = cert_pw >= c_target

        # Joint-ALL
        t2 = time.perf_counter()
        cert_all, _n_all = joint_cell_cert_all_windows(
            np.asarray(c_arr, dtype=np.int32), int(S), int(d), float(c_target))
        t_all_total += time.perf_counter() - t2
        all_cert = cert_all >= c_target

        if qp_cert: n_qp_cert += 1
        if pw_cert: n_pw_cert += 1
        if all_cert: n_all_cert += 1
        if pw_cert and not all_cert:
            n_pw_NOT_all += 1
            if len(bug_examples) < 3:
                bug_examples.append((list(c), cert_pw, cert_all))
        if all_cert and not pw_cert:
            n_all_NOT_pw += 1
        if qp_cert and not pw_cert: n_qp_NOT_pw += 1
        if qp_cert and not all_cert: n_qp_NOT_all += 1

    elapsed = time.time() - t0
    print(f"  Total compositions: {n_total:,}, pruned at grid: {n_pruned:,}, "
          f"wall: {elapsed:.2f}s")
    print(f"    joint-PW  total time: {t_pw_total:.3f}s")
    print(f"    joint-ALL total time: {t_all_total:.3f}s   "
          f"(ratio {t_all_total / max(t_pw_total, 1e-9):.2f}x)")
    print(f"  Certified by per-window QP : {n_qp_cert:,}")
    print(f"  Certified by joint-PW      : {n_pw_cert:,}  (+{max(0,n_pw_cert-n_qp_cert):,} over per-window QP)")
    print(f"  Certified by joint-ALL     : {n_all_cert:,}  (+{n_all_NOT_pw:,} over joint-PW)")
    if n_pw_NOT_all > 0:
        print(f"  *** SOUNDNESS BUG: {n_pw_NOT_all} compositions joint-PW-certified but NOT joint-ALL ***")
        for (c, cp, ca) in bug_examples:
            print(f"      c={c}: cert_pw={cp:.6f}, cert_all={ca:.6f}, c_target={c_target}")
    if n_qp_NOT_pw > 0:
        # qp_cert ⇒ joint_PW cert is normally implied (joint ≥ per-window).
        # Actually NO — per-window QP and joint-PW are different objects:
        # per-window QP uses the per-window QP_W upper bound on max_δ TV_W;
        # joint-PW solves the multi-window saddle. They CAN disagree per cell.
        # We log it but it's not necessarily a soundness violation of the
        # joint-PW vs joint-ALL comparison.
        pass

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_total': n_total, 'n_pruned': n_pruned,
        'n_qp_cert': n_qp_cert, 'n_pw_cert': n_pw_cert,
        'n_all_cert': n_all_cert,
        'n_all_extra_over_pw': n_all_NOT_pw,
        'n_pw_extra_over_qp': max(0, n_pw_cert - n_qp_cert),
        'soundness_violations_pw_NOT_all': n_pw_NOT_all,
        'wall_sec': elapsed,
        't_pw_sec': t_pw_total,
        't_all_sec': t_all_total,
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='Skip the second/third configs (smoke test).')
    args = ap.parse_args()

    # Configs from the task brief
    configs = [
        (4, 20, 1.20),
        (4, 30, 1.25),
        (6, 15, 1.25),
    ]
    if args.quick:
        configs = configs[:1]

    # Warm up numba JIT once (so the first config isn't dominated by compile)
    print("Warming up numba JIT...")
    c0 = np.array([1, 1, 1, 1], dtype=np.int32)
    _ = joint_cell_cert_all_windows(c0, 4, 4, 1.0)
    _ = joint_cell_cert_for_composition(c0, 4, 4, 1.0)
    print("  done.")

    results = []
    for (d, S, ct) in configs:
        results.append(run_bench(d, S, ct))

    print("\n" + "="*72)
    print("Summary:")
    for r in results:
        print(f"  d={r['d']}, S={r['S']}, c={r['c_target']}: "
              f"qp={r['n_qp_cert']:,} -> joint-PW={r['n_pw_cert']:,} "
              f"-> joint-ALL={r['n_all_cert']:,}  "
              f"(joint-ALL +{r['n_all_extra_over_pw']:,} over PW)  "
              f"sound: {r['soundness_violations_pw_NOT_all']}  "
              f"wall: {r['wall_sec']:.1f}s "
              f"(PW {r['t_pw_sec']:.2f}s / ALL {r['t_all_sec']:.2f}s)")

    out = os.path.join(_dir, '_qp_joint_all_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
