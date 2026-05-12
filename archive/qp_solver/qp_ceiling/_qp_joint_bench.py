"""Bench: triangle vs single-window QP vs joint multi-window QP.

For each pruned composition (TV > c_target at grid pt), check which methods
certify the cell. Method ranking (mathematically proven, larger ⇒ certifies more):

  joint_QP ≥ max_W per_window_QP ≥ max_W triangle

Empirical test: how often does the chain strictly improve?
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from qp_bound import (build_window_matrix, grad_for_window,
                      qp_bound_vertex, qp_bound_for_composition)
from qp_bound_joint import (joint_cell_cert_for_composition,
                             joint_bound_vertex)


def find_pruning_windows(c_int, S, d, c_target):
    """Return list of (ell, s_lo) where TV_W(c) > c_target at grid pt."""
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


def cell_var_triangle(c_int, S, d, ell, s_lo):
    """v2-style triangle bound for one window."""
    h = 1.0 / (2.0 * S)
    A_W = build_window_matrix(d, ell, s_lo)
    grad = grad_for_window(c_int, A_W, S, d, ell)
    grad_sorted = np.sort(grad)
    cell_var = sum(grad_sorted[d-1-k] - grad_sorted[k] for k in range(d // 2)) / (2.0 * S)
    n_pairs = int(np.sum(A_W))
    R_bound = (2.0 * d / ell) * n_pairs / (4.0 * S * S)
    return cell_var + R_bound


def test_one_composition(c_int, S, d, c_target, verbose=False):
    """Test all three certification methods. Returns dict of results."""
    # Find pruning windows (where TV > c_target at grid pt)
    pwins = find_pruning_windows(c_int, S, d, c_target)
    if not pwins:
        return None  # not pruned at grid point — n/a

    results = {
        'pruning_windows': len(pwins),
        'triangle_certified_by': None,  # window (ell, s_lo) if any
        'qp_certified_by': None,
        'joint_certified': False,
        'best_net_triangle': -np.inf,
        'best_net_qp': -np.inf,
        'cert_joint': None,
    }

    S_sq = float(S) * float(S)

    # Triangle and per-window QP
    for (ell, s_lo, ws) in pwins:
        tv = float(ws) * 2.0 * d / (S_sq * ell)
        margin = tv - c_target

        B_triangle = cell_var_triangle(c_int, S, d, ell, s_lo)
        net_tri = margin - B_triangle
        if net_tri > results['best_net_triangle']:
            results['best_net_triangle'] = net_tri
        if net_tri > 0 and results['triangle_certified_by'] is None:
            results['triangle_certified_by'] = (ell, s_lo)

        B_qp = qp_bound_for_composition(np.asarray(c_int, dtype=np.float64), S, d, ell, s_lo)
        net_qp = margin - B_qp
        if net_qp > results['best_net_qp']:
            results['best_net_qp'] = net_qp
        if net_qp > 0 and results['qp_certified_by'] is None:
            results['qp_certified_by'] = (ell, s_lo)

    # Joint multi-window
    cert_joint, n_pw = joint_cell_cert_for_composition(
        np.asarray(c_int, dtype=np.int32), int(S), int(d), float(c_target))
    results['cert_joint'] = cert_joint
    results['joint_certified'] = cert_joint >= c_target

    if verbose:
        print(f"  c={list(c_int)}: tri_net={results['best_net_triangle']:.5f}, "
              f"qp_net={results['best_net_qp']:.5f}, "
              f"joint={cert_joint:.5f} (target={c_target})")

    return results


def enumerate_compositions(d, S):
    """Brute-force enumerate all integer compositions of S into d non-neg parts."""
    if d == 1:
        yield [S]
        return
    for v in range(S + 1):
        for rest in enumerate_compositions(d - 1, S - v):
            yield [v] + rest


def run_bench(d, S, c_target, max_compositions=None):
    print(f"\n=== d={d}, S={S}, c_target={c_target} ===")
    n_pruned = 0
    n_tri_cert = 0
    n_qp_cert = 0
    n_joint_cert = 0
    n_tri_NOT_qp = 0   # tri certs but qp doesn't (impossible if math right)
    n_qp_NOT_joint = 0 # qp certs but joint doesn't (impossible if math right)
    n_joint_NOT_qp = 0 # joint certs but qp doesn't (the WIN — cells joint can certify but QP cannot)
    n_qp_NOT_tri = 0   # qp certs but tri doesn't

    t0 = time.time()
    n_total = 0
    for c in enumerate_compositions(d, S):
        n_total += 1
        if max_compositions is not None and n_total > max_compositions:
            break
        c_arr = np.array(c, dtype=np.int32)
        # Symmetry: skip non-palindromic to avoid duplicate work?
        # For coarse bench, enumerate all to capture full picture.
        r = test_one_composition(c_arr, S, d, c_target)
        if r is None:
            continue  # not pruned at grid pt
        n_pruned += 1
        tri = r['triangle_certified_by'] is not None
        qp = r['qp_certified_by'] is not None
        joint = r['joint_certified']
        if tri: n_tri_cert += 1
        if qp: n_qp_cert += 1
        if joint: n_joint_cert += 1
        if tri and not qp: n_tri_NOT_qp += 1   # SOUNDNESS BUG if >0
        if qp and not joint:
            n_qp_NOT_joint += 1  # SOUNDNESS BUG if >0
            if n_qp_NOT_joint <= 3:
                print(f"  [BUG] c={list(c)}: qp cert (best_net={r['best_net_qp']:.6e}), "
                      f"joint cert_value={r['cert_joint']:.6f} < c_target={c_target}")
        if joint and not qp: n_joint_NOT_qp += 1  # the WIN
        if qp and not tri: n_qp_NOT_tri += 1

    elapsed = time.time() - t0
    print(f"  Total compositions: {n_total:,}, pruned at grid: {n_pruned:,}, time: {elapsed:.1f}s")
    print(f"  Certified by triangle: {n_tri_cert:,}")
    print(f"  Certified by QP single: {n_qp_cert:,}  (+{n_qp_NOT_tri:,} over triangle)")
    print(f"  Certified by QP joint:  {n_joint_cert:,}  (+{n_joint_NOT_qp:,} over QP single)")
    if n_tri_NOT_qp > 0:
        print(f"  *** SOUNDNESS BUG: {n_tri_NOT_qp} compositions tri-certified but not qp ***")
    if n_qp_NOT_joint > 0:
        print(f"  *** SOUNDNESS BUG: {n_qp_NOT_joint} compositions qp-certified but not joint ***")
    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_total': n_total, 'n_pruned': n_pruned,
        'n_tri_cert': n_tri_cert, 'n_qp_cert': n_qp_cert, 'n_joint_cert': n_joint_cert,
        'n_qp_extra_over_tri': n_qp_NOT_tri,
        'n_joint_extra_over_qp': n_joint_NOT_qp,
        'soundness_violations_tri_NOT_qp': n_tri_NOT_qp,
        'soundness_violations_qp_NOT_joint': n_qp_NOT_joint,
        'wall_sec': elapsed,
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    # Small d, vary S and c_target to find regime where bound matters
    if args.quick:
        configs = [
            (4, 20, 1.20),
            (4, 20, 1.25),
            (4, 30, 1.25),
            (6, 15, 1.25),
        ]
    else:
        configs = [
            (4, 20, 1.20),
            (4, 30, 1.20),
            (4, 30, 1.25),
            (4, 40, 1.25),
            (6, 15, 1.20),
            (6, 20, 1.25),
            (6, 30, 1.25),
            (8, 12, 1.20),
            (8, 15, 1.20),
        ]
    results = []
    for d, S, ct in configs:
        results.append(run_bench(d, S, ct))
    print("\n" + "="*70 + "\nSummary:")
    for r in results:
        print(f"  d={r['d']}, S={r['S']}, c={r['c_target']}: "
              f"tri={r['n_tri_cert']:,} -> qp={r['n_qp_cert']:,} "
              f"(+{r['n_qp_extra_over_tri']:,}) -> joint={r['n_joint_cert']:,} "
              f"(+{r['n_joint_extra_over_qp']:,})  "
              f"sound: {r['soundness_violations_tri_NOT_qp']}, {r['soundness_violations_qp_NOT_joint']}")
    out = os.path.join(_dir, '_qp_joint_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
