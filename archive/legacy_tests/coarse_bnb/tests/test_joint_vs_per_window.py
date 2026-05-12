"""Compare joint multi-window cell cert vs per-window QP.

For each test composition:
  - per-window QP: cert if margin_W > QP_W for SOME W
  - joint cert:    cert if min_δ max_W TV_W(c+δ) >= c_target

The joint cert is provably TIGHTER. Quantify the gain.

Then run on a sweep of compositions where per-window FAILS to see how many
joint can RECOVER. Each recovered composition is a cell where the cascade
v2/QP marks UNSOUND but joint certifies as sound.
"""
from __future__ import annotations

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cloninger-steinerberger', 'cpu'))

from qp_bound import build_window_matrix, grad_for_window, qp_bound_vertex
from qp_bound_joint import joint_cell_cert_for_composition


def cell_var_v2(grad, S):
    d = grad.shape[0]
    g = np.sort(grad)
    h = 1.0 / (2.0 * S)
    total = 0.0
    for k in range(d // 2):
        total += g[d - 1 - k] - g[k]
    return h * total


def quad_corr_v2(d, S, ell, s):
    N_W, M_W = 0, 0
    for k in range(s, s + ell - 1):
        n_k = min(k + 1, d, 2 * d - 1 - k)
        if n_k <= 0:
            continue
        N_W += n_k
        if k % 2 == 0 and (k // 2) < d:
            M_W += 1
    cross_W = N_W - M_W
    pair_bound = min(cross_W, d * d - N_W)
    if pair_bound <= 0:
        return 0.0
    return (2.0 * d / ell) * pair_bound / (4.0 * S * S)


def per_window_cert(c_int, d, S, c_target):
    """Per-window cert: max over windows of (margin - bound)."""
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        ci = int(c_int[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = int(c_int[j])
                if cj != 0:
                    conv[i + j] += 2 * ci * cj

    max_ell = 2 * d
    S_sq = float(S) * float(S)
    d_d = float(d)
    h = 1.0 / (2.0 * S)
    best_net_qp = -1e30
    best_net_v2 = -1e30
    n_pruning = 0

    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        ws = 0
        for k in range(n_cv):
            ws += int(conv[k])
        ell_f = float(ell)
        thr = c_target * ell_f * S_sq / (2.0 * d_d) - 1e-9
        scale_qp = 2.0 * d_d / ell_f

        for s_lo in range(n_windows):
            if s_lo > 0:
                ws += int(conv[s_lo + n_cv - 1]) - int(conv[s_lo - 1])
            if ws > thr:
                n_pruning += 1
                tv = float(ws) * 2.0 * d_d / (S_sq * ell_f)
                margin = tv - c_target

                A = build_window_matrix(d, ell, s_lo)
                grad = grad_for_window(c_int.astype(np.int64), A, S, d, ell)
                qp = qp_bound_vertex(grad, A, scale_qp, h, d)
                cv = cell_var_v2(grad, S)
                qc = quad_corr_v2(d, S, ell, s_lo)
                tri = cv + qc
                net_qp = margin - qp
                net_v2 = margin - tri
                if net_qp > best_net_qp:
                    best_net_qp = net_qp
                if net_v2 > best_net_v2:
                    best_net_v2 = net_v2

    return best_net_qp, best_net_v2, n_pruning


def random_composition(d, S, rng):
    bars = rng.choice(S + d - 1, size=d - 1, replace=False)
    bars = np.sort(bars)
    parts = np.empty(d, dtype=np.int64)
    prev = -1
    for i in range(d - 1):
        parts[i] = bars[i] - prev - 1
        prev = bars[i]
    parts[-1] = S + d - 2 - prev
    assert parts.sum() == S
    return parts


def main():
    print("=" * 78)
    print("Joint multi-window cell cert vs per-window QP / v2 triangle")
    print("=" * 78)

    cases = [
        # (d, S, c_target)
        (4, 10, 1.28),
        (4, 10, 1.30),
        (6, 12, 1.28),
        (6, 15, 1.28),
        (8, 12, 1.28),
        (8, 12, 1.29),
    ]

    rng = np.random.default_rng(0)
    for (d, S, c_target) in cases:
        print(f"\n--- d={d}, S={S}, c_target={c_target} ---")
        n_total = 0
        n_pruned = 0
        n_v2_cert = 0
        n_qp_cert = 0
        n_joint_cert = 0
        n_recover_joint_over_qp = 0
        worst_qp_when_joint_recovers = (None, None, None)

        # Sample 200 random compositions
        for trial in range(200):
            c = random_composition(d, S, rng)
            n_total += 1

            net_qp, net_v2, n_pw = per_window_cert(c, d, S, c_target)
            if n_pw == 0:
                continue
            n_pruned += 1
            if net_v2 > 0:
                n_v2_cert += 1
            if net_qp > 0:
                n_qp_cert += 1

            cert_joint, _ = joint_cell_cert_for_composition(
                c.astype(np.int64), S, d, c_target)
            joint_passed = cert_joint >= c_target - 1e-9
            if joint_passed:
                n_joint_cert += 1
            if joint_passed and net_qp <= 0:
                n_recover_joint_over_qp += 1
                if worst_qp_when_joint_recovers[0] is None or \
                   net_qp < worst_qp_when_joint_recovers[1]:
                    worst_qp_when_joint_recovers = (
                        tuple(int(x) for x in c), net_qp, cert_joint)

        print(f"  Total: {n_total}, Pruned at grid: {n_pruned}")
        print(f"  v2 cert:    {n_v2_cert} / {n_pruned}")
        print(f"  per-W QP:   {n_qp_cert} / {n_pruned}")
        print(f"  Joint cert: {n_joint_cert} / {n_pruned}")
        print(f"  Joint recovers (vs QP): {n_recover_joint_over_qp}")
        if worst_qp_when_joint_recovers[0]:
            ex, qp_n, j = worst_qp_when_joint_recovers
            print(f"    Example: c={ex} qp_net={qp_n:.5f} joint_cert={j:.5f}")


if __name__ == "__main__":
    main()
