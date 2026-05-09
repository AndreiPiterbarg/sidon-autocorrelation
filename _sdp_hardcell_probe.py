"""Probe: Lasserre order-2 SDP on hard cells (joint QP fails to certify).

Definition of "hard cell": composition c that is pruned at the grid pt
(some window has TV(c) > c_target) but joint multi-window QP cannot certify
the surrounding cell box.

For each hard cell, run lasserre_box_lb_float on the box [c/S - h, c/S + h]^d
where h = 1/(2S), and report:
  - SDP lower bound u*
  - certified iff u* >= c_target
  - timing
  - sanity: SDP LB >= joint QP cert value (must hold mathematically)
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from qp_bound_joint import joint_cell_cert_for_composition
from interval_bnb.windows import build_windows
from interval_bnb.lasserre_cert import lasserre_box_lb_float


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
                out.append((ell, s_lo))
    return out


def enumerate_compositions(d, S):
    if d == 1:
        yield [S]; return
    for v in range(S + 1):
        for rest in enumerate_compositions(d - 1, S - v):
            yield [v] + rest


def find_hard_cells(d, S, c_target):
    """Return list of hard cells: pruned at grid but joint QP cert < c_target."""
    hard = []
    n_total = 0
    n_pruned = 0
    n_joint_cert = 0
    for c in enumerate_compositions(d, S):
        n_total += 1
        c_arr = np.array(c, dtype=np.int32)
        pwins = find_pruning_windows(c_arr, S, d, c_target)
        if not pwins:
            continue
        n_pruned += 1
        cert_joint, n_pw = joint_cell_cert_for_composition(c_arr, S, d, c_target)
        if cert_joint >= c_target:
            n_joint_cert += 1
        else:
            hard.append({'c': c, 'cert_joint': float(cert_joint),
                          'n_pwin': int(n_pw)})
    return hard, n_total, n_pruned, n_joint_cert


def sdp_certify_cell(c_int, S, d, windows, c_target, *, order=2, solver="CLARABEL"):
    """Run Lasserre SDP on the box around c/S. Returns (LB, time_s)."""
    h = 1.0 / (2.0 * S)
    mu_star = np.asarray(c_int, dtype=np.float64) / float(S)
    lo = np.maximum(mu_star - h, 0.0)
    hi = mu_star + h
    t0 = time.time()
    lb = lasserre_box_lb_float(lo, hi, windows, d, order=order, solver=solver)
    dt = time.time() - t0
    return lb, dt


if __name__ == '__main__':
    d, S, c_target = 6, 15, 1.25
    print(f"=== Finding hard cells at d={d}, S={S}, c_target={c_target} ===")
    t0 = time.time()
    hard, n_total, n_pruned, n_joint_cert = find_hard_cells(d, S, c_target)
    print(f"Total compositions: {n_total:,}, pruned at grid: {n_pruned:,}, "
          f"joint-QP-certified: {n_joint_cert:,}, HARD: {len(hard):,}, "
          f"({time.time()-t0:.1f}s)")

    # Sample 8 hard cells with most pruning windows (likely most informative)
    hard_sorted = sorted(hard, key=lambda x: -x['n_pwin'])
    sample = hard_sorted[:8]
    print(f"\n=== SDP on {len(sample)} hard cells ===")

    windows = build_windows(d)
    print(f"Building windows: {len(windows)} windows for d={d}")

    results = []
    for idx, h in enumerate(sample):
        c = h['c']
        cert_joint = h['cert_joint']
        gap_joint = c_target - cert_joint
        try:
            lb, dt = sdp_certify_cell(c, S, d, windows, c_target)
        except Exception as e:
            print(f"  [{idx}] c={c} EXCEPTION: {type(e).__name__}: {e}")
            continue
        sdp_certifies = lb >= c_target
        sane = lb >= cert_joint - 1e-6  # SDP must be >= joint QP
        gap_sdp = c_target - lb
        print(f"  [{idx}] c={c}: joint_cert={cert_joint:.6f} (gap={gap_joint:+.6f}), "
              f"SDP_LB={lb:.6f} (gap={gap_sdp:+.6f}), "
              f"SDP_certs={sdp_certifies}, SDP>=joint? {sane}, t={dt:.2f}s")
        results.append({'c': c, 'cert_joint': cert_joint, 'sdp_lb': lb,
                        'sdp_certifies': sdp_certifies, 'sane': sane, 'time_s': dt})

    if results:
        wall_total = sum(r['time_s'] for r in results)
        per_cell = wall_total / len(results)
        n_cert = sum(1 for r in results if r['sdp_certifies'])
        n_sane = sum(1 for r in results if r['sane'])
        print(f"\nSummary: {n_cert}/{len(results)} SDP-certified; "
              f"{n_sane}/{len(results)} sane (SDP>=joint); "
              f"per-cell mean time {per_cell:.2f}s")
        # Scaling extrapolation
        n_hard_d6 = len(hard)
        n_hard_d8_est = 5000  # per task brief
        print(f"\nScaling extrapolation (mean per-cell {per_cell:.2f}s):")
        print(f"  d=6, S=15, c=1.25: {n_hard_d6} hard -> "
              f"{n_hard_d6 * per_cell:.0f}s = {n_hard_d6 * per_cell/60:.1f} min")
        print(f"  d=8, hard~{n_hard_d8_est}: {n_hard_d8_est * per_cell:.0f}s = "
              f"{n_hard_d8_est * per_cell/60:.1f} min")
        print(f"  (note: d=8 SDP is LARGER -- per-cell will be SLOWER than d=6)")

    out = {'d': d, 'S': S, 'c_target': c_target,
           'n_total': n_total, 'n_pruned': n_pruned,
           'n_joint_cert': n_joint_cert, 'n_hard': len(hard),
           'sample_results': results}
    out_path = os.path.join(_dir, '_sdp_hardcell_probe_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
