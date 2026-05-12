"""Two-phase rigorous proof of val(10) >= 1.2 via interval BnB + Lasserre SDP.

PHASE 1: BnB with cheap cascade (natural / autoconv / McCormick / epi-LP)
         dumps stuck boxes when in_flight stops draining.

PHASE 2: For every stuck box, run order-2 Lasserre at K=32 (selective PSD).
         The SDP audit (sdp_stuck_audit.json) showed 100% close on the
         sample at K=32 in ~1.8s/box.

If every box certifies in either phase, val(10) >= 1.2 is proved
(modulo the standard half-simplex orbit cover which the BnB driver
already handles).

Disables tiers that broke the prior run (centroid + boundary-split
returned 0 certs at d=10, and centroid was slowing workers to a halt
on init).
"""
import os
import sys
import time
import json
import glob
import numpy as np
from fractions import Fraction

# --- Phase-1 settings: cheap cascade only, dump stuck boxes on time-out
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '8'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '12'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
os.environ['INTERVAL_BNB_PC_DEPTH'] = '14'
os.environ['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '14'
# Tiers that contributed 0 certs in prior runs at d=10 — disable.
os.environ['INTERVAL_BNB_ANCHOR_DEPTH'] = '999'
os.environ['INTERVAL_BNB_CENTROID_DEPTH'] = '999'
os.environ['INTERVAL_BNB_BOUNDARY_SPLIT_DEPTH'] = '999'
# SDP escalation OFF in phase 1 — we'll do it as a cleaner phase-2 sweep
# that processes stuck boxes in batch with K=32.
os.environ['INTERVAL_BNB_SDP_DEPTH'] = '999'
# Dumping
os.environ['INTERVAL_BNB_DUMP_BOXES'] = 'proof_d10_phase1'
os.environ['INTERVAL_BNB_INSTANT_DUMP'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def phase1(time_budget_s):
    """Run BnB cheap cascade until time-budget; dump stuck boxes."""
    from interval_bnb.parallel import parallel_branch_and_bound
    print("#" * 72, flush=True)
    print(f"# PHASE 1: cheap-cascade BnB, budget {time_budget_s}s", flush=True)
    print("#" * 72, flush=True)
    t0 = time.time()
    r = parallel_branch_and_bound(
        d=10, target_c='1.2',
        workers=8,
        init_split_depth=12,
        donate_threshold_floor=2,
        time_budget_s=time_budget_s,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n[phase1] elapsed={elapsed:.1f}s  success={r['success']}  "
          f"coverage={100*r['coverage_fraction']:.5f}%  "
          f"in_flight_final={r['in_flight_final']}  "
          f"certs={r['total_leaves_certified']}", flush=True)
    if r['success']:
        return None  # already done
    # Collect dumped stuck boxes from master queue + worker stacks
    los_all, his_all = [], []
    for path in sorted(glob.glob('proof_d10_phase1_*.npz')):
        npz = np.load(path)
        los_all.append(npz['lo'])
        his_all.append(npz['hi'])
        print(f"[phase1] dump {path}: {len(npz['lo'])} boxes", flush=True)
    if not los_all:
        print("[phase1] no stuck-box dumps found!", flush=True)
        return np.zeros((0, 10)), np.zeros((0, 10))
    los = np.concatenate(los_all, axis=0)
    his = np.concatenate(his_all, axis=0)
    print(f"[phase1] total stuck boxes: {len(los)}", flush=True)
    return los, his


def phase2(los, his, time_limit_s_per_box=20.0):
    """For each stuck box, attempt order-2 SDP cert at K=32."""
    from interval_bnb.windows import build_windows
    from interval_bnb.box import Box
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast, bound_sdp_escalation_int_ge_fast,
    )
    print("#" * 72, flush=True)
    print(f"# PHASE 2: SDP K=32 on {len(los)} stuck boxes", flush=True)
    print("#" * 72, flush=True)
    windows = build_windows(10)
    cache = build_sdp_escalation_cache_fast(10, windows, target=1.2)

    n_cert = 0
    n_fail = 0
    fail_indices = []
    times = []
    t0 = time.time()
    for k in range(len(los)):
        lo = los[k].astype(np.float64)
        hi = his[k].astype(np.float64)
        B = Box(lo=lo, hi=hi)
        lo_int, hi_int = B.to_ints()
        ts = time.time()
        try:
            cert = bound_sdp_escalation_int_ge_fast(
                lo_int, hi_int, windows, 10,
                target_num=12, target_den=10,
                cache=cache, n_window_psd_cones=32,
                n_threads=1, time_limit_s=time_limit_s_per_box,
            )
        except Exception as e:
            print(f"  box {k}: EXCEPTION {type(e).__name__}: {e}", flush=True)
            cert = False
        dt = time.time() - ts
        times.append(dt)
        if cert:
            n_cert += 1
        else:
            n_fail += 1
            fail_indices.append(k)
        if (k + 1) % 25 == 0:
            print(f"[phase2] {k+1}/{len(los)}  cert={n_cert}  fail={n_fail}  "
                  f"avg_t={np.mean(times):.2f}s  elapsed={time.time()-t0:.1f}s",
                  flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"PHASE 2 SUMMARY: cert={n_cert}/{len(los)} fail={n_fail}", flush=True)
    print(f"  median_t={np.median(times):.2f}s  total_wall={time.time()-t0:.1f}s",
          flush=True)
    if fail_indices:
        print(f"  failed indices (first 10): {fail_indices[:10]}", flush=True)
    print("=" * 72, flush=True)
    return n_cert, n_fail, fail_indices


def main():
    res = phase1(time_budget_s=120.0)  # 2 min should be plenty for d=10
    if res is None:
        print("\n[VERDICT] BnB cheap cascade certified val(10) >= 1.2 alone.",
              flush=True)
        return
    los, his = res
    if len(los) == 0:
        print("\n[VERDICT] no stuck boxes captured — inconclusive.", flush=True)
        return
    n_cert, n_fail, fail_indices = phase2(los, his)
    print()
    if n_fail == 0:
        print("=" * 72, flush=True)
        print("VERDICT: val(10) >= 1.2 is PROVED.", flush=True)
        print(f"  {n_cert} stuck boxes all closed via order-2 Lasserre @ K=32.",
              flush=True)
        print("=" * 72, flush=True)
    else:
        print("=" * 72, flush=True)
        print(f"VERDICT: {n_fail}/{len(los)} stuck boxes UNCERTIFIED.", flush=True)
        print(f"  Need to escalate (K=999, order-3, or further BnB).", flush=True)
        print("=" * 72, flush=True)
    # Save result
    np.savez('proof_d10_t1p2_result.npz',
             phase2_cert=n_cert, phase2_fail=n_fail,
             fail_indices=np.array(fail_indices, dtype=np.int64),
             stuck_lo=los, stuck_hi=his)


if __name__ == "__main__":
    main()
