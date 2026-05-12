"""Parameterised two-phase rigorous proof of val(d) >= target.

PHASE 1: cheap-cascade BnB (natural / autoconv / McCormick / epi-LP).
         Disables tiers that contributed 0 certs at d=10 to avoid
         worker-startup stalls (anchor / centroid / boundary-split).
         Dumps stuck boxes on time-budget exhaust.

PHASE 2: K-cascade SDP cleanup, parallel via multiprocessing.Pool.
         K1 (default = ceil(0.085 * |W|)): expected ~88-94% close
         K2 (default = ceil(0.170 * |W|)): expected ~100% close
         K_full (= |W|):                   guaranteed full-PSD baseline

Each K phase only attempts boxes that the previous phase failed.

Usage:
    python prove_dN_tT.py --d 30 --target 1.2805 --workers 80 \\
        --phase1_budget_s 14400 --phase2_time_limit_s 600

Output: proof_d{d}_t{target}_result.json with cert counts per phase.
"""
import os
import sys
import time
import json
import glob
import argparse
import subprocess
import multiprocessing as mp
import numpy as np
from fractions import Fraction


def _phase1_run(d, target_str, workers, init_split_depth,
                time_budget_s, dump_prefix):
    """Runs Phase 1 in a child process (so MOSEK doesn't stay loaded).
    Returns dump_prefix path roots."""
    # Set ENV before importing parallel.
    os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '8'
    os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
    os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '12'
    os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
    os.environ['INTERVAL_BNB_PC_DEPTH'] = '14'
    os.environ['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '14'
    os.environ['INTERVAL_BNB_ANCHOR_DEPTH'] = '999'
    os.environ['INTERVAL_BNB_CENTROID_DEPTH'] = '999'
    os.environ['INTERVAL_BNB_BOUNDARY_SPLIT_DEPTH'] = '999'
    os.environ['INTERVAL_BNB_SDP_DEPTH'] = '999'
    os.environ['INTERVAL_BNB_DUMP_BOXES'] = dump_prefix
    os.environ['INTERVAL_BNB_INSTANT_DUMP'] = '1'

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from interval_bnb.parallel import parallel_branch_and_bound
    print("#" * 72, flush=True)
    print(f"# PHASE 1: cheap-cascade BnB d={d} target={target_str}", flush=True)
    print(f"#   workers={workers} budget={time_budget_s}s", flush=True)
    print("#" * 72, flush=True)
    t0 = time.time()
    r = parallel_branch_and_bound(
        d=d, target_c=target_str,
        workers=workers,
        init_split_depth=init_split_depth,
        donate_threshold_floor=2,
        time_budget_s=time_budget_s,
        verbose=True,
    )
    print(f"\n[phase1] elapsed={time.time()-t0:.1f}s success={r['success']} "
          f"coverage={100*r['coverage_fraction']:.5f}% "
          f"in_flight_final={r['in_flight_final']} certs={r['total_leaves_certified']}",
          flush=True)
    return bool(r['success'])


def _collect_stuck_boxes(dump_prefix):
    """Concatenate master-queue + worker-stack dumps."""
    los_all, his_all = [], []
    for path in sorted(glob.glob(f'{dump_prefix}_*.npz')):
        try:
            npz = np.load(path)
            los_all.append(npz['lo'])
            his_all.append(npz['hi'])
            print(f"  {path}: {len(npz['lo'])} boxes", flush=True)
        except Exception as e:
            print(f"  {path}: {type(e).__name__}: {e}", flush=True)
    if not los_all:
        return np.zeros((0, 1)), np.zeros((0, 1))
    los = np.concatenate(los_all, axis=0)
    his = np.concatenate(his_all, axis=0)
    # Deduplicate by exact equality.
    keys = np.concatenate([los, his], axis=1)
    _, uniq = np.unique(keys, axis=0, return_index=True)
    return los[uniq], his[uniq]


_WORKER_GLOBALS = {}


def _worker_init(d, target_num, target_den):
    """Per-worker MOSEK + cache setup. Runs once per process."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast,
    )
    target_f = float(target_num) / float(target_den)
    windows = build_windows(d)
    cache = build_sdp_escalation_cache_fast(d, windows, target=target_f)
    _WORKER_GLOBALS['d'] = d
    _WORKER_GLOBALS['windows'] = windows
    _WORKER_GLOBALS['cache'] = cache
    _WORKER_GLOBALS['target_num'] = target_num
    _WORKER_GLOBALS['target_den'] = target_den


def _worker_solve(args):
    """Solve a single box at given K. Returns (idx, cert_bool, sec)."""
    box_idx, lo, hi, K, time_limit_s = args
    import time as _time
    from interval_bnb.box import Box
    from interval_bnb.bound_sdp_escalation_fast import bound_sdp_escalation_int_ge_fast
    B = Box(lo=np.asarray(lo, dtype=np.float64),
            hi=np.asarray(hi, dtype=np.float64))
    lo_int, hi_int = B.to_ints()
    t0 = _time.time()
    try:
        cert = bound_sdp_escalation_int_ge_fast(
            lo_int, hi_int,
            _WORKER_GLOBALS['windows'],
            _WORKER_GLOBALS['d'],
            target_num=_WORKER_GLOBALS['target_num'],
            target_den=_WORKER_GLOBALS['target_den'],
            cache=_WORKER_GLOBALS['cache'],
            n_window_psd_cones=K,
            n_threads=1,
            time_limit_s=time_limit_s,
        )
    except Exception:
        cert = False
    return box_idx, bool(cert), _time.time() - t0


def _phase2_K(stuck_lo, stuck_hi, indices_to_try, d, target_num, target_den,
              K, time_limit_s, n_workers, label):
    """Run K-batch SDP cert on subset of stuck boxes.
    Returns list of indices that did NOT cert."""
    print(f"\n#### {label}: K={K} on {len(indices_to_try)} boxes "
          f"(workers={n_workers}, time_limit={time_limit_s}s) ####", flush=True)
    if not len(indices_to_try):
        return []
    t0 = time.time()
    args_list = [
        (int(i), stuck_lo[i].tolist(), stuck_hi[i].tolist(), K, time_limit_s)
        for i in indices_to_try
    ]
    cert_ids = []
    fail_ids = []
    times = []
    n_done = 0
    with mp.Pool(processes=n_workers,
                  initializer=_worker_init,
                  initargs=(d, target_num, target_den)) as pool:
        for idx, cert, sec in pool.imap_unordered(_worker_solve, args_list,
                                                    chunksize=1):
            n_done += 1
            times.append(sec)
            if cert:
                cert_ids.append(idx)
            else:
                fail_ids.append(idx)
            if n_done % max(1, n_workers // 4) == 0 or n_done <= 10:
                avg_t = float(np.mean(times)) if times else 0.0
                print(f"  [{n_done}/{len(args_list)}] "
                      f"cert={len(cert_ids)} fail={len(fail_ids)} "
                      f"avg_t={avg_t:.1f}s elapsed={time.time()-t0:.0f}s",
                      flush=True)
    print(f"\n[{label}] cert={len(cert_ids)}/{len(args_list)} "
          f"fail={len(fail_ids)} wall={time.time()-t0:.1f}s "
          f"avg_per_box={np.mean(times):.1f}s", flush=True)
    return fail_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--target', type=str, required=True,
                    help='target c, e.g. 1.2805 or 12805/10000')
    ap.add_argument('--workers', type=int, default=64,
                    help='workers for phase 1 (BnB) — phase 2 uses --sdp_workers')
    ap.add_argument('--sdp_workers', type=int, default=80,
                    help='workers for phase 2 SDP pool')
    ap.add_argument('--init_split_depth', type=int, default=None,
                    help='default = min(d, 22)')
    ap.add_argument('--phase1_budget_s', type=float, default=14400,
                    help='phase 1 wall-clock budget (default 4 h)')
    ap.add_argument('--phase2_time_limit_s', type=float, default=600,
                    help='per-box SDP time limit (default 600 s)')
    ap.add_argument('--K1_frac', type=float, default=0.085,
                    help='K1 / |W| (default 0.085 = 8.5%)')
    ap.add_argument('--K2_frac', type=float, default=0.170,
                    help='K2 / |W| (default 0.170 = 17%)')
    ap.add_argument('--skip_phase1', action='store_true',
                    help='skip phase 1, expect existing dump from --dump_prefix')
    ap.add_argument('--dump_prefix', type=str, default=None,
                    help='dump prefix for phase 1 boxes (default: proof_d{d}_t{target_safe}_phase1)')
    args = ap.parse_args()

    d = args.d
    target_str = args.target
    if '/' in target_str:
        target_q = Fraction(target_str)
    else:
        target_q = Fraction(target_str)
    target_f = float(target_q)
    target_num = target_q.numerator
    target_den = target_q.denominator

    target_safe = target_str.replace('.', 'p').replace('/', '_')
    dump_prefix = args.dump_prefix or f"proof_d{d}_t{target_safe}_phase1"
    init_split_depth = args.init_split_depth or min(d, 22)

    print("#" * 72)
    print(f"# rigorous proof  val({d}) >= {target_str}  (= {target_f:.6f})")
    print(f"# {target_q}  num={target_num} den={target_den}")
    print(f"# dump_prefix={dump_prefix}  init_split_depth={init_split_depth}")
    print("#" * 72, flush=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # ---- PHASE 1 ----
    if not args.skip_phase1:
        # Run phase 1 in a child python so MOSEK isn't loaded yet (cheap cascade
        # doesn't need it; loading it eagerly would balloon worker memory).
        # Actually we just call it inline since the BnB doesn't need MOSEK
        # (we disabled INTERVAL_BNB_SDP_DEPTH).
        success_p1 = _phase1_run(d, target_str, args.workers, init_split_depth,
                                  args.phase1_budget_s, dump_prefix)
        if success_p1:
            print(f"\n[VERDICT] BnB cheap cascade alone certified val({d}) >= {target_str}.",
                  flush=True)
            with open(f'proof_d{d}_t{target_safe}_result.json', 'w') as f:
                json.dump({'success': True, 'phase': 1,
                            'd': d, 'target': target_str}, f, indent=2)
            return

    # Collect stuck boxes
    stuck_lo, stuck_hi = _collect_stuck_boxes(dump_prefix)
    n_stuck = len(stuck_lo)
    print(f"\n[phase1] collected {n_stuck} stuck boxes", flush=True)
    if n_stuck == 0:
        print("[VERDICT] no stuck boxes — phase 1 succeeded?", flush=True)
        return

    # ---- PHASE 2 ----
    from interval_bnb.windows import build_windows
    nW = len(build_windows(d))
    K1 = max(1, int(np.ceil(args.K1_frac * nW)))
    K2 = max(K1 + 1, int(np.ceil(args.K2_frac * nW)))
    K_full = nW
    print(f"\n# |W|={nW}  K1={K1} ({100*K1/nW:.1f}%)  "
          f"K2={K2} ({100*K2/nW:.1f}%)  K_full={K_full}", flush=True)

    pending_ids = list(range(n_stuck))
    phase_records = []

    for label, K in [('phase2a-K1', K1), ('phase2b-K2', K2),
                      ('phase2c-Kfull', K_full)]:
        if not pending_ids:
            break
        # Memory-aware: K_full uses much more RAM, drop worker count.
        if K >= nW * 0.5:
            workers_K = max(1, args.sdp_workers // 4)
        elif K >= nW * 0.10:
            workers_K = max(1, args.sdp_workers // 2)
        else:
            workers_K = args.sdp_workers
        n_in = len(pending_ids)
        fail_ids = _phase2_K(stuck_lo, stuck_hi, pending_ids,
                              d, target_num, target_den,
                              K, args.phase2_time_limit_s, workers_K,
                              label)
        phase_records.append({'label': label, 'K': K,
                                'n_input': n_in,
                                'n_cert': n_in - len(fail_ids),
                                'n_fail': len(fail_ids)})
        pending_ids = fail_ids

    # ---- VERDICT ----
    print("\n" + "=" * 72, flush=True)
    if not pending_ids:
        print(f"VERDICT: val({d}) >= {target_str} is PROVED.", flush=True)
        print("  All {} stuck boxes closed.".format(n_stuck), flush=True)
        success = True
    else:
        print(f"VERDICT: {len(pending_ids)}/{n_stuck} boxes UNCERTIFIED.", flush=True)
        print(f"  Need order-3 Lasserre or higher d.", flush=True)
        success = False
    print("=" * 72, flush=True)

    out = {
        'success': success,
        'd': d,
        'target': target_str,
        'n_stuck_total': n_stuck,
        'phases': phase_records,
        'pending_indices': pending_ids[:200],
    }
    out_path = f'proof_d{d}_t{target_safe}_result.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
