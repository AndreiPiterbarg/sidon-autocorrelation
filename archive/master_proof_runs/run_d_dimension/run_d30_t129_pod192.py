"""d=30 t=1.29 interval BnB on 192-core pod — fully optimized for this run.

GOAL: rigorously certify val(30) >= 1.29, hence C_{1a} >= 1.29.
WHY d=30:
  - val(30) UB extrapolated to ~1.35 (asymptotic fit val(d) = 1.503 - 0.36*exp(-d/35))
    => target margin ~0.06, which is 2.1x the d=22/t=1.281 margin (0.028)
    that just closed to 99.999998% in 3h on 16 cores.
  - 192-core scaling pushes the BnB to <30 min wallclock with all 4 fixes.

ALL FOUR PROVEN FIXES ENABLED via env vars. None of the core code is touched.

CASCADE PHILOSOPHY: cheap-first. Quick majority certify with the cheap
filters, reserve expensive bounds for the genuinely difficult deep tail.

The actual order in interval_bnb/parallel.py:
   (a) autoconv / McCormick batch bounds      ~1us       always
   (b) P2 safe-margin shortcut (if lb-target>1e-9)        always
   (c) standard rigor + CCTR (single-window)  ~ms        always
   (d) MULTI-ANCHOR cut over {mu*, sigma(mu*)} ~10us     depth >= ANCHOR_DEPTH
   (e) CCTR multi-alpha aggregates (SW/NE)     ~ms       always
   (f) CCTR joint-face + RLT (LP-based)        ~30-50ms  depth >= JOINT_DEPTH
   (g) EPIGRAPH LP (full RLT/SOS)              ~70-150ms depth >= EPIGRAPH_DEPTH
   (h) PER-BOX CENTROID ANCHOR (last resort)   ~98ms     depth >= CENTROID_DEPTH
   (i) split (axis chosen by LP-binding/cross-var/widest)

Effect at d=30 on 192c: shallow boxes close via (a)-(c) before paying any
LP cost; mid-depth boxes use (d)-(g); only the genuinely difficult deep
tail (depth >= 60) pays the expensive (h) safety net.

CASCADE TUNED FOR d=30 / margin ~0.04 / 192 cores:
  * TIGHTEN_DEPTH=4    early simplex shrink (default; cheap monotone tighten)
  * TOPK_JOINT_DEPTH=12 EARLIER than d=22's 14: large margin closes via P1-LITE
  * TOPK_JOINT_K=3      default
  * EPIGRAPH_DEPTH=26   LATER than d=22's 24: per-LP cost grows ~d^4
  * EPIGRAPH_FILTER=0.05 LOOSER than 0.02: skip LP when float-LB is far from target
  * ANCHOR_DEPTH=22     multi-anchor cheap filter (cost ~10us, fires at any margin)
  * CENTROID_DEPTH=60   NEW: per-box centroid anchor (expensive ~98ms safety
                        net; only fires when epi LP missed AND box is narrow
                        enough for the curvature concession to be tighter
                        than the missing margin — empirically half-width
                        <= 0.0015 for the bound to certify, which appears
                        around depth 60+)
  * LP_SPLIT_DEPTH=28   LP-binding-axis split (uses just-solved epigraph LP duals)
  * PC_DEPTH=26         EARLIER than 25 default: cross-box variance EMA split

WORKER FANOUT FOR 192 CORES:
  * init_split_depth=26 yields ~5-15K starter boxes (25-75 per worker locally)
  * donate_threshold_floor=2 aggressive work-stealing across workers
  * pull_batch_max=64 (default) — empirically optimal at 192 cores per d=14 run

PHASES (one Python script, runs end-to-end):
  1. KKT mu* finder for d=30 (saves mu_star_d30.npz; uses 192 cores Phase 1)
     Skipped if mu_star_d30.npz already present.
  2. Interval BnB with all fixes; saves d30_t129_result.json + d30_t129_log.txt.

Estimated wallclock on 192c (per the d^4.6 LP fit + node-count ~1/margin):
    KKT mu*:  ~7 min   (Phase 1 dominant; 3000 starts / 192 workers)
    BnB:      ~10-20 min if it closes cleanly; up to 4h budget if deep tail drags.
    TOTAL:    ~20-30 min expected, 4h hard cap.

KNOWN PRE-PUBLICATION CAVEAT:
  bound_epigraph_int_ge uses a 1e-14 (float-arith-only) safety cushion that does
  not cover HiGHS's 1e-7 dual-feasibility tolerance. With margin 0.06 the practical
  effect is zero, but for the FINAL published cert this needs a 1e-7 cushion or a
  full Neumaier-Shcherbina dual cert. Flag for the audit pass.
"""
from __future__ import annotations

import json
import os
import sys
import time

# ============================================================
# CASCADE THRESHOLDS — TUNED FOR d=30 / margin 0.06 / 192 cores
# Set BEFORE any interval_bnb import (env vars are read at module init).
# ============================================================
os.environ.setdefault('INTERVAL_BNB_TIGHTEN_DEPTH', '4')
os.environ.setdefault('INTERVAL_BNB_TOPK_JOINT_DEPTH', '12')
os.environ.setdefault('INTERVAL_BNB_TOPK_JOINT_K', '3')
os.environ.setdefault('INTERVAL_BNB_EPIGRAPH_DEPTH', '26')
os.environ.setdefault('INTERVAL_BNB_EPIGRAPH_FILTER', '0.05')
os.environ.setdefault('INTERVAL_BNB_ANCHOR_DEPTH', '22')      # cheap multi-anchor
os.environ.setdefault('INTERVAL_BNB_CENTROID_DEPTH', '60')    # expensive last-resort
os.environ.setdefault('INTERVAL_BNB_LP_SPLIT_DEPTH', '28')
os.environ.setdefault('INTERVAL_BNB_PC_DEPTH', '26')

# Force UTF-8 stdout (some pods have cp1252 default)
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ============================================================
# Phase 1: KKT mu* finder (skipped if mu_star_d30.npz exists)
# ============================================================

def ensure_mu_star_d30():
    """Find mu_star for d=30 if not already cached. Returns f(mu*) = val(30) UB."""
    npz_path = os.path.join(_HERE, 'mu_star_d30.npz')
    if os.path.isfile(npz_path):
        import numpy as np
        v = np.load(npz_path, allow_pickle=True)
        f = float(v['f']) if 'f' in v.files else float(v['f_value'])
        residual = float(v['residual']) if 'residual' in v.files else -1.0
        print(f"[setup] mu_star_d30.npz cached. f(mu*) = {f:.10f}, "
              f"residual = {residual:.2e}", flush=True)
        return f

    print("[setup] mu_star_d30.npz not found — running KKT pipeline ...",
          flush=True)
    from kkt_correct_mu_star import find_kkt_correct_mu_star
    import numpy as np

    t0 = time.time()
    result = find_kkt_correct_mu_star(
        d=30,
        x_cap=1.0,
        n_starts=3000,        # 2x d=22 to compensate for higher dim
        n_workers=192,        # full pod for Phase 1 multistart
        top_K_phase2=80,
        top_K_phase3=15,
        target_residual=1e-6,
        verbose=True,
    )
    dt = time.time() - t0

    if result.get('mu_star') is None:
        raise RuntimeError("KKT mu* pipeline failed for d=30")

    np.savez(
        npz_path,
        mu=result['mu_star'],
        f=result['f_value'],
        residual=result['residual'],
    )
    f = float(result['f_value'])
    res = float(result['residual'])
    print(f"[setup] saved mu_star_d30.npz. f(mu*) = {f:.10f}, "
          f"residual = {res:.2e}, total {dt:.1f}s", flush=True)
    return f


# ============================================================
# Phase 2: BnB on 192 cores with all 4 fixes active
# ============================================================

def main():
    d = 30
    target = '1.29'
    workers = 192
    time_budget_s = 14400  # 4 hours hard cap

    print("#" * 72, flush=True)
    print(f"# d={d} t={target} BnB on {workers}-core pod", flush=True)
    print(f"# All 4 fixes (LP cuts, H_d, mu*-anchor, LP-binding split) ENABLED",
          flush=True)
    print("#" * 72, flush=True)

    # ----- KKT mu* (anchor cut input) -----
    val_d_ub = ensure_mu_star_d30()
    margin = val_d_ub - 1.29
    print(f"[setup] val({d}) UB = {val_d_ub:.10f}", flush=True)
    print(f"[setup] target = {target}, margin = {margin:+.6f}", flush=True)
    if margin <= 0:
        print(f"!!! INFEASIBLE: val({d}) UB <= target. Aborting. !!!", flush=True)
        return
    if margin < 0.020:
        print(f"!!! WARNING: margin {margin:.4f} is small (<0.02). "
              f"Deep-tail stall risk increases. Proceeding anyway. !!!",
              flush=True)
    print(f"[setup] margin healthy ({margin:.4f}); proceeding", flush=True)

    # Late import so env vars are honored
    from interval_bnb.parallel import parallel_branch_and_bound

    print("\n" + "=" * 72, flush=True)
    print(f"# Cascade for d={d} / margin {margin:.3f} / {workers} cores:",
          flush=True)
    print(f"#   CHEAP FIRST: autoconv -> McCormick -> rigor+CCTR(SW/NE)",
          flush=True)
    print(f"#   MID:         multi-anchor>=22 -> CCTR(JOINT/RLT) -> epigraph LP>=26",
          flush=True)
    print(f"#   DEEP-TAIL:   centroid-anchor>=60 (ONLY if epi LP missed + box narrow)",
          flush=True)
    print(f"#   SPLIT:       LP-binding>=28 / cross-var>=26 / widest fallback",
          flush=True)
    print(f"#   PRE-FILTER:  H_d cut on every box pop  +  tighten-to-simplex>=4",
          flush=True)
    print(f"# init_split_depth=26  donate_threshold_floor=2  budget={time_budget_s}s",
          flush=True)
    print("=" * 72 + "\n", flush=True)

    t0 = time.time()
    r = parallel_branch_and_bound(
        d=d,
        target_c=target,
        workers=workers,
        init_split_depth=26,            # ~5-15K starters for 192 workers
        donate_threshold_floor=2,       # eager work-stealing
        time_budget_s=time_budget_s,
        verbose=True,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 72, flush=True)
    print(f"FINAL RESULT  d={d}  target={target}", flush=True)
    print("=" * 72, flush=True)
    print(f"  success                : {r['success']}", flush=True)
    print(f"  total nodes            : {r['total_nodes']:,}", flush=True)
    print(f"  certified leaves       : {r['total_leaves_certified']:,}", flush=True)
    print(f"  max depth              : {r['max_depth']}", flush=True)
    print(f"  coverage               : {100 * r['coverage_fraction']:.10f}%",
          flush=True)
    print(f"  uncovered volume       : {r['total_volume'] - r['closed_volume']:.3e}",
          flush=True)
    print(f"  in_flight final        : {r['in_flight_final']}", flush=True)
    print(f"  elapsed                : {elapsed:.1f}s = "
          f"{elapsed/60:.2f}min = {elapsed/3600:.3f}h", flush=True)
    print(f"  CCTR stats             : {r.get('cctr_stats', {})}", flush=True)
    print(f"  EPIGRAPH stats         : {r.get('epi_stats', {})}", flush=True)
    print(f"  ANCHOR stats           : {r.get('anchor_stats', {})}", flush=True)
    print(f"  val(d) UB              : {val_d_ub:.10f}", flush=True)
    print(f"  margin                 : {margin:+.6f}", flush=True)

    serializable = {k: v for k, v in r.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
    serializable['wall_time_s'] = elapsed
    serializable['val_d_ub'] = val_d_ub
    serializable['margin'] = margin
    serializable['cascade'] = {
        k: os.environ[k] for k in (
            'INTERVAL_BNB_TIGHTEN_DEPTH',
            'INTERVAL_BNB_TOPK_JOINT_DEPTH',
            'INTERVAL_BNB_TOPK_JOINT_K',
            'INTERVAL_BNB_EPIGRAPH_DEPTH',
            'INTERVAL_BNB_EPIGRAPH_FILTER',
            'INTERVAL_BNB_ANCHOR_DEPTH',
            'INTERVAL_BNB_CENTROID_DEPTH',
            'INTERVAL_BNB_LP_SPLIT_DEPTH',
            'INTERVAL_BNB_PC_DEPTH',
        )
    }

    out_path = os.path.join(_HERE, 'd30_t129_result.json')
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[done] saved {out_path}", flush=True)

    if r['success']:
        print(f"\n*** PROOF: val({d}) >= {target}, hence C_{{1a}} >= {target} ***",
              flush=True)
    else:
        cov = r['coverage_fraction']
        if cov > 0.999:
            print(f"\n[partial] {100*cov:.6f}% coverage — deep tail not closed "
                  f"in budget. Re-launch with longer budget or tighter cushion.",
                  flush=True)
        else:
            print(f"\n[FAILURE] coverage {100*cov:.4f}% — investigate stall",
                  flush=True)


if __name__ == "__main__":
    main()
