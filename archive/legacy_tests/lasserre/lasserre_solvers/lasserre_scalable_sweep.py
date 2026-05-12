#!/usr/bin/env python
"""
Sweep: compare scalable Lasserre modes against full Lasserre.

Runs configurations that were previously infeasible (OOM) to measure
how much the memory wall has been pushed back.

Expected results (based on analysis):
  L3 d=16 linear:   should work (~300MB), was OOM at ~60GB full
  L3 d=16 cg:       exact full bound with ~20 PSD windows
  L4 d=8  linear:   should match/beat L4 d=8 full (1.120)
  L4 d=16 linear:   NEW — was completely infeasible
  L3 d=32 linear:   NEW — was completely infeasible
"""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_scalable import solve_lasserre_scalable

val_d = {4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
         12: 1.271, 14: 1.284, 16: 1.319}


def gap_closure(lb, d):
    v = val_d.get(d, 0)
    if v <= 1:
        return 0
    return max(0, (lb - 1.0) / (v - 1.0) * 100)


def run_config(desc, d, order, mode, c_target=1.28, **kwargs):
    """Run one configuration and return result dict."""
    print(f"\n{'#'*60}")
    print(f"# {desc}")
    print(f"{'#'*60}\n", flush=True)
    t0 = time.time()
    try:
        r = solve_lasserre_scalable(
            d, c_target, order=order, mode=mode, **kwargs)
        elapsed = time.time() - t0
        lb = r['lb']
        gc = gap_closure(lb, d)
        print(f"\n  => lb={lb:.6f}, val({d})={val_d.get(d, '?')}, "
              f"gap_closure={gc:.1f}%, time={elapsed:.1f}s\n")
        return {'desc': desc, 'lb': lb, 'gc': gc, 'time': elapsed,
                'mode': mode, 'd': d, 'order': order, 'status': 'ok'}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return {'desc': desc, 'lb': 0, 'gc': 0, 'time': elapsed,
                'mode': mode, 'd': d, 'order': order, 'status': str(e)}


def main():
    print("=" * 60)
    print("SCALABLE LASSERRE SWEEP")
    print("=" * 60)
    print()

    results = []

    # --- Tier 1: Quick sanity checks (should take seconds) ---
    results.append(run_config(
        "L2 d=4 linear (sanity check)", 4, 2, 'linear'))
    results.append(run_config(
        "L2 d=8 linear (sanity check)", 8, 2, 'linear'))
    results.append(run_config(
        "L3 d=4 linear", 4, 3, 'linear'))

    # --- Tier 2: Previously feasible (compare bounds) ---
    results.append(run_config(
        "L2 d=16 linear", 16, 2, 'linear'))
    results.append(run_config(
        "L2 d=16 linear+upper", 16, 2, 'linear', add_upper_loc=True))
    results.append(run_config(
        "L3 d=4 fw (verify FW converges)", 4, 3, 'fw', fw_iters=50))

    # --- Tier 3: Previously infeasible — the main event ---
    results.append(run_config(
        "L3 d=16 linear (was OOM@60GB)", 16, 3, 'linear'))
    results.append(run_config(
        "L3 d=16 cg (exact full bound)", 16, 3, 'cg',
        n_bisect=15, cg_rounds=3, cg_add_per_round=15))
    results.append(run_config(
        "L4 d=8 linear (compare to full: 1.120)", 8, 4, 'linear'))
    results.append(run_config(
        "L4 d=10 linear (was OOM@>1TB)", 10, 4, 'linear'))

    # --- Tier 4: Pushing the frontier ---
    results.append(run_config(
        "L4 d=16 linear (new territory)", 16, 4, 'linear'))
    results.append(run_config(
        "L3 d=32 linear (new territory)", 32, 3, 'linear',
        add_upper_loc=False))  # skip upper-loc to save memory

    # --- Summary ---
    print("\n" + "=" * 80)
    print(f"{'Config':<40} {'lb':>10} {'val(d)':>8} "
          f"{'Gap%':>7} {'Time':>8} {'Status':>8}")
    print("-" * 80)
    for r in results:
        v = val_d.get(r['d'], 0)
        v_str = f"{v:.3f}" if v else "?"
        lb_str = f"{r['lb']:.6f}" if r['lb'] > 0 else "---"
        gc_str = f"{r['gc']:.1f}%" if r['lb'] > 0 else "---"
        t_str = f"{r['time']:.1f}s"
        print(f"{r['desc']:<40} {lb_str:>10} {v_str:>8} "
              f"{gc_str:>7} {t_str:>8} {r['status']:>8}")
    print("=" * 80)


if __name__ == '__main__':
    main()
