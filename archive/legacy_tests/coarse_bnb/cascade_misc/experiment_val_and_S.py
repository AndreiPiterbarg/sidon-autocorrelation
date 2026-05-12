#!/usr/bin/env python
"""
EXPERIMENTS A+C: Determine val(d) and push S for box cert at c=1.30.

EXPERIMENT A: For d0=12..30, test increasing S to find where val(d) >= 1.30.
  - If survivors persist at large S, val(d) < 1.30.
  - If 0 survivors at large S, val(d) >= 1.30.

EXPERIMENT C: For dimensions where val(d) >= 1.30, push S to find the
  crossover point where box cert net goes from negative to positive.
  Track how net scales with S to extrapolate S_needed.

Output: printed to stdout (tee to log), plus JSON summary at end.
"""
import sys
import os
import time
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

import numpy as np
from pruning import count_compositions


def run_l0(d0, S, c_target):
    """Run L0 coarse and return dict of results."""
    from run_cascade import run_level0
    n = count_compositions(d0, S)
    t0 = time.time()
    r = run_level0(d0 / 2.0, 20, c_target, verbose=False,
                   d0=d0, coarse_S=S)
    elapsed = time.time() - t0
    return {
        'd0': d0, 'S': S, 'c_target': c_target,
        'survivors': r['n_survivors'],
        'min_net': r.get('min_cert_net'),
        'box_ok': r.get('box_certified', False),
        'elapsed': round(elapsed, 2),
        'comps': n,
    }


def experiment_A():
    """Find where val(d) crosses 1.30."""
    print("\n" + "=" * 75)
    print("EXPERIMENT A: Determine val(d) for d=12..30 at c_target=1.30")
    print("  Strategy: increase S until survivors appear (val<1.30)")
    print("  or confirm 0 survivors at large S (val>=1.30)")
    print("=" * 75)

    val_status = {}  # d0 -> 'above' or 'below' or 'unknown'
    all_results = []

    for d0 in [12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30]:
        print(f"\n--- d0={d0} ---")
        max_surv_S = 0  # largest S with survivors
        max_zero_S = 0  # largest S with 0 survivors

        S_list = []
        if d0 <= 14:
            S_list = [10, 12, 15, 18, 20, 25, 30, 35, 40, 50]
        elif d0 <= 18:
            S_list = [10, 12, 15, 18, 20, 25, 30, 35]
        elif d0 <= 24:
            S_list = [10, 12, 15, 18, 20, 25]
        else:
            S_list = [10, 12, 15, 18, 20]

        for S in S_list:
            n = count_compositions(d0, S)
            if n > 500_000_000:
                print(f"  S={S:>3}: {n:>15,} comps -- SKIP")
                break

            r = run_l0(d0, S, 1.30)
            all_results.append(r)

            surv = r['survivors']
            mn = r['min_net']
            mn_s = f"{mn:.6f}" if mn is not None else "N/A"
            box = "PASS" if r['box_ok'] else "fail"
            print(f"  S={S:>3}: {n:>12,} comps, {surv:>6} surv, "
                  f"net={mn_s:>12}, box={box}, {r['elapsed']:.1f}s",
                  flush=True)

            if surv > 0:
                max_surv_S = max(max_surv_S, S)
            else:
                max_zero_S = max(max_zero_S, S)

            # Early stop: if survivors at S>=25, val(d) clearly < 1.30
            if surv > 0 and S >= 25:
                break

        # Verdict
        if max_surv_S >= 20:
            val_status[d0] = 'below'
            print(f"  >>> val({d0}) < 1.30 (survivors at S={max_surv_S})")
        elif max_zero_S >= 25 and max_surv_S == 0:
            val_status[d0] = 'above'
            print(f"  >>> val({d0}) >= 1.30 (0 surv through S={max_zero_S})")
        elif max_surv_S > 0:
            val_status[d0] = 'borderline'
            print(f"  >>> val({d0}) ~ 1.30 (0 at S={max_zero_S}, "
                  f"surv at S={max_surv_S})")
        else:
            val_status[d0] = 'likely_above'
            print(f"  >>> val({d0}) likely >= 1.30 "
                  f"(0 surv through S={max_zero_S})")

    # Summary
    print("\n" + "=" * 75)
    print("EXPERIMENT A SUMMARY: val(d) vs 1.30")
    print("=" * 75)
    for d0 in sorted(val_status.keys()):
        print(f"  d={d0:>2}: {val_status[d0]}")

    return all_results, val_status


def experiment_C(val_status, A_results):
    """Push S at dimensions where val(d) >= 1.30 to find box cert crossover."""
    print("\n" + "=" * 75)
    print("EXPERIMENT C: Push S for box cert at promising dimensions")
    print("  For d0 where val(d)>=1.30, find S where box cert net -> 0+")
    print("=" * 75)

    # Find dimensions where val(d) >= 1.30
    good_d0s = sorted([d for d, st in val_status.items()
                       if st in ('above', 'likely_above')])
    borderline_d0s = sorted([d for d, st in val_status.items()
                             if st == 'borderline'])

    if not good_d0s and not borderline_d0s:
        print("  No dimensions with val(d) >= 1.30 found!")
        return []

    all_results = []

    # Strategy: for each good d0, push S as high as feasible
    # Track (S, net) pairs to extrapolate crossover
    for d0 in good_d0s + borderline_d0s:
        print(f"\n--- d0={d0} (val status: {val_status[d0]}) ---")

        # Collect existing results from experiment A
        existing = [(r['S'], r['min_net'], r['survivors'])
                    for r in A_results if r['d0'] == d0]

        # Determine S range to test (go higher than experiment A)
        tested_S = set(r['S'] for r in A_results if r['d0'] == d0)

        # Build list of S values to test, going as high as feasible
        if d0 <= 14:
            S_candidates = list(range(10, 201, 5))
        elif d0 <= 16:
            S_candidates = list(range(10, 101, 5))
        elif d0 <= 20:
            S_candidates = list(range(10, 61, 5))
        elif d0 <= 24:
            S_candidates = list(range(10, 41, 3))
        else:
            S_candidates = list(range(10, 31, 2))

        nets = []  # (S, net) pairs for extrapolation

        # Include existing data
        for S, net, surv in existing:
            if surv == 0 and net is not None:
                nets.append((S, net))

        found_proof = False

        for S in S_candidates:
            if S in tested_S:
                continue

            n = count_compositions(d0, S)
            if n > 2_000_000_000:
                print(f"  S={S:>3}: {n:>15,} comps -- SKIP (too many)")
                continue
            if n > 500_000_000:
                # Still try but note it's slow
                est_time = n / 5_000_000  # rough: 5M comps/sec
                if est_time > 600:  # > 10 min
                    print(f"  S={S:>3}: {n:>15,} comps -- SKIP (est {est_time:.0f}s)")
                    continue
                print(f"  S={S:>3}: {n:>15,} comps (est {est_time:.0f}s)...")

            r = run_l0(d0, S, 1.30)
            all_results.append(r)

            surv = r['survivors']
            mn = r['min_net']
            mn_s = f"{mn:.6f}" if mn is not None else "N/A"
            box = "PASS" if r['box_ok'] else "fail"
            print(f"  S={S:>3}: {n:>12,} comps, {surv:>6} surv, "
                  f"net={mn_s:>12}, box={box}, {r['elapsed']:.1f}s",
                  flush=True)

            if surv == 0 and mn is not None:
                nets.append((S, mn))

            if r['box_ok']:
                print(f"\n  *** RIGOROUS PROOF: C_{{1a}} >= 1.30 ***")
                print(f"  *** d0={d0}, S={S}, net={mn:.6f} ***")
                found_proof = True
                break

            if surv > 0:
                print(f"  (survivors appeared at S={S} — grid too fine)")
                # Don't stop, try even larger S (survivors are rounding)
                continue

        # Extrapolation analysis
        if len(nets) >= 3 and not found_proof:
            nets.sort()
            print(f"\n  Scaling analysis (S vs net):")
            for S_val, net_val in nets:
                print(f"    S={S_val:>3}: net={net_val:.6f}")

            # Fit: net(S) ≈ a - b/S  (cell_var ~ 1/S dominates)
            # Linear regression on net vs 1/S
            inv_S = np.array([1.0/s for s, _ in nets])
            net_vals = np.array([n for _, n in nets])

            if len(nets) >= 2:
                # Fit net = a + b*(1/S)
                A_mat = np.vstack([np.ones(len(nets)), inv_S]).T
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A_mat, net_vals, rcond=None)
                    a, b = coeffs
                    # net = 0 when S = -b/a (if a > 0 and b < 0)
                    if a > 0 and b < 0:
                        S_crossover = -b / a
                        n_cross = count_compositions(d0, int(S_crossover) + 1)
                        print(f"\n  Linear fit: net ≈ {a:.4f} + {b:.2f}/S")
                        print(f"  Predicted box cert crossover: S ≈ {S_crossover:.0f}")
                        print(f"  Compositions at that S: {n_cross:,}")
                        if n_cross < 10_000_000_000:
                            print(f"  >>> FEASIBLE! Should test S={int(S_crossover)+5}")
                        else:
                            print(f"  >>> Too many comps. Need different approach.")
                    elif a > 0:
                        print(f"\n  Net is already trending positive (a={a:.6f})")
                        print(f"  Should reach box cert soon with larger S")
                    else:
                        print(f"\n  Fit: a={a:.6f}, b={b:.2f} — net not converging")
                except Exception as e:
                    print(f"\n  Fit failed: {e}")

    # Final summary
    print("\n" + "=" * 75)
    print("EXPERIMENT C SUMMARY")
    print("=" * 75)

    proofs = [r for r in all_results if r.get('box_ok')]
    if proofs:
        best = min(proofs, key=lambda r: r['S'])
        print(f"\n  *** RIGOROUS PROOF FOUND ***")
        print(f"  C_{{1a}} >= 1.30 at d0={best['d0']}, S={best['S']}")
        print(f"  net={best['min_net']:.6f}, time={best['elapsed']:.1f}s")
    else:
        print("\n  No rigorous proof yet. Best results:")
        for d0 in good_d0s + borderline_d0s:
            d_res = [r for r in (A_results + all_results) if r['d0'] == d0
                     and r['survivors'] == 0 and r.get('min_net') is not None]
            if d_res:
                best = max(d_res, key=lambda r: r['min_net'])
                print(f"  d0={d0}: best net={best['min_net']:.6f} at S={best['S']}")

    return all_results


def main():
    t_start = time.time()

    # Run experiment A
    A_results, val_status = experiment_A()

    # Run experiment C
    C_results = experiment_C(val_status, A_results)

    elapsed_total = time.time() - t_start

    # Save JSON summary
    summary = {
        'experiment': 'val_d_and_S_push',
        'c_target': 1.30,
        'total_time_s': round(elapsed_total, 1),
        'val_status': val_status,
        'A_results': A_results,
        'C_results': C_results,
    }
    out_path = os.path.join(os.path.dirname(__file__), '..',
                            'data', 'experiment_val_S.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*75}")
    print(f"TOTAL TIME: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")
    print(f"Results saved to {out_path}")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
