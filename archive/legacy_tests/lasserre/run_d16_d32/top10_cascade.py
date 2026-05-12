#!/usr/bin/env python
"""
TOP 10 CASCADE EXPERIMENTS to prove C_{1a} >= 1.30

Uses run_cascade_coarse_v2 (parallelized kernel, sound box cert).
Runs at MAXIMUM capacity (all CPU cores).

Key unknowns we're resolving:
  1. Does the cascade CONVERGE at c=1.30? (grid-point proof)
  2. What do box cert nets look like at EACH LEVEL?
  3. Does higher S fix box cert at the critical levels?
  4. What's the sweet spot for d0 vs S?

Experiments ordered by priority (fastest/most informative first).
"""
import sys
import os
import time
import json
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

# Use ALL cores
N_WORKERS = mp.cpu_count()


def run_experiment(exp_id, d0, S, c_target, max_levels=10):
    """Run one cascade experiment and report results."""
    # Use v1 run_cascade which has Gray code + incremental conv (10-100x faster)
    from run_cascade import run_cascade

    print(f"\n{'#'*75}")
    print(f"# EXPERIMENT {exp_id}: d0={d0}, S={S}, c_target={c_target}, "
          f"max_levels={max_levels}")
    print(f"# Workers: {N_WORKERS} (all cores)")
    print(f"# Kernel: Gray code + incremental conv (v1)")
    print(f"{'#'*75}", flush=True)

    t0 = time.time()
    os.makedirs('data_experiment', exist_ok=True)
    result = run_cascade(n_half=d0/2.0, m=20, c_target=c_target,
                         max_levels=max_levels, n_workers=N_WORKERS,
                         verbose=True, output_dir='data_experiment',
                         coarse_S=S, d0=d0)
    elapsed = time.time() - t0

    # Summary
    proven = 'proven_at' in result
    box_ok = result.get('box_certified', False)

    print(f"\n--- EXPERIMENT {exp_id} RESULT ---")
    print(f"  d0={d0}, S={S}, c={c_target}")
    print(f"  Time: {elapsed:.1f}s")

    if proven:
        print(f"  GRID-POINT PROOF at {result['proven_at']}")
        if box_ok:
            print(f"  *** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
    else:
        print(f"  NOT CONVERGED")

    # Per-level details (v1 format)
    l0_surv = result.get('l0_survivors', '?')
    l0_info = result.get('l0', {})
    l0_net = l0_info.get('min_cert_net', None)
    l0_box = l0_info.get('box_certified', False)
    l0_net_s = f"{l0_net:.6f}" if l0_net is not None and l0_net < 1e29 else "N/A"
    print(f"  L0: {l0_surv} surv, net={l0_net_s}, "
          f"box={'PASS' if l0_box else 'FAIL'}")

    for lv in result.get('levels', []):
        box_s = 'PASS' if lv.get('box_certified') else 'FAIL'
        mn = lv.get('min_cert_net', None)
        mn_s = f"{mn:.6f}" if mn is not None and mn < 1e29 else "N/A"
        children = lv.get('total_children', lv.get('children', '?'))
        survivors = lv.get('survivors_out', lv.get('survivors', '?'))
        print(f"  L{lv['level']} (d={lv['d_child']}): "
              f"{children:,} children, "
              f"{survivors:,} surv, "
              f"net={mn_s}, box={box_s}, "
              f"{lv.get('elapsed', lv.get('time', 0)):.1f}s")

    return {
        'exp_id': exp_id,
        'd0': d0, 'S': S, 'c_target': c_target,
        'proven': proven,
        'proven_at': result.get('proven_at'),
        'box_certified': box_ok,
        'elapsed': round(elapsed, 1),
        'l0': l0_info,
        'l0_survivors': l0_surv,
        'levels': result.get('levels', []),
    }


def main():
    t_start = time.time()

    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Using all {N_WORKERS} workers")

    # ===================================================================
    # TOP 10 EXPERIMENTS
    # ===================================================================
    #
    # Strategy: vary d0 and S to find the sweet spot.
    # - Small d0 (2-4): many cascade levels, but L0 is tiny
    # - Medium d0 (6-8): fewer levels, L0 is moderate
    # - S controls grid resolution AND box cert quality
    #
    # Priority: fastest first, then scale up S for box cert.

    experiments = [
        # --- GROUP A: Quick cascade convergence tests (does it converge?) ---

        # Exp 1: Baseline cascade, small S
        # d0=2, S=30: 31 L0 comps. Quick. See how many levels to converge.
        (1, 2, 30, 1.30, 10),

        # Exp 2: d0=2, S=50: 51 L0 comps. Better S for box cert.
        (2, 2, 50, 1.30, 10),

        # Exp 3: d0=4, S=20: 1771 L0 comps. Fewer cascade levels needed.
        (3, 4, 20, 1.30, 8),

        # Exp 4: d0=4, S=30: 5456 L0 comps. Better grid.
        (4, 4, 30, 1.30, 8),

        # --- GROUP B: Push S for box cert at the critical levels ---

        # Exp 5: d0=2, S=75: see if higher S improves box cert nets
        (5, 2, 75, 1.30, 10),

        # Exp 6: d0=2, S=100: aggressive S push
        (6, 2, 100, 1.30, 10),

        # Exp 7: d0=4, S=50: moderate d0 with good S
        (7, 4, 50, 1.30, 8),

        # --- GROUP C: Higher d0 to skip early levels ---

        # Exp 8: d0=6, S=30: 324K L0 comps. Skip 2 cascade levels.
        (8, 6, 30, 1.30, 6),

        # Exp 9: d0=6, S=50: 3.5M L0 comps. Best box cert at d0=6.
        (9, 6, 50, 1.30, 6),

        # --- GROUP D: High S frontier ---

        # Exp 10: d0=2, S=150: maximum S push for box cert
        (10, 2, 150, 1.30, 10),
    ]

    all_results = []
    for exp_id, d0, S, c, max_lev in experiments:
        # Clean checkpoints between experiments
        for f in os.listdir('data_experiment'):
            if f.startswith('checkpoint_'):
                try:
                    os.remove(os.path.join('data_experiment', f))
                except OSError:
                    pass

        try:
            r = run_experiment(exp_id, d0, S, c, max_lev)
            all_results.append(r)

            # If we got a rigorous proof, still run remaining experiments
            # to find optimal parameters
            if r['box_certified']:
                print(f"\n{'*'*75}")
                print(f"* RIGOROUS PROOF FOUND! C_{{1a}} >= {c}")
                print(f"* d0={d0}, S={S}")
                print(f"{'*'*75}")

        except Exception as e:
            print(f"\n  EXPERIMENT {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'exp_id': exp_id, 'd0': d0, 'S': S,
                'error': str(e)
            })

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    elapsed_total = time.time() - t_start

    print(f"\n{'='*75}")
    print(f"FINAL SUMMARY — TOP 10 EXPERIMENTS")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")
    print(f"{'='*75}")

    print(f"\n{'Exp':>4} {'d0':>3} {'S':>4} {'proven':>8} {'box':>5} "
          f"{'levels':>7} {'time':>8} {'worst_net':>12}")
    print(f"  {'-'*60}")

    for r in all_results:
        if 'error' in r:
            print(f"  {r['exp_id']:>2}  {r['d0']:>3} {r['S']:>4}  ERROR: {r['error'][:30]}")
            continue

        proven = r.get('proven_at', 'NO')
        box = 'YES' if r.get('box_certified') else 'no'
        elapsed = r.get('elapsed', 0)

        # Find worst net across all levels
        worst = r.get('l0', {}).get('min_cert_net', 1e30)
        for lv in r.get('levels', []):
            mn = lv.get('min_cert_net', 1e30)
            if mn < worst:
                worst = mn
        worst_s = f"{worst:.6f}" if worst < 1e29 else "N/A"

        n_levels = len(r.get('levels', []))
        print(f"  {r['exp_id']:>2}  {r['d0']:>3} {r['S']:>4}  {proven:>6} "
              f"{box:>5} {n_levels:>5}L {elapsed:>7.1f}s {worst_s:>12}")

    # Box cert analysis: how does net scale with S at each level?
    print(f"\n--- BOX CERT NET vs S (per cascade level) ---")
    # Group by (d0, level_d_child)
    from collections import defaultdict
    level_data = defaultdict(list)  # (d0, d_child) -> [(S, net)]

    for r in all_results:
        if 'error' in r:
            continue
        d0 = r['d0']
        l0 = r.get('l0', {})
        l0_net = l0.get('min_cert_net', 1e30)
        if l0_net < 1e29:
            level_data[(d0, d0)].append((r['S'], l0_net))
        for lv in r.get('levels', []):
            mn = lv.get('min_cert_net', 1e30)
            if mn < 1e29:
                level_data[(d0, lv['d_child'])].append((r['S'], mn))

    for (d0, d_child), points in sorted(level_data.items()):
        points.sort()
        print(f"\n  d0={d0}, d_child={d_child}:")
        for S_val, net_val in points:
            bar = '+' * max(0, int((net_val + 2) * 20)) if net_val > -2 else ''
            print(f"    S={S_val:>4}: net={net_val:>10.6f}  {bar}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..',
                            'data', 'top10_cascade_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Rigorous proofs found?
    proofs = [r for r in all_results if r.get('box_certified')]
    if proofs:
        best = min(proofs, key=lambda r: r['S'])
        print(f"\n*** BEST RIGOROUS PROOF: C_{{1a}} >= {best['c_target']} ***")
        print(f"*** d0={best['d0']}, S={best['S']}, "
              f"proved at {best['proven_at']} ***")
    else:
        print(f"\nNo rigorous proofs yet.")
        # Find closest to passing
        best_net = -1e30
        for r in all_results:
            if 'error' in r:
                continue
            for lv in r.get('levels', []):
                mn = lv.get('min_cert_net', 1e30)
                if mn < 1e29 and mn > best_net:
                    best_net = mn
            l0_net = r.get('l0', {}).get('min_cert_net', 1e30)
            if l0_net < 1e29 and l0_net > best_net:
                best_net = l0_net
        print(f"Closest box cert net to 0: {best_net:.6f}")


if __name__ == '__main__':
    main()
