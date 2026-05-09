"""Pod test: v2 vs v3 cascade head-to-head at multiple configs.

Run on pod with: python3 _pod_v2v3_compare.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                  'cloninger-steinerberger', 'cpu'))
from run_cascade_coarse_v2 import run_cascade as run_v2
from run_cascade_coarse_v3 import run_cascade as run_v3


def fmt_levels(levels):
    return [(lv['level'], lv['survivors']) for lv in levels]


def main():
    configs = [
        (2, 100, 1.25),
        (2, 200, 1.20),
        (2, 100, 1.28),
        (4, 50, 1.25),
        (4, 60, 1.20),
        (2, 80, 1.25),
    ]
    results = []
    for (d0, S, c) in configs:
        print(f"\n=== d0={d0}, S={S}, c={c} ===", flush=True)
        t0 = time.time()
        r2 = run_v2(d0=d0, S=S, c_target=c, max_levels=3,
                    n_workers=8, verbose=False)
        t2 = time.time() - t0
        l2 = fmt_levels(r2['levels'])
        print(f"  v2: pa={r2.get('proven_at')} t={t2:.1f}s "
              f"L0={r2['l0']['survivors']} L1+={l2}", flush=True)

        t0 = time.time()
        r3 = run_v3(d0=d0, S=S, c_target=c, max_levels=3,
                    n_workers=8, verbose=False)
        t3 = time.time() - t0
        l3 = fmt_levels(r3['levels'])
        print(f"  v3: pa={r3.get('proven_at')} t={t3:.1f}s "
              f"L0={r3['l0']['survivors']} L1+={l3}", flush=True)

        # Soundness invariant: v3 ⊆ v2 means at each level,
        # v3 survivors ≤ v2 survivors.
        v2_l0 = r2['l0']['survivors']
        v3_l0 = r3['l0']['survivors']
        if v3_l0 > v2_l0:
            print(f"  *** SOUNDNESS BUG L0: v3 has more survivors than v2 ***")
        for (l2_lv, l3_lv) in zip(r2['levels'], r3['levels']):
            if l3_lv['survivors'] > l2_lv['survivors']:
                print(f"  *** SOUNDNESS BUG L{l2_lv['level']}: "
                      f"v3 {l3_lv['survivors']} > v2 {l2_lv['survivors']} ***")

        results.append({
            'd0': d0, 'S': S, 'c': c,
            'v2_time': t2, 'v3_time': t3,
            'v2_l0_surv': v2_l0, 'v3_l0_surv': v3_l0,
            'v2_levels': l2, 'v3_levels': l3,
            'v2_proven_at': r2.get('proven_at'),
            'v3_proven_at': r3.get('proven_at'),
        })

    print("\n\n=== SUMMARY ===")
    for r in results:
        ratio_l0 = (r['v3_l0_surv'] / max(1, r['v2_l0_surv'])) * 100
        speedup = r['v2_time'] / max(0.001, r['v3_time'])
        print(f"d0={r['d0']}, S={r['S']}, c={r['c']}: "
              f"L0 v2={r['v2_l0_surv']} v3={r['v3_l0_surv']} "
              f"({ratio_l0:.1f}%) | v2 t={r['v2_time']:.1f}s "
              f"v3 t={r['v3_time']:.1f}s ({speedup:.2f}x)")

    with open('_pod_v2v3_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
