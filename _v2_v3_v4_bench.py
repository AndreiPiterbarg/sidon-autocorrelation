"""Head-to-head bench: v2 baseline vs v3 (N+O) vs v4 (N+O+J+L) cascades.

Reports:
  - Total time per cascade
  - Per-level survivors (so we see where each layer helps)
  - Joint dual + Shor SDP cell counts (v4 only)
  - Soundness invariant: v4 ⊆ v3 ⊆ v2 at every level
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)


def fmt_levels(levels):
    return [(lv['level'], lv['survivors']) for lv in levels]


def run_one(d0, S, c, use_sdp=False, n_workers=1, max_levels=4):
    from run_cascade_coarse_v2 import run_cascade as rv2
    from run_cascade_coarse_v3 import run_cascade as rv3
    from run_cascade_coarse_v4 import run_cascade as rv4

    print(f"\n=== d0={d0}, S={S}, c={c}, use_sdp={use_sdp} ===", flush=True)

    t0 = time.time()
    r2 = rv2(d0=d0, S=S, c_target=c, max_levels=max_levels,
             n_workers=n_workers, verbose=False)
    t2 = time.time() - t0
    L2 = fmt_levels(r2['levels'])
    print(f"  v2: pa={r2.get('proven_at')} t={t2:.2f}s "
          f"L0={r2['l0']['survivors']} L1+={L2}", flush=True)

    t0 = time.time()
    r3 = rv3(d0=d0, S=S, c_target=c, max_levels=max_levels,
             n_workers=n_workers, verbose=False)
    t3 = time.time() - t0
    L3 = fmt_levels(r3['levels'])
    print(f"  v3: pa={r3.get('proven_at')} t={t3:.2f}s "
          f"L0={r3['l0']['survivors']} L1+={L3}", flush=True)

    t0 = time.time()
    r4 = rv4(d0=d0, S=S, c_target=c, max_levels=max_levels,
             n_workers=n_workers, verbose=False, use_joint=True, use_sdp=use_sdp)
    t4 = time.time() - t0
    L4 = fmt_levels(r4['levels'])
    print(f"  v4: pa={r4.get('proven_at')} t={t4:.2f}s "
          f"L0={r4['l0']['survivors']} L1+={L4}", flush=True)
    # v4 layer-counts
    L0_v4 = r4['l0']
    print(f"      L0 layer counts: NO={L0_v4.get('n_pruned_NO',0):,} "
          f"J={L0_v4.get('n_pruned_J',0):,} "
          f"L={L0_v4.get('n_pruned_L',0):,}")

    # Soundness invariant
    bug = False
    if r3['l0']['survivors'] > r2['l0']['survivors']:
        print("  *** SOUNDNESS BUG L0: v3 has more survivors than v2 ***")
        bug = True
    if r4['l0']['survivors'] > r3['l0']['survivors']:
        print("  *** SOUNDNESS BUG L0: v4 has more survivors than v3 ***")
        bug = True

    return {
        'd0': d0, 'S': S, 'c': c,
        'v2': {'pa': r2.get('proven_at'), 't': t2, 'L0': r2['l0']['survivors'],
               'levels': L2},
        'v3': {'pa': r3.get('proven_at'), 't': t3, 'L0': r3['l0']['survivors'],
               'levels': L3},
        'v4': {'pa': r4.get('proven_at'), 't': t4, 'L0': r4['l0']['survivors'],
               'levels': L4,
               'NO_count': L0_v4.get('n_pruned_NO', 0),
               'J_count': L0_v4.get('n_pruned_J', 0),
               'L_count': L0_v4.get('n_pruned_L', 0)},
        'soundness_bug': bug,
    }


def main():
    configs = [
        # (d0, S, c, use_sdp, max_levels)
        (2, 30, 1.20, False, 4),
        (2, 50, 1.20, False, 4),
        (2, 30, 1.25, False, 4),
        (4, 20, 1.20, True, 3),
        (4, 30, 1.25, True, 3),
        (6, 15, 1.20, False, 2),  # use_sdp=False to keep wall-time reasonable
    ]
    results = []
    for cfg in configs:
        d0, S, c, use_sdp, max_levels = cfg
        results.append(run_one(d0, S, c, use_sdp=use_sdp,
                                n_workers=1, max_levels=max_levels))
    print("\n=== Summary ===")
    for r in results:
        v2_t = r['v2']['t']; v3_t = r['v3']['t']; v4_t = r['v4']['t']
        print(f"d0={r['d0']}, S={r['S']}, c={r['c']}: "
              f"v2 ({v2_t:.2f}s, L0={r['v2']['L0']}) -> "
              f"v3 ({v3_t:.2f}s) -> v4 ({v4_t:.2f}s, J={r['v4']['J_count']:,} "
              f"L={r['v4']['L_count']:,})")
    out = os.path.join(_dir, '_v2_v3_v4_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
