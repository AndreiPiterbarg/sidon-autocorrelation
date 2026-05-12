#!/usr/bin/env python
"""High-d Lasserre sweep via sparse PSD decomposition.

Targets d=128 L2 with aggressive CG + secant search.
Requires memory fixes in lasserre_fusion.py and lasserre_scalable.py.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_enhanced import solve_enhanced, val_d_known

val_d_known.update({
    32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448,
})


def run(desc, d, order, psd_mode, bw, n_bisect=15, cg_rounds=10,
        cg_add=20, search='secant'):
    print(f"\n{'#'*70}")
    print(f"# {desc}")
    print(f"{'#'*70}\n", flush=True)
    t0 = time.time()
    try:
        r = solve_enhanced(
            d, 1.28, order=order, psd_mode=psd_mode,
            search_mode=search, add_upper_loc=True,
            max_cg_rounds=cg_rounds, max_add_per_round=cg_add,
            n_bisect=n_bisect, sparse_bandwidth=bw, verbose=True)
        elapsed = time.time() - t0
        lb = r['lb']
        v = val_d_known.get(d, 0)
        gc = (lb - 1.0) / (v - 1.0) * 100 if v > 1 else 0
        print(f"\n  >>> lb={lb:.8f}, val({d})={v}, "
              f"gap_closure={gc:.1f}%, time={elapsed:.1f}s\n", flush=True)
        r['gap_closure'] = gc
        r['time'] = elapsed
        r['desc'] = desc
        return r
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED ({elapsed:.1f}s): {e}", flush=True)
        import traceback; traceback.print_exc()
        return {'desc': desc, 'lb': 0, 'time': elapsed, 'd': d,
                'order': order, 'status': str(e)}


def main():
    print("=" * 70)
    print("HIGH-D LASSERRE SWEEP — SPARSE PSD + SECANT SEARCH")
    print("=" * 70, flush=True)

    results = []

    # Tier 1: Calibration (validate sparse works)
    results.append(run("L2 d=8 sparse bw=6 (calibrate)",
                       8, 2, 'sparse', 6, n_bisect=14, cg_rounds=5))

    results.append(run("L2 d=16 sparse bw=8 (calibrate)",
                       16, 2, 'sparse', 8, n_bisect=14, cg_rounds=5))

    # Tier 2: Previously infeasible configs
    results.append(run("L2 d=32 sparse bw=10",
                       32, 2, 'sparse', 10, n_bisect=16, cg_rounds=8))

    results.append(run("L2 d=64 sparse bw=12",
                       64, 2, 'sparse', 12, n_bisect=18, cg_rounds=10,
                       cg_add=25))

    # Tier 3: The big target
    results.append(run("L2 d=128 sparse bw=16",
                       128, 2, 'sparse', 16, n_bisect=20, cg_rounds=15,
                       cg_add=30))

    # Tier 4: If time permits — L3 at moderate d
    results.append(run("L3 d=16 sparse bw=8",
                       16, 3, 'sparse', 8, n_bisect=14, cg_rounds=8,
                       cg_add=20))

    results.append(run("L3 d=32 sparse bw=10",
                       32, 3, 'sparse', 10, n_bisect=16, cg_rounds=10,
                       cg_add=25))

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Config':<40} {'lb':>12} {'val(d)':>8} "
          f"{'Gap%':>8} {'Time':>10}")
    print("-" * 80)
    for r in results:
        v = val_d_known.get(r.get('d', 0), 0)
        v_str = f"{v:.3f}" if v else "?"
        lb = r.get('lb', 0)
        lb_str = f"{lb:.8f}" if lb > 0 else "FAILED"
        gc_str = f"{r.get('gap_closure', 0):.1f}%" if lb > 0 else "---"
        t_str = f"{r.get('time', 0):.0f}s"
        print(f"{r.get('desc', '?'):<40} {lb_str:>12} {v_str:>8} "
              f"{gc_str:>8} {t_str:>10}")
    print("=" * 80)

    # Save results
    outpath = os.path.join('data', f"highd_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs('data', exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump([{k: v for k, v in r.items()
                    if not isinstance(v, (type(None),))}
                   for r in results], f, indent=2, default=str)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
