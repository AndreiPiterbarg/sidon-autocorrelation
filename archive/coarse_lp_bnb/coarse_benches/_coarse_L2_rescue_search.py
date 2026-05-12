"""Search for actual L2 rescues: pre-screen with Shor, then run L2 on Shor-failing
cells, looking for cases where L2 LB jumps over c_target.

Strategy: among the 'closest savable' cells (sorted by tri_net descending), try
to find any 'just-failed-by-Shor' cell where L2 saves it.
"""
import json, time
import numpy as np
from _coarse_L_bench import cell_cert_shor
from _coarse_L2_bench import cell_cert_lasserre2


def search_rescue(d, S, c_target, n_screen=80, solver='MOSEK'):
    print(f"\n=== d={d} S={S} c={c_target}: searching for L2 rescues ===")
    with open('_coarse_L2_hardcells_cache.json') as f:
        cache = json.load(f)
    key = f'{d}_{S}_{c_target}'
    rows = cache[key]['rows']
    print(f"    total hard cells: {len(rows):,}")

    # Phase 1: Shor on best-W (cheapest screen)
    print(f"    Phase 1: Shor on first {n_screen} cells (sorted by tri_net DESC)...")
    rows_sorted = sorted(rows, key=lambda r: -r['tri_net'])[:n_screen]

    shor_failing = []
    n_shor_cert_first = 0
    t0 = time.time()
    for k, row in enumerate(rows_sorted):
        c = np.asarray(row['c'], dtype=np.int32)
        Wstar = tuple(row['tri_W'])
        lb, _ = cell_cert_shor(c, S, d, c_target, Wstar, solver=solver)
        cert = lb >= c_target - 1e-9
        if cert:
            n_shor_cert_first += 1
        else:
            shor_failing.append({**row, 'shor_lb': float(lb)})
    t_shor_phase = time.time() - t0
    print(f"    Phase 1 done in {t_shor_phase:.1f}s")
    print(f"    Shor cert: {n_shor_cert_first}/{n_screen}")
    print(f"    Shor failing: {len(shor_failing)}")

    if not shor_failing:
        print(f"    No Shor-failing cells in screen. No rescue possible.")
        return {'d': d, 'S': S, 'c_target': c_target,
                'n_screen': n_screen, 'n_shor_failing': 0,
                'n_l2_rescue': 0}

    # Sort shor_failing by Shor's LB (DESCENDING, closest to c_target first = most rescuable)
    shor_failing.sort(key=lambda r: -r['shor_lb'])

    # Phase 2: L2 on Shor-failing cells (closest to c_target first)
    n_l2_rescue = 0
    n_l2_cert = 0
    n_l2_strict = 0
    times_l2 = []
    print(f"    Phase 2: L2 on {min(20, len(shor_failing))} Shor-failing cells...")
    detail = []
    for k, row in enumerate(shor_failing[:20]):
        c = np.asarray(row['c'], dtype=np.int32)
        Wstar = tuple(row['tri_W'])
        t0 = time.time()
        l2_lb, _ = cell_cert_lasserre2(c, S, d, c_target, Wstar, solver=solver)
        dt = time.time() - t0
        times_l2.append(dt)

        cert = l2_lb >= c_target - 1e-9
        rescue = cert  # by construction (Shor failed at this cell)
        gap = l2_lb - row['shor_lb']

        if cert: n_l2_cert += 1
        if rescue: n_l2_rescue += 1
        if gap > 1e-8: n_l2_strict += 1

        tag = 'RESCUE' if rescue else 'fail'
        if k < 12:
            print(f"    [{k:3d}] tri_net={row['tri_net']:+.6f}  "
                  f"shor={row['shor_lb']:.5f}(f) L2={l2_lb:.5f}({'C' if cert else 'f'})  "
                  f"gap=+{gap:.2e}  {tag}  T_L2={dt*1000:.0f}ms")

        detail.append({
            'c': row['c'],
            'tri_net': row['tri_net'],
            'shor_lb': row['shor_lb'],
            'l2_lb': float(l2_lb),
            'gap': float(gap),
            'rescue': rescue,
            't_l2_ms': 1000*dt,
        })

    print(f"\n    --- Summary ---")
    print(f"    Cells screened           : {n_screen}")
    print(f"    Shor failing             : {len(shor_failing)}")
    print(f"    L2 tested (closest)      : {min(20, len(shor_failing))}")
    print(f"    L2 rescues (Shor fail to L2 cert): {n_l2_rescue}")
    print(f"    L2 strict gain over Shor : {n_l2_strict}")
    if times_l2:
        print(f"    L2 time/cell (ms)        : "
              f"med={1000*np.median(times_l2):.1f}  "
              f"max={1000*np.max(times_l2):.1f}")
    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_screen': n_screen,
        'n_shor_failing': len(shor_failing),
        'n_l2_tested': min(20, len(shor_failing)),
        'n_l2_rescue': n_l2_rescue,
        'n_l2_strict': n_l2_strict,
        't_l2_med_ms': float(1000 * np.median(times_l2)) if times_l2 else None,
        'detail': detail,
    }


if __name__ == '__main__':
    results = []
    for d, S, c, n_scr in [(4, 20, 1.20, 79), (6, 15, 1.20, 100), (8, 12, 1.20, 100)]:
        r = search_rescue(d, S, c, n_screen=n_scr)
        results.append(r)

    with open('_coarse_L2_rescue_search.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== OVERALL RESCUE SUMMARY ===")
    for r in results:
        rate = 100.0 * r['n_l2_rescue'] / max(1, r['n_l2_tested']) if r.get('n_l2_tested') else 0.0
        print(f"  d={r['d']} S={r['S']}: rescues = {r['n_l2_rescue']}/{r.get('n_l2_tested',0)} "
              f"({rate:.1f}%); Shor failing screened: {r['n_shor_failing']}/{r['n_screen']}")
