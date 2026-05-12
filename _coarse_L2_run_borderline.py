"""Targeted bench: focus on 'borderline' cells where tri_net is near zero.
These are the cells most likely to be rescued by a tighter SDP.

Sorts hard cells by tri_net DESCENDING (closest-to-zero first).
"""
import json, time
import numpy as np
from _coarse_L_bench import cell_cert_shor
from _coarse_L2_bench import cell_cert_lasserre2


def bench_borderline(d, S, c_target, max_cells=15, solver='MOSEK'):
    print(f"\n=== d={d} S={S} c={c_target} (BORDERLINE: most savable) ===")
    with open('_coarse_L2_hardcells_cache.json') as f:
        cache = json.load(f)
    key = f'{d}_{S}_{c_target}'
    rows = cache[key]['rows']
    n_grid_pass = cache[key]['n_grid_pass']
    n_tri_cert = cache[key]['n_tri_cert']
    print(f"    grid passers: {n_grid_pass:,}  tri_cert: {n_tri_cert:,}  hard: {len(rows):,}")

    # Sort by tri_net DESCENDING (largest = closest-to-zero first; most savable)
    rows_sorted = sorted(rows, key=lambda r: -r['tri_net'])
    cells = rows_sorted[:max_cells]

    n_shor_cert = 0; n_l2_cert = 0; n_l2_strict = 0; n_l2_rescue = 0
    times_shor = []; times_l2 = []
    detail = []
    for k, row in enumerate(cells):
        c = np.asarray(row['c'], dtype=np.int32)
        Wstar = tuple(row['tri_W'])

        t0 = time.time()
        shor_lb, _ = cell_cert_shor(c, S, d, c_target, Wstar, solver=solver)
        dt_shor = time.time() - t0

        t0 = time.time()
        l2_lb, _ = cell_cert_lasserre2(c, S, d, c_target, Wstar, solver=solver)
        dt_l2 = time.time() - t0

        shor_cert = shor_lb >= c_target - 1e-9
        l2_cert = l2_lb >= c_target - 1e-9

        if shor_cert: n_shor_cert += 1
        if l2_cert: n_l2_cert += 1
        if l2_lb > shor_lb + 1e-8: n_l2_strict += 1
        if l2_cert and not shor_cert: n_l2_rescue += 1

        times_shor.append(dt_shor); times_l2.append(dt_l2)
        detail.append({
            'c': row['c'],
            'tri_net': row['tri_net'],
            'tri_W': row['tri_W'],
            'shor_lb': float(shor_lb),
            'l2_lb': float(l2_lb),
            'gap': float(l2_lb - shor_lb),
            'shor_cert': shor_cert,
            'l2_cert': l2_cert,
            'l2_rescue': l2_cert and not shor_cert,
            't_shor_ms': 1000*dt_shor,
            't_l2_ms': 1000*dt_l2,
        })
        if k < 12:
            tag = 'RESCUE' if (l2_cert and not shor_cert) else (
                'cert' if shor_cert else 'fail')
            print(f"    [{k:3d}] tri_net={row['tri_net']:+.6f}  "
                  f"shor={shor_lb:.5f}({'C' if shor_cert else 'f'}) "
                  f"L2={l2_lb:.5f}({'C' if l2_cert else 'f'}) "
                  f"gap=+{l2_lb-shor_lb:.2e} {tag}  "
                  f"T={dt_shor*1000:.0f}/{dt_l2*1000:.0f}ms")

    times_shor = np.asarray(times_shor); times_l2 = np.asarray(times_l2)
    print(f"\n    --- Summary ---")
    print(f"    Cells tested            : {len(cells)}")
    print(f"    Shor certified          : {n_shor_cert}/{len(cells)}")
    print(f"    Lasserre-2 certified    : {n_l2_cert}/{len(cells)}")
    print(f"    L2 strictly > Shor LB   : {n_l2_strict}")
    print(f"    L2 rescues (Shor fail->L2 cert): {n_l2_rescue}")
    print(f"    Shor time/cell (ms)     : med={1000*np.median(times_shor):.1f}")
    print(f"    L2  time/cell (ms)      : med={1000*np.median(times_l2):.1f}")
    print(f"    Time ratio L2/Shor      : "
          f"med={np.median(times_l2)/max(1e-9,np.median(times_shor)):.2f}x")

    return {
        'd': d, 'S': S, 'c_target': c_target, 'mode': 'borderline',
        'n_total_hard': len(rows),
        'n_cells_tested': len(cells),
        'n_shor_cert': n_shor_cert,
        'n_l2_cert': n_l2_cert,
        'n_l2_strict': n_l2_strict,
        'n_l2_rescue': n_l2_rescue,
        't_shor_med_ms': float(1000 * np.median(times_shor)),
        't_l2_med_ms': float(1000 * np.median(times_l2)),
        't_shor_p95_ms': float(1000 * np.percentile(times_shor, 95)),
        't_l2_p95_ms': float(1000 * np.percentile(times_l2, 95)),
        'detail': detail,
    }


if __name__ == '__main__':
    results = []
    for d, S, c in [(4, 20, 1.20), (6, 15, 1.20), (8, 12, 1.20)]:
        r = bench_borderline(d, S, c, max_cells=12)
        results.append(r)

    with open('_coarse_L2_borderline.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== OVERALL BORDERLINE SUMMARY ===")
    for r in results:
        rescue_rate = 100.0 * r['n_l2_rescue'] / max(1, r['n_cells_tested'])
        print(f"  d={r['d']} S={r['S']}: {r['n_l2_rescue']}/{r['n_cells_tested']} L2 rescues "
              f"({rescue_rate:.1f}%)  time L2/Shor = {r['t_l2_med_ms']/max(1,r['t_shor_med_ms']):.1f}x")
