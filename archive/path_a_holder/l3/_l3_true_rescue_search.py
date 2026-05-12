"""Search across many (d, S, c_target) for cells where Shor fails BUT vertex_ub
suggests certifiable.  Then run L2 and L3 to see if either can rescue.

A 'true rescue candidate' is a cell where:
    shor_lb < c_target  (Shor doesn't certify)
    vertex_ub >= c_target  (an exact QP solver MIGHT certify)

If L3 can match vertex_ub on such a cell, it rescues.
"""
import warnings, sys, time, os, json
warnings.filterwarnings('ignore')
os.environ['CVXPY_VERBOSE'] = '0'
import numpy as np
from _coarse_L_bench import (
    find_hard_cells, cell_cert_shor, qp_min_vertex_eval, tv_at, all_windows,
)
from _coarse_L2_bench import cell_cert_lasserre2
from _lasserre3_cell_cert import cell_cert_lasserre3

# Test multiple (d, S, c) configs, looking for true rescues
configs = [
    (4, 20, 1.281),
    (4, 40, 1.281),
    (4, 80, 1.281),
    (6, 15, 1.281),
]

all_candidates = []
for d, S, c_target in configs:
    print(f'\n=== d={d} S={S} c={c_target} ===', flush=True)
    t0 = time.time()
    hard, _, _, _ = find_hard_cells(d=d, S=S, c_target=c_target, max_eval=10**5)
    print(f'  {len(hard)} hard cells in {time.time()-t0:.0f}s', flush=True)
    hard.sort(key=lambda kv: -kv[1]['net'])

    # For each hard cell, check Shor on every window where TV>c_target
    # and vertex_ub on every such window.
    # Rescue candidate if:  there is a window where shor<c, vert_ub>=c
    candidates = []
    h = 1/(2*S)
    t0 = time.time()
    for k, (c, tri) in enumerate(hard):
        ws = [W for W in all_windows(d) if tv_at(c, S, d, *W) > c_target]
        for W in ws:
            ell, s_lo = W
            v_min = qp_min_vertex_eval(c, S, d, ell, s_lo, h)
            tv0 = tv_at(c, S, d, ell, s_lo)
            if v_min != v_min:
                continue
            v_ub = tv0 + v_min
            if v_ub < c_target:
                continue  # not even potentially certifiable on this window
            shor_lb, _ = cell_cert_shor(c, S, d, c_target, W, solver='MOSEK', tol=1e-9)
            if shor_lb < c_target - 1e-9:
                # Found a candidate!
                candidates.append({
                    'd': d, 'S': S, 'c_target': c_target,
                    'c': c.tolist(), 'W': list(W),
                    'shor_lb': float(shor_lb),
                    'v_ub': float(v_ub),
                    'tri_W': list(tri['W']),
                    'gap_v_to_target': float(v_ub - c_target),
                })
                if len(candidates) >= 30:
                    break
        if len(candidates) >= 30:
            break
        if k % 200 == 0 and k > 0:
            print(f'    progress {k}/{len(hard)} cands={len(candidates)} t={time.time()-t0:.0f}s', flush=True)

    print(f'  Rescue candidates: {len(candidates)} (t={time.time()-t0:.0f}s)', flush=True)
    if candidates:
        for cand in candidates[:5]:
            print(f'    {cand}', flush=True)
    all_candidates.extend(candidates)

print(f'\n=== Total rescue candidates: {len(all_candidates)} ===\n', flush=True)

# Now run L2 and L3 on each candidate
results = []
for k, cand in enumerate(all_candidates):
    d_, S_, c_target_ = cand['d'], cand['S'], cand['c_target']
    c = np.asarray(cand['c'])
    W = tuple(cand['W'])

    t1 = time.time()
    l2_lb, l2_status = cell_cert_lasserre2(c, S_, d_, c_target_, W, solver='MOSEK', tol=1e-9)
    dt_l2 = time.time() - t1

    t2 = time.time()
    l3_lb, l3_status = cell_cert_lasserre3(c, S_, d_, c_target_, W, solver='MOSEK', tol=1e-9)
    dt_l3 = time.time() - t2

    l2_cert = l2_lb >= c_target_ - 1e-9
    l3_cert = l3_lb >= c_target_ - 1e-9

    results.append({**cand,
                    'l2_lb': float(l2_lb), 'l3_lb': float(l3_lb),
                    'l2_cert': bool(l2_cert), 'l3_cert': bool(l3_cert),
                    't_l2_s': dt_l2, 't_l3_s': dt_l3})

    print(f'  [{k:2d}] d={d_} S={S_} c={cand["c"]} W={W} '
          f'shor={cand["shor_lb"]:.5f} L2={l2_lb:.5f}({"C" if l2_cert else "f"}) '
          f'L3={l3_lb:.5f}({"C" if l3_cert else "f"}) v_ub={cand["v_ub"]:.5f} '
          f't_L2={dt_l2*1000:.0f}/t_L3={dt_l3*1000:.0f}ms', flush=True)

n_l2_rescue = sum(1 for r in results if r['l2_cert'])
n_l3_rescue = sum(1 for r in results if r['l3_cert'])
print(f'\n=== Final summary ===')
print(f'Rescue candidates: {len(results)}')
print(f'L2 rescues : {n_l2_rescue}')
print(f'L3 rescues : {n_l3_rescue}')

with open('_l3_true_rescue_search.json', 'w') as fp:
    json.dump({'configs': configs, 'n_candidates': len(results),
               'n_l2_rescue': n_l2_rescue, 'n_l3_rescue': n_l3_rescue,
               'results': results}, fp, indent=2)
print(f'\nWrote _l3_true_rescue_search.json')
