"""Test L2 and L3 specifically on cells where Shor is loose (Shor < vertex_ub).

These are the cells where the Lasserre hierarchy could in principle help:
the QP min is genuinely larger than what Shor's PSD relaxation reports.
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

d, S, c_target = 4, 80, 1.281
print(f'Loading hard cells d={d} S={S} c_target={c_target}...', flush=True)
hard, _, _, _ = find_hard_cells(d=d, S=S, c_target=c_target, max_eval=10**5)
print(f'  {len(hard)} hard cells', flush=True)
hard.sort(key=lambda kv: -kv[1]['net'])

# Find the Shor-loose cells (Shor < vertex_ub, so SDP order >1 might rescue)
print('Identifying Shor-loose cells...', flush=True)
loose_cells = []
t0 = time.time()
for k, (c, tri) in enumerate(hard):
    ell, s_lo = tri['W']
    h = 1/(2*S)
    v_min = qp_min_vertex_eval(c, S, d, ell, s_lo, h)
    tv0 = tv_at(c, S, d, ell, s_lo)
    v_ub = tv0 + v_min
    shor_lb, _ = cell_cert_shor(c, S, d, c_target, tri['W'], solver='MOSEK', tol=1e-9)
    if v_ub - shor_lb > 1e-7:
        loose_cells.append({'c': c, 'tri': tri, 'shor_lb': shor_lb, 'v_ub': v_ub})
    if len(loose_cells) >= 30:
        print(f'  Got 30 loose cells (scanned {k+1}/{len(hard)}, t={time.time()-t0:.0f}s)', flush=True)
        break

print(f'Found {len(loose_cells)} Shor-loose cells', flush=True)

# Run L2 and L3 on each loose cell
results = []
n_l2_strict = 0
n_l3_strict_vs_l2 = 0
n_l3_strict_vs_shor = 0
n_l3_match_vertex = 0
times_l2 = []
times_l3 = []

for k, cell in enumerate(loose_cells):
    c = cell['c']
    W = cell['tri']['W']
    shor_lb = cell['shor_lb']
    v_ub = cell['v_ub']

    t1 = time.time()
    l2_lb, l2_status = cell_cert_lasserre2(c, S, d, c_target, W, solver='MOSEK', tol=1e-9)
    dt_l2 = time.time() - t1
    times_l2.append(dt_l2)

    t2 = time.time()
    l3_lb, l3_status = cell_cert_lasserre3(c, S, d, c_target, W, solver='MOSEK', tol=1e-9)
    dt_l3 = time.time() - t2
    times_l3.append(dt_l3)

    if l2_lb > shor_lb + 1e-7:
        n_l2_strict += 1
    if l3_lb > l2_lb + 1e-7:
        n_l3_strict_vs_l2 += 1
    if l3_lb > shor_lb + 1e-7:
        n_l3_strict_vs_shor += 1
    if abs(l3_lb - v_ub) < 1e-5:
        n_l3_match_vertex += 1

    results.append({
        'k': k, 'c': c.tolist(), 'W': list(W),
        'tri_net': float(cell['tri']['net']),
        'shor_lb': float(shor_lb),
        'l2_lb': float(l2_lb) if l2_lb != float('-inf') else None,
        'l3_lb': float(l3_lb) if l3_lb != float('-inf') else None,
        'vertex_ub': float(v_ub),
        'gap_shor_to_vert': float(v_ub - shor_lb),
        'gap_l2_to_shor': float(l2_lb - shor_lb) if l2_lb != float('-inf') else None,
        'gap_l3_to_l2': float(l3_lb - l2_lb) if l3_lb != float('-inf') and l2_lb != float('-inf') else None,
        'gap_l3_to_vert': float(v_ub - l3_lb) if l3_lb != float('-inf') else None,
        't_l2_s': dt_l2, 't_l3_s': dt_l3,
        'shor_cert': bool(shor_lb >= c_target - 1e-9),
        'l2_cert': bool(l2_lb >= c_target - 1e-9),
        'l3_cert': bool(l3_lb >= c_target - 1e-9),
    })

    if k < 12:
        print(f'  [{k:2d}] c={c.tolist()} shor={shor_lb:.6f} L2={l2_lb:.6f} L3={l3_lb:.6f} '
              f'vert={v_ub:.6f} L2-Sh={l2_lb-shor_lb:+.2e} L3-L2={l3_lb-l2_lb:+.2e} '
              f'L3-vert={l3_lb-v_ub:+.2e} T_L2={dt_l2*1000:.0f}/T_L3={dt_l3*1000:.0f}ms', flush=True)

print('\n--- Summary on Shor-loose cells ---')
print(f'Cells tested: {len(loose_cells)}')
print(f'L2 strictly > Shor    : {n_l2_strict}')
print(f'L3 strictly > L2      : {n_l3_strict_vs_l2}')
print(f'L3 strictly > Shor    : {n_l3_strict_vs_shor}')
print(f'L3 matches vertex_ub  : {n_l3_match_vertex}  (out of {len(loose_cells)})')
print(f'L2 time/cell: med={1000*np.median(times_l2):.0f}ms p95={1000*np.percentile(times_l2,95):.0f}ms')
print(f'L3 time/cell: med={1000*np.median(times_l3):.0f}ms p95={1000*np.percentile(times_l3,95):.0f}ms')

with open('_l3_rescue_test_results.json', 'w') as fp:
    json.dump({
        'd': d, 'S': S, 'c_target': c_target,
        'n_loose_tested': len(loose_cells),
        'n_l2_strict': n_l2_strict,
        'n_l3_strict_vs_l2': n_l3_strict_vs_l2,
        'n_l3_strict_vs_shor': n_l3_strict_vs_shor,
        'n_l3_match_vertex': n_l3_match_vertex,
        't_l2_med_ms': float(1000*np.median(times_l2)),
        't_l3_med_ms': float(1000*np.median(times_l3)),
        'rows': results,
    }, fp, indent=2)
print('\nWrote _l3_rescue_test_results.json')
