"""Quick scan of d=4 S=80 hard cells looking for any Shor looseness vs vertex_ub."""
import warnings, sys, time
warnings.filterwarnings('ignore')
import os
os.environ['CVXPY_VERBOSE'] = '0'
import numpy as np
from _coarse_L_bench import find_hard_cells, cell_cert_shor, qp_min_vertex_eval, tv_at

d, S, c_target = 4, 80, 1.281
print(f'Scanning d={d} S={S} c_target={c_target}...', flush=True)
t0 = time.time()
hard, _, _, _ = find_hard_cells(d=d, S=S, c_target=c_target, max_eval=10**5)
print(f'  {len(hard)} hard cells loaded in {time.time()-t0:.1f}s', flush=True)
hard.sort(key=lambda kv: -kv[1]['net'])  # nearest to 0 first

print('  Running Shor on each...', flush=True)
t0 = time.time()
shor_fail = 0
shor_loose = 0
shor_loose_examples = []
for k, (c, tri) in enumerate(hard):
    ell, s_lo = tri['W']
    h = 1/(2*S)
    v_min = qp_min_vertex_eval(c, S, d, ell, s_lo, h)
    tv0 = tv_at(c, S, d, ell, s_lo)
    v_ub = tv0 + v_min
    shor_lb, _ = cell_cert_shor(c, S, d, c_target, tri['W'], solver='MOSEK', tol=1e-9)
    if shor_lb < c_target - 1e-9:
        shor_fail += 1
    if v_ub - shor_lb > 1e-7:
        shor_loose += 1
        if len(shor_loose_examples) < 5:
            shor_loose_examples.append((c, tri, shor_lb, v_ub))
    if (k+1) % 100 == 0:
        print(f'    {k+1}/{len(hard)} t={time.time()-t0:.0f}s shor_fail={shor_fail} shor_loose={shor_loose}', flush=True)

print(f'Total: cells={len(hard)}, shor_fail={shor_fail}, shor_loose={shor_loose}, time={time.time()-t0:.0f}s')
for c, tri, slb, vub in shor_loose_examples:
    print(f'  loose: c={c.tolist()} W={tri["W"]} shor={slb:.7f} vert={vub:.7f}')
