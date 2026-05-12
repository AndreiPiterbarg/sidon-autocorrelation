"""Pared-down d=22 Z/2-only test — measures Z/2 SDP wall time on a single box.

Goal: get a fast measurement of Z/2 solve time at d=22 to compare to the
known full-SDP timeout (>25min at 48 threads). If Z/2 finishes in
reasonable time, we win.
"""
import sys, time, os
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)

from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation_z2 import (
    build_sdp_escalation_cache_z2, bound_sdp_escalation_z2_lb_float,
    is_box_sigma_symmetric,
)


d = int(os.environ.get('TB_D', '22'))
target = float(os.environ.get('TB_T', '1.281'))
time_lim = float(os.environ.get('TB_TIME', '1800'))
n_thr = int(os.environ.get('TB_THREADS', '48'))
radius = float(os.environ.get('TB_R', '5e-3'))
mode = os.environ.get('TB_MODE', 'z2_sym')   # z2_sym | full_sym | z2_force_asym | full_asym

print(f"=== d={d} mode={mode} threads={n_thr} radius={radius} target={target} ===")
windows = build_windows(d)

# Build appropriate cache
if mode.startswith('z2'):
    print("Building Z/2 cache (full + Z/2)...", flush=True)
    t0 = time.time()
    cache = build_sdp_escalation_cache_z2(d, windows, target=target)
    print(f"  cache build: {time.time()-t0:.1f}s")
    print(f"  n_loc full={cache['n_loc_full']} z2={cache['n_loc_z2']}")
    print(f"  n_win full={cache['n_win_full']} z2={cache['n_win_z2']}")
    print(f"  n_basis full={cache['n_basis_full']} sym={cache['n_basis_sym_z2']} anti={cache['n_basis_anti_z2']}")
    print(f"  full bar_entries={cache['info'].get('n_bar_entries', 'NA'):,}")
    print(f"  Z/2  bar_entries={cache['info_z2'].get('n_bar_entries', 'NA'):,}")
    solve_fn = bound_sdp_escalation_z2_lb_float
else:
    from interval_bnb.bound_sdp_escalation import build_sdp_escalation_cache, bound_sdp_escalation_lb_float
    print("Building full cache...", flush=True)
    t0 = time.time()
    cache = build_sdp_escalation_cache(d, windows, target=target)
    print(f"  cache build: {time.time()-t0:.1f}s")
    solve_fn = bound_sdp_escalation_lb_float

# Build box: use mode suffix to decide.
if mode.endswith('_asym'):
    rng = np.random.default_rng(42)
    mu = rng.dirichlet(np.full(d, 2.0))
    print(f"Box: random Dirichlet (asymmetric)")
else:
    mu = np.full(d, 1.0/d)
    print("Box: uniform sym mu = 1/d (sigma-symmetric)")
lo = np.maximum(mu - radius, 0.0); hi = np.minimum(mu + radius, 1.0)
print(f"  sigma_sym? {is_box_sigma_symmetric(lo, hi, tol=1e-12)}")

# Solve
print(f"\nSolving with n_threads={n_thr}, time_limit={time_lim}s...")
t0 = time.time()
extra = {}
if mode == 'z2_force_asym':
    extra['force_z2'] = True
elif mode == 'full_sym' or mode == 'full_asym':
    pass  # full_solver doesn't take force flags
res = solve_fn(lo, hi, windows, d, cache=cache, target=target,
                time_limit_s=time_lim, n_threads=n_thr, **extra)
dt = time.time() - t0
print(f"\nRESULT: solve={dt:.2f}s verdict={res.get('verdict')} "
      f"lambda={res.get('lambda_star', float('nan')):.6f} "
      f"used_z2={res.get('used_z2', 'N/A')}")
print("=== Done ===")
