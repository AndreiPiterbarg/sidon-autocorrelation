"""Z/2 vs full per-box dual-Farkas SDP benchmark at d=22.

Tests:
  A. Symmetric box (uniform mu_sym = 1/d): Z/2 vs full timing.
  B. Asymmetric box (mu_star_d22 if available, else perturbed uniform):
     auto-fallback path (uses full); also compare force_z2.
  C. Symmetric box but using force_full path: control.
  D. Random asymmetric Dirichlet: tests broader sample of boxes.

Reports:
  solve_time, verdict, lambda_star, used_z2 for each (config x threads).
"""
import sys, time, os
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)

from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, bound_sdp_escalation_lb_float,
)
from interval_bnb.bound_sdp_escalation_z2 import (
    build_sdp_escalation_cache_z2, bound_sdp_escalation_z2_lb_float,
    is_box_sigma_symmetric,
)


d = int(os.environ.get('TB_D', '22'))
target = float(os.environ.get('TB_T', '1.281'))
time_lim = float(os.environ.get('TB_TIME', '900'))
threads_list = [int(x) for x in os.environ.get('TB_THREADS', '48').split(',')]
radius = float(os.environ.get('TB_R', '5e-3'))

print(f"=== Z/2 vs Full per-box SDP bench: d={d} target={target} radius={radius} ===")
windows = build_windows(d)

# Build caches
print("Building full cache...", flush=True)
t0 = time.time()
cache_full = build_sdp_escalation_cache(d, windows, target=target)
print(f"  full cache built in {time.time()-t0:.1f}s")

print("Building Z/2 cache...", flush=True)
t0 = time.time()
cache_z2 = build_sdp_escalation_cache_z2(d, windows, target=target)
print(f"  Z/2 cache built in {time.time()-t0:.1f}s")
print(f"  n_loc full={cache_z2['n_loc_full']} z2={cache_z2['n_loc_z2']}")
print(f"  n_win full={cache_z2['n_win_full']} z2={cache_z2['n_win_z2']}")
print(f"  n_basis full={cache_z2['n_basis_full']} sym={cache_z2['n_basis_sym_z2']} anti={cache_z2['n_basis_anti_z2']}")
print(f"  full info: n_bar_entries={cache_z2['info'].get('n_bar_entries', 'N/A'):,}")
print(f"  Z/2  info: n_bar_entries={cache_z2['info_z2'].get('n_bar_entries', 'N/A'):,}")

# Test boxes
boxes = []

# (A) Symmetric uniform box.
mu_sym = np.full(d, 1.0/d)
lo_A = np.maximum(mu_sym - radius, 0.0); hi_A = np.minimum(mu_sym + radius, 1.0)
boxes.append(('A_uniform_sym', lo_A, hi_A))

# (B) mu_star if available, else perturbed uniform asymmetric.
mu_star_path = os.path.expanduser(f'~/sidon/mu_star_d{d}.npz')
if os.path.exists(mu_star_path):
    mu_star = np.load(mu_star_path)['mu']
    lo_B = np.maximum(mu_star - radius, 0.0); hi_B = np.minimum(mu_star + radius, 1.0)
    boxes.append(('B_mustar_asym', lo_B, hi_B))
else:
    print(f"  (no {mu_star_path})")
    mu = np.full(d, 1.0/d); mu[0] *= 0.5; mu[-1] *= 1.5; mu /= mu.sum()
    if mu[0] > mu[-1]: mu = mu[::-1]
    lo_B = np.maximum(mu - radius, 0.0); hi_B = np.minimum(mu + radius, 1.0)
    boxes.append(('B_perturb_asym', lo_B, hi_B))

# (C) Symmetric mu_sym box (sigma-symmetric construction with perturbation).
rng = np.random.default_rng(42)
mu_sym2 = rng.dirichlet(np.full(d, 2.0))
mu_sym2 = (mu_sym2 + mu_sym2[::-1]) / 2  # force sigma-symmetric
lo_C = np.maximum(mu_sym2 - radius, 0.0); hi_C = np.minimum(mu_sym2 + radius, 1.0)
boxes.append(('C_dirichlet_sym', lo_C, hi_C))

# (D) Random asymmetric Dirichlet box.
mu_D = rng.dirichlet(np.full(d, 2.0))
lo_D = np.maximum(mu_D - radius, 0.0); hi_D = np.minimum(mu_D + radius, 1.0)
boxes.append(('D_dirichlet_asym', lo_D, hi_D))


for n_thr in threads_list:
    print(f"\n--- n_threads = {n_thr} ---")
    for name, lo, hi in boxes:
        is_sym = is_box_sigma_symmetric(lo, hi, tol=1e-12)
        print(f"\n  Box {name}: sigma_sym={is_sym}")
        # Run full
        t0 = time.time()
        res_full = bound_sdp_escalation_lb_float(
            lo, hi, windows, d, cache=cache_full, target=target,
            time_limit_s=time_lim, n_threads=n_thr,
        )
        dt_full = time.time() - t0
        v_full = res_full.get('verdict')
        l_full = res_full.get('lambda_star', float('nan'))
        print(f"    FULL:        solve={dt_full:7.2f}s verdict={v_full} lambda={l_full:.6f}")

        # Run Z/2 (auto-dispatch)
        t0 = time.time()
        res_z2 = bound_sdp_escalation_z2_lb_float(
            lo, hi, windows, d, cache=cache_z2, target=target,
            time_limit_s=time_lim, n_threads=n_thr,
        )
        dt_z2 = time.time() - t0
        v_z2 = res_z2.get('verdict')
        l_z2 = res_z2.get('lambda_star', float('nan'))
        used_z2 = res_z2.get('used_z2')
        print(f"    Z/2 auto:    solve={dt_z2:7.2f}s verdict={v_z2} lambda={l_z2:.6f} used_z2={used_z2}")

        # Force Z/2 (sound-but-tighter on asymmetric)
        if not is_sym:
            t0 = time.time()
            res_z2f = bound_sdp_escalation_z2_lb_float(
                lo, hi, windows, d, cache=cache_z2, target=target,
                time_limit_s=time_lim, n_threads=n_thr, force_z2=True,
            )
            dt_z2f = time.time() - t0
            v_z2f = res_z2f.get('verdict')
            l_z2f = res_z2f.get('lambda_star', float('nan'))
            print(f"    Z/2 forced:  solve={dt_z2f:7.2f}s verdict={v_z2f} lambda={l_z2f:.6f}")

print("\n=== Done ===")
