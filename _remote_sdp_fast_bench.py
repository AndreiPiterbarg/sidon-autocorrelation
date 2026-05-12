"""Benchmark the FAST per-box dual-Farkas SDP cert at d=22.

Sweeps configurations: baseline (full PSD, no early-stop, tol=1e-7),
early-stop only, early-stop+loose-tol, all-three with selective windows
(K in {0, 16, 32, 64, 128, 256}).

Test box: peaked centroid (mu = 1/d, mu[0]*=0.5, mu[-1]*=1.5, normalize),
radius=5e-3.

Env vars:
    TB_D       (default 22)
    TB_T       (default 1.281)
    TB_TIME    (default 1500 s outer cap)
    TB_THREADS (default 48)
    TB_KS      (default '0,16,32,64,128,256,1000')  K sweep for selective.
    TB_BASELINE (default '1' = include the baseline run)
"""
import sys, time, os
import numpy as np
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, bound_sdp_escalation_lb_float,
)
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast, bound_sdp_escalation_lb_float_fast,
)

d = int(os.environ.get('TB_D', '22'))
target = float(os.environ.get('TB_T', '1.281'))
time_lim = float(os.environ.get('TB_TIME', '1500'))
n_threads = int(os.environ.get('TB_THREADS', '48'))
ks = [int(x) for x in os.environ.get('TB_KS', '0,16,32,64,128,256,1000').split(',')]
include_baseline = os.environ.get('TB_BASELINE', '1') == '1'

print(f"d={d} target={target} time_lim={time_lim}s n_threads={n_threads}")
print(f"Selective-window K sweep: {ks}")

windows = build_windows(d)
mu = np.full(d, 1.0/d); mu[0]*=0.5; mu[-1]*=1.5; mu/=mu.sum()
if mu[0] > mu[-1]: mu = mu[::-1]
radius = 5e-3
lo = np.maximum(mu-radius, 0.0); hi = np.minimum(mu+radius, 1.0)
print(f"Box: mu_min={mu.min():.4e}, mu_max={mu.max():.4e}, radius={radius}")
print(f"     sum(lo)={lo.sum():.4f}, sum(hi)={hi.sum():.4f}")
print(f"n_windows={len(windows)}")

# ----------------------------------------------------------------
# Baseline: existing full-PSD code, original (1e-7) tolerances, no early stop.
# (We have to override the new defaults to recover the baseline.)
# ----------------------------------------------------------------
if include_baseline:
    print("\n=== BASELINE (full PSD, no early-stop, tol=1e-7) ===")
    cache0 = build_sdp_escalation_cache(d, windows, target=target)
    t0 = time.time()
    res = bound_sdp_escalation_lb_float(
        lo, hi, windows, d, cache=cache0, target=target,
        time_limit_s=time_lim, n_threads=n_threads,
        tol_feas=1e-7, tol_gap_rel=1e-7, tol_gap_abs=1e-7,
        early_stop=False,
    )
    dt = time.time() - t0
    v = res.get('verdict')
    lam = res.get('lambda_star', float('nan'))
    print(f"  baseline: solve={dt:7.2f}s  verdict={v}  lambda*={lam:.4f}",
          flush=True)

    # + early stop only
    print("\n=== EARLY-STOP only (full PSD, tol=1e-7) ===")
    cache0 = build_sdp_escalation_cache(d, windows, target=target)
    t0 = time.time()
    res = bound_sdp_escalation_lb_float(
        lo, hi, windows, d, cache=cache0, target=target,
        time_limit_s=time_lim, n_threads=n_threads,
        tol_feas=1e-7, tol_gap_rel=1e-7, tol_gap_abs=1e-7,
        early_stop=True,
    )
    dt = time.time() - t0
    v = res.get('verdict')
    lam = res.get('lambda_star', float('nan'))
    es = res.get('early_stop', {})
    print(f"  early-stop: solve={dt:7.2f}s  verdict={v}  lambda*={lam:.4f}  "
          f"early={es.get('triggered', False)} iter={es.get('iter', '-')}",
          flush=True)

    # + early stop + loose tol
    print("\n=== EARLY-STOP + LOOSE TOL (full PSD, tol=1e-5) ===")
    cache0 = build_sdp_escalation_cache(d, windows, target=target)
    t0 = time.time()
    res = bound_sdp_escalation_lb_float(
        lo, hi, windows, d, cache=cache0, target=target,
        time_limit_s=time_lim, n_threads=n_threads,
        tol_feas=1e-5, tol_gap_rel=1e-5, tol_gap_abs=1e-5,
        early_stop=True,
    )
    dt = time.time() - t0
    v = res.get('verdict')
    lam = res.get('lambda_star', float('nan'))
    es = res.get('early_stop', {})
    print(f"  early-stop+loose-tol: solve={dt:7.2f}s  verdict={v}  "
          f"lambda*={lam:.4f}  early={es.get('triggered', False)} "
          f"iter={es.get('iter', '-')}", flush=True)

# ----------------------------------------------------------------
# FAST variant: selective window cones + early-stop + loose tol.
# Sweep K ∈ {0, 16, 32, 64, 128, 256, 1000}.
# ----------------------------------------------------------------
print("\n=== FAST (selective windows + early-stop + tol=1e-5) ===")
cache_fast = build_sdp_escalation_cache_fast(d, windows, target=target)

for K in ks:
    print(f"\n-- K={K} --", flush=True)
    t0 = time.time()
    res = bound_sdp_escalation_lb_float_fast(
        lo, hi, windows, d, cache=cache_fast, target=target,
        n_window_psd_cones=K,
        early_stop=True,
        tol=1e-5, max_iter=80,
        n_threads=n_threads,
        time_limit_s=time_lim,
        verbose=False,
    )
    dt = time.time() - t0
    v = res.get('verdict')
    lam = res.get('lambda_star', float('nan'))
    es = res.get('early_stop', {})
    print(f"  K={K:>4}  total={dt:7.2f}s  "
          f"(lp={res.get('lp_pre_s', 0):.2f} build={res.get('build_s', 0):.2f} "
          f"solve={res.get('solve_s', 0):.2f}) "
          f"verdict={v}  lambda*={lam:.4f}  "
          f"n_psd={res.get('n_psd_actual', '-')} "
          f"n_lin={res.get('n_lin_actual', '-')} "
          f"early={es.get('triggered', False)} iter={es.get('iter', '-')}",
          flush=True)
