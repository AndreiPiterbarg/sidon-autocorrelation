"""Tiny probe: measure ShorSDPTemplate_v6 compile time vs solve time at d=8.

Goal: confirm caching dominates solve cost (compile is one-shot), and
quantify per-solve cost for d=4/6/8.
"""
from __future__ import annotations
import os, sys, time, logging, warnings
logging.getLogger('cvxpy').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v6 as v6


def probe(d, S, c_target=1.281, n_solves=8):
    windows = v3.build_all_windows(d)
    bundle = v6.get_bundle(windows)
    # Build a simple cell
    c = np.full(d, S / d, dtype=np.float64)
    cell = v3.Cell.from_integer_composition(c, S)
    cache = v3.CellCache.build(cell)

    # Compile (template construction)
    t0 = time.perf_counter()
    tmpl = v6.ShorSDPTemplate_v6(d)
    t_compile_ms = (time.perf_counter() - t0) * 1000

    # First solve (cold canonicalization)
    W = windows[0]
    mu = cache.mu_star
    margin = W.Q_coef * float(mu @ W.A @ mu) - c_target
    grad = W.grad_coef * (W.A @ mu)
    Q = W.Q_coef * W.A

    t0 = time.perf_counter()
    tmpl.solve(cache.lo_eps, cache.hi_eps, mu, Q, grad, margin)
    t_first_ms = (time.perf_counter() - t0) * 1000

    # Subsequent solves
    solve_times = []
    for k in range(n_solves):
        W = windows[k % len(windows)]
        margin = W.Q_coef * float(mu @ W.A @ mu) - c_target
        grad = W.grad_coef * (W.A @ mu)
        Q = W.Q_coef * W.A
        t0 = time.perf_counter()
        tmpl.solve(cache.lo_eps, cache.hi_eps, mu, Q, grad, margin)
        solve_times.append((time.perf_counter() - t0) * 1000)
    return {
        'd': d, 'S': S, 'n_windows': len(windows),
        't_compile_ms': t_compile_ms,
        't_first_solve_ms': t_first_ms,
        't_subsequent_ms_mean': float(np.mean(solve_times)),
        't_subsequent_ms_min': float(np.min(solve_times)),
        't_subsequent_ms_max': float(np.max(solve_times)),
    }


if __name__ == '__main__':
    for d, S in [(4, 80), (6, 30), (8, 16)]:
        r = probe(d, S)
        print(r)
