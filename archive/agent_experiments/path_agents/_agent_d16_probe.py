"""Probe: M(d) = max_W TV_W(uniform μ=1/d) for d in {12,14,16,18,20}.

Also reports composition counts at d=16 for S=64 (n=1) and S=128 (n=2).
"""
from __future__ import annotations
import os, sys, math, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v4 as v4  # noqa: E402

THRESHOLD = 1.281


def M_uniform(d: int):
    """Return (M, best_W) where M = max_W (Q_coef * (1/d)^T A_W (1/d))."""
    windows = v4.build_all_windows(d)
    mu = np.full(d, 1.0 / d)
    best = -np.inf
    best_W = None
    for W in windows:
        v = W.Q_coef * float(mu @ W.A @ mu)
        if v > best:
            best = v
            best_W = (W.ell, W.s_lo)
    return best, best_W


def comp_count(S: int, d: int) -> int:
    """Number of integer compositions of S into d nonneg parts = C(S+d-1, d-1)."""
    return math.comb(S + d - 1, d - 1)


def main():
    results = {"threshold": THRESHOLD, "M_uniform_by_d": {}}
    for d in [12, 14, 16, 18, 20]:
        M, bestW = M_uniform(d)
        margin = M - THRESHOLD
        results["M_uniform_by_d"][str(d)] = {
            "M": M, "best_window_ell_s_lo": bestW, "margin_vs_1.281": margin,
        }
        print(f"d={d:2d}  M={M:.6f}  bestW(ell,s_lo)={bestW}  margin={margin:+.6f}")

    print()
    d = 16
    for n, S in [(1, 4 * d), (2, 8 * d)]:
        N = comp_count(S, d)
        results.setdefault("comp_counts", {})[f"d{d}_n{n}_S{S}"] = N
        print(f"d={d}  n={n}  S={S}  N_cells = C({S+d-1},{d-1}) = {N:,}")

    # Save
    out = os.path.join(_dir, "_agent_d16_probe.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out}")
    return results


if __name__ == "__main__":
    main()
