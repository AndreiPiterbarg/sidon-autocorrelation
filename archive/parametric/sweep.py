"""Meta-script sweeping (L, N) for the parametric dual bound.

Usage
-----
    from parametric.sweep import sweep_grid, sweep_to_csv

    results = sweep_grid(Ls=[1, 3, 5, 7, 9, 11], Ns=[1, 2, 4, 6, 8], solver="CLARABEL")
    sweep_to_csv(results, "data/parametric_sweep.csv")

Each entry has (L, N, bound, status, solver_time).
"""
from __future__ import annotations

import time
from typing import Iterable, List, Optional

import numpy as np

from .outer_sdp import solve_outer_sdp


def sweep_grid(
    Ls: Iterable[int] = (1, 3, 5, 7, 9),
    Ns: Iterable[int] = (1, 2, 3, 4, 5),
    solver: str = "CLARABEL",
    verbose: bool = False,
    require_2N_ge_Lm1: bool = True,
) -> List[dict]:
    """Run outer SDP on a grid of (L, N).  Skip infeasible (2N < L - 1) pairs."""
    out: List[dict] = []
    for L in Ls:
        if L % 2 == 0:
            if verbose:
                print(f"[sweep] L={L} skipped (must be odd).")
            continue
        for N in Ns:
            if require_2N_ge_Lm1 and 2 * N < L - 1:
                continue
            t0 = time.time()
            try:
                res = solve_outer_sdp(L=L, N=N, solver=solver, verbose=False)
                bound = res["bound"]
                status = res["status"]
            except Exception as exc:  # pragma: no cover
                bound = None
                status = f"EXC: {exc}"
            elapsed = time.time() - t0
            row = {"L": L, "N": N, "bound": bound, "status": status, "time_s": elapsed}
            out.append(row)
            if verbose:
                print(f"[sweep] L={L:2d} N={N:2d}  bound={bound!r}  status={status}  time={elapsed:.2f}s")
    return out


def sweep_to_csv(rows: List[dict], path: str) -> None:
    """Write sweep results to CSV."""
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def extrapolate_limit(rows: List[dict]) -> Optional[float]:
    """Fit a simple trend to bound-vs-(L, N) to guess the limit.

    Uses the largest-N row for each L, then fits bound ~ bound_inf - a / L^p.

    Returns the extrapolated bound_inf if fit succeeds, else None.
    """
    # Largest-N per L.
    per_L = {}
    for r in rows:
        if r["bound"] is None:
            continue
        key = r["L"]
        if key not in per_L or r["N"] > per_L[key]["N"]:
            per_L[key] = r
    if len(per_L) < 3:
        return None
    Ls = np.array(sorted(per_L.keys()), dtype=float)
    bs = np.array([per_L[int(L)]["bound"] for L in Ls])
    # Fit b = b_inf - a * L^(-p).  Log-space fit.
    # Use grid of p values and pick one minimizing residuals.
    best = None
    for p in np.linspace(0.2, 3.0, 29):
        # Linear regression: b = b_inf - a * L^(-p).  X = L^(-p), Y = b.
        X = Ls ** (-p)
        A = np.vstack([X, np.ones_like(X)]).T
        coef, res, *_ = np.linalg.lstsq(A, bs, rcond=None)
        a, b_inf = coef
        # a could be negative, that's fine (saturating from below).
        resid = bs - (b_inf - (-a) * X)
        sse = float(np.sum(resid * resid)) if res.size == 0 else float(res[0])
        if best is None or sse < best[0]:
            best = (sse, p, a, b_inf)
    return best[3] if best else None
