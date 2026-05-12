"""
F2 Heat Kernel Probe (extension): probe inf_{f admissible} max u(t,.)
as t -> 0+, to see whether it converges from below to inf max(f*f) = C_{1a}.

The smoothed problem at small t may be MORE TRACTABLE (analytic in f) than
the original nonsmooth problem, even though the LB at any t > 0 is strictly
weaker.  If at t = 1e-6, the random search inf is around 1.27 (slightly
below 1.2802), that's evidence the smoothed problem at very small t is a
near-tight relaxation, motivating an SDP attack.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

OUTDIR = Path(__file__).resolve().parent
LOG = OUTDIR / "run.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def build_grid(N: int, half_box: float):
    x = np.linspace(-half_box, half_box, N)
    dx = x[1] - x[0]
    return x, dx


def autoconv(f, dx):
    return np.correlate(f, f, mode="full") * dx


def heat_smooth(g, x_grid, t):
    if t <= 0.0:
        return g.copy()
    dx = x_grid[1] - x_grid[0]
    var = 2.0 * t
    sigma = np.sqrt(var)
    half_kernel_width = max(8.0 * sigma, 5.0 * dx)
    Nk = int(np.ceil(half_kernel_width / dx))
    k_grid = np.arange(-Nk, Nk + 1) * dx
    phi = np.exp(-k_grid ** 2 / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)
    phi /= np.sum(phi) * dx
    full = np.convolve(g, phi, mode="full") * dx
    return full[Nk : Nk + len(g)]


def search_smoothed(N, half_box, d, t, n_random=2000, n_local=2000, seed=0):
    """Random + local search for f piecewise-const on d bins of [-1/4,1/4]
    minimising max u(t,.)|_{|x|<=1/2}."""
    x, dx = build_grid(N, half_box)
    g_grid = (np.arange(2 * N - 1) - (N - 1)) * dx
    mask = np.abs(g_grid) <= 0.5
    bin_edges = np.linspace(-0.25, 0.25, d + 1)
    bin_width = 0.5 / d
    bin_masks = []
    for i in range(d):
        bin_masks.append((x >= bin_edges[i]) & (x < bin_edges[i + 1]))
    rng = np.random.default_rng(seed)

    def evaluate(a):
        a = np.clip(a, 0.0, None)
        s = np.sum(a) * bin_width
        if s <= 0:
            return np.inf, None
        a = a / s
        f = np.zeros_like(x)
        for i, mb in enumerate(bin_masks):
            f[mb] = a[i]
        ff = autoconv(f, dx)
        u = heat_smooth(ff, g_grid, t)
        return float(np.max(u[mask])), a

    best_max = np.inf
    best_a = None
    # Random init
    for trial in range(n_random):
        a = rng.uniform(0.0, 1.0, size=d)
        m, ar = evaluate(a)
        if m < best_max:
            best_max = m
            best_a = ar
    # Local refinement
    sigma = 0.05
    for step in range(n_local):
        a_try = best_a + rng.normal(0.0, sigma, size=d)
        m, ar = evaluate(a_try)
        if m < best_max:
            best_max = m
            best_a = ar
            sigma = max(0.001, sigma * 0.99)
        else:
            sigma = sigma * 1.001
        sigma = min(sigma, 0.1)
    return best_max, best_a


def main():
    log("=== F2 Heat Kernel Probe (extension: limit as t -> 0+) ===")
    t_start = time.perf_counter()

    N = 4097
    half_box = 1.0
    log(f"grid: N={N}, half_box={half_box}")

    # Range of t from very small to moderate
    t_grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2]
    d_grid = [10, 16, 24]

    results = {}
    for d in d_grid:
        log(f"--- d = {d} ---")
        per_t = {}
        for t in t_grid:
            tic = time.perf_counter()
            best_max, best_a = search_smoothed(
                N, half_box, d, t, n_random=400, n_local=600, seed=hash((d, t)) & 0xFFFFFFFF
            )
            toc = time.perf_counter()
            per_t[float(t)] = {"best_max": best_max, "elapsed": toc - tic}
            log(f"  d={d}, t={t:.1e}: best_max ~ {best_max:.6f}  ({toc-tic:.1f}s)")
        results[d] = per_t

    out = {
        "approach": "smoothed-problem inf as t -> 0+",
        "results": {str(d): per_t for d, per_t in results.items()},
        "elapsed_total": time.perf_counter() - t_start,
        "note": (
            "best_max[d, t] is an UPPER BOUND on inf_{f piecewise-const on d bins} "
            "max u(t, .)  on  [-1/2, 1/2].  This in turn UPPER-BOUNDS the true "
            "infimum at d = infinity.  If best_max is small (close to 1) for small t, "
            "this means the smoothed problem at t > 0 admits f's with arbitrarily "
            "low ||u(t,.)||_inf, which means heat-flow LB is WEAKER than 1.2802 by "
            "a margin that grows with t."
        ),
    }
    with open(OUTDIR / "results_limit.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log(f"=== done in {time.perf_counter()-t_start:.1f}s ===")
    return out


if __name__ == "__main__":
    main()
