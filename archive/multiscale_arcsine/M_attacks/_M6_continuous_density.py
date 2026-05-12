"""Agent M6 - Continuous Density Limit of multi-scale arcsine kernels.

Generalize the discrete K_hat(xi) = sum_i lambda_i J0(pi delta_i xi)^2
to the continuous form:
    K_hat(xi) = int_0^DELTA_MAX rho(delta') J0(pi delta' xi)^2 d delta'
where rho >= 0 with int rho = 1.

We discretize rho on a fine grid of delta' values, convert to (deltas, lambdas)
weights and call the existing pipeline (eval_at) from _K26_full_sweep_reopt.

Phases:
  1. Test parametric rho families (uniform, power, gaussian, two-bump,
     pseudo-arcsine, linear).
  2. Differential-evolution search over a 4-coefficient polynomial basis
     for rho on [0, DELTA_MAX].
  3. Compare to discrete 2-scale (M=1.29005) and report ρ-concentration.

Reuses K_hat_ms, solve_QP, K_2_quad, M_cert, eval_at from _K26_full_sweep_reopt.
"""
from __future__ import annotations

import json
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

from _K26_full_sweep_reopt import (
    K_hat_ms, solve_QP, K_2_quad, M_cert, eval_at, U, N_QP, DELTA as DELTA_MAX
)


# -----------------------------------------------------------------------
# Density discretization helpers
# -----------------------------------------------------------------------

N_DELTA_GRID = 100  # number of delta' nodes for the continuous discretization

# A uniform grid of delta' values strictly inside (0, DELTA_MAX].
# Avoid delta'=0 (it contributes a constant kernel and dominates the
# admissible cone trivially; the kernel J0(0)=1 means rho(0) gives a flat term.)
# We use mid-point rule to keep ints accurate.

def delta_grid():
    dp = DELTA_MAX / N_DELTA_GRID
    grid = np.linspace(dp / 2.0, DELTA_MAX - dp / 2.0, N_DELTA_GRID)
    return grid, dp


def normalize_rho(rho_vals, dp):
    """Make rho_vals nonneg and integrate to 1 against the mid-point rule."""
    rho_vals = np.maximum(rho_vals, 0.0)
    mass = rho_vals.sum() * dp
    if mass <= 0:
        return None
    return rho_vals / mass


def lambdas_from_rho(rho_vals, dp):
    """Convert rho values on grid to lambda weights with sum=1."""
    lam = rho_vals * dp
    s = lam.sum()
    if s <= 0:
        return None
    return lam / s


def eval_density(rho_vals):
    """Take rho values on the standard delta_grid, build (deltas, lambdas),
    call eval_at, and return result dict (or None on failure)."""
    grid, dp = delta_grid()
    rho_norm = normalize_rho(rho_vals, dp)
    if rho_norm is None:
        return None
    lambdas = lambdas_from_rho(rho_norm, dp)
    if lambdas is None:
        return None
    # Drop zero weights so QP weights stay positive and arrays compact.
    mask = lambdas > 1e-12
    deltas_eff = grid[mask]
    lambdas_eff = lambdas[mask]
    # Re-normalize lambdas after mask
    lambdas_eff = lambdas_eff / lambdas_eff.sum()
    r = eval_at(list(deltas_eff.astype(float)), list(lambdas_eff.astype(float)))
    return r


# -----------------------------------------------------------------------
# Parametric density families
# -----------------------------------------------------------------------

def rho_uniform(grid):
    return np.ones_like(grid)

def rho_power(grid, p):
    # rho ~ delta'^p
    return np.power(grid, p)

def rho_truncated_gaussian(grid, mu, sigma):
    return np.exp(-0.5 * ((grid - mu) / sigma) ** 2)

def rho_two_bump(grid, d1, d2, w1):
    # Discrete approximation: spikes at the two nearest grid nodes.
    rho = np.zeros_like(grid)
    i1 = int(np.clip(np.round(d1 / DELTA_MAX * N_DELTA_GRID - 0.5), 0, N_DELTA_GRID - 1))
    i2 = int(np.clip(np.round(d2 / DELTA_MAX * N_DELTA_GRID - 0.5), 0, N_DELTA_GRID - 1))
    rho[i1] += w1
    rho[i2] += (1.0 - w1)
    return rho

def rho_pseudo_arcsine(grid, eps=1e-3):
    # rho ~ 1 / sqrt(d (DELTA_MAX - d)), regularized
    arg = np.clip(grid * (DELTA_MAX - grid), eps * DELTA_MAX * DELTA_MAX, None)
    return 1.0 / np.sqrt(arg)

def rho_linear(grid, a, b):
    # rho ~ a + b * delta'/DELTA_MAX  (clipped >=0)
    return np.maximum(a + b * grid / DELTA_MAX, 0.0)


# -----------------------------------------------------------------------
# Phase 1: parametric sweep
# -----------------------------------------------------------------------

def phase1():
    grid, dp = delta_grid()
    runs = []

    def record(label, rho):
        t0 = time.time()
        r = eval_density(rho)
        dt = time.time() - t0
        Mc = None if r is None else r.get("M_cert")
        # rho concentration summary
        rho_n = normalize_rho(rho, dp)
        if rho_n is not None:
            cum = np.cumsum(rho_n * dp)
            # median (cum=0.5) and 25%/75% quantiles
            q25 = float(grid[np.searchsorted(cum, 0.25)])
            q50 = float(grid[np.searchsorted(cum, 0.5)])
            q75 = float(grid[np.searchsorted(cum, 0.75)])
            top_idx = int(np.argmax(rho_n))
            mode = float(grid[top_idx])
        else:
            q25 = q50 = q75 = mode = None
        runs.append({
            "label": label,
            "M_cert": (float(Mc) if Mc is not None else None),
            "k_1": (float(r.get("k_1")) if r else None),
            "K_2": (float(r.get("K_2")) if r else None),
            "min_G": (float(r.get("min_G")) if r else None),
            "S_1": (float(r.get("S_1")) if r else None),
            "rho_q25": q25, "rho_q50": q50, "rho_q75": q75, "rho_mode": mode,
            "time_s": dt,
        })
        Mc_str = (f"{Mc:.5f}" if Mc is not None else "FAIL")
        print(f"  {label:<35s} M={Mc_str}  mode~{mode}  q25/50/75={q25}/{q50}/{q75}  ({dt:.1f}s)")
        return Mc

    print("\nPhase 1: parametric density families")
    print("-" * 80)

    # 1. Uniform
    record("uniform", rho_uniform(grid))

    # 2. Power
    for p in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]:
        record(f"power_p={p:+.2f}", rho_power(grid, p))

    # 3. Truncated Gaussian
    for mu in [0.045, 0.07, 0.10, 0.115, 0.138]:
        for sigma in [0.01, 0.02, 0.03, 0.05]:
            record(f"gauss_mu={mu:.3f}_sig={sigma:.3f}",
                   rho_truncated_gaussian(grid, mu, sigma))

    # 4. Two-bump (sanity check: this should recover 2-scale ~1.29005)
    for (d1, d2, w1) in [
        (0.138, 0.045, 0.85),
        (0.138, 0.045, 0.90),
        (0.138, 0.050, 0.92),
        (0.138, 0.030, 0.85),
        (0.130, 0.050, 0.85),
    ]:
        record(f"twobump_d1={d1:.3f}_d2={d2:.3f}_w1={w1:.2f}",
               rho_two_bump(grid, d1, d2, w1))

    # 5. Pseudo-arcsine
    record("pseudo_arcsine", rho_pseudo_arcsine(grid))

    # 6. Linear (a + b * d/DELTA_MAX), positivity ensured by clip
    for (a, b) in [(1.0, 0.0), (1.0, 1.0), (1.0, -0.5), (1.0, 2.0),
                   (0.5, 2.0), (2.0, -1.0), (0.1, 3.0)]:
        record(f"linear_a={a:.2f}_b={b:+.2f}", rho_linear(grid, a, b))

    return runs


# -----------------------------------------------------------------------
# Phase 2: DE over 4-coefficient polynomial basis
# -----------------------------------------------------------------------

# rho(d) = c0 + c1 (d/D) + c2 (d/D)^2 + c3 (d/D)^3, clipped to >=0, normalized.
# We optimize c0..c3 in a box, then post-clip and renormalize.

def rho_poly(grid, coeffs):
    t = grid / DELTA_MAX
    return coeffs[0] + coeffs[1] * t + coeffs[2] * t**2 + coeffs[3] * t**3


def neg_M_poly(coeffs):
    grid, _ = delta_grid()
    rho = np.maximum(rho_poly(grid, coeffs), 0.0)
    if rho.sum() < 1e-9:
        return 0.0
    r = eval_density(rho)
    if r is None or r.get("M_cert") is None:
        return 0.0
    return -float(r["M_cert"])


def phase2(maxiter=30, popsize=12, seed=1):
    print("\nPhase 2: DE over 4-coefficient polynomial basis")
    print("-" * 80)
    bounds = [(-2.0, 5.0)] * 4
    t0 = time.time()
    de_iters = {"i": 0, "best": np.inf}
    def cb(xk, convergence):
        de_iters["i"] += 1
        cur = neg_M_poly(xk)
        if cur < de_iters["best"]:
            de_iters["best"] = cur
            print(f"  DE iter ~{de_iters['i']}: M={-cur:.5f} coeffs={[round(float(x),3) for x in xk]}")
        return False
    res = differential_evolution(
        neg_M_poly,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=True,
        tol=1e-6,
        workers=1,
        callback=cb,
    )
    dt = time.time() - t0
    print(f"  DE done in {dt:.0f}s, fun={res.fun:.6f}, M_cert={-res.fun:.5f}")
    print(f"  coeffs: {res.x}")

    coeffs = res.x.tolist()
    grid, dp = delta_grid()
    rho_opt = np.maximum(rho_poly(grid, coeffs), 0.0)
    rho_norm = normalize_rho(rho_opt, dp)

    # Density concentration of optimum
    cum = np.cumsum(rho_norm * dp)
    q05 = float(grid[np.searchsorted(cum, 0.05)])
    q25 = float(grid[np.searchsorted(cum, 0.25)])
    q50 = float(grid[np.searchsorted(cum, 0.5)])
    q75 = float(grid[np.searchsorted(cum, 0.75)])
    q95 = float(grid[np.searchsorted(cum, 0.95)])
    top_idx = int(np.argmax(rho_norm))

    # Estimate number of "peaks" by counting local maxima above 50% of global max
    peaks = []
    for i in range(1, len(rho_norm) - 1):
        if rho_norm[i] > rho_norm[i - 1] and rho_norm[i] > rho_norm[i + 1]:
            if rho_norm[i] > 0.5 * rho_norm.max():
                peaks.append({"delta": float(grid[i]), "rho_val": float(rho_norm[i])})

    return {
        "coeffs": coeffs,
        "M_cert": -float(res.fun),
        "time_s": dt,
        "quantiles": {"q05": q05, "q25": q25, "q50": q50, "q75": q75, "q95": q95},
        "mode_delta": float(grid[top_idx]),
        "rho_values_on_grid": rho_norm.tolist(),
        "grid_delta": grid.tolist(),
        "n_peaks_geq_50pct_max": len(peaks),
        "peaks": peaks,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 80)
    print("Agent M6 - Continuous density limit of multi-scale arcsine kernel")
    print(f"DELTA_MAX={DELTA_MAX}, U={U}, N_QP={N_QP}, N_DELTA_GRID={N_DELTA_GRID}")
    print("=" * 80)

    # Baseline reference: discrete 2-scale 1.29005
    print("\nBaseline (discrete 2-scale, delta_1=0.138, delta_2=0.045, lam_1=0.85):")
    r_ref = eval_at([0.138, 0.045], [0.85, 0.15])
    print(f"  M_cert = {r_ref.get('M_cert')!r}")
    baseline_2scale = float(r_ref["M_cert"]) if r_ref.get("M_cert") else None

    phase1_runs = phase1()

    phase2_res = phase2(maxiter=25, popsize=10)

    # Phase 3 analysis
    finite = [r for r in phase1_runs if r["M_cert"] is not None]
    best_phase1 = max(finite, key=lambda r: r["M_cert"]) if finite else None

    summary = {
        "DELTA_MAX": DELTA_MAX,
        "N_DELTA_GRID": N_DELTA_GRID,
        "baseline_M_cert_MV": 1.27481,
        "baseline_M_cert_2scale": baseline_2scale,
        "phase1_best": best_phase1,
        "phase1_runs": phase1_runs,
        "phase2": phase2_res,
        "total_time_s": time.time() - t0,
    }
    summary["beats_2scale"] = (
        phase2_res["M_cert"] > (baseline_2scale or 0)
        or (best_phase1 and best_phase1["M_cert"] > (baseline_2scale or 0))
    )
    with open("_M6_continuous_density.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"2-scale baseline M_cert = {baseline_2scale}")
    if best_phase1:
        print(f"Phase 1 best M_cert = {best_phase1['M_cert']:.5f}  ({best_phase1['label']})")
    print(f"Phase 2 (DE poly)  M_cert = {phase2_res['M_cert']:.5f}")
    print(f"DE mode delta = {phase2_res['mode_delta']:.4f}")
    print(f"DE quantiles = {phase2_res['quantiles']}")
    print(f"DE n peaks >=50%max = {phase2_res['n_peaks_geq_50pct_max']}")
    print(f"Total time: {summary['total_time_s']:.0f}s")
    print("Wrote _M6_continuous_density.json")


if __name__ == "__main__":
    main()
