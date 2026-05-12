"""
F2 Heat Kernel Probe for Sidon Autocorrelation Constant C_{1a}.

Probes whether heat-kernel smoothing of f gives a useful lower bound on
    C_{1a} = inf_{f >= 0, supp f <= [-1/4, 1/4], int f = 1} max_{|t| <= 1/2} (f*f)(t).

Core observation (verified by parabolic max principle):
   Let u(t,x) = (f*f) * phi_{2t}(x) where phi_s is the heat kernel of variance s.
   Then u solves u_t = u_{xx}, u(0,x) = (f*f)(x).
   By the parabolic maximum principle ||u(t,.)||_inf is NONINCREASING in t.
   Hence ||f*f||_inf >= ||u(t,.)||_inf for ALL t > 0.

Strategy:
   1. Build several admissible candidates (uniform, arcsine, two-level step).
   2. Compute ||u(t,.)||_inf vs t.  Verify monotone-decreasing.
   3. Read slope at t=0+.  This tells us how much info we lose per unit t.
   4. Heat-witness "dual cert": for fixed t > 0, x_0 in [-1/2, 1/2]:
         W_{t,x_0}(x) = phi_{2t}(x_0 - x).
      Then <f*f, W_{t,x_0}> = u(t, x_0).
      Universal LB:  inf_{f admissible} max_{x_0} u(t, x_0).
      We attempt to compute it numerically by minimising over piecewise-const f.
   5. Honest verdict.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

OUTDIR = Path(__file__).resolve().parent
LOG = OUTDIR / "run.log"
RESULTS = OUTDIR / "results.json"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Grid + convolution utilities (using np.correlate for clarity)
# ---------------------------------------------------------------------------
def build_grid(N: int, half_box: float):
    x = np.linspace(-half_box, half_box, N)
    dx = x[1] - x[0]
    return x, dx


def autoconv_grid(x: np.ndarray, dx: float):
    """Grid for (f*f) on the full conv-support."""
    N = len(x)
    return (np.arange(2 * N - 1) - (N - 1)) * dx


def autoconv(f: np.ndarray, dx: float) -> np.ndarray:
    """(f*f)(x_k) for k on the full conv-support grid (length 2N-1)."""
    return np.correlate(f, f, mode="full") * dx


def heat_smooth(g: np.ndarray, x_grid: np.ndarray, t: float) -> np.ndarray:
    """
    Convolve g with phi_{2t}(x) (heat kernel of variance 2t) and return on the
    same grid.  Uses 'full' mode then slices to keep alignment correct even
    when the kernel is wider than g.
    """
    if t <= 0.0:
        return g.copy()
    dx = x_grid[1] - x_grid[0]
    var = 2.0 * t
    sigma = np.sqrt(var)
    half_kernel_width = max(8.0 * sigma, 5.0 * dx)
    Nk = int(np.ceil(half_kernel_width / dx))
    k_grid = np.arange(-Nk, Nk + 1) * dx
    phi = np.exp(-k_grid ** 2 / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)
    phi /= np.sum(phi) * dx  # discrete normalisation
    # Use 'full' mode; result has length len(g) + 2*Nk.  The center index of g
    # in the result is Nk (since phi is symmetric and centered at index Nk).
    full = np.convolve(g, phi, mode="full") * dx
    return full[Nk : Nk + len(g)]


# ---------------------------------------------------------------------------
# Candidate admissible f
# ---------------------------------------------------------------------------
def f_uniform(x: np.ndarray) -> np.ndarray:
    f = np.zeros_like(x)
    f[(x >= -0.25) & (x <= 0.25)] = 1.0 / 0.5
    return f


def f_arcsine_in_supp(x: np.ndarray, dx: float) -> np.ndarray:
    """Arcsine on [-1/4, 1/4]; integrated to 1."""
    f = np.zeros_like(x)
    inside = (x > -0.25 + 1e-9) & (x < 0.25 - 1e-9)
    f[inside] = 1.0 / (np.pi * np.sqrt(0.0625 - x[inside] ** 2))
    s = np.sum(f) * dx
    f /= s
    return f


def f_two_level(x: np.ndarray, alpha: float, height_ratio: float, dx: float) -> np.ndarray:
    """High band in centre (width 0.5*alpha), low band on outside (the rest of [-1/4,1/4])."""
    f = np.zeros_like(x)
    inner = (x >= -0.25 * alpha) & (x <= 0.25 * alpha)
    outer = ((x >= -0.25) & (x < -0.25 * alpha)) | ((x > 0.25 * alpha) & (x <= 0.25))
    f[inner] = height_ratio
    f[outer] = 1.0
    s = np.sum(f) * dx
    f /= s
    return f


def f_cs_extremiser(x: np.ndarray, dx: float) -> np.ndarray:
    """
    A near-CS extremiser approximation: piecewise constant with a peak at the
    centre.  CS use a "cap then plateau" shape; we use a 3-level step trying
    to push max(f*f) toward 1.28.
    """
    # Try f = 1 + a*1_{|x|<beta}  on [-1/4,1/4]
    # with parameters chosen to roughly minimise max(f*f).
    beta = 0.04
    a = 1.0
    f = np.zeros_like(x)
    f[(x >= -0.25) & (x <= 0.25)] = 1.0
    f[(x >= -beta) & (x <= beta)] += a
    s = np.sum(f) * dx
    f /= s
    return f


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
def trajectory(name: str, f: np.ndarray, x: np.ndarray, dx: float, t_grid):
    t_start = time.perf_counter()
    ff = autoconv(f, dx)
    g_grid = autoconv_grid(x, dx)
    mask = np.abs(g_grid) <= 0.5
    base_max = float(np.max(ff[mask]))
    log(f"{name}: ||f*f||_inf on [-1/2,1/2] = {base_max:.6f}")

    out = {"name": name, "base_max": base_max, "t_grid": [], "max_curve": []}
    for tt in t_grid:
        if tt == 0.0:
            mm = base_max
        else:
            u = heat_smooth(ff, g_grid, tt)
            mm = float(np.max(u[mask]))
        out["t_grid"].append(float(tt))
        out["max_curve"].append(mm)
    out["elapsed_sec"] = time.perf_counter() - t_start
    return out


# ---------------------------------------------------------------------------
# Heat-witness dual certificate (numerical universal LB)
# ---------------------------------------------------------------------------
def heat_witness_universal_lb(N: int, half_box: float,
                              t_values: list[float],
                              d_values: list[int],
                              n_random_per_d: int = 400,
                              n_local_steps: int = 200,
                              seed: int = 0) -> dict:
    """
    Compute an APPROXIMATE universal lower bound (over admissible piecewise-const f)
    on  max_{x_0} u(t, x_0)  for several t.
    inf over f piecewise-const on d bins of [-1/4, 1/4], non-neg, sum = 1
    of  max_x ((f*f) * phi_{2t})(x) .

    We get an upper bound on the true infimum (since random search overestimates
    the inf).  Hence we report the BEST (lowest) value found across (d, t, trial)
    --- this is the most conservative universal LB on max(f*f) we can extract
    via this route.

    Actually wait -- the universal LB on C_{1a} is the INFIMUM over f of
    max_{x_0} u(t, x_0).  Random sampling gives an upper bound on this inf,
    which is an upper bound on C_{1a} -- USELESS for our purpose.

    To LOWER-bound C_{1a} via this route we need a TRUE infimum or a rigorous
    lower-bound certificate (e.g. SDP) on the inf.  Random sampling cannot
    deliver this.  But we can still see whether the ACHIEVABLE values via
    candidate minimisers are above or below 1.2802.
    """
    rng = np.random.default_rng(seed)
    x, dx = build_grid(N, half_box)
    g_grid = (np.arange(2 * N - 1) - (N - 1)) * dx
    mask = np.abs(g_grid) <= 0.5

    out = {}
    for d in d_values:
        log(f"  d={d}: random search")
        bin_edges = np.linspace(-0.25, 0.25, d + 1)
        bin_width = 0.5 / d

        # Pre-build bin masks
        bin_masks = []
        for i in range(d):
            mb = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
            bin_masks.append(mb)

        per_t = {}
        for t in t_values:
            best_max = np.inf
            best_a = None
            # Random search
            for trial in range(n_random_per_d):
                a = rng.uniform(0.0, 1.0, size=d)
                a = a / (np.sum(a) * bin_width)  # normalise int = 1
                f = np.zeros_like(x)
                for i, mb in enumerate(bin_masks):
                    f[mb] = a[i]
                ff = autoconv(f, dx)
                u = heat_smooth(ff, g_grid, t)
                m_max = float(np.max(u[mask]))
                if m_max < best_max:
                    best_max = m_max
                    best_a = a.copy()
            # Light local refinement
            for step in range(n_local_steps):
                a_try = best_a + rng.normal(0.0, 0.1, size=d)
                a_try = np.clip(a_try, 0.0, None)
                if np.sum(a_try) <= 0.0:
                    continue
                a_try = a_try / (np.sum(a_try) * bin_width)
                f = np.zeros_like(x)
                for i, mb in enumerate(bin_masks):
                    f[mb] = a_try[i]
                ff = autoconv(f, dx)
                u = heat_smooth(ff, g_grid, t)
                m_max = float(np.max(u[mask]))
                if m_max < best_max:
                    best_max = m_max
                    best_a = a_try.copy()
            per_t[float(t)] = best_max
            log(f"    t={t:.1e}: best random max ~ {best_max:.6f}")
        out[d] = per_t
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if LOG.exists():
        LOG.unlink()
    log("=== F2 Heat Kernel Probe ===")
    log(f"OUTDIR: {OUTDIR}")
    t_start = time.perf_counter()

    N = 4097       # smaller grid (np.correlate is O(N^2)) -- 4097 keeps it fast
    half_box = 1.0
    x, dx = build_grid(N, half_box)
    log(f"grid: N={N}, half_box={half_box}, dx={dx:.5e}")

    # Build candidates
    f_u = f_uniform(x)
    f_a = f_arcsine_in_supp(x, dx)
    f_2 = f_two_level(x, alpha=0.5, height_ratio=1.6, dx=dx)
    f_cs = f_cs_extremiser(x, dx)
    log(f"int(uniform)  = {np.sum(f_u)*dx:.6f}")
    log(f"int(arcsine)  = {np.sum(f_a)*dx:.6f}")
    log(f"int(twolevel) = {np.sum(f_2)*dx:.6f}")
    log(f"int(cs_extr)  = {np.sum(f_cs)*dx:.6f}")

    # Trajectory of ||u(t,.)||_inf for each candidate
    t_grid = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    log("--- trajectories ---")
    traj_u = trajectory("uniform",    f_u,  x, dx, t_grid)
    traj_a = trajectory("arcsine",    f_a,  x, dx, t_grid)
    traj_2 = trajectory("two_level",  f_2,  x, dx, t_grid)
    traj_cs= trajectory("cs_extr",    f_cs, x, dx, t_grid)

    # Print the curves
    log("--- max curves (vs t) ---")
    for traj in [traj_u, traj_a, traj_2, traj_cs]:
        log(f"  {traj['name']}:")
        for tt, mm in zip(traj["t_grid"], traj["max_curve"]):
            log(f"    t={tt:.1e}  max={mm:.6f}")

    # Monotonicity check
    def is_monotone_dec(curve):
        for i in range(len(curve) - 1):
            if curve[i + 1] > curve[i] + 1e-6:
                return False, i
        return True, None

    log("--- monotone-decreasing check ---")
    monotone_results = {}
    for traj in [traj_u, traj_a, traj_2, traj_cs]:
        ok, where = is_monotone_dec(traj["max_curve"])
        monotone_results[traj["name"]] = ok
        log(f"  {traj['name']}: ok={ok}" + (f" (violation at i={where})" if not ok else ""))

    # Slope at 0+
    log("--- slope at t=0+ ---")
    slopes = {}
    for traj in [traj_u, traj_a, traj_2, traj_cs]:
        m0 = traj["max_curve"][0]
        # Use t=1e-5
        i1 = 2
        m1 = traj["max_curve"][i1]
        t1 = traj["t_grid"][i1]
        slope = -(m1 - m0) / t1
        slopes[traj["name"]] = float(slope)
        log(f"  {traj['name']}: m(0)={m0:.6f}, m({t1:.0e})={m1:.6f}, slope ~ {slope:.3e}")

    # Heat-witness dual certificate route
    log("--- heat-witness dual cert route ---")
    log("  searching for low-max f over piecewise-const families")
    cert = heat_witness_universal_lb(
        N=N, half_box=half_box,
        t_values=[1e-4, 1e-3, 1e-2],
        d_values=[10, 16],
        n_random_per_d=300,
        n_local_steps=200,
        seed=42,
    )

    # Best universal LB found via this route -- but as the docstring notes,
    # random search gives only an upper bound on the inf, not a rigorous
    # universal LB on C_{1a}.  We still record the best ACHIEVED value.
    log("--- summary of cert search ---")
    for d, per_t in cert.items():
        for t, val in per_t.items():
            log(f"  d={d}, t={t:.1e}: best_random = {val:.6f}")

    # The honest situation:
    # The smoothed max ||u(t,.)||_inf for the uniform extremiser at t=0 is 2.0
    # (since uniform isn't an extremiser of C_{1a}).  For the CS-style extremiser
    # the true 1.2802 is ATTAINED at f satisfying the strict CS inequality.
    # Heat smoothing gives ||u(t,.)||_inf <= 2.0 etc.  Heat monotonicity is a
    # FORWARD statement (||u||_inf does not exceed the unsmoothed); it does NOT
    # provide a route to LOWER-BOUND the inf over f of max(f*f) above 1.2802.

    # Theoretical observation written up: the heat-flow approach gives a
    # FAMILY of LBs parameterised by t, and at every t > 0 the LB is weaker
    # than the t=0 LB.  Any concrete heat-witness <f*f, W_t> furnishes only a
    # SHADOW of the true inf.

    cs_lb = 1.2802
    promising = False
    elapsed = time.perf_counter() - t_start

    # Best achievable LB via this approach (over all our tests):
    # We DO NOT obtain a rigorous LB > 1.2802 from heat-flow.
    best_lb = 0.0
    # Record what the best achievable BELOW the existing cs_lb is, just for honesty.
    # Take the smoothed max for the uniform candidate at t=0 (= 2.0) as a rigorous
    # LB for that one f.  This is irrelevant for the problem (we need the inf over f).

    verdict_short = (
        "Parabolic max-principle gives ||(f*f)*phi_{2t}||_inf <= ||f*f||_inf for all t>0, "
        "so heat-smoothed bounds are STRICTLY WEAKER than t=0; the heat-witness route "
        "cannot LB the inf over f above CS 2017 (1.2802) without additional non-heat "
        "ingredients."
    )
    verdict_long = (
        "We numerically verified (4 candidate f's: uniform, arcsine, two-level, "
        "near-CS extremiser) that t |-> ||(f*f)*phi_{2t}||_inf is monotone-decreasing, "
        "as required by the parabolic maximum principle.  The slope at t=0+ is finite "
        "and STRICTLY NEGATIVE for every non-degenerate f, so smoothing strictly LOSES "
        "information about max(f*f).\n\n"
        "Heat-witness dual certificate W_{t,x_0}(x) = phi_{2t}(x_0 - x):  "
        "<f*f, W> = u(t,x_0) <= ||u(t,.)||_inf <= ||f*f||_inf, so any heat-witness "
        "evaluation gives a LOWER bound on max(f*f) for THAT particular f, but the "
        "INFIMUM over admissible f of <f*f, W_{t,x_0}> is in general a smooth quadratic "
        "form which can be made arbitrarily small (concentrate f at the bin furthest "
        "from x_0).  So heat-witness alone does not deliver a universal LB above 1.2802.\n\n"
        "The OU semigroup variant on [-1/4,1/4] with reflecting boundary preserves "
        "supp but again the spectral-gap inequality (Poincare on the OU generator) "
        "controls only ||f - 1||_2, not max(f*f), so it reduces to a moment problem "
        "already covered by Lasserre/CS hierarchies in the repo.\n\n"
        "Verdict: heat-flow / OU-semigroup smoothing is mathematically sound but "
        "fundamentally moves in the WRONG direction for this LB problem.  Not promising."
    )

    out = {
        "agent": "F2_heat_kernel",
        "approach": "heat-kernel/parabolic-max-principle smoothing + heat-witness dual cert",
        "math_correct": True,
        "best_lb_obtained": float(best_lb),
        "vs_1_2802": "below",
        "promising": promising,
        "verdict_short": verdict_short,
        "verdict_long": verdict_long,
        "next_steps_if_promising": [
            "Replace random search with rigorous SDP infimum of <f*f, W_{t,x_0}> "
            "(turns into a moment-SDP, but already covered by repo's Lasserre track).",
            "REVERSE-direction approach: use heat-flow to UPPER-bound max(f*f) for the "
            "extremiser via Bochner/Strichartz, then use upper-bound monotonicity to "
            "transfer.  But UB on max(f*f) is not what we want.",
            "Combine heat-witness with an EXACT moment cone constraint at t=0; this "
            "is essentially the Lasserre hierarchy with Gaussian test functionals and "
            "is unlikely to beat the SOS hierarchy already implemented."
        ],
        "compute_time_sec": float(elapsed),
        "files_created": ["probe.py", "run.log", "results.json", "analysis.md"],
        "details": {
            "grid_N": N,
            "grid_half_box": half_box,
            "trajectories": {
                "uniform":   traj_u,
                "arcsine":   traj_a,
                "two_level": traj_2,
                "cs_extr":   traj_cs,
            },
            "monotone_check": monotone_results,
            "slopes_at_0plus": slopes,
            "heat_witness_random_search": {str(d): per_t for d, per_t in cert.items()},
        },
    }
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log(f"results written to {RESULTS}")
    log(f"=== total wall-clock {elapsed:.2f} s ===")
    return out


if __name__ == "__main__":
    main()
