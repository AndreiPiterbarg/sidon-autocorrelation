"""Probe 2: deeper analysis.

(a) Verify free convolution implementation against the analytic case:
    semicircle distribution sc_R on [-2R, 2R] satisfies sc_R boxed_plus sc_R
    = sc_{R sqrt(2)}, density 1/(pi*2*R^2) sqrt(8R^2 - t^2) for |t|<=2R sqrt(2).

(b) Larger search for minimum free max, including symmetric configurations.

(c) Test whether the *infimum* of ||mu boxed_plus mu||_inf over admissible
    measures on [-1/4, 1/4] could even be > 1.2802 in principle, by checking
    structural bounds.
"""
import numpy as np, time, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from probe import (G_step, free_self_conv_density, classical_self_conv_density,
                   step_density, _json_default)

START = time.time()
HERE = Path(__file__).parent
LOGFILE = HERE / "run.log"

def log(msg):
    elapsed = time.time() - START
    line = f"[probe2 {elapsed:7.2f}s] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# (a) Semicircle sanity check: discretize semicircle on [-1/4, 1/4] (R = 1/8)
# Then mu boxed_plus mu = semicircle on [-R sqrt 8, R sqrt 8] = [-sqrt(2)/4, sqrt(2)/4]
# Peak density of semicircle of radius 2R is 1/(pi R) at t=0.
# For sc_R on [-2R, 2R], density = (1/(2 pi R^2)) sqrt(4R^2 - t^2).
# Peak = 1/(pi R).
# After free conv: sc_{R sqrt 2}, peak = 1/(pi R sqrt 2).
# With R = 1/8: peak_input = 8/pi = 2.546, peak_output = 8/(pi sqrt 2) = 1.801.
# ---------------------------------------------------------------------------
def discretize_semicircle(d):
    """Approximate semicircle on [-1/4, 1/4] by step density on d bins."""
    bin_centers = -0.25 + (np.arange(d) + 0.5) * (0.5 / d)
    R = 0.125  # radius
    # density at center
    dens = (1.0 / (2 * np.pi * R * R)) * np.sqrt(np.clip(4 * R * R - bin_centers ** 2, 0.0, None))
    bin_width = 0.5 / d
    a = dens * bin_width
    a = a / a.sum()
    return a


def test_semicircle():
    log("--- (a) Semicircle sanity ---")
    d = 64
    a = discretize_semicircle(d)
    R = 0.125
    expected_input_peak = 1.0 / (np.pi * R)  # 8/pi
    expected_output_peak = 1.0 / (np.pi * R * np.sqrt(2))
    log(f"  Theory: input peak = {expected_input_peak:.4f}, output peak (free) = {expected_output_peak:.4f}")
    # check input
    bin_width = 0.5 / d
    f_max_input = (a / bin_width).max()
    log(f"  Step semicircle input max density: {f_max_input:.4f}")

    # Classical conv: f * f for semicircle does NOT give semicircle; it gives a
    # different shape with peak ~= int f^2 by Cauchy-Schwarz... actually classical
    # convolution of semicircle with itself is a known explicit function.
    t_grid = np.linspace(-0.499, 0.499, 401)
    cl = classical_self_conv_density(a, t_grid)
    log(f"  Classical f*f peak = {cl.max():.4f}")

    # Free conv
    rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
    valid = ~np.isnan(rho_free)
    log(f"  Free solver: {valid.sum()}/{len(t_grid)} pts ok")
    fmax = float(np.nanmax(rho_free))
    log(f"  Free f boxed_plus f peak = {fmax:.4f}  (theory = {expected_output_peak:.4f})")
    rel_err = abs(fmax - expected_output_peak) / expected_output_peak
    log(f"  Relative error vs theory = {rel_err*100:.2f}%")
    return {"theory_peak": expected_output_peak, "computed_peak": fmax, "rel_err": rel_err}


# ---------------------------------------------------------------------------
# (b) Symmetric search for minimum free max
# ---------------------------------------------------------------------------
def symmetric_search(d=10, n_trials=30, seed=11):
    log(f"--- (b) Symmetric search d={d}, n={n_trials} ---")
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(-0.499, 0.499, 401)
    half = (d + 1) // 2
    best = (np.inf, None, None)
    for trial in range(n_trials):
        # Generate symmetric a: mirror left half from right half
        h = rng.dirichlet(np.ones(half) * 2.0)
        if d % 2 == 0:
            a = np.concatenate([h[::-1], h])
        else:
            a = np.concatenate([h[1:][::-1], h])  # avoids middle bin doubled
        a = a / a.sum()
        cl = classical_self_conv_density(a, t_grid)
        c_max = cl.max()
        rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
        valid = ~np.isnan(rho_free)
        if valid.sum() < 200:
            continue
        f_max = float(np.nanmax(rho_free))
        if f_max < best[0]:
            best = (f_max, c_max, a.copy())
        if (trial + 1) % 10 == 0:
            log(f"  trial {trial+1}: best free_max so far = {best[0]:.4f}")
    log(f"  Best symmetric: free_max = {best[0]:.4f}, classical_max at that pt = {best[1]:.4f}")
    return best


# ---------------------------------------------------------------------------
# (c) Spike + uniform mixtures (parametric study)
# Try a = (1-eps)/d * ones + eps * delta concentrated at edges.
# Free convolution behaves very differently from classical.
# ---------------------------------------------------------------------------
def edge_concentration_study(d=12):
    log(f"--- (c) Edge concentration study d={d} ---")
    t_grid = np.linspace(-0.499, 0.499, 401)
    eps_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    results = []
    for eps in eps_list:
        a = np.ones(d) / d
        bulk_mass = 1.0 - eps
        a = a * bulk_mass
        a[0] += eps / 2
        a[-1] += eps / 2
        cl = classical_self_conv_density(a, t_grid)
        c_max = cl.max()
        rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
        valid = ~np.isnan(rho_free)
        f_max = float(np.nanmax(rho_free)) if valid.sum() > 100 else None
        log(f"  eps={eps}: classical={c_max:.4f}, free={f_max}")
        results.append({"eps": eps, "classical_max": c_max, "free_max": f_max})
    return results


# ---------------------------------------------------------------------------
# (d) Bimodal / two-spike test: can free convolution have a SHARP single peak?
# Key insight: free convolution of a discrete measure with itself can be
# absolutely continuous and smoother than classical.
# ---------------------------------------------------------------------------
def two_spike_study(d=20):
    log(f"--- (d) Two-spike d={d} ---")
    t_grid = np.linspace(-0.499, 0.499, 401)
    results = []
    for gap_idx in [1, 2, 3, 5, 7, 9]:
        a = np.zeros(d)
        # two spikes symmetric around center, separated by gap
        center = d // 2
        a[center - gap_idx] = 0.5
        a[center + gap_idx - 1] = 0.5
        cl = classical_self_conv_density(a, t_grid)
        c_max = cl.max()
        rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
        valid = ~np.isnan(rho_free)
        f_max = float(np.nanmax(rho_free)) if valid.sum() > 100 else None
        log(f"  gap_idx={gap_idx}: classical={c_max:.4f}, free={f_max}")
        results.append({"gap_idx": gap_idx, "classical_max": c_max, "free_max": f_max})
    return results


# ---------------------------------------------------------------------------
# (e) Nesting of supports: support of mu boxed_plus mu vs supp(mu * mu)
#   Both lie in [-1/2, 1/2] for mu on [-1/4, 1/4]?  Actually NO for free.
#   Free convolution support can EXCEED classical support! Check.
# ---------------------------------------------------------------------------
def support_check(d=12):
    log(f"--- (e) Support comparison d={d} ---")
    a = np.array([0.18, 0.10, 0.07, 0.15, 0.15, 0.07, 0.10, 0.18])
    a = a / a.sum()
    # extend to d=12
    a12 = np.zeros(12)
    a12[2:10] = a
    a12 = a12 / a12.sum()

    t_grid = np.linspace(-0.6, 0.6, 601)
    cl = classical_self_conv_density(a12, t_grid)
    rho_free, _, _ = free_self_conv_density(a12, t_grid, eps=2e-3)
    valid = ~np.isnan(rho_free)
    log(f"  classical: support detected via density>0.001:")
    cl_supp = t_grid[cl > 0.001]
    if len(cl_supp) > 0:
        log(f"    [{cl_supp.min():.4f}, {cl_supp.max():.4f}]  (theoretical: [-1/2, 1/2] = [-0.5, 0.5])")
    log(f"  free: support detected via density>0.001:")
    valid_dense = valid & (rho_free > 0.001)
    fr_supp = t_grid[valid_dense]
    if len(fr_supp) > 0:
        log(f"    [{fr_supp.min():.4f}, {fr_supp.max():.4f}]")
    return {"classical_support": [float(cl_supp.min()), float(cl_supp.max())] if len(cl_supp) > 0 else None,
            "free_support": [float(fr_supp.min()), float(fr_supp.max())] if len(fr_supp) > 0 else None}


def main():
    out = {}
    log("Begin probe2")
    try:
        out["semicircle_check"] = test_semicircle()
    except Exception as e:
        log(f"  semicircle failed: {e}")
        out["semicircle_check"] = {"error": str(e)}
    try:
        b = symmetric_search()
        out["symmetric_search"] = {
            "best_free_max": b[0],
            "best_classical_max": b[1],
            "best_a": b[2].tolist() if b[2] is not None else None,
        }
    except Exception as e:
        log(f"  symmetric search failed: {e}")
    try:
        out["edge_concentration"] = edge_concentration_study()
    except Exception as e:
        log(f"  edge concentration failed: {e}")
    try:
        out["two_spike"] = two_spike_study()
    except Exception as e:
        log(f"  two spike failed: {e}")
    try:
        out["support_check"] = support_check()
    except Exception as e:
        log(f"  support check failed: {e}")
    out["compute_time_sec"] = time.time() - START
    out_path = HERE / "results_probe2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    log(f"  Wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
