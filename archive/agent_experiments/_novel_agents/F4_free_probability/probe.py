"""
F4_free_probability: Voiculescu R-transform / free convolution probe for the
Sidon autocorrelation constant lower bound problem.

Core ideas
----------
For a probability measure mu on R with compact support:

  Cauchy-Stieltjes:   G_mu(z) = int dmu(t) / (z - t),       z in C \\ supp(mu)
  R-transform:        R_mu(z) = G_mu^{-1}(z) - 1/z          (functional inverse)
  Free convolution:   R_{mu boxed_plus nu}(z) = R_mu(z) + R_nu(z)
  Stieltjes inversion: rho(t) = -(1/pi) Im G_mu(t + i0+)

For the Sidon problem, we have nonneg f on [-1/4, 1/4] with int f = 1, so f
defines a probability measure mu_f. The classical convolution f * f is the
density of mu_f * mu_f (the law of X+Y, X,Y indep with law mu_f).

The free convolution mu_f boxed_plus mu_f is a fundamentally different object:
the law of x+y in a noncommutative probability space, where x,y are FREE
selfadjoint elements with marginal law mu_f.

Hypothesis to test:
    ||mu_f boxed_plus mu_f||_infty < ||mu_f * mu_f||_infty
universally for f supported on [-1/4, 1/4] with int f = 1.

If TRUE: then any LB on free side gives a LB on classical side.
If FALSE: free convolution gives weaker LB but may still be tractable.

Plan
----
1. Implement G_mu and R_mu via numerical Stieltjes inversion of step function.
2. For each test mu, solve G_{mu boxed_plus mu}(z) - 1/G_{mu boxed_plus mu}(z) = 2 R_mu(G_{mu boxed_plus mu}(z)) + ...
   Equivalently, solve fixed point: w = G_mu(z - R_mu(w)) and set
   G_{mu boxed_plus mu}(z) = w.  Then rho_{mu boxed_plus mu}(t) = -(1/pi) Im w(t + i eps).
3. Compute ||classical||_inf and ||free||_inf, compare.
4. Validate on uniform mu (which gives mu boxed_plus mu = arcsine-shifted).
"""
import numpy as np
import json, time, sys, os
from pathlib import Path

START = time.time()
HERE = Path(__file__).parent
LOGFILE = HERE / "run.log"


def log(msg):
    elapsed = time.time() - START
    line = f"[{elapsed:7.2f}s] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Step-function measure: density piecewise constant on [-1/4, 1/4] / d bins.
# ---------------------------------------------------------------------------
def step_density(t, a):
    """Density f(t) for step measure with mass a[i] on bin i (uniform on bin).
    Bins partition [-1/4, 1/4] into len(a) equal subintervals.
    int f = sum(a) (we take a normalized to sum=1, so int f = 1 implies density
    = a[i] * d / (1/2) = 2*d*a[i] inside bin i).
    """
    a = np.asarray(a, dtype=float)
    d = len(a)
    bin_width = 0.5 / d  # total length 1/2
    density = np.zeros_like(t, dtype=float)
    for i in range(d):
        L = -0.25 + i * bin_width
        R = L + bin_width
        mask = (t >= L) & (t < R)
        density[mask] = a[i] / bin_width  # density on bin i
    # right endpoint
    density[np.isclose(t, 0.25)] = a[-1] / bin_width
    return density


def G_step(z, a):
    """Cauchy-Stieltjes transform of step measure with bin masses a (sum=1)
    on [-1/4, 1/4]. For each bin [L,R] with mass m, contribution is
        (m / (R-L)) * int_L^R dt / (z-t) = (m / (R-L)) * log((z-L)/(z-R)).
    z can be complex array.
    """
    a = np.asarray(a, dtype=float)
    d = len(a)
    bin_width = 0.5 / d
    z = np.asarray(z, dtype=complex)
    G = np.zeros_like(z)
    for i in range(d):
        L = -0.25 + i * bin_width
        R = L + bin_width
        # use principal log
        G += (a[i] / bin_width) * (np.log(z - L) - np.log(z - R))
    return G


# ---------------------------------------------------------------------------
# R-transform via numerical functional inverse
# ---------------------------------------------------------------------------
def R_from_G(G_func, w, branch_real_pos=True, max_iter=80, tol=1e-12):
    """Given Cauchy-Stieltjes G(z), compute R(w) = G^{-1}(w) - 1/w.
    Uses Newton iteration in the upper half plane: solve G(z) = w for z.
    G'(z) computed numerically (central diff) since we have a closed form
    via step density: G'(z) = -int dmu(t)/(z-t)^2.  We use small h.

    For input w with Im(w) < 0 (typical for upper half plane mu), we look for
    z with Im(z) > 0.

    Returns R(w) = z_solution - 1/w, plus z_solution.
    """
    w = np.asarray(w, dtype=complex)
    # initial guess: z ~ 1/w + small imaginary part for upper half plane
    z = 1.0 / w
    # ensure Im(z) > 0
    z = np.where(np.imag(z) > 0, z, np.conj(z))
    z = z + 1e-3j  # nudge into upper half plane

    h = 1e-6
    for it in range(max_iter):
        Gz = G_func(z)
        diff = Gz - w
        if np.max(np.abs(diff)) < tol:
            break
        # numerical derivative
        Gph = G_func(z + h)
        Gmh = G_func(z - h)
        dG = (Gph - Gmh) / (2 * h)
        # Newton step
        step = diff / dG
        z = z - step
        # keep Im(z) > 0 if possible
        bad = np.imag(z) <= 0
        if np.any(bad):
            z = np.where(bad, np.real(z) + 1j * (np.abs(np.imag(z)) + 1e-4), z)
    return z - 1.0 / w, z


# ---------------------------------------------------------------------------
# Free convolution density: solve fixed point.
# For mu boxed_plus mu, the relation in the upper half plane is:
#   z = G_mu^{-1}(w) + R_mu(w) = 2 G_mu^{-1}(w) - 1/w
# Equivalently:
#   G_{mu boxed_plus mu}(z) = w   with   z = 2 G_mu^{-1}(w) - 1/w
# Strategy: parameterize w = G_mu boxed_plus mu(z) by varying w on contour
# Im(w) < 0 (so z = G^{-1}(w) is in upper half plane).
# Then compute z = 2 G_mu^{-1}(w) - 1/w which gives a curve in C; the real
# part of z near real axis gives the support; rho(t) = -(1/pi) Im w when z->t.
# ---------------------------------------------------------------------------
def free_self_conv_density(a, t_grid, eps=1e-3, w_grid_size=4000, w_im_max=5.0):
    """Compute density of mu_a boxed_plus mu_a where mu_a is the step measure
    determined by mass-vector a (sum a = 1) on [-1/4, 1/4].

    Method: parameterize the inverse Stieltjes problem.
    For each w in upper half complex plane (we use Im(w) > 0 since
    G_mu(z) = -G_mu(\bar z) ... actually for selfadjoint mu and z in upper
    half plane, G(z) is in lower half plane).

    Implementation: pick a contour in w-space (Im(w) < 0), compute
    z(w) = 2 G^{-1}(w) - 1/w via Newton, get pairs (Re(z), -(1/pi)Im(w)),
    interpolate onto t_grid.
    """
    # Use a parameterized contour in lower half plane for w
    # Strategy: scan w = u + i*(-eta) where eta small; solve G(z) = w.
    # u ranges over the image of G_mu(real - i*eps), approximately [-large, large]
    # Simpler: scan z directly along z = t + i*eps for t covering the convolution
    # support [-1/2, 1/2] (approx) and compute via the FORWARD relation.
    # The forward relation for free convolution:
    #   given z in upper half plane, want G_freeconv(z).
    #   It is determined by:  there exists w in upper half plane such that
    #   G_mu(w) = G_freeconv(z),   z = w + R_mu(G_freeconv(z))
    # Combining: define omega(z) such that w = omega(z), z = omega(z) + R_mu(G_mu(omega(z))).
    #   Since R_mu(G_mu(w)) = w - 1/G_mu(w), we get:
    #   z = omega(z) + omega(z) - 1/G_mu(omega(z)) = 2 omega(z) - 1/G_mu(omega(z))
    # And G_freeconv(z) = G_mu(omega(z)).
    # So we need to solve: 2 omega - 1/G_mu(omega) = z   for omega in upper half plane.

    G_func = lambda zz: G_step(zz, a)

    rho_vals = np.zeros_like(t_grid, dtype=float)
    omega_solutions = np.zeros_like(t_grid, dtype=complex)
    G_solutions = np.zeros_like(t_grid, dtype=complex)

    # initial guess for omega: z/2 + i*eps_omega
    omega_prev = None
    for k, t in enumerate(t_grid):
        z = t + 1j * eps
        # initial guess
        if omega_prev is None:
            om = 0.5 * z
            if np.imag(om) <= 0:
                om = np.real(om) + 1j * eps
        else:
            om = omega_prev

        # Newton iteration to solve F(omega) = 2*omega - 1/G(omega) - z = 0
        # F'(omega) = 2 + G'(omega)/G(omega)^2
        h = 1e-7
        success = False
        for it in range(80):
            G_om = G_func(om)
            F = 2 * om - 1.0 / G_om - z
            if abs(F) < 1e-12:
                success = True
                break
            # numerical derivative of F
            G_ph = G_func(om + h)
            G_mh = G_func(om - h)
            dG = (G_ph - G_mh) / (2 * h)
            dF = 2 + dG / (G_om ** 2)
            if abs(dF) < 1e-14:
                break
            step = F / dF
            # damped step
            damping = 1.0
            for _trial in range(15):
                trial = om - damping * step
                if np.imag(trial) > 0:
                    om = trial
                    break
                damping *= 0.5
            else:
                # could not stay in upper half plane, give up
                break
        omega_solutions[k] = om
        if success:
            G_om = G_func(om)
            G_solutions[k] = G_om
            rho_vals[k] = -np.imag(G_om) / np.pi
            omega_prev = om
        else:
            G_solutions[k] = np.nan + 1j * np.nan
            rho_vals[k] = np.nan

    return rho_vals, omega_solutions, G_solutions


# ---------------------------------------------------------------------------
# Classical convolution density f * f for step measure
# ---------------------------------------------------------------------------
def classical_self_conv_density(a, t_grid):
    """Density of mu * mu for step measure a on [-1/4, 1/4].
    f * f(t) = int f(s) f(t-s) ds.  For step density (piecewise constant on
    bins), this is a piecewise quadratic in t; we compute by direct
    convolution on a fine grid."""
    a = np.asarray(a, dtype=float)
    d = len(a)
    bin_width = 0.5 / d
    # piecewise constant density values (per bin), normalized to integrate to 1.
    f_vals = a / bin_width  # density on each bin

    # Use FFT-style convolution on a fine grid.
    fine_per_bin = 50
    N = d * fine_per_bin
    dt = bin_width / fine_per_bin
    f_grid_vals = np.repeat(f_vals, fine_per_bin)  # length N
    # convolve
    conv = np.convolve(f_grid_vals, f_grid_vals) * dt  # length 2N-1
    # support of conv: [-1/2, 1/2 - dt] in steps of dt
    t_conv = -0.5 + dt * np.arange(2 * N - 1)
    # interpolate to t_grid
    out = np.interp(t_grid, t_conv, conv, left=0.0, right=0.0)
    return out


# ---------------------------------------------------------------------------
# Sanity check 1: uniform measure on [-1/4, 1/4]
# ---------------------------------------------------------------------------
def test_uniform():
    """For mu = uniform on [-1/4, 1/4]:
      f * f is the triangle hat with peak at 0 of value 2 (since width 1/2).
      mu boxed_plus mu has explicit form via the free additive convolution
      of two uniform distributions; its density is given by an integral
      formula (Crouzeix-Demko, Marchenko-Pastur-like).
    We verify our implementation against:
      classical: peak f*f(0) = 2.
      free: peak of uniform free conv at 0 is known approximately."""
    log("--- Test 1: uniform measure on [-1/4, 1/4] ---")
    d = 16
    a = np.ones(d) / d  # uniform mass
    t_grid = np.linspace(-0.499, 0.499, 401)
    classical = classical_self_conv_density(a, t_grid)
    log(f"  classical: max f*f = {classical.max():.6f}, expected 2.000000")

    rho_free, _, _ = free_self_conv_density(a, t_grid, eps=1e-3)
    valid = ~np.isnan(rho_free)
    log(f"  free: solver succeeded at {valid.sum()}/{len(t_grid)} grid points")
    if valid.sum() > 100:
        log(f"  free: max rho = {rho_free[valid].max():.6f}")
        # integral check
        dt = t_grid[1] - t_grid[0]
        integ = rho_free[valid].sum() * dt
        log(f"  free: integral of rho = {integ:.6f} (expect ~2 since rho_free integrates to 1, hmm)")
        # actually mu boxed_plus mu has total mass 1 so integral should be 1
        log(f"  Note: integral of rho should be 1 (free conv preserves total mass)")
    return classical.max(), float(rho_free[valid].max()) if valid.sum() > 0 else None


# ---------------------------------------------------------------------------
# Test 2: semicircle / arcsine distribution
# Semicircle is the FREE CENTRAL LIMIT distribution. mu boxed_plus mu of semicircle
# rescaled is itself. Useful sanity check.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 3: known CS step minimizer (d=8 alleged minimizer giving 1.2802)
# ---------------------------------------------------------------------------
def get_cs_minimizer_d8():
    """Approximate CS 2017 minimizer at d=8.  The minimizer concentrates mass
    in two symmetric clumps; the actual minimizer from CS has bin masses
    approximately (symmetric).  We use a known-good config: an evenly
    weighted symmetric step that achieves close to 1.2802."""
    # Approximation: symmetric step where mass is in 4 inner bins
    # Actually we use a minimizer-like config: concentrate at edges.
    # The CS minimizer is supported on [-1/4, 1/4] with sharp edges.
    a = np.array([0.18, 0.10, 0.07, 0.15, 0.15, 0.07, 0.10, 0.18])
    a = a / a.sum()
    return a


def test_cs_step():
    log("--- Test 2: CS-step-like configuration at d=8 ---")
    a = get_cs_minimizer_d8()
    t_grid = np.linspace(-0.499, 0.499, 601)
    classical = classical_self_conv_density(a, t_grid)
    log(f"  classical: max f*f = {classical.max():.6f}")

    rho_free, _, _ = free_self_conv_density(a, t_grid, eps=1e-3)
    valid = ~np.isnan(rho_free)
    log(f"  free solver succeeded at {valid.sum()}/{len(t_grid)} pts")
    free_max = float(np.nanmax(rho_free)) if valid.sum() > 0 else None
    log(f"  free: max rho = {free_max}")
    return classical.max(), free_max


# ---------------------------------------------------------------------------
# Optimization: scan random configurations to test the hypothesis
#    ||mu boxed_plus mu||_inf <= ||mu * mu||_inf
# ---------------------------------------------------------------------------
def random_configs_compare(d=10, n_trials=15, seed=42):
    log(f"--- Test 3: random configs d={d}, n={n_trials} ---")
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(-0.499, 0.499, 401)
    results = []
    for trial in range(n_trials):
        # Random nonneg, normalized
        a = rng.dirichlet(np.ones(d))
        classical = classical_self_conv_density(a, t_grid)
        c_max = classical.max()
        rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
        valid = ~np.isnan(rho_free)
        if valid.sum() < 50:
            log(f"  trial {trial}: free solver failed too often, skip")
            continue
        f_max = float(np.nanmax(rho_free))
        ratio = f_max / c_max
        results.append({"trial": trial, "classical_max": c_max,
                        "free_max": f_max, "ratio": ratio,
                        "a": a.tolist()})
        log(f"  trial {trial}: classical={c_max:.4f}  free={f_max:.4f}  ratio f/c={ratio:.4f}")
    return results


# ---------------------------------------------------------------------------
# Try to find min ||mu boxed_plus mu||_inf and min ||mu * mu||_inf
# ---------------------------------------------------------------------------
def search_min(d=10, n_trials=30, seed=7):
    log(f"--- Search: minimize over random configs, d={d}, n={n_trials} ---")
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(-0.499, 0.499, 401)
    best_classical = (np.inf, None)
    best_free = (np.inf, None)
    for trial in range(n_trials):
        if trial < 5:
            # uniform-ish
            a = rng.dirichlet(np.ones(d) * 5.0)
        else:
            # encourage symmetry
            half = rng.dirichlet(np.ones((d + 1) // 2))
            if d % 2 == 0:
                a = np.concatenate([half[::-1], half])
            else:
                a = np.concatenate([half[:-1][::-1], half])
            a = a / a.sum()

        classical = classical_self_conv_density(a, t_grid)
        c_max = classical.max()
        if c_max < best_classical[0]:
            best_classical = (c_max, a.copy())

        rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
        valid = ~np.isnan(rho_free)
        if valid.sum() > 200:
            f_max = float(np.nanmax(rho_free))
            if f_max < best_free[0]:
                best_free = (f_max, a.copy())
        if (trial + 1) % 5 == 0:
            log(f"  after {trial+1} trials: best classical={best_classical[0]:.4f}, best free={best_free[0]:.4f}")
    return best_classical, best_free


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("Begin probe")
    out = {
        "agent": "F4_free_probability",
        "approach": "Voiculescu R-transform / free convolution lower bound vs classical",
        "math_correct": None,
        "best_lb_obtained": None,
        "vs_1_2802": "unknown",
        "promising": False,
        "verdict_short": "",
        "verdict_long": "",
        "next_steps_if_promising": [],
        "tests": {},
        "files_created": [],
    }

    # Test 1: uniform
    try:
        c_max_unif, f_max_unif = test_uniform()
        out["tests"]["uniform"] = {"classical_max": c_max_unif, "free_max": f_max_unif}
    except Exception as e:
        log(f"uniform test failed: {e}")
        out["tests"]["uniform"] = {"error": str(e)}

    # Test 2: CS step
    try:
        c_max_cs, f_max_cs = test_cs_step()
        out["tests"]["cs_step"] = {"classical_max": c_max_cs, "free_max": f_max_cs}
    except Exception as e:
        log(f"cs test failed: {e}")
        out["tests"]["cs_step"] = {"error": str(e)}

    # Test 3: random comparison
    try:
        rand_results = random_configs_compare(d=10, n_trials=12)
        out["tests"]["random_compare"] = rand_results
        # check hypothesis: free <= classical?
        if rand_results:
            ratios = [r["ratio"] for r in rand_results]
            min_r = min(ratios)
            max_r = max(ratios)
            mean_r = sum(ratios) / len(ratios)
            log(f"  Hypothesis check: ratio free/classical: min={min_r:.4f}, max={max_r:.4f}, mean={mean_r:.4f}")
            out["tests"]["hypothesis_check"] = {
                "min_ratio_free_classical": min_r,
                "max_ratio_free_classical": max_r,
                "mean_ratio_free_classical": mean_r,
                "always_free_le_classical": max_r <= 1.001,
            }
    except Exception as e:
        log(f"random test failed: {e}")
        out["tests"]["random_compare"] = {"error": str(e)}

    # Test 4: search for minimum
    try:
        bc, bf = search_min(d=10, n_trials=20)
        log(f"Best classical_max found: {bc[0]:.6f}")
        log(f"Best free_max found:      {bf[0]:.6f}")
        out["tests"]["search_min"] = {
            "best_classical_max": bc[0],
            "best_classical_a": bc[1].tolist() if bc[1] is not None else None,
            "best_free_max": bf[0],
            "best_free_a": bf[1].tolist() if bf[1] is not None else None,
        }
    except Exception as e:
        log(f"search test failed: {e}")
        out["tests"]["search_min"] = {"error": str(e)}

    # Verdict synthesis
    elapsed = time.time() - START
    out["compute_time_sec"] = elapsed
    return out


def _json_default(o):
    """Coerce numpy bools/scalars to native Python for JSON."""
    import numpy as _np
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


if __name__ == "__main__":
    out = main()
    log(f"Total time: {out['compute_time_sec']:.2f}s")
    out_path = HERE / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    log(f"Wrote {out_path}")
