"""Probe 4: explore connection between free and classical convolution
through OTHER norms / functionals.

Key observation from probe3:
- min free_max ~ 1.22 < 1.28 = min classical_max  (free is LOWER on min)
- So free does NOT directly bound classical from below.

Question: is there a NORM combination
   F(mu * mu) >= G(mu boxed_plus mu)
that goes the right way?

(a) ||f||_2^2 = int f^2 = (mu * tilde mu)(0). For symmetric mu, this is (mu*mu)(0).
    Free analog:  Voiculescu's free L2 distance / ||mu boxed_plus mu||_2.
    Compute these and see relationships.

(b) Free entropy chi(mu) = int int log|x-y| dmu(x) dmu(y) + (3/4 + log 2pi) / 2.
    Voiculescu's free EPI: chi(mu boxed_plus mu) >= 2 chi(mu) + (1/2) log 2.
    Free Costa-Cover: dchi(mu_t)/dt >= 0 along free heat flow.

(c) Test: is min ||mu boxed_plus mu||_inf over admissible mu approaching 1.2802?
    Maybe MV-arcsine free_max = 1.2797 is the truthful free minimum?

(d) Compute exact peak of Wigner / arcsine free conv (analytic).
"""
import numpy as np, time, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from probe import (G_step, free_self_conv_density, classical_self_conv_density,
                   _json_default)

START = time.time()
HERE = Path(__file__).parent
LOGFILE = HERE / "run.log"


def log(msg):
    elapsed = time.time() - START
    line = f"[probe4 {elapsed:7.2f}s] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# (a) Free entropy for step measure
# chi(mu) = int int log|x-y| dmu(x) dmu(y) + 3/4 + log(2pi)/2  (Voiculescu)
# For step density a (mass on bins of width 1/d/2 = bw):
#   chi = sum_{i,j} a_i a_j log|x_i - x_j|_avg + entropy correction
# ---------------------------------------------------------------------------
def free_entropy(a):
    a = np.asarray(a)
    d = len(a)
    bw = 0.5 / d
    centers = -0.25 + (np.arange(d) + 0.5) * bw
    # double integral: sum a_i a_j log|x-y|, but bins have continuous distribution
    # within. Rough approximation: use centers.
    chi = 0.0
    for i in range(d):
        for j in range(d):
            if i == j:
                # within-bin self-energy: int_bin int_bin log|x-y| / bw^2
                # = log(bw) - 3/2  (standard formula for uniform)
                chi += a[i] * a[i] * (np.log(bw) - 1.5)
            else:
                chi += a[i] * a[j] * np.log(abs(centers[i] - centers[j]))
    chi += 0.75 + 0.5 * np.log(2 * np.pi)
    return chi


def classical_l2(a):
    """||f||_2^2 for step density."""
    bw = 0.5 / len(a)
    return float(np.sum((a / bw) ** 2 * bw))  # == sum a_i^2 / bw


# ---------------------------------------------------------------------------
# (b) Compute ||mu boxed_plus mu||_2 numerically; compare to ||mu * mu||_2
# Note: ||mu*mu||_2 = sum over translates -- not the same as ||mu boxed_plus mu||_2
# Both should be expressible via moments.
# ---------------------------------------------------------------------------
def conv_l2(a, classical=True):
    t_grid = np.linspace(-0.499, 0.499, 1001)
    if classical:
        density = classical_self_conv_density(a, t_grid)
    else:
        density, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
        density = np.where(np.isnan(density), 0, density)
    dt = t_grid[1] - t_grid[0]
    return float((density ** 2).sum() * dt)


# ---------------------------------------------------------------------------
# (c) Investigate the analytic free convolution of arcsine on [-1/4, 1/4]
# Arcsine on [-r, r]: density 1 / (pi sqrt(r^2 - t^2)).
# Stieltjes:  G(z) = 1 / sqrt(z^2 - r^2)  (principal branch, real on z > r)
# Actually for arcsine on [-r, r]: G(z) = 1/sqrt(z^2 - r^2).
# R-transform: G^{-1}(w) = sqrt(1/w^2 + r^2) = (sqrt(1 + r^2 w^2))/w.
# So R(w) = (sqrt(1 + r^2 w^2) - 1) / w.
# 2R(w) = 2(sqrt(1 + r^2 w^2) - 1)/w. The free conv has density determined by
# inverting G + 2R = G^{-1}_{conv}.
# ---------------------------------------------------------------------------
def analytic_arcsine_free_conv():
    """For arcsine on [-r, r] with r=1/4:
    G_arc(z) = 1 / sqrt(z^2 - r^2)
    R_arc(w) = (sqrt(1 + r^2 w^2) - 1) / w
    G_freeconv(z) satisfies z = w + R_arc(w) where w = G_freeconv(z), R_arc(G_freeconv(z))
    Wait, properly: if G_{mu boxed_plus mu}(z) = w, then
       z = G^{-1}_{mu}(w) + R_mu(w) where G^{-1}_mu(w) - 1/w = R_mu(w)
    Combining: z = G^{-1}_mu(w) + (G^{-1}_mu(w) - 1/w) = 2 G^{-1}_mu(w) - 1/w.
    For arcsine: G^{-1}_arc(w) = sqrt(1/w^2 + r^2).
    So z = 2 sqrt(1/w^2 + r^2) - 1/w.
    Solve for w numerically along z = t + i eps for t in real line.
    """
    log("--- (c) Analytic arcsine free conv ---")
    r = 0.25
    t_grid = np.linspace(-0.499, 0.499, 401)
    eps = 1e-3

    rho_vals = []
    for t in t_grid:
        z = t + 1j * eps
        # solve z = 2 sqrt(1/w^2 + r^2) - 1/w for w in upper half plane
        # initial guess: w = 1/z (small w means z = 2*r - large_negative)
        w = 1.0 / z
        if np.imag(w) <= 0:
            w = np.real(w) + 1j * eps
        for it in range(80):
            sq = np.sqrt(1.0 / (w ** 2) + r ** 2)
            F = 2 * sq - 1.0 / w - z
            if abs(F) < 1e-12:
                break
            # dF/dw = 2 * (-1/w^3) / sq + 1/w^2  [careful with branches]
            dsq = -1.0 / (w ** 3 * sq)
            dF = 2 * dsq + 1.0 / (w ** 2)
            if abs(dF) < 1e-14:
                break
            w = w - F / dF
            if np.imag(w) <= 0:
                w = np.real(w) + 1j * (abs(np.imag(w)) + 1e-4)
        rho_vals.append(-np.imag(w) / np.pi if abs(F) < 1e-6 else np.nan)

    rho = np.array(rho_vals)
    valid = ~np.isnan(rho)
    log(f"  Analytic arcsine free conv: max = {np.nanmax(rho):.6f}, succeeded {valid.sum()}/{len(t_grid)}")
    log(f"  Note: arcsine is NOT supported on [-1/4, 1/4] in the proper sense")
    log(f"       (it has weight diverging at endpoints), but supp is [-1/4, 1/4].")

    # Check: classical conv of arcsine: f * f
    # arcsine has density 1/(pi sqrt(r^2 - t^2)).  Classical f * f peak?
    # f(t) = 1/(pi sqrt(r^2 - t^2)).  At t=0: ||f||_2^2 = int f^2 = int 1/(pi^2 (r^2-t^2)) dt
    # = (1/pi^2) * (1/r) * arctanh diverges at t=r. So arcsine is NOT in L^2.
    # Hence f * f is finite (but possibly unbounded) -- the discretization gives
    # a finite answer.
    return float(np.nanmax(rho))


# ---------------------------------------------------------------------------
# (d) MAIN INSIGHT INVESTIGATION:
# Does there exist mu on [-1/4, 1/4] with int=1 and ||mu * mu||_inf < 1.28?
# (Would refute current LB.)  We use our known minimizers and search over
# nearby configurations focusing on classical_max.
#
# Compare: what is min classical_max in our random search?  We saw 2.03,
# but the literature suggests the true minimum is near 1.28-1.5.
# Why the gap?  Our 'random' configs are not optimized; the minimizer concentrates
# mass into specific peaks/edges.
# ---------------------------------------------------------------------------
def classical_min_attempt(d=10, n_trials=80, seed=77):
    """Try to find classical configs with low max."""
    log(f"--- (d) Classical_max minimization d={d} n={n_trials} ---")
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(-0.499, 0.499, 401)
    half = (d + 1) // 2
    best = (np.inf, None)
    for trial in range(n_trials):
        # symmetric, concentrated structures
        h = rng.dirichlet(np.ones(half) * 0.3)
        if d % 2 == 0:
            a = np.concatenate([h[::-1], h])
        else:
            a = np.concatenate([h[1:][::-1], h])
        a = a / a.sum()
        cl = classical_self_conv_density(a, t_grid)
        c_max = cl.max()
        if c_max < best[0]:
            best = (c_max, a.copy())
        if (trial+1) % 20 == 0:
            log(f"  trial {trial+1}: best classical_max = {best[0]:.6f}")
    # local search around best
    if best[1] is not None:
        a = best[1].copy()
        for step in range(50):
            half = (d + 1) // 2
            delta = rng.normal(0, 0.01, half)
            if d % 2 == 0:
                delta_full = np.concatenate([delta[::-1], delta])
            else:
                delta_full = np.concatenate([delta[1:][::-1], delta])
            new_a = np.maximum(a + delta_full, 0)
            if new_a.sum() < 1e-6:
                continue
            new_a = new_a / new_a.sum()
            cl = classical_self_conv_density(new_a, t_grid)
            c_max = cl.max()
            if c_max < best[0]:
                best = (c_max, new_a.copy())
                a = new_a
        log(f"  After local search: best classical_max = {best[0]:.6f}")
    return best


# ---------------------------------------------------------------------------
# (e) Now compare free_max for the classical-minimizing configs
# ---------------------------------------------------------------------------
def free_at_classical_min(a):
    log(f"--- (e) Free_max at the classical minimizer ---")
    t_grid = np.linspace(-0.499, 0.499, 401)
    rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
    valid = ~np.isnan(rho_free)
    f_max = float(np.nanmax(rho_free)) if valid.sum() > 100 else None
    log(f"  At classical-near-min config: free_max = {f_max}")
    return f_max


def main():
    log("Begin probe4")
    out = {}

    # (c) analytic arcsine
    out["analytic_arcsine_free_max"] = analytic_arcsine_free_conv()

    # (d) try to minimize classical
    bcl = classical_min_attempt()
    out["classical_min_attempt"] = {"best_classical_max": bcl[0]}
    if bcl[1] is not None:
        out["free_at_classical_min"] = free_at_classical_min(bcl[1])
        out["classical_min_attempt"]["a"] = bcl[1].tolist()
        # entropy of this config
        chi = free_entropy(bcl[1])
        log(f"  Free entropy of classical-min: chi = {chi:.4f}")
        out["free_entropy_at_min"] = chi

    out["compute_time_sec"] = time.time() - START
    p = HERE / "results_probe4.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    log(f"Wrote {p}")
    return out


if __name__ == "__main__":
    main()
