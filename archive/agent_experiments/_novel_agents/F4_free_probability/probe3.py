"""Probe 3: deeper search for minimum free_max, AND careful study of the
support issue.

Key questions:
(1) The free convolution mu boxed_plus mu of mu on [-1/4, 1/4] has support
    exceeding [-1/2, 1/2]. So the SAME measure mu has DIFFERENT supremum norms
    for classical vs free conv -- crucially, free conv can spread mass outside
    [-1/2, 1/2] but the constraint is on supp_t |t| <= 1/2 in the original
    problem. Need to clarify what we're comparing.

(2) Despite (1), MAX of free conv density may still be a useful proxy.
    Refine the symmetric search to push lower than 1.28.

(3) Investigate: is min ||mu boxed_plus mu||_inf over mu on [-1/4, 1/4]
    BOUNDED below by some explicit positive constant? E.g., free entropy
    bounds, free Brunn-Minkowski.

(4) Compute: in random tests, does the relation
        ||mu boxed_plus mu||_inf >= c * ||mu * mu||_inf^p ?
    hold for some explicit (c, p)?
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
    line = f"[probe3 {elapsed:7.2f}s] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# (1) Larger symmetric search
# ---------------------------------------------------------------------------
def big_symmetric_search(d=10, n_trials=120, seed=99):
    log(f"--- (1) Big symmetric search d={d} n={n_trials} ---")
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(-0.499, 0.499, 401)
    half = (d + 1) // 2
    best_free = (np.inf, None, None)
    best_classical = (np.inf, None, None)

    # also track distribution
    free_maxes = []
    cl_maxes = []
    for trial in range(n_trials):
        # Diverse generation strategies
        choice = trial % 5
        if choice == 0:
            h = rng.dirichlet(np.ones(half) * 1.0)
        elif choice == 1:
            h = rng.dirichlet(np.ones(half) * 0.3)
        elif choice == 2:
            h = rng.dirichlet(np.ones(half) * 5.0)
        elif choice == 3:
            # exponentially weighted
            h = np.exp(rng.normal(0, 1.5, half))
            h = h / h.sum()
        else:
            # bimodal
            h = np.abs(rng.normal(0, 1, half)) + 0.01
            h[half//2:] *= 3
            h = h / h.sum()
        if d % 2 == 0:
            a = np.concatenate([h[::-1], h])
        else:
            a = np.concatenate([h[1:][::-1], h])
        a = a / a.sum()

        cl = classical_self_conv_density(a, t_grid)
        c_max = cl.max()
        cl_maxes.append(c_max)
        if c_max < best_classical[0]:
            best_classical = (c_max, a.copy(), cl)

        rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
        valid = ~np.isnan(rho_free)
        if valid.sum() < 200:
            continue
        f_max = float(np.nanmax(rho_free))
        free_maxes.append(f_max)
        if f_max < best_free[0]:
            best_free = (f_max, a.copy(), rho_free)

        if (trial + 1) % 20 == 0:
            log(f"  after {trial+1}: best free={best_free[0]:.4f}, best classical={best_classical[0]:.4f}")

    log(f"  Final: best free_max = {best_free[0]:.6f}")
    log(f"  Final: best classical_max = {best_classical[0]:.6f}")
    log(f"  Free distribution: min={min(free_maxes):.4f}, median={sorted(free_maxes)[len(free_maxes)//2]:.4f}, max={max(free_maxes):.4f}")
    return best_free, best_classical, free_maxes, cl_maxes


# ---------------------------------------------------------------------------
# (2) Local search around the best symmetric config
# ---------------------------------------------------------------------------
def local_perturbation_search(a_init, d, t_grid, n_steps=40, sigma=0.02, seed=11):
    log(f"--- (2) Local search around best, sigma={sigma}, n={n_steps} ---")
    rng = np.random.default_rng(seed)
    a = a_init.copy()
    rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
    valid = ~np.isnan(rho_free)
    f_max = float(np.nanmax(rho_free)) if valid.sum() > 200 else np.inf
    best = (f_max, a.copy())
    log(f"  initial f_max = {f_max:.6f}")
    for step in range(n_steps):
        # random symmetric perturbation
        half = (d + 1) // 2
        delta = rng.normal(0, sigma, half)
        if d % 2 == 0:
            delta_full = np.concatenate([delta[::-1], delta])
        else:
            delta_full = np.concatenate([delta[1:][::-1], delta])
        new_a = a + delta_full
        new_a = np.maximum(new_a, 0.0)
        if new_a.sum() < 1e-6:
            continue
        new_a = new_a / new_a.sum()
        rho_free, _, _ = free_self_conv_density(new_a, t_grid, eps=2e-3)
        valid = ~np.isnan(rho_free)
        if valid.sum() < 200:
            continue
        new_fmax = float(np.nanmax(rho_free))
        if new_fmax < best[0] - 1e-6:
            best = (new_fmax, new_a.copy())
            a = new_a
            log(f"    step {step}: improved to {new_fmax:.6f}")
        if (step + 1) % 10 == 0:
            log(f"    after step {step+1}: best = {best[0]:.6f}")
    return best


# ---------------------------------------------------------------------------
# (3) Test the structural relation:  ||free||_inf >= c ||classical||_inf^p ?
# ---------------------------------------------------------------------------
def correlation_study(free_maxes, cl_maxes):
    log("--- (3) Correlation between free and classical maxes ---")
    fm = np.array(free_maxes)
    cm = np.array(cl_maxes[:len(fm)])
    log(f"  N = {len(fm)} configurations")
    log(f"  Pearson correlation: {np.corrcoef(fm, cm)[0,1]:.4f}")
    # log-log linear fit:  log(fm) = a + b*log(cm)
    valid = (fm > 0) & (cm > 0)
    A = np.vstack([np.log(cm[valid]), np.ones(valid.sum())]).T
    b, a = np.linalg.lstsq(A, np.log(fm[valid]), rcond=None)[0]
    log(f"  Log-log fit: log(free_max) = {a:.4f} + {b:.4f} * log(classical_max)")
    log(f"  Equivalently: free_max ~ {np.exp(a):.4f} * classical_max^{b:.4f}")
    return {"pearson": float(np.corrcoef(fm, cm)[0,1]),
            "loglog_intercept": float(a), "loglog_slope": float(b)}


# ---------------------------------------------------------------------------
# (4) Test particular symmetric "MV-arcsine"-like measure (which is the
# classical-side conjectured minimizer for some related problems)
# ---------------------------------------------------------------------------
def mv_arcsine_test(d=20):
    """Discretize arcsine on [-1/4, 1/4] with d bins, compute classical and free."""
    log(f"--- (4) MV-arcsine discretized at d={d} ---")
    bin_centers = -0.25 + (np.arange(d) + 0.5) * (0.5 / d)
    # arcsine density on [-1/4, 1/4]:  1 / (pi sqrt(1/16 - t^2))
    arc = 1.0 / (np.pi * np.sqrt(np.maximum(1.0/16.0 - bin_centers ** 2, 1e-10)))
    a = arc * (0.5 / d)
    a = a / a.sum()
    t_grid = np.linspace(-0.499, 0.499, 401)
    cl = classical_self_conv_density(a, t_grid)
    log(f"  classical max = {cl.max():.4f}")
    rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
    valid = ~np.isnan(rho_free)
    log(f"  free solver: {valid.sum()}/{len(t_grid)}")
    f_max = float(np.nanmax(rho_free)) if valid.sum() > 100 else None
    log(f"  free max = {f_max}")
    return {"classical_max": cl.max(), "free_max": f_max, "a": a.tolist()}


# ---------------------------------------------------------------------------
# (5) Explicit insight: free convolution can put mass OUTSIDE [-1/2, 1/2].
#     But for the SIDON problem we care about classical sup norm in [-1/2, 1/2].
#     So find: among mu on [-1/4, 1/4] with int = 1, can ||mu boxed_plus mu||_inf
#     restricted to t in [-1/2, 1/2] be < 1.2802?
# ---------------------------------------------------------------------------
def restrict_to_unit(d=10, n_trials=40, seed=33):
    log(f"--- (5) ||mu boxed_plus mu||_inf restricted to |t|<=1/2 ---")
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(-0.499, 0.499, 401)
    best = (np.inf, None)
    for trial in range(n_trials):
        half = (d + 1) // 2
        h = rng.dirichlet(np.ones(half) * 0.5)
        if d % 2 == 0:
            a = np.concatenate([h[::-1], h])
        else:
            a = np.concatenate([h[1:][::-1], h])
        a = a / a.sum()
        rho_free, _, _ = free_self_conv_density(a, t_grid, eps=2e-3)
        valid = ~np.isnan(rho_free)
        if valid.sum() < 200:
            continue
        # restrict to |t| <= 1/2 (already)
        f_max = float(np.nanmax(rho_free))
        if f_max < best[0]:
            best = (f_max, a.copy())
        if (trial+1) % 10 == 0:
            log(f"  trial {trial+1}: best = {best[0]:.6f}")
    log(f"  Final: min free_max (on |t|<=1/2) = {best[0]:.6f}")
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("Begin probe3")
    out = {}

    # (1) Big symmetric search
    bf, bc, fm, cm = big_symmetric_search(d=10, n_trials=80)
    out["big_symmetric_search"] = {
        "best_free_max": bf[0],
        "best_classical_max": bc[0],
        "n_evaluated": len(fm),
    }

    # (2) Local search around best
    if bf[1] is not None:
        t_grid = np.linspace(-0.499, 0.499, 401)
        local = local_perturbation_search(bf[1], d=10, t_grid=t_grid, n_steps=30, sigma=0.02)
        out["local_search"] = {"best_free_max": local[0]}
        log(f"After local search around best: free_max = {local[0]:.6f}")

    # (3) Correlation
    out["correlation"] = correlation_study(fm, cm)

    # (4) Arcsine
    out["mv_arcsine"] = mv_arcsine_test()

    # (5) Restricted-to-[-1/2, 1/2] free max
    r = restrict_to_unit()
    out["restricted_min"] = {"best_free_max": r[0]}

    out["compute_time_sec"] = time.time() - START
    p = HERE / "results_probe3.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    log(f"Wrote {p}")
    return out


if __name__ == "__main__":
    main()
