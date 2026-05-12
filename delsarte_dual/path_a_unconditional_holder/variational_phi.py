"""
Variational ascent on Phi(M) := sup { ||f*f||_2^2 / ||f*f||_inf : f admissible, ||f*f||_inf <= M }.

Hyp_R(c=log16/pi, M_max) holds iff Phi(M) <= 0.88254 for all M <= M_max.

We compute Phi numerically with multiple parameterizations:
  - Sum of K Gaussians (centers, widths, weights)
  - K-step (atomic) mixtures (positions, weights)
  - K-spline / piecewise-linear bumps (knots)
  - Adversarial: subsampled Boyer-Li witness, scaled

Strategy: maximize ||f*f||_2^2 / ||f*f||_inf subject to:
  - f >= 0 (parametrically guaranteed)
  - integral f = 1 (renormalize)
  - support in [-1/4, 1/4]
  - ||f*f||_inf <= M (penalty)

We use scipy.optimize with multiple random restarts, both symmetric and asymmetric inits.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import json
import sys

# Numerical grid: f sampled on [-1/4, 1/4] -> N points
# f*f sampled on [-1/2, 1/2] -> 2N-1 points
N = 1024  # grid size on [-1/4, 1/4]
DX = 0.5 / N  # step size  (interval length 1/2 over N points; centers shifted)
xs = np.linspace(-0.25 + DX/2, 0.25 - DX/2, N)


def conv_self(f):
    """Compute (f*f) on grid of length 2N-1, with proper scaling."""
    # Discrete convolution * dx gives Riemann approximation to integral
    g = np.convolve(f, f) * DX  # length 2N-1
    return g


def metrics(f):
    """Returns (||f*f||_inf, ||f*f||_2^2, ||f*f||_1, c_emp = L2^2/Linf)."""
    g = conv_self(f)
    Linf = np.max(g)
    L2sq = np.sum(g * g) * DX
    L1 = np.sum(g) * DX
    return Linf, L2sq, L1, L2sq / Linf if Linf > 0 else 0


def normalize_f(f):
    """Scale f so that integral f * dx = 1. Clip to nonneg first."""
    f = np.maximum(f, 0.0)
    s = np.sum(f) * DX
    if s <= 1e-12:
        return None
    return f / s


# === Parameterization 1: Sum of K Gaussians ===
def f_from_gaussians(params, K):
    """params: 3K floats: [logweights, centers (in [-1/4,1/4]), log_widths]"""
    lw = params[:K]
    cs = params[K:2*K]
    lws = params[2*K:3*K]
    w = np.exp(lw)
    sigmas = np.exp(lws)
    f = np.zeros_like(xs)
    for i in range(K):
        f += w[i] * np.exp(-0.5 * ((xs - cs[i])/sigmas[i])**2) / (sigmas[i] * np.sqrt(2*np.pi))
    return normalize_f(f)


# === Parameterization 2: Piecewise-constant (K-step) ===
def f_from_steps(params, K):
    """params: K nonneg log-weights. f = exp(params[i]) on i-th block."""
    w = np.exp(params)
    block = N // K
    f = np.zeros_like(xs)
    for i in range(K):
        s = i * block
        e = (i+1) * block if i < K-1 else N
        f[s:e] = w[i]
    return normalize_f(f)


# === Parameterization 3: B-spline knots / piecewise-linear ===
def f_from_pwlinear(params, K):
    """params: K nonneg log-values at K equispaced knots."""
    vals = np.exp(params)
    knots = np.linspace(-0.25, 0.25, K)
    f = np.interp(xs, knots, vals)
    return normalize_f(f)


# === Adversarial: scaled Boyer-Li witness ===
def load_BL():
    with open('delsarte_dual/restricted_holder/coeffBL.txt') as fh:
        s = fh.read().strip().strip('{}').split(',')
    return np.array([int(x) for x in s], dtype=float)


def f_from_BL_truncated(scale_factor=1.0, drop_tail=False):
    """Boyer-Li witness, optionally truncate the long tail. The full BL witness has
    M ~ 1.652. Truncating its tail reduces M.

    scale_factor = relative width of support (1.0 = full BL on [-1/4,1/4]).
    drop_tail: if True, truncate to first ~50 nonzero coefficients (the dense head).
    """
    v = load_BL()
    if drop_tail:
        # Keep first 50 + last 350 (the dense regions, drop sparse middle).
        # BL has dense head (0..50), sparse middle, dense tail.
        v_head = v[:50]
        v_tail = v[-350:]
        v_full = np.concatenate([v_head, v_tail])
    else:
        v_full = v
    L = len(v_full)
    # Map onto [-1/4, 1/4] -> N grid by aggregating BL bins
    # support of original = [0, 575] -> shift by -287.5 -> [-287.5, 287.5]
    # rescale to [-1/4, 1/4]: x_BL_in_unit = (k - L/2) / (2L) gives values in [-1/4, 1/4)
    # We want f on grid xs. Each BL coefficient v[k] is mass v[k] over an interval of width 1/(2L).
    # Convert to density: density at xs[i] = v[k] * (2L) where k is the BL bin xs[i] falls in.
    # For sub-support: scale_factor < 1 means concentrate in [-scale_factor/4, scale_factor/4].
    half = 0.25 * scale_factor
    # bin j in BL covers x in [-half + j*(2*half/L), -half + (j+1)*(2*half/L))
    f = np.zeros_like(xs)
    for j in range(L):
        x_lo = -half + j * (2*half/L)
        x_hi = -half + (j+1) * (2*half/L)
        # find xs in this bin
        mask = (xs >= x_lo) & (xs < x_hi)
        f[mask] = v_full[j]
    return normalize_f(f)


# === The ascent objective ===
def neg_obj(params, K, parametrization, M_target, penalty=1e3):
    """We want to MAXIMIZE c_emp = L2sq/Linf subject to Linf <= M_target.
    Returns: -c_emp + penalty * max(0, Linf - M_target)^2.
    """
    if parametrization == 'gauss':
        f = f_from_gaussians(params, K)
    elif parametrization == 'step':
        f = f_from_steps(params, K)
    elif parametrization == 'pwlin':
        f = f_from_pwlinear(params, K)
    else:
        raise ValueError(parametrization)
    if f is None:
        return 1e6
    Linf, L2sq, L1, c_emp = metrics(f)
    # Penalty: violating Linf > M_target is bad
    pen = penalty * max(0.0, Linf - M_target)**2
    return -c_emp + pen


def init_params(K, parametrization, asymmetric=False, seed=0):
    rng = np.random.default_rng(seed)
    if parametrization == 'gauss':
        # log-weights, centers, log-widths
        lw = rng.normal(0.0, 0.5, K)
        if asymmetric:
            cs = rng.uniform(-0.20, 0.20, K)
        else:
            # symmetric: pair up
            half = (K+1)//2
            tmp = rng.uniform(0.0, 0.20, half)
            cs = np.concatenate([-tmp, tmp[:K-half]])
        lws = rng.normal(np.log(0.05), 0.3, K)
        return np.concatenate([lw, cs, lws])
    elif parametrization == 'step':
        return rng.normal(0.0, 0.5, K)
    elif parametrization == 'pwlin':
        return rng.normal(0.0, 0.5, K)


def ascend(K, parametrization, M_target, n_restarts=8, asymmetric=True, verbose=False):
    """Run ascent with multiple restarts, return best (c_emp, params, f, info)."""
    best = (-np.inf, None, None, None)
    for r in range(n_restarts):
        x0 = init_params(K, parametrization, asymmetric=asymmetric, seed=r)
        try:
            res = minimize(neg_obj, x0, args=(K, parametrization, M_target),
                           method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-9})
        except Exception as e:
            if verbose:
                print(f"  restart {r}: error {e}")
            continue
        # Compute final metrics (no penalty)
        if parametrization == 'gauss':
            f = f_from_gaussians(res.x, K)
        elif parametrization == 'step':
            f = f_from_steps(res.x, K)
        elif parametrization == 'pwlin':
            f = f_from_pwlinear(res.x, K)
        if f is None:
            continue
        Linf, L2sq, L1, c_emp = metrics(f)
        # Reject if hard violation
        if Linf > M_target * 1.005:  # small tol
            continue
        if c_emp > best[0]:
            best = (c_emp, res.x, f, {'Linf': Linf, 'L2sq': L2sq, 'L1': L1, 'restart': r})
            if verbose:
                print(f"  restart {r}: c_emp={c_emp:.6f}, Linf={Linf:.4f}")
    return best


def asymmetry(f):
    """Measure of asymmetry: || f - f(-x) ||_1 / || f + f(-x) ||_1."""
    f_rev = f[::-1]
    num = np.sum(np.abs(f - f_rev)) * DX
    den = np.sum(np.abs(f + f_rev)) * DX
    return num / den if den > 0 else 0


def support_extent(f, threshold_frac=0.001):
    """Return (xmin, xmax) where f > threshold * max(f)."""
    thr = threshold_frac * np.max(f)
    mask = f > thr
    if not mask.any():
        return (0, 0)
    idx = np.where(mask)[0]
    return xs[idx[0]], xs[idx[-1]]


def describe_f(f, c_emp, M):
    Linf, L2sq, L1, _ = metrics(f)
    asym = asymmetry(f)
    sup_lo, sup_hi = support_extent(f)
    # peak location of f
    peak_x = xs[np.argmax(f)]
    # peak location of f*f
    g = conv_self(f)
    g_xs = np.linspace(-0.5, 0.5, 2*N-1)
    peak_g_x = g_xs[np.argmax(g)]
    return {
        'c_emp': float(c_emp),
        'M': float(Linf),
        'L2sq': float(L2sq),
        'L1': float(L1),
        'asymmetry': float(asym),
        'support_lo': float(sup_lo),
        'support_hi': float(sup_hi),
        'support_width': float(sup_hi - sup_lo),
        'peak_f_x': float(peak_x),
        'peak_g_x': float(peak_g_x),
        'f_max': float(np.max(f)),
        'K_eq_norm_f_sq': float(np.sum(f*f) * DX),  # ||f||_2^2
    }


def adversarial_BL_attempts(M_target):
    """Try adversarial constructions: BL-like, scaled to fit in M_target."""
    results = []
    # Strategy 1: full BL truncated to head
    for drop_tail in [True, False]:
        for sf in [0.5, 0.7, 0.85, 1.0]:
            try:
                f = f_from_BL_truncated(scale_factor=sf, drop_tail=drop_tail)
                if f is None:
                    continue
                Linf, L2sq, L1, c_emp = metrics(f)
                # Rescale: if Linf > M_target, we cannot use this directly.
                # But we can mix BL with a flat background to lower Linf.
                # f_mix = alpha * f_BL + (1-alpha) * indicator(2)
                # Linf scales roughly by alpha^2.
                # Try several alphas
                ind = np.full_like(xs, 2.0)  # f = 2 on [-1/4,1/4], integrates to 1
                for alpha in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    f_mix = alpha * f + (1-alpha) * ind
                    f_mix = normalize_f(f_mix)
                    if f_mix is None:
                        continue
                    Linf2, L2sq2, L1_2, c_emp2 = metrics(f_mix)
                    if Linf2 <= M_target * 1.005:
                        results.append((c_emp2, f_mix, {'kind': 'BL_mix',
                                                          'drop_tail': drop_tail,
                                                          'sf': sf, 'alpha': alpha,
                                                          'Linf': Linf2}))
            except Exception:
                continue
    return results


def main():
    # Hyp_R threshold
    c_star = np.log(16) / np.pi  # ~0.88254
    print(f"Threshold c_star = log(16)/pi = {c_star:.10f}")
    print(f"Grid: N={N}, dx={DX:.6e}, support [-1/4,1/4]")
    print()

    M_grid = [1.27, 1.30, 1.33, 1.35, 1.378]
    out = {'c_star': c_star, 'N': N, 'results': {}}

    for M in M_grid:
        print(f"=" * 60)
        print(f"M_target = {M}")
        print(f"=" * 60)
        best_overall = (-np.inf, None, None)
        # Multiple parameterizations & K
        configs = [
            ('gauss', 3), ('gauss', 5), ('gauss', 7), ('gauss', 10),
            ('step', 8), ('step', 16), ('step', 32),
            ('pwlin', 8), ('pwlin', 16), ('pwlin', 24),
        ]
        for (param, K) in configs:
            for asym in [False, True]:
                t0 = time.time()
                c_emp, x, f, info = ascend(K, param, M, n_restarts=6, asymmetric=asym, verbose=False)
                if f is None:
                    continue
                tag = f"{param}-K{K}-{'asy' if asym else 'sym'}"
                print(f"  {tag:20s}: c_emp={c_emp:.6f}, Linf={info['Linf']:.4f}, t={time.time()-t0:.1f}s")
                if c_emp > best_overall[0]:
                    best_overall = (c_emp, f, {'param': param, 'K': K, 'asym': asym, **info})
        # Adversarial attempts
        adv = adversarial_BL_attempts(M)
        for (c_emp, f, info) in adv:
            if c_emp > best_overall[0]:
                best_overall = (c_emp, f, info)
                print(f"  ADV {info['kind']:20s}: c_emp={c_emp:.6f}, Linf={info['Linf']:.4f}")
        # Best for this M
        c_best, f_best, info_best = best_overall
        desc = describe_f(f_best, c_best, M)
        print(f"\n  BEST at M={M}: c_emp={c_best:.6f} ({'>' if c_best > c_star else '<='} c_star={c_star:.6f}), info={info_best}")
        print(f"  Worst-case f description: {desc}")
        out['results'][str(M)] = {
            'best_c_emp': float(c_best),
            'best_info': {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in info_best.items()},
            'description': desc,
            'distance_to_threshold': float(c_star - c_best),
        }

    # Save
    with open('delsarte_dual/path_a_unconditional_holder/phi_curve.json', 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nSaved to phi_curve.json")

    # VERDICT
    print("\n" + "=" * 60)
    print("PHI(M) CURVE")
    print("=" * 60)
    print(f"  c_star (threshold) = {c_star:.6f}")
    for M in M_grid:
        cb = out['results'][str(M)]['best_c_emp']
        margin = c_star - cb
        verdict = "PROMISING" if margin > 0.01 else ("NEAR-CRITICAL" if margin > -0.001 else "DISPROVED")
        print(f"  M={M}: Phi_emp >= {cb:.6f}, margin = {margin:+.6f}, {verdict}")

    return out


if __name__ == '__main__':
    out = main()
