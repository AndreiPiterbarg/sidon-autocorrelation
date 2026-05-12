"""
Extra adversarial tests for the Sidon Reverse Young conjecture, focused on
families that COULD plausibly violate. Run after the main stress test.

Approach idea: at SS we have ratio = pi/8 ~ 0.3927. This case is highly
asymmetric (mass concentrated at one endpoint). We probe perturbations:
  - SS + delta * (perturbation localized in interior)
  - Two singular endpoints, asymmetric heights
  - 1-parameter SS-style: f(x) = (2x+1/2)^{-a}, scan a in (0, 1/2]
  - Add small bumps in the bulk
  - Mass distributions formed by power-law family that beats SS
  - Tighter sup grid near the suspected peak
"""
import numpy as np
import json, os, time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_a_stress_test import (ratio, conv_at, sup_conv, integrate_f,
                                 integrate_f_15, make_alpha, C0)
from scipy.optimize import differential_evolution, minimize_scalar

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# 1. Perturbed SS: f(x) = (2x+0.5)^{-1/2} + eps * g(x)
def fam_ss_perturb(p):
    """eps, c, sigma, k for bump c, width sigma, exponent k. Bump amp = eps; can be neg.
    Negative perturbation must keep f >= 0."""
    eps, c, sigma = p
    if not (-2.0 <= eps <= 5.0): return np.inf
    if not (-0.24 <= c <= 0.24): return np.inf
    if not (0.005 <= sigma <= 0.5): return np.inf
    def f(x):
        if not (-0.25 < x <= 0.25): return 0.0
        ss = (2*x + 0.5)**(-0.5)
        bump = eps * np.exp(-0.5*((x-c)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
        v = ss + bump
        return max(v, 0.0)
    sing = [-0.25]
    r, _ = ratio(f, sing_pts_f=sing, n_scan=400)
    return r if np.isfinite(r) else np.inf

# 2. Generalized SS: f(x) = (2x+0.5)^{-a} for a in (0, 1/2]
def fam_ss_alpha(p):
    a = p[0]
    if not (0.05 <= a <= 0.5): return np.inf
    def f(x):
        if not (-0.25 < x <= 0.25): return 0.0
        return (2*x + 0.5)**(-a)
    sing = [-0.25]
    r, _ = ratio(f, sing_pts_f=sing, n_scan=400)
    return r if np.isfinite(r) else np.inf

# 3. Two SS-like singular endpoints
def fam_two_endpoints(p):
    """f = w*(2x+0.5)^{-a1} + (1-w)*(0.5-2x)^{-a2}"""
    a1, a2, w = p
    if not (0.05 <= a1 <= 0.5 and 0.05 <= a2 <= 0.5): return np.inf
    if not (0 <= w <= 1): return np.inf
    def f(x):
        if not (-0.25 < x < 0.25): return 0.0
        return w*(2*x+0.5)**(-a1) + (1-w)*(0.5-2*x)**(-a2)
    sing = [-0.25, 0.25]
    r, _ = ratio(f, sing_pts_f=sing, n_scan=400)
    return r if np.isfinite(r) else np.inf

# 4. SS + general power on the OTHER side
def fam_ss_plus_alt(p):
    """f(x) = (2x+0.5)^{-1/2} + s * (b - 2x)^{-a} with b > 0.5"""
    s, a, b = p
    if not (0.0 <= s <= 5.0): return np.inf
    if not (0.05 <= a <= 0.5): return np.inf
    if not (0.5 <= b <= 1.0): return np.inf
    def f(x):
        if not (-0.25 < x <= 0.25): return 0.0
        v1 = (2*x+0.5)**(-0.5)
        v2 = (b - 2*x)**(-a) if (b - 2*x) > 0 else 0.0
        return v1 + s*v2
    sing = [-0.25]
    if b == 0.5: sing.append(0.25)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=400)
    return r if np.isfinite(r) else np.inf

# 5. f = (2x + b)^{-1/2}  i.e. SS shifted (peak NOT at exact endpoint)
def fam_ss_shifted(p):
    """f(x) = (2x + b)^{-a}, b can place singularity outside [-1/4, 1/4]"""
    a, b = p
    if not (0.05 <= a <= 0.5): return np.inf
    if not (0.5 <= b <= 1.5): return np.inf
    def f(x):
        if not (-0.25 < x <= 0.25): return 0.0
        v = 2*x + b
        return v**(-a) if v > 0 else 0.0
    sing = []
    if -0.25 <= -b/2 <= 0.25: sing.append(-b/2)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=400)
    return r if np.isfinite(r) else np.inf

# 6. Alpha + uniform background
def fam_alpha_plus_unif(p):
    a, u = p
    if not (0.05 <= a <= 0.5): return np.inf
    if not (0.0 <= u <= 5.0): return np.inf
    def f(x):
        if not (-0.25 < x <= 0.25): return 0.0
        return (2*x+0.5)**(-a) + u
    sing = [-0.25]
    r, _ = ratio(f, sing_pts_f=sing, n_scan=400)
    return r if np.isfinite(r) else np.inf

# 7. Alpha + linear ramp
def fam_alpha_plus_lin(p):
    a, A, B = p
    if not (0.05 <= a <= 0.5): return np.inf
    if not (-2.0 <= A <= 5.0 and -2.0 <= B <= 5.0): return np.inf
    def f(x):
        if not (-0.25 < x <= 0.25): return 0.0
        v = (2*x+0.5)**(-a) + A + B*(x+0.25)
        return max(v, 0)
    sing = [-0.25]
    r, _ = ratio(f, sing_pts_f=sing, n_scan=400)
    return r if np.isfinite(r) else np.inf

# 8. Sum of two power-laws at left edge: (2x+0.5)^{-a1} + s*(2x+0.5)^{-a2}
def fam_double_power(p):
    a1, a2, s = p
    if not (0.05 <= a1 <= 0.5 and 0.05 <= a2 <= 0.5): return np.inf
    if not (0 <= s <= 5.0): return np.inf
    def f(x):
        if not (-0.25 < x <= 0.25): return 0.0
        u = 2*x+0.5
        return u**(-a1) + s*u**(-a2)
    sing = [-0.25]
    r, _ = ratio(f, sing_pts_f=sing, n_scan=400)
    return r if np.isfinite(r) else np.inf

# 9. Beta-like family on [-1/4, 1/4]
def fam_beta(p):
    """f(x) = (2x+0.5)^{p1} * (0.5-2x)^{p2} with p1, p2 in [-1/2, 5]"""
    p1, p2 = p
    if not (-0.5 <= p1 <= 3.0 and -0.5 <= p2 <= 3.0): return np.inf
    def f(x):
        if not (-0.25 < x < 0.25): return 0.0
        u = 2*x+0.5; v = 0.5-2*x
        if u<=0 or v<=0: return 0.0
        return (u**p1) * (v**p2)
    sing = []
    if p1 < 0: sing.append(-0.25)
    if p2 < 0: sing.append(0.25)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=400)
    return r if np.isfinite(r) else np.inf

def main():
    results = {'c0': C0, 'tests': []}
    overall = np.inf
    overall_info = None

    def update(r, name, x):
        nonlocal overall, overall_info
        if np.isfinite(r) and r < overall:
            overall = r; overall_info = dict(family=name, x=list(x), ratio=r)

    runs = [
        ('ss_alpha', fam_ss_alpha, [(0.05, 0.5)], 200, 30),
        ('ss_perturb', fam_ss_perturb, [(-2.0, 5.0), (-0.24, 0.24), (0.005, 0.5)], 200, 30),
        ('two_endpoints', fam_two_endpoints, [(0.05, 0.5), (0.05, 0.5), (0, 1)], 250, 30),
        ('ss_plus_alt', fam_ss_plus_alt, [(0, 5), (0.05, 0.5), (0.5, 1.0)], 200, 30),
        ('ss_shifted', fam_ss_shifted, [(0.05, 0.5), (0.5, 1.5)], 200, 30),
        ('alpha_plus_unif', fam_alpha_plus_unif, [(0.05, 0.5), (0, 5)], 200, 30),
        ('alpha_plus_lin', fam_alpha_plus_lin, [(0.05, 0.5), (-2, 5), (-2, 5)], 200, 30),
        ('double_power', fam_double_power, [(0.05, 0.5), (0.05, 0.5), (0, 5)], 200, 30),
        ('beta', fam_beta, [(-0.5, 3), (-0.5, 3)], 200, 30),
    ]

    for name, fn, bnds, mit, ps in runs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        try:
            res = differential_evolution(fn, bnds, maxiter=mit, popsize=ps, seed=hash(name)%2**31,
                                          tol=1e-8, workers=1, init='sobol', updating='immediate', polish=True)
            print(f"  best ratio = {res.fun:.6f} at x={res.x.tolist()}, time={time.time()-t0:.1f}s")
            results['tests'].append(dict(family=name, fun=float(res.fun), x=res.x.tolist(),
                                          time=time.time()-t0, nfev=int(res.nfev)))
            update(res.fun, name, res.x)
        except Exception as e:
            print(f"  ERROR: {e}")
            results['tests'].append(dict(family=name, error=str(e)))

    # 1500+ random across all families
    print("\n=== Random 1500 across ss_perturb + two_endpoints + beta ===")
    rng = np.random.default_rng(2026)
    n_finite = 0; best = np.inf; best_info = None
    for i in range(1500):
        sel = rng.integers(0, 3)
        if sel == 0:
            x = [rng.uniform(-2, 5), rng.uniform(-0.24, 0.24), rng.uniform(0.005, 0.5)]
            r = fam_ss_perturb(x); fam='ss_perturb'
        elif sel == 1:
            x = [rng.uniform(0.05, 0.5), rng.uniform(0.05, 0.5), rng.uniform(0,1)]
            r = fam_two_endpoints(x); fam='two_endpoints'
        else:
            x = [rng.uniform(-0.5, 3), rng.uniform(-0.5, 3)]
            r = fam_beta(x); fam='beta'
        if np.isfinite(r):
            n_finite += 1
            if r < best:
                best = r; best_info = dict(family=fam, x=x, ratio=r)
    print(f"  random: {n_finite}/1500 finite, best={best:.6f}, fam={best_info['family'] if best_info else None}")
    results['random_1500'] = dict(n_finite=n_finite, best=float(best) if np.isfinite(best) else None,
                                    best_info=best_info)
    if np.isfinite(best): update(best, best_info['family'], best_info['x'])

    results['overall_min'] = overall
    results['overall_min_info'] = overall_info
    print(f"\n\n=== EXTRA OVERALL MIN: {overall:.6f}, c0 = {C0:.6f} ===")
    print(f"  margin: {overall - C0:.2e}")
    if overall_info:
        print(f"  achieved by: {overall_info['family']} at x={overall_info['x']}")

    out = os.path.join(OUTDIR, 'path_a_extra_adversarial.json')
    with open(out, 'w') as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nWrote {out}")

if __name__ == '__main__':
    main()
