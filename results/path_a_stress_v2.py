"""
v2 stress test: faster, simpler search.
Strategy:
 - For each parametric family, do RANDOM SAMPLING (cheaper, parallel-safe).
 - Then local refine top-K candidates with scipy.optimize.minimize.
 - Avoid differential_evolution which is slow due to deep integration costs.
 - Track total wall-clock per family to avoid runaway.
"""
import numpy as np
import json, os, sys, time, signal
import warnings; warnings.filterwarnings('ignore')
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar

OUTDIR = os.path.dirname(os.path.abspath(__file__))
C0 = np.pi / 8

def safe_int(f, a, b, sing_pts=None, limit=400):
    if sing_pts is None: sing_pts = []
    pts = sorted(set([p for p in sing_pts if a < p < b]))
    eps = 1e-12
    if not pts:
        try:
            v, _ = quad(f, a, b, limit=limit)
            return v
        except Exception:
            return np.nan
    total = 0.0
    edges = [a] + pts + [b]
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        if i > 0: x0 = x0 + eps
        if i < len(edges)-2: x1 = x1 - eps
        try:
            v, _ = quad(f, x0, x1, limit=limit)
            total += v
        except Exception:
            return np.nan
    return total

def conv_at(f, t, sing_pts_f=None):
    a = max(-0.25, t - 0.25); b = min(0.25, t + 0.25)
    if a >= b: return 0.0
    if sing_pts_f is None: sing_pts_f = []
    pts = list(sing_pts_f) + [t - p for p in sing_pts_f]
    pts = [p for p in pts if a < p < b]
    return safe_int(lambda x: f(x)*f(t-x), a, b, sing_pts=pts, limit=400)

def sup_conv(f, sing_pts_f=None, n_scan=120, zoom_iters=2):
    ts = np.linspace(-0.499, 0.499, n_scan)
    vals = np.array([conv_at(f, t, sing_pts_f=sing_pts_f) for t in ts])
    if not np.all(np.isfinite(vals)): return np.inf
    idx = int(np.argmax(vals))
    best_t = ts[idx]; best_v = vals[idx]
    width = (ts[1]-ts[0]) * 4
    for _ in range(zoom_iters):
        lo = max(-0.499, best_t - width); hi = min(0.499, best_t + width)
        try:
            res = minimize_scalar(lambda t: -conv_at(f, t, sing_pts_f=sing_pts_f),
                                  bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-7, 'maxiter': 30})
            v = -res.fun
            if v > best_v:
                best_v = v; best_t = res.x
        except Exception:
            pass
        width *= 0.25
    return best_v

def ratio_of(f, sing_pts_f=None):
    I  = safe_int(f, -0.25, 0.25, sing_pts=sing_pts_f)
    if not (np.isfinite(I) and I > 0): return np.inf, None
    g = lambda x: max(f(x), 0.0)**1.5
    I3 = safe_int(g, -0.25, 0.25, sing_pts=sing_pts_f)
    if not (np.isfinite(I3) and I3 > 0): return np.inf, None
    S  = sup_conv(f, sing_pts_f=sing_pts_f)
    if not np.isfinite(S): return np.inf, None
    return S * I / (I3**2), dict(I=I, I3=I3, S=S)

# ---------------- families ----------------

def fam_alpha(p):
    a, b = p
    if not (0 < a < 0.5 - 1e-4 and 0.5 <= b <= 1.0): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        v = b - 2*x
        return v**(-a) if v > 0 else 0.0
    sing = [b/2] if -0.25 <= b/2 <= 0.25 else []
    r, _ = ratio_of(f, sing_pts_f=sing)
    return r

def fam_alpha_left(p):
    a, b = p
    if not (0 < a <= 0.5 and 0.5 <= b <= 1.0): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        v = b + 2*x
        return v**(-a) if v > 0 else 0.0
    sing = [-b/2] if -0.25 <= -b/2 <= 0.25 else []
    r, _ = ratio_of(f, sing_pts_f=sing)
    return r

def fam_two_endpoints(p):
    """w*(2x+0.5)^{-a1} + (1-w)*(0.5-2x)^{-a2}"""
    a1, a2, w = p
    if not (0 < a1 <= 0.5 and 0 < a2 <= 0.5): return np.inf
    if not (0 <= w <= 1): return np.inf
    def f(x):
        if x <= -0.25 or x >= 0.25: return 0.0
        return w*(2*x+0.5)**(-a1) + (1-w)*(0.5-2*x)**(-a2)
    sing = [-0.25, 0.25]
    r, _ = ratio_of(f, sing_pts_f=sing)
    return r

def fam_ss_perturb(p):
    eps, c, sigma = p
    if not (-2 <= eps <= 5 and -0.24 <= c <= 0.24 and 0.005 <= sigma <= 0.5): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        ss = (2*x + 0.5)**(-0.5)
        bump = eps*np.exp(-0.5*((x-c)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        v = ss + bump
        return max(v, 0)
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def fam_alpha_plus_unif(p):
    a, u = p
    if not (0.05 <= a <= 0.5 and 0 <= u <= 5): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        return (2*x+0.5)**(-a) + u
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def fam_alpha_plus_lin(p):
    a, A, B = p
    if not (0.05 <= a <= 0.5 and -2 <= A <= 5 and -2 <= B <= 5): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        v = (2*x+0.5)**(-a) + A + B*(x+0.25)
        return max(v, 0)
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def fam_double_power(p):
    a1, a2, s = p
    if not (0.05 <= a1 <= 0.5 and 0.05 <= a2 <= 0.5 and 0 <= s <= 5): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        u = 2*x + 0.5
        return u**(-a1) + s*u**(-a2)
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def fam_beta(p):
    p1, p2 = p
    if not (-0.5 <= p1 <= 3 and -0.5 <= p2 <= 3): return np.inf
    def f(x):
        if x <= -0.25 or x >= 0.25: return 0.0
        u = 2*x+0.5; v = 0.5-2*x
        if u<=0 or v<=0: return 0.0
        return u**p1 * v**p2
    sing = []
    if p1 < 0: sing.append(-0.25)
    if p2 < 0: sing.append(0.25)
    r, _ = ratio_of(f, sing_pts_f=sing)
    return r

def fam_smooth(p):
    c, k = p
    if not (-1 <= c <= 1 and 0.05 <= k <= 8): return np.inf
    def f(x):
        if x < -0.25 or x > 0.25: return 0.0
        u = 4*x - c
        v = 1 - u*u
        return v**k if v > 0 else 0.0
    r, _ = ratio_of(f, sing_pts_f=[])
    return r

def fam_combo(p):
    w0, wu, wb, c, sg = p
    if not (0 <= w0 <= 1 and 0 <= wu <= 1 and 0 <= wb <= 1): return np.inf
    if not (-0.25 <= c <= 0.25 and 0.005 <= sg <= 0.5): return np.inf
    if w0+wu+wb < 1e-6: return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        v = wu*2.0
        if w0>0: v += w0*(2*x+0.5)**(-0.5)
        if wb>0: v += wb*np.exp(-0.5*((x-c)/sg)**2)/(sg*np.sqrt(2*np.pi))
        return v
    sing = [-0.25] if w0>0 else []
    r, _ = ratio_of(f, sing_pts_f=sing)
    return r

def fam_piecewise(p):
    """N=len(p) piecewise constant on equal intervals of [-0.25, 0.25]."""
    vals = np.maximum(np.array(p), 0)
    if vals.sum() < 1e-6: return np.inf
    N = len(vals)
    nodes = np.linspace(-0.25, 0.25, N+1)
    def f(x):
        if x < -0.25 or x > 0.25: return 0.0
        idx = int(min(N-1, max(0, np.floor((x+0.25)/0.5*N))))
        return vals[idx]
    r, _ = ratio_of(f, sing_pts_f=[])
    return r

def fam_mv_style(p):
    """Two pieces, each (b - 2|x|)^{-a} but flexible, scale-invariant."""
    a1, a2, w, split = p
    if not (0 < a1 <= 0.5 and 0 < a2 <= 0.5): return np.inf
    if not (0 <= w <= 1 and -0.24 < split < 0.24): return np.inf
    def f(x):
        if x <= -0.25 or x >= 0.25: return 0.0
        if x < split:
            return w*(2*x+0.5)**(-a1)
        else:
            return (1-w)*(0.5-2*x)**(-a2)
    sing = [-0.25, 0.25]
    r, _ = ratio_of(f, sing_pts_f=sing)
    return r

# ---------------- driver ----------------
def random_search(fn, bounds, n=300, seed=0):
    rng = np.random.default_rng(seed)
    bs = np.array(bounds); lo, hi = bs[:,0], bs[:,1]
    best = np.inf; best_x = None; nfin = 0
    for _ in range(n):
        x = lo + (hi-lo)*rng.random(len(bounds))
        v = fn(x)
        if np.isfinite(v):
            nfin += 1
            if v < best: best = v; best_x = x.tolist()
    return best, best_x, nfin

def local_refine(fn, x0, bounds, maxiter=50):
    try:
        res = minimize(fn, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < fn(x0):
            return float(res.fun), res.x.tolist()
    except Exception:
        pass
    return fn(x0), x0

def study(name, fn, bounds, n_random=300, n_refine=4, seed=0, max_time=180):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    bs = np.array(bounds); lo, hi = bs[:,0], bs[:,1]
    samples = []
    nfin = 0
    for i in range(n_random):
        if time.time() - t0 > max_time:
            print(f"  random search timed out after {i} iters", flush=True)
            break
        x = lo + (hi-lo)*rng.random(len(bounds))
        v = fn(x)
        if np.isfinite(v):
            nfin += 1
            samples.append((v, x.tolist()))
    samples.sort(key=lambda s: s[0])
    rs_best = samples[0] if samples else (np.inf, None)
    print(f"  random {nfin}/{n_random} finite, best ratio = {rs_best[0]:.6f}", flush=True)

    # local refine top-K
    refined = []
    for k in range(min(n_refine, len(samples))):
        if time.time() - t0 > max_time:
            print(f"  local refine timed out at k={k}", flush=True)
            break
        v0, x0 = samples[k]
        v1, x1 = local_refine(fn, x0, bounds, maxiter=40)
        refined.append((v1, x1))
        print(f"    refine {k}: {v0:.6f} -> {v1:.6f}", flush=True)
    refined.sort(key=lambda s: s[0])
    best = refined[0] if refined else rs_best
    elapsed = time.time() - t0
    print(f"  best after refine: {best[0]:.6f}, time={elapsed:.1f}s", flush=True)
    return dict(name=name, n_random=n_random, n_finite=nfin,
                rs_best_fun=float(rs_best[0]) if np.isfinite(rs_best[0]) else None,
                rs_best_x=rs_best[1],
                best_fun=float(best[0]) if np.isfinite(best[0]) else None,
                best_x=best[1], time=elapsed,
                top5_random=[(float(v), x) for v, x in samples[:5]])

def main():
    overall = np.inf; overall_info = None
    out = {'c0': C0, 'studies': []}
    t_start = time.time()

    studies = [
        ('alpha', fam_alpha, [(0.05, 0.499), (0.5, 1.0)], 250, 4, 60),
        ('alpha_left', fam_alpha_left, [(0.05, 0.5), (0.5, 1.0)], 250, 4, 60),
        ('two_endpoints', fam_two_endpoints, [(0.05, 0.5), (0.05, 0.5), (0,1)], 350, 5, 90),
        ('ss_perturb', fam_ss_perturb, [(-2, 5), (-0.24, 0.24), (0.005, 0.5)], 400, 6, 100),
        ('alpha_plus_unif', fam_alpha_plus_unif, [(0.05, 0.5), (0, 5)], 250, 4, 60),
        ('alpha_plus_lin', fam_alpha_plus_lin, [(0.05, 0.5), (-2, 5), (-2, 5)], 350, 5, 80),
        ('double_power', fam_double_power, [(0.05, 0.5), (0.05, 0.5), (0, 5)], 350, 5, 80),
        ('beta', fam_beta, [(-0.5, 3), (-0.5, 3)], 250, 4, 60),
        ('smooth', fam_smooth, [(-1, 1), (0.05, 8)], 250, 4, 50),
        ('combo', fam_combo, [(0,1)]*3 + [(-0.25, 0.25), (0.005, 0.5)], 350, 5, 80),
        ('mv_style', fam_mv_style, [(0.05, 0.5), (0.05, 0.5), (0,1), (-0.24, 0.24)], 350, 5, 90),
        ('piecewise8', fam_piecewise, [(0.0, 5.0)]*8, 400, 5, 80),
        ('piecewise16', fam_piecewise, [(0.0, 5.0)]*16, 400, 5, 100),
        ('piecewise32', fam_piecewise, [(0.0, 5.0)]*32, 400, 5, 120),
    ]
    for name, fn, bnds, nr, nf, mt in studies:
        s = study(name, fn, bnds, n_random=nr, n_refine=nf, seed=hash(name)%(2**31), max_time=mt)
        out['studies'].append(s)
        if s.get('best_fun') is not None and s['best_fun'] < overall:
            overall = s['best_fun']; overall_info = dict(family=name, x=s['best_x'], ratio=s['best_fun'])

    # Now mass random sampling: heavy mixed family
    print("\n=== Mass random across families ===", flush=True)
    rng = np.random.default_rng(123)
    fns_with_b = [
        (fam_alpha, [(0.05, 0.499), (0.5, 1.0)]),
        (fam_alpha_left, [(0.05, 0.5), (0.5, 1.0)]),
        (fam_two_endpoints, [(0.05, 0.5), (0.05, 0.5), (0,1)]),
        (fam_ss_perturb, [(-2, 5), (-0.24, 0.24), (0.005, 0.5)]),
        (fam_alpha_plus_unif, [(0.05, 0.5), (0, 5)]),
        (fam_double_power, [(0.05, 0.5), (0.05, 0.5), (0, 5)]),
        (fam_beta, [(-0.5, 3), (-0.5, 3)]),
        (fam_combo, [(0,1)]*3 + [(-0.25, 0.25), (0.005, 0.5)]),
        (fam_mv_style, [(0.05, 0.5), (0.05, 0.5), (0,1), (-0.24, 0.24)]),
    ]
    n_mass = 1500
    n_fin_mass = 0
    best_mass = np.inf; best_mass_info = None
    t_mass = time.time()
    cap_time_mass = 360
    for i in range(n_mass):
        if time.time() - t_mass > cap_time_mass:
            print(f"  mass random capped at {i} iters", flush=True)
            break
        fn, bnds = fns_with_b[rng.integers(0, len(fns_with_b))]
        bs = np.array(bnds); lo, hi = bs[:,0], bs[:,1]
        x = lo + (hi-lo)*rng.random(len(bnds))
        v = fn(x)
        if np.isfinite(v):
            n_fin_mass += 1
            if v < best_mass:
                best_mass = v
                best_mass_info = dict(fn=fn.__name__, x=x.tolist(), ratio=float(v))
    out['mass_random'] = dict(n=n_mass, n_finite=n_fin_mass,
                               best=float(best_mass) if np.isfinite(best_mass) else None,
                               best_info=best_mass_info, time=time.time()-t_mass)
    print(f"  mass: {n_fin_mass}/{n_mass} finite, best={best_mass:.6f}", flush=True)
    if np.isfinite(best_mass) and best_mass < overall:
        overall = best_mass; overall_info = best_mass_info

    out['overall_min'] = overall
    out['overall_min_info'] = overall_info
    out['elapsed'] = time.time() - t_start

    print(f"\n\n=== OVERALL MIN: {overall:.6f}, c0=pi/8={C0:.6f}, margin={overall-C0:.2e} ===")
    if overall_info: print(f"  by: {overall_info}")

    with open(os.path.join(OUTDIR, 'path_a_stress_test.json'), 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\nWrote path_a_stress_test.json. elapsed={out['elapsed']:.1f}s")

if __name__ == '__main__':
    main()
