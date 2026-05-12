"""
Exhaustive numerical stress-test for the conjectured reverse Young inequality:
   sup_{|t|<=1/2} (f*f)(t)  >=  c0 * ||f||_{3/2}^3,    c0 = pi/8

with f >= 0, supp f c [-1/4, 1/4], int f = 1, sup f*f finite.
Equality at f0(x) = (2x+1/2)^{-1/2}.

We MINIMIZE the ratio
   R(f) = sup(f*f) / ||f||_{3/2}^3 = sup(f*f) / (int f^{3/2})^2
where we drop the int f = 1 constraint since R is scale-invariant
(numerator has degree 2 in f, denominator has degree 3 in f^{1/2}? Let's check:
  sup f*f scales as lambda^2 if f -> lambda f.
  (int f^{3/2})^2 scales as lambda^3.
So R scales as 1/lambda. So we MUST normalize int f = 1, OR equivalently fix
the scaling somehow. Easiest: enforce int f = 1.)

Actually wait. With f -> lambda f, sup(f*f) -> lambda^2 * sup(f*f), and
||f||_{3/2}^3 = (int f^{3/2})^2 -> (lambda^{3/2})^2 * (int f^{3/2})^2 = lambda^3 * (...).
So R -> lambda^2 / lambda^3 * R = R/lambda. Thus R is NOT scale-invariant in lambda.
We must enforce int f = 1.

Equivalently, we can compute R for any f then for f normalized by lambda=1/int f
gives ratio scaling: R_norm = R_f * (int f)^1, so we just compute
  ratio = sup(f*f) * (int f) / (int f^{3/2})^2
to get the scale-invariant version. (Substitute lambda = 1/int f, ratio becomes
   sup(f*f)/lambda^2 * (1/lambda^3 * (int f^{3/2})^2)... let me redo carefully.)

Let g = f / (int f), so int g = 1. Then
  sup(g*g) = sup(f*f) / (int f)^2
  ||g||_{3/2}^3 = (int g^{3/2})^2 = ((int f^{3/2})/(int f)^{3/2})^2
              = (int f^{3/2})^2 / (int f)^3
  R(g) = sup(g*g)/||g||_{3/2}^3 = [sup(f*f)/(int f)^2] / [(int f^{3/2})^2/(int f)^3]
       = sup(f*f) * (int f) / (int f^{3/2})^2

So we use this scale-invariant form.
"""
import numpy as np
import json
import time
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, differential_evolution
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
C0 = np.pi / 8

# ---------------------------------------------------------------------------
# Core utilities: integrals & convolution
# ---------------------------------------------------------------------------
def safe_int(f, a, b, sing_pts=None, limit=2000):
    """Integrate f from a to b, splitting at singular points."""
    if sing_pts is None:
        sing_pts = []
    pts = sorted(set([p for p in sing_pts if a < p < b]))
    eps = 1e-13
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
        # nudge away from singularities
        if i > 0: x0 = x0 + eps
        if i < len(edges)-2: x1 = x1 - eps
        try:
            v, _ = quad(f, x0, x1, limit=limit)
            total += v
        except Exception:
            return np.nan
    return total

def integrate_f(f, sing_pts=None):
    """int_{-1/4}^{1/4} f."""
    return safe_int(f, -0.25, 0.25, sing_pts=sing_pts)

def integrate_f_15(f, sing_pts=None):
    """int_{-1/4}^{1/4} f^{3/2}."""
    g = lambda x: max(f(x), 0.0)**1.5
    return safe_int(g, -0.25, 0.25, sing_pts=sing_pts)

def conv_at(f, t, sing_pts_f=None):
    """(f*f)(t) = int f(x) f(t-x) dx over x in [-1/4,1/4] with t-x in [-1/4,1/4]."""
    a = max(-0.25, t - 0.25)
    b = min(0.25, t + 0.25)
    if a >= b: return 0.0
    if sing_pts_f is None: sing_pts_f = []
    # Singular points of integrand in x:
    # f(x) singular at p in sing_pts_f -> include p
    # f(t-x) singular at p -> at x = t-p
    pts = list(sing_pts_f) + [t - p for p in sing_pts_f]
    pts = [p for p in pts if a < p < b]
    return safe_int(lambda x: f(x)*f(t-x), a, b, sing_pts=pts, limit=2000)

def sup_conv(f, sing_pts_f=None, n_scan=200, zoom_iters=3):
    """Approximate sup_{t in [-1/2,1/2]} (f*f)(t).
    By symmetry of t -> -t when f is even-ish, we scan but always cover all of [-1/2,1/2]."""
    ts = np.linspace(-0.4999, 0.4999, n_scan)
    vals = np.array([conv_at(f, t, sing_pts_f=sing_pts_f) for t in ts])
    if not np.all(np.isfinite(vals)):
        return np.inf
    idx = int(np.argmax(vals))
    best_t = ts[idx]
    best_v = vals[idx]
    # zoom
    width = (ts[1] - ts[0]) * 4
    for _ in range(zoom_iters):
        lo = max(-0.4999, best_t - width)
        hi = min(0.4999, best_t + width)
        try:
            res = minimize_scalar(lambda t: -conv_at(f, t, sing_pts_f=sing_pts_f),
                                  bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            v = -res.fun
            if v > best_v:
                best_v = v; best_t = res.x
        except Exception:
            pass
        width *= 0.25
    return best_v

def ratio(f, sing_pts_f=None, n_scan=400):
    """Scale-invariant ratio = sup(f*f) * (int f) / (int f^{3/2})^2."""
    I  = integrate_f(f, sing_pts=sing_pts_f)
    I3 = integrate_f_15(f, sing_pts=sing_pts_f)
    if not (np.isfinite(I) and np.isfinite(I3) and I > 0 and I3 > 0):
        return np.nan, dict(I=I, I3=I3)
    S  = sup_conv(f, sing_pts_f=sing_pts_f, n_scan=n_scan)
    if not np.isfinite(S):
        return np.nan, dict(I=I, I3=I3, S=S)
    r = S * I / (I3**2)
    return r, dict(I=I, I3=I3, S=S)

# ---------------------------------------------------------------------------
# Schinzel-Schmidt analytic check
# ---------------------------------------------------------------------------
def f0(x):
    return (2*x + 0.5 + 1e-300)**(-0.5) if (2*x + 0.5) > 0 else 0.0

def test_ss():
    f = lambda x: 0.0 if x <= -0.25 else (2*x + 0.5)**(-0.5)
    r, info = ratio(f, sing_pts_f=[-0.25])
    return r, info

# ---------------------------------------------------------------------------
# PARAMETRIC FAMILIES
# ---------------------------------------------------------------------------

# Family 1: single-endpoint alpha
# f(x) = (b - 2x)^{-a} on (-1/4, 1/4), then optionally mirrored.
def make_alpha(a, b, mirror=False):
    """f(x) ~ (b-2x)^{-a} where if not mirrored, the singular endpoint is x=b/2.
    For singularity in [-1/4, 1/4] we want b/2 = 1/4, i.e. b=0.5 -> SS-like.
    More generally: domain (-1/4, 1/4), and (b-2x) > 0 throughout iff b > 2*(1/4) = 1/2.
    If b = 0.5 + delta, no singularity in interior; if b = 0.5 exactly, singular at x=1/4.
    If we want singular at x=-1/4 then mirror: f(x) = ((2x+0.5)+delta)^{-a}? Already SS if a=1/2.
    """
    # We allow b in (0.5, 1) so that (b-2x) in (b-0.5, b+0.5), strictly positive.
    # When mirrored: f(x) = (b + 2x)^{-a}, singularity at x = -b/2.
    if mirror:
        def f(x):
            v = b + 2*x
            return v**(-a) if v > 0 else 0.0
        sing = [-b/2] if -0.25 <= -b/2 <= 0.25 else []
    else:
        def f(x):
            v = b - 2*x
            return v**(-a) if v > 0 else 0.0
        sing = [b/2] if -0.25 <= b/2 <= 0.25 else []
    return f, sing

def fam_alpha(p):
    """p = [a, b]; constraints a in [0.05, 0.5], b in [0.5, 1.0]."""
    a, b = p
    if not (0 < a <= 0.5): return np.inf
    if not (0.5 <= b <= 1.0): return np.inf
    f, sing = make_alpha(a, b, mirror=False)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=150)
    return r if np.isfinite(r) else np.inf

# Family 2: two-piece (MV style)
# w * c1 (b1 - 2x)^{-a1} on (-1/4, 0)  +  (1-w) * c2 (b2 - 2x)^{-a2} on (0, 1/4)
# Actually scale-invariant form: just the unnormalized values, w in [0,1] is just a weight.
def make_twopiece(a1, b1, a2, b2, w, mirror1=False, mirror2=False):
    def piece(x, a, b, mir):
        if mir:
            v = b + 2*x
        else:
            v = b - 2*x
        return v**(-a) if v > 0 else 0.0
    sing = []
    if mirror1:
        if -0.25 <= -b1/2 <= 0.0: sing.append(-b1/2)
    else:
        if -0.25 <= b1/2 <= 0.0: sing.append(b1/2)
    if mirror2:
        if 0.0 <= -b2/2 <= 0.25: sing.append(-b2/2)
    else:
        if 0.0 <= b2/2 <= 0.25: sing.append(b2/2)
    def f(x):
        if x < 0:
            return w * piece(x, a1, b1, mirror1)
        else:
            return (1-w) * piece(x, a2, b2, mirror2)
    return f, sing

def fam_twopiece(p):
    a1, b1, a2, b2, w = p
    if not (0 < a1 <= 0.5 and 0 < a2 <= 0.5): return np.inf
    if not (0.5 <= b1 <= 1.0 and 0.5 <= b2 <= 1.0): return np.inf
    if not (0 <= w <= 1): return np.inf
    # try multiple mirror combos? For default fam, assume singular at -1/4 (left piece) and 1/4 (right piece) -> mirror1=True, mirror2=False
    f, sing = make_twopiece(a1, b1, a2, b2, w, mirror1=True, mirror2=False)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=150)
    return r if np.isfinite(r) else np.inf

# Family 3: three-piece
# split [-1/4, 1/4] into [-1/4, c1], [c1, c2], [c2, 1/4]
def make_threepiece(a1, a2, a3, b1, b2, b3, w1, w2, c1, c2, mirror_left=True, mirror_right=False):
    sing = []
    if mirror_left and -0.25 <= -b1/2 <= c1: sing.append(-b1/2)
    if mirror_right and c2 <= -b3/2 <= 0.25: sing.append(-b3/2)
    if (not mirror_right) and c2 <= b3/2 <= 0.25: sing.append(b3/2)
    def f(x):
        if x < c1:
            v = (b1 + 2*x) if mirror_left else (b1 - 2*x)
            return w1 * (v**(-a1) if v > 0 else 0.0)
        elif x < c2:
            v = (b2 + 2*x)
            return w2 * (v**(-a2) if v > 0 else 0.0)
        else:
            v = (b3 + 2*x) if mirror_right else (b3 - 2*x)
            wr = max(0.0, 1 - w1 - w2)
            return wr * (v**(-a3) if v > 0 else 0.0)
    return f, sing

def fam_threepiece(p):
    a1, a2, a3, b1, b2, b3, w1, w2, c1, c2 = p
    if not (0 < a1 <= 0.5 and 0 < a2 <= 0.5 and 0 < a3 <= 0.5): return np.inf
    if not (0.5 <= b1 <= 1.0 and 0.5 <= b2 <= 1.0 and 0.5 <= b3 <= 1.0): return np.inf
    if not (-0.25 < c1 < c2 < 0.25): return np.inf
    if not (0 <= w1 and 0 <= w2 and w1+w2 <= 1): return np.inf
    f, sing = make_threepiece(a1, a2, a3, b1, b2, b3, w1, w2, c1, c2)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=150)
    return r if np.isfinite(r) else np.inf

# Family 4: convex combos of f0 + uniform + bumps
def make_combo(w0, w_unif, w_bump, c_bump, sigma_bump):
    """f = w0 * f0 + w_unif * 2 + w_bump * gaussian-ish bump on [-1/4,1/4]."""
    def f(x):
        if not (-0.25 <= x <= 0.25): return 0.0
        v = 0.0
        if w0 > 0:
            u = 2*x + 0.5
            if u > 0: v += w0 * u**(-0.5)
        v += w_unif * 2.0  # uniform of total mass 1 has height 2
        if w_bump > 0:
            v += w_bump * np.exp(-((x - c_bump)/sigma_bump)**2 / 2) / (sigma_bump * np.sqrt(2*np.pi))
        return v
    sing = [-0.25] if w0 > 0 else []
    return f, sing

def fam_combo(p):
    w0, w_unif, w_bump, c_bump, sigma_bump = p
    if w0 < 0 or w_unif < 0 or w_bump < 0: return np.inf
    if not (-0.25 <= c_bump <= 0.25): return np.inf
    if not (0.005 <= sigma_bump <= 0.5): return np.inf
    if w0 + w_unif + w_bump < 1e-6: return np.inf
    f, sing = make_combo(w0, w_unif, w_bump, c_bump, sigma_bump)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=150)
    return r if np.isfinite(r) else np.inf

# Family 5: smooth bumps (1 - (4x-c)^2)^k+
def make_smooth_bump(c, k):
    def f(x):
        u = 4*x - c
        v = 1 - u*u
        return v**k if v > 0 else 0.0
    return f, []

def fam_smooth(p):
    c, k = p
    if not (-1.0 <= c <= 1.0): return np.inf
    if not (0.0 < k <= 10): return np.inf
    f, sing = make_smooth_bump(c, k)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=150)
    return r if np.isfinite(r) else np.inf

# Family 6: piecewise-linear with N nodes
def make_piecewise(values, N):
    """values: vector of length N giving f at nodes x_i = -1/4 + (i+0.5)/N * 1/2."""
    nodes = np.linspace(-0.25, 0.25, N+1)
    vals = np.maximum(values, 0)
    def f(x):
        if x < -0.25 or x > 0.25: return 0.0
        # find interval
        idx = int(np.clip(np.floor((x + 0.25) / 0.5 * N), 0, N-1))
        x0, x1 = nodes[idx], nodes[idx+1]
        if vals[idx] is None: return 0.0
        # linear interpolate? we have N+1 nodes, N intervals; let's piecewise constant on intervals
        return vals[idx]
    return f, []

def fam_piecewise(p):
    vals = np.array(p)
    if np.any(vals < 0): return np.inf
    if vals.sum() < 1e-6: return np.inf
    f, sing = make_piecewise(vals, len(vals))
    r, _ = ratio(f, sing_pts_f=sing, n_scan=150)
    return r if np.isfinite(r) else np.inf

# Family 7: alpha + scaled translate (2-mass mixture, both alpha-singular)
def fam_2alpha(p):
    """w * (b1 - 2x)^{-a1} on left half + (1-w)*(b2 + 2x)^{-a2} on right half (both singular at OWN edge of own half)
    Also allow swap.
    Even more general: signs and locations."""
    a1, b1, a2, b2, w = p
    if not (0 < a1 <= 0.5 and 0 < a2 <= 0.5): return np.inf
    if not (0.5 <= b1 <= 1.5 and 0.5 <= b2 <= 1.5): return np.inf
    if not (0 <= w <= 1): return np.inf
    def f(x):
        if x < 0:
            v = b1 + 2*x  # singular at x=-b1/2
            left = w * (v**(-a1) if v > 0 else 0.0)
            return left
        else:
            v = b2 - 2*x  # singular at x=b2/2
            right = (1-w) * (v**(-a2) if v > 0 else 0.0)
            return right
    sing = []
    if -0.25 <= -b1/2 <= 0: sing.append(-b1/2)
    if 0 <= b2/2 <= 0.25: sing.append(b2/2)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=150)
    return r if np.isfinite(r) else np.inf

# Family 8: random nonneg piecewise constant (high-res)
def fam_random_piecewise(p, N=64):
    vals = np.array(p)
    if len(vals) != N: return np.inf
    if np.any(vals < 0): return np.inf
    if vals.sum() < 1e-6: return np.inf
    f, sing = make_piecewise(vals, N)
    r, _ = ratio(f, sing_pts_f=sing, n_scan=200)
    return r if np.isfinite(r) else np.inf

# ---------------------------------------------------------------------------
# Optimization runners
# ---------------------------------------------------------------------------
def run_de(fn, bounds, name, maxiter=120, popsize=24, seed=0, tol=1e-7):
    print(f"  DE on {name}: bounds={bounds}, maxiter={maxiter}, popsize={popsize}")
    t0 = time.time()
    try:
        res = differential_evolution(fn, bounds, maxiter=maxiter, popsize=popsize,
                                     seed=seed, tol=tol, polish=True, workers=1,
                                     init='sobol', updating='immediate')
        return dict(x=list(res.x), fun=float(res.fun), success=bool(res.success),
                    nit=int(res.nit), nfev=int(res.nfev), time=time.time()-t0)
    except Exception as e:
        return dict(error=str(e), time=time.time()-t0)

def run_random(fn, bounds, name, n=2000, seed=0):
    print(f"  RandomSearch on {name}: n={n}")
    rng = np.random.default_rng(seed)
    bs = np.array(bounds)
    lo, hi = bs[:,0], bs[:,1]
    best_v = np.inf
    best_x = None
    n_finite = 0
    for i in range(n):
        x = lo + (hi-lo) * rng.random(len(bounds))
        v = fn(x)
        if np.isfinite(v):
            n_finite += 1
            if v < best_v:
                best_v = v
                best_x = x.tolist()
    return dict(best_x=best_x, best_fun=float(best_v) if np.isfinite(best_v) else None,
                n=n, n_finite=n_finite)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = {'c0': C0, 'tests': []}
    overall_min = np.inf
    overall_min_info = None

    # Sanity: SS
    print("=== SS sanity ===")
    r_ss, info_ss = test_ss()
    print(f"  SS ratio = {r_ss:.6f}  (expected pi/8 = {C0:.6f})")
    results['ss_check'] = dict(ratio=r_ss, **info_ss, expected=C0)

    def update_min(r, name, x):
        nonlocal overall_min, overall_min_info
        if np.isfinite(r) and r < overall_min:
            overall_min = r
            overall_min_info = dict(family=name, x=list(x) if hasattr(x,'__iter__') else x, ratio=r)

    # Family 1: alpha
    print("\n=== Family 1: alpha (single-endpoint) ===", flush=True)
    de_alpha = run_de(fam_alpha, [(0.05, 0.5), (0.5, 1.0)], 'alpha',
                      maxiter=80, popsize=20, seed=0)
    rs_alpha = run_random(fam_alpha, [(0.05, 0.5), (0.5, 1.0)], 'alpha', n=400, seed=1)
    print(f"  DE best ratio: {de_alpha.get('fun')}")
    print(f"  RS best ratio: {rs_alpha.get('best_fun')}")
    results['tests'].append(dict(family='alpha', de=de_alpha, rs=rs_alpha))
    if de_alpha.get('fun') is not None: update_min(de_alpha['fun'], 'alpha', de_alpha['x'])
    if rs_alpha.get('best_fun') is not None: update_min(rs_alpha['best_fun'], 'alpha', rs_alpha['best_x'])

    # Family 2: two-piece
    print("\n=== Family 2: two-piece ===", flush=True)
    bnds = [(0.05, 0.5), (0.5, 1.0), (0.05, 0.5), (0.5, 1.0), (0.0, 1.0)]
    de_2p = run_de(fam_twopiece, bnds, 'twopiece', maxiter=80, popsize=20, seed=2)
    rs_2p = run_random(fam_twopiece, bnds, 'twopiece', n=500, seed=3)
    print(f"  DE best ratio: {de_2p.get('fun')}")
    print(f"  RS best ratio: {rs_2p.get('best_fun')}")
    results['tests'].append(dict(family='twopiece', de=de_2p, rs=rs_2p))
    if de_2p.get('fun') is not None: update_min(de_2p['fun'], 'twopiece', de_2p['x'])
    if rs_2p.get('best_fun') is not None: update_min(rs_2p['best_fun'], 'twopiece', rs_2p['best_x'])

    # Family 7: 2-alpha both singular at edges
    print("\n=== Family 7: 2alpha-edge ===", flush=True)
    bnds = [(0.05, 0.5), (0.5, 1.5), (0.05, 0.5), (0.5, 1.5), (0.0, 1.0)]
    de_2a = run_de(fam_2alpha, bnds, '2alpha', maxiter=80, popsize=20, seed=7)
    rs_2a = run_random(fam_2alpha, bnds, '2alpha', n=500, seed=8)
    print(f"  DE best ratio: {de_2a.get('fun')}")
    print(f"  RS best ratio: {rs_2a.get('best_fun')}")
    results['tests'].append(dict(family='2alpha', de=de_2a, rs=rs_2a))
    if de_2a.get('fun') is not None: update_min(de_2a['fun'], '2alpha', de_2a['x'])
    if rs_2a.get('best_fun') is not None: update_min(rs_2a['best_fun'], '2alpha', rs_2a['best_x'])

    # Family 3: three-piece
    print("\n=== Family 3: three-piece ===", flush=True)
    bnds = [(0.05, 0.5)]*3 + [(0.5, 1.0)]*3 + [(0.0, 1.0), (0.0, 1.0), (-0.24, 0.0), (0.0, 0.24)]
    de_3p = run_de(fam_threepiece, bnds, 'threepiece', maxiter=80, popsize=20, seed=4)
    rs_3p = run_random(fam_threepiece, bnds, 'threepiece', n=500, seed=5)
    print(f"  DE best ratio: {de_3p.get('fun')}")
    print(f"  RS best ratio: {rs_3p.get('best_fun')}")
    results['tests'].append(dict(family='threepiece', de=de_3p, rs=rs_3p))
    if de_3p.get('fun') is not None: update_min(de_3p['fun'], 'threepiece', de_3p['x'])
    if rs_3p.get('best_fun') is not None: update_min(rs_3p['best_fun'], 'threepiece', rs_3p['best_x'])

    # Family 4: combos
    print("\n=== Family 4: combos (f0+unif+bump) ===", flush=True)
    bnds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-0.25, 0.25), (0.005, 0.5)]
    de_c = run_de(fam_combo, bnds, 'combo', maxiter=60, popsize=18, seed=10)
    rs_c = run_random(fam_combo, bnds, 'combo', n=400, seed=11)
    print(f"  DE best ratio: {de_c.get('fun')}")
    print(f"  RS best ratio: {rs_c.get('best_fun')}")
    results['tests'].append(dict(family='combo', de=de_c, rs=rs_c))
    if de_c.get('fun') is not None: update_min(de_c['fun'], 'combo', de_c['x'])
    if rs_c.get('best_fun') is not None: update_min(rs_c['best_fun'], 'combo', rs_c['best_x'])

    # Family 5: smooth bump
    print("\n=== Family 5: smooth bump (1-(4x-c)^2)^k ===", flush=True)
    bnds = [(-1.0, 1.0), (0.05, 8.0)]
    de_s = run_de(fam_smooth, bnds, 'smooth', maxiter=60, popsize=15, seed=12)
    rs_s = run_random(fam_smooth, bnds, 'smooth', n=400, seed=13)
    print(f"  DE best ratio: {de_s.get('fun')}")
    print(f"  RS best ratio: {rs_s.get('best_fun')}")
    results['tests'].append(dict(family='smooth', de=de_s, rs=rs_s))
    if de_s.get('fun') is not None: update_min(de_s['fun'], 'smooth', de_s['x'])
    if rs_s.get('best_fun') is not None: update_min(rs_s['best_fun'], 'smooth', rs_s['best_x'])

    # Family 6: piecewise constant N=8, 16, 32
    for N in [8, 16, 32]:
        print(f"\n=== Family 6: piecewise N={N} ===", flush=True)
        bnds = [(0.0, 5.0)] * N
        mit = {8: 60, 16: 50, 32: 40}[N]
        ps  = {8: 18, 16: 18, 32: 16}[N]
        rs_n = {8: 400, 16: 400, 32: 400}[N]
        de_p = run_de(fam_piecewise, bnds, f'piecewise{N}', maxiter=mit, popsize=ps, seed=20+N)
        rs_p = run_random(fam_piecewise, bnds, f'piecewise{N}', n=rs_n, seed=21+N)
        print(f"  DE best ratio: {de_p.get('fun')}")
        print(f"  RS best ratio: {rs_p.get('best_fun')}")
        results['tests'].append(dict(family=f'piecewise{N}', de=de_p, rs=rs_p))
        if de_p.get('fun') is not None: update_min(de_p['fun'], f'piecewise{N}', de_p['x'])
        if rs_p.get('best_fun') is not None: update_min(rs_p['best_fun'], f'piecewise{N}', rs_p['best_x'])

    # Random walks: large N random
    print(f"\n=== Random search: pure random piecewise (N=64), 800 trials ===", flush=True)
    rng = np.random.default_rng(99)
    n_trials = 800
    n_finite = 0
    best_v = np.inf
    best_x = None
    for i in range(n_trials):
        N = rng.choice([8, 16, 32, 64])
        kind = rng.integers(0, 3)
        if kind == 0:
            v = rng.random(N)
        elif kind == 1:
            # exponential heavy tail near edges
            xs = np.linspace(-0.25, 0.25, N+1); xs = 0.5*(xs[:-1]+xs[1:])
            edge = np.minimum(xs+0.25, 0.25-xs)
            edge = np.maximum(edge, 1e-6)
            a = rng.uniform(0.05, 0.5)
            side = rng.integers(0, 3)
            if side == 0: v = (xs+0.25 + 1e-6)**(-a)
            elif side == 1: v = (0.25-xs + 1e-6)**(-a)
            else: v = edge**(-a)
            v *= rng.random(N) * 0.5 + 0.5
        else:
            # bumps
            n_b = rng.integers(1, 4)
            xs = np.linspace(-0.25, 0.25, N+1); xs = 0.5*(xs[:-1]+xs[1:])
            v = np.zeros(N)
            for _ in range(n_b):
                c = rng.uniform(-0.24, 0.24); s = rng.uniform(0.02, 0.2)
                amp = rng.random()
                v += amp*np.exp(-0.5*((xs-c)/s)**2)
        v = np.maximum(v, 0)
        if v.sum() < 1e-6: continue
        f, sing = make_piecewise(v, N)
        r, _ = ratio(f, sing_pts_f=sing, n_scan=200)
        if np.isfinite(r):
            n_finite += 1
            if r < best_v:
                best_v = r
                best_x = (N, v.tolist())
    print(f"  pure random: {n_finite}/{n_trials} finite, best ratio = {best_v:.6f}")
    results['tests'].append(dict(family='random_pure', n=n_trials, n_finite=n_finite,
                                 best_fun=float(best_v) if np.isfinite(best_v) else None,
                                 best_x_summary=str(best_x[0]) if best_x else None))
    if np.isfinite(best_v): update_min(best_v, 'random_pure', best_x[0] if best_x else None)

    # Mirrored variations: alpha mirrored
    print("\n=== Family 1m: mirrored alpha ===", flush=True)
    def fam_alpha_m(p):
        a, b = p
        if not (0 < a <= 0.5 and 0.5 <= b <= 1.0): return np.inf
        f, sing = make_alpha(a, b, mirror=True)
        r, _ = ratio(f, sing_pts_f=sing, n_scan=200)
        return r if np.isfinite(r) else np.inf
    de_am = run_de(fam_alpha_m, [(0.05, 0.5), (0.5, 1.0)], 'alpha_m',
                   maxiter=60, popsize=18, seed=30)
    print(f"  DE best ratio: {de_am.get('fun')}")
    results['tests'].append(dict(family='alpha_mirror', de=de_am))
    if de_am.get('fun') is not None: update_min(de_am['fun'], 'alpha_mirror', de_am['x'])

    # Save
    results['overall_min'] = overall_min
    results['overall_min_info'] = overall_min_info
    print(f"\n\n=== OVERALL MIN ratio across all families: {overall_min:.6f} ===")
    print(f"    c0 = pi/8 = {C0:.6f}")
    print(f"    margin = {overall_min - C0:.6f}")
    if overall_min_info: print(f"    achieved by: {overall_min_info['family']}")

    out_path = os.path.join(OUTDIR, 'path_a_stress_test.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nWrote {out_path}")
    return results

if __name__ == '__main__':
    main()
