"""Faster adversarial tests focused on perturbations of SS."""
import numpy as np, json, os, sys, time
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_a_stress_v2 import (ratio_of, sup_conv, conv_at, safe_int, C0,
                                fam_two_endpoints, fam_ss_perturb,
                                fam_alpha_plus_unif, fam_alpha_plus_lin,
                                fam_double_power, fam_beta, fam_combo,
                                fam_mv_style, study, local_refine)

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Extra: refined two-endpoints
def fam_two_endpoints_h(p):
    """heavy two-endpoints with both at exactly 0.5 — must hit exactly singularity"""
    a1, a2, w = p
    if not (0.05 <= a1 <= 0.5 and 0.05 <= a2 <= 0.5 and 0 <= w <= 1): return np.inf
    def f(x):
        if x <= -0.25 or x >= 0.25: return 0.0
        return w*(2*x+0.5)**(-a1) + (1-w)*(0.5-2*x)**(-a2)
    sing = [-0.25, 0.25]
    r, _ = ratio_of(f, sing_pts_f=sing)
    return r

# 1D: fam_alpha at edge precisely (vs SS)
def fam_alpha_at_edge(p):
    a = p[0]
    if not (0.05 <= a <= 0.499): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        return (2*x + 0.5)**(-a)
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def fam_perturb_2(p):
    """SS + linear delta_lin*x"""
    s = p[0]
    if not (-3 <= s <= 5): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        v = (2*x + 0.5)**(-0.5) + s*(x+0.25)
        return max(v, 0)
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def fam_perturb_3(p):
    """SS + quadratic delta_q*x^2"""
    s = p[0]
    if not (-3 <= s <= 5): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        v = (2*x + 0.5)**(-0.5) + s*(x+0.25)**2
        return max(v, 0)
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def fam_perturb_4(p):
    """SS + sin perturbation"""
    A, k, phi = p
    if not (-2 <= A <= 2 and 1 <= k <= 30 and 0 <= phi <= 2*np.pi): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        v = (2*x + 0.5)**(-0.5) + A*np.sin(k*x + phi)
        return max(v, 0)
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def fam_perturb_5(p):
    """SS rescaled mass + uniform: w*SS + (1-w)*2"""
    w = p[0]
    if not (0 <= w <= 1): return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        return w*(2*x+0.5)**(-0.5) + (1-w)*2.0
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def fam_perturb_6(p):
    """SS + multiple bumps"""
    e1, c1, s1, e2, c2, s2 = p
    bnd_ok = (-2 <= e1 <= 5 and -2 <= e2 <= 5 and
              -0.24 <= c1 <= 0.24 and -0.24 <= c2 <= 0.24 and
              0.005 <= s1 <= 0.5 and 0.005 <= s2 <= 0.5)
    if not bnd_ok: return np.inf
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        v = (2*x+0.5)**(-0.5)
        v += e1*np.exp(-0.5*((x-c1)/s1)**2)/(s1*np.sqrt(2*np.pi))
        v += e2*np.exp(-0.5*((x-c2)/s2)**2)/(s2*np.sqrt(2*np.pi))
        return max(v, 0)
    r, _ = ratio_of(f, sing_pts_f=[-0.25])
    return r

def main():
    overall = np.inf; overall_info = None
    out = {'c0': C0, 'studies': []}
    studies = [
        ('alpha_at_edge_1D', fam_alpha_at_edge, [(0.05, 0.499)], 200, 4, 60),
        ('perturb_lin', fam_perturb_2, [(-3, 5)], 200, 4, 60),
        ('perturb_quad', fam_perturb_3, [(-3, 5)], 200, 4, 60),
        ('perturb_sin', fam_perturb_4, [(-2, 2), (1, 30), (0, 2*np.pi)], 350, 5, 80),
        ('SSplus_unif_w', fam_perturb_5, [(0, 1)], 200, 4, 50),
        ('SS_doublebump', fam_perturb_6,
         [(-2, 5), (-0.24, 0.24), (0.005, 0.5)]*2, 600, 6, 150),
        ('two_endpoints_v2', fam_two_endpoints_h, [(0.05, 0.5), (0.05, 0.5), (0,1)], 350, 5, 80),
    ]
    t_start = time.time()
    for name, fn, bnds, nr, nf, mt in studies:
        s = study(name, fn, bnds, n_random=nr, n_refine=nf, seed=hash(name)%(2**31), max_time=mt)
        out['studies'].append(s)
        if s.get('best_fun') is not None and s['best_fun'] < overall:
            overall = s['best_fun']; overall_info = dict(family=name, x=s['best_x'], ratio=s['best_fun'])

    out['overall_min'] = overall
    out['overall_min_info'] = overall_info
    out['elapsed'] = time.time() - t_start
    print(f"\n=== EXTRA OVERALL MIN: {overall:.6f}, c0=pi/8={C0:.6f}, margin={overall-C0:.2e} ===")
    if overall_info: print(f"  {overall_info}")
    with open(os.path.join(OUTDIR, 'path_a_extra_adversarial.json'), 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\nWrote path_a_extra_adversarial.json  elapsed={out['elapsed']:.1f}s")

if __name__ == '__main__':
    main()
