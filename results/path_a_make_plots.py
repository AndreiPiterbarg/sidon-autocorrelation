"""Plot lowest-ratio candidates from the v2 stress test."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
from path_a_stress_v2 import (ratio_of, conv_at, sup_conv, C0)

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Reconstruct f from family name + parameter list
def f_from(family, x):
    sing = []
    if family == 'alpha':
        a, b = x
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            v = b - 2*xv
            return v**(-a) if v > 0 else 0.0
        if -0.25 <= b/2 <= 0.25: sing = [b/2]
        return f, sing
    if family == 'alpha_left':
        a, b = x
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            v = b + 2*xv
            return v**(-a) if v > 0 else 0.0
        if -0.25 <= -b/2 <= 0.25: sing = [-b/2]
        return f, sing
    if family == 'two_endpoints' or family == 'two_endpoints_v2':
        a1, a2, w = x
        def f(xv):
            if xv <= -0.25 or xv >= 0.25: return 0.0
            return w*(2*xv+0.5)**(-a1) + (1-w)*(0.5-2*xv)**(-a2)
        return f, [-0.25, 0.25]
    if family == 'ss_perturb':
        eps, c, sigma = x
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            ss = (2*xv + 0.5)**(-0.5)
            bump = eps*np.exp(-0.5*((xv-c)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
            return max(ss + bump, 0.0)
        return f, [-0.25]
    if family == 'beta':
        p1, p2 = x
        def f(xv):
            if xv <= -0.25 or xv >= 0.25: return 0.0
            u = 2*xv+0.5; v = 0.5-2*xv
            if u<=0 or v<=0: return 0.0
            return u**p1 * v**p2
        if p1 < 0: sing.append(-0.25)
        if p2 < 0: sing.append(0.25)
        return f, sing
    if family == 'alpha_plus_unif':
        a, u = x
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            return (2*xv+0.5)**(-a) + u
        return f, [-0.25]
    if family == 'alpha_plus_lin':
        a, A, B = x
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            v = (2*xv+0.5)**(-a) + A + B*(xv+0.25)
            return max(v, 0)
        return f, [-0.25]
    if family == 'double_power':
        a1, a2, s = x
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            u = 2*xv + 0.5
            return u**(-a1) + s*u**(-a2)
        return f, [-0.25]
    if family == 'combo':
        w0, wu, wb, c, sg = x
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            v = wu*2.0
            if w0>0: v += w0*(2*xv+0.5)**(-0.5)
            if wb>0: v += wb*np.exp(-0.5*((xv-c)/sg)**2)/(sg*np.sqrt(2*np.pi))
            return v
        if w0 > 0: sing = [-0.25]
        return f, sing
    if family == 'mv_style':
        a1, a2, w, split = x
        def f(xv):
            if xv <= -0.25 or xv >= 0.25: return 0.0
            if xv < split:
                return w*(2*xv+0.5)**(-a1)
            else:
                return (1-w)*(0.5-2*xv)**(-a2)
        return f, [-0.25, 0.25]
    if family == 'smooth':
        c, k = x
        def f(xv):
            if xv < -0.25 or xv > 0.25: return 0.0
            u = 4*xv - c
            v = 1 - u*u
            return v**k if v > 0 else 0.0
        return f, []
    if family == 'alpha_at_edge_1D':
        a = x[0]
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            return (2*xv + 0.5)**(-a)
        return f, [-0.25]
    if family == 'perturb_lin':
        s = x[0]
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            return max((2*xv+0.5)**(-0.5) + s*(xv+0.25), 0)
        return f, [-0.25]
    if family == 'perturb_quad':
        s = x[0]
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            return max((2*xv+0.5)**(-0.5) + s*(xv+0.25)**2, 0)
        return f, [-0.25]
    if family == 'SS_doublebump':
        e1, c1, s1, e2, c2, s2 = x
        def f(xv):
            if xv <= -0.25 or xv > 0.25: return 0.0
            v = (2*xv+0.5)**(-0.5)
            v += e1*np.exp(-0.5*((xv-c1)/s1)**2)/(s1*np.sqrt(2*np.pi))
            v += e2*np.exp(-0.5*((xv-c2)/s2)**2)/(s2*np.sqrt(2*np.pi))
            return max(v, 0)
        return f, [-0.25]
    if family == 'piecewise8' or family == 'piecewise16' or family == 'piecewise32':
        N = int(family.replace('piecewise',''))
        vals = np.array(x)
        def f(xv):
            if xv < -0.25 or xv > 0.25: return 0.0
            idx = int(min(N-1, max(0, np.floor((xv+0.25)/0.5*N))))
            return max(vals[idx], 0)
        return f, []
    return None, None

def main():
    cands = []
    for fname in ['path_a_stress_test.json', 'path_a_extra_adversarial.json']:
        path = os.path.join(OUTDIR, fname)
        if not os.path.exists(path): continue
        with open(path) as fp:
            data = json.load(fp)
        for s in data.get('studies', []):
            name = s.get('name')
            if s.get('best_fun') is not None:
                cands.append((name, s['best_x'], s['best_fun']))
            for tup in s.get('top5_random', []) or []:
                v, x = tup
                if v is not None and np.isfinite(v):
                    cands.append((name, x, float(v)))
        if 'mass_random' in data and data['mass_random'].get('best_info'):
            mri = data['mass_random']['best_info']
            cands.append(('mass_'+mri.get('fn',''), mri.get('x'), mri.get('ratio')))

    # add SS reference
    cands.append(('alpha_at_edge_1D', [0.5], C0))

    # filter
    cands = [c for c in cands if c[2] is not None and np.isfinite(c[2])]
    cands.sort(key=lambda c: c[2])
    print(f'Candidates: {len(cands)}, top 8 lowest:')
    for c in cands[:8]:
        print(f'  {c[0]}: r={c[2]:.6f}, x={c[1]}')

    # Plot top 6 distinct families
    seen = set()
    plot_list = []
    for c in cands:
        key = c[0]
        if key in seen: continue
        seen.add(key)
        plot_list.append(c)
        if len(plot_list) >= 6: break

    fig, axes = plt.subplots(len(plot_list), 2, figsize=(12, 2.5*len(plot_list)))
    if len(plot_list) == 1: axes = axes.reshape(1, 2)
    for i, (fam, x, r) in enumerate(plot_list):
        ff = f_from(fam, x)
        if ff[0] is None:
            print(f'  cannot plot {fam}'); continue
        f, sing = ff
        xs = np.linspace(-0.2499, 0.2499, 600)
        ys = np.array([f(xv) for xv in xs])
        ymax = ys.max() if ys.max() > 0 else 1
        ymin = max(ys.min(), 1e-10)
        axes[i,0].plot(xs, ys)
        axes[i,0].set_title(f'{fam}: r={r:.5f}\nx={[round(v,4) for v in x]}', fontsize=9)
        axes[i,0].set_xlabel('x'); axes[i,0].set_ylabel('f(x)')
        if ymax/ymin > 50: axes[i,0].set_yscale('log')
        ts = np.linspace(-0.4999, 0.4999, 200)
        cs = np.array([conv_at(f, t, sing_pts_f=sing) for t in ts])
        axes[i,1].plot(ts, cs)
        axes[i,1].set_title(f'(f*f)(t)')
        axes[i,1].set_xlabel('t')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'path_a_lowest_ratio_candidates.png'), dpi=110)
    print('Wrote path_a_lowest_ratio_candidates.png')

    # SS reference plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    f, sing = f_from('alpha_at_edge_1D', [0.5])
    xs = np.linspace(-0.2499, 0.2499, 600)
    ax[0].plot(xs, [f(xv) for xv in xs])
    ax[0].set_title('Schinzel-Schmidt f0(x) = (2x+1/2)^{-1/2}')
    ax[0].set_xlabel('x'); ax[0].set_yscale('log')
    ts = np.linspace(-0.4999, 0.4999, 200)
    cs = [conv_at(f, t, sing_pts_f=sing) for t in ts]
    ax[1].plot(ts, cs)
    ax[1].axhline(np.pi/2, ls='--', alpha=0.5, label=r'$\pi/2$')
    ax[1].legend()
    ax[1].set_title('(f0*f0)(t); analytic max = pi/2 at t=0')
    ax[1].set_xlabel('t')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'path_a_ss_reference.png'), dpi=110)
    print('Wrote path_a_ss_reference.png')

    # Ratio histogram
    if cands:
        rs = np.array([c[2] for c in cands])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(rs[rs < 2], bins=80)
        ax.axvline(C0, color='r', ls='--', label=f'pi/8 = {C0:.5f}')
        ax.axvline(rs.min(), color='g', ls=':', label=f'min r = {rs.min():.5f}')
        ax.set_xlabel('ratio'); ax.set_ylabel('count')
        ax.set_title('Distribution of ratios across all candidate evaluations')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'path_a_ratio_histogram.png'), dpi=110)
        print('Wrote path_a_ratio_histogram.png')

if __name__ == '__main__':
    main()
