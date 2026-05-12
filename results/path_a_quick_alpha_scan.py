"""Quick: SS-alpha 1D scan a in (0, 0.5]."""
import numpy as np, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
from path_a_stress_test import ratio, C0

OUTDIR = os.path.dirname(os.path.abspath(__file__))

def f_ss_alpha(a):
    def f(x):
        if x <= -0.25 or x > 0.25: return 0.0
        return (2*x + 0.5)**(-a)
    return f, [-0.25]

# scan a from 0.05 to 0.5
print('a, ratio, S, I, I32')
results = []
t0 = time.time()
for a in np.linspace(0.05, 0.5, 91):
    f, sing = f_ss_alpha(a)
    r, info = ratio(f, sing_pts_f=sing, n_scan=400)
    results.append({'a': float(a), 'ratio': float(r), **{k: float(v) for k, v in info.items()}})
    print(f'{a:.4f}, {r:.6f}, S={info.get("S")}, I={info.get("I")}, I32={info.get("I3")}')
print(f'time = {time.time()-t0:.1f}s')

# also more refined near 0.5
for a in np.linspace(0.45, 0.5, 21):
    f, sing = f_ss_alpha(a)
    r, info = ratio(f, sing_pts_f=sing, n_scan=400)
    results.append({'a': float(a), 'ratio': float(r), **{k: float(v) for k, v in info.items()}})
    print(f'{a:.4f}, {r:.6f}')

# minimum
fin = [r for r in results if np.isfinite(r['ratio'])]
fin.sort(key=lambda r: r['ratio'])
print(f'\nMin ratio: {fin[0]}')

with open(os.path.join(OUTDIR, 'path_a_quick_alpha_scan.json'), 'w') as fp:
    json.dump({'c0': C0, 'results': results}, fp, indent=2)
