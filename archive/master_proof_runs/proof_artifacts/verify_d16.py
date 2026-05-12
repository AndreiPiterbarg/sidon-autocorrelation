"""Independent verification of d=16 result."""
import sys
import numpy as np

sys.path.insert(0, '.')
from kkt_correct_mu_star import build_window_data, evaluate_tv_per_window

data = np.load('mu_star_d16.npz')
mu = data['mu_star']
print("d=16 finding:")
print(f"  f = {data['f_value']}")
print(f"  sum(mu) = {mu.sum():.10f}")
print(f"  min(mu) = {mu.min():.4e}, max = {mu.max():.4f}")
print(f"  num zero entries (<1e-8) = {int((mu < 1e-8).sum())}")
A, c = build_window_data(16)
T = evaluate_tv_per_window(mu, A, c)
fmax = float(T.max())
print(f"  max TV (independent): {fmax:.10f}")
print(f"  margin to c=1.281: {fmax - 1.281:+.6f}")
verdict = "<" if fmax < 1.281 else ">="
print(f"  --> val(16) <= {fmax:.6f}")
print(f"  --> val(16) {verdict} 1.281 (UB: {fmax:.6f} {verdict} 1.281)")
