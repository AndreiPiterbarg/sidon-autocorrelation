"""For d=2 cells, compute TRUE val(cell) = inf over x in cell of max_W m^2*TV_W(a).

For d=2: x = (v, 80-v), v in [c_0-1, c_0+1] (assuming c_0 >= 1).
Convolution: conv = [v^2, 2v(80-v), (80-v)^2]
m^2*TV_W = (1/(4n*ell)) sum_{(i,j) in W} x_i x_j = window_sum / (4n*ell) [in m^2 units]

For each cell c, compute over fine grid of v: max_W m^2*TV_W(a). Then take MIN over v.
If this MIN >= 1.281, the cell is genuinely prunable. If < 1.281, the cell is genuinely
not prunable.
"""
import sys, os
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

import numpy as np
from _Q_bench import _build_windows
from _M1_bench import prune_F

n_half = 1
m = 20
c_target = 1.281
d = 2
S = 80

windows, ell_int_sums = _build_windows(d)
print(f"Windows for d=2: {windows}")

# Build A_W matrices
def build_A(d, ell, s_lo):
    s_hi = s_lo + ell - 2
    A = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                A[i, j] = 1.0
    return A

A_mats = [build_A(d, ell, s_lo) for (ell, s_lo) in windows]

def cell_max_TV(v, n_half, m):
    """For x=(v, 80-v), compute max_W m^2*TV_W = max over windows of x^T A_W x / (4n*ell)."""
    x = np.array([v, S - v])
    best = -np.inf
    best_w = None
    for A_mat, (ell, s_lo) in zip(A_mats, windows):
        val = float(x @ A_mat @ x) / (4.0 * n_half * ell)
        if val > best:
            best = val
            best_w = (ell, s_lo)
    return best, best_w

def cell_min_max_TV(c, n_half, m):
    """Find min over v in [c0-1, c0+1] of max_W m^2*TV_W(x).

    Returns (min_max, argmin_v).
    """
    c0 = c[0]
    v_lo = max(0.0, c0 - 1.0)
    v_hi = c0 + 1.0
    # Fine grid sweep
    n_grid = 10001
    vs = np.linspace(v_lo, v_hi, n_grid)
    best_v = vs[0]
    best_val = np.inf
    for v in vs:
        val, _ = cell_max_TV(v, n_half, m)
        if val < best_val:
            best_val = val
            best_v = v
    return best_val, best_v

# Get F-survivor cells
all_comps = [[k, S - k] for k in range(0, S + 1)]
batch = np.array(all_comps, dtype=np.int32)
sF = prune_F(batch, n_half, m, c_target)
canonical = [c for c, s in zip(all_comps, sF) if s and c[0] <= c[1]]

print(f"\nFor each L0 cell, computing TRUE val(cell) = min_v max_W m^2*TV_W:")
print(f"  c_target = {c_target}, m^2 = {m*m}, threshold (m^2 units) = {c_target*m*m}")
print()
results = []
for c in canonical:
    cell_val, v_star = cell_min_max_TV(c, n_half, m)
    cell_val_norm = cell_val / (m*m)  # in non-m^2 units
    is_prunable = cell_val_norm > c_target
    results.append((c, cell_val_norm, v_star, is_prunable))
    print(f"  c={c}: val(cell)/m^2 = {cell_val_norm:.6f}, "
          f"v* = {v_star:.4f}, prunable={'YES' if is_prunable else 'NO'}")

n_prunable = sum(1 for _, _, _, p in results if p)
print(f"\n*** TRUE prunable cells: {n_prunable} / {len(canonical)} ***")
print(f"*** TRUE NON-prunable: {len(canonical) - n_prunable} cells ***")
