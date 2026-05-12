"""Properly verified per-cell methods on d=2 borderline cells.

Critical insight: at (n_half=1, m=20, c_target=1.281), d=2 cells have
ALL true val(cell) <= 1.28 < 1.281. Therefore NO sound method can prune
any of them. The cascade is mathematically blocked at d=2.

This test confirms: methods reporting "pruned" must be UNSOUND (numerical bugs).

We also test method 6: a TRUE BOUND on val(cell) using exact 1D analysis
which is rigorous for d=2 (since d=2 is exactly 1-DOF).
"""
import sys, os, time, json
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

import numpy as np
from itertools import combinations_with_replacement
from _Q_bench import _build_windows
from _M1_bench import prune_F
from _L_bench import _build_A_matrices, _shor_feasibility, _lasserre2_feasibility, _make_cell

n_half = 1
m = 20
c_target = 1.281
d = 2
S = 4 * n_half * m

windows, ell_int_sums = _build_windows(d)
A_mats = _build_A_matrices(d, windows)

all_comps = [[k, S - k] for k in range(0, S + 1)]
batch = np.array(all_comps, dtype=np.int32)
sF = prune_F(batch, n_half, m, c_target)
canonical = [c for c, s in zip(all_comps, sF) if s and c[0] <= c[1]]
test_cells = [c for c in canonical if c[0] != c[1]]
print(f"Testing {len(test_cells)} cells\n")


# Method 6: EXACT 1D analysis (d=2 has 1 DOF)
# For x = (v, S-v), val(cell) = min over v in [c0-1, c0+1] of
#   max_W m^2*TV_W(x). Each m^2*TV_W is a quadratic in v.
# We analytically compute the minimum: for each window W, m^2*TV_W is
# a quadratic q_W(v). The pointwise max is a piecewise-quadratic.
# Min over [v_lo, v_hi] of max_W q_W(v).
#
# This is a sound, exact computation for d=2. It directly gives the TRUE
# value of val(cell), and val(cell) > c_target iff cell is genuinely
# prunable.
def method6_exact_1d(c):
    """Exact 1D: compute min over v of max_W m^2*TV_W(v, S-v).

    Sound (rigorous, no numerical relaxation). Returns (prunable, val_normalized).
    """
    c0 = c[0]
    v_lo = max(0.0, c0 - 1.0)
    v_hi = float(c0 + 1.0)

    # Each window W has q_W(v) = (1/(4n*ell)) * x^T A_W x
    # x^T A_W x = sum_{(i,j) in W} x_i x_j with x_0=v, x_1=S-v
    # = a_W * v^2 + b_W * v (S-v) + c_W * (S-v)^2 [for d=2]
    # where a_W = number of (i,j)=(0,0), b_W = sum (0,1)+(1,0) [unordered counted once],
    # actually A_W is symmetric, so x^T A_W x = sum_ij A_W[i,j] x_i x_j
    # For d=2 with x_0=v, x_1=S-v: x^T A_W x = A[0,0] v^2 + 2*A[0,1] v(S-v) + A[1,1] (S-v)^2

    # Find min over [v_lo, v_hi] using a fine grid + analytic check at vertices/intersections
    # For exact: enumerate all critical points (window-wise minima, boundary points,
    # window-pair intersections).
    crit_points = [v_lo, v_hi]

    def q_W(v, A_mat, ell):
        x = np.array([v, S - v])
        return float(x @ A_mat @ x) / (4.0 * n_half * ell)

    # Per-window minimum of q_W (interior)
    for A_mat, (ell, _) in zip(A_mats, windows):
        # q_W(v) = a v^2 + 2b v(S-v) + c (S-v)^2
        # Expand: = a v^2 + 2b (Sv - v^2) + c (S^2 - 2Sv + v^2)
        # = (a - 2b + c) v^2 + 2(b S - c S) v + c S^2
        # = (a + c - 2b) v^2 + 2 S (b - c) v + c S^2
        a = A_mat[0, 0]
        b = A_mat[0, 1]
        c_ = A_mat[1, 1]
        A2 = a + c_ - 2 * b
        B2 = 2 * S * (b - c_)
        if abs(A2) > 1e-12:
            v_crit = -B2 / (2 * A2)
            if v_lo <= v_crit <= v_hi:
                crit_points.append(v_crit)

    # Pairwise window intersections: q_W1(v) = q_W2(v) -> linear or quadratic equation
    for i in range(len(A_mats)):
        for j in range(i + 1, len(A_mats)):
            A1, A2 = A_mats[i], A_mats[j]
            ell1, ell2 = windows[i][0], windows[j][0]
            # q_W = (1/4n/ell) x^T A x = quadratic in v
            # Equate: (1/(4n*ell1)) * (a1+c1-2b1)v^2 + 2S(b1-c1)/4n/ell1 v + c1 S^2 / (4n*ell1)
            #         = (1/(4n*ell2)) * ...
            ai, bi, ci = A1[0,0], A1[0,1], A1[1,1]
            aj, bj, cj = A2[0,0], A2[0,1], A2[1,1]
            sa = (ai + ci - 2*bi) / (4*n_half*ell1)
            sb = 2*S*(bi - ci) / (4*n_half*ell1)
            sc = ci * S**2 / (4*n_half*ell1)
            ta = (aj + cj - 2*bj) / (4*n_half*ell2)
            tb = 2*S*(bj - cj) / (4*n_half*ell2)
            tc = cj * S**2 / (4*n_half*ell2)
            # Equation: (sa-ta)v^2 + (sb-tb)v + (sc-tc) = 0
            A_eq = sa - ta
            B_eq = sb - tb
            C_eq = sc - tc
            if abs(A_eq) > 1e-12:
                disc = B_eq**2 - 4*A_eq*C_eq
                if disc >= 0:
                    sd = np.sqrt(disc)
                    for v_root in [(-B_eq + sd) / (2*A_eq), (-B_eq - sd) / (2*A_eq)]:
                        if v_lo <= v_root <= v_hi:
                            crit_points.append(v_root)
            elif abs(B_eq) > 1e-12:
                v_root = -C_eq / B_eq
                if v_lo <= v_root <= v_hi:
                    crit_points.append(v_root)

    # Evaluate max over windows at each critical point + dense grid for safety
    crit_points = sorted(set(crit_points))
    # Add dense grid for numerical safety (should match)
    dense_v = np.linspace(v_lo, v_hi, 1001)
    crit_points = sorted(set(crit_points + list(dense_v)))

    best_val = np.inf
    best_v = None
    for v in crit_points:
        max_w = -np.inf
        for A_mat, (ell, _) in zip(A_mats, windows):
            val = q_W(v, A_mat, ell)
            if val > max_w:
                max_w = val
        if max_w < best_val:
            best_val = max_w
            best_v = v

    return best_val / (m * m), best_v


print("=" * 70)
print("Method 6: EXACT 1D analytic (d=2, sound, rigorous)")
print("=" * 70)
n6_pruned = 0
times6 = []
for c in test_cells:
    t0 = time.time()
    val, v_star = method6_exact_1d(c)
    t = time.time() - t0
    times6.append(t)
    is_prunable = val > c_target
    if is_prunable:
        n6_pruned += 1
    print(f"  c={c}: val(cell)={val:.6f}, v*={v_star:.4f}, "
          f"prunable={'YES' if is_prunable else 'NO'} ({t*1000:.1f} ms)")
print(f"  TOTAL: {n6_pruned} of {len(test_cells)} pruned (exact, sound)")
print(f"  Wall: {sum(times6):.2f}s, med {1000*np.median(times6):.1f} ms/cell")

# Test slightly higher c_target where some cells become prunable
print()
print("=" * 70)
print("Sensitivity: at what c_target do these cells become prunable?")
print("=" * 70)
for c in test_cells:
    val, _ = method6_exact_1d(c)
    print(f"  c={c}: val(cell)={val:.6f} (need c_target < {val:.4f} to prune)")
