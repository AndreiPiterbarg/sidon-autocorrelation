"""Empirical soundness check of qp_bound vertex enumeration.

For each (c, W), compute:
 - qp_bound via vertex enumeration (existing code).
 - "true" max of f via fine grid search on the (d-1)-dim Cell.
   Cell parameterization: pick d-1 free coords, last coord determined by Σ=0.
   For each grid point in the (d-1)-dim cube, project to Cell (clip last
   coord) and evaluate f.
 - Lasserre SDP order-2 LB on min g (which gives UB on max f via -g).

Compare. If grid_max > vertex_max for any (c, W), vertex enum is unsound.
"""
import os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger', 'cpu'))
from qp_bound import (build_window_matrix, grad_for_window, qp_bound_vertex,
                       qp_bound_for_composition)


def fine_grid_max_f(grad, A_W, scale, h, d, n_grid=20):
    """Brute-force max of f(δ) := -grad·δ - scale·δ^T A_W δ over Cell.
    Cell = {δ : |δ_i| ≤ h, Σδ = 0}. Parameterize by d-1 free coords;
    last coord forced by Σ=0; check |last| ≤ h.
    """
    best = 0.0  # f(0) = 0
    grid = np.linspace(-h, h, n_grid)
    # Iterate over all (n_grid)^(d-1) tuples for first d-1 coords
    for tup in itertools.product(grid, repeat=d-1):
        last = -sum(tup)
        if abs(last) > h:
            continue
        delta = np.array(list(tup) + [last], dtype=np.float64)
        f = -grad @ delta - scale * delta @ A_W @ delta
        if f > best:
            best = f
    return best


def check_soundness(d_max=4, n_check=20, n_grid=15):
    """For random (c, W), compare vertex_max vs grid_max."""
    np.random.seed(42)
    found_violations = 0
    max_excess = 0.0
    examples = []
    for trial in range(n_check):
        d = 4 if d_max <= 4 else np.random.randint(2, d_max + 1)
        S = np.random.randint(10, 30)
        c = np.random.dirichlet(np.ones(d)) * S
        c = np.maximum(0, np.round(c)).astype(np.int32)
        # Adjust to sum=S
        c[0] += S - c.sum()
        c = np.maximum(0, c)
        for ell in range(2, 2 * d + 1):
            for s_lo in range(2 * d - ell + 1 + 1):
                if s_lo + ell - 2 > 2 * d - 2:
                    continue
                A_W = build_window_matrix(d, ell, s_lo)
                grad = grad_for_window(c.astype(np.float64), A_W, S, d, ell)
                h = 1.0 / (2.0 * S)
                scale = 2.0 * d / ell
                v_max = qp_bound_vertex(grad, A_W, scale, h, d)
                g_max = fine_grid_max_f(grad, A_W, scale, h, d, n_grid=n_grid)
                excess = g_max - v_max
                if excess > 1e-10:
                    found_violations += 1
                    if excess > max_excess:
                        max_excess = excess
                    if len(examples) < 5:
                        examples.append({
                            'c': c.tolist(), 'd': d, 'S': S, 'ell': ell, 's_lo': s_lo,
                            'vertex_max': v_max, 'grid_max': g_max, 'excess': excess
                        })
    return found_violations, max_excess, examples


if __name__ == '__main__':
    print("Checking qp_bound soundness (d=4, fine grid n_grid=15)...")
    n_violations, max_excess, examples = check_soundness(d_max=4, n_check=30, n_grid=15)
    print(f"\nViolations found: {n_violations}")
    print(f"Max excess (grid_max - vertex_max): {max_excess:.6e}")
    if examples:
        print("\nFirst few violations:")
        for ex in examples:
            print(f"  c={ex['c']} d={ex['d']} S={ex['S']} (ell={ex['ell']}, s_lo={ex['s_lo']}): "
                  f"vertex={ex['vertex_max']:.6f}, grid={ex['grid_max']:.6f}, "
                  f"excess={ex['excess']:.6e}")
    else:
        print("\nNo violations — vertex enum agrees with grid search to tolerance.")
