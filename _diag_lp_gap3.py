"""Final diagnosis. Use boxes that DO contain a simplex feasible point.

Key insight: at depth ~30-40, BnB has shrunk most boxes to width 1e-9 or so.
The 99.8% stuck boxes are likely SMALL (deep) boxes whose LP value is below
1.281 even though true min on the box is well above 1.281.

For an in-flight stuck box, what is the actual width? The pod log says
1.68M certifications but stuck on 0.2% volume. Let me reason from depth:
volume_per_box ≈ 0.002 / 1.68e6 ≈ 1.2e-9 (volume), so per-axis width ≈
1.2e-9^(1/19) ≈ 0.32 — but that's mean width over axes. In 19D the box has
many axes of small width and a few large. More likely: depth ~ 30 puts most
axes at width 1e-9, but a few axes are much wider (e.g. unsplittable due
to D_SHIFT saturation at 2^-60).

Actually the box on which LP runs has width ~ 1/2^depth. At depth 24 we have
width ≈ 6e-8. So the LP gap shouldn't matter at all here per the McCormick
hypothesis. So what IS the bottleneck?

I'll construct boxes that are sliver-shaped (most axes 1e-9, one or two axes
1e-3 — saturation case). And check LP behavior.
"""
import os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import _solve_epigraph_lp
from _diag_lp_gap import _solve_epigraph_lp_with_primal


def make_simplex_centered_box(center, widths_per_axis):
    """Return (lo, hi). center should sum to 1; widths is per-axis half-width."""
    d = len(center)
    lo = np.maximum(center - widths_per_axis, 0.0)
    hi = np.minimum(center + widths_per_axis, 1.0)
    # Adjust so that simplex passes through: ensure sum(lo) <= 1 <= sum(hi).
    return lo, hi


def evaluate_lp_and_min(lo, hi, windows, d, A_stack, c_W, n_pgd=4):
    """Solve LP, find a min via PGD, return (lp_val, min_val, primal)."""
    lp_val, primal = _solve_epigraph_lp_with_primal(lo, hi, windows, d)
    if not np.isfinite(lp_val):
        return lp_val, None, None

    # PGD min
    rng = np.random.RandomState(0)
    def project(v, lo, hi):
        for _ in range(80):
            v = np.minimum(np.maximum(v, lo), hi)
            n = len(v)
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u) - 1
            idx = np.arange(1, n + 1)
            rho_arr = np.where(u - cssv / idx > 0)[0]
            if len(rho_arr) == 0: return None
            rho = rho_arr[-1]
            theta = cssv[rho] / (rho + 1)
            v = np.maximum(v - theta, 0.0)
            if abs(v.sum() - 1) < 1e-10 and (v >= lo - 1e-12).all() and (v <= hi + 1e-12).all():
                return v
        return None
    best = np.inf; best_mu = None
    for k in range(n_pgd):
        mu = primal[d*d:d*d+d].copy() if k == 0 else (lo + hi) / 2
        if k > 1:
            mu = primal[d*d:d*d+d] + 1e-6 * rng.randn(d) * (hi - lo)
        mu_p = project(mu, lo, hi)
        if mu_p is None: continue
        mu = mu_p
        lr = 0.05
        for it in range(1500):
            Amu = A_stack @ mu
            tv = c_W * (Amu * mu).sum(axis=1)
            m = tv.max()
            w = np.exp(200.0 * (tv - m)); w /= w.sum()
            grad = 2.0 * (w[:, None] * c_W[:, None] * Amu).sum(axis=0)
            mn = project(mu - lr * grad, lo, hi)
            if mn is None: break
            mu = mn
        Amu = A_stack @ mu
        tv = c_W * (Amu * mu).sum(axis=1)
        v = tv.max()
        if v < best:
            best, best_mu = v, mu.copy()
    return lp_val, best, primal


def main():
    d = 20
    print("Building data...")
    from kkt_correct_mu_star import build_window_data
    A_stack, c_W = build_window_data(d)
    windows = build_windows(d)

    target = 1.281

    # Use mu_star from previous run (computed there); we need a NEAR-mu* box.
    # Quick re-compute mu_star coarsely.
    rng = np.random.RandomState(0)
    print("Re-finding approximate mu_star...")
    from _diag_lp_gap import find_mu_star_quick
    mu_star, val_star, _, _ = find_mu_star_quick(d, n_starts=20, n_iters=2000)
    print(f"  val_star ≈ {val_star:.4f}")
    print(f"  mu_star = {np.round(mu_star, 4)}")

    # ===== Test 1: BAR-shaped box around a non-optimal point =====
    # Make centers off mu_star and check LP gap
    print("\n=== Test: tight-and-uniform-width boxes around mu_star at depth 24 to 40 ===")
    print(f"{'depth':>5s} {'width':>10s} {'lp_val':>12s} {'pgd_min':>12s} {'gap':>10s} {'closes?':>8s}")
    for depth in [24, 26, 28, 30, 33, 36, 40, 50]:
        width = 2.0 ** -depth
        # build box around mu_star
        lo = np.maximum(mu_star - width/2, 0.0)
        hi = np.minimum(mu_star + width/2, 1.0)
        if lo.sum() > 1: lo = mu_star - width
        if hi.sum() < 1: hi = mu_star + width
        if not (lo.sum() <= 1.0 + 1e-15 <= hi.sum()):
            # adjust: shift center so simplex passes through. Just widen by 0:
            pass
        lp_val, min_val, primal = evaluate_lp_and_min(lo, hi, windows, d, A_stack, c_W, n_pgd=2)
        gap = (min_val - lp_val) if (min_val is not None and np.isfinite(lp_val)) else float('nan')
        passes = "yes" if (lp_val > target) else "NO"
        if min_val is None: min_val = float('nan')
        print(f"{depth:>5d} {width:>10.3e} {lp_val:>12.5f} {min_val:>12.5f} {gap:>10.5e} {passes:>8s}")

    # ===== Test 2: SLIVER boxes (mostly tiny, one or two wide axes) =====
    # Mimics D_SHIFT saturation case
    print("\n=== Test: sliver boxes (most axes 1e-9, 1-2 axes wider) — saturation case ===")
    print(f"{'wide_axes':>10s} {'wide_width':>11s} {'lp_val':>12s} {'pgd_min':>12s} {'gap':>10s}")
    for n_wide in [1, 2, 3, 5]:
        for wide_w in [1e-2, 1e-3, 1e-4]:
            half = np.full(d, 5e-10)  # very tight
            wide_axes = rng.choice(d, n_wide, replace=False)
            half[wide_axes] = wide_w / 2
            lo = np.maximum(mu_star - half, 0.0)
            hi = np.minimum(mu_star + half, 1.0)
            if not (lo.sum() <= 1.0 <= hi.sum()):
                # tweak center: move within wide axes
                pass
            lp_val, min_val, primal = evaluate_lp_and_min(lo, hi, windows, d, A_stack, c_W, n_pgd=2)
            gap = (min_val - lp_val) if (min_val is not None and np.isfinite(lp_val)) else float('nan')
            mv = min_val if min_val is not None else float('nan')
            print(f"{n_wide:>10d} {wide_w:>11.3e} {lp_val:>12.5f} {mv:>12.5f} {gap:>10.5e}")

    # ===== Test 3: tight uniform 1e-9 box with mu_lp drift inspection =====
    print("\n=== Test: tight uniform width=1e-9 box around mu_star; print mu_lp - mu_star ===")
    width = 1e-9
    lo = np.maximum(mu_star - width/2, 0.0)
    hi = np.minimum(mu_star + width/2, 1.0)
    lp_val, primal = _solve_epigraph_lp_with_primal(lo, hi, windows, d)
    if primal is not None:
        n_y = d*d
        Y = primal[:n_y].reshape(d,d)
        mu_lp = primal[n_y:n_y+d]
        z = primal[n_y+d]
        TV_at_mu = c_W * (A_stack @ mu_lp * mu_lp).sum(axis=1)
        TV_at_mu_max = TV_at_mu.max()
        print(f"  width={width:.0e}, lp_val={lp_val:.6f}, max TV at mu_lp = {TV_at_mu_max:.6f}")
        print(f"  z - max_TV(mu_lp) = {z - TV_at_mu_max:.3e}  (LP underestimates true TV by this)")
        Yconv = mu_lp[:,None] * mu_lp[None,:]
        print(f"  max|Y - mu_lp⊗mu_lp| = {np.abs(Y-Yconv).max():.3e}")
        print(f"  ||mu_lp - mu_star||_inf = {np.abs(mu_lp - mu_star).max():.3e}")

    # ===== Test 4: same but with fixed mu (eliminate mu drift). Check LP at boundary =====
    print("\n=== Test: degenerate box {mu_star} — what is LP value? ===")
    width = 0  # try lo == hi == mu_star
    lo = mu_star.copy(); hi = mu_star.copy()
    lp_val, primal = _solve_epigraph_lp_with_primal(lo, hi, windows, d)
    if primal is not None:
        n_y = d*d
        Y = primal[:n_y].reshape(d,d)
        mu_lp = primal[n_y:n_y+d]
        z = primal[n_y+d]
        Yconv = mu_lp[:,None] * mu_lp[None,:]
        TV_at_mu = c_W * (A_stack @ mu_lp * mu_lp).sum(axis=1)
        print(f"  LP_val (mu pinned) = {lp_val:.8f}")
        print(f"  TV_max(mu_star) = {TV_at_mu.max():.8f}")
        print(f"  gap = {TV_at_mu.max() - lp_val:.3e}")
        print(f"  max|Y - mu_lp⊗mu_lp| = {np.abs(Y-Yconv).max():.3e}")

    # ===== Test 5: Aggressive — what if we ADD RLT-symmetry? =====
    print("\n=== Test: would Y_ij = Y_ji equality help? Check if asymmetric Y exploited. ===")
    # rebuild last small box and inspect Y asymmetry magnitude relative to lp_val
    width = 1e-9
    lo = np.maximum(mu_star - width/2, 0.0)
    hi = np.minimum(mu_star + width/2, 1.0)
    lp_val, primal = _solve_epigraph_lp_with_primal(lo, hi, windows, d)
    if primal is not None:
        n_y = d*d
        Y = primal[:n_y].reshape(d,d)
        Y_asym_inf = np.abs(Y - Y.T).max()
        print(f"  ||Y - Y^T||_inf = {Y_asym_inf:.3e}; lp_val = {lp_val:.5f}")

    # ===== Test 6: Try ZERO-width box with extreme mu (mu away from mu_star) =====
    # Pick a non-optimal but feasible mu — compute its TV
    print("\n=== Test: LP at degenerate box {mu0} for mu0 = uniform: what's it return? ===")
    mu0 = np.ones(d) / d
    TV_unif = c_W * (A_stack @ mu0 * mu0).sum(axis=1)
    print(f"  TV_max(uniform) = {TV_unif.max():.6f}")
    lp_val, primal = _solve_epigraph_lp_with_primal(mu0, mu0, windows, d)
    print(f"  LP value at degenerate box: {lp_val:.6f}")


if __name__ == "__main__":
    main()
