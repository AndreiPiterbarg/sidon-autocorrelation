"""Followup: where is the LP gap coming from?

At width=1e-4, McCormick is essentially tight. Y ≈ mu_lp⊗mu_lp.
But LP gap is 8e-4. Hypothesis: the LP is exploiting that mu_lp can drift
within the box (not pinned to mu_star) AND that Y in the LP is the convex
hull of mu*mu^T over the box, but actually Y_LP ≈ rank-1 mu_lp⊗mu_lp.

So LP is computing: min_{mu in box ∩ simplex} max_W TV_W(mu)  ... approximately.
The gap is because true_max_on_box >> min_on_box (the box contains points
much worse than mu*).

Actually re-read: "the LP value is a LOWER bound on min_B max_W TV_W". That's
exactly right — the BnB closes a box only if LP_lb >= target. If true min_B
max_W = val(20) ≈ 1.298 and target = 1.281, the LP_lb just needs to clear 1.281.

But we observe LP_lb = 1.34279 — well above 1.281! So this BOX would close.

But the BnB stalled at 99.8% volume. So the stuck boxes are NOT at mu_star.
They are at some other location where the LP ≤ 1.281 but the true min ≥ 1.281.

Let me check what the LP gives at random box centers, and at different scales.
"""
import os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import _solve_epigraph_lp
from _diag_lp_gap import _solve_epigraph_lp_with_primal


def evaluate_min_box_via_pgd(lo, hi, A_stack, c_W, n_starts=20, n_iters=2000, beta=200.0, seed=0):
    """Smoothed PGD for min_{mu in box ∩ simplex} max_W TV_W(mu)."""
    rng = np.random.RandomState(seed)
    d = len(lo)
    best_val = np.inf
    best_mu = None
    def project_to_box_simplex(v, lo, hi):
        # Iterate: clip to box; project to simplex; clip; ...
        for _ in range(80):
            v = np.minimum(np.maximum(v, lo), hi)
            # project to simplex
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

    for k in range(n_starts):
        v = rng.uniform(lo, hi)
        # check feasibility
        mu = project_to_box_simplex(v, lo, hi)
        if mu is None: continue
        lr = 0.05
        for it in range(n_iters):
            Amu = A_stack @ mu
            tv = c_W * (Amu * mu).sum(axis=1)
            m = tv.max()
            w = np.exp(beta * (tv - m)); w /= w.sum()
            grad = 2.0 * (w[:, None] * c_W[:, None] * Amu).sum(axis=0)
            mu_new = project_to_box_simplex(mu - lr * grad, lo, hi)
            if mu_new is None: break
            mu = mu_new
        Amu = A_stack @ mu
        tv = c_W * (Amu * mu).sum(axis=1)
        v_max = tv.max()
        if v_max < best_val:
            best_val, best_mu = v_max, mu.copy()
    return best_val, best_mu


def main():
    d = 20
    print("Building data...")
    # Build A_stack/c_W and windows
    from kkt_correct_mu_star import build_window_data
    A_stack, c_W = build_window_data(d)
    windows = build_windows(d)

    # Generate stuck-tail-like boxes: random simplex-feasible boxes of various widths,
    # NOT centered at mu_star. (The stall is at 99.8%, so these are tiny remaining slivers.)
    rng = np.random.RandomState(42)

    print("\n=== Width-scaling sweep ===")
    print(f"{'width':>10s} {'lp_val':>10s} {'min_pgd':>10s} {'gap':>10s} {'>1.281?':>8s}")
    print("-" * 60)
    for width in [1e-2, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
        # Build a box NOT centered at the global min — pick a random center on the simplex.
        center = rng.dirichlet(np.ones(d) * 0.8)
        lo = np.maximum(center - width/2, 0.0)
        hi = np.minimum(center + width/2, 1.0)
        if lo.sum() > 1 or hi.sum() < 1:
            # rescale center so that simplex passes through box
            scale = 1.0 / center.sum()
            center = center * scale
            lo = np.maximum(center - width/2, 0.0)
            hi = np.minimum(center + width/2, 1.0)
        lp_val, _, _, _, _ = _solve_epigraph_lp(lo, hi, windows, d)
        min_val, _ = evaluate_min_box_via_pgd(lo, hi, A_stack, c_W, n_starts=4, n_iters=800)
        gap = min_val - lp_val if np.isfinite(min_val) and np.isfinite(lp_val) else float('nan')
        passes = "yes" if lp_val > 1.281 else "NO"
        print(f"{width:>10.2e} {lp_val:>10.4f} {min_val:>10.4f} {gap:>10.4f} {passes:>8s}")

    # Now: at the values that DON'T pass, what does the LP exploit?
    # Take the random-center box at width=1e-4
    print("\n=== Diagnose a (random-center, 1e-4)-box where LP < 1.281 ===")
    found = False
    for trial in range(20):
        center = rng.dirichlet(np.ones(d) * 0.8)
        lo = np.maximum(center - 5e-5, 0.0)
        hi = np.minimum(center + 5e-5, 1.0)
        if lo.sum() > 1 or hi.sum() < 1:
            continue
        lp_val, primal = _solve_epigraph_lp_with_primal(lo, hi, windows, d)
        if lp_val < 1.281:
            print(f"  Found a box with LP_val = {lp_val:.5f}, center sum = {center.sum():.4f}")
            n_y = d*d
            Y = primal[:n_y].reshape(d,d)
            mu_lp = primal[n_y:n_y+d]
            z = primal[n_y+d]
            # Compute true minimum on this box
            min_val, min_mu = evaluate_min_box_via_pgd(lo, hi, A_stack, c_W, n_starts=8, n_iters=2000)
            print(f"  true_min on box (PGD) = {min_val:.5f}")
            print(f"  GAP = {min_val - lp_val:.5f}")
            # Y vs mu_lp ⊗ mu_lp
            Yconv = mu_lp[:, None] * mu_lp[None, :]
            print(f"  max |Y - mu_lp ⊗ mu_lp| = {np.abs(Y - Yconv).max():.3e}")
            print(f"  Y - mu_lp ⊗ mu_lp: min = {(Y-Yconv).min():.3e}, max = {(Y-Yconv).max():.3e}")
            # Symmetry of Y
            print(f"  ||Y - Y^T||_inf = {np.abs(Y-Y.T).max():.3e}")
            # What the LP "sees" as max TV via Y
            TV_via_Y = np.zeros(len(windows))
            for kw, w in enumerate(windows):
                s = sum(Y[i,j] for (i,j) in w.pairs_all)
                TV_via_Y[kw] = w.scale * s
            kbind = int(np.argmax(TV_via_Y))
            wb = windows[kbind]
            # Compare to TV at mu_lp directly
            TVs_at_mu = c_W * (A_stack @ mu_lp * mu_lp).sum(axis=1)
            print(f"  Binding window: W{kbind} (ell={wb.ell}, s={wb.s_lo}, scale={wb.scale:.3f})")
            print(f"  TV_via_Y[binding] = {TV_via_Y[kbind]:.6f}, TV_at_mu_lp[binding] = {TVs_at_mu[kbind]:.6f}")
            print(f"  Σ Y over S_W = {sum(Y[i,j] for (i,j) in wb.pairs_all):.6f}")
            print(f"  Σ mu_lp ⊗ mu_lp over S_W = {sum(mu_lp[i]*mu_lp[j] for (i,j) in wb.pairs_all):.6f}")
            print(f"  max TV at mu_lp = {TVs_at_mu.max():.6f}, argmax = W{int(TVs_at_mu.argmax())}")
            # Are mu_lp values pinned to box bounds?
            on_lo = ((mu_lp - lo) < 1e-6).sum()
            on_hi = ((hi - mu_lp) < 1e-6).sum()
            print(f"  mu_lp pinned to lo: {on_lo}/{d}, pinned to hi: {on_hi}/{d}")
            # Print mu_lp - center
            print(f"  ||mu_lp - center||_inf = {np.abs(mu_lp - center).max():.3e}")
            found = True
            break
    if not found:
        print("  (no random box found with LP<1.281; would need stuck-tail box)")


if __name__ == "__main__":
    main()
