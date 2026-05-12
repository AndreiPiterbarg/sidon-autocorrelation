"""Diagnose: does the bound stack actually close as box width → 0?

Construct boxes around mu_star_d10 with shrinking widths, and for each
box run every bound tier in the cascade (natural, autoconv, McCormick
SW/NE, joint-face LP, epigraph LP, low-K SDP). Report each tier's
returned LB versus the true val(B) ≈ val(d) = 1.2249, and against the
target = 1.2 (slack = 0.025).

If some tier returns LB ≥ 1.2 at small enough width → BnB can close.
If every tier plateaus below 1.2 even at width 1e-6 → bound stack is
fundamentally inadequate; gap doesn't vanish, BnB cannot terminate.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.windows import build_windows
from interval_bnb.bound_eval import (
    bound_natural, bound_autoconv, bound_mccormick_joint_face_lp,
)
from interval_bnb.bound_epigraph import bound_epigraph_int_ge_with_marginals


def build_box_around_mu(mu, width, d):
    """Box of given half-width around mu, projected to feasible simplex.

    Returns (lo, hi) such that lo <= mu <= hi, lo >= 0, hi <= 1, and the
    box intersects the simplex {sum=1}. Boundary axes (mu_i ≈ 0) get
    lo_i = 0 (preserving the boundary-axis structure that's the
    suspected stall driver).
    """
    lo = np.maximum(mu - width, 0.0)
    hi = np.minimum(mu + width, 1.0)
    # Force lo=0 on truly-zero axes so we test the boundary-degeneracy
    # case the BnB hits in practice.
    lo[mu < 1e-9] = 0.0
    return lo, hi


def true_val_at_mu(mu, windows):
    """f(mu) = max_W mu^T M_W mu — the best-window objective at mu."""
    best = -np.inf
    for w in windows:
        A = np.zeros((len(mu), len(mu)))
        for (i, j) in w.pairs_all:
            A[i, j] = 1.0
        v = float(w.scale) * float(mu @ A @ mu)
        if v > best:
            best = v
    return best


def best_window_at_mu(mu, windows):
    """argmax_W f_W(mu)."""
    best_v = -np.inf
    best_w = None
    for k, w in enumerate(windows):
        A = np.zeros((len(mu), len(mu)))
        for (i, j) in w.pairs_all:
            A[i, j] = 1.0
        v = float(w.scale) * float(mu @ A @ mu)
        if v > best_v:
            best_v = v
            best_w = (k, w)
    return best_w, best_v


def main():
    d = 10
    target = 1.2
    npz = np.load('mu_star_d10.npz')
    mu_star = np.asarray(npz['mu'], dtype=np.float64)
    f_star = float(npz['f'])
    print(f"mu* = {mu_star}")
    print(f"f(mu*) = val({d}) UB = {f_star:.6f}")
    print(f"target = {target}  slack = {f_star - target:.6f}")
    print(f"boundary axes (mu* < 1e-9): {[i for i in range(d) if mu_star[i] < 1e-9]}")

    windows = build_windows(d)
    n_W = len(windows)
    print(f"|W_d| = {n_W}")

    # Pick the active (best) window at mu*
    (kw, w_best), v_best = best_window_at_mu(mu_star, windows)
    print(f"argmax_W at mu* = window #{kw}; f_W(mu*) = {v_best:.6f}")
    print()

    # Test a sequence of shrinking widths around mu*.
    widths = [0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]
    print(f"{'width':>10} {'natural':>10} {'autoconv':>10} {'jointMcC':>10} "
          f"{'epiLP':>10} {'best':>10}  cert(t={target})?")
    print("-" * 90)

    for hw in widths:
        lo, hi = build_box_around_mu(mu_star, hw, d)
        # Natural and autoconv evaluated at the BEST WINDOW for mu*.
        nat = bound_natural(lo, hi, w_best)
        auto = bound_autoconv(lo, hi, w_best, d)
        # Joint-face McCormick LP at the BEST window
        try:
            jmc = bound_mccormick_joint_face_lp(lo, hi, w_best, float(w_best.scale))
        except Exception as e:
            jmc = float('-inf')
        # Epigraph LP — multi-window, returns the LP min over ALL windows
        try:
            cert, lp_val, _ = bound_epigraph_int_ge_with_marginals(
                lo, hi, windows, d, target,
            )
        except Exception as e:
            lp_val = float('-inf')
        best = max(nat, auto, jmc, lp_val)
        cert_flag = "YES" if best >= target else "no"
        print(f"{hw:10.1e} {nat:10.4f} {auto:10.4f} {jmc:10.4f} "
              f"{lp_val:10.4f} {best:10.4f}  {cert_flag}")

    print()
    print("Interpretation:")
    print("  - If any column reaches >= 1.2 as width shrinks → bound is OK,")
    print("    the box CAN close eventually (BnB terminates in principle).")
    print("  - If every column plateaus below 1.2 → bound stack has an")
    print("    asymptotic gap > slack on these boundary boxes → BnB cannot")
    print("    terminate, no matter how deep we split.")


if __name__ == "__main__":
    main()
