"""Read dumped stuck boxes and analyse why each tier failed.

For each stuck box B:
  - geometry: width, depth, # boundary axes (lo<=1e-12), # tight axes
  - bounds (vs. target=1.2): natural, autoconv, joint-McCormick, epi LP
  - true val_B: f at centroid (UB) AND f at boxed-projected mu* (LB)

We dump the 30 worst-case boxes (those with the smallest gap to
target — i.e. where the bound is closest to but below 1.2) and
characterise the geometry. If they're all near a single attractor,
that attractor is what we need to handle. If they're scattered,
the issue is something else.
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


def f_at(mu, windows):
    """Compute f(mu) = max_W mu^T M_W mu at a single point."""
    best = -np.inf
    for w in windows:
        s = 0.0
        for (i, j) in w.pairs_all:
            s += mu[i] * mu[j]
        v = float(w.scale) * s
        if v > best:
            best = v
    return best


def project_to_simplex(mu):
    """Euclidean projection onto Δ_d (Wang & Carreira-Perpiñán)."""
    n = len(mu)
    u = np.sort(mu)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.where(u - cssv / np.arange(1, n + 1) > 0)[0]
    if len(rho) == 0:
        return np.maximum(mu, 0)  # degenerate
    rho = rho[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(mu - theta, 0)


def project_to_box_simplex(mu, lo, hi):
    """Project mu to box ∩ simplex by clipping then re-normalising."""
    mu = np.clip(mu, lo, hi)
    s = mu.sum()
    if abs(s - 1.0) < 1e-12:
        return mu
    # Distribute the gap into still-flexible coords proportional to slack.
    if s < 1.0:
        slack = hi - mu
    else:
        slack = mu - lo
    tot_slack = slack.sum()
    if tot_slack < 1e-15:
        return mu
    direction = 1.0 if s < 1.0 else -1.0
    return mu + direction * abs(s - 1.0) * slack / tot_slack


def main():
    target = 1.2
    d = 10

    npz = np.load('stuck_d10_master_queue.npz')
    los = npz['lo']  # (N, d)
    his = npz['hi']
    deps = npz['depths']
    N = len(los)
    print(f"Loaded {N} stuck boxes from master queue.")
    print(f"depth: min={int(deps.min())} max={int(deps.max())} "
          f"median={int(np.median(deps))}")

    npz_mu = np.load('mu_star_d10.npz')
    mu_star = npz_mu['mu']
    f_star = float(npz_mu['f'])
    print(f"val(10) UB = {f_star:.6f}  target = {target}  slack = {f_star - target:.4f}")

    windows = build_windows(d)
    n_W = len(windows)

    # Per-box geometry stats
    widths = (his - los).max(axis=1)        # max axis width per box
    n_boundary = (los <= 1e-12).sum(axis=1)  # # axes pinned at lo=0
    n_tight = ((his - los) <= 1e-9).sum(axis=1)  # # axes ~ collapsed

    print(f"\nGeometry across {N} stuck boxes:")
    print(f"  width: min={widths.min():.2e} max={widths.max():.2e} "
          f"median={np.median(widths):.2e}")
    print(f"  boundary axes (lo<=1e-12): min={n_boundary.min()} "
          f"max={n_boundary.max()} median={int(np.median(n_boundary))}")
    print(f"  tight axes (width<=1e-9): min={n_tight.min()} "
          f"max={n_tight.max()} median={int(np.median(n_tight))}")

    # Pick 20 boxes spanning depth range for detailed analysis.
    # Sort by depth, sample evenly.
    order = np.argsort(deps)
    pick = order[np.linspace(0, len(order) - 1, 20).astype(int)]

    print("\n" + "=" * 110)
    print(f"{'idx':>5} {'depth':>5} {'width':>10} {'bdry':>4} {'tight':>5} "
          f"{'natural':>8} {'autoconv':>8} {'jntMcC':>8} {'epiLP':>8} "
          f"{'f(cent)':>8} {'gap':>7}")
    print("=" * 110)

    for k in pick:
        lo = los[k].astype(np.float64)
        hi = his[k].astype(np.float64)
        depth = int(deps[k])
        wmax = float(np.max(hi - lo))
        nb = int((lo <= 1e-12).sum())
        nt = int(((hi - lo) <= 1e-9).sum())

        # f at centroid (projected to box ∩ simplex) — UB on val_B
        cent = (lo + hi) / 2.0
        cent_p = project_to_box_simplex(cent, lo, hi)
        f_cent = f_at(cent_p, windows)

        # Bounds: scan all windows (slow but rigorous), pick best
        nat_best = -np.inf
        auto_best = -np.inf
        for w in windows:
            nat_best = max(nat_best, bound_natural(lo, hi, w))
            auto_best = max(auto_best, bound_autoconv(lo, hi, w, d))

        # Joint-McCormick at the best window heuristically (use centroid's
        # argmax window for speed — full scan is expensive)
        best_w = None
        best_v = -np.inf
        for w in windows:
            s = 0.0
            for (i, j) in w.pairs_all:
                s += cent_p[i] * cent_p[j]
            v = float(w.scale) * s
            if v > best_v:
                best_v = v
                best_w = w
        try:
            jmc = bound_mccormick_joint_face_lp(lo, hi, best_w, float(best_w.scale))
        except Exception:
            jmc = float('-inf')

        # Epigraph LP — multi-window
        try:
            cert, lp_val, _ = bound_epigraph_int_ge_with_marginals(
                lo, hi, windows, d, target,
            )
        except Exception:
            lp_val = float('-inf')

        gap = target - max(nat_best, auto_best, jmc, lp_val)
        print(f"{k:5d} {depth:5d} {wmax:10.2e} {nb:4d} {nt:5d} "
              f"{nat_best:8.4f} {auto_best:8.4f} {jmc:8.4f} {lp_val:8.4f} "
              f"{f_cent:8.4f} {gap:+7.4f}")

    print()
    print("Reading 'gap' col: positive = the bound stack misses the target")
    print("by that much. The closer to 0 (but positive), the closer the box")
    print("is to certifying — those are the 'almost done' boxes.")
    print()
    print("'f(cent)' ≥ target = 1.2 always (since the centroid lies in B and")
    print("val_B = max over B). If f(cent) >> target, the box has lots of")
    print("headroom but the bound stack just can't reach it.")


if __name__ == "__main__":
    main()
