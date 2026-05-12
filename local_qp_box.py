"""Empirical val_B audit on stuck d=10 boxes via local QP / projected subgradient.

For each of K=30 sampled stuck boxes, we estimate
    val_B = min_{mu in B intersect Delta_10} f(mu),
    f(mu) = max_W scale_W * sum_{(i,j) in pairs_W} mu_i * mu_j
by running R=20 random restarts of 200 steps of projected subgradient descent.

The subgradient at mu is 2 * scale_W^* * A_{W^*} mu where W^* = argmax over windows.
Projection: clip into [lo, hi], then redistribute residual mass into per-coordinate
slack until sum=1 (sound projection onto B intersect simplex when B is feasible).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from interval_bnb.windows import build_windows, WindowMeta  # noqa: E402
from interval_bnb.bound_epigraph import bound_epigraph_lp_float  # noqa: E402


D = 10
TARGET = 1.2
N_BOXES = 30
N_RESTARTS = 20
N_STEPS = 200
SEED = 42


def precompute_window_arrays(windows: List[WindowMeta], d: int):
    """Return per-window pairs_all as int arrays for fast subgradient eval."""
    Is = []
    Js = []
    scales = []
    for w in windows:
        ij = np.asarray(w.pairs_all, dtype=np.int64)  # (k, 2)
        Is.append(ij[:, 0])
        Js.append(ij[:, 1])
        scales.append(w.scale)
    return Is, Js, np.asarray(scales, dtype=np.float64)


def f_value(mu: np.ndarray, Is, Js, scales) -> Tuple[float, int]:
    """Return (max value, argmax window index)."""
    best = -np.inf
    best_w = -1
    for w in range(len(scales)):
        v = scales[w] * float(np.dot(mu[Is[w]], mu[Js[w]]))
        if v > best:
            best = v
            best_w = w
    return best, best_w


def subgradient_at(mu: np.ndarray, w: int, Is, Js, scales) -> np.ndarray:
    """Subgradient of f at mu given argmax window w.
    g_k = 2 * scale_w * sum_{(k,j) in pairs_all} mu_j (since pairs_all has both orderings)."""
    g = np.zeros_like(mu)
    # f = scale * sum_{(i,j)} mu_i * mu_j over ordered pairs (both orderings included)
    # df/dmu_k = scale * (sum_{j: (k,j) in P} mu_j + sum_{i: (i,k) in P} mu_i)
    #         = 2 * scale * sum_{j: (k,j) in P} mu_j  (since P is symmetric).
    Iw = Is[w]
    Jw = Js[w]
    # accumulate: for each pair (i,j), add scale * mu_j to g[i]
    np.add.at(g, Iw, mu[Jw])
    # By symmetry of pairs_all (both orderings), this already gives the full gradient
    # without needing to also do np.add.at(g, Jw, mu[Iw]).
    g *= scales[w]
    return g


def project_box_simplex(mu: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                        max_iter: int = 50) -> np.ndarray:
    """Project mu onto {x : lo <= x <= hi, sum x = 1}.
    Clip, then iteratively redistribute residual into per-coord slack."""
    x = np.clip(mu, lo, hi)
    for _ in range(max_iter):
        s = x.sum()
        r = 1.0 - s
        if abs(r) < 1e-14:
            break
        if r > 0:
            slack = hi - x
            tot = slack.sum()
            if tot <= 1e-15:
                break
            x = x + (r / tot) * slack
            x = np.minimum(x, hi)
        else:
            slack = x - lo
            tot = slack.sum()
            if tot <= 1e-15:
                break
            x = x - (-r / tot) * slack
            x = np.maximum(x, lo)
    return x


def sample_in_box_simplex(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Draw a point uniformly in [lo, hi]^d then project onto simplex slice."""
    u = rng.random(lo.shape)
    x = lo + u * (hi - lo)
    return project_box_simplex(x, lo, hi)


def descend(mu0: np.ndarray, lo: np.ndarray, hi: np.ndarray,
            Is, Js, scales, n_steps: int) -> Tuple[float, np.ndarray]:
    mu = mu0.copy()
    best_val, _ = f_value(mu, Is, Js, scales)
    best_mu = mu.copy()
    # Step size schedule: 1/sqrt(t)
    for t in range(1, n_steps + 1):
        val, w = f_value(mu, Is, Js, scales)
        if val < best_val:
            best_val = val
            best_mu = mu.copy()
        g = subgradient_at(mu, w, Is, Js, scales)
        # remove component along simplex normal (1-vector), to keep on sum=1 manifold
        g = g - g.mean()
        eta = 0.05 / np.sqrt(t)
        mu = mu - eta * g
        mu = project_box_simplex(mu, lo, hi)
    val, _ = f_value(mu, Is, Js, scales)
    if val < best_val:
        best_val = val
        best_mu = mu.copy()
    return best_val, best_mu


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    data = np.load(os.path.join(_REPO_ROOT, "stuck_d10_master_queue.npz"))
    LO = data["lo"]
    HI = data["hi"]
    n_total = LO.shape[0]
    print(f"Loaded {n_total} stuck boxes; sampling {N_BOXES} with seed={SEED}")

    idx = rng.choice(n_total, size=N_BOXES, replace=False)
    idx.sort()

    windows = build_windows(D)
    Is, Js, scales = precompute_window_arrays(windows, D)
    print(f"Built {len(windows)} windows for d={D}")

    results = []
    any_below = []
    for k, bi in enumerate(idx):
        lo = LO[bi].astype(np.float64)
        hi = HI[bi].astype(np.float64)
        # Sanity: feasible?
        assert lo.sum() <= 1.0 + 1e-9 and hi.sum() >= 1.0 - 1e-9, f"box {bi} infeasible"

        # LP relaxation value
        try:
            lp_val = float(bound_epigraph_lp_float(lo, hi, windows, D))
        except Exception as e:
            lp_val = float("nan")

        best_val = np.inf
        best_mu = None
        for r in range(N_RESTARTS):
            mu0 = sample_in_box_simplex(rng, lo, hi)
            v, mu = descend(mu0, lo, hi, Is, Js, scales, N_STEPS)
            if v < best_val:
                best_val = v
                best_mu = mu

        slack = best_val - TARGET
        gap_lp = best_val - lp_val if np.isfinite(lp_val) else float("nan")
        below = best_val < TARGET
        if below:
            any_below.append((int(bi), best_val, lo.tolist(), hi.tolist(), best_mu.tolist()))

        results.append({
            "box_idx": int(bi),
            "val_B_star": float(best_val),
            "slack_vs_1p2": float(slack),
            "lp_relax": float(lp_val),
            "gap_to_lp": float(gap_lp),
            "below_1p2": bool(below),
            "lo": lo.tolist(),
            "hi": hi.tolist(),
            "witness_mu": best_mu.tolist() if best_mu is not None else None,
        })
        elapsed = time.time() - t0
        print(f"[{k+1:2d}/{N_BOXES}] box={bi:4d}  val_B*={best_val:.6f}  "
              f"slack={slack:+.6f}  LP={lp_val:.6f}  gap_to_LP={gap_lp:+.6f}  "
              f"{'BELOW 1.2!' if below else ''}  ({elapsed:.1f}s)")

    vals = np.array([r["val_B_star"] for r in results])
    summary = {
        "n_boxes": N_BOXES,
        "n_restarts": N_RESTARTS,
        "n_steps": N_STEPS,
        "seed": SEED,
        "min_val_B_star": float(vals.min()),
        "median_val_B_star": float(np.median(vals)),
        "max_val_B_star": float(vals.max()),
        "n_below_1p2": int((vals < TARGET).sum()),
        "any_below_details": any_below,
    }

    print()
    print("=" * 60)
    print(f"min(val_B*)    = {summary['min_val_B_star']:.6f}")
    print(f"median(val_B*) = {summary['median_val_B_star']:.6f}")
    print(f"max(val_B*)    = {summary['max_val_B_star']:.6f}")
    print(f"# below 1.2    = {summary['n_below_1p2']}")
    print("=" * 60)

    out = {"summary": summary, "boxes": results}
    out_path = os.path.join(_REPO_ROOT, "local_qp_audit.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
