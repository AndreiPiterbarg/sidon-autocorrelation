"""Diagnose d=14 t=1.253 stuck-box geometry.

Runs SERIAL BnB with the full epigraph LP cascade enabled at every box
beyond `EPI_DEPTH`. Boxes at `depth >= STUCK_DEPTH` whose epigraph LP
value is still below `target` are saved as "stuck" and analysed.

For each stuck box we record:
  - lo, hi
  - centroid (and centroid normalised onto simplex)
  - max_W TV_W at centroid (true objective)
  - epigraph LP value (current relaxation)
  - GAP = max_W TV - epigraph_lp
  - winning W index

Aggregate analysis:
  - PCA (top 5 directions, explained variance)
  - Distance-from-mean distribution
  - Coordinate min/max profile (which coords are "U-shaped")
  - Effective rank of stuck-box manifold
  - Typical GAP and dominant-window histogram

This is a diagnostic only -- not part of the BnB hot path.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
import time
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from interval_bnb.box import Box  # noqa: E402
from interval_bnb.bound_eval import (  # noqa: E402
    batch_bounds_full,
    _adjacency_matrix,
)
from interval_bnb.bound_epigraph import _solve_epigraph_lp  # noqa: E402
from interval_bnb.symmetry import half_simplex_cuts  # noqa: E402
from interval_bnb.windows import build_windows  # noqa: E402


def _max_w_tv(mu: np.ndarray, A_tensor: np.ndarray,
              scales: np.ndarray) -> Tuple[float, int]:
    """Return (max_W TV_W(mu), arg_W)."""
    # mu^T A_W mu for every W: einsum.
    vals = np.einsum("i,wij,j->w", mu, A_tensor, mu) * scales
    w_idx = int(np.argmax(vals))
    return float(vals[w_idx]), w_idx


def diagnose(
    d: int,
    target_str: str,
    *,
    max_nodes: int = 200_000,
    time_budget_s: float = 600.0,
    stuck_depth: int = 30,
    epi_depth: int = 20,
    n_save: int = 200,
    out_path: str = "stuck_d14.pkl",
):
    """Run serial BnB; save stuck boxes; print analysis."""
    target_q = Fraction(target_str)
    target_f = float(target_q)
    sym_cuts = half_simplex_cuts(d)
    windows = build_windows(d)
    A_tensor, scales = window_tensor_local(windows, d)

    initial = Box.initial(d, sym_cuts)

    # Stack entries: (Box, depth, parent_cache, axis, which_end).
    stack: List[Tuple[Box, int, Optional[tuple], int, Optional[str]]] = []
    stack.append((initial, 0, None, -1, None))

    stuck_boxes: List[Dict] = []
    n_certified = 0
    n_processed = 0
    n_split = 0
    max_depth_seen = 0
    epi_attempts = 0
    epi_certs = 0

    t0 = time.time()
    last_log = t0
    while stack:
        if n_processed >= max_nodes:
            print(f"[diag] max_nodes={max_nodes} reached")
            break
        if time.time() - t0 > time_budget_s:
            print(f"[diag] time_budget_s={time_budget_s} exceeded")
            break
        if len(stuck_boxes) >= n_save:
            print(f"[diag] saved n_save={n_save} stuck boxes; stopping")
            break

        B, depth, parent_cache, _, _ = stack.pop()
        n_processed += 1
        if depth > max_depth_seen:
            max_depth_seen = depth

        if not B.intersects_simplex():
            continue

        # Standard fast bound (autoconv + McCormick + natural).
        lb_fast, w_idx, which, _, my_cache = batch_bounds_full(
            B.lo, B.hi, A_tensor, scales, target_f,
        )

        if lb_fast >= target_f:
            n_certified += 1
            continue

        # If depth >= EPI_DEPTH, try the epigraph LP.
        epi_val = None
        if depth >= epi_depth:
            epi_attempts += 1
            lp_val, *_ = _solve_epigraph_lp(B.lo, B.hi, windows, d)
            if np.isfinite(lp_val):
                epi_val = lp_val
                if lp_val >= target_f - 1e-9:
                    epi_certs += 1
                    n_certified += 1
                    continue

        # Box NOT certified. If at stuck depth, log it.
        if depth >= stuck_depth:
            centroid = 0.5 * (B.lo + B.hi)
            s = centroid.sum()
            centroid_simplex = centroid / s if s > 0 else centroid
            tv_max, tv_argw = _max_w_tv(centroid_simplex, A_tensor, scales)
            # Compute epigraph LP if not done yet.
            if epi_val is None:
                lp_val, *_ = _solve_epigraph_lp(B.lo, B.hi, windows, d)
                epi_val = lp_val if np.isfinite(lp_val) else None
            stuck_boxes.append({
                "depth": depth,
                "lo": B.lo.copy(),
                "hi": B.hi.copy(),
                "centroid": centroid.copy(),
                "centroid_simplex": centroid_simplex.copy(),
                "lb_fast": float(lb_fast),
                "lb_fast_window": int(w_idx),
                "lb_fast_which": str(which),
                "tv_at_centroid": float(tv_max),
                "tv_argw": int(tv_argw),
                "epi_lp_val": (float(epi_val) if epi_val is not None
                               else float("nan")),
                "max_width": float(B.max_width()),
            })

        # Split widest axis. Push both children.
        if B.max_width() < 1e-12:
            # Numerical bottom; skip.
            continue
        axis = int(np.argmax(B.hi - B.lo))
        left, right = B.split(axis)
        n_split += 1
        if right.intersects_simplex():
            stack.append((right, depth + 1, my_cache, axis, "lo"))
        if left.intersects_simplex():
            stack.append((left, depth + 1, my_cache, axis, "hi"))

        if time.time() - last_log > 5.0:
            elapsed = time.time() - t0
            print(f"[diag] t={elapsed:6.1f}s  n_proc={n_processed:>8d}  "
                  f"queue={len(stack):>6d}  cert={n_certified:>6d}  "
                  f"stuck={len(stuck_boxes):>4d}  depth={max_depth_seen:>3d}  "
                  f"epi_certs={epi_certs}/{epi_attempts}")
            last_log = time.time()

    elapsed = time.time() - t0
    print(f"\n[diag] FINAL t={elapsed:.1f}s")
    print(f"  n_processed={n_processed}  n_certified={n_certified}")
    print(f"  n_split={n_split}  max_depth={max_depth_seen}")
    print(f"  epi: {epi_certs}/{epi_attempts} certified")
    print(f"  stuck_boxes saved: {len(stuck_boxes)}")

    # Save raw boxes for offline analysis.
    with open(out_path, "wb") as f:
        pickle.dump({
            "d": d, "target": target_str,
            "stuck_boxes": stuck_boxes,
            "stuck_depth": stuck_depth,
            "epi_depth": epi_depth,
            "n_processed": n_processed,
            "n_certified": n_certified,
            "elapsed_s": elapsed,
        }, f)
    print(f"  saved -> {out_path}")

    # ---- ANALYSIS ----
    if len(stuck_boxes) < 5:
        print("[diag] too few stuck boxes for analysis; aborting.")
        return

    analyse(stuck_boxes, d, windows)


def window_tensor_local(windows, d):
    from interval_bnb.bound_eval import window_tensor
    return window_tensor(windows, d)


def analyse(stuck_boxes, d, windows):
    print(f"\n{'='*70}\nSTUCK-BOX ANALYSIS (n={len(stuck_boxes)}, d={d})\n{'='*70}")
    # --- centroids matrix ---
    C = np.array([b["centroid_simplex"] for b in stuck_boxes])  # (n, d)
    n = C.shape[0]
    mean = C.mean(axis=0)
    print(f"\n[A] Mean centroid (across stuck boxes):")
    print(f"  mu_mean = {np.round(mean, 5).tolist()}")
    print(f"  argmax = {int(np.argmax(mean))}, argmin = {int(np.argmin(mean))}")
    print(f"  min/max coord = {mean.min():.5f} / {mean.max():.5f}")

    # Coordinate-by-coordinate spread (std).
    std = C.std(axis=0)
    print(f"\n[B] Per-coordinate spread (std across stuck boxes):")
    print(f"  std = {np.round(std, 5).tolist()}")
    print(f"  max std at coord {int(np.argmax(std))} (val {std.max():.5f})")
    print(f"  min std at coord {int(np.argmin(std))} (val {std.min():.5f})")

    # --- PCA (SVD on centred centroids) ---
    Cc = C - mean
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    var = (S ** 2) / max(n - 1, 1)
    var_frac = var / var.sum() if var.sum() > 0 else var
    cum = np.cumsum(var_frac)
    print(f"\n[C] PCA of stuck-box centroids:")
    print(f"  Top-5 explained variance fraction: {np.round(var_frac[:5], 4).tolist()}")
    print(f"  Cumulative: {np.round(cum[:8], 4).tolist()}")
    # Effective rank: number of components for 95% variance.
    eff_rank_95 = int(np.searchsorted(cum, 0.95) + 1)
    eff_rank_99 = int(np.searchsorted(cum, 0.99) + 1)
    print(f"  Effective rank (95%): {eff_rank_95},  (99%): {eff_rank_99}")
    print(f"  --> manifold dim ~{eff_rank_95}")

    # --- Top PCA directions ---
    print(f"\n[D] Top-3 principal directions (signed coords):")
    for k in range(min(3, Vt.shape[0])):
        v = Vt[k]
        # Find the largest |coords|.
        order = np.argsort(-np.abs(v))[:6]
        topcoords = [(int(i), round(float(v[i]), 4)) for i in order]
        print(f"  PC{k+1} (var_frac={var_frac[k]:.4f}): {topcoords}")

    # --- Distance from mean ---
    dists = np.linalg.norm(Cc, axis=1)
    print(f"\n[E] L2 distance of centroids from mean:")
    print(f"  min={dists.min():.5f}  median={np.median(dists):.5f}  "
          f"max={dists.max():.5f}  std={dists.std():.5f}")

    # --- GAPs ---
    tv = np.array([b["tv_at_centroid"] for b in stuck_boxes])
    epi = np.array([b["epi_lp_val"] for b in stuck_boxes])
    lb_fast = np.array([b["lb_fast"] for b in stuck_boxes])
    valid = np.isfinite(epi)
    gap_epi = tv[valid] - epi[valid]
    gap_fast = tv - lb_fast
    print(f"\n[F] GAP analysis (max_W TV at centroid - relaxation LB):")
    print(f"  fast-bound GAP:  min={gap_fast.min():.6f}  median={np.median(gap_fast):.6f}  "
          f"max={gap_fast.max():.6f}")
    if valid.any():
        print(f"  epigraph GAP:    min={gap_epi.min():.6f}  median={np.median(gap_epi):.6f}  "
              f"max={gap_epi.max():.6f}")
        print(f"  epi LP val:      min={epi[valid].min():.6f}  median={np.median(epi[valid]):.6f}  "
              f"max={epi[valid].max():.6f}")

    # --- Window histogram ---
    win_hist = {}
    for b in stuck_boxes:
        ww = b["tv_argw"]
        win_hist[ww] = win_hist.get(ww, 0) + 1
    top_windows = sorted(win_hist.items(), key=lambda x: -x[1])[:5]
    print(f"\n[G] Argmax-window histogram (which W binds at centroid):")
    for w_idx, count in top_windows:
        w = windows[w_idx]
        print(f"  W{w_idx} (ell={w.ell}, s={w.s_lo}): {count} boxes "
              f"({100*count/n:.1f}%)")

    # --- Coordinate "U-shape" fingerprint ---
    # Sort coordinates by index; print mean profile.
    print(f"\n[H] Coordinate-by-coordinate mean profile (look for U-shape):")
    for i in range(d):
        bar = "*" * int(50 * mean[i] / max(mean.max(), 1e-9))
        print(f"  mu_{i:2d}: {mean[i]:.4f} ± {std[i]:.4f}  |{bar}")

    # --- Manifold check: are stuck centroids clustered around 1 or many points? ---
    # Quick: relative std vs cluster radius.
    rel_spread = dists.std() / max(dists.mean(), 1e-9)
    print(f"\n[I] Cluster tightness: rel-std(distance from mean) = {rel_spread:.4f}")
    print(f"    (low = single tight cluster; high = multimodal)")

    # --- Box-width profile (which axis still wide at stuck depth?) ---
    widths = np.array([b["hi"] - b["lo"] for b in stuck_boxes])  # (n, d)
    mean_widths = widths.mean(axis=0)
    print(f"\n[J] Mean width per axis at stuck boxes (axis still un-split):")
    for i in range(d):
        print(f"  axis {i:2d}: width={mean_widths[i]:.4e}  "
              f"(=2^{np.log2(max(mean_widths[i], 1e-30)):.2f})")
    widest = int(np.argmax(mean_widths))
    print(f"  widest axis (mean): {widest}, width={mean_widths[widest]:.4e}")

    # --- Save analysis JSON ---
    analysis_out = {
        "n_stuck": n,
        "mean_centroid": mean.tolist(),
        "std_centroid": std.tolist(),
        "mean_widths": mean_widths.tolist(),
        "pca_var_frac": var_frac.tolist(),
        "pca_cumulative": cum.tolist(),
        "eff_rank_95": eff_rank_95,
        "eff_rank_99": eff_rank_99,
        "pc_directions": Vt[:5].tolist(),
        "dist_from_mean_stats": {
            "min": float(dists.min()), "median": float(np.median(dists)),
            "max": float(dists.max()), "std": float(dists.std()),
            "rel_spread": float(rel_spread),
        },
        "gap_fast": {
            "min": float(gap_fast.min()), "median": float(np.median(gap_fast)),
            "max": float(gap_fast.max()),
        },
        "gap_epi": ({
            "min": float(gap_epi.min()), "median": float(np.median(gap_epi)),
            "max": float(gap_epi.max()),
        } if valid.any() else None),
        "top_windows": [
            {"w_idx": int(w_idx), "ell": int(windows[w_idx].ell),
             "s_lo": int(windows[w_idx].s_lo), "count": int(c)}
            for w_idx, c in top_windows
        ],
    }
    out_json = "stuck_d14_analysis.json"
    with open(out_json, "w") as f:
        json.dump(analysis_out, f, indent=2)
    print(f"\n[diag] analysis saved -> {out_json}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=14)
    p.add_argument("--target", type=str, default="1.253")
    p.add_argument("--max-nodes", type=int, default=200_000)
    p.add_argument("--time-budget", type=float, default=600.0)
    p.add_argument("--stuck-depth", type=int, default=30)
    p.add_argument("--epi-depth", type=int, default=20)
    p.add_argument("--n-save", type=int, default=200)
    p.add_argument("--out", type=str, default="stuck_d14.pkl")
    args = p.parse_args()

    diagnose(
        args.d, args.target,
        max_nodes=args.max_nodes,
        time_budget_s=args.time_budget,
        stuck_depth=args.stuck_depth,
        epi_depth=args.epi_depth,
        n_save=args.n_save,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
