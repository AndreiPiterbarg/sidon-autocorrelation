"""Comprehensive diagnostic for d=10 t=1.208 BnB stall.

Strategy: re-implement the BnB main loop in this script so we can
intercept boxes whose `lb_fast < target` AT MODERATE WIDTH (i.e. before
they would get split into oblivion). This gives us a LARGE population of
stuck-or-near-stuck boxes that we can analyze structurally.

We harvest two populations:
  STUCK: lb_fast < target AND max_width < harvest_width  (would-fail)
  NEAR:  lb_fast in (target - 0.05, target) at max_width in (h, 2h)
         (would-stall)

For each, we record:
  - centroid, sorted top coordinates
  - lb_fast (best across all bound types and all windows)
  - winning window (ell, s_lo)
  - max_W TV_W at centroid
  - distance to mu_star (the illustrative one), distance to the
    nearest corner e_i, the nearest 2-corner edge convex combination,
    and the smallest coordinate of centroid

We aggregate distributions of these features and write STALL_DIAGNOSTIC.md.
"""
from __future__ import annotations

import os
import sys
import time
import json
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.box import Box
from interval_bnb.bnb import _to_fraction
from interval_bnb.bound_eval import (
    _adjacency_matrix,
    batch_bounds_full,
    eval_all_window_lbs_from_cached,
    gap_weighted_split_axis,
    window_tensor,
    bound_mccormick_joint_face_lp,
)
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


@dataclass
class StuckRecord:
    centroid: np.ndarray  # (d,)
    lo: np.ndarray
    hi: np.ndarray
    max_width: float
    min_width: float
    depth: int
    lb_fast: float
    win_ell: int
    win_s: int
    win_idx: int
    top3_lb: List[Tuple[int, int, float]]  # (ell, s_lo, lb)
    centroid_top_tvs: List[Tuple[int, int, float]]
    centroid_argmax_tv_window: Tuple[int, int]
    n_zeros_lo: int    # # coords with lo[i] near 0
    n_pinned_zero: int # # coords with hi[i] near 0 (forced 0)
    smallest_coord: float


def harvest_stuck_boxes(
    d: int,
    target_c: str,
    *,
    harvest_width: float = 1e-4,
    harvest_depth: int = 50,
    near_band: float = 0.05,
    max_nodes: int = 2_000_000,
    time_budget_s: float = 60.0,
    max_records: int = 500,
    verbose: bool = True,
):
    """Run BnB but collect boxes (rather than terminate on first failure)."""
    sym_cuts = half_simplex_cuts(d)
    initial = Box.initial(d, sym_cuts)

    target_q = _to_fraction(target_c)
    target_f = float(target_q)
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)

    stack: List[Tuple[Box, int, Optional[tuple], int, Optional[str]]] = []
    stack.append((initial, 0, None, -1, None))

    stuck: List[StuckRecord] = []
    near: List[StuckRecord] = []

    n_processed = 0
    n_certified = 0
    n_split = 0
    max_depth = 0
    max_q = 0
    t0 = time.time()
    last_log = t0

    def make_record(B, depth, lb_fast, w_idx, lb_all=None) -> StuckRecord:
        center = 0.5 * (B.lo + B.hi)
        s = float(center.sum())
        centroid = center / s if s > 0 else center
        widths = B.hi - B.lo
        # Compute TV at centroid for every window.
        tv_per_w = np.empty(len(windows))
        for wi, w in enumerate(windows):
            A = _adjacency_matrix(w, d)
            tv_per_w[wi] = w.scale * float(centroid @ A @ centroid)
        top_tv_idx = np.argsort(-tv_per_w)[:5]
        centroid_top = [(int(windows[i].ell), int(windows[i].s_lo),
                         float(tv_per_w[i])) for i in top_tv_idx]
        argmax_w = top_tv_idx[0]
        centroid_argmax = (int(windows[argmax_w].ell), int(windows[argmax_w].s_lo))
        # Top 3 windows by LB across the BOX (lb_fast ranking).
        if lb_all is not None:
            top3_idx = np.argsort(-lb_all)[:3]
            top3_lb = [(int(windows[i].ell), int(windows[i].s_lo),
                        float(lb_all[i])) for i in top3_idx]
        else:
            top3_lb = []
        win = windows[w_idx]
        n_zeros_lo = int(np.sum(B.lo < 1e-10))
        n_pinned_zero = int(np.sum(B.hi < 1e-10))
        smallest = float(centroid.min())
        return StuckRecord(
            centroid=centroid,
            lo=B.lo.copy(), hi=B.hi.copy(),
            max_width=float(widths.max()),
            min_width=float(widths.min()),
            depth=depth,
            lb_fast=float(lb_fast),
            win_ell=int(win.ell), win_s=int(win.s_lo), win_idx=int(w_idx),
            top3_lb=top3_lb,
            centroid_top_tvs=centroid_top,
            centroid_argmax_tv_window=centroid_argmax,
            n_zeros_lo=n_zeros_lo,
            n_pinned_zero=n_pinned_zero,
            smallest_coord=smallest,
        )

    while stack:
        if n_processed >= max_nodes:
            if verbose:
                print(f"[harvest] max_nodes={max_nodes} reached")
            break
        if time.time() - t0 > time_budget_s:
            if verbose:
                print(f"[harvest] time_budget {time_budget_s}s exceeded")
            break

        B, depth, parent_cache, changed_k, which_end = stack.pop()
        n_processed += 1
        if depth > max_depth:
            max_depth = depth
        if len(stack) > max_q:
            max_q = len(stack)

        if not B.intersects_simplex():
            continue

        # Always full recompute -- simpler than rank-1, and we need lb_all anyway.
        lb_fast, w_idx, which, _mu, my_cache = batch_bounds_full(
            B.lo, B.hi, A_tensor, scales, target_f,
        )

        # If certified easy, move on (don't record).
        if lb_fast >= target_f:
            n_certified += 1
            continue

        # NEAR-CERTIFIED: harvest if box is moderate width and gap small.
        gap = target_f - lb_fast
        max_w = B.max_width()
        is_stuck = (max_w < harvest_width) or (depth >= harvest_depth)
        if is_stuck:
            # STUCK at fine width or deep depth.
            if len(stuck) < max_records:
                Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum, _, _ = my_cache
                lb_all = eval_all_window_lbs_from_cached(
                    B.lo, B.hi, A_tensor, scales,
                    Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum,
                )
                stuck.append(make_record(B, depth, lb_fast, w_idx, lb_all))
            # Stop subdividing this box; it's a stall sample.
            continue
        elif gap < near_band and depth >= harvest_depth - 10:
            if len(near) < max_records:
                Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum, _, _ = my_cache
                lb_all = eval_all_window_lbs_from_cached(
                    B.lo, B.hi, A_tensor, scales,
                    Alo, Ahi, loAlo, hiAhi, lo_sum, hi_sum,
                )
                near.append(make_record(B, depth, lb_fast, w_idx, lb_all))
            # Keep subdividing.

        # Split.
        if depth >= 4 and w_idx >= 0:
            axis = gap_weighted_split_axis(B.lo, B.hi, windows[w_idx], d)
        else:
            axis = B.widest_axis()
        left, right = B.split(axis)
        n_split += 1
        if right.intersects_simplex():
            stack.append((right, depth + 1, my_cache, axis, "lo"))
        if left.intersects_simplex():
            stack.append((left, depth + 1, my_cache, axis, "hi"))

        if verbose and n_processed % 20000 == 0 and time.time() - last_log > 1.0:
            t = time.time() - t0
            rate = n_processed / max(t, 1e-9)
            print(f"[harvest] t={t:5.1f}s nodes={n_processed} stack={len(stack)} "
                  f"stuck={len(stuck)} near={len(near)} cert={n_certified} "
                  f"depth={max_depth} rate={rate:.0f}/s")
            last_log = time.time()

    if verbose:
        print(f"\n[harvest] DONE: {n_processed} nodes, {n_certified} certified, "
              f"{len(stuck)} stuck, {len(near)} near. depth={max_depth}.")
    return stuck, near, {
        "n_processed": n_processed,
        "n_certified": n_certified,
        "max_depth": max_depth,
        "wall_time_s": time.time() - t0,
    }


def analyze(stuck: List[StuckRecord], near: List[StuckRecord], d: int):
    """Aggregate structural properties of stuck boxes."""
    target_f = 1.208

    print("\n" + "=" * 70)
    print(f"AGGREGATE ANALYSIS: {len(stuck)} STUCK + {len(near)} NEAR boxes")
    print("=" * 70)

    if not stuck and not near:
        return {}

    boxes = stuck + near
    n = len(boxes)

    # Centroids matrix
    C = np.array([r.centroid for r in boxes])     # (n, d)
    LB = np.array([r.lb_fast for r in boxes])
    GAPS = target_f - LB
    DEPTHS = np.array([r.depth for r in boxes])
    WIDTHS = np.array([r.max_width for r in boxes])
    SMALLEST = np.array([r.smallest_coord for r in boxes])
    N_ZEROS = np.array([r.n_zeros_lo for r in boxes])

    # 1. Where are they geometrically?
    # mean centroid
    print("\n[1] CENTROID DISTRIBUTION")
    mean_c = C.mean(axis=0)
    std_c = C.std(axis=0)
    print(f"  Mean centroid (per coord):   {np.round(mean_c, 4).tolist()}")
    print(f"  Stddev centroid (per coord): {np.round(std_c, 4).tolist()}")
    print(f"  Mean centroid sum: {mean_c.sum():.4f}")

    # 2. Distance to mu_star illustrative
    mu_star = np.array([0.176, 0.049, 0.055, 0.005, 0.001, 0.001, 0.05, 0.05, 0.05, 0.165])
    mu_star = mu_star / mu_star.sum()
    dists_mustar = np.linalg.norm(C - mu_star, axis=1)
    print(f"\n[2] DIST TO mu_star (illustrative):")
    print(f"  min={dists_mustar.min():.4f}  median={np.median(dists_mustar):.4f}  "
          f"max={dists_mustar.max():.4f}  mean={dists_mustar.mean():.4f}")

    # 3. Distance to nearest simplex corner e_i
    # ||c - e_i||^2 = sum c_j^2 + 1 - 2 c_i. Minimised when c_i largest.
    sumsq = (C ** 2).sum(axis=1)
    max_c = C.max(axis=1)
    dists_corner = np.sqrt(sumsq + 1.0 - 2.0 * max_c)
    print(f"\n[3] DIST TO NEAREST CORNER e_i:")
    print(f"  min={dists_corner.min():.4f}  median={np.median(dists_corner):.4f}  "
          f"max={dists_corner.max():.4f}")
    # which i is the argmax most often?
    argmax_i = np.argmax(C, axis=1)
    cnt_argmax = Counter(argmax_i.tolist())
    print(f"  argmax-coord histogram: {dict(sorted(cnt_argmax.items()))}")

    # 4. Boundary structure: how many coords near 0?
    print(f"\n[4] BOUNDARY (mu_i ~ 0):")
    eps_near = 0.01
    n_near_zero = (C < eps_near).sum(axis=1)
    cnt = Counter(n_near_zero.tolist())
    print(f"  # coords with centroid < {eps_near}: {dict(sorted(cnt.items()))}")
    print(f"  smallest_coord stats: min={SMALLEST.min():.4f} median={np.median(SMALLEST):.4f}")

    # 5. Symmetry edges: # coords approximately equal to their pair?
    # Half-simplex cut enforces mu_i <= mu_{d-1-i}. Edge {mu_i = mu_{d-1-i}}.
    print(f"\n[5] SYMMETRY EDGE PROXIMITY |mu_i - mu_{{d-1-i}}|:")
    pair_diffs = []
    for r in boxes:
        c = r.centroid
        diffs = []
        for i in range(d // 2):
            j = d - 1 - i
            diffs.append(abs(c[i] - c[j]))
        pair_diffs.append(min(diffs))  # closest sym edge
    pair_diffs = np.array(pair_diffs)
    print(f"  min |mu_i - mu_{{d-1-i}}|: median={np.median(pair_diffs):.4f}  "
          f"min={pair_diffs.min():.4f}  max={pair_diffs.max():.4f}")
    eps_edge = 0.005
    on_edge = pair_diffs < eps_edge
    print(f"  fraction on symmetry edge (|.|<{eps_edge}): {on_edge.mean():.2%}")

    # 6. Width distribution
    print(f"\n[6] BOX SIZE STATS:")
    print(f"  max_width: min={WIDTHS.min():.2e} median={np.median(WIDTHS):.2e} "
          f"max={WIDTHS.max():.2e}")
    print(f"  depth: min={DEPTHS.min()} median={np.median(DEPTHS):.0f} "
          f"max={DEPTHS.max()}")

    # 7. LB / gap distribution
    print(f"\n[7] LB / GAP TO TARGET={target_f}:")
    print(f"  lb_fast: min={LB.min():.5f} median={np.median(LB):.5f} max={LB.max():.5f}")
    print(f"  gap = target - lb_fast: min={GAPS.min():.5f} median={np.median(GAPS):.5f}")

    # 8. Window structure: which (ell, s_lo) win?
    print(f"\n[8] WINNING WINDOWS (best LB at the box):")
    win_cnt = Counter([(r.win_ell, r.win_s) for r in boxes])
    for (ell, s), c in sorted(win_cnt.items(), key=lambda x: -x[1])[:10]:
        print(f"  W(ell={ell:>2d}, s={s:>2d}): {c} boxes")
    ells = np.array([r.win_ell for r in boxes])
    print(f"  ell distribution: min={ells.min()} median={int(np.median(ells))} "
          f"max={ells.max()}")
    s_lo = np.array([r.win_s for r in boxes])
    print(f"  s_lo distribution: min={s_lo.min()} median={int(np.median(s_lo))} "
          f"max={s_lo.max()}")

    # 9. What is max_W TV at the centroid?
    centroid_max_tvs = np.array([r.centroid_top_tvs[0][2] for r in boxes])
    print(f"\n[9] max_W TV(centroid) (the TRUE objective at centroid):")
    print(f"  min={centroid_max_tvs.min():.5f}  median={np.median(centroid_max_tvs):.5f} "
          f"max={centroid_max_tvs.max():.5f}")
    print(f"  fraction of centroids with max_W TV >= target ({target_f}): "
          f"{(centroid_max_tvs >= target_f).mean():.2%}")
    # which window achieves max TV at centroid
    centroid_argmax_w = Counter([r.centroid_argmax_tv_window for r in boxes])
    print(f"  argmax-TV window at centroid (top 5):")
    for (ell, s), c in sorted(centroid_argmax_w.items(), key=lambda x: -x[1])[:5]:
        print(f"    W(ell={ell:>2d}, s={s:>2d}): {c}")

    # 10. Cluster centroids: project to PCA, count clusters
    if n >= 5:
        # Center
        Cm = C - C.mean(axis=0, keepdims=True)
        cov = Cm.T @ Cm / n
        evals, evecs = np.linalg.eigh(cov)
        evals = evals[::-1]; evecs = evecs[:, ::-1]
        total = evals.sum()
        if total > 1e-12:
            print(f"\n[10] PCA OF CENTROIDS:")
            cum = 0.0
            print(f"  variance explained:")
            for k in range(min(5, d)):
                cum += evals[k]
                print(f"    PC{k+1}: {evals[k]:.4e}  cum={cum/total:.2%}")

    # Build summary dict for JSON
    summary = {
        "n_stuck": len(stuck),
        "n_near": len(near),
        "centroid_mean": mean_c.tolist(),
        "centroid_std": std_c.tolist(),
        "dist_mustar_median": float(np.median(dists_mustar)),
        "dist_corner_median": float(np.median(dists_corner)),
        "argmax_coord_histogram": {int(k): int(v) for k, v in cnt_argmax.items()},
        "n_near_zero_coord_histogram": {int(k): int(v) for k, v in cnt.items()},
        "fraction_on_sym_edge": float(on_edge.mean()),
        "max_width_median": float(np.median(WIDTHS)),
        "depth_median": float(np.median(DEPTHS)),
        "lb_fast_median": float(np.median(LB)),
        "gap_median": float(np.median(GAPS)),
        "winning_windows_top10": [
            {"ell": int(e), "s_lo": int(s), "count": int(c)}
            for (e, s), c in sorted(win_cnt.items(), key=lambda x: -x[1])[:10]
        ],
        "centroid_max_tv_median": float(np.median(centroid_max_tvs)),
        "centroid_max_tv_above_target_fraction": float((centroid_max_tvs >= target_f).mean()),
        "ell_min": int(ells.min()), "ell_median": int(np.median(ells)),
        "ell_max": int(ells.max()),
    }
    return summary


def write_report(stuck, near, stats, summary, d, target_f):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "STALL_DIAGNOSTIC.md")
    lines = []
    lines.append(f"# Interval BnB Stall Diagnostic: d={d}, target={target_f}\n")
    lines.append(f"## BnB Run Stats\n")
    lines.append(f"- nodes processed: {stats['n_processed']}")
    lines.append(f"- certified leaves: {stats['n_certified']}")
    lines.append(f"- max depth: {stats['max_depth']}")
    lines.append(f"- wall time: {stats['wall_time_s']:.1f}s")
    lines.append(f"- harvest threshold: width < 1e-4 (STUCK), in-band (NEAR)")
    lines.append(f"- STUCK boxes harvested: {len(stuck)}")
    lines.append(f"- NEAR boxes harvested: {len(near)}\n")

    if not summary:
        lines.append("No boxes harvested -- BnB succeeded or returned no failures.\n")
    else:
        lines.append("## Geometric Location of Unprovable Boxes\n")
        lines.append(f"- Mean centroid (sums to ~1): `{[round(x, 4) for x in summary['centroid_mean']]}`")
        lines.append(f"- Centroid stddev per coord: `{[round(x, 4) for x in summary['centroid_std']]}`")
        lines.append(f"- Distance to illustrative mu_star: median {summary['dist_mustar_median']:.4f}")
        lines.append(f"- Distance to nearest corner e_i: median {summary['dist_corner_median']:.4f}")
        lines.append(f"- argmax-coord histogram (which coordinate dominates the centroid):")
        lines.append("```")
        for k in sorted(summary['argmax_coord_histogram'].keys()):
            v = summary['argmax_coord_histogram'][k]
            lines.append(f"  coord {k}: {v}")
        lines.append("```")
        lines.append(f"- # near-zero (<0.01) coords histogram (boundary count):")
        lines.append("```")
        for k in sorted(summary['n_near_zero_coord_histogram'].keys()):
            v = summary['n_near_zero_coord_histogram'][k]
            lines.append(f"  {k} coords near 0: {v}")
        lines.append("```")
        lines.append(f"- Fraction of centroids on symmetry edge (|μ_i - μ_{{d-1-i}}|<0.005): "
                     f"{summary['fraction_on_sym_edge']:.2%}\n")

        lines.append(f"## Width / Depth\n")
        lines.append(f"- max_width median: {summary['max_width_median']:.2e}")
        lines.append(f"- depth median: {summary['depth_median']:.0f}\n")

        lines.append(f"## Gap to Target\n")
        lines.append(f"- lb_fast median: {summary['lb_fast_median']:.5f}")
        lines.append(f"- gap = (target - lb_fast) median: {summary['gap_median']:.5f}")
        lines.append(f"- max_W TV(centroid) median: {summary['centroid_max_tv_median']:.5f}")
        lines.append(f"- fraction of centroids with TRUE objective >= target: "
                     f"{summary['centroid_max_tv_above_target_fraction']:.2%}\n")

        lines.append(f"## Winning Windows\n")
        lines.append(f"- ell range: [{summary['ell_min']}, {summary['ell_max']}], "
                     f"median {summary['ell_median']}")
        lines.append(f"- top-10 winning windows:")
        lines.append("```")
        for w in summary['winning_windows_top10']:
            lines.append(f"  W(ell={w['ell']:>2d}, s={w['s_lo']:>2d}): {w['count']} boxes")
        lines.append("```\n")

        # First 10 sample boxes for manual inspection
        lines.append(f"## Sample Stuck Boxes (first 10)\n")
        for i, r in enumerate((stuck + near)[:10]):
            lines.append(f"### Box {i+1}: depth={r.depth} max_w={r.max_width:.2e}")
            lines.append(f"- centroid: `{[round(x, 4) for x in r.centroid]}`")
            lines.append(f"- lb_fast={r.lb_fast:.5f} (gap to target={target_f - r.lb_fast:.5f})")
            lines.append(f"- winning W: ell={r.win_ell}, s_lo={r.win_s}")
            lines.append(f"- max_W TV at centroid: ell={r.centroid_argmax_tv_window[0]}, "
                         f"s={r.centroid_argmax_tv_window[1]}, value={r.centroid_top_tvs[0][2]:.5f}")
            lines.append("")

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {p}")
    return p


def main():
    d = 10
    target_c = "1.208"
    target_f = float(Fraction(target_c))
    print(f"=== Interval BnB stall diagnostic at d={d}, target={target_c} ===\n")

    stuck, near, stats = harvest_stuck_boxes(
        d=d, target_c=target_c,
        harvest_width=1e-4,
        harvest_depth=55,
        near_band=0.05,
        max_nodes=6_000_000,
        time_budget_s=180.0,
        max_records=600,
        verbose=True,
    )
    summary = analyze(stuck, near, d)
    write_report(stuck, near, stats, summary, d, target_f)
    # Also save raw JSON.
    raw = {
        "stats": stats,
        "summary": summary,
        "stuck": [
            {
                "centroid": r.centroid.tolist(),
                "lo": r.lo.tolist(),
                "hi": r.hi.tolist(),
                "max_width": r.max_width, "min_width": r.min_width,
                "depth": r.depth, "lb_fast": r.lb_fast,
                "win_ell": r.win_ell, "win_s": r.win_s,
                "top3_lb": r.top3_lb,
                "centroid_top_tvs": r.centroid_top_tvs,
                "centroid_argmax_tv_window": r.centroid_argmax_tv_window,
                "n_zeros_lo": r.n_zeros_lo, "n_pinned_zero": r.n_pinned_zero,
                "smallest_coord": r.smallest_coord,
            }
            for r in stuck[:200]
        ],
    }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "stall_diag_d10_t1208.json")
    with open(p, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
