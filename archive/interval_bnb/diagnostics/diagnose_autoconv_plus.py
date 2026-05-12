"""Diagnostic: measure autoconv⁺ (RLT-LP) vs standard lb_fast on
d=16 t=1.25 stall-region boxes.

Samples boxes at depth ≥ MIN_DEPTH where lb_fast < target. For each,
runs autoconv⁺ LP on the top window AND the top-3 windows (by lb_fast)
and reports how often it crosses target.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from interval_bnb.bound_eval import (
    batch_bounds_full,
    batch_bounds_rank1_hi,
    batch_bounds_rank1_lo,
    bound_autoconv_plus_lp,
    window_tensor,
)
from interval_bnb.box import Box
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


def main():
    d = int(os.environ.get("DIAG_D", "16"))
    target_f = float(os.environ.get("DIAG_TARGET", "1.25"))
    time_budget = float(os.environ.get("DIAG_TIME", "180"))
    min_depth = int(os.environ.get("DIAG_MIN_DEPTH", "30"))
    max_samples = int(os.environ.get("DIAG_MAX_SAMPLES", "40"))
    topk = int(os.environ.get("DIAG_TOPK", "3"))

    sym = half_simplex_cuts(d)
    initial = Box.initial(d, sym)
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)

    stack = [(initial, 0, None, -1, None)]
    t0 = time.time()
    nodes = 0
    samples = []
    while stack:
        if time.time() - t0 > time_budget:
            break
        if len(samples) >= max_samples:
            break
        B, depth, parent_cache, changed_k, which_end = stack.pop()
        nodes += 1
        if not B.intersects_simplex():
            continue
        if parent_cache is None:
            lb_fast, w_idx, which, _, my_cache = batch_bounds_full(
                B.lo, B.hi, A_tensor, scales, target_f,
            )
        elif which_end == "lo":
            lb_fast, w_idx, which, _, my_cache = batch_bounds_rank1_lo(
                A_tensor, scales, parent_cache, B.lo, changed_k, target_f,
            )
        else:
            lb_fast, w_idx, which, _, my_cache = batch_bounds_rank1_hi(
                A_tensor, scales, parent_cache, B.hi, changed_k, target_f,
            )

        if depth >= min_depth and lb_fast < target_f and w_idx >= 0:
            # Evaluate autoconv+ on top-k windows by auto+mcc max.
            # Recompute per-window lb to rank windows.
            # Quick: use lb_fast as the representative; run autoconv+ on
            # just the top window and the top-3 windows picked by the
            # float-path selection.
            w = windows[w_idx]
            t_lp = time.time()
            lb_plus_top = bound_autoconv_plus_lp(B.lo, B.hi, w, d, w.scale)
            lp_ms_top = (time.time() - t_lp) * 1000

            # Try top-3 windows in terms of simple natural-interval
            # heuristic (since lb_fast picks one window, but others
            # might be closer if autoconv+ tightens differently).
            # Use autoconv raw to rank: 1 - (sum hi)^2 + hi^T A_w hi.
            sum_hi = float(B.hi.sum())
            rankings = []
            for wi, ww in enumerate(windows):
                hAh = float(B.hi @ (A_tensor[wi] @ B.hi))
                auto_raw = 1.0 - sum_hi * sum_hi + hAh
                rankings.append((ww.scale * max(auto_raw, 0.0), wi))
            rankings.sort(reverse=True)
            top_window_idxs = [rankings[k][1] for k in range(min(topk, len(rankings)))]
            best_plus = lb_plus_top
            best_w = w_idx
            lp_ms_total = lp_ms_top
            for wi in top_window_idxs:
                if wi == w_idx:
                    continue
                ww = windows[wi]
                t_lp2 = time.time()
                lb_plus_i = bound_autoconv_plus_lp(B.lo, B.hi, ww, d, ww.scale)
                lp_ms_total += (time.time() - t_lp2) * 1000
                if lb_plus_i > best_plus:
                    best_plus = lb_plus_i
                    best_w = wi

            samples.append({
                "depth": depth,
                "lb_fast": lb_fast,
                "lb_plus_top": lb_plus_top,
                "lb_plus_best": best_plus,
                "best_w": best_w,
                "lb_plus_crosses": best_plus >= target_f,
                "gap_top": lb_plus_top - lb_fast,
                "gap_best": best_plus - lb_fast,
                "max_width": B.max_width(),
                "lp_ms_total": lp_ms_total,
            })
            tag = "CROSSES" if best_plus >= target_f else "below"
            print(f"[d={depth:3d}] lb_fast={lb_fast:.5f} "
                  f"lb+={lb_plus_top:.5f} lb+_best={best_plus:.5f} "
                  f"[{tag}] lp_ms={lp_ms_total:.1f}")
            continue  # don't recurse

        if lb_fast >= target_f:
            continue
        if B.max_width() < 1e-12:
            continue
        axis = B.widest_axis()
        left, right = B.split(axis)
        if right.intersects_simplex():
            stack.append((right, depth + 1, my_cache, axis, "lo"))
        if left.intersects_simplex():
            stack.append((left, depth + 1, my_cache, axis, "hi"))

    elapsed = time.time() - t0
    print()
    print(f"[diag] d={d} target={target_f} elapsed={elapsed:.1f}s")
    print(f"[diag] nodes={nodes} samples={len(samples)}")
    if samples:
        crosses = sum(1 for s in samples if s["lb_plus_crosses"])
        gaps_top = sorted(s["gap_top"] for s in samples)
        gaps_best = sorted(s["gap_best"] for s in samples)
        lbplus = sorted(s["lb_plus_best"] for s in samples)
        lpms = sorted(s["lp_ms_total"] for s in samples)
        print(f"[diag] autoconv+ crosses target = {crosses}/{len(samples)}")
        print(f"[diag] lb+_best min={lbplus[0]:.5f} median={lbplus[len(lbplus)//2]:.5f} "
              f"max={lbplus[-1]:.5f} (target={target_f})")
        print(f"[diag] gap over lb_fast (top window) min={gaps_top[0]:+.5f} "
              f"median={gaps_top[len(gaps_top)//2]:+.5f} max={gaps_top[-1]:+.5f}")
        print(f"[diag] gap over lb_fast (best of top-{topk}) min={gaps_best[0]:+.5f} "
              f"median={gaps_best[len(gaps_best)//2]:+.5f} max={gaps_best[-1]:+.5f}")
        print(f"[diag] lp_ms per box (k={topk} wins) min={lpms[0]:.1f} "
              f"median={lpms[len(lpms)//2]:.1f} max={lpms[-1]:.1f}")


if __name__ == "__main__":
    main()
