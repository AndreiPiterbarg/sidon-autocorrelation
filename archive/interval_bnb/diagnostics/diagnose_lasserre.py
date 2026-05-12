"""Diagnostic: Lasserre L2 SDP LB (float) vs standard lb_fast on
d=16 t=1.25 stall-region boxes.

Uses interval_bnb.lasserre_cert.lasserre_box_lb_float which builds the
full L2 SDP over box ∩ simplex (box localizers L_k, U_k plus moment
matrix M_2 PSD plus simplex consistency).

Returns: does SDP cross target on stall boxes? How often, how much?
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
    window_tensor,
)
from interval_bnb.box import Box
from interval_bnb.lasserre_cert import lasserre_box_lb_float
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


def main():
    d = int(os.environ.get("DIAG_D", "16"))
    target_f = float(os.environ.get("DIAG_TARGET", "1.25"))
    time_budget = float(os.environ.get("DIAG_TIME", "600"))
    min_depth = int(os.environ.get("DIAG_MIN_DEPTH", "50"))
    max_samples = int(os.environ.get("DIAG_MAX_SAMPLES", "8"))
    order = int(os.environ.get("DIAG_ORDER", "2"))
    solver = os.environ.get("DIAG_SOLVER", "CLARABEL")

    sym = half_simplex_cuts(d)
    initial = Box.initial(d, sym)
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)

    stack = [(initial, 0, None, -1, None)]
    t0 = time.time()
    nodes = 0
    samples = []
    print(f"[diag] d={d} target={target_f} min_depth={min_depth} "
          f"samples={max_samples} order={order} solver={solver}")

    while stack:
        if time.time() - t0 > time_budget:
            print("[diag] time budget exhausted")
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
            t_sdp = time.time()
            lb_sdp = lasserre_box_lb_float(
                B.lo, B.hi, windows, d,
                order=order, solver=solver, verbose=False,
            )
            sdp_ms = (time.time() - t_sdp) * 1000
            crosses = lb_sdp >= target_f
            samples.append({
                "depth": depth,
                "lb_fast": lb_fast,
                "lb_sdp": lb_sdp,
                "crosses": crosses,
                "gap": lb_sdp - lb_fast,
                "max_width": B.max_width(),
                "sdp_ms": sdp_ms,
            })
            tag = "CROSSES" if crosses else "below"
            print(f"[d={depth:3d}] lb_fast={lb_fast:.5f} "
                  f"lb_sdp={lb_sdp:.5f} gap={lb_sdp - lb_fast:+.5f} "
                  f"[{tag}] sdp_ms={sdp_ms:.1f}")
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
    print(f"[diag] d={d} target={target_f} elapsed={elapsed:.1f}s  order={order}")
    print(f"[diag] nodes={nodes} samples={len(samples)}")
    if samples:
        crosses = sum(1 for s in samples if s["crosses"])
        lbs = sorted(s["lb_sdp"] for s in samples)
        gaps = sorted(s["gap"] for s in samples)
        mss = sorted(s["sdp_ms"] for s in samples)
        print(f"[diag] SDP crosses target = {crosses}/{len(samples)}")
        print(f"[diag] lb_sdp min={lbs[0]:.5f} median={lbs[len(lbs)//2]:.5f} "
              f"max={lbs[-1]:.5f} (target={target_f})")
        print(f"[diag] gap over lb_fast min={gaps[0]:+.5f} "
              f"median={gaps[len(gaps)//2]:+.5f} max={gaps[-1]:+.5f}")
        print(f"[diag] sdp_ms min={mss[0]:.0f} median={mss[len(mss)//2]:.0f} "
              f"max={mss[-1]:.0f}")


if __name__ == "__main__":
    main()
