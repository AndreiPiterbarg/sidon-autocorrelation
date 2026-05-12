"""Diagnostic: Shor (eigenvalue) lower bound vs standard lb_fast on
d=16 t=1.25 stall-region boxes.

For window W with adjacency matrix A_W = Σ_k λ_k v_k v_k^T (symmetric),
mu^T A_W mu = Σ_k λ_k (v_k^T mu)^2. Over box ∩ simplex with linear
functional ranges [a_k, b_k] = [min v_k^T mu, max v_k^T mu]:

  LB_Shor(W, B) = scale * [
      Σ_{λ_k ≥ 0} λ_k · L_k
    + Σ_{λ_k < 0} λ_k · U_k
  ]

  L_k = 0 if 0 ∈ [a_k, b_k] else min(a_k², b_k²)   (convex floor)
  U_k = max(a_k², b_k²)

Each a_k, b_k is one greedy LP on box ∩ simplex → O(d log d) each.
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
    _adjacency_matrix,
    batch_bounds_full,
    batch_bounds_rank1_hi,
    batch_bounds_rank1_lo,
    window_tensor,
)
from interval_bnb.box import Box
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


def _min_max_linear_box_simplex(g, lo, hi):
    """Return (min_val, max_val) of g^T mu over {sum mu = 1, lo ≤ mu ≤ hi}.

    Greedy: min sorts g ascending, fills from lo_sum up to 1; max sorts
    descending.
    """
    d = len(g)
    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    if lo_sum > 1.0 + 1e-12 or hi_sum < 1.0 - 1e-12:
        return float("nan"), float("nan")
    remaining = 1.0 - lo_sum
    # min
    order_min = np.argsort(g, kind="stable")
    val_min = float((g * lo).sum())
    rem = remaining
    for i in order_min:
        if rem <= 0:
            break
        cap = float(hi[i] - lo[i])
        add = cap if cap < rem else rem
        val_min += float(g[i]) * add
        rem -= add
    # max
    order_max = order_min[::-1]
    val_max = float((g * lo).sum())
    rem = remaining
    for i in order_max:
        if rem <= 0:
            break
        cap = float(hi[i] - lo[i])
        add = cap if cap < rem else rem
        val_max += float(g[i]) * add
        rem -= add
    return val_min, val_max


def shor_bound(lo, hi, w, d, A_cache):
    """Shor eigenvalue LB on mu^T M_W mu with M_W = scale * A_W."""
    A = A_cache
    eigvals, eigvecs = np.linalg.eigh(A)
    lb = 0.0
    for k in range(d):
        lam = float(eigvals[k])
        v = eigvecs[:, k]
        a, b = _min_max_linear_box_simplex(v, lo, hi)
        if not np.isfinite(a) or not np.isfinite(b):
            return float("-inf")
        a2 = a * a
        b2 = b * b
        U_k = max(a2, b2)
        if a <= 0.0 <= b:
            L_k = 0.0
        else:
            L_k = min(a2, b2)
        if lam >= 0:
            lb += lam * L_k
        else:
            lb += lam * U_k
    if lb < 0:
        lb = 0.0
    return w.scale * lb


def main():
    d = int(os.environ.get("DIAG_D", "16"))
    target_f = float(os.environ.get("DIAG_TARGET", "1.25"))
    time_budget = float(os.environ.get("DIAG_TIME", "120"))
    min_depth = int(os.environ.get("DIAG_MIN_DEPTH", "50"))
    max_samples = int(os.environ.get("DIAG_MAX_SAMPLES", "30"))
    eval_all_windows = os.environ.get("DIAG_ALL_WINDOWS", "0") == "1"

    sym = half_simplex_cuts(d)
    initial = Box.initial(d, sym)
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)
    # precompute eigendecomp for every window
    A_mats = [_adjacency_matrix(w, d) for w in windows]

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
            # Evaluate Shor on the float-path winning window.
            w = windows[w_idx]
            A = A_mats[w_idx]
            t_sh = time.time()
            lb_shor_top = shor_bound(B.lo, B.hi, w, d, A)
            ms_top = (time.time() - t_sh) * 1000

            # Optionally evaluate Shor over ALL windows and take max.
            best_shor = lb_shor_top
            best_w = w_idx
            ms_total = ms_top
            if eval_all_windows:
                for wi, ww in enumerate(windows):
                    if wi == w_idx:
                        continue
                    t2 = time.time()
                    lbs = shor_bound(B.lo, B.hi, ww, d, A_mats[wi])
                    ms_total += (time.time() - t2) * 1000
                    if lbs > best_shor:
                        best_shor = lbs
                        best_w = wi

            samples.append({
                "depth": depth,
                "lb_fast": lb_fast,
                "lb_shor_top": lb_shor_top,
                "lb_shor_best": best_shor,
                "best_w": best_w,
                "lb_shor_crosses": best_shor >= target_f,
                "lb_combined_crosses": max(lb_fast, best_shor) >= target_f,
                "gap_top": lb_shor_top - lb_fast,
                "gap_best": best_shor - lb_fast,
                "max_width": B.max_width(),
                "ms_total": ms_total,
            })
            tag = "CROSSES" if best_shor >= target_f else "below"
            print(f"[d={depth:3d}] lb_fast={lb_fast:.5f} "
                  f"shor_top={lb_shor_top:.5f} shor_best={best_shor:.5f} "
                  f"[{tag}] ms={ms_total:.1f}")
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
    print(f"[diag] d={d} target={target_f} elapsed={elapsed:.1f}s  all_windows={eval_all_windows}")
    print(f"[diag] nodes={nodes} samples={len(samples)}")
    if samples:
        shor_crosses = sum(1 for s in samples if s["lb_shor_crosses"])
        comb_crosses = sum(1 for s in samples if s["lb_combined_crosses"])
        lb_shors = sorted(s["lb_shor_best"] for s in samples)
        gaps_top = sorted(s["gap_top"] for s in samples)
        gaps_best = sorted(s["gap_best"] for s in samples)
        mss = sorted(s["ms_total"] for s in samples)
        print(f"[diag] Shor alone crosses target = {shor_crosses}/{len(samples)}")
        print(f"[diag] max(lb_fast, Shor) crosses = {comb_crosses}/{len(samples)}")
        print(f"[diag] lb_shor_best min={lb_shors[0]:.5f} median={lb_shors[len(lb_shors)//2]:.5f} "
              f"max={lb_shors[-1]:.5f} (target={target_f})")
        print(f"[diag] gap over lb_fast (top win) min={gaps_top[0]:+.5f} "
              f"median={gaps_top[len(gaps_top)//2]:+.5f} max={gaps_top[-1]:+.5f}")
        print(f"[diag] gap over lb_fast (best) min={gaps_best[0]:+.5f} "
              f"median={gaps_best[len(gaps_best)//2]:+.5f} max={gaps_best[-1]:+.5f}")
        print(f"[diag] ms per box min={mss[0]:.1f} median={mss[len(mss)//2]:.1f} "
              f"max={mss[-1]:.1f}")


if __name__ == "__main__":
    main()
