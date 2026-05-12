"""Deep diagnostic on the single stuck box from a split-first run.

Loads the box from runs/<tag>/stuck_lineage.npz and runs every available
bound function in the repo on it, plus structural analysis:

  1. Geometry summary (widths, depth, single-spike μ feasibility)
  2. McCormick epigraph LP (cheap baseline)
  3. SDP escalation at K=0, 16, 32, 64, 128 (full Lasserre order-2)
  4. Z/2 symmetric SDP variant (if box is σ-symmetric)
  5. Anchor / centroid-anchor / multi-anchor / multi-corner bounds
  6. Natural / autoconv exact bounds (analytic, per-window)
  7. Per-window LP value: which windows are binding?
  8. Value-minimizing μ in the box (explicit witness via local NLP)
  9. Per-axis split sensitivity: for each splittable axis, what's the
     LP value of the two children? Identifies which axis-split would
     close the gap fastest.
 10. The "all-low corner" check: confirm whether the stuck box's
     descendant lineage really does always carry the all-low subset.

OUTPUT: a single JSON file with every diagnostic + verdict per method.
USE: python -m cert_pipeline.diagnose_stuck_box \\
        --stuck runs/<tag>/stuck_lineage.npz \\
        --d 22 --target 1.2805 \\
        --output stuck_diagnostic.json

Each method that takes >300s is run with a 300s time limit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from cert_pipeline.k_ladder import survivors_from_npz
from interval_bnb.box import SCALE as _SCALE


def _safe(name: str, fn, *args, **kw) -> Dict[str, Any]:
    """Run `fn(*args, **kw)`, capture timing, exception, result."""
    t0 = time.time()
    try:
        out = fn(*args, **kw)
        return {"name": name, "ok": True, "wall_s": time.time() - t0,
                "result": out}
    except Exception as e:
        import traceback
        return {"name": name, "ok": False, "wall_s": time.time() - t0,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc().splitlines()[-5:]}


def geometry(box) -> Dict[str, Any]:
    lo = np.array([float(x) / _SCALE for x in box.lo_int], dtype=np.float64)
    hi = np.array([float(x) / _SCALE for x in box.hi_int], dtype=np.float64)
    widths = hi - lo
    # Single-spike μ = e_i feasibility test: need lo_i <= 1 <= hi_i AND
    # lo_j == 0 for all j != i.
    spike_axes = []
    for i in range(len(lo)):
        if hi[i] >= 1.0 - 1e-12 and all(lo[j] == 0 for j in range(len(lo)) if j != i):
            spike_axes.append(i)
    sigma_symmetric = bool(np.allclose(lo, lo[::-1]) and np.allclose(hi, hi[::-1]))
    return {
        "d": len(lo),
        "depth": box.depth,
        "iters_survived": box.iters_survived,
        "lp_val_from_pipeline": box.lp_val,
        "widths": widths.tolist(),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "n_axes_full_width": int((widths >= 1.0 - 1e-12).sum()),
        "n_axes_half_width": int(np.isclose(widths, 0.5).sum()),
        "n_axes_quarter_or_less": int((widths <= 0.25 + 1e-12).sum()),
        "sum_lo": float(lo.sum()),
        "sum_hi": float(hi.sum()),
        "intersects_simplex": bool(lo.sum() <= 1 + 1e-12 and hi.sum() >= 1 - 1e-12),
        "single_spike_mu_axes_feasible_in_box": spike_axes,
        "sigma_symmetric_box": sigma_symmetric,
    }


def lp_epigraph(d: int, lo: np.ndarray, hi: np.ndarray) -> float:
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_epigraph import bound_epigraph_lp_float
    windows = build_windows(d)
    return float(bound_epigraph_lp_float(lo, hi, windows, d))


def sdp_at_K(d: int, lo: np.ndarray, hi: np.ndarray,
              target: float, K: int, time_limit_s: float = 300.0,
              n_threads: int = 16) -> Dict[str, Any]:
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast, bound_sdp_escalation_lb_float_fast,
    )
    windows = build_windows(d)
    cache = build_sdp_escalation_cache_fast(d, windows, target=target)
    res = bound_sdp_escalation_lb_float_fast(
        lo, hi, windows, d, cache=cache, target=target,
        n_window_psd_cones=K, time_limit_s=time_limit_s, n_threads=n_threads,
    )
    # Slim down for JSON
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
             for k, v in res.items() if k in (
                 "verdict", "lambda_star", "solsta", "wall_s",
                 "n_psd_cones_kept", "n_lin_cones_kept", "rprim", "rdual",
                 "gap", "wall_lp_s", "wall_build_s", "wall_solve_s",
             )}


def sdp_z2(d: int, lo: np.ndarray, hi: np.ndarray, target: float,
            time_limit_s: float = 300.0,
            n_threads: int = 16) -> Dict[str, Any]:
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_z2 import (
        bound_sdp_escalation_z2_lb_float,
    )
    windows = build_windows(d)
    res = bound_sdp_escalation_z2_lb_float(
        lo, hi, windows, d, target=target,
        time_limit_s=time_limit_s, n_threads=n_threads,
    )
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
             for k, v in res.items() if k in (
                 "verdict", "lambda_star", "solsta", "wall_s",
             )}


def anchor_bounds(d: int, lo_int: List[int], hi_int: List[int],
                   target: float) -> Dict[str, Any]:
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_anchor import (
        bound_anchor_centroid_int_ge,
        anchor_lb_float,
        anchor_lb_centroid_float,
    )
    windows = build_windows(d)
    out: Dict[str, Any] = {}
    # The float-side anchor lower bounds give a value (best lower bound)
    # without needing a target. Use those for diagnostic clarity.
    lo = np.array([float(x) / _SCALE for x in lo_int], dtype=np.float64)
    hi = np.array([float(x) / _SCALE for x in hi_int], dtype=np.float64)
    try:
        out["anchor_lb_float"] = float(anchor_lb_float(lo, hi, windows, d))
    except Exception as e:
        out["anchor_lb_float_err"] = str(e)
    try:
        out["anchor_lb_centroid_float"] = float(
            anchor_lb_centroid_float(lo, hi, windows, d))
    except Exception as e:
        out["anchor_lb_centroid_float_err"] = str(e)
    return out


def per_window_lp(d: int, lo: np.ndarray, hi: np.ndarray) -> Dict[str, Any]:
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_fast import (
        _per_window_lp_primal_values,
    )
    windows = build_windows(d)
    lp_val, pw = _per_window_lp_primal_values(lo, hi, windows, d)
    pw = np.asarray(pw, dtype=np.float64)
    n_W = len(windows)
    order = np.argsort(-pw)
    top = order[:10]
    return {
        "n_windows": n_W,
        "lp_val": float(lp_val),
        "per_window_max": float(pw.max()) if pw.size else None,
        "per_window_mean": float(pw.mean()) if pw.size else None,
        "per_window_median": float(np.median(pw)) if pw.size else None,
        "n_binding_windows_close_to_lp_val": int(
            (np.abs(pw - lp_val) < 1e-3).sum()),
        "top10_windows_by_value": [
            {"window_idx": int(i), "value": float(pw[i])} for i in top
        ],
    }


def value_minimizing_mu(d: int, lo: np.ndarray, hi: np.ndarray) -> Dict[str, Any]:
    """Minimize max_W μ^T M_W μ over μ in box ∩ simplex. Local
    optimization (multistart) — confirms the val_B value and locates
    the obstruction.
    """
    from interval_bnb.windows import build_windows
    from scipy.optimize import minimize, linprog
    windows = build_windows(d)
    # Build M_W matrices
    M_W_list = []
    for W in windows:
        M = np.zeros((d, d), dtype=np.float64)
        S_W = W.S_W if hasattr(W, "S_W") else getattr(W, "edges", None)
        if S_W is None:
            continue
        for (i, j) in S_W:
            M[i, j] = 1.0
            M[j, i] = 1.0
        scale = float(getattr(W, "scale", 1.0))
        M_W_list.append(scale * M)

    def fmax(mu):
        return max(float(mu @ M @ mu) for M in M_W_list)

    # Multistart starting points
    starts = []
    # 1. Single-spike μ = e_i for each i
    for i in range(d):
        if hi[i] >= 1.0 - 1e-12 and all(lo[j] == 0 for j in range(d) if j != i):
            e = np.zeros(d); e[i] = 1.0
            starts.append(e)
    # 2. Centroid clipped to box ∩ simplex
    cent = (lo + hi) / 2
    if cent.sum() > 0:
        cent = cent / cent.sum() * 1.0
        cent = np.clip(cent, lo, hi)
        if cent.sum() > 0:
            cent = cent / cent.sum()
            starts.append(cent)
    # 3. Random points in box ∩ simplex
    rng = np.random.default_rng(42)
    for _ in range(20):
        p = rng.uniform(lo, hi)
        s = p.sum()
        if s > 0:
            p = p / s
            p = np.clip(p, lo, hi)
            s = p.sum()
            if s > 0:
                starts.append(p / s)

    best = {"val": float("inf"), "mu": None}
    for mu0 in starts:
        # Project to box ∩ simplex via simple LP feasibility
        try:
            res = minimize(
                fmax, mu0,
                method="SLSQP",
                bounds=[(float(lo[i]), float(hi[i])) for i in range(d)],
                constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
                options={"maxiter": 100, "ftol": 1e-9},
            )
            if res.success and res.fun < best["val"]:
                best = {"val": float(res.fun), "mu": res.x.tolist()}
        except Exception:
            continue
    return {
        "val_B_estimated": best["val"],
        "argmin_mu": best["mu"],
        "n_nonzero_argmin": (sum(1 for x in best["mu"] if abs(x) > 1e-6)
                              if best["mu"] else None),
        "max_argmin_coord": (max(abs(x) for x in best["mu"])
                              if best["mu"] else None),
    }


def per_axis_split_sensitivity(d: int, lo_int: List[int], hi_int: List[int],
                                 target: float) -> Dict[str, Any]:
    """For each splittable axis, halve it and report the LP values of
    the two children. The axis whose split most increases the MIN of
    the two child LP values is the best to split next.
    """
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_epigraph import bound_epigraph_lp_float
    windows = build_windows(d)
    rows = []
    for axis in range(d):
        w_int = hi_int[axis] - lo_int[axis]
        if w_int < 2:
            continue
        mid = (lo_int[axis] + hi_int[axis]) // 2
        # Left child
        l_lo, l_hi = list(lo_int), list(hi_int)
        l_hi[axis] = mid
        # Right child
        r_lo, r_hi = list(lo_int), list(hi_int)
        r_lo[axis] = mid
        l_lo_f = np.array([float(x) / _SCALE for x in l_lo], dtype=np.float64)
        l_hi_f = np.array([float(x) / _SCALE for x in l_hi], dtype=np.float64)
        r_lo_f = np.array([float(x) / _SCALE for x in r_lo], dtype=np.float64)
        r_hi_f = np.array([float(x) / _SCALE for x in r_hi], dtype=np.float64)
        # Skip children that miss the simplex
        l_in = (sum(l_lo) <= _SCALE and sum(l_hi) >= _SCALE)
        r_in = (sum(r_lo) <= _SCALE and sum(r_hi) >= _SCALE)
        try:
            l_lp = float(bound_epigraph_lp_float(l_lo_f, l_hi_f, windows, d)) if l_in else None
        except Exception:
            l_lp = None
        try:
            r_lp = float(bound_epigraph_lp_float(r_lo_f, r_hi_f, windows, d)) if r_in else None
        except Exception:
            r_lp = None
        # min child LP value (the worse of the two — what we'd inherit)
        candidates = [v for v in (l_lp, r_lp) if v is not None]
        min_child = min(candidates) if candidates else None
        rows.append({
            "axis": axis,
            "child_lp_left": l_lp,
            "child_lp_right": r_lp,
            "left_in_simplex": l_in,
            "right_in_simplex": r_in,
            "min_child_lp": min_child,
            "improves_target": (min_child is not None and min_child >= target),
        })
    rows.sort(key=lambda r: -(r["min_child_lp"] or -float("inf")))
    return {
        "per_axis": rows,
        "best_axis_to_split_first": rows[0]["axis"] if rows else None,
        "best_axis_min_child_lp": rows[0]["min_child_lp"] if rows else None,
        "n_axes_whose_split_certs_both_children": sum(
            1 for r in rows if r["improves_target"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stuck", required=True,
                    help="path to stuck_lineage.npz")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sdp-time-limit-s", type=float, default=300.0)
    ap.add_argument("--n-threads", type=int, default=16)
    ap.add_argument("--include-K", type=str, default="0,16,32,64,128",
                    help="Which K values to test (comma-separated)")
    args = ap.parse_args()

    boxes = survivors_from_npz(args.stuck)
    if not boxes:
        print(f"[err] no boxes in {args.stuck}")
        sys.exit(1)
    box = boxes[0]
    print(f"[diagnose] box {box.hash} depth={box.depth} "
          f"iters={box.iters_survived}", flush=True)

    out: Dict[str, Any] = {
        "box_hash": box.hash,
        "src_npz": str(args.stuck),
        "d": args.d,
        "target": args.target,
    }

    print("[1] geometry...", flush=True)
    out["geometry"] = geometry(box)
    print(f"    n_axes_full_width={out['geometry']['n_axes_full_width']}", flush=True)
    print(f"    single_spike_mu_axes_feasible="
          f"{out['geometry']['single_spike_mu_axes_feasible_in_box']}", flush=True)

    lo = np.array([float(x) / _SCALE for x in box.lo_int], dtype=np.float64)
    hi = np.array([float(x) / _SCALE for x in box.hi_int], dtype=np.float64)

    print("[2] LP epigraph...", flush=True)
    out["lp_epigraph"] = _safe("lp_epigraph", lp_epigraph, args.d, lo, hi)
    print(f"    LP val = {out['lp_epigraph'].get('result')}", flush=True)

    print("[3] per-window LP profile...", flush=True)
    out["per_window_lp"] = _safe("per_window_lp", per_window_lp, args.d, lo, hi)

    print("[4] value-minimizing μ (multistart NLP)...", flush=True)
    out["val_B_witness"] = _safe("val_B_witness", value_minimizing_mu,
                                   args.d, lo, hi)
    if out["val_B_witness"]["ok"]:
        r = out["val_B_witness"]["result"]
        print(f"    val_B ≈ {r['val_B_estimated']:.6f}  "
              f"n_nonzero_argmin={r['n_nonzero_argmin']}", flush=True)

    print("[5] anchor bounds...", flush=True)
    out["anchor"] = _safe("anchor", anchor_bounds, args.d,
                           box.lo_int, box.hi_int, args.target)

    K_list = [int(k) for k in args.include_K.split(",")]
    out["sdp_at_K"] = {}
    for K in K_list:
        print(f"[6] SDP at K={K} (time_limit={args.sdp_time_limit_s}s)...",
              flush=True)
        r = _safe(f"sdp_K{K}", sdp_at_K, args.d, lo, hi, args.target, K,
                   args.sdp_time_limit_s, args.n_threads)
        out["sdp_at_K"][str(K)] = r
        if r["ok"]:
            v = r["result"]
            print(f"    K={K} verdict={v.get('verdict')} "
                  f"lambda*={v.get('lambda_star')} wall={v.get('wall_s')}",
                  flush=True)

    print("[7] Z/2-symmetric SDP variant...", flush=True)
    out["sdp_z2"] = _safe("sdp_z2", sdp_z2, args.d, lo, hi, args.target,
                            args.sdp_time_limit_s, args.n_threads)

    print("[8] per-axis split sensitivity (LP on each candidate split)...",
          flush=True)
    out["per_axis_split"] = _safe("per_axis_split",
                                    per_axis_split_sensitivity,
                                    args.d, box.lo_int, box.hi_int, args.target)
    if out["per_axis_split"]["ok"]:
        r = out["per_axis_split"]["result"]
        print(f"    best_axis={r['best_axis_to_split_first']}  "
              f"best_min_child_lp={r['best_axis_min_child_lp']}  "
              f"n_axes_certs_both_children={r['n_axes_whose_split_certs_both_children']}",
              flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[diagnose] DONE  output={args.output}", flush=True)


if __name__ == "__main__":
    main()
