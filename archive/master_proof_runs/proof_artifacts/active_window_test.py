"""Active-window restriction audit for d=10 stuck boxes.

Hypothesis: restricting the bound stack to the global active set A
(windows binding at mu*) -- or to a per-box top-k by centroid g_W(c_B)
-- gives a STRICTLY tighter LP value than batching ALL 190 windows.

This script:
  1. Loads stuck_d10_master_queue.npz  (lo, hi, depths over 748 stuck boxes)
     and samples 30 boxes uniformly with seed=42.
  2. Loads mu_star_d10.npz                       (mu* and val(10) ~ 1.2249).
  3. Builds windows = build_windows(10).
  4. Computes A = {W : g_W(mu*) >= val* - 1e-6}.
  5. For each sampled box B and each restriction r in
       {ALL, ACTIVE, TOP5}:
       a. Counts integer rigor-gate certs at target=Fraction(12,10),
          using bound_natural_int_ge / bound_mccormick_sw_int_ge / NE
          per-window then taking max.
       b. Solves bound_epigraph_int_ge_with_marginals (float epi-LP),
          records lp_val and (lp_val >= 1.2).
  6. Saves a json report and prints the comparative summary.

Restricting to a SUBSET of windows in the epi-LP gives
  min_mu max_{W in subset} g_W(mu) <= true val_B,
so subset-LP-cert >= target is a SOUND certificate (still a valid
lower bound on val_B, just not necessarily tighter than ALL).
"""
from __future__ import annotations

import json
import os
import sys
import time
from fractions import Fraction
from typing import List

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from interval_bnb.windows import build_windows
from interval_bnb.bound_eval import (
    bound_natural_int_ge,
    bound_mccormick_sw_int_ge,
    bound_mccormick_ne_int_ge,
    window_tensor,
)
from interval_bnb.bound_epigraph import bound_epigraph_int_ge_with_marginals
from interval_bnb.box import SCALE


D = 10
TARGET_NUM = 12
TARGET_DEN = 10
TARGET_F = TARGET_NUM / TARGET_DEN  # 1.2
N_SAMPLES = 30
SEED = 42
TOP_K = 5


def gW_value(mu: np.ndarray, A_tensor: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Return per-window value g_W(mu) = scale * mu^T A_W mu."""
    Amu = A_tensor @ mu                              # (W, d)
    quad = (Amu * mu[None, :]).sum(axis=1)           # (W,)
    return scales * quad


def main():
    print(f"[load] mu_star_d10.npz", flush=True)
    mus = np.load(os.path.join(REPO, "mu_star_d10.npz"))
    mu_star = np.asarray(mus["mu"], dtype=np.float64)
    val_star = float(mus["f"])
    print(f"        mu* sum={mu_star.sum():.10f}  val*={val_star:.10f}", flush=True)

    print(f"[load] stuck_d10_master_queue.npz", flush=True)
    q = np.load(os.path.join(REPO, "stuck_d10_master_queue.npz"))
    lo_all = np.asarray(q["lo"], dtype=np.float64)
    hi_all = np.asarray(q["hi"], dtype=np.float64)
    depths = np.asarray(q["depths"], dtype=np.int64)
    n_box = lo_all.shape[0]
    print(f"        {n_box} boxes, depth range {depths.min()}..{depths.max()}", flush=True)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(n_box, size=N_SAMPLES, replace=False)
    sample_idx.sort()
    print(f"[sample] {N_SAMPLES} indices: {list(sample_idx)}", flush=True)

    print(f"[windows] build_windows({D})", flush=True)
    windows = build_windows(D)
    W = len(windows)
    A_tensor, scales = window_tensor(windows, D)
    print(f"          {W} windows", flush=True)

    # --- Global active set A: windows binding at mu* ---
    g_star = gW_value(mu_star, A_tensor, scales)
    max_star = float(g_star.max())
    tol = 1e-6
    A_mask = g_star >= max_star - tol
    A_idx = np.where(A_mask)[0]
    A_windows = [windows[i] for i in A_idx]
    print(f"[active] |A| = {len(A_idx)} of {W}  "
          f"(threshold = max - 1e-6 = {max_star - tol:.10f})", flush=True)
    print(f"         g_max@mu* = {max_star:.10f}  vs  f = {val_star:.10f}",
          flush=True)

    # --- Restriction definitions ---
    # We compute TOP5 per-box from g_W(c_B); ALL and ACTIVE are static.
    restrictions = ["ALL", "ACTIVE", "TOP5"]

    # Per-box results
    per_box_records: List[dict] = []
    summary = {r: {"lp_vals": [], "lp_certs": 0,
                   "int_nat_certs": 0, "int_mcc_sw_certs": 0,
                   "int_mcc_ne_certs": 0, "int_any_certs": 0}
               for r in restrictions}

    target_q = Fraction(TARGET_NUM, TARGET_DEN)
    print(f"\n[run] target = {target_q} = {float(target_q):.6f}", flush=True)

    t0 = time.time()
    for n, bi in enumerate(sample_idx):
        lo = lo_all[bi]
        hi = hi_all[bi]
        depth = int(depths[bi])
        c_B = 0.5 * (lo + hi)
        # Project centroid onto simplex by simple normalisation (mass need
        # not equal 1 for a generic box centroid, but g_W is quadratic in
        # mu so we want a representative interior point).
        s = c_B.sum()
        if s > 0:
            c_B = c_B / s
        g_c = gW_value(c_B, A_tensor, scales)
        argmax_c = int(np.argmax(g_c))
        topk_idx = np.argsort(-g_c)[:TOP_K]
        # ensure argmax is included (it already is since we sort desc)
        top5_set = set(int(i) for i in topk_idx)
        top5_set.add(argmax_c)
        top5_idx = sorted(top5_set)
        top5_windows = [windows[i] for i in top5_idx]

        # Integer endpoints (the queue should be on dyadic grid)
        lo_int = [int(round(float(x) * SCALE)) for x in lo]
        hi_int = [int(round(float(x) * SCALE)) for x in hi]

        # ---- Integer rigor-gate certs (per window, take any) ----
        win_subsets = {
            "ALL": (windows, list(range(W))),
            "ACTIVE": (A_windows, list(A_idx)),
            "TOP5": (top5_windows, top5_idx),
        }

        record = {"box_idx": int(bi), "depth": depth,
                  "argmax_w@cB": argmax_c, "top5_idx": top5_idx}

        for r, (wlist, _) in win_subsets.items():
            any_nat = any(
                bound_natural_int_ge(lo_int, hi_int, w, TARGET_NUM, TARGET_DEN)
                for w in wlist
            )
            any_sw = any(
                bound_mccormick_sw_int_ge(lo_int, hi_int, w, D,
                                          TARGET_NUM, TARGET_DEN)
                for w in wlist
            )
            any_ne = any(
                bound_mccormick_ne_int_ge(lo_int, hi_int, w, D,
                                          TARGET_NUM, TARGET_DEN)
                for w in wlist
            )
            any_int = any_nat or any_sw or any_ne
            summary[r]["int_nat_certs"]    += int(any_nat)
            summary[r]["int_mcc_sw_certs"] += int(any_sw)
            summary[r]["int_mcc_ne_certs"] += int(any_ne)
            summary[r]["int_any_certs"]    += int(any_int)
            record[f"int_{r}_nat"] = bool(any_nat)
            record[f"int_{r}_sw"] = bool(any_sw)
            record[f"int_{r}_ne"] = bool(any_ne)
            record[f"int_{r}_any"] = bool(any_int)

        # ---- Float epi-LP per restriction ----
        for r, (wlist, _) in win_subsets.items():
            cert, lp_val, _ = bound_epigraph_int_ge_with_marginals(
                lo, hi, wlist, D, TARGET_F,
            )
            summary[r]["lp_vals"].append(float(lp_val))
            summary[r]["lp_certs"] += int(cert)
            record[f"lp_{r}_val"] = float(lp_val)
            record[f"lp_{r}_cert"] = bool(cert)

        per_box_records.append(record)

        if (n + 1) % 5 == 0 or n == 0:
            print(f"  [{n+1:>2}/{N_SAMPLES}] box={bi:>4} d={depth:>3}  "
                  f"lp ALL={record['lp_ALL_val']:.6f}  "
                  f"ACTIVE={record['lp_ACTIVE_val']:.6f}  "
                  f"TOP5={record['lp_TOP5_val']:.6f}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n[done] elapsed = {elapsed:.1f}s", flush=True)

    # --- Aggregate ---
    print(f"\n=== SUMMARY (n={N_SAMPLES} boxes, d={D}, target={TARGET_F}) ===")
    print(f"|A| = {len(A_idx)} / {W} windows")
    print(f"{'restr':<8} {'int_certs':>10} {'lp_certs':>9} "
          f"{'mean_lp':>10} {'min_lp':>10} {'max_lp':>10}")
    agg = {}
    for r in restrictions:
        lp_vals = np.array(summary[r]["lp_vals"], dtype=float)
        # Replace -inf (LP failure) with NaN for stats; track separately
        finite = lp_vals[np.isfinite(lp_vals)]
        mean_lp = float(finite.mean()) if finite.size > 0 else float("nan")
        min_lp  = float(finite.min())  if finite.size > 0 else float("nan")
        max_lp  = float(finite.max())  if finite.size > 0 else float("nan")
        agg[r] = {
            "int_any_certs": summary[r]["int_any_certs"],
            "int_nat_certs": summary[r]["int_nat_certs"],
            "int_mcc_sw_certs": summary[r]["int_mcc_sw_certs"],
            "int_mcc_ne_certs": summary[r]["int_mcc_ne_certs"],
            "lp_certs": summary[r]["lp_certs"],
            "mean_lp": mean_lp, "min_lp": min_lp, "max_lp": max_lp,
            "n_lp_finite": int(finite.size),
            "lp_vals": [float(v) for v in lp_vals],
        }
        print(f"{r:<8} {summary[r]['int_any_certs']:>10d} "
              f"{summary[r]['lp_certs']:>9d} "
              f"{mean_lp:>10.6f} {min_lp:>10.6f} {max_lp:>10.6f}")

    # Pairwise mean-delta vs ALL
    lp_all    = np.array(summary["ALL"]["lp_vals"], dtype=float)
    lp_act    = np.array(summary["ACTIVE"]["lp_vals"], dtype=float)
    lp_top5   = np.array(summary["TOP5"]["lp_vals"], dtype=float)
    msk = np.isfinite(lp_all) & np.isfinite(lp_act) & np.isfinite(lp_top5)
    delta_act_vs_all  = lp_act[msk]  - lp_all[msk]
    delta_top5_vs_all = lp_top5[msk] - lp_all[msk]
    print(f"\nDelta(ACTIVE - ALL):  mean={delta_act_vs_all.mean():+.6e}  "
          f"min={delta_act_vs_all.min():+.6e}  max={delta_act_vs_all.max():+.6e}")
    print(f"Delta(TOP5   - ALL):  mean={delta_top5_vs_all.mean():+.6e}  "
          f"min={delta_top5_vs_all.min():+.6e}  max={delta_top5_vs_all.max():+.6e}")
    n_act_strict_higher  = int(np.sum(delta_act_vs_all  > 1e-12))
    n_top5_strict_higher = int(np.sum(delta_top5_vs_all > 1e-12))
    print(f"# boxes ACTIVE strictly > ALL:  {n_act_strict_higher}/{int(msk.sum())}")
    print(f"# boxes TOP5   strictly > ALL:  {n_top5_strict_higher}/{int(msk.sum())}")

    payload = {
        "d": D,
        "target": TARGET_F,
        "target_q": [TARGET_NUM, TARGET_DEN],
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "top_k": TOP_K,
        "n_windows": W,
        "active_set_size": int(len(A_idx)),
        "active_set_indices": [int(i) for i in A_idx],
        "val_star": val_star,
        "max_g_at_mu_star": max_star,
        "elapsed_s": elapsed,
        "by_restriction": agg,
        "delta_active_vs_all": {
            "mean": float(delta_act_vs_all.mean()) if delta_act_vs_all.size else None,
            "min":  float(delta_act_vs_all.min())  if delta_act_vs_all.size else None,
            "max":  float(delta_act_vs_all.max())  if delta_act_vs_all.size else None,
            "n_strict_higher": n_act_strict_higher,
        },
        "delta_top5_vs_all": {
            "mean": float(delta_top5_vs_all.mean()) if delta_top5_vs_all.size else None,
            "min":  float(delta_top5_vs_all.min())  if delta_top5_vs_all.size else None,
            "max":  float(delta_top5_vs_all.max())  if delta_top5_vs_all.size else None,
            "n_strict_higher": n_top5_strict_higher,
        },
        "per_box": per_box_records,
    }
    out = os.path.join(REPO, "active_window_audit.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
