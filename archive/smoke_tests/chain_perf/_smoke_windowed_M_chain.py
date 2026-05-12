"""Smoke test: windowed-test-value (CS-2017 dynamic threshold) at L0.

OBJECTIVE.  Establish C_{1a} > 1.281 by running the CS-2017 dynamic-threshold
certifier (Lean theorem `dynamic_threshold_sound_tight`) at L0 of
(n_half=1, m=20).  The certifier needs, for EVERY integer composition c with
sum 4·n_half·m = 80, a window (ell, s_lo) with

    TV_disc(c; ell, s_lo)  >  c_target  +  Δ_tight(n_half, m, c, ell, s_lo)

where TV_disc is the cascade's window-averaged test value and Δ_tight is the
D-v2 (factor 1) discretization error correction proven sound by Lemma 1
(`continuous_test_value_le_ratio`, proven) + Lemma 3 (`cs_eq1_tight`, axiom).

If this holds for all 41 canonical L0 compositions (or all 81 total, including
non-canonical), then by `dynamic_threshold_sound_tight` (proven from axioms)
every continuous f with canonical_discretization equal to some c has
||f*f||_∞ ≥ c_target.  Since canonical_discretization is surjective onto
the simplex of integer compositions of S, this gives C_{1a} ≥ c_target = 1.281.

QUESTION A (windowed M-chain version).  TV_disc is NOT the M-chain LB.
TV_disc(c) is purely a function of the integer composition c — no cell
allowance, no LB(k) — because the discretization correction Δ_tight already
absorbs the ±1 cell allowance via the |c_i/m - mu_i| ≤ 1/m rounding bound.
Therefore the natural windowed-cell-bound is just TV_disc(c) itself, and
the windowed M-chain question reduces to:

  for each L0 composition c, is there (ell, s_lo) with
    TV_disc(c; ell, s_lo) - Δ_tight  >  1.281?

This smoke runs that check.

USAGE:
    python _smoke_windowed_M_chain.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


# ----------------------------------------------------------------------
# Per-composition windowed test value + D-v2 tight correction.
# ----------------------------------------------------------------------
def _conv_int(c):
    """Integer autoconv: conv[k] = sum_{i+j=k} c_i c_j (length 2d-1)."""
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    conv = np.zeros(2 * d - 1, dtype=np.int64)
    for i in range(d):
        ci = int(c[i])
        if ci == 0:
            continue
        conv[2 * i] += ci * ci
        for j in range(i + 1, d):
            cj = int(c[j])
            if cj != 0:
                conv[i + j] += 2 * ci * cj
    return conv


def _ell_int_arr(d, n_half):
    """ell_int_arr[k] = #{(i,j) in [0,d-1]^2 : i+j = k}.

    Equivalent to n_pairs(k) = max(0, 2n - |k+1 - 2n|).
    """
    conv_len = 2 * d - 1
    arr = np.empty(conv_len, dtype=np.int64)
    two_n = 2 * n_half
    for k in range(conv_len):
        d_idx = abs((k + 1) - two_n)
        v = two_n - d_idx
        if v < 0:
            v = 0
        arr[k] = v
    return arr


def _tv_disc(c, ell, s_lo, n_half, m):
    """TV_disc(c; ell, s_lo) = (1/(4 n_half ell m^2)) * sum_{s=s_lo..s_lo+ell-2} conv[s](c).

    Matches Lean `test_value` (Defs.lean:57) and Python `compute_test_value_single`
    (test_values.py:127).
    """
    conv = _conv_int(c)
    n_cv = ell - 1
    if s_lo + n_cv > len(conv):
        return None
    ws = int(conv[s_lo : s_lo + n_cv].sum())
    norm = 4.0 * n_half * ell * (m ** 2)
    return ws / norm


def _delta_tight_d_v2(c, ell, s_lo, n_half, m):
    """Δ_tight (D-v2) = (ell-1)·W_int/(2n·ell·m²) + ell_int_sum/(4n·ell·m²).

    This is the strictly tighter version of Δ_tight than D-v1 (factor 3).
    Lean reference: `tight_correction` in TightContinuousBridge.lean,
    proven sound by `cs_eq1_tight` (axiom).

    Args:
      c:       integer composition (length d = 2 n_half)
      ell:     window length (>= 2)
      s_lo:    window start in conv-positions
      n_half:  cascade param
      m:       discretization scale

    Returns:
      Δ_tight in TV space (units of (f*f) value).
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    n_cv = ell - 1

    # W_int = sum_{i in contributing bins} c_i
    # Contributing bins = [lo_bin, hi_bin] where lo_bin = max(0, s_lo - (d-1))
    # and hi_bin = min(d-1, s_lo + ell - 2).
    lo_bin = max(0, s_lo - (d - 1))
    hi_bin = min(d - 1, s_lo + ell - 2)
    W_int = int(c[lo_bin : hi_bin + 1].sum())

    # ell_int_sum = sum_{s=s_lo..s_lo+ell-2} ell_int_arr[s]
    ell_int = _ell_int_arr(d, n_half)
    if s_lo + n_cv > len(ell_int):
        return None
    ell_int_sum = int(ell_int[s_lo : s_lo + n_cv].sum())

    # D-v2 correction
    n_half_f = float(n_half)
    ell_f = float(ell)
    m_sq = float(m * m)
    corr_lin = (ell - 1) * W_int / (2.0 * n_half_f * ell_f * m_sq)
    corr_quad = ell_int_sum / (4.0 * n_half_f * ell_f * m_sq)
    return corr_lin + corr_quad


def _delta_tight_d_v1(c, ell, s_lo, n_half, m):
    """Δ_tight (D-v1, what _stage1_bench.prune_D actually uses).

    corr = (ell-1)·W_int/(2n·ell·m²) + 3·ell_int_sum/(4n·ell·m²).

    Matches the Python prune_D in _stage1_bench.py:367-373.
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
    n_cv = ell - 1
    lo_bin = max(0, s_lo - (d - 1))
    hi_bin = min(d - 1, s_lo + ell - 2)
    W_int = int(c[lo_bin : hi_bin + 1].sum())
    ell_int = _ell_int_arr(d, n_half)
    if s_lo + n_cv > len(ell_int):
        return None
    ell_int_sum = int(ell_int[s_lo : s_lo + n_cv].sum())
    n_half_f = float(n_half)
    ell_f = float(ell)
    m_sq = float(m * m)
    corr_lin = (ell - 1) * W_int / (2.0 * n_half_f * ell_f * m_sq)
    corr_quad = 3.0 * ell_int_sum / (4.0 * n_half_f * ell_f * m_sq)
    return corr_lin + corr_quad


def cert_one(c, n_half, m, c_target, variant="D-v2"):
    """For a single integer composition c, find best (ell, s_lo) s.t.
    TV_disc(c; ell, s_lo) - Δ_tight  >  c_target.

    Returns dict with best_margin (= TV - Δ - c_target), best (ell, s_lo),
    TV value, Δ value, and pass flag.
    """
    d = len(c)
    conv_len = 2 * d - 1
    max_ell = 2 * d  # window can span all conv positions

    best = None
    best_margin = -float("inf")

    # Compute conv once
    conv = _conv_int(c)

    delta_fn = _delta_tight_d_v2 if variant == "D-v2" else _delta_tight_d_v1

    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            ws = int(conv[s_lo : s_lo + n_cv].sum())
            tv = ws / (4.0 * n_half * ell * (m ** 2))
            delta = delta_fn(c, ell, s_lo, n_half, m)
            if delta is None:
                continue
            margin = tv - delta - c_target
            if margin > best_margin:
                best_margin = margin
                best = {
                    "ell": ell,
                    "s_lo": s_lo,
                    "TV_disc": tv,
                    "Delta_tight": delta,
                    "TV_minus_Delta": tv - delta,
                    "margin": margin,
                }
    if best is None:
        return {"c": list(int(x) for x in c), "passes": False, "best_margin": -float("inf")}
    return {
        "c": [int(x) for x in c],
        "passes": best_margin > 0,
        **best,
    }


def _generate_all_d2(n_half, m):
    """Generate ALL c=(c0, c1) with c0 + c1 = 4*n_half*m, c_i >= 0.

    For d = 2 (n_half = 1), this is 4n*m + 1 = 81 compositions for m=20.
    """
    S = 4 * n_half * m
    return np.array([[a, S - a] for a in range(0, S + 1)], dtype=np.int32)


def _generate_canonical_d2(n_half, m):
    """All canonical c=(c0, c1) with c0+c1 = 4*n_half*m and c0 <= c1.

    For (n_half=1, m=20) this is 41 compositions (c0 in [0, 40]).
    """
    S = 4 * n_half * m
    return np.array([[a, S - a] for a in range(0, S // 2 + 1)], dtype=np.int32)


def run(n_half, m, c_target, output_path=None):
    print("=" * 72)
    print(f"WINDOWED CS-2017 DYNAMIC-THRESHOLD CERTIFIER")
    print(f"  n_half={n_half}, m={m}, c_target={c_target}")
    print(f"  d = 2 n_half = {2 * n_half}, S = 4 n_half m = {4 * n_half * m}")
    print("=" * 72)

    # All compositions (including non-canonical)
    batch_all = _generate_all_d2(n_half, m)
    batch_can = _generate_canonical_d2(n_half, m)
    print(f"  total all comps:       {len(batch_all)}")
    print(f"  total canonical comps: {len(batch_can)} (c0 <= c1)")

    # ----- D-v1 (current Python prune_D) ------
    t0 = time.time()
    results_v1 = [cert_one(c, n_half, m, c_target, variant="D-v1")
                  for c in batch_all]
    n_pass_v1 = sum(1 for r in results_v1 if r["passes"])
    fails_v1 = [r for r in results_v1 if not r["passes"]]
    elapsed_v1 = time.time() - t0
    print(f"\n[D-v1, factor 3] elapsed={elapsed_v1:.2f}s")
    print(f"  pass {n_pass_v1}/{len(batch_all)}, "
          f"fail {len(fails_v1)}.  ALL pass = {n_pass_v1 == len(batch_all)}")

    # ----- D-v2 (Lean cs_eq1_tight, factor 1) ------
    t0 = time.time()
    results_v2 = [cert_one(c, n_half, m, c_target, variant="D-v2")
                  for c in batch_all]
    n_pass_v2 = sum(1 for r in results_v2 if r["passes"])
    fails_v2 = [r for r in results_v2 if not r["passes"]]
    elapsed_v2 = time.time() - t0
    print(f"\n[D-v2, factor 1, Lean cs_eq1_tight] elapsed={elapsed_v2:.2f}s")
    print(f"  pass {n_pass_v2}/{len(batch_all)}, "
          f"fail {len(fails_v2)}.  ALL pass = {n_pass_v2 == len(batch_all)}")

    # Failures
    if fails_v2:
        print(f"\n  Top-{min(10, len(fails_v2))} D-v2 failing compositions "
              f"(margin = TV_disc - Delta - c_target):")
        for r in sorted(fails_v2, key=lambda r: -r.get("best_margin", -1))[:10]:
            margin = r.get("margin", r.get("best_margin", "?"))
            print(f"    c={r['c']}  best_margin={margin:.6f}  "
                  f"(ell={r.get('ell', '?')}, s_lo={r.get('s_lo', '?')})")
    else:
        print(f"\n  *** ALL {len(batch_all)} COMPOSITIONS CERTIFIED via D-v2 ***")
        print(f"  C_{{1a}} > {c_target} is established at L0 (n={n_half}, m={m})")

    # Detailed results for canonical comps
    can_set = set(map(tuple, batch_can.tolist()))
    canonical_v2 = [r for r in results_v2 if tuple(r["c"]) in can_set]
    n_pass_can_v2 = sum(1 for r in canonical_v2 if r["passes"])
    fails_can_v2 = [r for r in canonical_v2 if not r["passes"]]
    print(f"\n  Canonical (c0 <= c1) D-v2: pass {n_pass_can_v2}/{len(canonical_v2)}")

    out = {
        "config": {
            "n_half": n_half,
            "m": m,
            "c_target": c_target,
            "d": 2 * n_half,
            "S": 4 * n_half * m,
            "total_all_comps": int(len(batch_all)),
            "total_canonical": int(len(batch_can)),
        },
        "D_v1": {
            "n_pass": int(n_pass_v1),
            "n_fail": int(len(fails_v1)),
            "all_pass": bool(n_pass_v1 == len(batch_all)),
            "elapsed_s": float(elapsed_v1),
            "fails_top10": sorted(fails_v1, key=lambda r: -r.get("best_margin", -1))[:10],
        },
        "D_v2": {
            "n_pass": int(n_pass_v2),
            "n_fail": int(len(fails_v2)),
            "all_pass": bool(n_pass_v2 == len(batch_all)),
            "elapsed_s": float(elapsed_v2),
            "fails_top10": sorted(fails_v2, key=lambda r: -r.get("best_margin", -1))[:10],
        },
        "canonical_D_v2": {
            "n_pass": int(n_pass_can_v2),
            "n_fail": int(len(fails_can_v2)),
            "all_pass": bool(n_pass_can_v2 == len(canonical_v2)),
            "details": canonical_v2,
        },
    }

    # ---------------- Verdict ----------------
    if n_pass_v2 == len(batch_all):
        verdict = "C_1A_PROVED_AT_L0"
        out["verdict"] = verdict
        out["proof_chain"] = [
            "Lemma 1 (continuous_test_value_le_ratio, PROVEN in TestValueBounds.lean:382)",
            "Lemma 3 / cs_eq1_tight (AXIOM, paper-grade derivable, in TightContinuousBridge.lean:203)",
            "dynamic_threshold_sound_tight (PROVEN from above, in TightContinuousBridge.lean:381)",
            f"Computational: TV_disc(c) - Delta_tight > {c_target} for all {len(batch_all)} compositions.",
        ]
    else:
        verdict = "C_1A_NOT_PROVED_AT_L0"
        out["verdict"] = verdict
        out["proof_blocker"] = (
            f"D-v2 fails on {len(fails_v2)} compositions; cascade must "
            f"refine to deeper L1 to handle these."
        )

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_smoke_windowed_M_chain.json",
        )
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {output_path}")

    return out


if __name__ == "__main__":
    n_half = 1
    m = 20
    c_target = 1.281
    run(n_half=n_half, m=m, c_target=c_target)
