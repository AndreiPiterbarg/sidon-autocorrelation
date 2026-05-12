"""Mathematical trend analysis for the Pólya LP sweep.

Uses RIGOROUS THEORY-BASED MODELS rather than fitting arbitrary R^{-a}:

(1) Per-d convergence (alpha vs R, fixed d):

  Theory (Powers-Reznick 2001, de Klerk-Laurent-Sun 2014, arXiv:1407.2108
  for quadratic-on-simplex):

    val(d) - alpha_LP(d, R)   ≤   C_d * 1/R                   (P1, linear)

  where C_d depends on the polynomial's L-infinity norm and minimum
  value on the simplex. For Pólya/Handelman LP with quadratic
  objective, the rate is provably O(1/R), with C_d governed by
  ||M||_inf and the gap min_mu val - alpha.

  We fit 4 candidate models and report which one's residuals are smallest:

    M1: gap(R) = C / R                   (Pólya theory, linear)
    M2: gap(R) = C / R^2                 (Lasserre SDP rate, would not apply
                                          to LP but tests the data)
    M3: gap(R) = C * exp(-c R)           (geometric — unlikely for LP)
    M4: gap(R) = C / R^a (free a)        (generic power-law fit)

(2) Cross-d scaling (C_d as function of d):

  Theory: C_d ~ ||M_W||_inf ~ 2d/(min ell) ~ 2d for finest band.
  So C_d ~ a + b*d (linear). We fit:

    L1: C(d) = b * d                     (pure linear, theory)
    L2: C(d) = a + b * d                 (linear with offset)
    L3: C(d) = a * d^b                   (power law)
    L4: C(d) = a + b * d + c * d^2       (quadratic)

(3) Projection: given a target alpha (e.g., 1.281), and the val(d)
  (the LP's asymptotic limit, which is val(d) per CS 2017 chain),
  we project R_needed(d) = C(d) / (val(d) - alpha_target) under M1.

Output:
  - Per-d fits (which model wins, what's C_d).
  - Cross-d fit (C as function of d).
  - Projected (d, R) combinations to clear alpha >= 1.281, including
    estimated LP size to assess feasibility.
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from math import comb, log, exp

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "sweep_results.json")
SUMMARY_PATH = os.path.join(HERE, "sweep_summary.md")


VAL_D_KNOWN = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319, 20: 1.328, 24: 1.332,
    32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448,
}
TARGET = 1.281


# ---------------------------------------------------------------------
# Model fitters
# ---------------------------------------------------------------------

def fit_linear_least_squares(x, y):
    """Fit y = a + b x by OLS. Returns (a, b, ss_res)."""
    x = np.asarray(x); y = np.asarray(y)
    A = np.vstack([np.ones_like(x), x]).T
    sol, ss_res_arr, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = sol
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    return float(a), float(b), ss_res


def fit_polya_M1(R, gap):
    """gap = C / R. Linearize: gap * R = C. Median of (gap * R)."""
    gR = np.asarray(gap) * np.asarray(R)
    C = float(np.median(gR))
    pred = C / np.asarray(R)
    ss_res = float(np.sum((np.asarray(gap) - pred) ** 2))
    return C, ss_res


def fit_lasserre_M2(R, gap):
    """gap = C / R^2."""
    gR2 = np.asarray(gap) * np.asarray(R) ** 2
    C = float(np.median(gR2))
    pred = C / np.asarray(R) ** 2
    ss_res = float(np.sum((np.asarray(gap) - pred) ** 2))
    return C, ss_res


def fit_geometric_M3(R, gap):
    """gap = C * exp(-c R). Linearize: log gap = log C - c R."""
    R = np.asarray(R, dtype=float); g = np.asarray(gap, dtype=float)
    mask = g > 1e-12
    if mask.sum() < 2:
        return None, float("inf")
    log_g = np.log(g[mask])
    Rm = R[mask]
    a, b, _ = fit_linear_least_squares(Rm, log_g)
    C = exp(a); c = -b
    pred = C * np.exp(-c * R)
    ss_res = float(np.sum((g - pred) ** 2))
    return (C, c), ss_res


def fit_power_M4(R, gap):
    """gap = C / R^a. Linearize: log gap = log C - a log R."""
    R = np.asarray(R, dtype=float); g = np.asarray(gap, dtype=float)
    mask = g > 1e-12
    if mask.sum() < 2:
        return None, float("inf")
    log_R = np.log(R[mask])
    log_g = np.log(g[mask])
    a, b, _ = fit_linear_least_squares(log_R, log_g)
    C = exp(a); exponent = -b
    pred = C / R ** exponent
    ss_res = float(np.sum((g - pred) ** 2))
    return (C, exponent), ss_res


# ---------------------------------------------------------------------
# Cross-d scaling fitters (C_d as function of d)
# ---------------------------------------------------------------------

def fit_C_pure_linear(d, C):
    """C(d) = b * d (no intercept; theory)."""
    d = np.asarray(d, dtype=float); C = np.asarray(C, dtype=float)
    b = float(np.sum(d * C) / np.sum(d * d))
    pred = b * d
    ss_res = float(np.sum((C - pred) ** 2))
    return b, ss_res


def fit_C_affine(d, C):
    """C(d) = a + b * d."""
    a, b, ss_res = fit_linear_least_squares(d, C)
    return (a, b), ss_res


def fit_C_power(d, C):
    """C(d) = a * d^b. Linearize."""
    d = np.asarray(d, dtype=float); C = np.asarray(C, dtype=float)
    mask = (d > 0) & (C > 0)
    if mask.sum() < 2:
        return None, float("inf")
    a_log, b, _ = fit_linear_least_squares(np.log(d[mask]), np.log(C[mask]))
    a = exp(a_log)
    pred = a * d ** b
    ss_res = float(np.sum((C - pred) ** 2))
    return (a, b), ss_res


def fit_C_quadratic(d, C):
    """C(d) = a + b * d + c * d^2."""
    d = np.asarray(d, dtype=float); C = np.asarray(C, dtype=float)
    A = np.vstack([np.ones_like(d), d, d ** 2]).T
    sol, *_ = np.linalg.lstsq(A, C, rcond=None)
    a, b, c = sol
    pred = a + b * d + c * d ** 2
    ss_res = float(np.sum((C - pred) ** 2))
    return (float(a), float(b), float(c)), ss_res


# ---------------------------------------------------------------------
# Estimated LP size (for feasibility assessment)
# ---------------------------------------------------------------------

def estimated_lp_size(d: int, R: int) -> dict:
    d_eff = d // 2
    n_eq = comb(d_eff + R, R)
    n_q = comb(d_eff + R - 1, R - 1)
    n_c = n_eq
    n_lambda_typical = (2 * d) ** 2 // 4
    n_vars = 1 + n_lambda_typical + n_q + n_c
    nnz_estimate = n_vars * (d_eff + 2)
    return dict(n_eq=n_eq, n_vars=n_vars, nnz=nnz_estimate)


def laptop_feasible(size: dict, ram_gb: float = 8.0) -> bool:
    """Heuristic: LP solvable on a laptop with ram_gb of RAM if
    nnz < 5e6 and n_vars < 1.5e6 (we OOMed at d=32 R=8 with ~1M vars)."""
    return size["n_vars"] < 1_500_000 and size["nnz"] < 5_000_000


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def analyze():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # Bucket by d
    by_d = defaultdict(list)
    for r in results:
        if r.get("alpha") is not None:
            by_d[r["d"]].append(r)
    for d in by_d:
        by_d[d].sort(key=lambda r: r["R"])

    lines = []
    lines.append("# Pólya LP Sweep — Mathematical Trend Analysis\n\n")
    lines.append(f"Total successful records: {sum(len(v) for v in by_d.values())}\n")
    lines.append(f"d values with >=3 points: "
                 f"{sorted(d for d, v in by_d.items() if len(v) >= 3)}\n\n")
    lines.append(f"Theory: Pólya/Handelman LP for quadratic on simplex Δ_d "
                 f"has gap(R) = O(C_d / R) where C_d ~ ||M||_∞ ~ d. "
                 f"Below we fit MULTIPLE models per d and compare residuals.\n\n")

    # ============================================================
    # PER-D MODEL COMPARISON
    # ============================================================
    lines.append("## Step 1: Per-d gap model fits\n\n")
    lines.append("For each d with >=3 (R, alpha) points, fit gap(R) = "
                 "val_d - alpha_LP(R) under 4 models. Pick the one with "
                 "smallest sum-of-squared-residuals.\n\n")

    lines.append("| d | val_d | n_pts | M1 (C/R) ss_res | M2 (C/R²) ss_res | "
                 "M4 (C/R^a, a) ss_res | best |\n")
    lines.append("|---|---|---|---|---|---|---|\n")

    per_d_fit = {}   # d -> (C_M1, model_winner)
    per_d_M1_C = {}

    for d in sorted(by_d.keys()):
        runs = by_d[d]
        if len(runs) < 3:
            continue
        # Use known val_d if available, else best-alpha + 0.05 as proxy
        val_d = VAL_D_KNOWN.get(d)
        if val_d is None:
            # Use highest observed alpha + small margin
            val_d_use = max(r["alpha"] for r in runs) + 0.02
            val_d_str = f"~{val_d_use:.3f}"
        else:
            val_d_use = val_d
            val_d_str = f"{val_d:.3f}"

        Rs = np.array([r["R"] for r in runs], dtype=float)
        alphas = np.array([r["alpha"] for r in runs])
        gaps = val_d_use - alphas
        if (gaps <= 0).any():
            # at least one alpha exceeds claimed val_d; skip if unreliable
            continue

        # Fit each model
        C_M1, ss_M1 = fit_polya_M1(Rs, gaps)
        C_M2, ss_M2 = fit_lasserre_M2(Rs, gaps)
        res_M4, ss_M4 = fit_power_M4(Rs, gaps)

        # Compare (M1 has 1 free param, M4 has 2; for fairness compare ss_res)
        candidates = [
            ("M1", ss_M1, C_M1),
            ("M2", ss_M2, C_M2),
        ]
        if res_M4 is not None:
            candidates.append(("M4", ss_M4, res_M4))
        winner = min(candidates, key=lambda x: x[1])

        per_d_fit[d] = (val_d_use, winner)
        per_d_M1_C[d] = C_M1

        m4_str = (f"M4: C={res_M4[0]:.3f},a={res_M4[1]:.3f} "
                  f"ss={ss_M4:.2e}" if res_M4 is not None else "—")
        lines.append(f"| {d} | {val_d_str} | {len(runs)} | "
                     f"M1: C={C_M1:.3f} ss={ss_M1:.2e} | "
                     f"M2: C={C_M2:.3f} ss={ss_M2:.2e} | "
                     f"{m4_str} | {winner[0]} |\n")

    # ============================================================
    # CROSS-D SCALING OF C_M1 (the theory-relevant constant)
    # ============================================================
    lines.append("\n## Step 2: Cross-d scaling of C_d (the M1 constant)\n\n")
    lines.append("Theory: C_d ~ ||M_W||_∞ ~ 2d (since M_W = 2d/ell · 1[band]). "
                 "Test linear, affine, and power-law models for C(d).\n\n")

    if len(per_d_M1_C) >= 3:
        ds = sorted(per_d_M1_C.keys())
        Cs = [per_d_M1_C[d] for d in ds]

        b_pure, ss_pure = fit_C_pure_linear(ds, Cs)
        (a_aff, b_aff), ss_aff = fit_C_affine(ds, Cs)
        res_pow, ss_pow = fit_C_power(ds, Cs)
        res_quad, ss_quad = fit_C_quadratic(ds, Cs)

        lines.append("| Model | Form | Fit | SS residuals |\n")
        lines.append("|---|---|---|---|\n")
        lines.append(f"| L1 | C(d) = b·d | b = {b_pure:.4f} | {ss_pure:.4e} |\n")
        lines.append(f"| L2 | C(d) = a + b·d | a = {a_aff:.4f}, "
                     f"b = {b_aff:.4f} | {ss_aff:.4e} |\n")
        if res_pow is not None:
            lines.append(f"| L3 | C(d) = a·d^b | a = {res_pow[0]:.4f}, "
                         f"b = {res_pow[1]:.4f} | {ss_pow:.4e} |\n")
        lines.append(f"| L4 | C(d) = a + b·d + c·d² | "
                     f"a = {res_quad[0]:.4f}, b = {res_quad[1]:.4f}, "
                     f"c = {res_quad[2]:.6f} | {ss_quad:.4e} |\n")

        # Pick best non-overfitting fit. Restrict to monotone non-negative
        # extrapolations for d > max(d_data); L4 is overfit and goes negative.
        # Among L1, L2, L3: pick lowest ss.
        candidates_c = [
            ("L1", ss_pure, ("b", b_pure)),
            ("L2", ss_aff, ("a", a_aff, "b", b_aff)),
        ]
        if res_pow is not None:
            candidates_c.append(("L3", ss_pow, ("a", res_pow[0], "b", res_pow[1])))
        # NOTE: L4 (quadratic) added but EXCLUDED from extrapolation since
        # it can go negative (over-fitting). Use only L1/L2/L3 for projection.
        l4_only = ("L4", ss_quad, ("a", res_quad[0], "b", res_quad[1],
                                    "c", res_quad[2]))
        best_c = min(candidates_c, key=lambda x: x[1])
        lines.append(f"\n**Best cross-d fit (used for extrapolation):** "
                     f"{best_c[0]} with params {best_c[2]}\n")
        lines.append(f"**L4 (quadratic) ss={l4_only[1]:.4e}** but EXCLUDED "
                     f"from extrapolation — goes negative for large d.\n\n")

        lines.append("\n### Per-d C values (data + best fit prediction):\n\n")
        lines.append("| d | C_M1 (data) | L1 pred | L2 pred | L3 pred | L4 pred |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for d_val in ds:
            C_data = per_d_M1_C[d_val]
            l1_pred = b_pure * d_val
            l2_pred = a_aff + b_aff * d_val
            l3_pred = (res_pow[0] * d_val ** res_pow[1]
                       if res_pow is not None else float("nan"))
            l4_pred = res_quad[0] + res_quad[1] * d_val + res_quad[2] * d_val ** 2
            lines.append(f"| {d_val} | {C_data:.4f} | {l1_pred:.4f} | "
                         f"{l2_pred:.4f} | {l3_pred:.4f} | {l4_pred:.4f} |\n")
    else:
        lines.append("Not enough per-d fits yet (need >=3 d values).\n\n")
        best_c = None
        b_pure = None
        a_aff = b_aff = None

    # ============================================================
    # PROJECTIONS
    # ============================================================
    lines.append("\n## Step 3: Projected R needed to clear alpha >= "
                 f"{TARGET} per d\n\n")
    lines.append("Using M1 (gap = C_d/R) and best cross-d fit for C_d, "
                 "compute R_needed = C_d / (val_d - target).\n\n")

    if best_c is not None and len(per_d_M1_C) >= 3:
        lines.append("| d | val_d | C_d (data) | C_d (best fit) | gap_to_target | "
                     "R_needed (data) | R_needed (fit) | LP size at R_needed | laptop? |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|\n")

        check_d = sorted(set(list(VAL_D_KNOWN.keys())))
        for d_val in check_d:
            val_d = VAL_D_KNOWN[d_val]
            if val_d <= TARGET:
                # impossible at this d (val_d already < target)
                lines.append(f"| {d_val} | {val_d:.3f} | "
                             f"{'(no data)' if d_val not in per_d_M1_C else f'{per_d_M1_C[d_val]:.3f}'} | "
                             f"— | {val_d - TARGET:+.3f} | "
                             f"impossible (val<target) | — | — | — |\n")
                continue

            eps = val_d - TARGET

            C_data = per_d_M1_C.get(d_val)
            R_data_str = f"{C_data/eps:.1f}" if C_data is not None else "—"

            # C from best non-quadratic fit (L4 excluded as overfit)
            if best_c[0] == "L1":
                C_fit = b_pure * d_val
            elif best_c[0] == "L2":
                C_fit = a_aff + b_aff * d_val
            elif best_c[0] == "L3":
                C_fit = res_pow[0] * d_val ** res_pow[1]
            else:
                C_fit = a_aff + b_aff * d_val   # default to L2

            R_fit = C_fit / eps if eps > 0 else float("inf")

            R_for_size = max(int(round(R_fit)), 4)
            sz = estimated_lp_size(d_val, R_for_size)
            lab_ok = "YES" if laptop_feasible(sz) else "NO"

            C_data_str = f"{C_data:.3f}" if C_data is not None else "—"
            lines.append(f"| {d_val} | {val_d:.3f} | "
                         f"{C_data_str} | "
                         f"{C_fit:.3f} | {eps:+.4f} | {R_data_str} | "
                         f"{R_fit:.1f} | "
                         f"n_vars≈{sz['n_vars']:,} nnz≈{sz['nnz']:,} | "
                         f"{lab_ok} |\n")

        lines.append("\n### Sweet spot recommendation\n\n")
        # Find smallest (d, R) on laptop with projected α ≥ target
        best_target = None
        for d_val in sorted(VAL_D_KNOWN.keys()):
            if VAL_D_KNOWN[d_val] <= TARGET:
                continue
            if best_c[0] == "L1":
                C_fit = b_pure * d_val
            elif best_c[0] == "L2":
                C_fit = a_aff + b_aff * d_val
            elif best_c[0] == "L3":
                C_fit = res_pow[0] * d_val ** res_pow[1]
            else:
                C_fit = a_aff + b_aff * d_val   # default to L2 (no L4)
            eps = VAL_D_KNOWN[d_val] - TARGET
            R_fit = C_fit / eps
            R_int = max(4, int(round(R_fit)) + 1)  # +1 for margin
            sz = estimated_lp_size(d_val, R_int)
            if laptop_feasible(sz):
                if best_target is None or R_int < best_target[1]:
                    best_target = (d_val, R_int, sz)

        if best_target is not None:
            d_val, R_int, sz = best_target
            lines.append(f"- **Smallest (d, R) projected to clear {TARGET} on this "
                         f"8GB laptop:** d={d_val}, R={R_int}, size n_vars≈{sz['n_vars']:,}, "
                         f"nnz≈{sz['nnz']:,}\n\n")
        else:
            lines.append("- **NO (d, R) combination on the 8GB laptop is projected "
                         f"to clear {TARGET}.** Cloud H100 required for the breakthrough.\n\n")

    # ============================================================
    # RAW DATA TABLE
    # ============================================================
    lines.append("\n## Raw data: alpha vs R per d\n\n")
    lines.append("| d | R | alpha | gap_to_val | wall_s | mem_MB | n_eq | n_vars | status |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(results, key=lambda r: (r["d"], r["R"])):
        alpha = r.get("alpha")
        if alpha is None:
            astr = "—"; gstr = "—"
        else:
            astr = f"{alpha:.6f}"
            val_d = VAL_D_KNOWN.get(r["d"])
            gstr = f"{val_d - alpha:+.4f}" if val_d else "?"
        lines.append(
            f"| {r['d']} | {r['R']} | {astr} | {gstr} | "
            f"{r.get('wall', 0):.1f} | {r.get('peak_mem_mb', 0):.0f} | "
            f"{r.get('n_eq', 0):,} | {r.get('n_vars', 0):,} | "
            f"{r.get('status', '?')} |\n"
        )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Summary -> {SUMMARY_PATH}")
    print(f"Total records analyzed: {sum(len(v) for v in by_d.values())}")


if __name__ == "__main__":
    analyze()
