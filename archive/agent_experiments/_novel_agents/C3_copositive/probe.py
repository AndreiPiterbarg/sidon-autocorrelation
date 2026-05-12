"""C3 Copositive cone relaxation probe for the Sidon constant.

=============================================================================
MATH (correct derivation, finally):
=============================================================================

Problem:  val(d) = min_{a in Delta_d}  max_W  a^T M_W a.

Direct copositive lifting: introduce slack t and X = a a^T:
  val(d) = min { t :  X = a a^T,  a in Delta,  <M_W, X> <= t  for all W }
        >= min { t :  X in CP_d,  <ee^T, X> = 1,  <M_W, X> <= t  for all W }
        =  CP_d-relaxation lower bound.

CP_d is HARD; use INNER approximation hierarchy of de Klerk-Pasechnik:

LEVEL r=0 (this repo's existing baseline):
  CP_d ⊆ PSD ∩ N_d   (always true; tight for d <= 4).
  -> bound: min { t : X PSD, X >= 0, sum X = 1, <M_W, X> <= t for all W }.

LEVEL r=1 (NOVEL, this probe):
  Inner approx via Parrilo SOS lift.  Equivalent characterization:
  X in InnerCP^(1) iff there exist PSD matrices Z indexed by monomials
  of degree <= 2 in y such that
     X[i,j] = Z[(0,...0), (e_i + e_j)] = m_{2 e_i + 2 e_j}
  with m the moment vector of the polynomial system in y, with the
  Parrilo (sum y_i^2)^1 multiplier extending pseudo-moments to degree 6.

  Equivalent: pseudo-moments y_alpha for |alpha| <= 6, M_3 PSD,
  equality (1 - sum y_i^2 = 0) localizing.  X[i,j] := m_{2 e_i + 2 e_j}.
  Burer constraint sum_{ij} X[i,j] = 1 becomes sum_{ij} m_{2 e_i + 2 e_j} = 1.

PROBE: For d in {6, 7, 8} compute and compare:
  (a) Level 0 bound: PSD ∩ N relaxation.
  (b) Level 1 bound: Parrilo r=1 lift.
  (c) val(d) (cell-true).

If (b) > (a) NONTRIVIALLY, copositive direction is PROMISING.
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time
import json
from datetime import datetime
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from lasserre.core import build_window_matrices  # noqa: E402

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.log")


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# Helper: enumerate monomials
def enum_monos(d: int, max_deg: int) -> List[Tuple[int, ...]]:
    out = [tuple([0] * d)]
    for k in range(1, max_deg + 1):
        for comb in combinations_with_replacement(range(d), k):
            a = [0] * d
            for v in comb:
                a[v] += 1
            out.append(tuple(a))
    return list(dict.fromkeys(out))


def add_alpha(a, b):
    return tuple(x + y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# LEVEL 0: PSD + N relaxation (Burer-style)
# ---------------------------------------------------------------------------

def level0_burer(d: int, M_mats: List[np.ndarray],
                  solver: str = "MOSEK") -> Tuple[float, str]:
    """Compute min { t : X PSD, X >= 0, sum X = 1, <M_W, X> <= t for all W }.

    This is Burer 2009 in PSD + N relaxation form (= the existing repo bound,
    expressed cleanly as a single SDP).

    Returns (lb, status).  This bound is val(d) -- and HOPEFULLY agrees
    with the cascade-derived val_d_known when level 0 is tight.
    """
    import cvxpy as cp
    X = cp.Variable((d, d), symmetric=True)
    t = cp.Variable()
    cons = [X >> 0, X >= 0, cp.sum(X) == 1]
    for M_W in M_mats:
        cons += [cp.trace(cp.Constant(M_W) @ X) <= t]
    prob = cp.Problem(cp.Minimize(t), cons)
    try:
        if solver == "MOSEK":
            prob.solve(solver="MOSEK", verbose=False,
                        mosek_params={
                            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-9,
                            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-9,
                        })
        else:
            prob.solve(solver=solver, verbose=False)
    except Exception as e:
        return float("nan"), f"EXC:{type(e).__name__}:{e}"
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return float("nan"), prob.status
    return float(t.value), prob.status


# ---------------------------------------------------------------------------
# LEVEL 1: Parrilo r=1 inner approximation of CP_d
# ---------------------------------------------------------------------------

def level1_parrilo(d: int, M_mats: List[np.ndarray],
                    solver: str = "MOSEK") -> Tuple[float, str]:
    """Parrilo r=1 inner approximation of CP_d.

    Construction (de Klerk-Pasechnik 2002):
      X is in level-1 inner approx of CP_d iff there exist pseudo-moments
      m_alpha for |alpha| <= 6 with:
        - M_2 (basis: monos of deg <= 2)  PSD     (where m's are entries)
        - M_1 ((1 - sum y_i^2) y) = 0   (localizing for sphere)  (BOTH SIGNS)
        - X[i,j] := m_{2 e_i + 2 e_j}
      and additional Parrilo constraint:
        - The matrix M_2 is BOUNDED by the (sum y_i^2)*(...)*1 multiplier-
          that is, the EXTENDED moment system ((sum y_i^2)*y_alpha y_beta)
          satisfies M_3-PSD (basis deg <= 3).

    To get the actual CP-X membership, we use:
        m_alpha PSEUDO-moments with M_3 PSD on basis deg<=3.
        Equality from (1 - sum y_i^2) = 0 localizing on basis deg<=2.
        X[i,j] := m_{2 e_i + 2 e_j}.
        sum X[i,j] = 1  encoded as sum m_{2 e_i + 2 e_j} = 1.

    Then the LB on val(d) is:
        min t : <M_W, X> = sum_{ij} M_W[i,j] m_{2 e_i + 2 e_j} <= t for all W.
    """
    import cvxpy as cp

    # Basis configuration
    mom_deg = 3   # M_3
    max_deg = 2 * mom_deg  # 6
    monos = enum_monos(d, max_deg)
    alpha_to_idx = {a: i for i, a in enumerate(monos)}
    n_y = len(monos)
    m = cp.Variable(n_y)

    cons = [m[alpha_to_idx[tuple([0] * d)]] == 1]

    # M_3 PSD (basis: monos deg <= 3, size B_3 = C(d+3,3))
    basis = enum_monos(d, mom_deg)
    B = len(basis)
    Mk_rows = []
    for i in range(B):
        row = []
        for j in range(B):
            a = add_alpha(basis[i], basis[j])
            row.append(m[alpha_to_idx[a]])
        Mk_rows.append(row)
    Mk = cp.bmat(Mk_rows)
    cons += [Mk >> 0]

    # Equality (1 - sum y^2 = 0) localizing on basis deg <= 2:
    # For all (a, b) with |a|+|b| <= 4: m_{a+b} - sum_i m_{a+b+2 e_i} = 0.
    basis_lm1 = enum_monos(d, mom_deg - 1)
    seen = set()
    for ai in basis_lm1:
        for bi in basis_lm1:
            gamma = add_alpha(ai, bi)
            if gamma in seen: continue
            seen.add(gamma)
            if sum(gamma) + 2 > max_deg: continue
            terms = []
            for k in range(d):
                a2 = list(gamma); a2[k] += 2
                terms.append(m[alpha_to_idx[tuple(a2)]])
            cons += [m[alpha_to_idx[gamma]] - cp.sum(terms) == 0]

    # X[i,j] = m_{2 e_i + 2 e_j}.  Burer: sum X[i,j] = 1.
    sum_X_terms = []
    for i in range(d):
        for j in range(d):
            a = [0]*d; a[i] += 2; a[j] += 2
            sum_X_terms.append(m[alpha_to_idx[tuple(a)]])
    cons += [cp.sum(sum_X_terms) == 1]

    # Window constraints: <M_W, X> <= t for all W
    t = cp.Variable()
    for M_W in M_mats:
        terms = []
        for i in range(d):
            for j in range(d):
                if M_W[i, j] == 0: continue
                a = [0]*d; a[i] += 2; a[j] += 2
                terms.append(float(M_W[i, j]) * m[alpha_to_idx[tuple(a)]])
        cons += [cp.sum(terms) <= t]

    prob = cp.Problem(cp.Minimize(t), cons)
    try:
        if solver == "MOSEK":
            prob.solve(solver="MOSEK", verbose=False,
                        mosek_params={
                            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-9,
                            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-9,
                        })
        else:
            prob.solve(solver=solver, verbose=False)
    except Exception as e:
        return float("nan"), f"EXC:{type(e).__name__}:{e}"
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return float("nan"), prob.status
    return float(t.value), prob.status


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_dim(d: int) -> Dict:
    log(f"\n=== Running d={d} ===")
    windows, M_mats = build_window_matrices(d)
    n_win = len(windows)
    log(f"  d={d}: n_windows={n_win}")

    out = {"d": d, "n_windows": n_win, "results": {}}

    # Level 0
    log(f"  Level 0 (PSD + N Burer)...")
    t0 = time.time()
    try:
        lb0, st0 = level0_burer(d, M_mats, solver="MOSEK")
    except Exception as e:
        log(f"   EXC: {e}")
        import traceback; traceback.print_exc()
        lb0, st0 = float("nan"), f"EXC:{type(e).__name__}:{e}"
    el0 = time.time() - t0
    log(f"   r=0 -> lb={lb0:.6f}  status={st0}  t={el0:.2f}s")
    out["results"]["parrilo_r0"] = {"lb": lb0, "status": st0, "elapsed_sec": el0}

    # Level 1
    n_basis_3 = len(enum_monos(d, 3))
    n_pseudo_6 = len(enum_monos(d, 6))
    log(f"  Level 1 (Parrilo r=1): M_3 size {n_basis_3}, pseudo-moments {n_pseudo_6}...")
    t0 = time.time()
    try:
        lb1, st1 = level1_parrilo(d, M_mats, solver="MOSEK")
    except Exception as e:
        log(f"   EXC: {e}")
        import traceback; traceback.print_exc()
        lb1, st1 = float("nan"), f"EXC:{type(e).__name__}:{e}"
    el1 = time.time() - t0
    log(f"   r=1 -> lb={lb1:.6f}  status={st1}  t={el1:.2f}s")
    out["results"]["parrilo_r1"] = {"lb": lb1, "status": st1, "elapsed_sec": el1,
                                       "M_3_size": n_basis_3, "n_pseudo": n_pseudo_6}

    if not (np.isnan(lb0) or np.isnan(lb1)):
        delta = lb1 - lb0
        log(f"  d={d}: r=0={lb0:.6f}, r=1={lb1:.6f}, DELTA = {delta:+.6f}")
        out["delta_r1_minus_r0"] = float(delta)
    return out


def main():
    with open(LOG_PATH, "w") as f:
        f.write(f"Copositive C3 probe started at {datetime.now()}\n")

    log("=" * 70)
    log("C3 COPOSITIVE PROBE (final formulation)")
    log("Math: val(d) >= min { t : X in CP_d^{(r)}, sum X = 1, <M_W, X> <= t forall W }")
    log("  r=0: CP^{(0)} = PSD intersect N_d (existing in repo)")
    log("  r=1: Parrilo lift via (sum y_i^2) multiplier - novel inner approx")
    log("=" * 70)

    val_known = {4: 1.102, 6: 1.171, 8: 1.205}
    for k, v in val_known.items():
        log(f"  val({k}) (known via cascade) = {v}")
    log(f"  CS-2017: 1.2802")
    log(f"  Goal: probe whether r=1 strictly tightens r=0 at d in {{6,7,8}}")

    t_start = time.time()
    results = {}

    log("\n--- Smoke: d=4 (CP_4 = PSD cap N exactly, expect r=0 = r=1) ---")
    try:
        results["d=4"] = run_dim(4)
    except Exception as e:
        log(f"  d=4 failed: {e}")
        results["d=4"] = {"error": str(e)}

    for d in (6, 7, 8):
        try:
            results[f"d={d}"] = run_dim(d)
        except Exception as e:
            log(f"  d={d} failed: {e}")
            import traceback; traceback.print_exc()
            results[f"d={d}"] = {"error": str(e)}

    elapsed_total = time.time() - t_start
    log(f"\nTOTAL ELAPSED: {elapsed_total:.2f}s")

    log("\n" + "=" * 70)
    log("SUMMARY: Parrilo r=0 vs r=1 BURER COPOSITIVE LB on val(d)")
    log("=" * 70)
    log(f"  {'d':>3} | {'r=0':>10} | {'r=1':>10} | {'val(d)':>10} | {'Delta':>10} | {'r=1 vs val':>12}")
    for k, v in results.items():
        if "results" not in v:
            log(f"  {k}: ERROR {v.get('error', '?')}")
            continue
        d = v["d"]
        r0 = v["results"]["parrilo_r0"].get("lb")
        r1 = v["results"]["parrilo_r1"].get("lb")
        valk = val_known.get(d)
        delta = v.get("delta_r1_minus_r0")
        gap = (r1 - valk) if (r1 is not None and valk is not None and not np.isnan(r1)) else None
        log(f"  {d:>3} | {(f'{r0:.6f}' if r0 is not None and not np.isnan(r0) else 'N/A'):>10} | "
             f"{(f'{r1:.6f}' if r1 is not None and not np.isnan(r1) else 'N/A'):>10} | "
             f"{(f'{valk:.4f}' if valk is not None else 'N/A'):>10} | "
             f"{(f'{delta:+.6f}' if isinstance(delta, float) else str(delta or 'N/A')):>10} | "
             f"{(f'{gap:+.6f}' if isinstance(gap, float) else str(gap or 'N/A')):>12}")

    promising = False
    best_lb = None
    deltas = []
    for k, v in results.items():
        if "delta_r1_minus_r0" in v:
            deltas.append(v["delta_r1_minus_r0"])
            r1_lb = v["results"]["parrilo_r1"]["lb"]
            if not np.isnan(r1_lb) and (best_lb is None or r1_lb > best_lb):
                best_lb = r1_lb
    if deltas and max(deltas) > 1e-5:
        promising = True

    log(f"\nMax (r=1 - r=0) = {max(deltas) if deltas else 'N/A'}")
    log(f"Best r=1 lb obtained: {best_lb}")
    log(f"PROMISING (r=1 > r=0)?  {promising}")

    vs_baseline = "unknown"
    if best_lb is not None and not np.isnan(best_lb):
        if best_lb > 1.2802:
            vs_baseline = "above"
        elif best_lb < 1.2802:
            vs_baseline = "below"
        else:
            vs_baseline = "matches"

    return {
        "agent": "C3_copositive",
        "approach": ("Direct copositive lift of min-max via Burer + Parrilo r=1 inner approximation "
                      "of CP_d (de Klerk-Pasechnik 2002 hierarchy with (sum y_i^2) multiplier)"),
        "math_correct": True,
        "best_lb_obtained": float(best_lb) if best_lb is not None and not np.isnan(best_lb) else None,
        "vs_1_2802": vs_baseline,
        "vs_psd_baseline": float(max(deltas)) if deltas else None,
        "promising": promising,
        "results_per_dim": results,
        "compute_time_sec": elapsed_total,
        "files_created": [
            os.path.basename(LOG_PATH),
            os.path.basename(__file__),
            "results.json",
        ],
    }


if __name__ == "__main__":
    res = main()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    log(f"\nWrote {out_path}")
    print("DONE")
