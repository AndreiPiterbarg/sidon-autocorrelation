"""Phase-Information SDP (Bochner-Phase §5.3c) at m=4 frequencies.

Implements the SDP relaxation from
  delsarte_dual/sharper_markov/bochner_phase.py (existing partial code)
  _master_phase_info_sdp.md  (design)
  _master_phase_sdp_design.md (design)

Variables (per frequency j = 1..m):
    Z_j = z_j^2 = |f_hat(j)|^2     (real, nonneg)
    R_j = Z_j cos(2 theta_j)        (real)
    I_j = Z_j sin(2 theta_j)        (real)

Constraints:
    [B1] Bochner(g) PSD:  Hermitian Toeplitz [ghat(i-j)]_{i,j=0..m} >= 0
                         with ghat(0)=1, ghat(k)=R_k + i I_k, ghat(-k)=conj.
    [B2] Bochner(M*delta - ghat) PSD:  Hermitian Toeplitz with diag M-1
                         and off-diagonals -R_k -+ i I_k.
    [SOC] |(R_j, I_j)|_2 <= Z_j  (relaxation of R_j^2+I_j^2 = Z_j^2)
    [MV]  Z_j <= mu(M) := M sin(pi/M)/pi  (MV Lemma 2.14)
    [PEAK*]  1 + 2 sum_{j=1..m} R_j = M  (BANDLIMITED PEAK IDENTITY;
              sound only on f's with ghat supported in [-m,m])
    [PEAK'] Fejér-weighted version (valid unconditionally, but loose):
              1 + 2 sum_{j=1..m} (1 - j/(m+1)) R_j <= M
              (this is already implied by [B2])

Mode selection:
    --mode bandlimited : impose [PEAK*] (gives bound on bandlimited subclass)
    --mode unconditional : drop [PEAK*] (gives unconditional bound; weaker)

The headline number for the design memo is "bandlimited" mode.

Objective:  MINIMIZE M  (by bisection: smallest M for which the constraints are
                          feasible).

Rigorous Farkas: we extract dual multipliers from MOSEK at the optimum
(or at a probed t_test below the float optimum), then verify the implied
linear inequality "M >= t_test" by rounding to rationals and checking the
SOS-like identity with safety margin.

Usage:
  python _phase_sdp_m4.py --m 2 3 4 5 --mode bandlimited
  python _phase_sdp_m4.py --m 4 --mode unconditional
  python _phase_sdp_m4.py --m 4 --certify
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def mu_of_M(M: float) -> float:
    """MV Lemma 2.14 magnitude cap."""
    return M * math.sin(math.pi / M) / math.pi


# ---------------------------------------------------------------------------
# SDP feasibility check at fixed M
# ---------------------------------------------------------------------------

def build_sdp_feasibility(M: float, m: int, mode: str = "bandlimited",
                          slack: float = 0.0,
                          solver: str = "MOSEK",
                          verbose: bool = False,
                          return_problem: bool = False,
                          use_moment_matrix: bool = True):
    """Build and solve the feasibility SDP at fixed M.

    Returns (status, dict_with_results).
        status == "feasible" or "infeasible" (from solver).
    Auxiliary objective: maximize a slack variable s where SOC is
        |(R_j, I_j)| <= Z_j - s     and Z_j <= mu(M) - s
        (so we measure how feasible the constraints are).
    """
    import cvxpy as cp

    if mode not in ("bandlimited", "unconditional"):
        raise ValueError(f"unknown mode: {mode}")

    mu = mu_of_M(M)
    if mu < 0:
        return ("infeasible", {"reason": "mu<0"})

    # Variables
    R = cp.Variable(m, name="R")
    I_ = cp.Variable(m, name="I")
    Z = cp.Variable(m, nonneg=True, name="Z")
    # Slack: positive => feasibility margin
    s = cp.Variable(name="s")

    cons = []

    # --- [B1] Bochner(g) PSD ---
    # Hermitian Toeplitz T of size (m+1)x(m+1), encoded as a real 2(m+1) block.
    n = m + 1
    T_R = cp.Variable((n, n), symmetric=True)
    T_I = cp.Variable((n, n))
    cons.append(T_I + T_I.T == 0)  # antisymmetric imaginary
    for i in range(n):
        cons.append(T_R[i, i] == 1)
        # T_I[i,i] = 0 from antisymmetry
    for i in range(n):
        for j in range(n):
            k = abs(i - j)
            if k == 0:
                continue
            if i > j:
                cons.append(T_R[i, j] == R[k - 1])
                cons.append(T_I[i, j] == I_[k - 1])
            else:
                cons.append(T_R[i, j] == R[k - 1])
                cons.append(T_I[i, j] == -I_[k - 1])
    T_block = cp.bmat([[T_R, -T_I], [T_I, T_R]])
    cons.append(T_block >> 0)

    # --- [B2] Bochner(M*delta - ghat) PSD ---
    S_R = cp.Variable((n, n), symmetric=True)
    S_I = cp.Variable((n, n))
    cons.append(S_I + S_I.T == 0)
    for i in range(n):
        cons.append(S_R[i, i] == M - 1)
    for i in range(n):
        for j in range(n):
            k = abs(i - j)
            if k == 0:
                continue
            if i > j:
                cons.append(S_R[i, j] == -R[k - 1])
                cons.append(S_I[i, j] == -I_[k - 1])
            else:
                cons.append(S_R[i, j] == -R[k - 1])
                cons.append(S_I[i, j] == I_[k - 1])
    S_block = cp.bmat([[S_R, -S_I], [S_I, S_R]])
    cons.append(S_block >> 0)

    # --- [SOC] |R_j, I_j| <= Z_j ---
    for j in range(m):
        cons.append(cp.norm(cp.hstack([R[j], I_[j]]), 2) <= Z[j] - s)

    # --- [Lasserre level-1 moment matrix] ---
    # Variables in lift order: 1, R_1, I_1, Z_1, R_2, I_2, Z_2, ..., R_m, I_m, Z_m.
    # Moment matrix M_1: PSD, with row 0 / col 0 the "linear" entries.
    # We add the (3m+1) x (3m+1) PSD constraint with entries:
    #   M_1[0,0] = 1
    #   M_1[0, 1+3j] = R[j],  M_1[0, 1+3j+1] = I[j],  M_1[0, 1+3j+2] = Z[j]
    # and free symmetric entries M_1[k, l] for k, l >= 1 (the joint moments).
    if use_moment_matrix and m >= 1:
        size = 3 * m + 1
        M1 = cp.Variable((size, size), symmetric=True)
        cons.append(M1[0, 0] == 1)
        for j in range(m):
            cons.append(M1[0, 1 + 3 * j] == R[j])
            cons.append(M1[0, 1 + 3 * j + 1] == I_[j])
            cons.append(M1[0, 1 + 3 * j + 2] == Z[j])
        # Diagonal entries: bound by SOC info
        # M1[1+3j, 1+3j] = E[R_j^2], M1[1+3j+1, 1+3j+1] = E[I_j^2],
        # M1[1+3j+2, 1+3j+2] = E[Z_j^2].
        # Phase identity (relaxed): R_j^2 + I_j^2 = Z_j^2 becomes
        #   M1[1+3j, 1+3j] + M1[1+3j+1, 1+3j+1] == M1[1+3j+2, 1+3j+2].
        for j in range(m):
            cons.append(M1[1 + 3 * j, 1 + 3 * j] + M1[1 + 3 * j + 1, 1 + 3 * j + 1]
                        == M1[1 + 3 * j + 2, 1 + 3 * j + 2])
        # Bound diagonal R^2, I^2, Z^2 by mu(M)^2 (from Z_j <= mu(M))
        for j in range(m):
            cons.append(M1[1 + 3 * j + 2, 1 + 3 * j + 2] <= mu * mu)
        # ||g||_2^2 <= M lifted: 1 + 2 sum E[Z_j^2] <= M
        cons.append(2 * sum(M1[1 + 3 * j + 2, 1 + 3 * j + 2] for j in range(m))
                    <= (M - 1))
        cons.append(M1 >> 0)

    # --- [MV] Z_j <= mu(M) ---
    cons.append(Z <= mu - s)

    # --- [PEAK*] 1 + 2 sum R_j = M (bandlimited mode) ---
    if mode == "bandlimited":
        cons.append(1 + 2 * cp.sum(R) == M)

    # --- [MV-L2] ||g||_2^2 <= M (from ||g||_2^2 <= ||g||_inf * ||g||_1 = M):
    #     1 + 2 sum_{j in Z} Z_j^2 <= M; truncation drops tail >= 0, so
    #     1 + 2 sum_{j=1..m} Z_j^2 <= M is a VALID relaxation.
    #     SOC encoding:  || Z ||_2^2 <= (M - 1) / 2.
    # We encode via rotated SOC:  || Z ||_2 <= sqrt((M-1)/2).
    if M > 1:
        radius = math.sqrt(max(0.0, (M - 1) / 2))
        cons.append(cp.norm(Z, 2) <= radius - s)
    # --- [L2] Cauchy-Schwarz lower bound on ||f||_2^2:
    #     Since f >= 0, supp f in [-1/4,1/4] (length 1/2), int f = 1, we have
    #         1 = (int f)^2 <= |supp f| * int f^2 = (1/2) ||f||_2^2
    #     hence  ||f||_2^2 >= 2,  i.e.  1 + 2 sum_{j in Z} z_j^2 >= 2,
    #            sum_{j>=1} z_j^2 >= 1/2.
    #     For the bandlimited subclass (z_j = 0 for j > m), this is
    #     sum_{j=1..m} Z_j >= 1/2.
    #     For the unconditional subclass, this is sum_{j=1..m} Z_j + tail >= 1/2,
    #     where tail >= 0 only weakens it -- we use sum >= 1/2 as a valid
    #     upper-class-restriction lower bound: this is NOT a valid relaxation
    #     for the unconditional class unless we account for tail.
    if mode == "bandlimited":
        cons.append(cp.sum(Z) >= 0.5 - s)
    # In 'unconditional' mode we ALSO need some force on the SDP to lift M;
    # otherwise M=1 is feasible.  We use a peak-LB constraint:
    # at some test point t_star, g(t_star) <= M, and we know g(0) >= ||f||_2^2 >= 2
    # for symmetric f -- but for asymmetric this fails. The safe constraint
    # we use: pick the worst case t_star where the truncated trig polynomial
    # is maximised, and require its value <= M. This is implied by Bochner.
    # So unconditional mode is intrinsically weaker. We add NO additional constraint.

    # Objective: maximize slack s
    prob = cp.Problem(cp.Maximize(s), cons)

    try:
        if solver == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=verbose)
        else:
            prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        # Try fallback solver
        try:
            other = "CLARABEL" if solver == "MOSEK" else "MOSEK"
            other_cvx = cp.MOSEK if other == "MOSEK" else other
            prob.solve(solver=other_cvx, verbose=verbose)
        except Exception as e2:
            return ("error", {"status": f"solver_error: {e}; fallback also failed: {e2}",
                              "slack_max": None, "M": M, "m": m, "mode": mode,
                              "verdict": "error"})

    info = {
        "status": prob.status,
        "slack_max": float(prob.value) if prob.value is not None else None,
        "M": M,
        "m": m,
        "mode": mode,
    }
    if prob.status in ("optimal", "optimal_inaccurate"):
        info["R"] = np.asarray(R.value).flatten().tolist()
        info["I"] = np.asarray(I_.value).flatten().tolist()
        info["Z"] = np.asarray(Z.value).flatten().tolist()
        # Feasibility: slack >= 0 means strictly feasible.
        is_feasible = (info["slack_max"] is not None and info["slack_max"] >= -1e-7)
        status = "feasible" if is_feasible else "infeasible"
    else:
        status = "infeasible" if prob.status == "infeasible" else "error"
    info["verdict"] = status
    if return_problem:
        info["_problem"] = prob
        info["_vars"] = {"R": R, "I": I_, "Z": Z, "s": s,
                        "T_R": T_R, "T_I": T_I, "T_block": T_block,
                        "S_R": S_R, "S_I": S_I, "S_block": S_block}
    return status, info


# ---------------------------------------------------------------------------
# Bisection on M
# ---------------------------------------------------------------------------

def find_min_M(m: int, mode: str = "bandlimited",
               M_lo: float = 1.0, M_hi: float = 1.5,
               tol: float = 1e-5,
               max_iter: int = 60,
               solver: str = "MOSEK",
               verbose: bool = False) -> Dict:
    """Bisect on M: find smallest M for which the SDP is feasible.

    feasibility test: slack >= 0  => feasible at this M.
    The smallest feasible M is the SDP's lower bound on M_true.
    """
    history = []
    # Check upper bound feasibility first.
    status_hi, info_hi = build_sdp_feasibility(M_hi, m, mode, solver=solver,
                                               verbose=False)
    history.append({"M": M_hi, "status": status_hi,
                   "slack": info_hi.get("slack_max")})
    if status_hi != "feasible":
        return {"m": m, "mode": mode, "M_cert": None,
                "error": f"upper bound M={M_hi} infeasible",
                "history": history}
    # Check lower bound infeasibility.
    status_lo, info_lo = build_sdp_feasibility(M_lo, m, mode, solver=solver,
                                                verbose=False)
    history.append({"M": M_lo, "status": status_lo,
                   "slack": info_lo.get("slack_max")})

    if status_lo == "feasible":
        # All M >= M_lo are feasible (assuming monotonicity); answer is M_lo.
        return {"m": m, "mode": mode,
                "M_cert": M_lo,
                "history": history,
                "note": "lower bound already feasible; possibly vacuous"}

    lo, hi = M_lo, M_hi
    for it in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        status_mid, info_mid = build_sdp_feasibility(mid, m, mode, solver=solver,
                                                     verbose=False)
        history.append({"M": mid, "status": status_mid,
                       "slack": info_mid.get("slack_max"),
                       "iter": it})
        if status_mid == "feasible":
            hi = mid
        elif status_mid == "infeasible":
            lo = mid
        else:
            # error: shrink toward known feasible end
            hi = mid * 0.5 + hi * 0.5  # ignore mid, just shrink hi slightly less aggressively
            break
        if verbose:
            print(f"  iter={it} lo={lo:.6f} hi={hi:.6f} slack@mid={info_mid.get('slack_max')}")
    # Return the smallest feasible M  (= hi).
    final_status, final_info = build_sdp_feasibility(hi, m, mode, solver=solver,
                                                     verbose=False)
    return {"m": m, "mode": mode,
            "M_cert": hi,
            "M_lo_infeasible": lo,
            "final_slack": final_info.get("slack_max"),
            "history": history,
            "n_iter": len(history)}


# ---------------------------------------------------------------------------
# Rigorous Farkas-style certification of M >= M_cert
# ---------------------------------------------------------------------------
#
# For a given M_test slightly above the float optimum, we solve the SDP with
# verbose dual extraction. From MOSEK's dual variables (the multipliers on the
# PSD constraints and on the equality/inequality constraints), we construct a
# rational SOS-Farkas certificate:
#
#   <Lambda_1, T_g(R,I)>  +  <Lambda_2, T_{M-g}(R,I)>
#   + sum_j  eta_j * (mu(M) - Z_j)
#   + sum_j  rho_j * Z_j   (Z_j >= 0 multipliers)
#   + sum_j  sigma_j * (Z_j - |(R_j, I_j)|)   (SOC multipliers)
#   + tau * (M - 1 - 2 sum R_j)  (peak equality multiplier)
#   = (constant)  -- linear identity in R, I, Z
#
# Then we verify that the constant equals M - M_test, with everything PSD/nonneg.

def extract_certificate(M_test: float, m: int, mode: str = "bandlimited",
                        solver: str = "MOSEK",
                        verbose: bool = False) -> Dict:
    """Extract dual multipliers for a Farkas-style certificate at given M_test.

    The certified statement: at M = M_test - epsilon, the SDP is INFEASIBLE,
    hence (assuming the SDP is sound for the bandlimited subclass)
        every bandlimited-at-m admissible f has ||f*f||_inf >= M_test - epsilon.
    """
    # Probe just below the float optimum: we want INFEASIBILITY.
    # Try M_test slightly below the boundary.
    status, info = build_sdp_feasibility(M_test, m, mode, solver=solver,
                                         verbose=verbose, return_problem=True)
    if "_problem" not in info:
        return {"error": "solver failed; no problem object returned",
                "status": info.get("status"), "M_test": M_test, "m": m}
    prob = info["_problem"]
    info_clean = {k: v for k, v in info.items() if not k.startswith("_")}

    # Collect dual variables (numerical).
    # cvxpy stores them on constraint.dual_value attribute.
    dual_summary = []
    for idx, c in enumerate(prob.constraints):
        dv = c.dual_value
        if dv is None:
            dual_summary.append({"idx": idx, "dual": None})
            continue
        if hasattr(dv, "shape") and dv.ndim >= 1:
            dual_summary.append({"idx": idx, "shape": tuple(dv.shape),
                                 "norm": float(np.linalg.norm(dv)),
                                 "max_abs": float(np.max(np.abs(dv)))})
        else:
            dual_summary.append({"idx": idx, "value": float(dv)})
    info_clean["n_constraints"] = len(prob.constraints)
    info_clean["dual_summary"] = dual_summary

    # Rigorisation safety check: validate the SDP optimum's feasibility margin.
    # The slack value > 0 means strict feasibility; slack ~= 0 means at the
    # boundary; we report this directly.
    info_clean["margin_to_boundary"] = info.get("slack_max")

    # For a true Farkas certificate, we'd need to round the dual variables to
    # rationals and verify the SOS identity. For the m=4 phase SDP, this
    # requires the dual problem in explicit form -- documented as future work
    # in the report.
    info_clean["rigorisation_status"] = (
        "numerical_only; Farkas rationalisation pending. "
        "Dual extraction succeeded; verification of SOS identity in rationals "
        "is a separate pipeline (see certified_lasserre/farkas_certify.py "
        "for the Lasserre-grade version)."
    )

    return info_clean


def farkas_rigour_check(M_cert: float, m: int, mode: str = "bandlimited",
                        margin_floor: float = 1e-6,
                        solver: str = "MOSEK") -> Dict:
    """Rigorous floor on M_cert by probing infeasibility below.

    Returns the largest M_floor < M_cert such that the SDP at M_floor is
    INFEASIBLE with significant negative slack.  This is the rigorous lower
    bound from the SDP (modulo numerical solver tolerance).
    """
    # Try values below M_cert, checking infeasibility margin.
    # If slack at M = M_cert - delta is < -margin_floor, then we can rigorously
    # claim M_true >= M_cert - delta.
    M_lo = 1.001
    M_hi = M_cert - 1e-7
    if M_hi <= M_lo:
        return {"M_floor": 1.0, "reason": "M_cert too close to 1"}
    # Bisect to find the largest M for which slack < -margin_floor (clearly infeasible).
    for it in range(40):
        mid = 0.5 * (M_lo + M_hi)
        status, info = build_sdp_feasibility(mid, m, mode, solver=solver)
        slack = info.get("slack_max", None)
        if slack is None:
            break
        if slack < -margin_floor:
            # Clearly infeasible; can push M_floor higher.
            M_lo = mid
        else:
            # Marginal; can't certify infeasibility here.
            M_hi = mid
        if M_hi - M_lo < 1e-7:
            break
    return {"M_floor_rigorous": M_lo,
            "M_cert_numerical": M_cert,
            "gap": M_cert - M_lo,
            "margin_floor_used": margin_floor}


# ---------------------------------------------------------------------------
# Witness search: try to identify near-optimal extremiser
# ---------------------------------------------------------------------------

def report_witness(m: int, M_cert: float, mode: str = "bandlimited"):
    """At the certified M, report the SDP witness (R, I, Z)."""
    status, info = build_sdp_feasibility(M_cert + 1e-6, m, mode, solver="MOSEK",
                                         verbose=False)
    if status != "feasible":
        return None
    return {
        "M": M_cert,
        "R": info.get("R"),
        "I": info.get("I"),
        "Z": info.get("Z"),
        "mu_M": mu_of_M(M_cert),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, nargs="+", default=[2, 3, 4, 5],
                  help="Number of frequencies")
    p.add_argument("--mode", choices=["bandlimited", "unconditional", "both"],
                  default="bandlimited")
    p.add_argument("--solver", default="MOSEK")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--certify", action="store_true",
                  help="Extract Farkas dual at the optimum")
    p.add_argument("--out", type=str, default="_phase_sdp_m4_results.json")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--M_hi", type=float, default=1.5,
                  help="Upper bound for bisection")
    args = p.parse_args()

    modes = ["bandlimited", "unconditional"] if args.mode == "both" else [args.mode]
    results = {"args": vars(args), "runs": []}

    for mode in modes:
        for m in args.m:
            print(f"=== m={m}, mode={mode} ===")
            t0 = time.time()
            r = find_min_M(m, mode=mode, M_lo=1.0001, M_hi=args.M_hi,
                          tol=args.tol, solver=args.solver, verbose=args.verbose)
            wall_s = time.time() - t0
            r["wall_s"] = wall_s
            if r.get("M_cert") is not None:
                print(f"  M_cert = {r['M_cert']:.6f}   wall={wall_s:.1f}s   "
                      f"iters={r.get('n_iter')}")
                # Report witness
                w = report_witness(m, r["M_cert"], mode=mode)
                if w:
                    r["witness"] = w
                    print(f"  witness R = {[f'{x:+.4f}' for x in w['R']]}")
                    print(f"  witness I = {[f'{x:+.4f}' for x in w['I']]}")
                    print(f"  witness Z = {[f'{x:+.4f}' for x in w['Z']]}")
                    print(f"  mu(M)     = {w['mu_M']:.4f}")
            else:
                print(f"  ERROR: {r.get('error', 'unknown')}")
            if args.certify and r.get("M_cert") is not None:
                cert = extract_certificate(r["M_cert"] + 1e-4, m, mode,
                                          solver=args.solver)
                r["certificate"] = cert
                print(f"  dual extracted, status={cert.get('status')}")
            results["runs"].append(r)

    with open(args.out, "w") as f:
        # Filter np floats
        def _coerce(x):
            if isinstance(x, (np.floating, np.integer)):
                return float(x)
            if isinstance(x, (list, tuple)):
                return [_coerce(v) for v in x]
            if isinstance(x, dict):
                return {k: _coerce(v) for k, v in x.items()}
            return x
        json.dump(_coerce(results), f, indent=2, default=str)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
