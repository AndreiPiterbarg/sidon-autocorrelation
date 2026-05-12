"""z_1 (and higher z_n) refinement of the MV master applied to K26 multi-scale arcsine.

Background
----------
The existing ``mv_master_M_cert(k_1, K_2, S_1)`` in ``_kernel_probe_helper.py``
ALREADY implements MV eq. (10): it computes ``mu_ = M sin(pi/M)/pi`` (Lemma 3.4
cap on |hat_f(1)|^2 = z_1^2), uses the unconstrained saddle ``y_star^2 =
k_1^2 (M-1)/(K_2-1)``, and picks whichever active branch (cap-saturated or
unconstrained sqrt) is appropriate.  Hence K26's published ``1.28013`` is
ALREADY a z_1-active value.  This script verifies that and explores
higher-frequency refinements (k_2, k_3, ...) per MV's open direction #3
(MV p. 7 lines 356-380).

Generalized master (extending MV eq. (9)/(10) to multiple harmonics)
--------------------------------------------------------------------
For z_n := |hat_f(n)|, n>=1, the period-1 Parseval decomposition gives

  R(M) := int(f*f) K dx + ||f*f||_inf
        = M + 1 + 2 sum_{n>=1} z_n^2 k_n + (tail)

with k_n = hat_K(n) >= 0 and z_n^2 <= mu(M) := M sin(pi/M) / pi  for ALL n
(verified by re-deriving Lemma 3.4 with cos(2 pi n x): the optimal h is
M * 1_{cos(2 pi n x) >= cos(pi/M)}, and the level-set integral is
n * sin(pi/M) / (pi n) = sin(pi/M)/pi, independent of n).

Cauchy-Schwarz on the j>=m+1 tail (using sum_j hat_f(j)^4 = ||f*f||_2^2 <= M
and sum_j hat_K(j)^2 = K_2):

  tail_sum_{n} = sum_{|j|>=m+1} z_j^2 k_j
              <= sqrt( M - 1 - 2 sum_{n=1..m} z_n^4 )
                * sqrt( K_2 - 1 - 2 sum_{n=1..m} k_n^2 ).

So R <= M + 1 + 2 sum_{n=1..m} z_n^2 k_n
        + sqrt( (M-1-2*sum z_n^4)(K_2-1-2*sum k_n^2) ).

The master inequality 2/u + a <= R becomes a constraint that bounds M from
below.  We optimize over z_n^2 in [0, mu(M)] to MAXIMIZE R (giving the
strongest lower bound on M).

This script:
  1. Verifies K26's 1.28013 already uses z_1 (eq. 10).
  2. Computes higher Fourier coeffs k_2, k_3, ... of the K26 best kernel.
  3. Applies the m=1, m=2, m=3 refined masters and reports M_cert.
  4. Sweeps (delta_2, lambda_1) under the m-refined master to see whether
     the optimal pair shifts.
  5. Saves results to ``_master_k26_z1_results.json``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import j0

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import DELTA, U, mv_master_M_cert  # noqa: E402
from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq  # noqa: E402


MV_COEFFS = np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])
N_QP = 119

# Xi grid for K_2 = int K_hat^2 d xi
N_XI_DEFAULT = 40001
XI_MAX_DEFAULT = 600.0


# ---------------------------------------------------------------------------
# K_hat helpers for multi-scale arcsine kernels
# ---------------------------------------------------------------------------
def K_hat_multiscale(xi: np.ndarray, deltas, lambdas) -> np.ndarray:
    """K_hat(xi) = sum_i lambdas[i] * J_0(pi * deltas[i] * xi)^2 >= 0."""
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * xi) ** 2
    return out


def K_quantities(deltas, lambdas, n_max: int = 5,
                 n_xi: int = N_XI_DEFAULT,
                 xi_max: float = XI_MAX_DEFAULT) -> dict:
    """Compute k_1..k_{n_max}, K_2, S_1 for the multi-scale kernel."""
    deltas = np.asarray(deltas, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    assert np.isclose(lambdas.sum(), 1.0), f"lambdas sum to {lambdas.sum()}"

    # k_n = K_hat(n) for n = 1..n_max  (period-1)
    ns = np.arange(1, n_max + 1)
    k_n = K_hat_multiscale(ns, deltas, lambdas)

    # K_2 = int K_hat(xi)^2 d xi  (Parseval)
    xi = np.linspace(0.0, xi_max, n_xi)
    dxi = xi[1] - xi[0]
    Kh = K_hat_multiscale(xi, deltas, lambdas)
    K_2 = float(2.0 * np.trapezoid(Kh ** 2, dx=dxi))

    # S_1 = sum a_j^2 / K_hat(j/u)
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = K_hat_multiscale(qp_xi, deltas, lambdas)
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))

    return {
        "deltas": deltas.tolist(),
        "lambdas": lambdas.tolist(),
        "k_n": k_n.tolist(),  # k_1, k_2, ..., k_{n_max}
        "K_2": K_2,
        "S_1": S_1,
    }


# ---------------------------------------------------------------------------
# Master inequality solvers
# ---------------------------------------------------------------------------
def M_cert_eq7_only(k_1: float, K_2: float, S_1: float) -> float | None:
    """MV eq. (7): no z_1 lift.   M + 1 + sqrt((M-1)(K_2-1)) >= 2/u + a."""
    a = (4.0 / U) / S_1
    target = 2.0 / U + a
    if K_2 <= 1.0:
        return None

    def f(M):
        return M + 1 + np.sqrt((M - 1) * (K_2 - 1)) - target

    try:
        if f(1.0 + 1e-10) >= 0 or f(2.0) <= 0:
            return None
        return brentq(f, 1.0 + 1e-10, 2.0, xtol=1e-10)
    except Exception:
        return None


def M_cert_z_refined(k_list: list, K_2: float, S_1: float,
                     m: int = 1, branch: str = "mv") -> tuple:
    """Generalized eq.(10): refine with first ``m`` harmonics.

    MV's eq.(10) uses ``z_n^2 = mu(M)`` (the Lemma 3.4 saturated boundary).
    MV substitutes z_1 = sqrt(mu) (the boundary) into

        R_upper(M, z_1) = M + 1 + 2 z_1^2 k_1 + sqrt((M-1-2 z_1^4)(K_2-1-2 k_1^2))

    and solves R_upper = target for M (l(z_1) is, per MV p.7 line 349, monotone
    decreasing in z_1 on [0, sqrt(mu(M*))] so the boundary z_1 = sqrt(mu) gives
    the tightest LB).  Generalized to m harmonics: substitute z_n^2 = mu(M)
    for all n=1..m (cap-saturated) at the boundary and solve.

    Parameters
    ----------
    branch : "mv" (cap-saturate per MV, matches helper) or "saddle" (interior
        max which equals eq.(7) when unconstrained-feasible).
    """
    assert m >= 0
    if m == 0:
        M = M_cert_eq7_only(k_list[0] if k_list else 0.0, K_2, S_1)
        return M, {"branch": "eq7-no-z", "z_star": []}

    k = np.array(k_list[:m], dtype=float)  # k_1..k_m
    sum_k2 = float(np.sum(k ** 2))
    rad2_const = K_2 - 1 - 2 * sum_k2
    if rad2_const <= 0:
        return None, {"branch": "infeasible-K_2-too-small"}

    a = (4.0 / U) / S_1
    target = 2.0 / U + a

    def R_at_cap(M: float) -> tuple[float, np.ndarray]:
        """R with z_n^2 = mu(M) for all n=1..m (MV-style boundary)."""
        if M <= 1.0:
            return float("-inf"), np.zeros(m)
        mu = M * np.sin(np.pi / M) / np.pi
        if mu <= 0:
            return float("-inf"), np.zeros(m)
        y = np.full(m, mu)
        rad1 = M - 1 - 2 * np.sum(y ** 2)
        if rad1 < 0:
            return float("-inf"), y
        R = M + 1 + 2.0 * np.sum(y * k) + np.sqrt(rad1 * rad2_const)
        return float(R), y

    def R_at_saddle(M: float) -> tuple[float, np.ndarray]:
        """R at the interior saddle (= eq.(7) value when unconstrained-feasible)."""
        if M <= 1.0:
            return float("-inf"), np.zeros(m)
        mu = M * np.sin(np.pi / M) / np.pi
        # Multi-coord unconstrained: y_n* = k_n sqrt((M-1)/(K_2-1))
        y_unc = k * np.sqrt((M - 1) / (K_2 - 1))
        if np.all(y_unc <= mu):
            # All interior feasible
            return float(M + 1 + np.sqrt((M - 1) * (K_2 - 1))), y_unc
        # Else cap some; numerical opt
        def neg_R(y):
            rad1 = M - 1 - 2 * np.sum(y ** 2)
            if rad1 < 0:
                return 1e10
            return -(2.0 * np.sum(y * k) + np.sqrt(rad1 * rad2_const))
        res = minimize(neg_R, np.minimum(y_unc, mu), method="L-BFGS-B",
                       bounds=[(0.0, mu)] * m,
                       options={"ftol": 1e-12, "gtol": 1e-10})
        return float(M + 1 - res.fun), res.x

    R_fn = R_at_cap if branch == "mv" else R_at_saddle

    def f(M):
        R, _ = R_fn(M)
        return R - target

    try:
        f_lo = f(1.0 + 1e-8)
        f_hi = f(2.0)
        if f_lo >= 0 or f_hi <= 0:
            return None, {"branch": "no-bracket"}
        M = brentq(f, 1.0 + 1e-8, 2.0, xtol=1e-10)
    except Exception as e:
        return None, {"branch": f"brentq-fail-{e}"}

    R, y_star = R_fn(M)
    return float(M), {
        "branch": branch,
        "m": m,
        "z_star_sq": y_star.tolist(),
        "mu_at_M": float(M * np.sin(np.pi / M) / np.pi),
    }


# ---------------------------------------------------------------------------
# Sweeping
# ---------------------------------------------------------------------------
def sweep_two_component(delta_1, delta_2_list, lambda_1_list, m: int,
                        branch: str = "mv"):
    results = []
    best = {"M_cert": -np.inf, "delta_2": None, "lambda_1": None}
    for d2 in delta_2_list:
        for lam in lambda_1_list:
            q = K_quantities([delta_1, d2], [lam, 1.0 - lam], n_max=m + 2)
            M, info = M_cert_z_refined(q["k_n"], q["K_2"], q["S_1"],
                                       m=m, branch=branch)
            rec = {
                "delta_1": float(delta_1),
                "delta_2": float(d2),
                "lambda_1": float(lam),
                "k_1": q["k_n"][0],
                "k_2": q["k_n"][1] if len(q["k_n"]) > 1 else None,
                "K_2": q["K_2"],
                "S_1": q["S_1"],
                "M_cert": M,
            }
            results.append(rec)
            if M is not None and M > best["M_cert"]:
                best = {"M_cert": float(M),
                        "delta_2": float(d2),
                        "lambda_1": float(lam),
                        "info": info}
    return results, best


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("z_1 (and higher) refinement of the K26 multi-scale arcsine master")
    print("=" * 78)

    out = {"theory": {}, "K26_best": {}, "higher_m": {}, "sweep_m1": {}}

    # ---- Theory note: Lemma 3.4 cap for general n ----
    note = (
        "Lemma 3.4 generalized: the level-set construction for max |hat_h(n)| "
        "(h>=0, h<=M, supp [-1/2,1/2], int h=1) gives mu(M) = M sin(pi/M)/pi "
        "for ALL integer n>=1, because cos(2 pi n x) tiles n periods in "
        "[-1/2,1/2] and the level-set integral per period sums to sin(pi/M)/pi."
    )
    out["theory"]["note"] = note
    print(note)

    # ---- Sanity: pure MV arcsine ----
    print("\n--- Pure MV arcsine: eq.(7) vs eq.(10) ---")
    q_pure = K_quantities([DELTA, 0.0], [1.0, 0.0], n_max=5)
    M_pure_eq7 = M_cert_eq7_only(q_pure["k_n"][0], q_pure["K_2"], q_pure["S_1"])
    M_pure_eq10, info_pure = M_cert_z_refined(q_pure["k_n"], q_pure["K_2"],
                                              q_pure["S_1"], m=1)
    M_pure_existing = mv_master_M_cert(q_pure["k_n"][0], q_pure["K_2"],
                                       q_pure["S_1"])
    print(f"  Pure arcsine:")
    print(f"    eq.(7)  no z_1     : M_cert = {M_pure_eq7:.6f}")
    print(f"    eq.(10) m=1 (z_1)  : M_cert = {M_pure_eq10:.6f}  (existing: {M_pure_existing:.6f})")
    print(f"    z_1 lift           : +{M_pure_eq10 - M_pure_eq7:.6f}")
    out["theory"]["pure_arcsine"] = {
        "M_eq7": M_pure_eq7,
        "M_eq10_m1": M_pure_eq10,
        "M_existing_helper": M_pure_existing,
        "z1_lift": M_pure_eq10 - M_pure_eq7,
        "k_n": q_pure["k_n"], "K_2": q_pure["K_2"], "S_1": q_pure["S_1"],
    }

    # ---- K26 best two-component (delta_2=0.055, lambda_1=0.9312) ----
    print("\n--- K26 best two-component: delta_2=0.055, lambda_1=0.9312 ---")
    K26_DELTAS = [0.138, 0.055]
    K26_LAMBDAS = [0.9312, 0.0688]
    q = K_quantities(K26_DELTAS, K26_LAMBDAS, n_max=5)
    print(f"  k_1, k_2, k_3, k_4, k_5 = "
          f"{q['k_n'][0]:.5f}, {q['k_n'][1]:.5f}, {q['k_n'][2]:.5f}, "
          f"{q['k_n'][3]:.5f}, {q['k_n'][4]:.5f}")
    print(f"  K_2 = {q['K_2']:.5f}   S_1 = {q['S_1']:.4f}")

    M_eq7 = M_cert_eq7_only(q["k_n"][0], q["K_2"], q["S_1"])
    M_existing = mv_master_M_cert(q["k_n"][0], q["K_2"], q["S_1"])
    print(f"  eq.(7) no z_1                  : M_cert = {M_eq7:.6f}")
    print(f"  existing mv_master_M_cert helper: M_cert = {M_existing:.6f}")
    z1_lift_helper = M_existing - M_eq7
    print(f"  -> existing helper z_1 lift    : +{z1_lift_helper:.6f}")
    out["K26_best"]["deltas"] = K26_DELTAS
    out["K26_best"]["lambdas"] = K26_LAMBDAS
    out["K26_best"]["k_n"] = q["k_n"]
    out["K26_best"]["K_2"] = q["K_2"]
    out["K26_best"]["S_1"] = q["S_1"]
    out["K26_best"]["M_eq7_no_z"] = M_eq7
    out["K26_best"]["M_existing_helper"] = M_existing

    print("\n  -- Branch 'mv' (cap-saturate per MV paper; matches helper) --")
    M_list_mv = {}
    z_star_list_mv = {}
    for m in [1, 2, 3, 4, 5]:
        M, info = M_cert_z_refined(q["k_n"], q["K_2"], q["S_1"], m=m, branch="mv")
        M_list_mv[m] = M
        z_star_list_mv[m] = info.get("z_star_sq") if M is not None else None
        if M is None:
            print(f"  m={m}  INFEASIBLE (rad1 = M-1-2*m*mu^2 < 0); reason: {info}")
        else:
            print(f"  m={m}  M_cert = {M:.6f}   z*^2 = {z_star_list_mv[m]}")
    out["K26_best"]["M_by_m_mv"] = {str(k): v for k, v in M_list_mv.items()}
    out["K26_best"]["z_star_sq_by_m_mv"] = {str(k): v for k, v in z_star_list_mv.items()}

    print("\n  -- Branch 'saddle' (interior max; rigorous concave-saddle) --")
    M_list_sd = {}
    for m in [1, 2, 3, 4, 5]:
        M, info = M_cert_z_refined(q["k_n"], q["K_2"], q["S_1"], m=m, branch="saddle")
        M_list_sd[m] = M
        if M is None:
            print(f"  m={m}  None ({info})")
        else:
            print(f"  m={m}  M_cert = {M:.6f}")
    out["K26_best"]["M_by_m_saddle"] = {str(k): v for k, v in M_list_sd.items()}

    # Use MV branch for sequence/sweep (to match helper)
    M_list = M_list_mv
    z_star_list = z_star_list_mv

    print(f"\n  Summary (K26 best, MV-cap branch):")
    print(f"    no z       : {M_eq7:.6f}")
    print(f"    m=1 (z_1)  : {M_list[1]:.6f}  (+{M_list[1]-M_eq7:.6f})")
    for mm in [2, 3, 4, 5]:
        v = M_list.get(mm)
        if v is None:
            print(f"    m={mm}        : INFEASIBLE under MV-cap (rad1 < 0)")
        else:
            ref = M_list.get(mm-1) or M_list[1]
            print(f"    m={mm}        : {v:.6f}  (+{v-ref:.6f})")

    # ---- (delta_2, lambda_1) sweep under m=1 z_1-refined master ----
    print("\n--- Sweep (delta_2, lambda_1) under m=1 master ---")
    d2_list = list(np.round(np.linspace(0.04, 0.075, 8), 5))
    l1_list = list(np.round(np.linspace(0.88, 0.97, 10), 5))
    sweep_results, sweep_best = sweep_two_component(DELTA, d2_list, l1_list, m=1)
    print(f"  best (m=1 sweep): M_cert={sweep_best['M_cert']:.6f} at "
          f"delta_2={sweep_best['delta_2']}, lambda_1={sweep_best['lambda_1']}")
    out["sweep_m1"]["d2_list"] = d2_list
    out["sweep_m1"]["l1_list"] = l1_list
    out["sweep_m1"]["best"] = sweep_best
    out["sweep_m1"]["results"] = sweep_results

    # ---- (delta_2, lambda_1) sweep under m=3 master (test optimum shift) ----
    print("\n--- Sweep (delta_2, lambda_1) under m=3 master ---")
    sweep_results_m3, sweep_best_m3 = sweep_two_component(
        DELTA, d2_list, l1_list, m=3
    )
    print(f"  best (m=3 sweep): M_cert={sweep_best_m3['M_cert']:.6f} at "
          f"delta_2={sweep_best_m3['delta_2']}, lambda_1={sweep_best_m3['lambda_1']}")
    out["sweep_m3"] = {
        "d2_list": d2_list,
        "l1_list": l1_list,
        "best": sweep_best_m3,
        "results": sweep_results_m3,
    }

    # ---- Sequence limit analysis (MV-style cap-sat) ----
    print("\n--- Sequence convergence MV-branch (M as function of m) ---")
    M_seq = []
    for m in [0, 1, 2, 3, 4, 5, 7, 10]:
        if m == 0:
            M = M_eq7
        else:
            q_big = K_quantities(K26_DELTAS, K26_LAMBDAS, n_max=max(m, 1) + 2)
            M, _ = M_cert_z_refined(q_big["k_n"], q_big["K_2"], q_big["S_1"],
                                    m=m, branch="mv")
        M_seq.append((m, M))
        print(f"  m={m:>2}: M = {M}")
    out["higher_m"]["sequence_mv"] = [(int(m), v) for m, v in M_seq]

    # ---- Optimal z_n^2 (not at boundary; sub-cap) for MV-branch --------------
    # An honest treatment: max R over z_n^2 in [0, mu(M)] subject to
    # M - 1 - 2 sum z_n^4 >= 0.  We try this directly to see if any setting of
    # z_n's gives more than the m=1 cap result.
    print("\n--- Direct numerical optimization of R over (z_1^2,...,z_m^2) ---")
    M_test = M_existing  # evaluate at K26 best
    mu_t = M_test * np.sin(np.pi / M_test) / np.pi
    print(f"  Evaluating R at M = {M_test:.6f}, mu = {mu_t:.6f}")
    target = 2.0 / U + (4.0 / U) / q["S_1"]
    print(f"  target = 2/u + a = {target:.6f}")
    direct_out = {}
    for m in [1, 2, 3, 5]:
        k = np.array(q["k_n"][:m])
        sum_k2 = float(np.sum(k**2))
        rad2c = q["K_2"] - 1 - 2 * sum_k2
        if rad2c <= 0:
            print(f"  m={m}: rad2 < 0, infeasible")
            direct_out[m] = None
            continue
        def neg_R(y):
            rad1 = M_test - 1 - 2 * np.sum(y**2)
            if rad1 < 0:
                return 1e10
            return -(2.0 * np.sum(y * k) + np.sqrt(rad1 * rad2c))
        from scipy.optimize import minimize as _min
        # multi-start
        best_R = -np.inf
        best_y = None
        for y0 in [np.zeros(m), np.full(m, mu_t * 0.5), np.full(m, mu_t * 0.9)]:
            try:
                r = _min(neg_R, y0, method="L-BFGS-B",
                         bounds=[(0.0, mu_t)] * m,
                         options={"ftol": 1e-13, "gtol": 1e-11})
                R = M_test + 1 - r.fun
                if R > best_R:
                    best_R = R
                    best_y = r.x
            except Exception:
                pass
        print(f"  m={m}: max R = {best_R:.6f}  (target {target:.6f}, diff {best_R-target:+.6f}), y*={best_y}")
        direct_out[m] = {"max_R_at_M_test": float(best_R), "y_star_sq": best_y.tolist() if best_y is not None else None}
    out["direct_R_max_at_K26_M"] = {str(k): v for k, v in direct_out.items()}

    # ---- Save ----
    outpath = os.path.join(REPO, "_master_k26_z1_results.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nWrote {outpath}")
    print("=" * 78)
    print(f"  K26 best (existing helper): {M_existing:.6f}")
    print(f"  K26 best with explicit z_1: {M_list[1]:.6f}")
    print(f"  Diff: {M_list[1] - M_existing:.2e}")
    m5 = M_list.get(5)
    if m5 is None:
        print(f"  K26 best with z_1..z_5    : INFEASIBLE (rad1<0 for m>=2 cap-saturated)")
    else:
        print(f"  K26 best with z_1..z_5    : {m5:.6f}")
        print(f"  Total higher-m lift over m=1: {m5 - M_list[1]:.6f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
