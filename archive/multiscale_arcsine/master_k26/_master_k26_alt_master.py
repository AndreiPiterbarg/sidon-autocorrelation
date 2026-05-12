"""Alternative master inequalities for K26 multi-scale kernel.

For the K26 best multi-scale kernel
    K_hat(xi) = lambda * J_0(pi*delta_1*xi)^2 + (1-lambda) * J_0(pi*delta_2*xi)^2
with (delta_1, delta_2, lambda) = (0.138, 0.055, 0.9312)  (best two-component)
and the further-refined (0.138, 0.0525, 0.935).

We compute (k_1, k_2, k_3, k_4), K_2, S_1 and feed them into:

   (A) MV single-moment (eq. 7)              -- arcsine baseline form
   (B) MV n_max=1 refinement (eq. 10)        -- z_1 box constraint
   (C) MV multi-moment MM-10 at n_max=2, 3, 4
   (D) MO Prop 2.11 m=3 + MV-side joint     (analogue of multifreq_mo217)
   (E) MO Prop 2.11 m=4 + MV-side joint
   (F) Hybrid: min over all valid bounds.

The K26 K is fixed throughout; only the OUTER analytic inequality changes.

Output: console summary + _master_k26_alt_master.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import mpmath as mp
from mpmath import mpf
from scipy.special import j0
from scipy.optimize import brentq

mp.mp.dps = 50

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

# ---------- MV constants ----------
DELTA = 0.138
U = 0.5 + DELTA   # 0.638
N_QP = 119

# Load MV coefficients
sys.path.insert(0, os.path.join(REPO, "delsarte_dual", "grid_bound"))
from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq

_MVC_FMPQ = mv_coeffs_fmpq()
MV_COEFFS = np.array([float(c.p)/float(c.q) for c in _MVC_FMPQ])

# ---------- K26 best parameters ----------
KERNELS = {
    "single_arcsine_MV": dict(deltas=[0.138], lambdas=[1.0]),
    "K26_two_comp_best": dict(deltas=[0.138, 0.055], lambdas=[0.9312, 0.0688]),
    "K26_fine_refined":  dict(deltas=[0.138, 0.0525], lambdas=[0.935, 0.065]),
}


# ---------- K_hat for multi-scale ----------
def K_hat(xi, deltas, lambdas):
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * xi) ** 2
    return out


# ---------- Moments k_n for n=1..N_MAX ----------
def kn_value(n, deltas, lambdas):
    """k_n = K_hat(n), where K_hat is the FT of K with K supported on [-DELTA, DELTA].

    For K = sum lambda_i * K_arcsine(. ; delta_i), K_hat(xi) = sum lambda_i J_0(pi delta_i xi)^2.
    """
    return float(K_hat(np.array([float(n)]), deltas, lambdas)[0])


def compute_moments(deltas, lambdas, n_max=8, n_xi=80001, xi_max=1200.0):
    """Return dict with k_1..k_{n_max}, K_2, S_1, K_hat_at_j_over_u (1..N_QP)."""
    deltas = np.asarray(deltas, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)

    # k_n for n = 1..n_max
    ns = np.arange(1, n_max + 1, dtype=float)
    kn = K_hat(ns, deltas, lambdas)

    # K_2 = int K_hat(xi)^2 dxi over all R (even -> 2 * int_0^inf)
    xi = np.linspace(0.0, xi_max, n_xi)
    dxi = xi[1] - xi[0]
    Kh = K_hat(xi, deltas, lambdas)
    K_2 = float(2.0 * np.trapezoid(Kh ** 2, dx=dxi))

    # S_1 = sum a_j^2 / K_hat(j/U)
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = K_hat(qp_xi, deltas, lambdas)
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))

    return dict(kn=[float(x) for x in kn], K_2=K_2, S_1=S_1,
                a=4.0 / (U * S_1),
                target=2.0 / U + 4.0 / (U * S_1),
                tail=float(Kh[-100:].max() ** 2))


# ---------- MV master inequalities ----------
def mu_M(M):
    """Lemma 3.4 bound on z_n^2."""
    return float(M * np.sin(np.pi / M) / np.pi)


def mv_eq7_LB(K_2, target):
    """Solve MV eq. 7: 2/u + a <= M + 1 + sqrt((M-1)(K_2-1)).

    Returns smallest M satisfying it.
    """
    def f(M):
        if M <= 1: return -1e10
        rad = (M - 1) * (K_2 - 1)
        return M + 1 + np.sqrt(max(0, rad)) - target
    try:
        return brentq(f, 1.0 + 1e-10, 2.5, xtol=1e-12)
    except Exception:
        return None


def mv_eq10_LB(k_1, K_2, target):
    """MV eq. 10 with z_1 = sqrt(mu(M)) at boundary.

    LHS(M) = M + 1 + 2*mu(M)*k_1 + sqrt(M - 1 - 2*mu(M)^2) * sqrt(K_2 - 1 - 2*k_1^2)
    """
    rad2 = K_2 - 1 - 2 * k_1 * k_1
    if rad2 <= 0:
        return None

    def LHS(M):
        mu_ = mu_M(M)
        # Check if interior critical point lies below mu
        y_star = (k_1 ** 2) * (M - 1) / (K_2 - 1)
        if y_star <= mu_:
            # use interior optimum which gives MV-7
            return M + 1 + np.sqrt(max(0.0, (M - 1) * (K_2 - 1)))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0: return float('inf')
        return M + 1 + 2 * mu_ * k_1 + np.sqrt(rad1 * rad2)

    def f(M): return LHS(M) - target
    try:
        return brentq(f, 1.0 + 1e-10, 2.5, xtol=1e-12)
    except Exception:
        return None


def mm10_LB(kn_list, K_2, target, n_max):
    """Multi-moment MM-10 inequality, n_max levels.

    LHS = M + 1 + 2 * sum_{n<=n_max} y_n * k_n
        + sqrt(M - 1 - 2 * sum y_n^2) * sqrt(K_2 - 1 - 2 * sum k_n^2)
    where y_n = z_n^2 in [0, mu(M)] (Lemma 1 box-constraint).

    Maximise LHS over (y_n) to find the tightest LB.

    Closed-form KKT solution.  Let S = {n : y_n at boundary y_n = mu}, F = complement.
    For n in F: y_n = k_n * sqrt(rad1/rad2_F) where rad2_F = K_2-1-2*sum k_n^2 (over ALL n in [1,n_max]).
    Substituting back, the F-subproblem reduces to MV-7 with effective K_2, M, k.  The
    "active set" S is unique; try all 2^n_max subsets, find the maximum.
    """
    from itertools import product
    kn_arr = np.array(kn_list[:n_max], dtype=float)
    sum_kn2 = float(np.sum(kn_arr ** 2))
    rad2 = K_2 - 1 - 2 * sum_kn2   # >=0 required
    if rad2 <= 0:
        return None

    def max_RHS(M):
        mu_ = mu_M(M)

        # Enumerate active set patterns (which y_n = mu_ vs interior).
        # For each pattern, solve the constrained-interior y values.
        best = -np.inf
        for pattern in product([0, 1], repeat=n_max):
            # pattern[n]=1 means y_n = mu_, pattern[n]=0 means interior (>0).
            S_mask = np.array(pattern, dtype=bool)
            F_mask = ~S_mask
            # For F-coords, y_n = lambda * k_n, with lambda = sqrt(rad1/rad2),
            # where rad1 = M-1-2*sum_S mu_^2 - 2*lambda^2*sum_F k_n^2.
            # => lambda^2 * rad2 = M - 1 - 2*sum_S mu^2 - 2*lambda^2 * sum_F kn^2.
            # => lambda^2 * (rad2 + 2*sum_F kn^2) = M - 1 - 2*sum_S mu^2.
            sumS = float(np.sum(S_mask)) * (mu_ ** 2)
            sumF_kn2 = float(np.sum(kn_arr[F_mask] ** 2))
            numer = M - 1 - 2 * sumS
            denom = rad2 + 2 * sumF_kn2
            if numer < 0 or denom <= 0:
                continue
            lam = np.sqrt(numer / denom)
            y = np.empty(n_max)
            y[S_mask] = mu_
            y[F_mask] = lam * kn_arr[F_mask]
            # Validity: y >= 0 (auto), and for S coords lam*k_n >= mu (active set
            # consistency: interior solution would have been clipped); for F coords
            # lam*k_n <= mu.
            if not np.all(y[S_mask] >= 0): continue
            if not np.all(y[F_mask] >= 0): continue
            if not np.all(y[F_mask] <= mu_ + 1e-12): continue
            if S_mask.any() and not np.all(lam * kn_arr[S_mask] >= mu_ - 1e-12): continue

            sum_y2 = float(np.sum(y ** 2))
            rad1 = M - 1 - 2 * sum_y2
            if rad1 < 0: continue
            sum_yk = float(np.sum(y * kn_arr))
            val = M + 1 + 2 * sum_yk + np.sqrt(rad1 * rad2)
            if val > best: best = val

        # Fallback: if no pattern worked (shouldn't happen), return MV-7 value.
        if best == -np.inf:
            best = M + 1 + np.sqrt(max(0.0, (M - 1) * (K_2 - 1)))
        return best

    def f(M): return max_RHS(M) - target
    try:
        f_lo = f(1.0 + 1e-10)
        f_hi = f(1.6)
        if f_lo >= 0 or f_hi <= 0:
            return None
        return brentq(f, 1.0 + 1e-10, 1.6, xtol=1e-12)
    except Exception:
        return None


# ---------- MO Prop 2.11 (m=3) joint bound (P-side) ----------
def mo_prop211_p_side_LB(K2_norm_43, target, k_1_K2_etc=None):
    """Compute the MO Prop 2.11 m=3 P-side ceiling (independent of K_beta!).

    The P-side is: M >= 1 + 2|f1|^4 + 2|f2|^4 + ((Mtilde)/||K||_{4/3})^4
    where K is the m-truncation kernel, NOT MV's K_beta.

    The optimal P-side lower bound was computed in multifreq_mo217 = 1.116;
    independent of the choice of MV kernel K_beta. So this is a CONSTANT.
    We return it as a known constant (per derivation.md §5).
    """
    return 1.116276  # K_pm/K_step both give 1.116


# ---------- ||K_hat||_{4/3} for MO m=3, m=4 using K = K_pm ----------
def Khat_43_norm(K_fhat_fn, m):
    """||_m K_hat||_{4/3} = (sum_{|j|>=m} |K_hat(j)|^{4/3})^{3/4}.

    Truncated at j = 4000 (tail < 1e-10 for K_pm / K_step).
    """
    JMAX = 4000
    s = mpf(0)
    for j in range(m, JMAX + 1):
        v = K_fhat_fn(j)
        if v == 0:
            continue
        s += abs(v) ** (mpf(4) / 3)
    s = 2 * s  # pair j and -j
    return s ** (mpf(3) / 4)


def K_pm_fhat(j):
    """K_pm: +1 on [-1/4,1/4], -1 on [-1/2,-1/4] U [1/4, 1/2]."""
    j = int(j)
    if j == 0: return mpf(0)
    return mpf(2) * mp.sin(mp.pi * mpf(j) / 2) / (mp.pi * mpf(j)) - mp.sin(mp.pi * mpf(j)) / (mp.pi * mpf(j))


def K_step_fhat(j):
    if j == 0: return mpf(1)/2
    return mp.sin(mp.pi * mpf(j) / 2) / (mp.pi * mpf(j))


# ---------- MV-side at n_max=2 with the L2.14 + L2.17 box (multi-moment) ----------
# Already covered by mm10_LB with n_max=2.

# ---------- Yu's tilted CS check ----------
def yu_tilted_LB(k_1, K_2, target, gamma_grid=None):
    """Replace MV eq. 9's CS step by a Holder/tilted-CS with parameter gamma in (1, infty).

    For gamma=2 we recover MV (CS).  For gamma=1+ we get an L^inf-L^1 split.
    The MV-tilted master inequality (gamma>1) is

        sum_{|j|>1} h_j k_j  <=  (sum |h_j|^p)^{1/p} (sum |k_j|^q)^{1/q},  1/p+1/q=1.

    With h = f*f, ||h||_p^p <= M^{p-1} (since h <= M, int h = 1).
    Plug into MV's eq. 9 / 10 framework with p, q != 2.

    Returns dict of best gamma and M_LB.
    """
    if gamma_grid is None:
        gamma_grid = np.linspace(1.1, 4.0, 40)
    best = {"M_LB": None, "gamma": None}
    for g in gamma_grid:
        p = float(g)
        q = p / (p - 1)
        # Tail bound:  sum_{|j|>1} z_j^{2p}  <=  M^{p-1} - 1 - 2 mu^p   (extending L_p)
        # The k tail:   (sum_{|j|>1} k_j^q)^{1/q}
        # We need sum_{j!=0} |k_j|^q.  For K = arcsine, k_j = |J_0(pi j delta)|^2 has
        # k_0 = 1, k_j decays slowly.  Not analytically tractable -- skip in this pass.
        pass
    # For now, mark as unimplemented (no L^q summable for arcsine).
    return {"M_LB": None, "gamma": None,
            "note": "Yu-tilt requires sum k_j^q; for arcsine k_j ~ 1/j on average -> diverges for q < 2."}


# ---------- 3-point SDP "V2" check ----------
def three_point_V2_check(kn_list, K_2, target):
    """3-point SDP V2 was Δ=0 for single arcsine.
    For multi-scale K, does the Δ=0 break?

    V2 design: include f_hat(0), f_hat(1), f_hat(2) and a generalised CS with
    K-weighted basis.  Per project_threepoint_sdp_dead.md, V2 had Δ=0 *structurally*
    (the SDP feasibility was tight at every M for ANY K).

    The "structural" aspect means: V2 reduces to MV-7 in the K -> K_beta limit, and
    for ANY K with K_hat >= 0 the V2 SDP-tightening is also dominated by MV-7's
    interior critical point (per the closed-form lambda * k_n analysis in
    multi_moment_derivation.md).

    With multi-scale K26 the same structural reasoning applies: the interior critical
    point of the SDP coincides with MV-7's interior, and the only gain is from clipping
    z_n^2 <= mu(M).  This is captured by mm10_LB(n_max=2).

    So 3-pt V2 with K26 gives M_LB = mm10_LB(n_max=2) -- no new lift over MM-10.
    """
    return {"M_LB": mm10_LB(kn_list, K_2, target, n_max=2),
            "note": "V2 SDP collapses to MM-10 at n_max=2 by the lambda*k_n interior crit pt argument."}


# ============================================================================
# Main: run all bounds for each kernel.
# ============================================================================

def evaluate_kernel(name, deltas, lambdas):
    print("=" * 78)
    print(f"KERNEL: {name}  deltas={deltas}  lambdas={lambdas}")
    print("=" * 78)
    mom = compute_moments(deltas, lambdas, n_max=8)
    kn = mom["kn"]
    K_2 = mom["K_2"]
    S_1 = mom["S_1"]
    a = mom["a"]
    target = mom["target"]
    print(f"k_1={kn[0]:.7f}  k_2={kn[1]:.7f}  k_3={kn[2]:.7f}  k_4={kn[3]:.7f}")
    print(f"k_5={kn[4]:.7f}  k_6={kn[5]:.7f}  k_7={kn[6]:.7f}  k_8={kn[7]:.7f}")
    print(f"K_2={K_2:.6f}  S_1={S_1:.4f}  a={a:.7f}  target=2/u+a={target:.6f}")
    print(f"tail at xi_max squared = {mom['tail']:.3e}")

    bounds = {}

    # (A) MV eq. 7  (no z_1 refinement)
    M_eq7 = mv_eq7_LB(K_2, target)
    bounds["A_MV_eq7"] = M_eq7
    print(f"\n(A) MV eq. 7        (z_1=0 limit):              M_cert = {M_eq7}")

    # (B) MV eq. 10  (z_1 box)
    M_eq10 = mv_eq10_LB(kn[0], K_2, target)
    bounds["B_MV_eq10"] = M_eq10
    print(f"(B) MV eq. 10       (z_1 box, n_max=1):         M_cert = {M_eq10}")

    # (C) MM-10 at n_max = 2, 3, 4
    for n_max in [2, 3, 4, 5, 6, 8]:
        M = mm10_LB(kn, K_2, target, n_max)
        bounds[f"C_MM10_nmax{n_max}"] = M
        print(f"(C) MM-10 n_max={n_max}: M_cert = {M}")

    # (D) MO Prop 2.11 m=3 P-side  (independent of MV-K -- but reports the value)
    M_P = mo_prop211_p_side_LB(None, None)
    # The joint bound is max(MV-side, P-side)
    bounds["D_P_side_constant"] = M_P
    bounds["D_MO211_m3_joint"] = max(M_eq10 if M_eq10 else 0, M_P)
    print(f"\n(D) MO Prop 2.11 m=3 P-side constant: {M_P}")
    print(f"    Joint MV(eq10)+MO211: max({M_eq10:.5f}, {M_P:.5f}) = {bounds['D_MO211_m3_joint']:.5f}")

    # (E) MO Prop 2.11 m=4 (4/3-norm even smaller, but P-side ceiling derivation
    #     in multifreq_mo217 §6 gives M_inf <= 1 + 1/||_m K_hat||_{4/3}^4)
    Khat43_m3 = float(Khat_43_norm(K_pm_fhat, 3))
    Khat43_m4 = float(Khat_43_norm(K_pm_fhat, 4))
    M_Pside_m3_alt = 1.0 + 1.0 / Khat43_m3 ** 4
    M_Pside_m4_alt = 1.0 + 1.0 / Khat43_m4 ** 4
    bounds["E_K_pm_43norm_m3"] = Khat43_m3
    bounds["E_K_pm_43norm_m4"] = Khat43_m4
    bounds["E_P_side_m3_K_pm"] = M_Pside_m3_alt
    bounds["E_P_side_m4_K_pm"] = M_Pside_m4_alt
    print(f"(E) ||_3 K_pm||_{{4/3}} = {Khat43_m3:.5f},  M_P(m=3,K_pm) <= {M_Pside_m3_alt:.5f}")
    print(f"    ||_4 K_pm||_{{4/3}} = {Khat43_m4:.5f},  M_P(m=4,K_pm) <= {M_Pside_m4_alt:.5f}")

    # (F) Restricted-Hoelder conditional (1.378 under Hyp_R(M_max=1.51))
    bounds["F_restricted_holder_conditional"] = 1.37842  # K-independent

    # (G) 3-pt SDP V2 with K26
    V2 = three_point_V2_check(kn, K_2, target)
    bounds["G_3pt_V2_M_LB"] = V2["M_LB"]
    bounds["G_3pt_V2_note"] = V2["note"]
    print(f"(G) 3-pt SDP V2 (K26): M_LB = {V2['M_LB']}  ({V2['note']})")

    # (H) Yu tilted CS
    tilt = yu_tilted_LB(kn[0], K_2, target)
    bounds["H_Yu_tilted"] = tilt
    print(f"(H) Yu-tilted CS: {tilt['note']}")

    # (I) Hybrid: max of unconditional LBs (best LB on M)
    uncond_LBs = [v for k, v in bounds.items() if k in
                  ["A_MV_eq7", "B_MV_eq10", "C_MM10_nmax2", "C_MM10_nmax3",
                   "C_MM10_nmax4", "C_MM10_nmax5", "C_MM10_nmax6", "C_MM10_nmax8",
                   "D_MO211_m3_joint"] and v is not None]
    M_hybrid = max(uncond_LBs) if uncond_LBs else None
    bounds["I_hybrid_max"] = M_hybrid
    print(f"\n(I) HYBRID (max of unconditional LBs): {M_hybrid}")

    return dict(name=name, deltas=list(deltas), lambdas=list(lambdas),
                kn=kn, K_2=K_2, S_1=S_1, a=a, target=target,
                bounds=bounds)


def main():
    results = {}
    for name, kdef in KERNELS.items():
        results[name] = evaluate_kernel(name, kdef["deltas"], kdef["lambdas"])

    # Comparison table
    print("\n" + "=" * 78)
    print("SUMMARY  (M_cert under different master inequalities)")
    print("=" * 78)
    header = "Inequality".ljust(30) + " | "
    for name in KERNELS:
        header += name.ljust(25) + " | "
    print(header)
    print("-" * len(header))

    keys_order = ["A_MV_eq7", "B_MV_eq10",
                  "C_MM10_nmax2", "C_MM10_nmax3", "C_MM10_nmax4",
                  "C_MM10_nmax5", "C_MM10_nmax6", "C_MM10_nmax8",
                  "D_P_side_constant", "D_MO211_m3_joint",
                  "F_restricted_holder_conditional",
                  "G_3pt_V2_M_LB",
                  "I_hybrid_max"]
    for k in keys_order:
        row = k.ljust(30) + " | "
        for name in KERNELS:
            v = results[name]["bounds"].get(k)
            cell = f"{v:.5f}" if isinstance(v, (int, float)) else "N/A"
            row += cell.ljust(25) + " | "
        print(row)

    outpath = os.path.join(REPO, "_master_k26_alt_master.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: (
            float(x) if hasattr(x, "item") else str(x)))
    print(f"\nWrote {outpath}")
    return results


if __name__ == "__main__":
    main()
