"""Frank-Wolfe optimization of the continuous probability measure nu
over delta in (0, 0.138] for the multi-scale arcsine kernel:

    K_hat(xi) = int J_0(pi*delta*xi)^2 d nu(delta)
              = sum_i lambda_i * J_0(pi*delta_i*xi)^2   (discretized)

Master objective (MV master inequality) defines M_cert(k_1, K_2, S_1)
implicitly via sup_R(M) = 2/u + a; we derive grad_lambda M_cert
ANALYTICALLY via implicit differentiation.

Pipeline (every step rigor-aware):
  1. Discretize delta on a fine uniform grid (N atoms).
  2. Pre-compute (using existing cache) the per-atom data
       C[i,j]  = int J_0(pi d_i xi)^2 J_0(pi d_j xi)^2 dxi  (exact via quad)
       Kh_qp[i,j] = J_0(pi d_i (j/u))^2
       Kh_1[i]    = J_0(pi d_i)^2
  3. Initialize nu at the v4 3-scale optimum (deltas=0.138/0.055/0.025,
     lambdas=0.85/0.10/0.05), projected to the grid.
  4. Frank-Wolfe:
        g_i = directional derivative dM_cert/dlambda toward vertex e_i
            (analytic, via implicit-function theorem on the master eq.)
        i*  = argmax g_i
        line-search s in [0,1]:  nu <- (1-s) nu + s e_{i*}
  5. Pairwise CD polish.
  6. RIGOROUS arb verification via certify_Nscale from _cohn_elkies_128_v5.
     (Drops atoms with lambda < threshold; runs cvxpy QP reopt for G.)

Deliverable: best rigorous M_cert at the converged nu, list of (delta_i, lambda_i)
support atoms, effective Caratheodory dimension.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import j0

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
# Walk up to locate the project root containing 'delsarte_dual' so the kernel
# helper can import MV_COEFFS.
_p = REPO
for _ in range(5):
    if (_p / "delsarte_dual").is_dir():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent

from _kernel_probe_helper import DELTA, MV_COEFFS, N_QP, U, mv_master_M_cert
assert MV_COEFFS is not None, "MV_COEFFS failed to load; delsarte_dual on sys.path?"
from _master_k26_continuous import (
    precompute_atoms,
    eval_nu,
    support_summary,
    pairwise_cd,
)

# Rigorous arb certifier
from _cohn_elkies_128_v5 import (
    certify_Nscale,
    solve_qp_Nscale,
    round_to_rationals,
    configure_precision,
    DELTA1_Q,
)


QP_XI = np.arange(1, N_QP + 1) / U


# ---------------------------------------------------------------------------
# Closed-form optimal G coefficients given K_hat(j/u) weights.
#
# QP:  min sum_j (a_j^2 / w_j)  s.t.  sum_j a_j cos(2 pi j x / u) >= 1
#                                    for all x in [0, 1/4].
#
# Lagrangian:   L = sum_j a_j^2 / w_j  -  int_{[0,1/4]} mu(x) (G(x) - 1) dx
# Stationarity: 2 a_j / w_j  =  int mu(x) cos(2 pi j x / u) dx
#               => a_j = (w_j / 2) * c_j   where c_j = int mu cos(...)
#               => S_1 = sum_j (w_j / 4) c_j^2
# This is an infinite-dim LP dual; we approximate it by a finite grid of x's
# and solve the QP via cvxpy each call.
# ---------------------------------------------------------------------------

# Pre-build the constraint matrix B[x_k, j] = cos(2 pi j x_k / u)  for x_k in [0,1/4]
_QP_NGRID = 501
_QP_XS = np.linspace(0.0, 0.25, _QP_NGRID)
_QP_B = np.zeros((_QP_NGRID, N_QP))
for _j in range(1, N_QP + 1):
    _QP_B[:, _j - 1] = np.cos(2.0 * math.pi * _j * _QP_XS / float(U))

# Cached parametric problem for fast resolving with new weights
_QP_PROB = None
_QP_PARAM_W = None
_QP_VAR_A = None


def _build_qp_problem():
    """Build a parametric cvxpy Problem once and re-use with Parameter weights.

    Bypasses cvxpy's re-canonicalisation each call.
    """
    global _QP_PROB, _QP_PARAM_W, _QP_VAR_A
    import cvxpy as cp
    w = cp.Parameter(N_QP, pos=True)
    a = cp.Variable(N_QP)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [_QP_B @ a >= 1.0]
    _QP_PROB = cp.Problem(obj, cons)
    _QP_PARAM_W = w
    _QP_VAR_A = a


def solve_qp_optimalG(kh_qp):
    """For weights w = kh_qp = K_hat(j/u), solve the QP for optimal G.

    Returns (a_opt, S_1_opt) where S_1_opt = sum a_j^2 / kh_qp[j].
    """
    if np.any(kh_qp < 1e-15):
        return None, np.inf
    if _QP_PROB is None:
        _build_qp_problem()
    _QP_PARAM_W.value = np.asarray(kh_qp, dtype=float).copy()
    for s_name in ("MOSEK", "CLARABEL", "SCS"):
        try:
            _QP_PROB.solve(solver=s_name, verbose=False, warm_start=True)
            if _QP_PROB.status in ("optimal", "optimal_inaccurate") and _QP_VAR_A.value is not None:
                break
        except Exception:
            continue
    if _QP_VAR_A.value is None:
        return None, np.inf
    a_val = np.asarray(_QP_VAR_A.value).flatten()
    S_1 = float(np.sum(a_val ** 2 / kh_qp))
    return a_val, S_1


def eval_nu_optG(lam, C, Kh_qp, Kh_1):
    """Evaluate M_cert for nu = lam with G *re-optimised* for this nu.

    Returns (M, k_1, K_2, S_1, a_opt)
    """
    K_2 = float(lam @ C @ lam)
    k_1 = float(lam @ Kh_1)
    kh_qp = lam @ Kh_qp
    if np.any(kh_qp < 1e-15):
        return None, k_1, K_2, None, None
    a_opt, S_1 = solve_qp_optimalG(kh_qp)
    if a_opt is None:
        return None, k_1, K_2, None, None
    M = mv_master_M_cert(k_1, K_2, S_1)
    return M, k_1, K_2, S_1, a_opt


# ---------------------------------------------------------------------------
# Analytic implicit-differentiation gradient  (fixed MV G, baseline)
# ---------------------------------------------------------------------------
def grad_M_analytic_optG(lam, C, Kh_qp, Kh_1, u_=U):
    """Analytic gradient WITH re-optimised G (envelope theorem).

    Returns (M, grad, k_1, K_2, S_1, a_opt).

    By the envelope theorem, dS_1/dlam_i at the optimal a* is computed
    holding a = a*:  dS_1/dlam_i = - sum_j (a*_j)^2 * Kh_qp[i,j] / kh_qp[j]^2
    """
    M, k_1, K_2, S_1, a_opt = eval_nu_optG(lam, C, Kh_qp, Kh_1)
    if M is None:
        return None, None, k_1, K_2, S_1, a_opt
    mu = M * math.sin(math.pi / M) / math.pi
    rad1 = M - 1.0 - 2.0 * mu * mu
    rad2 = K_2 - 1.0 - 2.0 * k_1 * k_1
    y_star_sq = (k_1 * k_1) * (M - 1.0) / (K_2 - 1.0)
    y_star = math.sqrt(max(0.0, y_star_sq))
    use_refined = (y_star > mu) and (rad1 > 0.0) and (rad2 > 0.0)
    kh_qp = lam @ Kh_qp
    coef = (a_opt ** 2) / (kh_qp ** 2)
    dS1 = -(Kh_qp @ coef)
    dk1 = Kh_1
    dK2 = 2.0 * (C @ lam)
    if use_refined:
        sqrtR = math.sqrt(rad1 * rad2)
        F_k1 = 2.0 * mu - 2.0 * k_1 * rad1 / sqrtR
        F_K2 = 0.5 * rad1 / sqrtR
        dmu_dM = (math.sin(math.pi / M) - (math.pi / M) * math.cos(math.pi / M)) / math.pi
        drad1_dM = 1.0 - 4.0 * mu * dmu_dM
        F_M = 1.0 + 2.0 * k_1 * dmu_dM + 0.5 * drad1_dM * rad2 / sqrtR
    else:
        sqrtR = math.sqrt((M - 1.0) * (K_2 - 1.0))
        F_k1 = 0.0
        F_K2 = 0.5 * (M - 1.0) / sqrtR
        F_M = 1.0 + 0.5 * (K_2 - 1.0) / sqrtR
    F_S1 = (4.0 / float(u_)) / (S_1 * S_1)
    if abs(F_M) < 1e-14:
        return M, None, k_1, K_2, S_1, a_opt
    num = F_k1 * dk1 + F_K2 * dK2 + F_S1 * dS1
    grad = -num / F_M
    return M, grad, k_1, K_2, S_1, a_opt


def grad_M_analytic(lam, C, Kh_qp, Kh_1, mv_coeffs=MV_COEFFS, u_=U):
    """Return (M, grad) where grad[i] = dM_cert/dlambda_i (directional, free
    component -- the simplex constraint is handled by FW's argmax step).

    Derivation:
      sup_R(M) = M + 1 + 2 mu k_1 + sqrt(rad1 * rad2)
        with mu  = M sin(pi/M)/pi
             rad1 = M - 1 - 2 mu^2
             rad2 = K_2 - 1 - 2 k_1^2
        (only the 'refined' branch; we verify the y_star regime in code)
      F(M, k_1, K_2, S_1) = sup_R(M) - 2/u - (4/u) * (minG^2) / S_1 = 0
      We use minG = 1 (ignore the G-reopt feedback in the FW gradient;
      the line-search and final arb verification use the true minG and reopt).

    Implicit diff:  dM/dlam_i = -[F_k1 dk1 + F_K2 dK2 + F_S1 dS1] / F_M

    where
      dk1/dlam_i = Kh_1[i]
      dK2/dlam_i = 2 (C lam)_i
      dS1/dlam_i = -sum_j (a_j^2) * Kh_qp[i,j] / kh_qp[j]^2

      F_k1 = 2 mu  +  (1/2)*rad1*(-4 k_1)/sqrt(rad1 rad2)
           = 2 mu  -  2 k_1 * rad1 / sqrt(rad1 rad2)         (= 2 mu  -  2 k_1 sqrt(rad1/rad2))
      F_K2 = (1/2) * rad1 / sqrt(rad1 rad2)                  (= (1/2) sqrt(rad1/rad2))
      F_S1 = (4/u) / S_1^2
      F_M  = 1 + 2 k_1 dmu/dM
             + (1/(2 sqrt(rad1 rad2))) * drad1/dM * rad2
           dmu/dM = (sin(pi/M) - (pi/M) cos(pi/M)) / pi
           drad1/dM = 1 - 4 mu dmu/dM
    """
    M, k_1, K_2, S_1 = eval_nu(lam, C, Kh_qp, Kh_1, mv_coeffs=mv_coeffs)
    if M is None:
        return None, None

    # Check that we're in the 'refined' (k_1 > mu) regime; the y_star <= mu
    # branch has gradient = no_z1 branch.  We'll handle both.
    mu = M * math.sin(math.pi / M) / math.pi
    rad1 = M - 1.0 - 2.0 * mu * mu
    rad2 = K_2 - 1.0 - 2.0 * k_1 * k_1
    y_star_sq = (k_1 * k_1) * (M - 1.0) / (K_2 - 1.0)
    y_star = math.sqrt(max(0.0, y_star_sq))
    use_refined = (y_star > mu) and (rad1 > 0.0) and (rad2 > 0.0)

    kh_qp = lam @ Kh_qp  # (N_QP,)
    # dS1/dlam_i = -sum_j a_j^2 Kh_qp[i,j] / kh_qp[j]^2
    coef = (mv_coeffs ** 2) / (kh_qp ** 2)
    dS1 = -(Kh_qp @ coef)  # shape (N,)
    dk1 = Kh_1  # shape (N,)
    dK2 = 2.0 * (C @ lam)  # shape (N,)

    if use_refined:
        sqrtR = math.sqrt(rad1 * rad2)
        F_k1 = 2.0 * mu - 2.0 * k_1 * rad1 / sqrtR
        F_K2 = 0.5 * rad1 / sqrtR
        # F_M
        dmu_dM = (math.sin(math.pi / M) - (math.pi / M) * math.cos(math.pi / M)) / math.pi
        drad1_dM = 1.0 - 4.0 * mu * dmu_dM
        F_M = 1.0 + 2.0 * k_1 * dmu_dM + 0.5 * drad1_dM * rad2 / sqrtR
    else:
        # no_z1 branch: sup_R = M + 1 + sqrt((M-1)(K_2 - 1))
        sqrtR = math.sqrt((M - 1.0) * (K_2 - 1.0))
        F_k1 = 0.0
        F_K2 = 0.5 * (M - 1.0) / sqrtR
        F_M = 1.0 + 0.5 * (K_2 - 1.0) / sqrtR
    F_S1 = (4.0 / float(u_)) / (S_1 * S_1)

    if abs(F_M) < 1e-14:
        return M, None
    # dM/dlam_i for each i
    num = F_k1 * dk1 + F_K2 * dK2 + F_S1 * dS1
    grad = -num / F_M
    return M, grad


# ---------------------------------------------------------------------------
# Frank-Wolfe with analytic gradient
# ---------------------------------------------------------------------------
def frank_wolfe_analytic(lam0, C, Kh_qp, Kh_1, n_iters=200, tol=1e-9,
                         verbose=False):
    """Frank-Wolfe with analytic gradient + 1D line search."""
    lam = lam0.copy()
    M_curr, grad = grad_M_analytic(lam, C, Kh_qp, Kh_1)
    if M_curr is None:
        return lam, -np.inf, []
    history = [M_curr]
    for it in range(n_iters):
        M_curr, grad = grad_M_analytic(lam, C, Kh_qp, Kh_1)
        if grad is None:
            break
        # FW vertex: argmax over simplex of <grad, v> is argmax_i grad[i]
        i_star = int(np.argmax(grad))
        # Frank-Wolfe gap: <grad, e_{i*} - lam>
        fw_gap = grad[i_star] - float(grad @ lam)
        if fw_gap < tol:
            if verbose:
                print(f"  FW it {it}: gap={fw_gap:.2e} < tol -> stop")
            break

        # Line search along  s -> (1-s) lam + s e_{i*}
        def neg_M(s):
            lam_s = (1.0 - s) * lam
            lam_s[i_star] += s
            M, _, _, _ = eval_nu(lam_s, C, Kh_qp, Kh_1)
            return -M if M is not None else np.inf

        res = minimize_scalar(neg_M, bounds=(0.0, 1.0), method='bounded',
                              options={'xatol': 1e-9})
        s_best = float(res.x)
        M_new = -res.fun
        if not (M_new > M_curr + 1e-12):
            if verbose:
                print(f"  FW it {it}: line search no improvement (gap={fw_gap:.2e})")
            break
        lam = (1.0 - s_best) * lam
        lam[i_star] += s_best
        M_curr = M_new
        history.append(M_curr)
        if verbose and (it < 10 or it % 10 == 0):
            print(f"  FW it {it}: i*={i_star} delta={i_star} s={s_best:.5f} "
                  f"gap={fw_gap:.3e} M={M_curr:.7f}")
    return lam, M_curr, history


# ---------------------------------------------------------------------------
# Frank-Wolfe with re-optimised G inside every evaluation
# ---------------------------------------------------------------------------
def frank_wolfe_optG(lam0, C, Kh_qp, Kh_1, n_iters=200, tol=1e-9,
                    verbose=False, ls_xatol=1e-9, checkpoint_path=None):
    lam = lam0.copy()
    history = []
    Mi, grad, _, _, _, _ = grad_M_analytic_optG(lam, C, Kh_qp, Kh_1)
    if Mi is None:
        return lam, -np.inf, history
    history.append(Mi)
    M_curr = Mi
    for it in range(n_iters):
        M_curr, grad, _, _, _, _ = grad_M_analytic_optG(lam, C, Kh_qp, Kh_1)
        if grad is None:
            break
        i_star = int(np.argmax(grad))
        fw_gap = grad[i_star] - float(grad @ lam)
        if fw_gap < tol:
            if verbose:
                print(f"  FWopt it {it}: gap={fw_gap:.2e} < tol -> stop")
            break

        def neg_M(s):
            lam_s = (1.0 - s) * lam
            lam_s[i_star] += s
            M, _, _, _, _ = eval_nu_optG(lam_s, C, Kh_qp, Kh_1)
            return -M if M is not None else np.inf

        res = minimize_scalar(neg_M, bounds=(0.0, 1.0), method='bounded',
                              options={'xatol': ls_xatol})
        s_best = float(res.x)
        M_new = -res.fun
        if not (M_new > M_curr + 1e-12):
            if verbose:
                print(f"  FWopt it {it}: no LS improvement (gap={fw_gap:.2e})")
            break
        lam = (1.0 - s_best) * lam
        lam[i_star] += s_best
        M_curr = M_new
        history.append(M_curr)
        if verbose and (it < 15 or it % 5 == 0):
            print(f"  FWopt it {it}: i*={i_star} s={s_best:.5f} "
                  f"gap={fw_gap:.3e} M={M_curr:.7f}", flush=True)
        if checkpoint_path is not None and (it % 10 == 0):
            np.savez(checkpoint_path, lam=lam, M_curr=M_curr, it=it,
                     history=np.array(history))
    return lam, M_curr, history


# ---------------------------------------------------------------------------
# Pairwise CD with re-optimised G
# ---------------------------------------------------------------------------
def pairwise_cd_optG(lam0, C, Kh_qp, Kh_1, n_sweeps=8, verbose=False,
                    tol=1e-9, max_active=15):
    lam = lam0.copy()
    M_curr, _, _, _, _ = eval_nu_optG(lam, C, Kh_qp, Kh_1)
    N = Kh_1.shape[0]
    for sweep in range(n_sweeps):
        improved = False
        active = list(np.where(lam > 1e-8)[0])
        # Restrict to the top-K heaviest atoms (CD on tail is wasteful)
        candidates = list(np.argsort(-lam)[:max_active])
        for i in candidates:
            for j in candidates:
                if i >= j:
                    continue
                m = lam[i] + lam[j]
                if m < 1e-9:
                    continue

                def neg_M(t):
                    if t < 0 or t > m:
                        return np.inf
                    lam_t = lam.copy()
                    lam_t[i] = t
                    lam_t[j] = m - t
                    M, _, _, _, _ = eval_nu_optG(lam_t, C, Kh_qp, Kh_1)
                    return -M if M is not None else np.inf

                res = minimize_scalar(neg_M, bounds=(0.0, m), method='bounded',
                                      options={'xatol': 1e-9})
                M_new = -res.fun
                if M_new > M_curr + tol:
                    lam[i] = float(res.x)
                    lam[j] = m - lam[i]
                    M_curr = M_new
                    improved = True
        if verbose:
            print(f"  CDopt sweep {sweep}: M={M_curr:.7f}  "
                  f"support={int(np.sum(lam > 1e-8))}")
        if not improved:
            break
    return lam, M_curr


# ---------------------------------------------------------------------------
# Initialise at v4 3-scale optimum
# ---------------------------------------------------------------------------
def init_v4_3scale(deltas):
    """Project the v4 3-scale optimum onto the discretized atom grid."""
    target_deltas = [0.138, 0.055, 0.025]
    target_lambdas = [0.85, 0.10, 0.05]
    N = len(deltas)
    lam = np.zeros(N)
    for d_t, l_t in zip(target_deltas, target_lambdas):
        idx = int(np.argmin(np.abs(np.asarray(deltas) - d_t)))
        lam[idx] += l_t
    lam /= lam.sum()
    return lam


# ---------------------------------------------------------------------------
# Rigorous arb verification of the final nu
# ---------------------------------------------------------------------------
def rigorous_verify(atoms, xi_max=10_000, verbose=True):
    """atoms: list of (delta_i, lambda_i), assumed pruned (small atoms dropped).
    Renormalize so lambdas sum to 1, then call certify_Nscale from v5.
    """
    from fractions import Fraction
    DEN = 10 ** 8
    # Pin delta_1 = 138/1000 if any atom is at 0.138 to use the cached frac
    deltas_q = []
    lambdas_q = []
    for d, lam in atoms:
        # exact rational delta on the grid
        d_q = Fraction(int(round(d * DEN)), DEN)
        deltas_q.append(d_q)
        lambdas_q.append(Fraction(int(round(lam * DEN)), DEN))
    # Make sure the largest atom is delta_1 = DELTA1_Q (138/1000), which is
    # required by v5's certifier (it fixes delta_1).
    # Pull out the largest atom and pin it to DELTA1_Q.
    order = sorted(range(len(deltas_q)), key=lambda i: -lambdas_q[i])
    # If the heaviest atom is already very close to 0.138, force it to 138/1000:
    if abs(float(deltas_q[order[0]]) - float(DELTA1_Q)) < 5e-4:
        deltas_q[order[0]] = DELTA1_Q
    # Normalize lambdas to sum to 1 exactly:
    s = sum(lambdas_q)
    if s == 0:
        raise ValueError("zero lambdas")
    lambdas_q = [l / s for l in lambdas_q]
    drift = Fraction(1) - sum(lambdas_q)
    lambdas_q[0] = lambdas_q[0] + drift
    if verbose:
        print("  Rigorous nu (rational):")
        for d, l in zip(deltas_q, lambdas_q):
            print(f"    delta = {float(d):.6f}    lambda = {float(l):.6f}")
    # Solve QP for G coefficients at this K_hat
    configure_precision(256)
    a_opt = solve_qp_Nscale(deltas_q, lambdas_q)
    coeffs_q = round_to_rationals(a_opt)
    r = certify_Nscale(deltas_q, lambdas_q, coeffs_q, xi_max=xi_max, verbose=verbose)
    return r


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(N_grid, n_fw_iters, do_cd_polish=True, xi_max_rigor=10_000,
        prune_thresh=5e-4, verbose=True, sweeps_cd=8):
    print(f"\n=== Continuous-nu Frank-Wolfe on N={N_grid} grid (DELTA={DELTA}) ===")
    t0 = time.time()
    deltas, C, Kh_qp, Kh_1 = precompute_atoms(N_grid, verbose=False, cache_dir=str(REPO))
    print(f"  precompute: {time.time() - t0:.1f}s  C shape={C.shape}")

    # Initialize at v4 3-scale optimum (projected to grid)
    lam0 = init_v4_3scale(deltas)
    M0_fixedG, _, _, _ = eval_nu(lam0, C, Kh_qp, Kh_1)
    M0_optG, k1_0, K2_0, S1_0, _ = eval_nu_optG(lam0, C, Kh_qp, Kh_1)
    print(f"\nInit (v4 3-scale projection):")
    print(f"  fixed-MV-G  M_cert = {M0_fixedG:.7f}  (mis-aligned w/ rigor)")
    print(f"  optimal-G   M_cert = {M0_optG:.7f}   k_1={k1_0:.5f} K_2={K2_0:.5f} S_1={S1_0:.4f}")
    print("  atoms:", support_summary(lam0, deltas))

    # --- Frank-Wolfe with re-optimised G ---
    print(f"\n--- Frank-Wolfe + reoptG (analytic grad, n_iters={n_fw_iters}) ---")
    t0 = time.time()
    ckpt_path = str(REPO / f"_continuous_nu_fw_ckpt_N{N_grid}.npz")
    lam_fw, M_fw, hist = frank_wolfe_optG(
        lam0, C, Kh_qp, Kh_1, n_iters=n_fw_iters, verbose=verbose,
        checkpoint_path=ckpt_path)
    print(f"  FWopt: M_cert = {M_fw:.7f}  in {time.time()-t0:.1f}s  "
          f"({len(hist)} accepted iters)")
    print("  support:", support_summary(lam_fw, deltas))

    # --- CD polish with re-optimised G ---
    if do_cd_polish:
        print(f"\n--- Pairwise CD polish + reoptG (n_sweeps={sweeps_cd}) ---")
        t0 = time.time()
        lam_cd, M_cd = pairwise_cd_optG(lam_fw, C, Kh_qp, Kh_1,
                                         n_sweeps=sweeps_cd, verbose=verbose)
        print(f"  CDopt: M_cert = {M_cd:.7f}  in {time.time()-t0:.1f}s")
        if M_cd > M_fw:
            lam_fw, M_fw = lam_cd, M_cd

    final_atoms = support_summary(lam_fw, deltas, thresh=1e-5)
    print(f"\nFinal (numerical, reoptG) M_cert = {M_fw:.7f}")
    print(f"Support size: {len(final_atoms)}")
    for d, lam in final_atoms:
        print(f"  delta = {d:.6f}    lambda = {lam:.7f}")

    # --- Effective Caratheodory dimension ---
    eff_atoms = [(d, l) for d, l in final_atoms if l > 1e-4]
    print(f"\nEffective Caratheodory dim (lambda > 1e-4): {len(eff_atoms)}")

    # --- Rigorous arb verification ---
    pruned_atoms = [(d, l) for d, l in final_atoms if l > prune_thresh]
    print(f"\n--- Rigorous arb verification at pruned nu "
          f"(thresh={prune_thresh}, support={len(pruned_atoms)}) ---")
    t0 = time.time()
    r_rigor = rigorous_verify(pruned_atoms, xi_max=xi_max_rigor, verbose=verbose)
    print(f"  arb cert: M_cert >= {r_rigor['M_cert_lower']:.7f}  "
          f"in {time.time()-t0:.1f}s")

    return {
        "N_grid": N_grid,
        "DELTA": DELTA,
        "init_atoms": support_summary(lam0, deltas),
        "init_M_cert_fixedG": float(M0_fixedG),
        "init_M_cert_optG": float(M0_optG),
        "fw_M_cert_numeric": float(M_fw),
        "final_atoms": final_atoms,
        "eff_caratheodory_dim": len(eff_atoms),
        "support_size": len(final_atoms),
        "rigorous": r_rigor,
        "prune_thresh": prune_thresh,
        "pruned_atoms": pruned_atoms,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=100, help="atom grid size")
    p.add_argument("--n_fw_iters", type=int, default=200)
    p.add_argument("--xi_max_rigor", type=int, default=10000)
    p.add_argument("--prune_thresh", type=float, default=5e-4)
    p.add_argument("--cd_sweeps", type=int, default=12)
    p.add_argument("--out", default="_continuous_nu_fw_results.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    out = run(args.N, args.n_fw_iters, do_cd_polish=True,
              xi_max_rigor=args.xi_max_rigor, prune_thresh=args.prune_thresh,
              sweeps_cd=args.cd_sweeps, verbose=not args.quiet)
    out_path = REPO / args.out
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
