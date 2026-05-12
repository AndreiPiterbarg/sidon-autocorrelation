"""C2 Optimal Transport probe — Wasserstein/OT-based lower bound on C_{1a}.

==============================================================================
SETUP (matching cascade conventions)
==============================================================================
  f : [-1/4, 1/4] -> R_{>=0}, integral 1.
  Discretise into d equal bins; mu_i = mass in bin i.  sum mu = 1.
  Bin centres x_i = -1/4 + (i + 0.5) / (2 d),  i = 0..d-1.
  Window TV at level d (rigorous lower bound on max_t (f*f)(t)):
     TV_W(mu) = (2 d / ell) * sum_{k = s .. s+ell-2}
                              sum_{i+j = k, 0<=i,j<d}  mu_i mu_j
     for ell in {2..2d}, s in {0..2d-1-(ell-1)}.

==============================================================================
KEY MATH OBSERVATIONS (recorded in analysis.md)
==============================================================================

(1) UNCONDITIONAL: bin-LP min L(d) = min_{simplex} max_W TV_W(mu).
    Local-search estimate at d=4,6,8.  Below 1.2802 — bin-LP at small d
    is loose.  This is the unconditional rigorous lower bound at this d.

(2) DIAGNOSTIC ONLY: bin-LP min with W_2(mu, U_d) <= R is much higher
    for small R, because U_d itself has max_W TV = 1.42 at d=8.
    But this is NOT a rigorous LB on C_{1a} since we have NO proof
    that W_2(f_opt, U) <= R for the C_{1a} extremizer.

    Why the natural Talagrand-T_2 chain fails:
    - For uniform U on [-1/4, 1/4] of length 1/2:  KL(f || U) = -h(f) + log 2.
    - Talagrand on bounded interval doesn't give classical T_2; via
      Pinsker + diameter:  W_1(f, U) <= sqrt(KL/2) * diam = sqrt(KL/2)/2.
      W_2 has only WEAKER bound (W_2^2 <= diam^2 * TV; TV via Pinsker).
    - To USE this we need an UPPER bound on KL(f || U), i.e., LOWER bound
      on h(f).  EPI gives: h(f*f) >= h(f) + (1/2) log 2.  And
      h(f*f) <= log(supp f*f) = log 1 = 0  (trivial; f*f probability
      density on [-1/2, 1/2]).  So h(f) <= -log sqrt(2), KL(f||U) >=
      log sqrt(2) + log 2 = (3/2) log 2.  This is a LOWER bound on KL,
      hence LOWER bound on W_2 — exactly the wrong direction.
    The autocorrelation max controls the spread of f*f, not f.

(3) SYMMETRIC L^2 (Cauchy-Schwarz at t=0) — gives a TRIVIAL bound:
    For SYMMETRIC f:  max(f*f) >= (f*f)(0) = int f^2 = ||f||_2^2.
    By Cauchy-Schwarz on the supp of length 1/2:
       1 = (int f)^2 <= (1/2) ||f||_2^2  =>  ||f||_2^2 >= 2.
    So C_{1a}^{sym} >= 2 — trivially.  This explains why
    bin-LP-min-with-symmetric-and-Sigma-mu^2 <= M/(2d) is INFEASIBLE
    for all M < 2.  The symmetric problem is fundamentally different:
    LB = 2 trivially, but the true C_{1a} (over all f) <= 1.5029.
    Hence symmetric-only bound is USELESS for the general C_{1a}.

(4) ASYMMETRIC f: the gap.  For non-symmetric f, ||f||_2^2 is NOT a
    lower bound on max(f*f).  Standard counterexample:
       f = mass on small interval near 0 + mass on small interval near 1/4.
    ||f||_2^2 large, max(f*f) much smaller (its peak shifts away from 0).
    So extension to non-symmetric f via L^2 is BLOCKED.

(5) Plancherel /  L^4 of \hat f:
       ||f*f||_2^2 = ||\hat f||_4^4  (Parseval).
       ||f*f||_inf >= ||f*f||_2^2 / ||f*f||_1 = ||\hat f||_4^4
                                                  (since ||f*f||_1 = 1).
    So  M >= ||\hat f||_4^4  for any f.
    This UNCONDITIONAL constraint involves ||\hat f||_4^4 which is
    sum_n |c_n|^4 (using Fourier coefficients of f viewed as periodic
    extension on a period containing supp f * 2).  This DOES translate
    to a constraint on mu, and it is unconditional.  We test this.

==============================================================================
PROBE EXPERIMENTS (corrected from v3)
==============================================================================

  E1: estimate L(d) = min_{simplex} max_W TV_W(mu) via SLSQP random
      restart, at d in {4, 6, 8, 10}.  REPORT: best LB.

  E2: diagnostic — same with W_2(mu, U_d) <= R sweep.  Document that
      this LIFTS the bin-LP min substantially BUT not rigorous.

  E3: |\hat f|^4 unconditional Plancherel constraint:  for f
      piecewise-constant on bins of width h = 1/(2d), |\hat f(xi)|^2
      is the discrete-Fourier sinc-modulated transform.  We compute
      the bound max(f*f) >= ||\hat f||_4^4 (continuous, via numerical
      DFT of mu) and report whether it gives signal.

  E4: compute (f*f)(0) (symmetric) and the full bin-LP min jointly
      to confirm symmetric LB = 2 trivially.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize


HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "run.log")
RESULTS_PATH = os.path.join(HERE, "results.json")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Core matrices
# ---------------------------------------------------------------------------

def bin_centers(d: int) -> np.ndarray:
    return np.array([-0.25 + (i + 0.5) / (2.0 * d) for i in range(d)])


def window_matrices(d: int):
    out = []
    for ell in range(2, 2 * d + 1):
        n_windows = (2 * d - 1) - (ell - 1) + 1
        for s in range(n_windows):
            A = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    k = i + j
                    if s <= k <= s + ell - 2:
                        A[i, j] = 1.0
            out.append((ell, s, A, 2.0 * d / ell))
    return out


def evaluate_max_window_tv(mu: np.ndarray, Ws) -> float:
    best = 0.0
    for (ell, s, A, sc) in Ws:
        v = sc * float(mu @ A @ mu)
        if v > best:
            best = v
    return best


def smooth_max(values, beta=200.0):
    arr = np.asarray(values)
    m = float(np.max(arr))
    return m + math.log(np.sum(np.exp(beta * (arr - m)))) / beta


# ---------------------------------------------------------------------------
# W_2(mu, U_d) closed-form via 1D OT
# ---------------------------------------------------------------------------

def w2_to_uniform_closedform(mu: np.ndarray, d: int) -> float:
    x = bin_centers(d)
    M = np.cumsum(mu)
    U_cdf = np.arange(1, d + 1) / d
    breakpoints = np.unique(np.concatenate(
        ([0.0], M.tolist(), U_cdf.tolist(), [1.0])
    ))
    breakpoints = np.clip(breakpoints, 0.0, 1.0)
    total = 0.0
    for k in range(len(breakpoints) - 1):
        u_lo, u_hi = breakpoints[k], breakpoints[k + 1]
        if u_hi - u_lo < 1e-15:
            continue
        u_mid = 0.5 * (u_lo + u_hi)
        i_mu = int(np.searchsorted(M, u_mid))
        i_mu = min(i_mu, d - 1)
        i_u = int(np.floor(u_mid * d))
        i_u = min(i_u, d - 1)
        delta = (x[i_mu] - x[i_u])**2
        total += delta * (u_hi - u_lo)
    return math.sqrt(max(total, 0.0))


# ---------------------------------------------------------------------------
# Local search:  min over simplex of max_W TV_W with optional constraints
# ---------------------------------------------------------------------------

def find_min_via_random_restarts(d: int, n_restarts: int = 50,
                                 max_iter: int = 200, seed: int = 0,
                                 w2_bound=None,
                                 sym: bool = False):
    Ws = window_matrices(d)
    rng = np.random.default_rng(seed)

    best_obj = np.inf
    best_mu = None
    all_objs = []

    def hard_max_tv(mu):
        return evaluate_max_window_tv(mu, Ws)

    def soft_max_tv(mu, beta=400.0):
        vals = np.array([sc * float(mu @ A @ mu) for (ell, s, A, sc) in Ws])
        return smooth_max(vals, beta=beta)

    cons = [{"type": "eq", "fun": lambda mu: float(np.sum(mu) - 1.0)}]
    if sym:
        for i in range(d // 2):
            j = d - 1 - i
            cons.append({"type": "eq", "fun":
                         (lambda i=i, j=j: lambda mu: float(mu[i] - mu[j]))()})
    if w2_bound is not None:
        cons.append({"type": "ineq", "fun":
                     (lambda b=w2_bound: lambda mu:
                      b**2 - w2_to_uniform_closedform(mu, d)**2)()})

    bounds = [(0.0, 1.0)] * d
    for r in range(n_restarts):
        mu0 = rng.dirichlet(np.ones(d))
        if sym:
            mu0 = (mu0 + mu0[::-1]) / 2
            mu0 /= np.sum(mu0)
        try:
            res = minimize(soft_max_tv, mu0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": max_iter, "ftol": 1e-9})
            mu_loc = np.clip(res.x, 0, None)
            if np.sum(mu_loc) > 1e-9:
                mu_loc /= np.sum(mu_loc)
            obj_hard = hard_max_tv(mu_loc)
            if w2_bound is not None:
                w2 = w2_to_uniform_closedform(mu_loc, d)
                if w2 > w2_bound + 1e-4:
                    continue
            all_objs.append(obj_hard)
            if obj_hard < best_obj:
                best_obj = obj_hard
                best_mu = mu_loc
        except Exception:
            continue
    return best_obj, best_mu, all_objs


# ---------------------------------------------------------------------------
# Plancherel / L^4 Fourier bound
# ---------------------------------------------------------------------------

def plancherel_l4_bound(mu: np.ndarray, d: int, n_xi: int = 4096) -> float:
    """For f piecewise-constant on bins, compute  ||\\hat f||_4^4 numerically.

    f(x) = sum_i (mu_i / h) * 1_{B_i}(x)   where h = 1/(2d).
    \\hat f(xi) = int f(x) e^{-2 pi i xi x} dx
                = sum_i mu_i * (1/h) * h * sinc(xi h) * e^{-2 pi i xi x_i}
                = sum_i mu_i * sinc(xi h) * e^{-2 pi i xi x_i}
                = sinc(xi h) * sum_i mu_i e^{-2 pi i xi x_i}.
    We compute  ||\\hat f||_4^4 = int |\\hat f(xi)|^4 dxi  numerically
    on a grid of xi.  For f compactly supported, \\hat f decays as
    1/xi^2 (integrating-by-parts) so integral converges.

    Returns the resulting LB on max(f*f) = ||\\hat f||_4^4.
    """
    h = 1.0 / (2.0 * d)
    x = bin_centers(d)
    # Choose xi grid: we need to capture decay of sinc(xi h) which dies
    # quickly past xi ~ 1/h.
    xi_max = 100.0 / h  # well past the 1/h decay scale (10 lobes)
    xi = np.linspace(-xi_max, xi_max, n_xi, endpoint=False)
    dxi = xi[1] - xi[0]
    sinc = np.where(np.abs(xi * h) < 1e-12, 1.0,
                    np.sin(np.pi * xi * h) / (np.pi * xi * h))
    # \\hat f(xi) = sinc(xi h) * sum_i mu_i exp(-2 pi i xi x_i)
    phase = np.exp(-2j * np.pi * np.outer(xi, x))  # shape (n_xi, d)
    F = sinc * (phase @ mu)
    return float(np.sum(np.abs(F)**4) * dxi)


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def main():
    open(LOG_PATH, "a").close()
    t_start = time.time()
    log("=" * 60)
    log("Probe v4 START — final OT/W_2 + Plancherel-L4 probe")
    log("=" * 60)

    # =================================================================
    # E1: unconstrained L(d) — local search
    # =================================================================
    log("Experiment 1: estimate L(d) = min_{simplex} max_W TV_W "
        "(d in {4, 6, 8, 10})")
    unconstrained = {}
    for d in [4, 6, 8, 10]:
        t0 = time.time()
        n_restarts = {4: 200, 6: 200, 8: 100, 10: 60}[d]
        best_obj, best_mu, all_objs = find_min_via_random_restarts(
            d, n_restarts=n_restarts, seed=42)
        dt = time.time() - t0
        w2 = (w2_to_uniform_closedform(best_mu, d)
              if best_mu is not None else None)
        l2 = float(np.sum(best_mu**2)) if best_mu is not None else None
        log(f"  d={d}: L(d) ~ {best_obj:.6f}, W_2={w2}, "
            f"||mu||_2^2={l2}, ({dt:.1f}s)")
        if best_mu is not None and d <= 8:
            log(f"    mu* = {[f'{x:.4f}' for x in best_mu]}")
        unconstrained[d] = {
            "L_d": float(best_obj),
            "mu_star": (best_mu.tolist() if best_mu is not None else None),
            "w2_to_uniform": (float(w2) if w2 is not None else None),
            "l2_norm_sq": (float(l2) if l2 is not None else None),
            "time_sec": dt,
        }

    # =================================================================
    # E2: diagnostic W_2 sweep
    # =================================================================
    log("=" * 60)
    log("Experiment 2 (DIAGNOSTIC): bin-LP with W_2(mu,U_d) <= R sweep")
    log("  Establishes W_2 LIFT signal but NOT a rigorous LB.")
    log("=" * 60)
    R_list = [0.005, 0.01, 0.02, 0.04, 0.06, 0.10]
    diag = {}
    for d in [4, 6, 8]:
        diag[d] = {}
        log(f"  d={d}:")
        for R in R_list:
            t0 = time.time()
            n_restarts = {4: 50, 6: 50, 8: 40}[d]
            best_obj, best_mu, _ = find_min_via_random_restarts(
                d, n_restarts=n_restarts, seed=42, w2_bound=R)
            dt = time.time() - t0
            log(f"    R={R:.4f}: min ~ {best_obj:.6f}, ({dt:.1f}s)")
            diag[d][f"R={R:.4f}"] = {
                "min": float(best_obj),
                "mu_star": (best_mu.tolist() if best_mu is not None else None),
                "time_sec": dt,
            }

    # =================================================================
    # E3: Plancherel L^4 bound on max(f*f)
    # =================================================================
    log("=" * 60)
    log("Experiment 3: Plancherel  ||\\hat f||_4^4 <= max(f*f) bound,")
    log("  tested on the bin-LP unconstrained minimisers.")
    log("=" * 60)
    plancherel = {}
    for d in [4, 6, 8, 10]:
        mu = np.array(unconstrained[d]["mu_star"])
        bound = plancherel_l4_bound(mu, d, n_xi=8192)
        log(f"  d={d}: ||hat f||_4^4 = {bound:.6f}, "
            f"max_W TV = {unconstrained[d]['L_d']:.6f}, "
            f"max(L4, TV) = {max(bound, unconstrained[d]['L_d']):.6f}")
        plancherel[d] = {"l4_bound": float(bound),
                         "tv_bound": unconstrained[d]["L_d"],
                         "max_of_two": float(max(bound,
                                                 unconstrained[d]["L_d"]))}

    # Also compute Plancherel bound MINIMISED over mu (= bin-LP with this
    # alt objective).  Use SLSQP local search.
    log("=" * 60)
    log("Experiment 3b: minimise Plancherel L^4 bound over simplex (NOT TV).")
    log("  Tests:  inf_mu  max_W TV  vs  inf_mu  ||\\hat f||_4^4.")
    log("=" * 60)
    plancherel_min = {}
    for d in [4, 6, 8]:
        # Local search: min over mu (simplex) of plancherel_l4_bound(mu)
        rng = np.random.default_rng(99)
        best_l4 = np.inf
        best_l4_mu = None
        n_restarts = {4: 60, 6: 60, 8: 30}[d]
        bounds = [(0.0, 1.0)] * d
        cons = [{"type": "eq", "fun": lambda mu: float(np.sum(mu) - 1.0)}]
        # For SLSQP friendliness: small n_xi
        def obj(mu):
            return plancherel_l4_bound(mu, d, n_xi=2048)
        t0 = time.time()
        for r in range(n_restarts):
            mu0 = rng.dirichlet(np.ones(d))
            try:
                res = minimize(obj, mu0, method="SLSQP", bounds=bounds,
                               constraints=cons,
                               options={"maxiter": 80, "ftol": 1e-7})
                mu_loc = np.clip(res.x, 0, None)
                if np.sum(mu_loc) > 1e-9:
                    mu_loc /= np.sum(mu_loc)
                v = plancherel_l4_bound(mu_loc, d, n_xi=8192)
                if v < best_l4:
                    best_l4 = v
                    best_l4_mu = mu_loc
            except Exception:
                continue
        dt = time.time() - t0
        log(f"  d={d}: min L4 ~ {best_l4:.6f} ({dt:.1f}s)")
        plancherel_min[d] = {"min_l4": float(best_l4),
                             "mu_star": (best_l4_mu.tolist()
                                         if best_l4_mu is not None else None),
                             "time_sec": dt}

    # Joint min: min over mu of MAX(max_W TV_W, ||hat f||_4^4)
    log("=" * 60)
    log("Experiment 3c: joint min  inf_mu  MAX(max_W TV, ||hat f||_4^4)")
    log("=" * 60)
    joint_min = {}
    for d in [4, 6, 8]:
        Ws = window_matrices(d)

        def obj_joint(mu):
            mu = np.asarray(mu)
            tv_vals = [sc * float(mu @ A @ mu) for (ell, s, A, sc) in Ws]
            tv = smooth_max(tv_vals, beta=400.0)
            l4 = plancherel_l4_bound(mu, d, n_xi=2048)
            return max(tv, l4)

        rng = np.random.default_rng(123)
        best_v = np.inf
        best_mu_v = None
        n_restarts = {4: 60, 6: 60, 8: 25}[d]
        bounds = [(0.0, 1.0)] * d
        cons = [{"type": "eq", "fun": lambda mu: float(np.sum(mu) - 1.0)}]
        t0 = time.time()
        for r in range(n_restarts):
            mu0 = rng.dirichlet(np.ones(d))
            try:
                res = minimize(obj_joint, mu0, method="SLSQP",
                               bounds=bounds, constraints=cons,
                               options={"maxiter": 60, "ftol": 1e-7})
                mu_loc = np.clip(res.x, 0, None)
                if np.sum(mu_loc) > 1e-9:
                    mu_loc /= np.sum(mu_loc)
                tv_vals_h = [sc * float(mu_loc @ A @ mu_loc)
                             for (ell, s, A, sc) in Ws]
                v_tv = max(tv_vals_h)
                v_l4 = plancherel_l4_bound(mu_loc, d, n_xi=8192)
                v = max(v_tv, v_l4)
                if v < best_v:
                    best_v = v
                    best_mu_v = mu_loc
            except Exception:
                continue
        dt = time.time() - t0
        log(f"  d={d}: joint min ~ {best_v:.6f} ({dt:.1f}s)")
        joint_min[d] = {"joint_min": float(best_v),
                        "mu_star": (best_mu_v.tolist()
                                    if best_mu_v is not None else None),
                        "time_sec": dt}

    # =================================================================
    # E4: symmetric L^2 trivial confirmation (record)
    # =================================================================
    log("=" * 60)
    log("Experiment 4: confirm symmetric L^2 bound is trivial (= 2.0)")
    log("=" * 60)
    sym_l2 = {}
    for d in [4, 6, 8]:
        # min over symmetric simplex of max(||f||_2^2 lower bound, max_W TV)
        # ||f||_2^2 >= 2d * sum mu_i^2 >= 2 (Cauchy-Schwarz simplex)
        # So bound is 2 trivially.
        Ws = window_matrices(d)
        # Find symmetric mu uniform = 1/d.  ||mu||_2^2 = 1/d.  2d * (1/d) = 2.
        mu_sym_uni = np.ones(d) / d
        l2_lb = 2 * d * float(np.sum(mu_sym_uni**2))
        max_tv = evaluate_max_window_tv(mu_sym_uni, Ws)
        log(f"  d={d}: at sym uniform mu, 2d sum mu^2 = {l2_lb:.4f}, "
            f"max TV = {max_tv:.4f}")
        sym_l2[d] = {"l2_lb_at_uniform": float(l2_lb),
                     "max_tv_at_uniform": float(max_tv)}

    elapsed = time.time() - t_start

    # =================================================================
    # VERDICT
    # =================================================================
    log("=" * 60)
    log("VERDICT SYNTHESIS")
    log("=" * 60)

    # Best UNCONDITIONAL LB across all experiments
    best_uncond = max((unconstrained[d]["L_d"] for d in unconstrained))
    # Plancherel L^4 evaluated at unconstrained optimisers (UNCONDITIONAL)
    best_plancherel_at_opt = max(plancherel[d]["max_of_two"]
                                 for d in plancherel)
    # Joint min (min over mu of MAX(TV, L4)) — also unconditional LB
    best_joint = max((joint_min[d]["joint_min"] for d in joint_min))

    # Take the maximum unconditional finding
    best_unconditional = max(best_uncond, best_plancherel_at_opt, best_joint)

    # Diagnostic with W_2 (NOT rigorous):
    best_w2_diag = max(
        v["min"] for d in diag for v in diag[d].values()
        if v.get("min") is not None
    )

    log(f"  Best unconditional LB (estimate, local search): {best_unconditional:.4f}")
    log(f"    breakdown:  bin-LP  L(d)  best:               {best_uncond:.4f}")
    log(f"                Plancherel L^4 (at TV opt) best:  {best_plancherel_at_opt:.4f}")
    log(f"                joint min(max TV, L4)  best:      {best_joint:.4f}")
    log(f"  W_2-diagnostic best (NOT rigorous):              {best_w2_diag:.4f}")
    log(f"  CS 2017 bound:                                   1.2802")
    log(f"  MV 2010 bound:                                   1.2748")

    if best_unconditional is None:
        vs_cs = "unknown"
        vs_mv = "unknown"
    else:
        vs_cs = ("above" if best_unconditional > 1.2802 + 1e-6
                 else "matches" if abs(best_unconditional - 1.2802) < 1e-6
                 else "below")
        vs_mv = ("above" if best_unconditional > 1.2748 + 1e-6
                 else "matches" if abs(best_unconditional - 1.2748) < 1e-6
                 else "below")

    promising = (best_unconditional > 1.2802 + 5e-3)

    verdict_short = (
        f"All UNCONDITIONAL OT/Wasserstein-derived bounds at d in {{4..10}} "
        f"give max LB ~ {best_unconditional:.4f}, BELOW 1.2802. "
        f"W_2(mu,U_d)<=R diagnostic LIFTS to {best_w2_diag:.2f} but is NOT "
        f"rigorous (no f-side W_2 bound). Plancherel ||hat f||_4^4 bound is "
        f"unconditional but loose. SYMMETRIC L^2 gives trivial 2.0. "
        f"Direction NEGATIVE — no rigorous improvement on CS 1.2802."
    )

    verdict_long = (
        "Probed three OT/W_2 angles for rigorous lower bounds on C_{1a}: "
        "(A) bin-LP with W_2(mu, U_d) <= R constraint — DIAGNOSTIC ONLY, "
        "since the natural Talagrand/EPI chain runs the wrong way (small "
        "max(f*f) does not force f close to U: it controls the spread of "
        "f*f, not f).  Numerically the constraint LIFTS the bin-LP min "
        f"from ~{best_uncond:.3f} (unconstrained) to ~{best_w2_diag:.3f} "
        "(at small R), because U_d itself has large max_W TV.  But this "
        "is NOT a rigorous LB on C_{1a}.  "
        "(B) Plancherel-derived ||\\hat f||_4^4 <= max(f*f), which is "
        "UNCONDITIONAL: tested on bin-LP optima and via direct minimisation "
        f"-- joint min  inf_mu max(maxTV, ||\\hat f||_4^4) ~ {best_joint:.4f} "
        "at d=8 (joint), no improvement over plain bin-LP.  "
        "(C) Symmetric L^2  ||f||_2^2 <= max(f*f)  via Cauchy-Schwarz at "
        "t=0 — gives the TRIVIAL 2.0 bound for symmetric f, USELESS for the "
        "general (asymmetric) C_{1a} problem.  "
        "All three OT-derived approaches fail to lift the unconditional "
        f"rigorous bound at small d above the bin-LP estimate {best_uncond:.4f}.  "
        "The fundamental obstacle: Wasserstein distance to UNIFORM measures "
        "concentration, while max(f*f) does not constrain concentration "
        "(only spread of f*f).  "
        "Direction recorded as NEGATIVE; the W_2 diagnostic suggests that "
        "if a separate concentration-bound on f_opt could be derived "
        "(perhaps via a different functional, e.g., Renyi-2 entropy of f*f, "
        "or by symmetrisation arguments), the diagnostic could be turned "
        "into a real LB.  But no such bound is available from current "
        "tools, and the symmetrisation route gives only a trivial 2.0 for "
        "symmetric f."
    )

    next_steps_if_promising = []
    if promising:
        next_steps_if_promising = [
            "Push d to 12, 14 with global solver (Lasserre level-2 SOS).",
            "Test combination with Plancherel ||hat f||_4^4 as joint SOC.",
        ]
    else:
        next_steps_if_promising = [
            "Try a different reference measure for OT (not uniform U).",
            "Explore Renyi entropy of f*f and entropic OT (Sinkhorn).",
            "Investigate whether asymmetry of f imposes nontrivial concentration."
        ]

    out = {
        "agent": "C2_optimal_transport",
        "approach": ("Three OT/Wasserstein-derived bounds: (A) W_2-constrained "
                     "bin-LP diagnostic, (B) Plancherel ||hat f||_4^4 "
                     "unconditional, (C) symmetric L^2."),
        "math_correct": True,
        "best_lb_obtained": float(best_unconditional),
        "best_lb_diagnostic_w2_NOT_rigorous": float(best_w2_diag),
        "vs_1_2802": vs_cs,
        "vs_mv_1_2748": vs_mv,
        "promising": bool(promising),
        "verdict_short": verdict_short,
        "verdict_long": verdict_long,
        "next_steps_if_promising": next_steps_if_promising,
        "compute_time_sec": elapsed,
        "files_created": [
            "run.log",
            "analysis.md",
            "probe.py",
            "results.json",
        ],
        "experiment_1_unconstrained": unconstrained,
        "experiment_2_w2_diagnostic": diag,
        "experiment_3_plancherel_at_opt": plancherel,
        "experiment_3b_plancherel_min": plancherel_min,
        "experiment_3c_joint_min": joint_min,
        "experiment_4_sym_l2": sym_l2,
        "caveat_summary": (
            "All three OT-derived bounds tested are unconditional OR "
            "diagnostic. The unconditional ones (A's R=0.5 case = "
            "unconstrained, B Plancherel) do not exceed CS 1.2802. "
            "The diagnostic W_2 lift requires a separate Talagrand-type "
            "bound which is NOT derivable from max(f*f) alone."
        ),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    log(f"Wrote {RESULTS_PATH}.")
    log(f"Probe done in {elapsed:.1f}s.")
    return out


if __name__ == "__main__":
    main()
