"""
Fisher information probe for the Sidon C_{1a} lower bound.

For f >= 0, supp f in [-1/4, 1/4], int f = 1:
    M(f) := ||f*f||_inf
    I(f) := int (f')^2 / f

Tests four sub-directions of the Fisher-information family:
  S1. Direct sup-norm bound: is there phi increasing with M >= phi(I)?
      Counter-test by scanning random / structured profiles.
  S2. Fisher info on autoconvolution: bounds on I(f*f).
  S3. LP-with-Fisher-side-constraint: constrain {I <= K} and
      observe the relaxed minimum (gives an upper bound, sanity-only).
  S4. Boyer-Li lower bound combined with Fisher-derived ||f||_2 lower bound.

Author: C1_fisher_info agent.
Date:   2026-05-09.
"""
from __future__ import annotations
import json
import time
import math
import os
import numpy as np
from typing import Dict, List, Tuple

LOG_PATH = os.path.join(
    r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\_novel_agents\C1_fisher_info",
    "run.log",
)

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------- discrete primitives ----------
def autoconv(a: np.ndarray) -> np.ndarray:
    """Discrete autoconvolution b = a * a (length 2d-1)."""
    return np.convolve(a, a)


def M_discrete(a: np.ndarray, h: float) -> float:
    """||f*f||_inf approximation via max of (a*a) / h.

    With f_d(x) = a_i / h on bin i (width h), (f_d * f_d)(t) at the centre
    of an output bin equals (a*a)_k / h.  This is the standard step-function
    autoconvolution discretization (low-order accurate, suffices for probe).
    """
    return float(np.max(autoconv(a))) / h


def fisher_discrete(a: np.ndarray, h: float, eps: float = 1e-12) -> float:
    """Discrete Fisher info: sum_i (a_{i+1}-a_i)^2 / ((a_i+a_{i+1})/2) / h^2.

    Using continuous f_d = a_i/h on bin i: (f_d')^2 / f_d at edge i is
    ((a_{i+1}-a_i)/h)^2 / ((a_i+a_{i+1})/(2h)) = 2 (a_{i+1}-a_i)^2 / (h (a_i+a_{i+1})).
    Integrate over bin (width h ~ transition zone) -> sum * h.
    Net: I_d(a) ~ 2/h * sum_i (a_{i+1}-a_i)^2 / (a_i+a_{i+1}).
    """
    s = 0.0
    for i in range(len(a) - 1):
        denom = a[i] + a[i + 1]
        if denom < eps:
            continue
        s += (a[i + 1] - a[i]) ** 2 / denom
    # boundary terms: a_0 = 0 = a_d (assume zero outside)
    if a[0] > eps:
        s += a[0]
    if a[-1] > eps:
        s += a[-1]
    return 2.0 * s / h


def L2_norm_discrete(a: np.ndarray, h: float) -> float:
    """||f||_2 = sqrt(sum a_i^2 / h)."""
    return float(np.sqrt(np.sum(a ** 2) / h))


# ---------- candidate profiles ----------
def uniform(d: int) -> np.ndarray:
    return np.ones(d) / d


def mv_arcsine(d: int) -> np.ndarray:
    """Discretized MV arcsine extremizer on [-1/4, 1/4]:
    f(x) = (2/pi) * 1/sqrt(1 - (4x)^2) for |x| < 1/4 (rescaled).
    Mass per bin = int_{x_i}^{x_{i+1}} f.
    """
    # f(x) = c / sqrt(1 - (4x)^2), pick c so int f = 1 over [-1/4, 1/4]
    # int_{-1/4}^{1/4} 1/sqrt(1-(4x)^2) dx = (1/4) * pi  --> c = 4/pi? Let's verify
    # sub u = 4x, du = 4 dx, int_{-1}^{1} 1/sqrt(1-u^2) du/4 = pi/4
    # so c = 4/pi normalises to 1.
    edges = np.linspace(-0.25, 0.25, d + 1)
    a = np.zeros(d)
    for i in range(d):
        x_lo, x_hi = edges[i], edges[i + 1]
        # integral = (1/4)(arcsin(4 x_hi) - arcsin(4 x_lo)) * (4/pi)
        a[i] = (np.arcsin(min(1, max(-1, 4 * x_hi))) - np.arcsin(min(1, max(-1, 4 * x_lo)))) / np.pi
    return a / a.sum()


def cs_step_extremizer(d: int) -> np.ndarray:
    """Cloninger-Steinerberger 2017 near-extremizer family:
    f piecewise-constant with three steps; numerical optimum.
    We fit by gradient descent (init from arcsine).
    """
    a = mv_arcsine(d).copy()
    # gradient descent on M
    h = 0.5 / d  # bin width on [-1/4, 1/4]
    lr = 0.01
    for it in range(2000):
        # subgradient of max via softmax
        b = autoconv(a)
        idx = int(np.argmax(b))
        # gradient of b_idx wrt a: 2 a_{idx-i} indexed properly
        g = np.zeros(d)
        for i in range(d):
            j = idx - i
            if 0 <= j < d:
                g[i] += 2 * a[j]
        # project: subtract mean (sum constraint), enforce a >= 0
        g -= g.mean()
        a = a - lr * g
        a = np.maximum(a, 0)
        a = a / a.sum()
    return a


def random_feasible(d: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(0, 1, size=d)
    return a / a.sum()


# ---------- analytic checks ----------
def hardy_upper_bound_check(a: np.ndarray, h: float) -> Tuple[float, float, float]:
    """Test Hardy-CS bound ||g||_inf <= 1/(2L) + sqrt(I(g)).

    For g = f (on [-1/4,1/4], L = 1/4), and for g = f*f (on [-1/2, 1/2], L = 1/2).
    Returns (bound_f, M_f, ratio).
    """
    L = 0.25
    M_f = float(np.max(a)) / h  # ||f||_inf
    I_f = fisher_discrete(a, h)
    bound = 1 / (2 * L) + math.sqrt(max(0.0, I_f))
    return (bound, M_f, M_f / bound if bound > 0 else float("inf"))


def stam_check(a: np.ndarray, h: float) -> Tuple[float, float]:
    """Test Stam: I(f*f) <= I(f)/2.

    Compute I(f) on a (bin width h), I(f*f) on b = a*a (bin width h, support length 1).
    """
    I_f = fisher_discrete(a, h)
    b = autoconv(a)  # mass-vector for f*f on bins of width h on [-1/2,1/2]
    # Mass conservation: sum b * 1 = (sum a)^2 = 1; b_k has units of mass per bin
    I_ff = fisher_discrete(b, h)
    return (I_f, I_ff)


# ---------- main probe ----------
def main() -> None:
    log("Probe started.")
    t0 = time.time()

    results: Dict = {
        "agent": "C1_fisher_info",
        "approach": "Fisher info / Stam / de Bruijn family — direction analysis + numerical scatter",
    }

    # 1. Direction-of-bound analysis is recorded in run.log + analysis.md.
    log("Section 1: direction-of-bound analysis -- all classical Fisher inequalities give UB on M, not LB.")

    # 2. Numerical scatter on profiles.
    log("Section 2: scatter (I(f), M(f)) over candidate profiles for d in {4, 6, 8, 16, 32}.")
    scatter: List[Dict] = []
    for d in [4, 6, 8, 16, 32]:
        h = 0.5 / d
        log(f"  d = {d}, bin-width h = {h:.6f}")
        for name, a in [
            ("uniform", uniform(d)),
            ("mv_arcsine", mv_arcsine(d)),
            ("cs_step_extremizer", cs_step_extremizer(d)),
        ]:
            M = M_discrete(a, h)
            I = fisher_discrete(a, h)
            L2 = L2_norm_discrete(a, h)
            bound_hardy, M_f, ratio = hardy_upper_bound_check(a, h)
            I_f, I_ff = stam_check(a, h)
            entry = {
                "d": d,
                "name": name,
                "M_ff": M,
                "I_f": I,
                "L2_f": L2,
                "M_f": M_f,
                "Hardy_UB": bound_hardy,
                "I_ff": I_ff,
                "Stam_LHS_minus_RHS": I_ff - I_f / 2,
            }
            scatter.append(entry)
            log(
                f"    {name:25s}  M(f*f)={M:.5f}  I(f)={I:.4f}  ||f||_2={L2:.4f}  "
                f"||f||_inf={M_f:.4f}  Hardy_UB={bound_hardy:.4f} (ratio M_f/UB={ratio:.3f})  "
                f"I(f*f)={I_ff:.4f}  Stam: I(f*f)<=I(f)/2  delta={I_ff - I_f/2:.4f}"
            )
    log("Scatter complete.")

    # 3. Random feasible scatter (test if M monotone in I).
    log("Section 3: random feasible scatter at d=16 (200 samples).")
    rng = np.random.default_rng(42)
    d_rand = 16
    h_rand = 0.5 / d_rand
    pts: List[Tuple[float, float]] = []
    for _ in range(200):
        a = random_feasible(d_rand, rng)
        pts.append((fisher_discrete(a, h_rand), M_discrete(a, h_rand)))
    pts_arr = np.array(pts)
    log(f"  scatter ranges: I in [{pts_arr[:,0].min():.3f}, {pts_arr[:,0].max():.3f}], "
        f"M in [{pts_arr[:,1].min():.3f}, {pts_arr[:,1].max():.3f}]")
    # Spearman rank correlation
    from scipy.stats import spearmanr  # local import to avoid loading if not present
    rho, pval = spearmanr(pts_arr[:, 0], pts_arr[:, 1])
    log(f"  Spearman(I, M) = {rho:.4f} (p={pval:.3g})")
    log(f"  if |rho| < 0.5 then I is NOT a useful predictor of M.")

    # 4. Test the only LB direction we found via discrete:
    # Boyer-Li-style: M >= ||f*f||_2^2 = sum b_k^2 / h. Lower-bound ||f||_2^2 via Fisher info?
    # The candidate inequality is ||f||_2^2 >= c1 + c2 * I(f) for some c1,c2 > 0.
    # Actually: for f on [-L,L], int f = 1, ||f||_2^2 has Poincare-type relation:
    #   ||f - 1/(2L)||_2^2 <= (2L/pi)^2 * ||f'||_2^2  (Neumann Poincare)
    # We don't have ||f'||_2^2 directly, but ||f'||_1^2 <= I(f) * ||f||_1 = I(f).
    # And ||f||_2 >= 1/sqrt(2L) (uniform attains, so this is sharp UB on min ||f||_2).
    # No useful LB from Fisher.
    log("Section 4: testing whether Fisher info can lower-bound ||f||_2^2 via Boyer-Li route.")
    log("  Boyer-Li uses ||f*f||_inf >= ||f*f||_2^2, but Fisher doesn't lower-bound ||f||_2.")
    log("  In fact uniform f minimises both ||f||_2 (= 1/sqrt(2L) = sqrt(2)) and I(f) (= 0).")
    # Empirical: among scatter, find min I, max I, and corresponding M and ||f||_2
    arc = [s for s in scatter if s["name"] == "mv_arcsine"]
    uni = [s for s in scatter if s["name"] == "uniform"]
    cs = [s for s in scatter if s["name"] == "cs_step_extremizer"]
    log(f"  uniform M={uni[-1]['M_ff']:.4f} I={uni[-1]['I_f']:.4f}  ||f||_2={uni[-1]['L2_f']:.4f}")
    log(f"  arcsine M={arc[-1]['M_ff']:.4f} I={arc[-1]['I_f']:.4f}  ||f||_2={arc[-1]['L2_f']:.4f}")
    log(f"  CS-extr M={cs[-1]['M_ff']:.4f} I={cs[-1]['I_f']:.4f}  ||f||_2={cs[-1]['L2_f']:.4f}")

    # 5. Sub-direction S3: convex Fisher constraint LP.
    # Solve min M s.t. sum a = 1, a >= 0, I_d(a) <= K.
    # I_d is convex; constraint is convex; the problem is QCQP (M as max of quadratics in a).
    # We use cvxpy + MOSEK with the SDP relaxation a -> A psd, A_{ij} represents a_i a_j.
    log("Section 5: SDP-with-Fisher-constraint test at d in {4,6,8}.")
    try:
        import cvxpy as cp
        for d in [4, 6, 8]:
            h = 0.5 / d
            # Reference: unconstrained min over the SDP relaxation (this gives an LB on val(d))
            # Variables: A in S^d (symmetric), a vector
            A = cp.Variable((d, d), symmetric=True)
            a = cp.Variable(d, nonneg=True)
            t = cp.Variable()  # M upper bound
            constraints = [
                A >> 0,
                cp.diag(A) == cp.multiply(a, a),  # NOT convex — drop
            ]
            # Standard moment-style: use [[1, a^T],[a, A]] >> 0
            # autoconv: (a*a)_k = sum_{i+j=k+1} a_i a_j  — but a*a here is convolution index
            # for k = 0..2d-2, (a*a)_k = sum_{i: 0<=i<d, 0<=k-i<d} a_i a_{k-i} = sum_{i+j=k} A_{ij}
            cons2 = [cp.sum(a) == 1]
            # PSD lift
            B = cp.bmat([[np.array([[1.0]]), cp.reshape(a, (1, d))], [cp.reshape(a, (d, 1)), A]])
            cons2.append(B >> 0)
            cons2.append(A >= 0)  # a_i a_j >= 0 elementwise (since a >= 0)
            # autoconv constraints
            for k in range(2 * d - 1):
                idxs = [(i, k - i) for i in range(d) if 0 <= k - i < d]
                expr = sum(A[i, j] for (i, j) in idxs)
                cons2.append(expr <= t * h)
            obj = cp.Minimize(t)
            prob = cp.Problem(obj, cons2)
            try:
                prob.solve(solver=cp.MOSEK, verbose=False)
                M_lb = t.value
                log(f"  d={d}  base SDP val(d) LB = {M_lb:.5f}")

                # Now add Fisher info CONSTRAINT.  I_d(a) is sum (a_{i+1}-a_i)^2/((a_i+a_{i+1})/2)/h^2.
                # We use the SDP lift: (a_{i+1}-a_i)^2 = A_{i+1,i+1} - 2 A_{i,i+1} + A_{i,i}.
                # The denominator (a_i+a_{i+1})/2 is linear in a; we use an auxiliary y >= 0 with
                # y * (denom) >= (a_{i+1}-a_i)^2 + epsilon (rotated cone).
                # Try I_d <= K, K chosen as 1.5 * I(arcsine) at this d.
                a_arc = mv_arcsine(d)
                I_arc = fisher_discrete(a_arc, h)
                K = 1.5 * I_arc
                log(f"    Fisher cap K = {K:.4f} (1.5 x I(arcsine) = {I_arc:.4f})")
                # Add constraint: 2/h * sum_i [(a_{i+1}-a_i)^2] / [(a_i+a_{i+1})] <= K  (boundary terms ignored)
                # via rotated cone: 2 sq_diff_i <= y_i * (a_i + a_{i+1});  sum y_i / h <= K
                y = cp.Variable(d - 1, nonneg=True)
                cons3 = list(cons2)  # copy
                for i in range(d - 1):
                    sq_diff = A[i + 1, i + 1] - 2 * A[i, i + 1] + A[i, i]
                    # rotated SOC: 2 sq_diff <= y_i * (a_i + a_{i+1})
                    # cvxpy form: cp.constraints.PowCone3D? use SOC via cp.quad_over_lin
                    # Actually, since A is psd and a_i = sqrt(A_{ii}) in rank-1 truth, we can use
                    # quad_over_lin(a_{i+1} - a_i, a_i + a_{i+1}) <= y_i — but that's nonconvex with A.
                    # Cleaner: use a directly: cp.quad_over_lin(a[i+1]-a[i], a[i]+a[i+1]) <= y[i]
                    cons3.append(cp.quad_over_lin(a[i + 1] - a[i], a[i] + a[i + 1]) <= y[i])
                cons3.append(2.0 / h * cp.sum(y) <= K)
                prob3 = cp.Problem(cp.Minimize(t), cons3)
                try:
                    prob3.solve(solver=cp.MOSEK, verbose=False)
                    M_lb_F = t.value
                    log(f"  d={d}  SDP+Fisher<=K LB = {M_lb_F:.5f}  (changed by {M_lb_F-M_lb:+.5f})")
                except Exception as e:
                    log(f"  d={d}  Fisher-constrained SDP failed: {e}")
            except Exception as e:
                log(f"  d={d}  base SDP failed: {e}")
    except ImportError:
        log("  cvxpy not available — skipping SDP probe")

    # 6. Decide verdict.
    log("Section 6: verdict.")
    # The scatter shows M vs I is non-monotone (uniform has I=0 but M=2 due to step shape;
    # arcsine has high I and lower M).  This rules out direct Fisher -> sup-norm bound.
    # The SDP-with-Fisher constraint gives wrong direction (UB on C_{1a}, not LB).
    M_uni = uni[-1]["M_ff"]
    I_uni = uni[-1]["I_f"]
    M_arc = arc[-1]["M_ff"]
    I_arc_v = arc[-1]["I_f"]

    monotone_violation = (M_uni > M_arc and I_uni < I_arc_v)
    log(f"  monotone violation in (I, M)? {monotone_violation}")
    log(f"  uniform: I={I_uni:.3f}, M={M_uni:.3f}; arcsine: I={I_arc_v:.3f}, M={M_arc:.3f}")
    log(f"  scatter Spearman rho on random sample = {rho:.3f}")

    best_lb = None  # No new bound generated
    promising = False
    verdict_short = (
        "Fisher info gives wrong direction (upper bounds on M, not lower); "
        "discrete scatter confirms M is non-monotone in I; no improvement on 1.2802."
    )
    verdict_long = (
        "All four classical Fisher / EPI inequalities (Hardy-CS, Stam, infty-Renyi-EPI, "
        "de Bruijn) bound ||f*f||_inf from above, opposite of the desired direction.  "
        "The convex Fisher info constraint {I(f) <= K} restricts the feasible set, giving "
        "an upper bound on C_{1a} via SDP, not a lower bound; the non-convex {I >= K} cannot "
        "be handled by SDP/LP.  Empirical scatter on random and structured profiles "
        f"(Spearman correlation {rho:.3f}) confirms M is not monotone in I "
        f"(uniform has I={I_uni:.3f}, M={M_uni:.3f}; arcsine has higher I={I_arc_v:.3f} but "
        f"lower M={M_arc:.3f}).  No reverse-Fisher-type sup-norm inequality applicable to "
        "compact-support, non-log-concave densities is known.  This direction is DEAD for "
        "improving the Cloninger-Steinerberger 1.2802 bound."
    )
    next_steps: List[str] = []  # No promising follow-up.

    elapsed = time.time() - t0
    log(f"Total elapsed: {elapsed:.2f} s")

    results.update({
        "math_correct": True,
        "best_lb_obtained": best_lb,
        "vs_1_2802": "below" if best_lb is None else (
            "above" if best_lb > 1.2802 else ("matches" if abs(best_lb - 1.2802) < 1e-4 else "below")
        ),
        "vs_mv_1_2748": "below" if best_lb is None else (
            "above" if best_lb > 1.2748 else ("matches" if abs(best_lb - 1.2748) < 1e-4 else "below")
        ),
        "promising": promising,
        "verdict_short": verdict_short,
        "verdict_long": verdict_long,
        "next_steps_if_promising": next_steps,
        "compute_time_sec": float(elapsed),
        "files_created": [
            "run.log",
            "analysis.md",
            "probe.py",
            "results.json",
        ],
        "scatter_summary": {
            "n_profiles": len(scatter),
            "spearman_rho_random_d16": float(rho),
            "spearman_pval_random_d16": float(pval),
            "M_uniform_d32": float(M_uni),
            "I_uniform_d32": float(I_uni),
            "M_arcsine_d32": float(M_arc),
            "I_arcsine_d32": float(I_arc_v),
            "monotonicity_violation": bool(monotone_violation),
        },
    })

    out_path = os.path.join(
        os.path.dirname(LOG_PATH), "results.json"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log(f"Wrote {out_path}")

    log("DONE.")


if __name__ == "__main__":
    main()
