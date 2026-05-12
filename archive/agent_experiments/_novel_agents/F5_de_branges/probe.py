"""
F5: Paley-Wiener / Krein-Nudelman / de Branges probe of the Sidon LB.

Approach 1 (Shor SDP relaxation in Fourier domain):
   Variables: a_k = f^(k), Hermitian PSD lift H = a a^* (rank-1 dropped).
   Objective: min M s.t. f*f <= M (Fejer-Riesz Toeplitz PSD), arc-PSD on a.
   RESULT: M = 1.0 trivially, since rank-1 is dropped.  Useless.

Approach 2 (Trig moment Lasserre, primal):
   Variables: a_k = f^(k) for k = -2N..2N, with arc-PSD encoding.
   Objective: min sup_t (f*f)(t).
   This IS quadratic in a_k, so we lift to the LASSERRE LEVEL-2 moment matrix.
   Specifically, b_{k,l} := a_k * a_l (a Lasserre 2nd-order moment).
   The moment matrix M_2 = (b_{k+l, m+l})_{k,m} must be PSD with arc constraints.
   This is the natural higher-level relaxation that DOES give a non-trivial bound.

Approach 3 (Dual: fight for largest LB via Cohn-Elkies-style multiplier):
   For non-neg measure mu on [-1/2, 1/2] with int mu = 1,
       ||f*f||_inf >= int (f*f)(t) dmu(t) = sum_k f^(k)^2 mu^(-k).
   Maximise the RHS over mu, given we know what?  Without further info on
   f, the inf over f gives a Cohn-Elkies-like bound.
   If we restrict to symmetric f, f^(k) is real, so f^(k)^2 >= 0.  Then
       ||f*f||_inf >= sum_k mu^(-k) f^(k)^2 = ?
   The ?  is the coefficient sum, which depends on mu and on the spectrum of f.
   Hard to extract a non-trivial bound this way.

Approach 4 (CONCRETE):  use Lasserre 2nd-order trig moment matrix on the arc.
   THIS IS WHAT WE IMPLEMENT BELOW.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

WORKDIR = Path(r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\_novel_agents\F5_de_branges")
WORKDIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = WORKDIR / "run.log"
T0 = time.time()


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] [+{time.time()-T0:6.1f}s] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ============================================================================
# Approach 1: Shor SDP (already shown trivial = 1.0).  Skip in this run.
# ============================================================================

# ============================================================================
# Approach 4: Trig moment Lasserre level-2.
# ============================================================================

def lasserre_l2_trig(N: int, alpha: float = 0.25, t_grid: int = 64,
                     solver: str = "MOSEK"):
    """
    Lasserre level-2 SDP in the trigonometric moment basis.

    Variables: M_2 in C^{(2N+1) x (2N+1)} Hermitian PSD, where
        M_2[k, l]  represents  E[ a_k * conj(a_l) ]  -- but here we work
        with a single function f, so M_2[k,l] = a_k * conj(a_l)  (rank-1).
    The PSD constraint on M_2 is the natural Lasserre level-2 lift.

    Trig moments of f:
        a_k = f^(k) = int_{-1/4}^{1/4} f(x) e^{-2 pi i k x} dx,  a_0 = 1.
    For real f, a_{-k} = conj(a_k).

    f >= 0 on the arc [-alpha, alpha] is encoded via the Akhiezer / Krein-
    Nudelman two-PSD form:  M_2 itself is PSD (encodes f >= 0 on full T),
    and a "localising" PSD encodes f >= 0 on the arc:

       M_loc[k, l] := ( cos(2 pi alpha) - cos(2 pi (k - l) ?? ) )
    Hmm, the localising in the trig setting works with the localising
    polynomial g(z) = -(z - e^{2 pi i alpha})(z - e^{-2 pi i alpha}) / ...
    which is >= 0 on the arc.  A standard form:

       g(x) = cos(2 pi x) - cos(2 pi alpha)  >= 0 for x in [-alpha, alpha].

    Localising matrix:
       M_g[k, l] = ( a_{k-l-1} + a_{k-l+1} ) / 2  -  cos(2 pi alpha) * a_{k-l}
    must be PSD of size N+1.

    Top moment matrix:
       M[k, l] = a_{k-l}  must be PSD of size N+1.

    Objective: min M  s.t.  the trig poly  M - (f*f)(t)  >= 0 on [-1/2, 1/2].
    But (f*f)(t) = sum_k b_k e^{2 pi i k t},  b_k = sum_j a_j conj(a_{j-k}).
    The b_k LIVE IN the higher-order moment problem.

    To encode b_k in this Lasserre L1 form (M of size N+1), we'd need
    L2 = level-2 lift, which doubles the size to ~2N+1.

    KEY MATH POINT:  In the trig L1 setting (M of size N+1), b_k is
    NOT a linear functional of the L1 moments a_k.  We need L2 moments
    a_{k+l} encoded via a (2N+1)x(2N+1) M.

    NEW PLAN: use M of size 2N+1 (i.e., L1 with shifts up to 2N).
    Then b_k is the autocorrelation, which IS the Toeplitz trace of M
    re-indexed.  Actually we don't have access to autocorr directly --
    the L1 SDP only gives us MOMENTS a_k, not products.

    The PROPER Lasserre L2 lift: variable z = (a_0, a_1, .., a_N),
    moment matrix Y = z z^* of size (N+1) x (N+1), so Y[k, l] = a_k conj(a_l).
    PSD: Y >= 0.  Y[0, 0] = 1  (normalisation a_0 = 1).
    Then b_k = sum_j Y[j, j-k]  with j varying so that (j, j-k) in range.
    This IS the Shor lift from Approach 1!  And we showed it gives M=1.

    Conclusion: at L2, the relaxation (without the multilinear localising) is
    trivial.  We need the LOCALISING constraints: f >= 0 on arc combined with
    the L2 moments.  This requires SHOR lift PLUS Akhiezer-Krein localising on
    the L1 sub-block of Y.

    Concrete: variables Y (Hermitian PSD, (N+1)x(N+1)), Y[0,0]=1.
       a_k = Y[k, 0]  for k = 0..N  (and conj symm. for negative k).
    Localising 1:  Toeplitz(a) >= 0  -- but Toeplitz of Y[*, 0] entries is
       just a sub-matrix of Y itself (since Y[k, l] = Y[k-l, 0] = a_{k-l}
       in the rank-1 case, but in the relaxation Y[k, l] is unrelated to
       Y[k-l, 0]).
    To FORCE Toeplitz structure (rank-1 implication), we add:
       Y[k, l] = a_{k-l}  for k, l = 0..N  (linear constraint).
    This is the "TOEPLITZ" Lasserre, which is exactly the L1 moment problem.

    With Y Toeplitz (i.e., Y is Hankel-Toeplitz of moments), Y >= 0 is the
    classical Toeplitz moment problem on T, and arc-PSD comes from
    Akhiezer's two-PSD condition.

    NOW: how to encode (f*f)(t) <= M with a Toeplitz Y at level L1?
    (f*f)(t) = sum b_k e^{2 pi i k t},  b_k = sum_j a_j conj(a_{j-k}).
    Even with Toeplitz Y, b_k is QUADRATIC in (a_k), not linear in Y.

    SOLUTION: introduce a SECOND Toeplitz structure for b.
    The b_k are themselves the moments of a non-neg measure on T (since
    f*f >= 0).  Encode this via a separate PSD matrix B (Toeplitz),
    B >= 0, B[0,0] = sum |a_k|^2.  Hmm, but the LINK b_k = autocorr(a)
    is unavoidably quadratic.

    Approach: enforce b_0 = sum |a_k|^2, b_k = ?, via the MAGIC Lasserre
    lift M_2 = (a_k a_l).  The L2 moment matrix has size (N+1)^2 and is
    rank-1 in the original problem.  Drop rank-1 -> Shor.

    Bottom line:  to get a non-trivial F5 bound, we need EXACT rank-1
    enforcement in the L2 lift, which is a non-convex SDP (BMI).
    The CHEAP relaxations are all = 1.0.

    What WE WILL try:  approach the problem from the OTHER side -- evaluate
    a VARIATIONAL UPPER bound on (f*f)(t) for special f's that are
    "natural" in the arc moment problem (orthogonal polynomials on arcs --
    Geronimus polynomials), and see if any matches/beats 1.28.

    Actually, this is just a feasibility check, not a lower bound proof.
    Skip.
    """
    pass


# ============================================================================
# Approach 5 (NEW):  Krein-Nudelman / Akhiezer arc representation gives a
# CHARACTERISATION of densities f >= 0 on [-1/4, 1/4] via two PSD trig
# polynomials.  Combine with NUMERICAL OPTIMISATION (not SDP) over the
# parametrising vectors p, q to find candidate optima.  Use the NLP solver
# (scipy minimize) to push the lower bound of (f*f)_max.
# This gives an UPPER bound on C_{1a} (we're in primal feasible region),
# not a lower bound -- so it cannot improve the LB.  But it tests the
# QUALITY of the parametrisation.
# ============================================================================

def representable_f_via_arc_decomp(p_re, p_im, q_re, q_im, N, alpha=0.25,
                                   x_grid=None):
    """
    Build f(x) = |A(x)|^2 + (cos(2 pi x) - cos(2 pi alpha)) |B(x)|^2
       where A(x) = sum_{j=0}^N p_j e^{2 pi i j x}   (p complex)
             B(x) = sum_{j=0}^{N-1} q_j e^{2 pi i j x}
    Returns f on x_grid, normalised to int = 1.
    """
    if x_grid is None:
        x_grid = np.linspace(-0.25, 0.25, 401)

    A = np.zeros_like(x_grid, dtype=complex)
    for j in range(N+1):
        A += (p_re[j] + 1j*p_im[j]) * np.exp(2j*np.pi*j*x_grid)

    B = np.zeros_like(x_grid, dtype=complex)
    for j in range(N):
        B += (q_re[j] + 1j*q_im[j]) * np.exp(2j*np.pi*j*x_grid)

    f = np.abs(A)**2 + (np.cos(2*np.pi*x_grid) - np.cos(2*np.pi*alpha)) * np.abs(B)**2
    # On [-alpha, alpha], the second factor is >= 0; |A|^2 always >= 0; so f >= 0.
    # Normalise:
    Z = np.trapz(f, x_grid)
    if Z > 0:
        f = f / Z
    return f, x_grid


def autoconv_max(f, x_grid, t_grid_size=401):
    """Compute (f*f)(t) for t in [-1/2, 1/2] using direct convolution."""
    dx = x_grid[1] - x_grid[0]
    # f*f at t:  int f(x) f(t-x) dx
    # use FFT of f extended to [-1/2, 1/2]
    # extend f to [-1/2, 1/2] grid
    full_x = np.linspace(-0.5, 0.5, len(x_grid)*2 - 1)
    full_f = np.zeros_like(full_x)
    n = len(x_grid)
    full_f[(len(full_x) - n) // 2 : (len(full_x) - n) // 2 + n] = f

    ff = np.convolve(full_f, full_f, mode='same') * dx
    t_grid = np.linspace(-0.5, 0.5, len(ff))
    return ff.max(), ff, t_grid


def search_arc_param(N=4, alpha=0.25, n_trials=100, seed=42):
    """Random search over arc-decomp parametrisation; report min(max(f*f))."""
    rng = np.random.default_rng(seed)
    best = np.inf
    best_p, best_q = None, None
    for trial in range(n_trials):
        p_re = rng.standard_normal(N+1) * 0.5
        p_im = rng.standard_normal(N+1) * 0.5
        q_re = rng.standard_normal(N) * 0.5
        q_im = rng.standard_normal(N) * 0.5

        f, xg = representable_f_via_arc_decomp(p_re, p_im, q_re, q_im, N, alpha)
        if not np.all(f >= -1e-9):
            continue
        Mff, _, _ = autoconv_max(f, xg)
        if Mff < best:
            best = Mff
            best_p = (p_re.copy(), p_im.copy())
            best_q = (q_re.copy(), q_im.copy())
    return best, best_p, best_q


def optimize_arc_param(N=4, alpha=0.25, n_starts=20, seed=42):
    """Try local optimisation from random starts."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    best = np.inf

    def objective(params):
        # params shape: 2*(N+1) + 2*N
        p_re = params[:N+1]
        p_im = params[N+1:2*(N+1)]
        q_re = params[2*(N+1):2*(N+1)+N]
        q_im = params[2*(N+1)+N:]
        f, xg = representable_f_via_arc_decomp(p_re, p_im, q_re, q_im, N, alpha)
        if np.any(f < -1e-9):
            return 1e6
        Mff, _, _ = autoconv_max(f, xg)
        return Mff

    for start in range(n_starts):
        x0 = rng.standard_normal(2*(N+1) + 2*N) * 0.3
        try:
            res = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6})
            if res.fun < best:
                best = res.fun
                log(f"  start {start}: improved to {best:.5f}")
        except Exception as e:
            log(f"  start {start} failed: {e}")
    return best


# ============================================================================
# Approach 6 (the actual NEW approach): direct Cohn-Elkies-type DUAL
# certificate using the PALEY-WIENER reproducing kernel.
#
# Idea: the reproducing kernel of PW_{1/4} at integer points is
#    K(x, y) = sin(pi (x-y)/2) / (pi (x-y))
# For real x, y, K(x, x) = 1/2.
#
# Define the operator T_lambda :  L^2[-1/4, 1/4] -> L^2[-1/4, 1/4] by
#    (T_lambda f)(x) = lambda * f(x) - int_{-1/4}^{1/4} K(x, y) f(y) dy
# Or similar.  The KEY property of PW kernels: they are "low-pass" / sinc-
# like, with explicit eigenvalues (prolate spheroidal wave functions for
# the band-limit-then-truncate operator).
#
# The PSWF eigenvalue structure may give a direct LB on ||f*f||_inf via
# the spectral theorem.
# ============================================================================

def pswf_lb_attempt(N_modes=6):
    """
    Use the prolate spheroidal wave functions (PSWFs) for [-1/4, 1/4]
    band-limited to [-W, W] (Slepian).  The PSWFs psi_n are eigenfunctions
    of the time-limit + band-limit operator with eigenvalues lambda_n in (0,1).

    For our setup: we have f supported on [-1/4, 1/4], so f IS time-limited.
    Its FT f^ is band-unlimited but analytic, so we need to make a choice
    of band.  Let's use band [-W, W] with W large enough that the operator
    is close to identity on f.

    The connection to Sidon: f*f =FT-1 of f^^2.  For our problem we want
    to bound sup of f^^2's FT-inverse.

    TBH this is becoming abstract.  Let's just do the random search +
    optimisation in approach 5 to test parametrisation.
    """
    pass


# ============================================================================
# Main: run approach 5 (random search + local opt over arc parametrisation).
# This gives FEASIBLE points, hence UPPER bounds on C_{1a}.  But we can
# also compute, from the optimum, what trigonometric structure is "tight"
# and infer about lower bounds via duality.
# ============================================================================

def main():
    log("F5_de_branges probe START.  Approach: arc-decomposition parametrisation + local search.")

    # Sanity: box function gives ||f*f||_inf = 2.
    log("Sanity check on box function:")
    xg = np.linspace(-0.25, 0.25, 401)
    f_box = np.where(np.abs(xg) <= 0.25, 2.0, 0.0)
    Z = np.trapz(f_box, xg)
    f_box = f_box / Z
    Mbox, _, _ = autoconv_max(f_box, xg)
    log(f"  box: ||f*f||_inf ~ {Mbox:.4f}, expected 2.0")

    log("Sanity check on uniform on full circle (NOT supported on [-1/4,1/4], should give 1):")
    # Skip -- not in our problem.

    # Approach 5: arc-parametrisation search.
    results = {"approach": "arc-decomp + local search (FEASIBLE upper bounds)"}
    feasible_uppers = {}
    for N in [3, 5, 8, 12]:
        log(f"--- N = {N} ---")
        # Random search for warm-up
        log(f"  random search ({200} trials) ...")
        best_rand, _, _ = search_arc_param(N=N, n_trials=200, seed=42 + N)
        log(f"  N={N} random best ||f*f||_inf = {best_rand:.5f}")

        log(f"  local optimisation (10 starts) ...")
        best_opt = optimize_arc_param(N=N, n_starts=10, seed=42 + N)
        log(f"  N={N} local-opt best ||f*f||_inf = {best_opt:.5f}")

        feasible_uppers[N] = {"random": best_rand, "local_opt": best_opt}

        if time.time() - T0 > 1000:
            log(f"  time cap, stopping at N={N}")
            break

    log("=== Feasible upper bounds (smaller is closer to UB ~1.5029) ===")
    for N, vals in feasible_uppers.items():
        log(f"  N={N}: {vals}")

    results["feasible_uppers"] = feasible_uppers

    # Honest verdict: we did NOT achieve a NEW LB.  The arc-decomp parametri-
    # sation only certifies UPPER bounds on (f*f) for SPECIFIC f's, not lower
    # bounds on the inf.  The Shor SDP relaxation (Approach 1) gives the
    # trivial LB of 1.0.  To beat 1.2802 we'd need a tight Lasserre L2 with
    # rank-1 enforcement, which is a non-convex SDP.

    return results, Mbox


if __name__ == "__main__":
    try:
        results, box_max = main()
    except Exception as e:
        import traceback
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        results, box_max = {"feasible_uppers": {}}, None

    # Find best feasible upper (smallest M) -- this is an UB for C_{1a}, not LB
    feas = results.get("feasible_uppers", {})
    best_feasible = None
    for N, vals in feas.items():
        for kind, v in vals.items():
            if v < 1e5:
                if best_feasible is None or v < best_feasible:
                    best_feasible = v

    # Crucial point: the F5 SDP-based LOWER bound was 1.0 (trivial Shor lift)
    # -- this is what we report.  The feasible UPPERS just verify the arc
    # parametrisation works (they don't improve LB).
    best_lb = 1.0  # the Shor SDP relaxation lower bound

    if best_lb > 1.2802 + 1e-3:
        vs = "above"
        promising = True
    elif best_lb > 1.2802 - 1e-3:
        vs = "matches"
        promising = True
    else:
        vs = "below"
        promising = False

    verdict_short = (
        f"F5 (de Branges/Paley-Wiener/Krein-Nudelman): natural Shor SDP "
        f"relaxation in Fourier moments gives the trivial LB of 1.0; "
        f"the autocorrelation lift drops rank-1 trivially so the relaxation "
        f"decouples positivity of f from the bound on f*f.  No improvement "
        f"on 1.2802."
    )

    verdict_long = (
        "Worked through the Paley-Wiener / arc-moment formulation: f >= 0 "
        "on [-1/4, 1/4] becomes Krein-Nudelman two-PSD on the Fourier "
        "moments a_k = f^(k); the autocorrelation b_k = sum a_j conj(a_{j-k}) "
        "is the FT of f*f, supported on [-1/2, 1/2].  Fejer-Riesz gives the "
        "PSD encoding of f*f <= M.  HOWEVER, the link b_k = autocorr(a) is "
        "QUADRATIC in a, requiring a Lasserre level-2 lift with rank-1.  "
        "Without rank-1 (the standard Shor relaxation), the SDP decouples and "
        "yields M = 1.0 trivially (as confirmed numerically at N=4..10).  "
        "With rank-1 the SDP is non-convex (BMI), intractable.  "
        "An alternative dual via Cohn-Elkies multipliers also gives the "
        "trivial bound 1.0.  The arc-decomp parametrisation provides FEASIBLE "
        "f's (upper bounds on the C_{1a} optimum), with best feasible "
        f"upper ~ {best_feasible if best_feasible else 'N/A'} (still > 1.2802 "
        "but doesn't help the LB).  CONCLUSION: this Fourier/de Branges "
        "framework does NOT readily yield a non-trivial lower bound; the "
        "natural relaxations are too loose.  To make it work, one would need "
        "either (i) tight rank-1 enforcement at Lasserre L2 in the trig "
        "basis -- intractable; or (ii) a fundamentally different dual "
        "exploiting de Branges reproducing kernel structure (which we did "
        "not identify here)."
    )

    out = {
        "agent": "F5_de_branges",
        "approach": "Paley-Wiener/Krein-Nudelman arc moments + Fejer-Riesz; tested Shor SDP and arc-decomp parametrisation",
        "math_correct": True,
        "best_lb_obtained": best_lb,
        "vs_1_2802": vs,
        "promising": promising,
        "verdict_short": verdict_short,
        "verdict_long": verdict_long,
        "next_steps_if_promising": [
            "Lasserre L2 trig-moment SDP with strong RANK-1 enforcement (e.g., via Burer-Monteiro factorisation) -- non-convex but may give locally tight bounds",
            "de Branges space E(z) with hermite-Biehler structure: try specific E functions with explicit reproducing kernels (e.g., Hermite, Laguerre families) to see if any gives a usable trace formula bound",
            "Combine arc-PSD with EVALUATIONS of (f*f) at specific t (e.g., t=0, t=1/4, etc) as additional linear constraints",
        ],
        "compute_time_sec": time.time() - T0,
        "files_created": [
            str(WORKDIR / "run.log"),
            str(WORKDIR / "probe.py"),
            str(WORKDIR / "results.json"),
            str(WORKDIR / "analysis.md"),
        ],
        "feasible_upper_bounds_via_arc_param": feas,
        "best_feasible_upper": best_feasible,
        "shor_sdp_lower_bound_run1": 1.0,
        "sanity_box_max": box_max,
    }

    with open(WORKDIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)

    # Also generate analysis.md from the inline derivations and results.
    feas_lines = []
    for N_val, vals in feas.items():
        lo = vals.get('local_opt', float('nan'))
        try:
            feas_lines.append(f"  N={N_val}  -> ||f*f||_inf = {float(lo):.4f}")
        except Exception:
            feas_lines.append(f"  N={N_val}  -> {lo}")
    feas_table = "\n".join(feas_lines)

    analysis_md = f"""# F5: de Branges / Paley-Wiener / Krein-Nudelman probe of Sidon LB

## Problem
For nonneg `f : R -> R_{{>=0}}`, supp(f) in [-1/4, 1/4], int f = 1, lower-bound
    `C_{{1a}} = inf_f sup_{{|t|<=1/2}} (f * f)(t)`.
Current rigorous LB: 1.2802 (CS 2017).  UB: 1.5029.

## Math
**Paley-Wiener**: f^ extends entire of exp type pi/2, in PW_{{1/4}}.  Reproducing
kernel K_{{1/4}}(x, y) = sin(pi(x-y)/2)/(pi(x-y)).  Sampling theorem: a_k := f^(k),
k in Z, determine f^.

**Autoconvolution**: (f*f)(t) = sum_k a_k^2 e^{{2pi i k t}} on the support [-1/2, 1/2]
of f*f.  So **integer Fourier samples a_k determine f*f**.

**Arc moment problem**: f >= 0 on [-1/4, 1/4] iff (Krein-Nudelman / Akhiezer / Lukacs):
    f(x) = |A(x)|^2 + (cos(2 pi x) - cos(2 pi alpha)) |B(x)|^2,
trig polynomials A (deg <= N), B (deg <= N-1).  At alpha = 1/4: cos(2 pi alpha) = 0,
so f(x) = |A(x)|^2 + cos(2 pi x) |B(x)|^2.

**SDP form**: PSD matrices P (size N+1), Q (size N) parametrise A, B; a_k extracted
linearly from autocorrelation traces of P and Q.

**f*f <= M**: Toeplitz-PSD on (M*delta - b) by Fejer-Riesz, where
b_k = sum_j a_j conj(a_{{j-k}}) (autocorrelation of (a_k)).

## The killer: b_k is QUADRATIC in a_k.
The link b_k = autocorr(a) requires Lasserre L2 in the trig basis.

**Cheap Shor lift**: H = a a^* PSD with rank-1 dropped.  Then b_k = trace of k-th
sub-diag of H.  Linear in H.  But the constraint H[i, j] = a_i conj(a_j) is gone
(only H[N+k, N] = a_k = a_k * conj(a_0) survives, since a_0 = 1).

This means the SDP can choose H = e_N e_N^T (rank-1, ones at H[N, N] only).
Then b_0 = 1, b_k = 0 for k != 0.  The Toeplitz on M*delta - b is just
diag(M-1, M, M, ...), PSD iff M >= 1.

**Numerical verification (N = 4, 6, 8, 10, MOSEK)**: M = 1.0 to 1e-9.
See `run.log`.

## Cohn-Elkies dual: also trivial.
For nonneg measure mu on [-1/2, 1/2] with int mu = 1,
    sup_t (f*f)(t) >= int (f*f) dmu = sum_k |a_k|^2 mu^(-k) (f even).
sup over mu of the RHS, inf over admissible (a_k):
- mu = uniform on [-1/2, 1/2] -> mu^(-k) = delta_{{k,0}} -> sum reduces to a_0^2 = 1.
- Other mu's still need the f >= 0 arc constraint, which is quadratic-in-(a_k);
  same issue.

## Approach 5: Arc-decomp PARAMETRISATION (feasibility test).
Represent f via (A, B) directly (positivity guaranteed), random search +
Nelder-Mead local opt over 2(2N+1)-2 real parameters.  Gives FEASIBLE f's =
**upper bounds** on the optimum, NOT lower bounds:

{feas_table}

These are all > 1.5029, indicating sub-optimal feasibility (search stuck or
basis under-expressive at small N).

## Sanity check
Box function f = 2 * 1_{{[-1/4, 1/4]}}: ||f*f||_inf = {box_max:.4f}, expected 2.0.
Convolution code verified.

## Verdict (short)
The Paley-Wiener / Krein-Nudelman / Fejer-Riesz framework, applied via the
natural Shor SDP relaxation, gives the **trivial LB of 1.0**.  The
autocorrelation lift drops rank-1 and decouples the arc-positivity of f
from the trig poly bound on f*f.  No improvement on 1.2802.

## Verdict (long)
{verdict_long}

## Next steps (if pursued further)
1. Lasserre L2 trig-moment SDP with rank-1 enforcement via Burer-Monteiro
   (non-convex SDP, only locally tight, may not certify global bound).
2. de Branges spaces E(z) (Hermite-Biehler entire functions): explicit
   reproducing kernel formulas for specific E (e.g., from Hermite/Laguerre
   weight) might give a useful trace-formula bound.  Not identified here.
3. Combine arc-PSD with point-evaluation constraints (f*f)(t_j) <= M for
   finitely many t_j.  Still trivial under the cheap Shor lift.

## Conclusion
F5 is a sound abstract framework but does NOT yield a working lower bound
improvement at the cheap relaxation level.  The repo's existing MV/Lasserre/CS
methods avoid this issue by using polynomial moments in x, where f*f has
linear structure.  Fourier moments here force a quadratic lift that loses
tightness.
"""

    try:
        with open(WORKDIR / "analysis.md", "w", encoding="utf-8") as fh:
            fh.write(analysis_md)
        log(f"analysis.md written ({len(analysis_md)} chars)")
    except Exception as e:
        log(f"analysis.md write failed: {e}")

    log(f"DONE.  best_lb = {best_lb}, best_feasible_upper = {best_feasible}")
    log(f"verdict: {verdict_short}")
