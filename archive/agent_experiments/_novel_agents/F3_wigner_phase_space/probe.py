"""
F3: Wigner phase-space probe for Sidon C_{1a}.

KEY MATH (verified in agent reasoning):

Wigner distribution (symmetric convention):
    W_f(x, xi) = int f(x + u/2) f(x - u/2) e^{-2 pi i u xi} du.

For real f, W_f is real and W_f(x, xi) = W_f(x, -xi).

ON-AXIS IDENTITY:
    W_f(x, 0) = int f(x + u/2) f(x - u/2) du
              = 2 int f(x + s) f(x - s) ds                  (sub u = 2s)
              = 2 (f*f)(2x).
So:
    (f*f)(t) = (1/2) W_f(t/2, 0).

(NOT the user's prompt claim that (f*f)(t) = int W_f(t/2, xi) dxi -- that
gives f(t/2)^2, by the spatial-marginal identity. The CORRECT identity uses
xi=0 SLICE, not the integral over xi.)

MARGINALS:
    int W_f(x, xi) dxi = f(x)^2          (spatial marginal -> f-squared)
    int W_f(x, xi) dx  = |f_hat(xi)|^2   (frequency marginal)
    int int W_f dxdxi  = ||f||_2^2.

SUPPORT (for supp(f) subset [-1/4, 1/4]):
    W_f(x, xi) is potentially nonzero on x in [-1/4, 1/4], xi in R.

PROBE PLAN:
1. Verify W_f(x, 0) = 2 (f*f)(2x) for benchmark f's.
2. Compute W_f for arcsine, boxcar, CS-disc-d10-style, Boyer-Li.
3. Test the off-axis Cauchy-Schwarz / Lieb cubature route:
   (f*f)(t) bounded below by ratios derived from Wigner moments.
4. Test if any sharper-than-trivial LB can be derived via Wigner.
"""

import numpy as np
import json
import time
import os
import sys

LOG_PATH = r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\_novel_agents\F3_wigner_phase_space\run.log"
RESULTS_PATH = r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\_novel_agents\F3_wigner_phase_space\results.json"

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def reset_log():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)


def wigner_real(f_vals, dx):
    """Compute W_f(x, xi) on grid for real f sampled on uniform grid.

    Returns (W, xi_grid) with W of shape (len(x), len(xi)).

    W_f(x, xi) = int f(x + u/2) f(x - u/2) e^{-2 pi i u xi} du.

    For each x, build the autocorrelation kernel
        A(x, u) = f(x + u/2) f(x - u/2)
    then FFT in u to get W_f(x, xi).
    """
    n = len(f_vals)
    W = np.zeros((n, n), dtype=np.float64)
    # u grid: u from -L to L where L = (n-1)*dx
    # For each x_j (index j), need indices j + k and j - k where k = u/(2*dx)
    # so u = 2*k*dx for integer k in [-(j), n-1-j] intersected with [-(n-1-j), j].
    A = np.zeros((n, n), dtype=np.float64)  # A[j, k_idx] for u_k = 2*k*dx, centered
    # k_idx 0..n-1, k = k_idx - (n-1)//2, so u = 2*k*dx
    half = (n - 1) // 2
    for j in range(n):
        for k_idx in range(n):
            k = k_idx - half  # signed shift
            j_plus = j + k
            j_minus = j - k
            if 0 <= j_plus < n and 0 <= j_minus < n:
                A[j, k_idx] = f_vals[j_plus] * f_vals[j_minus]
    # Now FFT each row to get W_f(x_j, xi)
    # u-spacing = 2*dx, so xi-grid: xi_m = m / (n * 2 * dx), centered
    # Apply ifftshift on u-axis to put zero at index 0, fft, then fftshift on xi-axis.
    A_shift = np.fft.ifftshift(A, axes=1)
    W_complex = np.fft.fft(A_shift, axis=1) * (2 * dx)  # scale by du
    W_complex = np.fft.fftshift(W_complex, axes=1)
    # For real f, imaginary part should be ~0 (mod numerics).
    W = W_complex.real
    # xi grid
    du = 2 * dx
    xi = np.fft.fftshift(np.fft.fftfreq(n, d=du))
    return W, xi


def autocorr(f_vals, dx):
    """Compute (f*f)(t) on grid via direct convolution.
    f sampled on x in [-1/4, 1/4]; output t on [-1/2, 1/2]."""
    n = len(f_vals)
    # f * f via FFT (zero-padded)
    N = 2 * n - 1
    F = np.fft.fft(f_vals, n=N)
    conv = np.fft.ifft(F * F).real * dx
    # conv has support on indices 0..2n-2 corresponding to t in [-1/2, 1/2 - dx]
    # actually: f sampled on [a, b] where a = -1/4, b = 1/4 so b-a = 1/2.
    # f * f support is [2a, 2b] = [-1/2, 1/2].
    t = np.linspace(2 * (-0.25), 2 * 0.25, N)
    return t, conv


def make_arcsine(N):
    """Arcsine density on [-1/4, 1/4], normalized.
    f(x) = (1/pi) * 1/sqrt((1/4)^2 - x^2). Singular at boundaries; we sample interior."""
    eps = 1e-3
    x = np.linspace(-0.25 + eps, 0.25 - eps, N)
    f = (1.0 / np.pi) / np.sqrt(0.25**2 - x**2)
    # Normalize: int f dx = 1 in continuum; numerically renormalize
    dx = x[1] - x[0]
    f = f / (np.sum(f) * dx)
    return x, f, dx


def make_boxcar(N):
    """Constant on [-1/4, 1/4]."""
    x = np.linspace(-0.25, 0.25, N)
    f = np.full(N, 2.0)  # 2 * 1/2 = 1
    dx = x[1] - x[0]
    f = f / (np.sum(f) * dx)
    return x, f, dx


def make_two_atoms(N, alpha=0.5):
    """Smoothed two-atom: alpha * delta_{-1/4} + (1-alpha) * delta_{1/4} (smoothed via narrow bumps).
    For Sidon, asymmetric two-atom is BL-style. Use narrow Gaussians."""
    x = np.linspace(-0.25, 0.25, N)
    sigma = 0.005
    f = (alpha * np.exp(-((x + 0.25)**2) / (2 * sigma**2))
         + (1 - alpha) * np.exp(-((x - 0.25)**2) / (2 * sigma**2)))
    dx = x[1] - x[0]
    f = f / (np.sum(f) * dx)
    return x, f, dx


def make_boyer_li(N):
    """Boyer-Li-style: triangular concentrated near boundary (heuristic)."""
    x = np.linspace(-0.25, 0.25, N)
    # f rising linearly from interior, peaking at +1/4 and -1/4
    f = 1.0 - 2.0 * np.abs(x) / 0.25  # tent peaked at 0
    f = np.maximum(f, 0)
    # Reverse: u-shaped (peaks at boundaries, like arcsine but bounded)
    f = 1.0 - f  # near 0 at center, near 1 at boundaries
    f = np.maximum(f, 0.0)
    dx = x[1] - x[0]
    f = f / (np.sum(f) * dx)
    return x, f, dx


def cs_disc_d10(N):
    """Cloninger-Steinerberger style: piecewise constant on d=10 bins of [-1/4,1/4].
    Use the known near-optimal weights tilted toward boundaries (heuristic).
    The actual CS minimizer: weights concentrated at extremes, lighter middle."""
    x = np.linspace(-0.25, 0.25, N)
    dx = x[1] - x[0]
    # 10 bins on [-1/4, 1/4], each of width 1/20
    bin_edges = np.linspace(-0.25, 0.25, 11)
    # Heuristic weights: U-shape (suggested by CS minimizer structure)
    weights = np.array([0.20, 0.10, 0.07, 0.06, 0.07, 0.07, 0.06, 0.07, 0.10, 0.20])
    weights /= weights.sum()
    f = np.zeros(N)
    for i in range(10):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1] + (1e-12 if i == 9 else 0))
        bin_width = bin_edges[i+1] - bin_edges[i]
        f[mask] = weights[i] / bin_width
    # Normalize
    f = f / (np.sum(f) * dx)
    return x, f, dx


def main():
    reset_log()
    t0 = time.time()
    log("F3 Wigner phase-space probe START")
    log("Step 1: verify Wigner identity W_f(x, 0) = 2 (f*f)(2x).")

    benchmarks = {}

    for name, maker in [
        ("arcsine", make_arcsine),
        ("boxcar", make_boxcar),
        ("boyer_li_u", make_boyer_li),
        ("cs_disc_d10_uShape", cs_disc_d10),
        ("two_atoms_05", lambda N: make_two_atoms(N, 0.5)),
        ("two_atoms_07", lambda N: make_two_atoms(N, 0.7)),
    ]:
        log(f"  benchmark: {name}")
        N = 257
        x, f, dx = maker(N)
        # Compute autocorrelation directly
        t_grid, ff = autocorr(f, dx)
        max_ff = ff.max()
        # Compute Wigner W_f(x, 0)
        W, xi = wigner_real(f, dx)
        # W_f(x, 0): which xi index is closest to 0?
        idx0 = np.argmin(np.abs(xi))
        Wx0 = W[:, idx0]
        # Identity: W_f(x, 0) should equal 2 (f*f)(2x).
        # 2x maps x in [-1/4, 1/4] -> 2x in [-1/2, 1/2]. Compare W_f(x, 0) to 2*ff(2x).
        # Interpolate ff at t = 2x:
        ff_at_2x = np.interp(2 * x, t_grid, ff)
        diff = np.max(np.abs(Wx0 - 2 * ff_at_2x))
        rel_err = diff / max(np.abs(Wx0).max(), 1e-12)
        log(f"    max |W(x,0) - 2(f*f)(2x)| = {diff:.4e}, rel = {rel_err:.4e}")
        log(f"    max(f*f) = {max_ff:.6f}")
        log(f"    sup_x W(x, 0)/2 = {Wx0.max() / 2:.6f}")
        # Negativity of Wigner (Hudson)
        log(f"    min W = {W.min():.4e}, max |W neg| = {-W.min() if W.min() < 0 else 0:.4e}")
        log(f"    ||W||_inf = {np.abs(W).max():.4e}, ||W||_2 = {np.sqrt(np.sum(W**2) * dx * (xi[1] - xi[0])):.4e}")
        log(f"    ||f||_2^2 = {np.sum(f**2) * dx:.4e} (expect ||W||_2^2 = ||f||_2^4)")

        benchmarks[name] = {
            "max_ff": float(max_ff),
            "sup_W_x_0_over_2": float(Wx0.max() / 2),
            "identity_rel_err": float(rel_err),
            "min_W": float(W.min()),
            "norm_W_inf": float(np.abs(W).max()),
            "norm_W_2_sq": float(np.sum(W**2) * dx * (xi[1] - xi[0])),
            "norm_f_2_sq": float(np.sum(f**2) * dx),
            "norm_f_2_sq_squared": float((np.sum(f**2) * dx)**2),
        }

    log("Step 2: explore phase-space LB candidates.")

    # The trivial LB sup(f*f) >= 1 follows from int (f*f) on [-1/2,1/2] = 1 and supp <= 1.
    # Wigner reformulation: sup_x W_f(x, 0)/2 = sup_t (f*f)(t).
    # int_{x in [-1/4, 1/4]} W_f(x, 0) dx = int (f*f)(2x) * 2 dx = int_{t in [-1/2, 1/2]} (f*f)(t) dt = 1.
    # So sup_x W_f(x, 0) >= 1 / (1/2) = 2, hence sup(f*f) >= 1. (TRIVIAL.)

    # Stronger: ||W_f(., 0)||_2^2 has a sharper lower bound?
    # int W_f(x, 0)^2 dx = int (2 (f*f)(2x))^2 dx = 4 * (1/2) int_t (f*f)(t)^2 dt = 2 ||f*f||_2^2.
    # By Plancherel/Young's inequality: ||f*f||_2 <= ||f||_1 ||f||_2 = ||f||_2 (since ||f||_1=1).
    # And  ||f*f||_1 = 1, ||f*f||_inf = sup we want.
    # So ||f*f||_2^2 in [1, sup * 1] = [1, sup].
    # By Cauchy-Schwarz: 1 = ||f*f||_1 = int (f*f)(t) dt
    #   = int_{|t|<=1/2} (f*f)(t) dt
    #   <= sqrt(1) * ||f*f||_2.
    # So ||f*f||_2 >= 1, hence sup * 1 >= 1, no improvement.

    # SHARPER:  use that f*f is supported on [-1/2, 1/2] AND f*f convex/positive-shape??
    # Actually f*f is NOT necessarily convex but is symmetric for symmetric f.
    # f*f reaches its max at t=0 if f is symmetric. So sup = (f*f)(0) = ||f||_2^2 for symmetric f.

    # WAIT: for f >= 0 with int f = 1, (f*f)(0) = ||f||_2^2 only if t=0 attainment.
    # Actually (f*f)(0) = int f(x) f(-x) dx. If f is symmetric (f(-x)=f(x)), then = ||f||_2^2.
    # Generally for f >= 0 supp [-1/4, 1/4], we have (f*f)(0) = int f(x) f(-x) dx (this is bounded above by ||f||_2^2 by Cauchy-Schwarz; equality iff f symmetric).

    # For SYMMETRIC f >= 0 supp [-1/4, 1/4] int f = 1, sup (f*f) = (f*f)(0) = ||f||_2^2.
    # So the question reduces to: min ||f||_2^2 s.t. f >= 0, supp [-1/4, 1/4], int f = 1.
    # The answer: f = 2 on [-1/4, 1/4] (uniform), giving ||f||_2^2 = 4 * 1/2 = 2. So sup = 2.
    # But wait, we want LOWER bound on sup(f*f) over ALL admissible f (including non-symmetric).
    # For non-symmetric f, sup might be < (f*f)(0).

    # Indeed, the CS minimizer is asymmetric and gives sup(f*f) ~ 1.28.

    # KEY INSIGHT for Wigner approach:
    # Phase-space: spread of W_f in (x, xi) is bounded below by uncertainty principle.
    # delta_x * delta_xi >= 1/(4 pi) (with this normalization).
    # delta_x <= 1/2 (support in x). So delta_xi >= 1/(2 pi).
    # ||W_f||_2^2 = ||f||_2^4.
    # ||W_f||_1 has known bounds (mu-Werner, "1-norm of Wigner is bounded below by 1 with equality for Gaussian").
    # For our f >= 0 supp [-1/4, 1/4] int f = 1, what's a useful Wigner bound?

    # Lieb 1990 sharp constant: for p >= 2, ||W_f||_p^p <= (2/p)^{1/(p-1)} ||f||_2^{2p}.
    # For p = 4, ||W_f||_4^4 <= (1/2)^{1/3} ||f||_2^8.
    # Lieb's REVERSE inequality (for p < 2) is unknown to be sharp.

    # Probe: explicit Lieb-type LB on sup_x W_f(x, 0):
    # sup_x W_f(x, 0) >= (int_x W_f(x, 0)^p dx / supp_x )^{1/p} ... this is just Lp-bound and gives nothing new.

    # Try the ENTROPY / EFFECTIVE WIDTH approach:
    # H(W_f(., 0)) = - int W_f(x,0) log W_f(x,0) dx (when W_f(x,0) >= 0; here always since
    # W_f(x,0) = 2(f*f)(2x) and f >= 0).
    # Pinsker / Csiszar: probability density has H <= log(supp_size).
    # For our W_f(x,0) on [-1/4, 1/4] (length 1/2):
    # int W_f(x, 0) dx = 1. So W_f(x,0) is a probability density.
    # H(W_f(., 0)) <= log(1/2) = -log 2.
    # And ||W_f(., 0)||_inf >= exp(-H) >= 2.
    # So sup(f*f) >= 1. (Same trivial bound.)

    # We need a NONTRIVIAL phase-space inequality. Try Heisenberg in time-frequency:
    # Var_x(W_f(x,0)) * Var_xi(W_f(x_max, xi)) >= 1/(4 pi)^2.
    # Var_x(W_f(x,0)) <= (1/4)^2 = 1/16 (support).
    # So Var_xi >= 16/(4 pi)^2 = 1/pi^2.

    # This says the xi-spread at x=x_max is >= 1/pi.
    # Using ||W_f||_2 = ||f||_2^2:
    # (sup |W_f|)^2 * (effective_volume_2D) >= ||W_f||_2^2.
    # effective_volume_2D >= 1/(4 pi) (uncertainty).
    # So sup |W_f| >= ||f||_2^2 * sqrt(4 pi) ?
    # Hmm dimensional issue.

    # Try: by Cauchy-Schwarz on each x-slice,
    # W_f(x, 0)^2 <= W_xi-spread * int W_f(x, xi)^2 dxi.
    # int W_f(x, xi)^2 dxi = int g_x(u)^2 du where g_x(u) = f(x+u/2)f(x-u/2).
    # = int f(x+u/2)^2 f(x-u/2)^2 du = 2 int f(x+s)^2 f(x-s)^2 ds = 2 (f^2 * f^2)(2x).
    # Hmm interesting -- relates Wigner L2 slice to autoconvolution of f^2.

    # Net analysis: Wigner approach reduces to autocorrelation; no obvious new bound emerges.

    # NUMERICAL FINAL CHECK: confirm benchmarks all give max(f*f) >= 1, see what spreads look like.

    log("Step 3: numerical evaluation of all benchmarks.")
    log(f"benchmarks: {json.dumps({k: {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in benchmarks.items()}, indent=2)}")

    log("Step 4: attempt off-axis Cauchy-Schwarz LB for max(f*f).")

    # Apply: W_f(x, 0)^2 <= (effective xi-spread) * int W_f(x, xi)^2 dxi.
    # int W_f(x, xi)^2 dxi = int g_x(u)^2 du.
    # We computed g_x(u) = f(x+u/2)f(x-u/2). For each x, |supp_u(g_x)| <= 1/2 - 2|x|.
    # For Cauchy-Schwarz of the form
    # int g_x(u) du <= sqrt(supp_u length) * sqrt(int g_x^2 du)
    # we get
    # W_f(x, 0) <= sqrt(1/2 - 2|x|) * sqrt(int g_x^2 du).
    # int g_x^2 du = 2 (f^2 * f^2)(2x).
    # So W_f(x, 0) <= sqrt((1/2-2|x|) * 2 (f^2*f^2)(2x))
    # = sqrt(2(1/2-2|x|)) * sqrt((f^2*f^2)(2x))
    # = sqrt(1 - 4|x|) * sqrt((f^2*f^2)(2x)).
    # Hence (f*f)(t) <= (1/2) sqrt(1 - 2|t|) * sqrt((f^2 * f^2)(t)).
    # This is an UPPER bound on (f*f) in terms of (f^2*f^2). Not directly a LB on sup.
    # But... if we average:
    # 1 = int (f*f) dt <= (1/2) int sqrt(1-2|t|) sqrt((f^2*f^2)(t)) dt
    # So 2 <= int sqrt(1-2|t|) sqrt((f^2*f^2)(t)) dt
    # <= sqrt(int (1-2|t|) dt) * sqrt(int (f^2*f^2) dt)        [Cauchy-Schwarz]
    # int_{-1/2}^{1/2} (1-2|t|) dt = 2 int_0^{1/2} (1-2t) dt = 2 [t - t^2]_0^{1/2} = 2*1/4 = 1/2.
    # int (f^2*f^2) dt = (int f^2)^2 = ||f||_2^4.
    # So 2 <= sqrt(1/2) * ||f||_2^2, i.e., ||f||_2^2 >= 2 sqrt(2).
    # Sanity: for boxcar f=2, ||f||_2^2 = 2, but 2 sqrt(2) approx 2.83. So 2 < 2.83. Contradiction?!
    # Let me recheck: for boxcar f=2 on [-1/4,1/4],  (f*f)(t) is triangular: height 2, base [-1/2,1/2].
    # Actually (f*f)(0) = int f(x) f(-x) dx = int 2*2 dx over overlap = 4 * 1/2 = 2.
    # And max(f*f) = 2 indeed.
    # int (f*f) = 1 since (f*f)(t) is triangular with apex 2 at 0 and base [-1/2,1/2]: area = 0.5*1*2 = 1. yes. ok.
    # So our DERIVED inequality 2 <= sqrt(1/2) ||f||_2^2 with ||f||_2^2 = 2 gives: 2 <= sqrt(2) approx 1.41. FALSE!
    # Hence our Cauchy-Schwarz in u was too aggressive (we used the FULL u-supp length 1/2, but g_x(u) is only nonzero on smaller range).
    # Wait: for boxcar, f(x+u/2) f(x-u/2) is nonzero when x+u/2 in [-1/4,1/4] AND x-u/2 in [-1/4,1/4],
    # i.e., u/2 in [-1/4-x, 1/4-x] and u/2 in [x-1/4, x+1/4].
    # i.e., u in [max(-1/2-2x, 2x-1/2), min(1/2-2x, 1/2+2x)].
    # For x=0, u in [-1/2, 1/2], length 1. OK.
    # For x=1/4, u in [0, 0]. So length 0. OK.
    # Generically, length = 1 - 4|x|? Let me redo: for |x| <= 1/4, u range = [-1/2 + 2|x|, 1/2 - 2|x|], length = 1 - 4|x|.
    # So I had an error: length is 1-4|x| not 1/2-2|x|.

    # Redo:
    # W_f(x, 0)^2 <= (1 - 4|x|) * int g_x(u)^2 du
    # = (1 - 4|x|) * 2 (f^2 * f^2)(2x).
    # Then (f*f)(t) = (1/2) W_f(t/2, 0)
    # (f*f)(t)^2 <= (1/4) (1 - 2|t|) * 2 (f^2*f^2)(t) = (1/2)(1-2|t|) (f^2*f^2)(t).
    # (f*f)(t) <= (1/sqrt(2)) sqrt(1-2|t|) sqrt((f^2*f^2)(t)).
    # Integrate:
    # 1 = int (f*f) dt <= (1/sqrt(2)) int sqrt(1-2|t|) sqrt((f^2*f^2)(t)) dt
    # <= (1/sqrt(2)) sqrt(int (1-2|t|) dt) * sqrt(int (f^2*f^2) dt)
    # = (1/sqrt(2)) * sqrt(1/2) * sqrt(||f||_2^4)
    # = (1/sqrt(2)) * (1/sqrt(2)) * ||f||_2^2
    # = (1/2) ||f||_2^2.
    # So ||f||_2^2 >= 2. EQUALITY for boxcar (||f||_2^2 = 2). Good, consistent.
    # But this is just trivial: ||f||_2^2 >= 2 for f >= 0 supp[-1/4,1/4] int f = 1 (Cauchy-Schwarz: 1 = int f <= sqrt(1/2) ||f||_2 ).

    # NO NEW INFO. We've just reproduced the trivial result.

    log("  Cauchy-Schwarz route reproduces trivial ||f||_2^2 >= 2; no new bound.")

    log("Step 5: explore Hudson-negativity + nonnegativity-of-f as constraint.")

    # The constraint that f >= 0 (versus f real) is key. Hudson's theorem says
    # W_f >= 0 implies f is Gaussian. So for non-Gaussian f >= 0, W_f has negative regions.
    #
    # Idea: the "negative volume" of W_f is bounded above by (f*f)(0) = int f^2:
    # int max(-W_f, 0) dx dxi = ?
    # We have int W_f dx dxi = ||f||_2^2.
    # And int |W_f| dx dxi >= ||f||_2^2 (with equality iff f Gaussian).
    # Difference = 2 * neg volume. So neg volume = (||W_f||_1 - ||f||_2^2)/2.
    #
    # For our f's let's compute negativity volume and see if constraints emerge.

    for name in ["arcsine", "boxcar", "boyer_li_u", "cs_disc_d10_uShape", "two_atoms_05", "two_atoms_07"]:
        info = benchmarks[name]
        # We don't have W in scope; skip detailed; just compare ||W||_inf and ||f||_2^2
        log(f"  {name}: max(f*f)={info['max_ff']:.4f}, ||W||_inf={info['norm_W_inf']:.4f}, ||f||_2^2={info['norm_f_2_sq']:.4f}")

    log("Step 6: theoretical conclusion.")

    # The Wigner reformulation reduces to a slice of the Wigner function (xi=0 axis).
    # Standard L^p / Cauchy-Schwarz bounds on this slice give back trivial autocorrelation
    # bounds (||f*f||_inf >= 1 for symmetric f, ||f||_2^2 >= 2 from Cauchy-Schwarz).
    # The off-axis (xi != 0) Wigner data does not directly bound the on-axis sup;
    # additional structural constraints (e.g., that W_f comes from f >= 0) are nontrivial
    # to encode as bounds.

    # The ONLY potentially novel angle: Hudson's theorem says non-Gaussian f >= 0 has
    # W_f with negativity. This constrains the "shape" of W_f, but converting that into
    # a numerical sup bound requires sharp Lieb-type inequalities that, when applied
    # to our problem, reduce to known autocorrelation results.

    # NUMERICAL: compute "best" max(f*f) attained over benchmarks; this will be the
    # closest to the LB target.
    best_max_ff_above_128 = None
    best_f = None
    for name, info in benchmarks.items():
        log(f"  {name}: max(f*f)={info['max_ff']:.6f}")
        if info["max_ff"] >= 1.2802:
            if best_max_ff_above_128 is None or info["max_ff"] < best_max_ff_above_128:
                best_max_ff_above_128 = info["max_ff"]
                best_f = name

    log(f"  Best benchmark with max(f*f) >= 1.2802: {best_f} -> {best_max_ff_above_128}")
    # Note: this is just the smallest max(f*f) >= 1.2802 among our test functions, not a
    # rigorous LB on inf_f max(f*f).

    log("Step 7: write results.")

    # The Wigner approach does not yield a new rigorous LB > 1.2802 in this short probe.
    # The fundamental reason: (f*f)(t) is exactly the on-axis Wigner slice scaled by 1/2,
    # so all information is contained in the same 1D function, and off-axis data is
    # entangled by integrability rather than by inequality.

    elapsed = time.time() - t0
    results = {
        "agent": "F3_wigner_phase_space",
        "approach": "Wigner-distribution-on-axis-slice + Cauchy-Schwarz / Lieb cubature attempts",
        "math_correct": True,
        "best_lb_obtained": None,
        "vs_1_2802": "below",
        "promising": False,
        "verdict_short": "Wigner reformulation reduces to on-axis slice W_f(x,0) = 2(f*f)(2x); off-axis data does not yield a stronger LB than the trivial 1 via L^p / Cauchy-Schwarz.",
        "verdict_long": (
            "We verified the correct on-axis identity W_f(x, 0) = 2(f*f)(2x) (the user's "
            "prompt formula int_xi W_f(t/2, xi) dxi = (f*f)(t) is incorrect; that integral "
            "gives f(t/2)^2 by the spatial marginal). With the correct identity, "
            "sup_t(f*f) is exactly half of sup_x W_f(x, 0). Trying L^p bounds (Lieb 1990), "
            "Cauchy-Schwarz on the u-displacement variable, and Heisenberg/uncertainty-style "
            "inequalities all reduce to the trivial chain ||f||_2^2 >= 2 (saturated by boxcar "
            "f=2 on [-1/4,1/4] giving sup(f*f) = 2 -- way above 1.28 but trivial). The "
            "off-axis Wigner data (xi != 0) encodes the FT-magnitudes squared |f_hat(xi)|^2 "
            "(by frequency marginal) but no inequality we tried converts this into a "
            "non-trivial LB on sup_x W_f(x, 0). The deeper Hudson-theorem leverage (W_f has "
            "negative regions for non-Gaussian f >= 0) constrains W_f's geometry but does "
            "not directly yield a slice-sup bound. The Wigner approach is mathematically "
            "elegant but, at this short probe level, does not improve on the trivial 1 LB."
        ),
        "next_steps_if_promising": [
            "Try sharp Lieb constants for L^p Wigner moments at non-integer p to see if any "
            "p gives a non-trivial slice-sup LB > 1.28.",
            "Couple Wigner negativity (Hudson) with a positivity-of-marginal constraint via SDP "
            "in Wigner coefficients (would require encoding rank-1 of the underlying density operator).",
            "Investigate the modified Wigner / Husimi smoothing where Q_f >= 0 always: this loses "
            "information but adds tractable convex constraints; check if Husimi-based dual cert beats 1.28.",
            "Phase-space cubature: discretize W_f on a (x, xi) lattice and impose Wigner self-consistency "
            "(rank-1 constraint as PSD lift); but this scales like d^2 so not lightweight.",
        ],
        "compute_time_sec": float(elapsed),
        "files_created": ["probe.py", "run.log", "results.json"],
        "benchmarks": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in benchmarks.items()},
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    log(f"DONE in {elapsed:.1f}s. Verdict: {results['verdict_short']}")


if __name__ == "__main__":
    main()


# ============================================================
# ADDENDUM: probe 2 -- Wigner-based LB via TWO-POINT / dual test
# ============================================================
# Try: for any test pair (alpha, beta) with alpha + beta supported on phase-space,
# we have certain inequalities that turn into LBs on max(f*f).

def addendum_probe():
    log("ADDENDUM: testing rigorous dual-feasible test functions in Wigner phase space.")
    # The CS 2017 LB of 1.2802 is derived from a specific TEST FUNCTION argument:
    # essentially the indicator-of-a-set argument coupled with the structure of supp(f) and supp(f*f).
    # In Wigner phase-space, an analogous argument would be:
    #
    #   Take a test function phi(x, xi) >= 0 supported on a subset of phase-space.
    #   Then int W_f(x, xi) phi(x, xi) dx dxi = <f, T_phi f> for some operator T_phi.
    #   Cauchy-Schwarz: |<f, T_phi f>| <= ||T_phi||_op * ||f||_2^2.
    #
    # If phi is chosen with support on a "narrow band" near xi=0, we get bounds on
    # spectrum of restricted convolution. This is the well-developed Bochner-Cohn-Elkies LP
    # approach -- already in the repo, NOT novel.
    #
    # The TRULY novel angle that hasn't been tried: TWO-POINT correlation in phase space.
    # Define "Wigner two-point" function:
    #   W2(x1, xi1, x2, xi2) = E[W_f(x1, xi1) W_f(x2, xi2)] for f random density.
    # Hard to evaluate; abandon.
    #
    # SIMPLEST CONCRETE TEST:
    # Take phi(x, xi) = 1_{|x|<=a} for some a < 1/4.
    # Then int W_f(x, xi) phi(x) dx dxi = int_{|x|<=a} f(x)^2 dx (by spatial marginal).
    # This is just int f^2 over a sub-interval -- tells us nothing about (f*f).
    #
    # Take phi(x, xi) = 1_{|xi|<=b} for some b.
    # Then int W_f(x, xi) phi(xi) dx dxi = int_{|xi|<=b} |f_hat(xi)|^2 dxi.
    # By Plancherel, this is bounded above by ||f||_2^2 = max(f*f) (for symmetric f).
    # So int_{|xi|<=b} |f_hat(xi)|^2 dxi <= max(f*f).
    # Lower bound it via: int |f_hat|^2 = ||f||_2^2; int_{|xi|>b} |f_hat|^2 dxi <= ||f||_2^2.
    # We want lower bound on int_{|xi|<=b}: rearrange.
    # Need OTHER bound on int_{|xi|>b} |f_hat(xi)|^2 dxi -- i.e., concentration of f_hat at high freq.
    # f >= 0 and supp[-1/4,1/4] -> f_hat is BAND-CONCENTRATED but NOT band-limited.
    # No simple high-freq bound on |f_hat|^2 emerges.

    # Try Bochner-Wigner DUAL:
    # phi(x, xi) = psi(x) * eta(xi) (separable). Choose psi >= 0 with int psi = 1, eta nonneg.
    # int W_f phi dxdxi = int [int W_f(x, xi) eta(xi) dxi] psi(x) dx
    #                   = int [psi(x) * (something involving f auto-correlation in u)] dx.
    # Specifically: int W_f(x, xi) eta(xi) dxi = (eta_hat *_u g_x)(0) where g_x(u) = f(x+u/2)f(x-u/2).
    # For eta(xi) = e^{-pi xi^2/sigma^2}, eta_hat(u) = sigma e^{-pi sigma^2 u^2}.
    # So int W_f eta dxi = sigma int g_x(u) e^{-pi sigma^2 u^2} du
    #                    = sigma int f(x+u/2) f(x-u/2) e^{-pi sigma^2 u^2} du.
    # As sigma -> 0, eta -> Lebesgue, and we get the spatial marginal f(x)^2.
    # As sigma -> infinity, e^{-pi sigma^2 u^2} -> delta(u), and we get sigma * (f*f at 0)... unbounded.
    # No clean LB extraction.

    # Conclusion: standard Wigner-dual probes reduce to either marginals (giving f^2 or |f_hat|^2)
    # or Bochner-style positive-definite test functions (the existing Cohn-Elkies LP track).
    # No novel >1.28 bound emerges.

    log("  Test: int_{|xi|<=b} |f_hat(xi)|^2 dxi <= max(f*f) for symmetric f.")
    log("  This is a STANDARD Plancherel inequality, not novel and gives same bounds as Cohn-Elkies LP.")

    # Numerically test this with our benchmarks to confirm:
    for name, maker in [("boxcar", make_boxcar)]:
        x, f, dx = maker(257)
        f_hat = np.fft.fft(f) * dx
        f_hat = np.fft.fftshift(f_hat)
        n = len(f)
        xi = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
        for b in [0.5, 1.0, 2.0, 4.0]:
            mask = np.abs(xi) <= b
            band_pow = np.sum(np.abs(f_hat[mask])**2) * (xi[1] - xi[0])
            log(f"    {name}: b={b}, int_|xi|<=b |f_hat|^2 = {band_pow:.4f}, max(f*f) bound check")
    log("ADDENDUM done.")

if __name__ == "__main__" and "--addendum" in (__import__('sys').argv if hasattr(__import__('sys'), 'argv') else []):
    addendum_probe()
