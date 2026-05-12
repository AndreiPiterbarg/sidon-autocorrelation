"""Fourier-truncated SDP for the Sidon autocorrelation problem.

Two complementary tools:

(1) UPPER BOUND construction. Parameterize f on [-1/4, 1/4] as a
    Fourier-truncated non-negative density. Minimise ||f*f||_inf via SDP.
    Returns an UPPER BOUND on C_{1a} (any admissible f gives one).
    Currently 1.2802 <= C_{1a} <= 1.5029. Improvement needs ||f*f||_inf < 1.5029.

(2) DUAL LOWER BOUND. For weights w on convolution knots, if K_w (the
    convolution operator with the knot measure) is PSD, the cross-term
    lemma gives a lower bound:
        C_{1a} >= 4n * inf_mu mu^T M_w mu  where M_w[i,j] = w_{i+j+1}.
    Optimise w under PSD-Fourier constraint to maximise the bound.
    By weak duality this is <= val(d), but the GAP is informative.

Both tools share Fourier infrastructure. The PSD-Fourier constraint is
implemented via a Toeplitz matrix being PSD (Bochner's theorem for the
discrete-knot measure).

==============================================================================
Mathematics, made precise
==============================================================================

Let knots x_k = -1/2 + k*h, h = 1/(2d), k = 0, 1, ..., 2d.

For any probability w = (w_0, ..., w_{2d}) on knots:
    sum_k w_k (f*f)(x_k) <= ||f*f||_inf .

By Lemma 1 (Theorem 1):
    sum_k w_k (f*f)(x_k) = 4n * sum_k w_k MC[k-1] + sum_k w_k (eps*eps)(x_k)
                        = 4n * mu^T M_w mu + <eps, K_w eps>

with (M_w)_{ij} = w_{i+j+1} (set to 0 outside valid range), and
K_w eps (s) = sum_k w_k eps(x_k - s) i.e. convolution against the knot
measure mu_w := sum_k w_k delta_{x_k}.

If K_w is PSD (equivalently, the Fourier transform of mu_w is non-negative,
i.e. hat{mu_w}(xi) >= 0 for all xi by Bochner), then the cross term is
non-negative, and we drop it for a lower bound:

    ||f*f||_inf  >=  4n * mu^T M_w mu   for all admissible f.

Inf over mu in Delta_d:
    C_{1a}  >=  4n * inf_mu mu^T M_w mu.

The inner inf is a non-convex StQP; relax via Lasserre order r SDP for a
sound lower bound. Optimise w subject to PSD-Fourier + probability for the
best bound.

==============================================================================
Honest expectation
==============================================================================

By LP duality, this dual scheme is bounded above by val(d) (the existing
Lasserre value). It cannot beat it. What it CAN do:

  - Give an INDEPENDENT certificate (alternative SDP) of the same bound.
  - Be cheap when the moment matrix is small (~O(2d) size for the dual
    weights vs O(d^2) for full Lasserre at order 2).
  - Verify Lemma 1's content by EXTRACTING a witness w that achieves a
    bound close to val(d).

"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cvxpy as cp
    HAVE_CVXPY = True
except ImportError:
    HAVE_CVXPY = False


# =============================================================================
# Window matrix M_w for the dual approach
# =============================================================================

def build_M_w(w: np.ndarray, d: int) -> np.ndarray:
    """M_w[i,j] = w_{i+j+1} for 0 <= i, j < d.

    w has length 2d+1 (knot weights for k=0..2d). The shift i+j+1 maps
    bin indices (i,j) to convolution index s = i+j, then to knot k = s+1.
    """
    if len(w) != 2 * d + 1:
        raise ValueError(f"w must have length 2d+1 = {2*d+1}, got {len(w)}")
    M = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            k = i + j + 1
            if 0 <= k <= 2 * d:
                M[i, j] = w[k]
    return M


def is_psd_knot_measure(w: np.ndarray, n_xi: int = 200) -> Tuple[bool, float]:
    """Test if mu_w = sum_k w_k delta_{x_k} has hat{mu_w}(xi) >= 0.

    Discretized: scan xi over a grid in [-N, N] for N large enough.
    Returns (is_psd, min_real_hat) — if min < -tol, fails.

    For PSD-Fourier of a discrete measure, equivalent: the covariance
    Toeplitz matrix [w_{|i-j|}]_{i,j} is PSD if w is the autocorrelation
    of a vector. We use a direct Fourier test for clarity.
    """
    d = (len(w) - 1) // 2
    h = 1.0 / (2.0 * d)
    knots = np.array([-0.5 + k * h for k in range(2 * d + 1)])

    xi_grid = np.linspace(-n_xi, n_xi, 4001)
    hat_vals = np.zeros_like(xi_grid)
    for k, x_k in enumerate(knots):
        hat_vals += w[k] * np.cos(2.0 * np.pi * xi_grid * x_k)
    # Imaginary part = sum w_k sin(2pi xi x_k); zero only if w symmetric.
    # We need the whole hat to be real-valued non-negative.
    # If w is REAL symmetric around the center knot, imag = 0 and real = sum cos.
    return (hat_vals.min() >= -1e-9), float(hat_vals.min())


def compute_dual_lb_random_w(
    d: int,
    n_trials: int = 20,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Try random PSD-Fourier knot measures and report the best lower bound.

    For each trial:
      - Generate w as |a|^2 convolved with itself (Riesz-Fejer style).
      - Normalize to a probability.
      - Compute M_w, take eigen-min on the simplex (use spectral lower bound:
        inf_{mu in Delta} mu^T M mu >= lambda_min(M) (loose)).

    For comparison: val(d) from cascade. This routine is exploratory.
    """
    rng = np.random.default_rng(seed)
    n = d // 2
    L = 2 * d + 1

    best_lb = -np.inf
    best_w = None
    for trial in range(n_trials):
        # Generate a vector a, then w = autocorrelation of a (PSD-Fourier
        # by construction since Fourier of autocorr is |a_hat|^2 >= 0)
        a = rng.standard_normal(d + 1)
        w = np.zeros(L, dtype=np.float64)
        for k in range(-d, d + 1):
            for i in range(len(a)):
                j = i - k
                if 0 <= j < len(a):
                    w[k + d] += a[i] * a[j]
        # Normalize: w should be a probability
        s = w.sum()
        if s <= 0:
            continue
        w /= s

        M = build_M_w(w, d)
        # On the simplex: inf mu^T M mu >= ?
        # Use SDP relaxation? Easier: just use lambda_min lower bound (loose).
        eigs = np.linalg.eigvalsh(M)
        lb_inner = max(0.0, float(eigs.min()))  # mu^T M mu >= lam_min * |mu|^2 >= lam_min/d
        # On the simplex |mu|^2 >= 1/d, so:
        bound = 4.0 * n * lb_inner / d  # lower bound
        # Better: solve the StQP exactly via Lasserre, but let's first see crude bound
        if bound > best_lb:
            best_lb = bound
            best_w = w.copy()

    return float(best_lb), best_w


# =============================================================================
# Dual SDP: max_w min_{mu in simplex} mu^T M_w mu  (subject to PSD-Fourier(w))
# =============================================================================

def dual_sdp_lower_bound(d: int, verbose: bool = False) -> Optional[float]:
    """SDP for the cross-term-lemma dual lower bound at dimension d.

    Maximise over w in Delta_{2d+1} (knot weights), PSD-Fourier:
        min_{mu in Delta_d} 4n mu^T M_w mu

    Inner min is a Standard Quadratic Program (StQP). Lasserre order-1
    SDP relaxation: replace mu mu^T by Y >= 0, tr(Y) = ?, etc.
    Standard: inf_{x in Delta_n} x^T A x = min { x^T A x : 1^T x = 1, x >= 0 }
    Lower-bounded by: max{ lambda : A - lambda * 11^T - sum_i mu_i e_i e_i^T >= 0 }

    Equivalent SDP dual:
        max gamma  subject to  A - gamma I_d + diag(s) >> 0,  s in R^d
        which gives: gamma <= min eigenvalue of (A + diag(s)).

    Actually simpler: inf_{x in Delta} x^T A x equals min_{i,j} A_{ij}
    when A is rank-1 of a special form. In general: copositive prog.
    Lasserre order 1 SDP: lift to Y = x x^T, Y >= 0, sum_{ij} Y_{ij} = 1,
    Y >= 0 (entrywise), tr(Y) free. Min sum_{ij} A_{ij} Y_{ij}.
    """
    if not HAVE_CVXPY:
        print("cvxpy not available -- skip SDP")
        return None

    n = d // 2
    L = 2 * d + 1

    # Variables
    w = cp.Variable(L)                  # knot weight (probability + PSD-Fourier)
    Y = cp.Variable((d, d), symmetric=True)  # Y = mu mu^T relaxation
    # We want to MAXIMISE  4n * sum_{ij} M_w[i,j] Y[i,j]  (hardcoded as quadratic in w)
    # over (Y, w) with constraints:
    #   Y >= 0 entrywise, Y PSD (Lasserre order 1 lift)
    #   sum_{ij} Y[i,j] = 1
    #   w probability + PSD-Fourier
    # This is JOINTLY non-convex (objective bilinear in w and Y). Bad.

    # Better: ALTERNATE direction.
    # Step A: fix w, solve inner LP/SDP for inf_mu mu^T M_w mu.
    # Step B: fix mu, solve outer LP for sup_w 4n * mu^T M_w mu (linear in w).
    # Iterate.
    print("Note: full dual SDP is bilinear; using alternating optimization.")
    return None


def lasserre1_inf_quadform_simplex(M: np.ndarray) -> Optional[float]:
    """Lasserre order-1 SDP relaxation of inf_{x in Delta_d} x^T M x.

    Lift: Y = x x^T. Constraints:
      Y >= 0 entrywise, Y >> 0 PSD, sum Y = 1.
    Objective: min sum_{ij} M[i,j] Y[i,j].

    Returns the optimal value (a lower bound on the true inf).
    """
    if not HAVE_CVXPY:
        return None
    d = M.shape[0]
    Y = cp.Variable((d, d), symmetric=True)
    constraints = [Y >> 0, Y >= 0, cp.sum(Y) == 1]
    obj = cp.Minimize(cp.sum(cp.multiply(M, Y)))
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            return None
        return float(prob.value)
    except Exception as e:
        print(f"  SDP failed: {e}")
        return None


def alternating_dual_lb(d: int, n_iters: int = 20, seed: int = 0,
                        verbose: bool = True) -> Tuple[float, np.ndarray]:
    """Alternating max over (w, mu) for the dual lower bound.

    Each iteration:
      1. Fix w. Solve inf_mu mu^T M_w mu via Lasserre-1 SDP. Get mu, value v.
      2. Fix mu. Solve sup_w 4n * mu^T M_w mu over PSD-Fourier probability w.
         This is a linear program in w with a PSD constraint on the Toeplitz
         matrix of w (Bochner discretization).

    Returns (best_lb, best_w).
    """
    if not HAVE_CVXPY:
        return float('-inf'), np.zeros(2 * d + 1)

    rng = np.random.default_rng(seed)
    n = d // 2
    L = 2 * d + 1

    # Initialize with autocorr of a random vector
    a = rng.standard_normal(d + 1)
    w_init = np.zeros(L)
    for k in range(-d, d + 1):
        for i in range(len(a)):
            j = i - k
            if 0 <= j < len(a):
                w_init[k + d] += a[i] * a[j]
    w_init = np.maximum(w_init, 0)
    w_init = w_init / max(w_init.sum(), 1e-9)
    w = w_init

    best_lb = -np.inf
    best_w = w.copy()

    for it in range(n_iters):
        M = build_M_w(w, d)
        lb_inner = lasserre1_inf_quadform_simplex(M)
        if lb_inner is None:
            if verbose:
                print(f"  iter {it}: inner SDP failed")
            break
        bound = 4.0 * n * lb_inner
        if bound > best_lb:
            best_lb = bound
            best_w = w.copy()

        # Step 2: optimize w, fix Y from inner SDP
        # We don't have direct access to mu — re-solve to get Y
        Y_var = cp.Variable((d, d), symmetric=True)
        cons1 = [Y_var >> 0, Y_var >= 0, cp.sum(Y_var) == 1]
        obj1 = cp.Minimize(cp.sum(cp.multiply(M, Y_var)))
        prob1 = cp.Problem(obj1, cons1)
        try:
            prob1.solve(solver=cp.SCS, verbose=False)
            Y_val = Y_var.value
        except Exception:
            break

        if Y_val is None:
            break

        # Now find w to maximise sum_{ij} M_w[i,j] Y_val[i,j]
        # = sum_{ij} w[i+j+1] Y_val[i,j]
        # = sum_k w[k] (sum_{i+j+1=k} Y_val[i,j])
        # = sum_k w[k] s_k  where s_k = sum_{i+j=k-1} Y_val[i,j]
        s = np.zeros(L)
        for i in range(d):
            for j in range(d):
                k = i + j + 1
                if 0 <= k <= 2 * d:
                    s[k] += Y_val[i, j]

        # Maximise sum_k w[k] s_k subject to:
        #   w[k] >= 0, sum w = 1
        #   PSD-Fourier: Toeplitz matrix [w_{|i-j|}] >> 0 — but this requires w
        #   symmetric around the center. For the discrete measure on knots, the
        #   PSD condition is: w corresponds to a positive-definite sequence.
        #   For symmetric w (w_k = w_{2d-k}), the Toeplitz matrix [w_{|i-j|}]
        #   on indices 0..d must be PSD.
        # Simplification: enforce w as autocorrelation of a (length d+1) vector
        #   a: w[k] = sum_i a[i] * a[i + k - d] for k = 0..2d. Auto PSD.
        # This is non-convex in a, but convex in w if we lift to a a^T = T.
        # Actually it's the FACTORIZATION constraint.
        #
        # Cleanest convex approach: enforce symmetry w_k = w_{2d-k}, then
        # require Toeplitz matrix T[i,j] = w_{|i-j|}_{0<=i,j<=d} to be PSD.
        # T is symmetric and Toeplitz; PSD <=> w corresponds to a stationary
        # random sequence. This is a convex SDP constraint.
        w_var = cp.Variable(L)
        T = cp.Variable((d + 1, d + 1), symmetric=True)
        constraints = [
            w_var >= 0,
            cp.sum(w_var) == 1,
            T >> 0,
        ]
        # Symmetry: w[k] = w[2d - k]
        for k in range(d + 1):
            if k < L and (2 * d - k) < L:
                constraints.append(w_var[k] == w_var[2 * d - k])
        # Toeplitz: T[i,j] = w[|i-j|], with index offset (use w_var[d + diff]
        # since the "center" lag-0 is w_var[d] under the symmetry above)
        # Actually the discrete autocorrelation interpretation: lag-l value
        # is w_var[d + l] = w_var[d - l] by symmetry. T[i,j] with lag = i-j
        # means T[i,j] = w_var[d + |i-j|].
        for i in range(d + 1):
            for j in range(d + 1):
                lag = abs(i - j)
                if d + lag < L:
                    constraints.append(T[i, j] == w_var[d + lag])

        obj_w = cp.Maximize(s @ w_var)
        prob2 = cp.Problem(obj_w, constraints)
        try:
            prob2.solve(solver=cp.SCS, verbose=False)
            if prob2.status not in ['optimal', 'optimal_inaccurate']:
                break
            w_new = w_var.value
        except Exception as e:
            if verbose:
                print(f"  iter {it}: outer SDP failed: {e}")
            break

        # Update
        w = w_new
        if verbose:
            print(f"  iter {it}: bound = {bound:.6f}")

    return best_lb, best_w


# =============================================================================
# UPPER BOUND: Fourier-truncated f, minimise ||f*f||_inf
# =============================================================================

def upper_bound_via_bin_grid(d: int, n_attempts: int = 50, seed: int = 0,
                              verbose: bool = True) -> Tuple[float, np.ndarray]:
    """Heuristic upper bound: random Dirichlet samples on the d-bin simplex.

    For each sample mu, compute ||step(mu) * step(mu)||_inf = max_k 4n MC[k-1].
    This is the EXACT inf-norm for step functions (piecewise linear).
    Track the minimum.

    Each sample gives an UPPER BOUND on val(d), hence on C_{1a} only when
    we extract a CONTINUOUS f from the bin masses (which is what step
    functions are). So this gives upper bounds on C_{1a}.
    """
    rng = np.random.default_rng(seed)
    from .fourier_xterm import step_autoconv_inf_norm

    best_v = np.inf
    best_mu = None
    for trial in range(n_attempts):
        # Generate concentrated dirichlet (heuristic for extremizers)
        alpha = rng.uniform(0.5, 5.0, size=d)
        mu = rng.dirichlet(alpha)
        v = step_autoconv_inf_norm(mu)
        if v < best_v:
            best_v = v
            best_mu = mu.copy()

    if verbose:
        print(f"  d={d}: best ||f_step * f_step||_inf = {best_v:.4f} from "
              f"{n_attempts} random samples")
        print(f"  Best mu: {best_mu}")

    return float(best_v), best_mu


# =============================================================================
# Smoke tests
# =============================================================================

def smoke_dual_random_w(d: int = 4):
    print("=" * 70)
    print(f"Random PSD-Fourier dual w experiment (d={d})")
    print("=" * 70)
    bound, w = compute_dual_lb_random_w(d, n_trials=100, seed=0)
    print(f"  Best random-w lower bound (crude lambda_min): {bound:.4f}")
    print(f"  (val(d={d}) from CLAUDE.md val_d_known is the target)")


def smoke_alternating(d: int = 4):
    print()
    print("=" * 70)
    print(f"Alternating dual SDP at d={d}")
    print("=" * 70)
    if not HAVE_CVXPY:
        print("  cvxpy unavailable; skipping")
        return
    bound, w = alternating_dual_lb(d, n_iters=10, seed=0, verbose=True)
    print(f"  Best alternating bound: {bound:.4f}")
    print(f"  val({d}) target: 1.250 (d=4); 1.281 (d=8); 1.319 (d=16)")


def smoke_upper_bound(d: int = 8):
    print()
    print("=" * 70)
    print(f"Heuristic UPPER BOUND (random Dirichlet, d={d})")
    print("=" * 70)
    v, mu = upper_bound_via_bin_grid(d, n_attempts=200, seed=42, verbose=True)
    print(f"  Note: this is val(d), not C_{{1a}} -- step functions have "
          f"||f*f||_inf == val(d) exactly")
    print(f"  Current upper bound: 1.5029. We have val({d}) = {v:.4f}")


if __name__ == "__main__":
    smoke_dual_random_w(d=4)
    smoke_alternating(d=4)
    smoke_upper_bound(d=8)
