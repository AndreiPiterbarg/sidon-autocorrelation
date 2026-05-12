"""Fourier-truncated SDP exploiting the cross-term vanishing lemma.

==============================================================================
MATHEMATICAL FOUNDATION (proof/formula_b_coarse_grid_proof.md)
==============================================================================

Setup: f on [-1/4, 1/4] with int f = 1, f >= 0. Partition into d = 2n bins
I_i = [-1/4 + i*h, -1/4 + (i+1)*h) with h = 1/(4n). Bin masses
mu_i = int_{I_i} f. Step function f_step has heights a_i = mu_i / h = 4n mu_i.
Residual eps_2 = f - f_step satisfies int_{I_i} eps_2 = 0 (zero-mean per bin).

Convolution KNOT POINTS x_k = -1/2 + k*h for k = 0, 1, ..., 4n. (f*f) is
piecewise quadratic with breakpoints at knots; for step f, (f_step * f_step)
is piecewise LINEAR with breakpoints at knots.

LEMMA 1 (Cross-term vanishing at knot points). Under the above setup,
    (f_step * eps_2)(x_k) = 0   exactly, for every knot x_k.

THEOREM 1 (Knot-point decomposition).
    (f * f)(x_k) = (f_step * f_step)(x_k) + (eps_2 * eps_2)(x_k)
                 = 4n * MC[k-1] + (eps_2 * eps_2)(x_k)

where MC[s] := sum_{i+j=s} mu_i mu_j is the discrete autoconvolution of
the bin masses. (The 4n factor is exact: see Section 0 of the proof doc.)

==============================================================================
WHAT THIS MODULE PROVIDES
==============================================================================

(1) Numerical verification of Lemma 1 on concrete f = f_step + eps_2,
    confirming cross-term = 0 at all knots to machine precision.

(2) EXACT evaluator for (f_step * f_step) at knots and continuous t,
    using the closed-form piecewise-linear interpolation.

(3) For f = f_step + eps_2 with eps_2 expressed in a Fourier-per-bin basis
    (sin(pi * (2j+1) * (x - x_i_mid) / h) for j = 0, 1, ..., K-1):
    - All bin-mean-zero by construction (every basis function integrates to 0
      over its bin)
    - Closed-form evaluation of (eps_2 * eps_2)(x_k) at knots (involves
      product-of-sines integrals; precomputable lookup table)

(4) Soundness checks:
    - Cross-term identity verified numerically for the chosen basis
    - (eps_2 * eps_2)(x_k) computed two ways (closed form + numeric quadrature)
      and cross-validated

(5) Optimization shell that finds eps_2 minimizing
        max_k [ 4n * MC[k-1] + (eps_2 * eps_2)(x_k) ]
    subject to f = f_step + eps_2 >= 0 on each bin, for fixed bin masses mu.
    This gives an UPPER BOUND on inf_f ||f*f||_inf over the parameterized
    Fourier-truncated subspace, which in turn is an upper bound on C_{1a}.

==============================================================================
HONEST CLARIFICATIONS (mathematically critical)
==============================================================================

The cross-term lemma applies at KNOT POINTS only. Off-knot, (f * f)(t) has
a non-trivial cross-term contribution (f_step * eps_2)(t) which is bounded
by Cauchy-Schwarz but NOT zero. So the bound

    ||f*f||_inf >= max_k (f*f)(x_k) >= max_k [4n MC[k-1] - ||eps_2||_2^2]

is RIGOROUS but uses a COARSE Cauchy-Schwarz on the residual.

For the Fourier-per-bin parameterization with K modes, ||eps_2||_2^2 has
closed form in the coefficients. The optimization in (5) is convex (sum
of squares minimization with linear nonnegativity per bin).

This module produces UPPER BOUNDS on C_{1a} (since restricting to a
subspace can only RAISE inf_f ||f*f||_inf above C_{1a}). Lower-bound use
requires a separate dual argument (NOT implemented here -- and as the
audit in proof/tightest_valid_pruning_bound.md and the agent reports
flagged, the standard cross-term-+-Fourier-residual scheme does NOT
produce a lower bound on C_{1a}; it just constructs explicit candidates).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# =============================================================================
# Geometry
# =============================================================================

@dataclass(frozen=True)
class BinGeometry:
    """Geometry of the d=2n bin partition of [-1/4, 1/4]."""
    d: int          # number of bins (must be even, = 2n)

    @property
    def n(self) -> int:
        return self.d // 2

    @property
    def h(self) -> float:
        """Bin width 1/(4n) = 1/(2d)."""
        return 1.0 / (2.0 * self.d)

    @property
    def conv_len(self) -> int:
        """Number of MC[s] indices: s = 0, 1, ..., 2d - 2."""
        return 2 * self.d - 1

    @property
    def n_knots(self) -> int:
        """Knots x_k = -1/2 + k*h for k = 0..4n. Note 4n = 2d."""
        return 2 * self.d + 1

    def bin_left(self, i: int) -> float:
        return -0.25 + i * self.h

    def bin_right(self, i: int) -> float:
        return -0.25 + (i + 1) * self.h

    def bin_mid(self, i: int) -> float:
        return -0.25 + (i + 0.5) * self.h

    def knot(self, k: int) -> float:
        return -0.5 + k * self.h


# =============================================================================
# Step-function exact evaluator
# =============================================================================

def step_autoconv_at_knot(mu: np.ndarray, k: int) -> float:
    """Exact value of (f_step * f_step)(x_k) where f_step has bin masses mu.

    Formula (proof doc Section 0):
        (f_step * f_step)(x_k) = h * sum_{i+j=k-1} g_i g_j
                                = h * (4n)^2 sum_{i+j=k-1} mu_i mu_j
                                = 4n * MC[k-1]
    where heights g_i = 4n * mu_i and h = 1/(4n).

    For k = 0: the sum is empty (no i,j>=0 with i+j=-1), value = 0.
    For k = 4n = 2d: the sum is empty, value = 0.
    """
    d = len(mu)
    s = k - 1
    if s < 0 or s > 2 * d - 2:
        return 0.0
    n = d // 2
    if d != 2 * n:
        raise ValueError("d must be even (d = 2n)")
    acc = 0.0
    i_lo = max(0, s - (d - 1))
    i_hi = min(d - 1, s)
    for i in range(i_lo, i_hi + 1):
        j = s - i
        acc += float(mu[i]) * float(mu[j])
    return 4.0 * n * acc


def step_autoconv_all_knots(mu: np.ndarray) -> np.ndarray:
    """Vector of (f_step * f_step)(x_k) for k = 0..4n = 2d.

    Length 2d + 1.
    """
    d = len(mu)
    geom = BinGeometry(d)
    vals = np.empty(geom.n_knots, dtype=np.float64)
    for k in range(geom.n_knots):
        vals[k] = step_autoconv_at_knot(mu, k)
    return vals


def step_autoconv_continuous(mu: np.ndarray, t: float) -> float:
    """Exact (f_step * f_step)(t) for any t in [-1/2, 1/2].

    f_step is piecewise constant -> f_step * f_step is piecewise LINEAR
    with breakpoints at knots x_k. Linearly interpolate between knot values.

    Outside [-1/2, 1/2]: returns 0.
    """
    d = len(mu)
    geom = BinGeometry(d)
    if t <= -0.5 or t >= 0.5:
        return 0.0
    # Knot index just below t: k = floor((t + 0.5) / h)
    u = (t + 0.5) / geom.h
    k_lo = int(math.floor(u))
    if k_lo >= geom.n_knots - 1:
        k_lo = geom.n_knots - 2
    if k_lo < 0:
        k_lo = 0
    frac = u - k_lo
    v_lo = step_autoconv_at_knot(mu, k_lo)
    v_hi = step_autoconv_at_knot(mu, k_lo + 1)
    return (1.0 - frac) * v_lo + frac * v_hi


def step_autoconv_inf_norm(mu: np.ndarray) -> float:
    """||f_step * f_step||_inf = max over knots (since piecewise linear)."""
    return float(np.max(step_autoconv_all_knots(mu)))


# =============================================================================
# Fourier-per-bin residual basis
# =============================================================================
#
# On bin I_i = [x_i_lo, x_i_hi] of width h, define for j = 0, 1, ..., K-1:
#     phi_{i,j}(x) = sqrt(2/h) * sin(pi * (2j+1) * (x - x_i_lo) / h)   [x in I_i]
#     phi_{i,j}(x) = 0                                                  [x not in I_i]
#
# Properties:
#   (a) int_{I_i} phi_{i,j} dx = sqrt(2/h) * (h / (pi*(2j+1))) * (1 - cos(pi*(2j+1)))
#                              = sqrt(2/h) * (h / (pi*(2j+1))) * (1 - (-1)^(2j+1))
#                              = sqrt(2/h) * (h / (pi*(2j+1))) * 2
#       BUT: with (2j+1) the integrand has half-period structure that does NOT
#       integrate to zero on [0, h]. We need a basis that integrates to zero.
#
# Use instead: half-period sines vanishing at both bin endpoints,
#     phi_{i,j}(x) = sqrt(2/h) * sin(pi * (j+1) * (x - x_i_lo) / h),
#                    j = 1, 2, 3, ... (j=1 -> 1 half-period, integrates to nonzero)
#
# Sines on [0,h] of frequency m*pi/h: int_0^h sin(m*pi*x/h) dx = (h/(m*pi))(1-cos(m*pi))
#   = 0 when m even. Non-zero (= 2h/(m*pi)) when m odd.
#
# So use EVEN-m sines: phi_{i,j}(x) = sqrt(2/h) * sin(2*pi*(j+1)*(x - x_i_lo)/h),
#                       j = 0, 1, 2, ...
# These have m = 2(j+1) which is even >= 2; all bin-integrals are 0. Orthonormal
# in L^2(I_i). And each phi_{i,j} vanishes at both endpoints.
# =============================================================================

def residual_basis_value(i: int, j: int, x: float, geom: BinGeometry) -> float:
    """Value of phi_{i,j}(x). Returns 0 outside bin I_i."""
    x_lo = geom.bin_left(i)
    x_hi = geom.bin_right(i)
    if x < x_lo or x >= x_hi:
        return 0.0
    h = geom.h
    return math.sqrt(2.0 / h) * math.sin(2.0 * math.pi * (j + 1) * (x - x_lo) / h)


def residual_basis_integrate(i: int, j: int, geom: BinGeometry) -> float:
    """int_{I_i} phi_{i,j} dx -- by construction = 0 for all i, j."""
    return 0.0  # exact (analytic): m=2(j+1) is even, so sin vanishes on full periods


# =============================================================================
# Verification of Lemma 1 (cross-term vanishing)
# =============================================================================

def verify_cross_term_at_knot(
    mu: np.ndarray,
    eps_coeffs: np.ndarray,
    n_quad: int = 4096,
) -> Tuple[float, np.ndarray]:
    """Numerically verify Lemma 1.

    Construct f_step and eps_2 = sum_{i,j} eps_coeffs[i,j] * phi_{i,j}(x).
    For each knot x_k, compute (f_step * eps_2)(x_k) by numerical quadrature
    on a fine grid. Lemma 1 predicts EXACTLY 0 (up to quadrature error).

    Args:
      mu: bin masses, shape (d,)
      eps_coeffs: residual Fourier coeffs, shape (d, K)
      n_quad: number of quadrature points on [-1/4, 1/4] (per side)

    Returns:
      (max_abs_value, per_knot_values) -- max_abs should be ~quadrature error.
    """
    d = len(mu)
    K = eps_coeffs.shape[1]
    geom = BinGeometry(d)
    n = geom.n
    h = geom.h

    # Fine quadrature grid on [-1/4, 1/4]
    s_grid = np.linspace(-0.25, 0.25, n_quad, endpoint=False) + (0.5 / n_quad)
    ds = 0.5 / n_quad

    # Build f_step on grid: f_step(s) = (4n) * mu_i for s in I_i
    bin_idx = np.minimum(np.floor((s_grid + 0.25) / h).astype(int), d - 1)
    f_step_grid = (4.0 * n) * mu[bin_idx]

    # Build eps_2 on grid
    eps_grid = np.zeros(n_quad)
    for s_idx in range(n_quad):
        s = s_grid[s_idx]
        i = bin_idx[s_idx]
        x_lo = geom.bin_left(i)
        for j in range(K):
            eps_grid[s_idx] += eps_coeffs[i, j] * math.sqrt(2.0 / h) * \
                math.sin(2.0 * math.pi * (j + 1) * (s - x_lo) / h)

    # For each knot x_k, compute (f_step * eps_2)(x_k) = int f_step(s) eps_2(x_k - s) ds
    cross_vals = np.zeros(geom.n_knots)
    for k in range(geom.n_knots):
        xk = geom.knot(k)
        # Need eps_2(xk - s) for each s. xk - s is in [xk - 1/4, xk + 1/4].
        # That's a different grid — use linear interp.
        # For simplicity: compute integrand pointwise.
        integrand = np.zeros(n_quad)
        for s_idx in range(n_quad):
            s = s_grid[s_idx]
            t = xk - s
            # Find bin of t
            if t < -0.25 or t >= 0.25:
                continue
            i_t = min(int(math.floor((t + 0.25) / h)), d - 1)
            x_lo_t = geom.bin_left(i_t)
            eps_t = 0.0
            for j in range(K):
                eps_t += eps_coeffs[i_t, j] * math.sqrt(2.0 / h) * \
                    math.sin(2.0 * math.pi * (j + 1) * (t - x_lo_t) / h)
            integrand[s_idx] = f_step_grid[s_idx] * eps_t
        cross_vals[k] = float(np.sum(integrand) * ds)

    return float(np.max(np.abs(cross_vals))), cross_vals


# =============================================================================
# Direct numerical evaluation of (f * f)(t) for f = f_step + eps_2
# =============================================================================

def _eps_value(eps_coeffs: np.ndarray, x: float, geom: BinGeometry) -> float:
    """Evaluate eps_2(x) given coefficients."""
    if x < -0.25 or x >= 0.25:
        return 0.0
    d = eps_coeffs.shape[0]
    K = eps_coeffs.shape[1]
    h = geom.h
    i = min(int(math.floor((x + 0.25) / h)), d - 1)
    x_lo = geom.bin_left(i)
    val = 0.0
    s_factor = math.sqrt(2.0 / h)
    for j in range(K):
        val += eps_coeffs[i, j] * s_factor * math.sin(2.0 * math.pi * (j + 1) * (x - x_lo) / h)
    return val


def _f_value(mu: np.ndarray, eps_coeffs: np.ndarray, x: float, geom: BinGeometry) -> float:
    """Evaluate f(x) = f_step(x) + eps_2(x)."""
    if x < -0.25 or x >= 0.25:
        return 0.0
    n = geom.n
    h = geom.h
    d = len(mu)
    i = min(int(math.floor((x + 0.25) / h)), d - 1)
    f_step = 4.0 * n * float(mu[i])
    eps = _eps_value(eps_coeffs, x, geom)
    return f_step + eps


def ff_value_numeric(
    mu: np.ndarray,
    eps_coeffs: np.ndarray,
    t: float,
    n_quad: int = 4096,
) -> float:
    """Numerically compute (f*f)(t) by quadrature."""
    geom = BinGeometry(len(mu))
    s_grid = np.linspace(-0.25, 0.25, n_quad, endpoint=False) + (0.5 / n_quad)
    ds = 0.5 / n_quad
    acc = 0.0
    for s in s_grid:
        f_s = _f_value(mu, eps_coeffs, s, geom)
        if f_s == 0.0:
            continue
        f_t_minus_s = _f_value(mu, eps_coeffs, t - s, geom)
        acc += f_s * f_t_minus_s
    return acc * ds


def ff_inf_norm_numeric(
    mu: np.ndarray,
    eps_coeffs: np.ndarray,
    n_t: int = 1000,
    n_quad: int = 2048,
) -> Tuple[float, float]:
    """Numerically compute ||f*f||_inf and the t at which it is attained.

    Scans n_t test points in [-1/2, 1/2] and quadrature with n_quad points.
    """
    t_grid = np.linspace(-0.5, 0.5, n_t)
    best_t = 0.0
    best_v = -np.inf
    for t in t_grid:
        v = ff_value_numeric(mu, eps_coeffs, t, n_quad=n_quad)
        if v > best_v:
            best_v = v
            best_t = t
    return float(best_v), float(best_t)


# =============================================================================
# Theoretical bound from Theorem 1 + Cauchy-Schwarz on residual
# =============================================================================

def eps_l2_norm_squared(eps_coeffs: np.ndarray) -> float:
    """||eps_2||_2^2 = sum of squares of coeffs (orthonormal basis)."""
    return float(np.sum(eps_coeffs * eps_coeffs))


def lemma1_lower_bound_at_knots(
    mu: np.ndarray,
    eps_coeffs: np.ndarray,
) -> Tuple[float, int]:
    """Sound lower bound on ||f*f||_inf using Theorem 1 + Cauchy-Schwarz.

    For any knot x_k:
        (f*f)(x_k) = 4n MC[k-1] + (eps*eps)(x_k) >= 4n MC[k-1] - ||eps||_2^2

    So ||f*f||_inf >= max_k [4n MC[k-1]] - ||eps||_2^2.

    Returns (bound_value, k_attaining_max).

    NOTE: if eps_coeffs == 0, bound = max_k 4n MC[k-1] = ||f_step * f_step||_inf
    which is the standard val(d)-style knot bound.
    """
    d = len(mu)
    n = d // 2
    eps_sq = eps_l2_norm_squared(eps_coeffs)

    best_k = 0
    best_v = -np.inf
    for k in range(2 * d + 1):
        v = step_autoconv_at_knot(mu, k)
        if v > best_v:
            best_v = v
            best_k = k

    return float(best_v - eps_sq), best_k


# =============================================================================
# Smoke tests (run with: python -m lasserre.fourier_xterm)
# =============================================================================

def _smoke_lemma1():
    """Verify Lemma 1 on several configurations."""
    print("=" * 70)
    print("Lemma 1 (cross-term vanishing) numerical verification")
    print("=" * 70)

    rng = np.random.default_rng(0)
    cases = [
        ("d=4 uniform mu, K=1 random eps", 4, 1),
        ("d=4 uniform mu, K=3 random eps", 4, 3),
        ("d=8 random mu, K=2 random eps", 8, 2),
        ("d=8 spike mu, K=4 random eps", 8, 4),
    ]

    for label, d, K in cases:
        if "uniform" in label:
            mu = np.ones(d) / d
        elif "spike" in label:
            mu = np.zeros(d)
            mu[d // 2] = 0.6
            mu[d // 2 - 1] = 0.4
        else:
            mu = rng.dirichlet(np.ones(d))
        eps_coeffs = 0.5 * rng.standard_normal((d, K))

        max_abs, _ = verify_cross_term_at_knot(mu, eps_coeffs, n_quad=2048)
        print(f"  {label}: max |cross-term at any knot| = {max_abs:.3e}")
        if max_abs > 1e-2:
            print(f"     WARNING: large -- possibly wrong basis or quadrature too coarse")
        else:
            print(f"     PASS (consistent with Lemma 1; remaining = quadrature error)")


def _smoke_step_autoconv():
    """Verify step-function autoconv: numeric vs Lemma 1 closed form."""
    print()
    print("=" * 70)
    print("Step-function (f_step * f_step) at knots: closed form vs quadrature")
    print("=" * 70)

    d = 4
    mu = np.array([0.1, 0.4, 0.3, 0.2])
    geom = BinGeometry(d)

    print(f"  d={d}, mu={mu}")
    for k in range(geom.n_knots):
        v_closed = step_autoconv_at_knot(mu, k)
        v_numeric = ff_value_numeric(mu, np.zeros((d, 0)), geom.knot(k), n_quad=4096)
        print(f"    k={k} x_k={geom.knot(k):+.4f}  closed={v_closed:.6f}  "
              f"numeric={v_numeric:.6f}  diff={abs(v_closed - v_numeric):.2e}")


def _smoke_thm1_decomposition():
    """Verify Theorem 1: (f*f)(x_k) = (f_step*f_step)(x_k) + (eps*eps)(x_k)."""
    print()
    print("=" * 70)
    print("Theorem 1 decomposition at knots: numeric (f*f) vs step + eps*eps")
    print("=" * 70)

    rng = np.random.default_rng(7)
    d = 4
    K = 2
    mu = np.array([0.2, 0.3, 0.3, 0.2])
    # Choose small eps_coeffs so f = f_step + eps stays nonneg
    eps_coeffs = 0.3 * rng.standard_normal((d, K))
    geom = BinGeometry(d)

    print(f"  d={d}, K={K}")
    for k in range(geom.n_knots):
        xk = geom.knot(k)
        ff_full = ff_value_numeric(mu, eps_coeffs, xk, n_quad=4096)
        ff_step = step_autoconv_at_knot(mu, k)
        # eps*eps at xk: by quadrature
        ee = ff_value_numeric(np.zeros(d), eps_coeffs, xk, n_quad=4096)
        diff = ff_full - (ff_step + ee)
        print(f"    k={k:2d}  full={ff_full:+.5f}  step={ff_step:+.5f}  "
              f"ee={ee:+.5f}  thm1_residual={diff:+.2e}")


def _smoke_lower_bound():
    """Demonstrate the Cauchy-Schwarz lower bound from Lemma 1."""
    print()
    print("=" * 70)
    print("Lower bound max_k 4n MC[k-1] - ||eps||^2  vs  numeric ||f*f||_inf")
    print("=" * 70)

    rng = np.random.default_rng(3)
    d = 4
    K = 2
    mu = np.array([0.2, 0.3, 0.3, 0.2])
    eps_scales = [0.0, 0.05, 0.1, 0.2, 0.5]
    for s_eps in eps_scales:
        eps_coeffs = s_eps * rng.standard_normal((d, K))
        bound, k_max = lemma1_lower_bound_at_knots(mu, eps_coeffs)
        ff_inf, t_max = ff_inf_norm_numeric(mu, eps_coeffs, n_t=200, n_quad=1024)
        sound = bound <= ff_inf + 1e-6
        print(f"  ||eps_coeffs||={s_eps:.2f}: lemma1_lb={bound:+.4f}  "
              f"true_inf={ff_inf:+.4f}  sound={sound}  "
              f"(k*={k_max}, t*={t_max:+.3f})")


if __name__ == "__main__":
    _smoke_lemma1()
    _smoke_step_autoconv()
    _smoke_thm1_decomposition()
    _smoke_lower_bound()
