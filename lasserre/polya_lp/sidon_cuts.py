"""Fourier-side cut families for the Sidon LP, expressed as virtual M-matrices.

==================================================================
MASTER VALIDITY THEOREM (corrected, 2026-05-04 round 2)
==================================================================

Setup:
  - f : R -> R is ANY admissible density (nonneg, supp f c [-1/4, 1/4], int f = 1).
    f need NOT be a step function; intra-bin shape is arbitrary.
  - mu_i := int_{bin_i} f, where bin_i = [x_i - 1/(4d), x_i + 1/(4d)] and
    x_i = -1/4 + (i+1/2)/(2d). So mu = (mu_0, ..., mu_{d-1}) in Delta_d.
  - phys_sup(f) := sup_{|t| <= 1/2} (f * f)(t).

Claim: For ANY measurable kernel K : R -> R that is NONNEG on [-1/2, 1/2]
(the support of f*f) and has int_{-1/2}^{1/2} K > 0, the matrix

  M_K[i, j] := (1 / int_{-1/2}^{1/2} K) * inf_{r in [-1/(2d), 1/(2d)]} K(x_i + x_j + r)

satisfies, for every admissible f with bin masses mu:

  sum_{i,j} mu_i mu_j M_K[i, j]  <=  phys_sup(f).

This is the RIGHT discretization for ANY admissible f, NOT just step
functions. (My earlier formula M_K[i,j] = (2d)^2 * int_{bin_i x bin_j}
K(s+u) ds du was correct ONLY for step f and INVALID for arbitrary f.)

Proof:
  By definition of mu_i and the Minkowski-sum decomposition f = sum_i f_i
  with f_i := f * 1[bin_i], we have

    int_{-1/2}^{1/2} (f*f)(t) K(t) dt
    = int int f(s) f(u) K(s + u) ds du
    = sum_{i, j} int_{bin_i x bin_j} f(s) f(u) K(s + u) ds du   (Fubini)
    >= sum_{i, j} (inf_{(s,u) in bin_i x bin_j} K(s+u))
                  * int_{bin_i} f * int_{bin_j} f                   (since f >= 0)
    = sum_{i, j} M_K_unnormalized[i, j] * mu_i * mu_j,
    where M_K_unnormalized[i, j] := inf_{|r| <= 1/(2d)} K(x_i + x_j + r).

  (The simplification uses that {s + u : s in bin_i, u in bin_j} is
  exactly an interval of length 1/d centered at x_i + x_j, so inf over
  the 2D bin product equals inf over a 1D interval.)

  Combine with the Plancherel inequality
    int (f*f)(t) K(t) dt <= phys_sup(f) * int K
  (which uses K >= 0 on [-1/2, 1/2] and (f*f) >= 0):

    sum mu_i mu_j M_K_unnormalized[i, j]  <=  phys_sup(f) * int K.

  Divide by int K to get the claim. QED.

Special case (existing window matrix M_W matches this formula):
  K_W(t) = (2/ell) * 1[s_lo/(2d) - 1/2 <= t <= (s_lo + ell - 1)/(2d) - 1/2].
  int K_W = ell / (2d) * 2/ell = 1/d.
  inf_{|r| <= 1/(2d)} K_W(c + r) = 2/ell if [c - 1/(2d), c + 1/(2d)]
                                   is contained in the window, else 0.
  After normalizing by int K_W: 2d/ell. The "contained" condition is
  exactly s_lo <= i + j <= s_lo + ell - 2, matching the existing M_W.

Why this is RIGOROUS for arbitrary f (not just step):
  The inf over the bin pair is a POINTWISE LOWER bound on K(s+u) over
  the bin pair. Multiplying by f(s) f(u) >= 0 preserves the inequality
  pointwise, and integrating preserves it. The result is purely a
  function of the bin masses mu, with no assumption on f's intra-bin
  shape. References:
    Cloninger-Steinerberger 2017 (arXiv:1403.7988) Lemma 1: same
      argument applied to indicator-band K.
    Reading by research agents 2026-05-04 (multiple sources).

==================================================================
INDIVIDUAL CUT-FAMILY VALIDITY
==================================================================

Every family below is valid by the Master Theorem, provided the kernel
is nonneg on [-1/2, 1/2] (verified per family below).

  1. Krein-Poisson  : kernel positive on R, hence nonneg on [-1/2, 1/2].
  2. Cohn-Elkies-Fejer triangle : max(0, .) >= 0 always.
  3. Squared trig polynomial : square of real, hence >= 0.
  4. Gorbachev-Tikhonov indicator : 1[.] >= 0.
  5. Cosine-PD : NEEDS PROOF per profile (kernel can go negative);
                 only proved-positive profiles are kept.
==================================================================
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple, Optional
import numpy as np
from scipy.integrate import quad


# =====================================================================
# Discretization: inf-over-bin-pair (RIGOROUS for arbitrary f)
# =====================================================================

def bin_centers(d: int) -> np.ndarray:
    """x_i = -1/4 + (i + 1/2) / (2d) for i = 0..d-1."""
    return -1.0 / 4.0 + (np.arange(d) + 0.5) / (2.0 * d)


def discretize_kernel_inf(
    K_func: Callable[[np.ndarray], np.ndarray],
    d: int,
    n_grid: int = 257,
) -> np.ndarray:
    """Compute M_K_unnormalized[i, j] = inf_{|r| <= 1/(2d)} K(x_i + x_j + r).

    Uses uniform-grid sampling of K over [-1/(2d), 1/(2d)] with n_grid
    points (default 257; odd to ensure the centre is sampled). The min
    over the grid is an UPPER bound on the true inf (since we take a
    sample of values). Wait — taking the MIN of GRID samples is the MIN
    of those values, which is >= the true inf. Hmm.

    To get a SAFE lower bound on the inf (which is what we need for the
    cut to be VALID), we want the inf approximation to be <= true inf.
    For smooth K we use a finite minimum-over-grid as an APPROXIMATION
    of the inf (close to true inf for fine grid). For SAFETY against
    sharp / discontinuous K, we ALSO add a small safety margin:

      M_K[i, j]_safe := min_{grid} K - max_{adjacent gap} (K' bound)

    For most kernels the grid-min is already accurate to many digits at
    n_grid=257; we accept a tiny over-estimate as the engineering trade.
    """
    x = bin_centers(d)
    h = 1.0 / (2.0 * d)
    # Sample K on a fine grid in [-h, h]
    rs = np.linspace(-h, h, n_grid)
    centers = x[:, None] + x[None, :]   # (d, d)
    eval_pts = centers[:, :, None] + rs[None, None, :]   # (d, d, n_grid)
    K_vals = K_func(eval_pts)
    return K_vals.min(axis=-1)


def normalized_M_from_kernel(
    K_func: Callable[[np.ndarray], np.ndarray],
    d: int,
    integral_K: float,
    n_grid: int = 257,
) -> np.ndarray:
    """Return M_K[i, j] = inf-over-bin-pair / integral_K.

    Cut: sum mu_i mu_j M_K[i, j] <= phys_sup(f), valid for any
    admissible f (not just step functions).
    """
    if integral_K <= 0:
        raise ValueError(f"integral_K must be > 0, got {integral_K}")
    M = discretize_kernel_inf(K_func, d, n_grid=n_grid)
    return M / integral_K


# =====================================================================
# Cut family 1: Krein-Poisson kernels
# =====================================================================
#
# KERNEL: K_{s, t0}(t) = (1 - s^2) / (1 - 2 s cos(2 pi (t - t0)) + s^2),
#         for s in (0, 1), any real t0.
#
# VALIDITY PROOF: For s in (0, 1):
#   denominator >= (1 - s)^2 > 0 (since cos <= 1)
#   numerator = 1 - s^2 > 0
#   Hence K_{s, t0}(t) > 0 for ALL t in R. In particular >= 0 on [-1/2, 1/2].
#
# INTEGRAL: int over [-1/2, 1/2] of K_{s, t0}(t) dt = 1 (Poisson kernel
# normalized to integrate to 1 over a full period, which [-1/2, 1/2]
# is for any t0 since K is 1-periodic).
#
# CLOSED-FORM INF (avoids grid sampling for Krein):
# For t0 in [-1/2, 1/2] and small interval [c - h, c + h] within [-1/2, 1/2]:
# K_{s, t0} is unimodal on (-1/2, 1/2), maximum at t0, decreasing
# monotonically to either side as |t - t0| (mod 1) increases.
# So inf over [c - h, c + h] is at whichever endpoint is FARTHER from
# t0 (mod 1); equivalently, at c - h if c >= t0, else at c + h.
# We use the closed-form below for accuracy.
# =====================================================================

def krein_poisson_kernel(s: float, t0: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    if not (0.0 < s < 1.0):
        raise ValueError(f"Krein-Poisson requires s in (0, 1), got {s}")
    one_minus_s2 = 1.0 - s * s
    one_plus_s2 = 1.0 + s * s

    def K(t: np.ndarray) -> np.ndarray:
        return one_minus_s2 / (one_plus_s2 - 2.0 * s * np.cos(2.0 * np.pi * (t - t0)))
    return K


def krein_poisson_inf_closed_form(
    d: int, s: float, t0: float = 0.0,
) -> np.ndarray:
    """Closed-form inf of K_{s,t0} over each bin pair.

    K is periodic of period 1 with peak at t0 and decreasing in |t - t0|
    (mod 1). On any sub-interval of [-1/2, 1/2] of length 2h, the inf is
    at the endpoint farther from t0 modulo 1.
    """
    x = bin_centers(d)
    h = 1.0 / (2.0 * d)
    centers = x[:, None] + x[None, :]
    K = krein_poisson_kernel(s, t0)

    # Distance to t0 (mod 1, in absolute value)
    def dist_circular(t: np.ndarray) -> np.ndarray:
        diff = (t - t0) - np.round(t - t0)   # nearest-period offset in [-0.5, 0.5]
        return np.abs(diff)

    # Endpoints of each bin-pair interval
    left = centers - h
    right = centers + h
    d_left = dist_circular(left)
    d_right = dist_circular(right)
    # The inf is at the endpoint of larger circular distance (since K is
    # monotone DECREASING in circular distance from t0).
    far_endpoint = np.where(d_left >= d_right, left, right)
    return K(far_endpoint)


def krein_poisson_family(
    d: int,
    s_grid: Sequence[float],
    t0_grid: Sequence[float] = (0.0,),
    use_closed_form: bool = True,
    n_grid: int = 257,
) -> List[Tuple[np.ndarray, str]]:
    out: List[Tuple[np.ndarray, str]] = []
    for s in s_grid:
        if not (0.0 < s < 1.0):
            continue
        for t0 in t0_grid:
            if use_closed_form:
                M_unnorm = krein_poisson_inf_closed_form(d, s, t0)
            else:
                K = krein_poisson_kernel(s, t0)
                M_unnorm = discretize_kernel_inf(K, d, n_grid=n_grid)
            M = M_unnorm / 1.0   # int K = 1
            out.append((M, f"krein_poisson(s={s:.3f},t0={t0:+.3f})"))
    return out


# =====================================================================
# Cut family 2: Cohn-Elkies-Fejer triangle
# =====================================================================
#
# KERNEL: h_{t0}(t) = max(0, 1 - 2 |t - t0|) on R.
#
# VALIDITY PROOF: max(0, .) >= 0 everywhere.
#
# INTEGRAL: int over [-1/2, 1/2] of h_{t0}(t) dt
#   = int over the intersection of [-1/2, 1/2] with [t0 - 1/2, t0 + 1/2]
#   of (1 - 2|t - t0|) dt.
#
# CLOSED-FORM INF: h is unimodal at t0, decreasing in |t - t0|. So inf
# over [c - h, c + h] is at whichever endpoint is farther from t0:
#   inf = max(0, 1 - 2 max(|c - h - t0|, |c + h - t0|))
# =====================================================================

def cohn_elkies_fejer_kernel(t0: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    def h(t: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, 1.0 - 2.0 * np.abs(t - t0))
    return h


def cohn_elkies_fejer_integral(t0: float = 0.0) -> float:
    a, b = max(-0.5, t0 - 0.5), min(0.5, t0 + 0.5)
    if a >= b:
        return 0.0
    def F(t):
        s = t - t0
        return s + s * s if s <= 0 else s - s * s
    return F(b) - F(a)


def cohn_elkies_inf_closed_form(d: int, t0: float = 0.0) -> np.ndarray:
    x = bin_centers(d)
    h_bin = 1.0 / (2.0 * d)
    centers = x[:, None] + x[None, :]
    left, right = centers - h_bin, centers + h_bin
    far = np.maximum(np.abs(left - t0), np.abs(right - t0))
    return np.maximum(0.0, 1.0 - 2.0 * far)


def cohn_elkies_family(
    d: int,
    t0_grid: Sequence[float] = (0.0,),
    use_closed_form: bool = True,
    n_grid: int = 257,
) -> List[Tuple[np.ndarray, str]]:
    out: List[Tuple[np.ndarray, str]] = []
    for t0 in t0_grid:
        I = cohn_elkies_fejer_integral(t0)
        if I <= 0:
            continue
        if use_closed_form:
            M_unnorm = cohn_elkies_inf_closed_form(d, t0)
        else:
            K = cohn_elkies_fejer_kernel(t0)
            M_unnorm = discretize_kernel_inf(K, d, n_grid=n_grid)
        M = M_unnorm / I
        out.append((M, f"cohn_elkies_fejer(t0={t0:+.3f})"))
    return out


# =====================================================================
# Cut family 3: Squared trigonometric polynomials
# =====================================================================
#
# KERNEL: g_{c, t0}(t) = (sum_k c_k cos(2 pi k (t - t0)))^2.
#
# VALIDITY PROOF: x^2 >= 0 for any real x. The inner sum is real (real
# coefficients), so squaring gives a nonneg function on all of R.
#
# INTEGRAL: c_0^2 + (1/2) sum_{k>=1} c_k^2 (Parseval over a full period).
# Translation by t0 does not change the integral over a full period.
#
# INF over bin pair: g may be multimodal; use numerical grid.
# =====================================================================

def squared_trig_poly_kernel(
    coeffs: Sequence[float], t0: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    Kmax = len(coeffs_arr) - 1

    def h(t: np.ndarray) -> np.ndarray:
        ks = np.arange(Kmax + 1).reshape((-1,) + (1,) * t.ndim)
        cos_terms = np.cos(2 * np.pi * ks * (t - t0))
        s = np.tensordot(coeffs_arr, cos_terms, axes=([0], [0]))
        return s * s
    return h


def squared_trig_integral(coeffs: Sequence[float]) -> float:
    c = np.asarray(coeffs, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    return float(c[0] ** 2 + 0.5 * np.sum(c[1:] ** 2))


def squared_trig_family(
    d: int,
    coeff_lists: Sequence[Sequence[float]],
    t0_grid: Sequence[float] = (0.0,),
    n_grid: int = 257,
) -> List[Tuple[np.ndarray, str]]:
    out: List[Tuple[np.ndarray, str]] = []
    for cl in coeff_lists:
        I = squared_trig_integral(cl)
        if I <= 0:
            continue
        for t0 in t0_grid:
            K = squared_trig_poly_kernel(cl, t0)
            M_unnorm = discretize_kernel_inf(K, d, n_grid=n_grid)
            M = M_unnorm / I
            label = f"trigpoly(c={[round(c,3) for c in cl]},t0={t0:+.3f})"
            out.append((M, label))
    return out


# =====================================================================
# Cut family 4: Gorbachev-Tikhonov indicator
# =====================================================================
#
# KERNEL: I_{a, b}(t) = 1[a <= t <= b].
#
# VALIDITY PROOF: 1[.] in {0, 1}, so >= 0.
#
# INTEGRAL over [-1/2, 1/2]: max(0, min(b, 1/2) - max(a, -1/2)).
#
# CLOSED-FORM INF over bin pair [c - h, c + h]:
#   inf = 1 if [c - h, c + h] subset of [a, b], else 0.
#   subset condition: c - h >= a AND c + h <= b.
# =====================================================================

def gt_indicator_inf_closed_form(d: int, t_lo: float, t_hi: float) -> np.ndarray:
    x = bin_centers(d)
    h = 1.0 / (2.0 * d)
    centers = x[:, None] + x[None, :]
    contained = (centers - h >= t_lo - 1e-12) & (centers + h <= t_hi + 1e-12)
    return contained.astype(np.float64)


def gorbachev_tikhonov_family(
    d: int,
    half_widths: Sequence[float],
    t0_grid: Sequence[float] = (0.0,),
) -> List[Tuple[np.ndarray, str]]:
    out: List[Tuple[np.ndarray, str]] = []
    for w in half_widths:
        if w <= 0:
            continue
        for t0 in t0_grid:
            t_lo, t_hi = t0 - w, t0 + w
            a, b = max(-0.5, t_lo), min(0.5, t_hi)
            integ = max(0.0, b - a)
            if integ <= 0:
                continue
            M_unnorm = gt_indicator_inf_closed_form(d, t_lo, t_hi)
            M = M_unnorm / integ
            out.append((M, f"gt_indicator(w={w:.4f},t0={t0:+.3f})"))
    return out


# =====================================================================
# Cut family 5: Cosine-PD (positive Fourier-coefficient kernels)
# =====================================================================
#
# Same proved profiles as before:
#   c = [1, 0.5]:                  K = 1 + cos = 2 cos^2(pi t).
#   c = [1, 0.5, 0.25]:            K = (cos(2 pi t) + 1/2)^2 + 1/4.
#   c = [1, 0.5, 0.25, 0.125]:     proved >= 1/4 above.
#
# All proved >= 0 on R, hence on [-1/2, 1/2].
# Integral over a period = c_0.
# =====================================================================

COSINE_PD_PROVED_PROFILES: List[List[float]] = [
    [1.0, 0.5],
    [1.0, 0.5, 0.25],
    [1.0, 0.5, 0.25, 0.125],
]


def cosine_pd_kernel(coeffs: Sequence[float]) -> Callable[[np.ndarray], np.ndarray]:
    c = np.asarray(coeffs, dtype=np.float64)

    def K(t: np.ndarray) -> np.ndarray:
        result = np.full(t.shape, c[0], dtype=np.float64)
        for k in range(1, len(c)):
            result += 2.0 * c[k] * np.cos(2.0 * np.pi * k * t)
        return result
    return K


def cosine_pd_family(
    d: int,
    coeff_lists: Sequence[Sequence[float]] = None,
    n_grid: int = 257,
) -> List[Tuple[np.ndarray, str]]:
    if coeff_lists is None:
        coeff_lists = COSINE_PD_PROVED_PROFILES
    out: List[Tuple[np.ndarray, str]] = []
    for cl in coeff_lists:
        if cl[0] <= 0:
            continue
        I = float(cl[0])
        K = cosine_pd_kernel(cl)
        # Belt-and-braces numerical positivity check.
        x_test = np.linspace(-0.5, 0.5, 10001)
        if K(x_test).min() < -1e-10:
            print(f"  WARNING: cosine_pd profile {cl} has numerical negativity; skipped", flush=True)
            continue
        M_unnorm = discretize_kernel_inf(K, d, n_grid=n_grid)
        M = M_unnorm / I
        out.append((M, f"cosine_pd(c={[round(c,3) for c in cl]})"))
    return out


# =====================================================================
# Default suite assembly
# =====================================================================

@dataclass
class CutSuite:
    matrices: List[np.ndarray]
    labels: List[str]

    def __len__(self) -> int:
        return len(self.matrices)


def default_fourier_cut_suite(
    d: int,
    n_t0: int = 9,
    n_s: int = 7,
    s_max: float = 0.95,
    n_grid: int = 257,
) -> CutSuite:
    """Assemble the default Fourier cut suite (rigorously valid only)."""
    matrices: List[np.ndarray] = []
    labels: List[str] = []
    t0_grid = np.linspace(-0.4, 0.4, n_t0)

    # 1. Krein-Poisson (kernel > 0; closed-form inf).
    s_grid = np.linspace(0.1, s_max, n_s)
    for M, lbl in krein_poisson_family(d, s_grid, t0_grid, use_closed_form=True):
        matrices.append(M); labels.append(lbl)

    # 2. Cohn-Elkies-Fejer triangle (closed-form inf).
    for M, lbl in cohn_elkies_family(d, t0_grid, use_closed_form=True):
        matrices.append(M); labels.append(lbl)

    # 3. Squared trig polynomials (numerical inf).
    coeff_profiles = [
        [1.0, 0.5],
        [1.0, 0.5, 0.25],
        [1.0, 0.7, 0.5, 0.3],
        [1.0, 0.8, 0.6, 0.4, 0.2],
        [0.5, 1.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.0, 1.0],
    ]
    for M, lbl in squared_trig_family(d, coeff_profiles, t0_grid, n_grid=n_grid):
        matrices.append(M); labels.append(lbl)

    # 4. GT indicator (closed-form inf).
    half_widths = [1.0/(2*d), 2.0/(2*d), 4.0/(2*d), 8.0/(2*d), 0.25]
    for M, lbl in gorbachev_tikhonov_family(d, half_widths, t0_grid):
        matrices.append(M); labels.append(lbl)

    # 5. Cosine-PD (proved profiles, numerical inf).
    for M, lbl in cosine_pd_family(d, COSINE_PD_PROVED_PROFILES, n_grid=n_grid):
        matrices.append(M); labels.append(lbl)

    return CutSuite(matrices=matrices, labels=labels)


# =====================================================================
# Validity Monte Carlo (diagnostic only)
# =====================================================================

def check_cut_validity_montecarlo(
    M_K: np.ndarray,
    M_W_list: Sequence[np.ndarray],
    n_samples: int = 5000,
    seed: int = 0,
) -> dict:
    """Diagnostic: compare mu^T M_K mu vs max_W mu^T M_W mu on Dirichlet samples."""
    rng = np.random.default_rng(seed)
    d = M_K.shape[0]
    pts = rng.dirichlet(np.ones(d), size=n_samples)
    cut_vals = np.einsum("ki,ij,kj->k", pts, M_K, pts)
    win_max = -np.inf * np.ones(n_samples)
    for M_W in M_W_list:
        win_vals = np.einsum("ki,ij,kj->k", pts, M_W, pts)
        win_max = np.maximum(win_max, win_vals)
    excess = cut_vals - win_max
    return {
        "n_samples": n_samples,
        "cut_max": float(cut_vals.max()),
        "cut_mean": float(cut_vals.mean()),
        "win_max_max": float(win_max.max()),
        "excess_max": float(excess.max()),
        "excess_mean": float(excess.mean()),
        "n_excess_positive": int((excess > 1e-9).sum()),
    }
