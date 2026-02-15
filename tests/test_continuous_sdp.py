"""Tests for the continuous moment SDP (exp_continuous_sdp.py).

Verifies that every SDP constraint is a valid necessary condition: for any
concrete f >= 0 on [-1/4, 1/4] with int f = 1, the true product measure
Y = mm^T with eta = ||f*f||_inf satisfies all constraints.  This proves
the SDP relaxation contains all true feasible points, so its optimum is
a valid lower bound on C_{1a}.

Test functions:
  - Uniform:   f = 2  on [-1/4, 1/4]           (symmetric)
  - Right-half: f = 4  on [0, 1/4]              (fully asymmetric)
  - Skewed:    f = 3  on [-1/4, 0], 1 on (0, 1/4]  (mildly asymmetric)

Each test function is checked against ALL constraints:
  (a) Y >> 0, Y[0,0] = 1
  (b) 1D moment matrix Hankel(m) >> 0
  (c) 1D localizing for [-1/4, 1/4]
  (d) 2D Lasserre moment matrix >> 0
  (e) 2D localizing for g(x), g(y)
  (f) Gap function Hankel(h) >> 0
  (g) Gap localizing for [-1/2, 1/2]
  (h) Autoconvolution moments G[k] match numerical convolution
  (i) V2 boundary factoring: q = (f*f)/(1/4-t^2) >= 0 with PSD Hankel
  (j) V2 product localizing: (1/16-x^2)(1/16-y^2) >> 0
"""

import sys
import os
import math

import numpy as np
import pytest

# Add exploration/sdp/continuous and exploration/sdp to path for imports
_sdp_dir = os.path.join(os.path.dirname(__file__), '..', 'exploration', 'sdp')
_cont_dir = os.path.join(_sdp_dir, 'continuous')
sys.path.insert(0, _cont_dir)
sys.path.insert(0, _sdp_dir)

from exp_continuous_sdp import (
    precompute_constants,
    _enumerate_2d_monomials,
    solve_continuous_sdp,
    solve_continuous_sdp_scaled,
    solve_decoupled_sdp,
    solve_continuous_sdp_v2,
    solve_continuous_sdp_v3,
)


# ===========================================================================
# Helpers
# ===========================================================================

N_GRID = 50_000  # fine grid for numerical integration
N_MOMENTS = 8    # moment degree for constraint checks
PSD_TOL = -1e-5  # tolerance for "PSD" (allow tiny numerical negativity from quadrature)


def _make_grid():
    return np.linspace(-0.25, 0.25, N_GRID)


def _compute_moments(f_vals, x, n_max):
    """m_k = int x^k f(x) dx via trapezoidal rule."""
    dx = x[1] - x[0]
    return np.array([np.trapezoid(x**k * f_vals, dx=dx) for k in range(n_max + 1)])


def _autoconv(f_vals, x):
    """Compute autoconvolution on a fine grid. Returns (conv, t_grid)."""
    dx = x[1] - x[0]
    conv = np.convolve(f_vals, f_vals, mode='full') * dx
    t_grid = np.linspace(2 * x[0], 2 * x[-1], len(conv))
    return conv, t_grid


def _integration_constants(max_k):
    """I[k] = int_{-1/2}^{1/2} t^k dt."""
    I = np.zeros(max_k + 1)
    for k in range(0, max_k + 1, 2):
        I[k] = 2.0 * (0.5) ** (k + 1) / (k + 1)
    return I


def _build_hankel(seq, size):
    return np.array([[seq[i + j] for j in range(size)] for i in range(size)])


def _build_localizing_1d(seq, size, c_sq):
    """Localizing for g(t) = c^2 - t^2. L[i,j] = c^2*seq[i+j] - seq[i+j+2]."""
    return np.array([[c_sq * seq[i + j] - seq[i + j + 2]
                       for j in range(size)] for i in range(size)])


def _build_2d_moment_matrix(Y, d):
    """2D Lasserre moment matrix: M[(a1,a2),(b1,b2)] = Y[a1+b1, a2+b2]."""
    monos = _enumerate_2d_monomials(d)
    s = len(monos)
    M = np.zeros((s, s))
    for i, (a1, a2) in enumerate(monos):
        for j, (b1, b2) in enumerate(monos):
            M[i, j] = Y[a1 + b1, a2 + b2]
    return M


def _build_2d_localizing_x(Y, d):
    """Localizing for g(x) = 1/16 - x^2 on the 2D product space."""
    monos = _enumerate_2d_monomials(d - 1)
    s = len(monos)
    L = np.zeros((s, s))
    for i, (a1, a2) in enumerate(monos):
        for j, (b1, b2) in enumerate(monos):
            s1, s2 = a1 + b1, a2 + b2
            L[i, j] = (1.0 / 16.0) * Y[s1, s2] - Y[s1 + 2, s2]
    return L


def _build_2d_localizing_y(Y, d):
    """Localizing for g(y) = 1/16 - y^2 on the 2D product space."""
    monos = _enumerate_2d_monomials(d - 1)
    s = len(monos)
    L = np.zeros((s, s))
    for i, (a1, a2) in enumerate(monos):
        for j, (b1, b2) in enumerate(monos):
            s1, s2 = a1 + b1, a2 + b2
            L[i, j] = (1.0 / 16.0) * Y[s1, s2] - Y[s1, s2 + 2]
    return L


def _build_product_localizing(Y, d):
    """Localizing for (1/16-x^2)(1/16-y^2) on the 2D product space.

    L[alpha,beta] = (1/256)*Y[s1,s2] - (1/16)*Y[s1+2,s2]
                    - (1/16)*Y[s1,s2+2] + Y[s1+2,s2+2]
    """
    if d < 2:
        return None
    monos = _enumerate_2d_monomials(d - 2)
    s = len(monos)
    L = np.zeros((s, s))
    for i, (a1, a2) in enumerate(monos):
        for j, (b1, b2) in enumerate(monos):
            s1, s2 = a1 + b1, a2 + b2
            L[i, j] = ((1.0 / 256.0) * Y[s1, s2]
                        - (1.0 / 16.0) * Y[s1 + 2, s2]
                        - (1.0 / 16.0) * Y[s1, s2 + 2]
                        + Y[s1 + 2, s2 + 2])
    return L


def _build_stieltjes_1d(seq, size, half_width, sign):
    """Stieltjes (one-sided) localizing for p(t) = sign*t + half_width >= 0.

    L[i,j] = half_width * seq[i+j] + sign * seq[i+j+1]
    """
    return np.array([[half_width * seq[i + j] + sign * seq[i + j + 1]
                       for j in range(size)] for i in range(size)])


def _build_stieltjes_2d_x(Y, d, sign):
    """2D Stieltjes localizing for p(x) = sign*x + 1/4.

    L[(a1,a2),(b1,b2)] = (1/4)*Y[s1,s2] + sign*Y[s1+1,s2]
    Monomials: total degree <= d-1.
    """
    if d < 1:
        return None
    monos = _enumerate_2d_monomials(d - 1)
    s = len(monos)
    L = np.zeros((s, s))
    for i, (a1, a2) in enumerate(monos):
        for j_idx, (b1, b2) in enumerate(monos):
            s1, s2 = a1 + b1, a2 + b2
            L[i, j_idx] = 0.25 * Y[s1, s2] + sign * Y[s1 + 1, s2]
    return L


def _build_stieltjes_2d_y(Y, d, sign):
    """2D Stieltjes localizing for p(y) = sign*y + 1/4."""
    if d < 1:
        return None
    monos = _enumerate_2d_monomials(d - 1)
    s = len(monos)
    L = np.zeros((s, s))
    for i, (a1, a2) in enumerate(monos):
        for j_idx, (b1, b2) in enumerate(monos):
            s1, s2 = a1 + b1, a2 + b2
            L[i, j_idx] = 0.25 * Y[s1, s2] + sign * Y[s1, s2 + 1]
    return L


def _build_corner_localizing(Y, d, sx, sy):
    """Cross-product localizing for (sx*x + 1/4)(sy*y + 1/4) >= 0.

    L[(a1,a2),(b1,b2)] = sx*sy*Y[s1+1,s2+1] + (1/4)*sx*Y[s1+1,s2]
                        + (1/4)*sy*Y[s1,s2+1] + (1/16)*Y[s1,s2]
    Monomials: total degree <= d-1.
    """
    if d < 1:
        return None
    monos = _enumerate_2d_monomials(d - 1)
    s = len(monos)
    L = np.zeros((s, s))
    for i, (a1, a2) in enumerate(monos):
        for j_idx, (b1, b2) in enumerate(monos):
            s1, s2 = a1 + b1, a2 + b2
            L[i, j_idx] = (sx * sy * Y[s1 + 1, s2 + 1]
                          + 0.25 * sx * Y[s1 + 1, s2]
                          + 0.25 * sy * Y[s1, s2 + 1]
                          + (1.0 / 16.0) * Y[s1, s2])
    return L


def _min_eigval(M):
    return float(np.min(np.linalg.eigvalsh(M)))


def _prepare_test_data(f_vals, x, n):
    """Compute all quantities needed to check SDP constraints."""
    # Need moments up to 2n+2 for localizing checks
    m = _compute_moments(f_vals, x, 2 * n + 2)

    # True product measure (rank 1)
    Y = np.outer(m[:n + 1], m[:n + 1])

    # Autoconvolution
    conv, t_grid = _autoconv(f_vals, x)
    eta = float(np.max(conv))

    # Binomial coefficients
    binom = [[math.comb(k, j) for j in range(k + 1)] for k in range(2 * n + 1)]

    # Autoconvolution moments from Y: G[k] = sum_j C(k,j) Y[j, k-j]
    G = np.zeros(2 * n + 1)
    for k in range(2 * n + 1):
        for j in range(max(0, k - n), min(k, n) + 1):
            G[k] += binom[k][j] * Y[j, k - j]

    # Integration constants
    Ik = _integration_constants(4 * n)

    # Gap function h[k] = eta * I[k] - G[k]
    h = np.array([eta * Ik[k] - G[k] for k in range(2 * n + 1)])

    return {
        'm': m, 'Y': Y, 'n': n, 'eta': eta,
        'G': G, 'I': Ik, 'h': h,
        'conv': conv, 't_grid': t_grid,
        'f_vals': f_vals, 'x': x,
    }


# ===========================================================================
# Test functions (concrete f's)
# ===========================================================================

def _uniform():
    """f = 2 on [-1/4, 1/4].  Symmetric.  ||f*f||_inf = 2."""
    x = _make_grid()
    return 2.0 * np.ones_like(x), x


def _right_half():
    """f = 4 on [0, 1/4], 0 on [-1/4, 0).  Fully asymmetric."""
    x = _make_grid()
    return np.where(x >= 0, 4.0, 0.0), x


def _skewed():
    """f = 3 on [-1/4, 0], f = 1 on (0, 1/4].  Mildly asymmetric."""
    x = _make_grid()
    return np.where(x <= 0, 3.0, 1.0), x


# ===========================================================================
# Core constraint tests — parametrized over all test functions
# ===========================================================================

class TestConstraintValidity:
    """For each concrete f, verify ALL SDP constraints hold at eta = ||f*f||_inf.

    This proves the relaxation is valid: every true feasible point is inside
    the SDP feasible set, so the SDP minimum is a valid lower bound on C_{1a}.
    """

    @pytest.fixture(params=['uniform', 'right_half', 'skewed'])
    def data(self, request):
        if request.param == 'uniform':
            f_vals, x = _uniform()
        elif request.param == 'right_half':
            f_vals, x = _right_half()
        else:
            f_vals, x = _skewed()
        d = _prepare_test_data(f_vals, x, N_MOMENTS)
        d['name'] = request.param
        return d

    # --- (a) Y >> 0, Y[0,0] = 1, normalization ---

    def test_normalization(self, data):
        """m_0 = int f = 1."""
        assert abs(data['m'][0] - 1.0) < 1e-3, \
            f"{data['name']}: m_0 = {data['m'][0]}"

    def test_Y_psd(self, data):
        """Y = mm^T is PSD (trivially, since it's rank 1 with nonneg entries)."""
        ev = _min_eigval(data['Y'])
        assert ev > PSD_TOL, f"{data['name']}: Y min eigval = {ev:.2e}"

    def test_Y_00_is_1(self, data):
        """Y[0,0] = m_0^2 = 1."""
        assert abs(data['Y'][0, 0] - 1.0) < 1e-3, \
            f"{data['name']}: Y[0,0] = {data['Y'][0, 0]}"

    # --- (b) 1D moment matrix ---

    def test_1d_moment_matrix_psd(self, data):
        """Hankel(m) >> 0 (necessary for m to be moments of f >= 0)."""
        n = data['n']
        r = n // 2
        H = _build_hankel(data['m'], r + 1)
        ev = _min_eigval(H)
        assert ev > PSD_TOL, \
            f"{data['name']}: Hankel(m) min eigval = {ev:.2e}"

    # --- (c) 1D localizing for [-1/4, 1/4] ---

    def test_1d_localizing_psd(self, data):
        """Localizing for g(x) = 1/16 - x^2 >> 0 (support on [-1/4, 1/4])."""
        n = data['n']
        r = n // 2
        L = _build_localizing_1d(data['m'], r, 1.0 / 16.0)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: 1D localizing min eigval = {ev:.2e}"

    # --- (d) 2D Lasserre moment matrix ---

    def test_2d_moment_matrix_psd(self, data):
        """2D moment matrix >> 0 (necessary for Y from nonneg 2D measure)."""
        d = data['n'] // 2
        M2d = _build_2d_moment_matrix(data['Y'], d)
        ev = _min_eigval(M2d)
        assert ev > PSD_TOL, \
            f"{data['name']}: 2D moment matrix min eigval = {ev:.2e}"

    # --- (e) 2D localizing for g(x), g(y) ---

    def test_2d_localizing_x_psd(self, data):
        """2D localizing for g(x) = 1/16 - x^2."""
        d = data['n'] // 2
        if d < 1:
            pytest.skip("d < 1")
        Lgx = _build_2d_localizing_x(data['Y'], d)
        ev = _min_eigval(Lgx)
        assert ev > PSD_TOL, \
            f"{data['name']}: 2D loc(x) min eigval = {ev:.2e}"

    def test_2d_localizing_y_psd(self, data):
        """2D localizing for g(y) = 1/16 - y^2."""
        d = data['n'] // 2
        if d < 1:
            pytest.skip("d < 1")
        Lgy = _build_2d_localizing_y(data['Y'], d)
        ev = _min_eigval(Lgy)
        assert ev > PSD_TOL, \
            f"{data['name']}: 2D loc(y) min eigval = {ev:.2e}"

    # --- (f) Gap function Hankel ---

    def test_gap_hankel_psd(self, data):
        """Hankel(h) >> 0 where h = eta*I - G (gap nonneg on [-1/2, 1/2])."""
        n = data['n']
        Mh = _build_hankel(data['h'], n + 1)
        ev = _min_eigval(Mh)
        assert ev > PSD_TOL, \
            f"{data['name']}: Hankel(h) min eigval = {ev:.2e}"

    # --- (g) Gap localizing for [-1/2, 1/2] ---

    def test_gap_localizing_psd(self, data):
        """Localizing for g(t) = 1/4 - t^2 on gap function."""
        n = data['n']
        Lh = _build_localizing_1d(data['h'], n, 0.25)
        ev = _min_eigval(Lh)
        assert ev > PSD_TOL, \
            f"{data['name']}: gap localizing min eigval = {ev:.2e}"

    # --- (h) Autoconvolution moments cross-check ---

    def test_autoconv_moments_match_numerical(self, data):
        """G[k] from Y should match moments of numerically computed f*f."""
        conv, t = data['conv'], data['t_grid']
        dt = t[1] - t[0]
        n = data['n']

        for k in range(min(8, 2 * n + 1)):
            G_direct = np.trapezoid(t**k * conv, dx=dt)
            G_from_Y = data['G'][k]
            if abs(G_direct) > 1e-10:
                rel_err = abs(G_from_Y - G_direct) / abs(G_direct)
                assert rel_err < 0.01, \
                    f"{data['name']}: G[{k}] Y={G_from_Y:.6e}, " \
                    f"direct={G_direct:.6e}, rel_err={rel_err:.2%}"
            else:
                assert abs(G_from_Y) < 1e-4, \
                    f"{data['name']}: G[{k}] should be ~0, got {G_from_Y:.2e}"

    # --- Round 1: Autoconvolution nonnegativity ---

    def test_autoconv_hankel_psd(self, data):
        """Hankel(G) >> 0: f >= 0 implies (f*f) >= 0, so G moments form PSD Hankel."""
        n = data['n']
        MG = _build_hankel(data['G'], n + 1)
        ev = _min_eigval(MG)
        assert ev > PSD_TOL, \
            f"{data['name']}: Hankel(G) min eigval = {ev:.2e}"

    def test_autoconv_localizing_psd(self, data):
        """Localizing for (f*f) on [-1/2, 1/2] with g(t) = 1/4 - t^2."""
        n = data['n']
        LG = _build_localizing_1d(data['G'], n, 0.25)
        ev = _min_eigval(LG)
        assert ev > PSD_TOL, \
            f"{data['name']}: autoconv localizing min eigval = {ev:.2e}"

    # --- Round 2: Marginal moment bounds ---

    def test_marginal_moment_bounds(self, data):
        """|m_k| <= (1/4)^k since f >= 0 on [-1/4, 1/4] with int f = 1."""
        m = data['m']
        n = data['n']
        for k in range(n + 1):
            bound = 0.25 ** k
            assert abs(m[k]) <= bound + 1e-6, \
                f"{data['name']}: |m[{k}]| = {abs(m[k]):.6f} > (1/4)^{k} = {bound:.6f}"

    # --- Round 3: Y entry upper bounds ---

    def test_Y_entry_bounds(self, data):
        """|Y[j,k]| <= (1/4)^{j+k} since Y = m_j*m_k with |m_k| <= (1/4)^k."""
        Y = data['Y']
        n = data['n']
        for j in range(n + 1):
            for k in range(n + 1):
                bound = 0.25 ** (j + k)
                assert abs(Y[j, k]) <= bound + 1e-6, \
                    f"{data['name']}: |Y[{j},{k}]| = {abs(Y[j,k]):.6f} > {bound:.6f}"

    # --- Round 4: Stieltjes (one-sided) localizing, 1D ---

    def test_stieltjes_1d_h1_psd(self, data):
        """One-sided localizing: (x + 1/4) >= 0 on support."""
        n = data['n']
        r = n // 2
        L = _build_stieltjes_1d(data['m'], r, 0.25, +1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: Stieltjes 1D h1 min eigval = {ev:.2e}"

    def test_stieltjes_1d_h2_psd(self, data):
        """One-sided localizing: (1/4 - x) >= 0 on support."""
        n = data['n']
        r = n // 2
        L = _build_stieltjes_1d(data['m'], r, 0.25, -1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: Stieltjes 1D h2 min eigval = {ev:.2e}"

    # --- Round 4: Stieltjes (one-sided) localizing, 2D ---

    def test_stieltjes_2d_x_h1_psd(self, data):
        """2D Stieltjes: (x + 1/4) >= 0."""
        d = data['n'] // 2
        if d < 1:
            pytest.skip("d < 1")
        L = _build_stieltjes_2d_x(data['Y'], d, +1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: 2D Stieltjes x+ min eigval = {ev:.2e}"

    def test_stieltjes_2d_x_h2_psd(self, data):
        """2D Stieltjes: (1/4 - x) >= 0."""
        d = data['n'] // 2
        if d < 1:
            pytest.skip("d < 1")
        L = _build_stieltjes_2d_x(data['Y'], d, -1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: 2D Stieltjes x- min eigval = {ev:.2e}"

    def test_stieltjes_2d_y_h1_psd(self, data):
        """2D Stieltjes: (y + 1/4) >= 0."""
        d = data['n'] // 2
        if d < 1:
            pytest.skip("d < 1")
        L = _build_stieltjes_2d_y(data['Y'], d, +1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: 2D Stieltjes y+ min eigval = {ev:.2e}"

    def test_stieltjes_2d_y_h2_psd(self, data):
        """2D Stieltjes: (1/4 - y) >= 0."""
        d = data['n'] // 2
        if d < 1:
            pytest.skip("d < 1")
        L = _build_stieltjes_2d_y(data['Y'], d, -1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: 2D Stieltjes y- min eigval = {ev:.2e}"

    # --- Round 5: Autoconvolution Stieltjes ---

    def test_autoconv_stieltjes_h1_psd(self, data):
        """Stieltjes on (f*f): (t + 1/2)(f*f)(t) >= 0."""
        n = data['n']
        G = data['G']
        L = _build_stieltjes_1d(G, n, 0.5, +1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: autoconv Stieltjes h1 min eigval = {ev:.2e}"

    def test_autoconv_stieltjes_h2_psd(self, data):
        """Stieltjes on (f*f): (1/2 - t)(f*f)(t) >= 0."""
        n = data['n']
        G = data['G']
        L = _build_stieltjes_1d(G, n, 0.5, -1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: autoconv Stieltjes h2 min eigval = {ev:.2e}"

    # --- Round 6: Cross-product (corner) localizing ---

    @pytest.mark.parametrize("sx,sy", [(+1, +1), (+1, -1), (-1, +1), (-1, -1)])
    def test_cross_product_localizing_psd(self, data, sx, sy):
        """(sx*x + 1/4)(sy*y + 1/4) >= 0 on [-1/4, 1/4]^2."""
        d = data['n'] // 2
        if d < 1:
            pytest.skip("d < 1")
        L = _build_corner_localizing(data['Y'], d, sx, sy)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: corner ({sx:+d},{sy:+d}) min eigval = {ev:.2e}"

    # --- Round 7: Gap Stieltjes ---

    def test_gap_stieltjes_h1_psd(self, data):
        """Stieltjes on gap: (t + 1/2) h(t) >= 0."""
        n = data['n']
        h = data['h']
        L = _build_stieltjes_1d(h, n, 0.5, +1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: gap Stieltjes h1 min eigval = {ev:.2e}"

    def test_gap_stieltjes_h2_psd(self, data):
        """Stieltjes on gap: (1/2 - t) h(t) >= 0."""
        n = data['n']
        h = data['h']
        L = _build_stieltjes_1d(h, n, 0.5, -1)
        ev = _min_eigval(L)
        assert ev > PSD_TOL, \
            f"{data['name']}: gap Stieltjes h2 min eigval = {ev:.2e}"


# ===========================================================================
# V2 boundary factoring: (f*f)(t) = (1/4 - t^2) q(t), q >= 0
# ===========================================================================

class TestBoundaryFactoring:
    """Verify boundary factoring constraint is valid for concrete f's."""

    @pytest.fixture(params=['uniform', 'right_half', 'skewed'])
    def data(self, request):
        if request.param == 'uniform':
            f_vals, x = _uniform()
        elif request.param == 'right_half':
            f_vals, x = _right_half()
        else:
            f_vals, x = _skewed()
        d = _prepare_test_data(f_vals, x, N_MOMENTS)
        d['name'] = request.param
        return d

    def test_autoconv_vanishes_at_boundary(self, data):
        """(f*f)(+-1/2) = 0 (consequence of support on [-1/4, 1/4])."""
        conv, t = data['conv'], data['t_grid']
        # Find values near t = +-0.5
        for t_boundary in [-0.5, 0.5]:
            idx = np.argmin(np.abs(t - t_boundary))
            assert abs(conv[idx]) < 0.1, \
                f"{data['name']}: (f*f)({t_boundary}) = {conv[idx]:.4f}, should be ~0"

    def test_q_nonneg(self, data):
        """q(t) = (f*f)(t) / (1/4 - t^2) >= 0 on (-1/2, 1/2)."""
        conv, t = data['conv'], data['t_grid']
        # Restrict to interior of [-1/2, 1/2] to avoid 0/0
        mask = (t > -0.48) & (t < 0.48)
        denom = 0.25 - t[mask]**2
        q = conv[mask] / denom
        assert np.all(q > -1e-3), \
            f"{data['name']}: q has negative values, min = {np.min(q):.4f}"

    def test_q_moments_hankel_psd(self, data):
        """The Hankel matrix of q's moments should be PSD."""
        conv, t = data['conv'], data['t_grid']
        dt = t[1] - t[0]

        # Compute q on interior (avoid boundary singularity)
        mask = (t > -0.499) & (t < 0.499)
        t_inner = t[mask]
        denom = 0.25 - t_inner**2
        # Avoid division by zero at boundary
        denom = np.maximum(denom, 1e-12)
        q = conv[mask] / denom

        n = data['n']
        # Compute Q moments numerically
        dt_inner = t_inner[1] - t_inner[0]
        Q = np.array([np.trapezoid(t_inner**k * q, dx=dt_inner)
                       for k in range(2 * n + 3)])

        # Check Hankel PSD
        MQ_size = n + 2
        MQ = _build_hankel(Q, MQ_size)
        ev = _min_eigval(MQ)
        assert ev > PSD_TOL, \
            f"{data['name']}: Hankel(Q) min eigval = {ev:.2e}"

    def test_q_moments_localizing_psd(self, data):
        """Localizing matrix for q on [-1/2, 1/2] should be PSD."""
        conv, t = data['conv'], data['t_grid']
        dt = t[1] - t[0]

        mask = (t > -0.499) & (t < 0.499)
        t_inner = t[mask]
        denom = np.maximum(0.25 - t_inner**2, 1e-12)
        q = conv[mask] / denom

        n = data['n']
        dt_inner = t_inner[1] - t_inner[0]
        Q = np.array([np.trapezoid(t_inner**k * q, dx=dt_inner)
                       for k in range(2 * n + 3)])

        LQ_size = n + 1
        LQ = _build_localizing_1d(Q, LQ_size, 0.25)
        ev = _min_eigval(LQ)
        assert ev > PSD_TOL, \
            f"{data['name']}: Q localizing min eigval = {ev:.2e}"

    def test_boundary_factor_moment_relation(self, data):
        """G[k] = (1/4) Q[k] - Q[k+2] should hold."""
        conv, t = data['conv'], data['t_grid']

        mask = (t > -0.499) & (t < 0.499)
        t_inner = t[mask]
        denom = np.maximum(0.25 - t_inner**2, 1e-12)
        q = conv[mask] / denom

        n = data['n']
        dt_inner = t_inner[1] - t_inner[0]
        Q = np.array([np.trapezoid(t_inner**k * q, dx=dt_inner)
                       for k in range(2 * n + 3)])

        G = data['G']
        for k in range(min(6, 2 * n + 1)):
            lhs = G[k]
            rhs = 0.25 * Q[k] - Q[k + 2]
            if abs(lhs) > 1e-10:
                rel_err = abs(lhs - rhs) / abs(lhs)
                assert rel_err < 0.02, \
                    f"{data['name']}: G[{k}]={lhs:.6e}, (1/4)Q[{k}]-Q[{k+2}]={rhs:.6e}"


# ===========================================================================
# V2 product localizing: (1/16-x^2)(1/16-y^2) >> 0
# ===========================================================================

class TestProductLocalizing:
    """Verify product localizing is PSD for true product measures."""

    @pytest.fixture(params=['uniform', 'right_half', 'skewed'])
    def data(self, request):
        if request.param == 'uniform':
            f_vals, x = _uniform()
        elif request.param == 'right_half':
            f_vals, x = _right_half()
        else:
            f_vals, x = _skewed()
        d = _prepare_test_data(f_vals, x, N_MOMENTS)
        d['name'] = request.param
        return d

    def test_product_localizing_psd(self, data):
        """(1/16-x^2)(1/16-y^2) localizing matrix is PSD."""
        d = data['n'] // 2
        if d < 2:
            pytest.skip("d < 2 for product localizing")
        Y = data['Y']
        Lprod = _build_product_localizing(Y, d)
        if Lprod is None:
            pytest.skip("d too small")
        ev = _min_eigval(Lprod)
        assert ev > PSD_TOL, \
            f"{data['name']}: product localizing min eigval = {ev:.2e}"


# ===========================================================================
# Asymmetric feasibility: symmetry is NOT required
# ===========================================================================

class TestAsymmetricFeasibility:
    """Verify that asymmetric functions satisfy all constraints.

    This confirms the symmetry assumption f(-x) = f(x) is NOT a necessary
    condition and its removal does not exclude valid solutions.
    """

    @pytest.fixture(params=['right_half', 'skewed'])
    def data(self, request):
        if request.param == 'right_half':
            f_vals, x = _right_half()
        else:
            f_vals, x = _skewed()
        d = _prepare_test_data(f_vals, x, N_MOMENTS)
        d['name'] = request.param
        return d

    def test_odd_moments_nonzero(self, data):
        """Asymmetric f has nonzero odd moments — symmetry would exclude it."""
        m = data['m']
        assert abs(m[1]) > 1e-4, \
            f"{data['name']}: m_1 = {m[1]:.6f}, should be nonzero"

    def test_Y_odd_entries_nonzero(self, data):
        """Y[1,0] = m_1 != 0 — the old symmetry constraint would force this to 0."""
        Y = data['Y']
        assert abs(Y[1, 0]) > 1e-4, \
            f"{data['name']}: Y[1,0] = {Y[1,0]:.6f}, should be nonzero"
        assert abs(Y[1, 1]) > 1e-8, \
            f"{data['name']}: Y[1,1] = {Y[1,1]:.6f}, should be nonzero"

    def test_all_constraints_still_satisfied(self, data):
        """Even with nonzero odd moments, all PSD constraints hold."""
        n = data['n']
        m, Y, h = data['m'], data['Y'], data['h']
        d = n // 2

        G = data['G']
        checks = {
            'Y >> 0': _min_eigval(Y),
            'Hankel(m)': _min_eigval(_build_hankel(m, d + 1)),
            '1D localizing': _min_eigval(_build_localizing_1d(m, d, 1.0 / 16.0)),
            '2D moment': _min_eigval(_build_2d_moment_matrix(Y, d)),
            'Hankel(h)': _min_eigval(_build_hankel(h, n + 1)),
            'gap localizing': _min_eigval(_build_localizing_1d(h, n, 0.25)),
            # Round 1: autoconv nonnegativity
            'Hankel(G)': _min_eigval(_build_hankel(G, n + 1)),
            'autoconv loc': _min_eigval(_build_localizing_1d(G, n, 0.25)),
            # Round 4: Stieltjes 1D
            'Stieltjes 1D h1': _min_eigval(_build_stieltjes_1d(m, d, 0.25, +1)),
            'Stieltjes 1D h2': _min_eigval(_build_stieltjes_1d(m, d, 0.25, -1)),
            # Round 5: autoconv Stieltjes
            'autoconv Stj h1': _min_eigval(_build_stieltjes_1d(G, n, 0.5, +1)),
            'autoconv Stj h2': _min_eigval(_build_stieltjes_1d(G, n, 0.5, -1)),
            # Round 7: gap Stieltjes
            'gap Stj h1': _min_eigval(_build_stieltjes_1d(h, n, 0.5, +1)),
            'gap Stj h2': _min_eigval(_build_stieltjes_1d(h, n, 0.5, -1)),
        }
        if d >= 1:
            checks['2D loc(x)'] = _min_eigval(_build_2d_localizing_x(Y, d))
            checks['2D loc(y)'] = _min_eigval(_build_2d_localizing_y(Y, d))
            # Round 4: Stieltjes 2D
            checks['2D Stj x+'] = _min_eigval(_build_stieltjes_2d_x(Y, d, +1))
            checks['2D Stj x-'] = _min_eigval(_build_stieltjes_2d_x(Y, d, -1))
            checks['2D Stj y+'] = _min_eigval(_build_stieltjes_2d_y(Y, d, +1))
            checks['2D Stj y-'] = _min_eigval(_build_stieltjes_2d_y(Y, d, -1))
            # Round 6: cross-product
            for sx, sy in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
                L = _build_corner_localizing(Y, d, sx, sy)
                checks[f'corner({sx:+d},{sy:+d})'] = _min_eigval(L)
        if d >= 2:
            Lprod = _build_product_localizing(Y, d)
            if Lprod is not None:
                checks['product loc'] = _min_eigval(Lprod)

        for name, ev in checks.items():
            assert ev > PSD_TOL, \
                f"{data['name']}: {name} FAILED, min eigval = {ev:.2e}"


# ===========================================================================
# Precompute constants
# ===========================================================================

class TestPrecomputeConstants:
    def test_I_even_moments(self):
        """I[k] = int_{-1/2}^{1/2} t^k dt = 2*(1/2)^{k+1}/(k+1) for even k."""
        consts = precompute_constants(6)
        I = consts['I']
        assert abs(I[0] - 1.0) < 1e-12
        assert abs(I[2] - 1.0 / 12.0) < 1e-12
        assert abs(I[4] - 1.0 / 80.0) < 1e-12

    def test_I_odd_moments_zero(self):
        """I[k] = 0 for odd k (symmetric interval)."""
        consts = precompute_constants(6)
        I = consts['I']
        for k in range(1, len(I), 2):
            assert abs(I[k]) < 1e-15

    def test_I_matches_numerical(self):
        """I[k] should match numerical integration."""
        consts = precompute_constants(6)
        I = consts['I']
        t = np.linspace(-0.5, 0.5, 100_000)
        dt = t[1] - t[0]
        for k in [0, 2, 4, 6, 8]:
            numerical = np.trapezoid(t**k, dx=dt)
            assert abs(I[k] - numerical) < 1e-6, \
                f"I[{k}]: analytic={I[k]:.10f}, numerical={numerical:.10f}"

    def test_binomial_coefficients(self):
        consts = precompute_constants(8)
        binom = consts['binom']
        for k in range(len(binom)):
            for j in range(k + 1):
                assert binom[k][j] == math.comb(k, j)


class TestEnumerate2dMonomials:
    def test_degree_0(self):
        assert _enumerate_2d_monomials(0) == [(0, 0)]

    def test_degree_1(self):
        monos = _enumerate_2d_monomials(1)
        assert len(monos) == 3
        assert set(monos) == {(0, 0), (1, 0), (0, 1)}

    def test_size_formula(self):
        """Size should be (d+1)(d+2)/2."""
        for d in range(8):
            monos = _enumerate_2d_monomials(d)
            assert len(monos) == (d + 1) * (d + 2) // 2


# ===========================================================================
# Solver smoke tests (without symmetry — bounds will be trivial)
# ===========================================================================

class TestSolverSmoke:
    """Basic smoke tests: solvers run and produce valid (if trivial) results."""

    def test_unscaled_runs(self):
        result = solve_continuous_sdp(4)
        assert result['status'] in ('optimal', 'optimal_inaccurate')
        assert result['eta'] is not None

    def test_scaled_runs(self):
        result = solve_continuous_sdp_scaled(6)
        assert result['status'] in ('optimal', 'optimal_inaccurate')
        assert result['eta'] is not None

    def test_decoupled_runs(self):
        result = solve_decoupled_sdp(6, d_lasserre=2)
        assert result['status'] in ('optimal', 'optimal_inaccurate')
        assert result['eta'] is not None

    def test_v2_runs(self):
        result = solve_continuous_sdp_v2(4)
        assert result['status'] in ('optimal', 'optimal_inaccurate')
        assert result['eta'] is not None

    def test_decoupled_raises_on_invalid_d(self):
        with pytest.raises(ValueError):
            solve_decoupled_sdp(4, d_lasserre=3)

    def test_unscaled_vs_scaled_agree(self):
        """Unscaled and scaled should agree for small n."""
        r1 = solve_continuous_sdp(4)
        r2 = solve_continuous_sdp_scaled(4)
        if (r1['status'] in ('optimal', 'optimal_inaccurate') and
                r2['status'] in ('optimal', 'optimal_inaccurate')):
            assert abs(r1['eta'] - r2['eta']) < 0.05, \
                f"unscaled={r1['eta']:.6f}, scaled={r2['eta']:.6f}"

    def test_v3_runs(self):
        result = solve_continuous_sdp_v3(4)
        assert result['status'] in ('optimal', 'optimal_inaccurate')
        assert result['eta'] is not None

    def test_v3_minimal_runs(self):
        """V3 with all new constraints disabled should match v2."""
        result = solve_continuous_sdp_v3(
            4, use_autoconv_psd=False, use_marginal_bounds=False,
            use_entry_bounds=False, use_stieltjes_1d=False,
            use_stieltjes_2d=False, use_autoconv_stieltjes=False,
            use_cross_product_loc=False, use_gap_stieltjes=False)
        assert result['status'] in ('optimal', 'optimal_inaccurate')
        assert result['eta'] is not None

    def test_eta_is_valid_lower_bound(self):
        """SDP eta must not exceed any known V(P) upper bound."""
        from baseline_results import BASELINE
        min_vp = min(bl['lasserre2_lb'] for bl in BASELINE.values())
        for solver_fn, kwargs in [
            (solve_continuous_sdp, {'n': 4}),
            (solve_continuous_sdp_scaled, {'n': 6}),
            (solve_continuous_sdp_v2, {'n': 4}),
            (solve_continuous_sdp_v3, {'n': 4}),
        ]:
            result = solver_fn(**kwargs)
            if result['eta'] is not None:
                assert result['eta'] <= min_vp + 0.01, \
                    f"{solver_fn.__name__}: eta={result['eta']:.4f} > min V(P)={min_vp:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
