"""
Basic validation tests for the Sidon autocorrelation optimizer.

Tests core mathematical functions from sidon_core.py:
- Simplex projection
- Autoconvolution computation
- LogSumExp smoothing
- Gradient correctness
- Known analytical solutions
"""

import sys
import os
import numpy as np
import pytest

# Add prev_attempts/ to path so we can import sidon_core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'prev_attempts'))
from sidon_core import (
    project_simplex_nb,
    convolve_full,
    autoconv_coeffs,
    logsumexp_nb,
    lse_obj_nb,
    lse_grad_nb,
    exact_val,
    warmup,
)

# Trigger Numba JIT compilation before tests run
warmup()


# ─── Simplex projection tests ───────────────────────────────────────────────

class TestSimplexProjection:
    def test_already_on_simplex(self):
        """A point already on the simplex should be unchanged."""
        x = np.array([0.2, 0.3, 0.5])
        proj = project_simplex_nb(x)
        assert np.allclose(proj, x, atol=1e-12)

    def test_sums_to_one(self):
        """Projection output must sum to 1."""
        x = np.array([2.0, -1.0, 0.5, 3.0])
        proj = project_simplex_nb(x)
        assert abs(proj.sum() - 1.0) < 1e-12

    def test_nonnegative(self):
        """Projection output must be nonnegative."""
        x = np.array([-5.0, -2.0, 10.0, 0.1])
        proj = project_simplex_nb(x)
        assert np.all(proj >= -1e-15)

    def test_uniform(self):
        """A constant vector should project to uniform distribution."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        proj = project_simplex_nb(x)
        assert np.allclose(proj, 0.25, atol=1e-12)


# ─── Autoconvolution tests ──────────────────────────────────────────────────

class TestAutoconvolution:
    def test_uniform_peak_is_two(self):
        """Uniform distribution f=2 on [-1/4,1/4] (integral=1) has peak
        autoconvolution = 2.0 at t=0, since (f*f)(0) = integral f^2 = 4*0.5 = 2."""
        P = 100
        x = np.ones(P) / P
        c = autoconv_coeffs(x, P)
        peak = np.max(c)
        assert abs(peak - 2.0) < 0.01, f"Uniform peak should be ~2.0, got {peak}"

    def test_dirac_peak(self):
        """A delta concentrated in one bin should have peak = 2P."""
        P = 50
        x = np.zeros(P)
        x[P // 2] = 1.0
        c = autoconv_coeffs(x, P)
        peak = np.max(c)
        assert abs(peak - 2 * P) < 0.1, f"Dirac peak should be {2*P}, got {peak}"

    def test_symmetry(self):
        """Autoconvolution of a symmetric function should be symmetric."""
        P = 20
        x = np.random.default_rng(42).dirichlet(np.ones(P))
        x_sym = 0.5 * (x + x[::-1])
        x_sym /= x_sym.sum()
        c = autoconv_coeffs(x_sym, P)
        n = len(c)
        # c should be symmetric: c[k] == c[n-1-k]
        assert np.allclose(c, c[::-1], atol=1e-10), "Autoconvolution of symmetric f should be symmetric"

    def test_convolution_length(self):
        """Autoconvolution of P-length vector should have length 2P-1."""
        P = 30
        x = np.ones(P) / P
        c = autoconv_coeffs(x, P)
        assert len(c) == 2 * P - 1

    def test_nonneg_output(self):
        """Autoconvolution of nonnegative f should be nonnegative."""
        P = 50
        x = np.random.default_rng(123).dirichlet(np.ones(P))
        c = autoconv_coeffs(x, P)
        assert np.all(c >= -1e-15), "Autoconvolution of nonneg function should be nonneg"


# ─── LogSumExp tests ────────────────────────────────────────────────────────

class TestLogSumExp:
    def test_upper_bound(self):
        """LSE_beta(c) >= max(c)."""
        c = np.array([1.0, 2.0, 3.0, 2.5])
        for beta in [1.0, 10.0, 100.0]:
            lse = logsumexp_nb(c, beta)
            assert lse >= np.max(c) - 1e-10, f"LSE({beta}) = {lse} < max = {np.max(c)}"

    def test_convergence_to_max(self):
        """LSE_beta(c) -> max(c) as beta -> infinity."""
        c = np.array([1.0, 2.0, 3.0, 2.5])
        lse_low = logsumexp_nb(c, 1.0)
        lse_high = logsumexp_nb(c, 10000.0)
        assert abs(lse_high - 3.0) < 0.001, f"LSE at large beta should be ~3.0, got {lse_high}"
        assert lse_low > lse_high, "LSE should decrease as beta increases"

    def test_equal_values(self):
        """LSE of equal values should return that value + log(n)/beta."""
        n = 5
        val = 2.0
        c = np.full(n, val)
        beta = 10.0
        expected = val + np.log(n) / beta
        lse = logsumexp_nb(c, beta)
        assert abs(lse - expected) < 1e-10


# ─── Gradient tests ─────────────────────────────────────────────────────────

class TestGradient:
    def test_finite_difference_agreement(self):
        """Analytical gradient should match central finite differences."""
        P = 10
        rng = np.random.default_rng(42)
        x = rng.dirichlet(np.ones(P))
        beta = 10.0
        eps = 1e-5

        grad_analytical = lse_grad_nb(x, P, beta)

        grad_fd = np.zeros(P)
        for i in range(P):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad_fd[i] = (lse_obj_nb(x_plus, P, beta) - lse_obj_nb(x_minus, P, beta)) / (2 * eps)

        rel_error = np.linalg.norm(grad_analytical - grad_fd) / (np.linalg.norm(grad_fd) + 1e-15)
        assert rel_error < 1e-4, f"Gradient relative error = {rel_error:.6e}, should be < 1e-4"


# ─── Exact evaluation tests ─────────────────────────────────────────────────

class TestExactEvaluation:
    def test_uniform_exact(self):
        """Exact evaluation of uniform distribution should give peak ~2.0."""
        P = 50
        x = np.ones(P) / P
        peak = exact_val(x, P)
        assert abs(peak - 2.0) < 0.02, f"Exact uniform peak should be ~2.0, got {peak}"

    def test_exact_vs_discrete_agreement(self):
        """Exact breakpoint evaluation should roughly agree with discrete coefficients."""
        P = 30
        rng = np.random.default_rng(99)
        x = rng.dirichlet(np.ones(P))
        discrete_peak = np.max(autoconv_coeffs(x, P))
        exact_peak = exact_val(x, P)
        rel_diff = abs(exact_peak - discrete_peak) / discrete_peak
        assert rel_diff < 0.02, f"Exact vs discrete differ by {rel_diff:.4f}, should be < 0.02"


# ─── Integration / optimization sanity tests ────────────────────────────────

class TestOptimizationSanity:
    def test_optimizer_improves_over_uniform(self):
        """A few steps of gradient descent should beat the uniform distribution."""
        from sidon_core import armijo_step_nb
        P = 20
        x = np.ones(P) / P
        beta = 10.0
        initial_obj = lse_obj_nb(x, P, beta)
        for _ in range(50):
            g = lse_grad_nb(x, P, beta)
            x, _, _ = armijo_step_nb(x, g, P, beta, 0.1)
        final_obj = lse_obj_nb(x, P, beta)
        assert final_obj < initial_obj, "Gradient descent should improve the objective"

    def test_simplex_constraint_maintained(self):
        """After optimization steps, solution should remain on the simplex."""
        from sidon_core import armijo_step_nb
        P = 20
        x = np.ones(P) / P
        beta = 10.0
        for _ in range(100):
            g = lse_grad_nb(x, P, beta)
            x, _, _ = armijo_step_nb(x, g, P, beta, 0.1)
        assert abs(x.sum() - 1.0) < 1e-10, f"Sum should be 1.0, got {x.sum()}"
        assert np.all(x >= -1e-15), "All entries should be nonneg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
