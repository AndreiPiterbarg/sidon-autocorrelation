"""Independent post-hoc verification of an LP solution.

Given a solver output (alpha, lambda, q, c), evaluate the certificate
identity:

  p_lambda(mu) - alpha == sum_beta c_beta mu^beta + q(mu) (sum mu - 1)

both symbolically (compare coefficient by coefficient) and numerically
(Monte Carlo over Delta_d). This is a soundness check: if the equality
holds and c >= 0, then on Delta_d we have p_lambda(mu) = alpha + sum
c_beta mu^beta >= alpha (since each mu^beta >= 0).
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from lasserre.polya_lp.build import BuildResult, _coeff_W_vector
from lasserre.polya_lp.poly import enum_monomials_le, index_map


def verify_certificate_symbolic(
    build: BuildResult,
    M_mats_eff,
    alpha: float,
    x: np.ndarray,
    tol: float = 1e-8,
) -> dict:
    """Coefficient-wise check that the certificate identity holds."""
    R = build.options.R
    monos_le_R = build.monos_le_R
    monos_le_Rm1 = build.monos_le_Rm1
    beta_to_idx = index_map(monos_le_R)
    q_to_idx = index_map(monos_le_Rm1)

    # Extract solver values
    lam = (x[build.lambda_idx] if build.fixed_lambda is None
           else build.fixed_lambda)
    q_vec = x[build.q_idx] if monos_le_Rm1 else np.zeros(0)
    c_vec = x[build.c_idx]

    # LHS coefficients: p_lambda(mu) - alpha
    lhs = np.zeros(len(monos_le_R), dtype=np.float64)
    # alpha contributes -alpha at beta = 0
    lhs[beta_to_idx[tuple([0] * (len(monos_le_R[0])))]] -= alpha
    # p_lambda contributes at |beta| = 2
    for w, M_W in enumerate(M_mats_eff):
        coeff = _coeff_W_vector(M_W, beta_to_idx)
        for beta_i, v in coeff.items():
            lhs[beta_i] += lam[w] * v

    # RHS coefficients: sum c_beta mu^beta + q(mu)(sum mu_i - 1)
    rhs = c_vec.copy()
    if len(q_vec) > 0:
        d = len(monos_le_R[0])
        for k_idx, K in enumerate(monos_le_Rm1):
            qk = q_vec[k_idx]
            if qk == 0:
                continue
            # +q_K * (-1) at beta = K (since q*(sum mu - 1) = q sum mu - q,
            # constant term is -q)
            row_K = beta_to_idx.get(K)
            if row_K is not None:
                rhs[row_K] -= qk
            for j in range(d):
                K_plus_ej = list(K)
                K_plus_ej[j] += 1
                row_idx = beta_to_idx.get(tuple(K_plus_ej))
                if row_idx is not None:
                    rhs[row_idx] += qk

    diff = lhs - rhs
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    n_violations = int((c_vec < -tol).sum())

    return {
        "max_coeff_residual": max_abs,
        "min_c_beta": float(c_vec.min()) if c_vec.size else 0.0,
        "n_negative_c_beta": n_violations,
        "passed": (max_abs < tol and n_violations == 0),
    }


def verify_certificate_montecarlo(
    build: BuildResult,
    M_mats_eff,
    alpha: float,
    x: np.ndarray,
    n_samples: int = 1000,
    seed: int = 0,
) -> dict:
    """Sample random mu in Delta_d and check p_lambda(mu) - alpha >= 0."""
    rng = np.random.default_rng(seed)
    d = len(build.monos_le_R[0])

    lam = (x[build.lambda_idx] if build.fixed_lambda is None
           else build.fixed_lambda)
    M_eff = sum(lam[w] * M_mats_eff[w] for w in range(len(M_mats_eff)))

    # Dirichlet samples for diversity
    pts = rng.dirichlet(alpha=np.ones(d), size=n_samples)
    vals = np.einsum("ki,ij,kj->k", pts, M_eff, pts)
    margins = vals - alpha
    return {
        "n_samples": n_samples,
        "min_margin": float(margins.min()),
        "median_margin": float(np.median(margins)),
        "n_violations": int((margins < -1e-9).sum()),
    }
