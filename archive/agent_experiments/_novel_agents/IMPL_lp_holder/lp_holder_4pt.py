"""4-point Hausdorff Lasserre for the L^2 Hoelder chain bound on C_{1a}.

================================================================================
RIGOROUS DERIVATION (no shortcuts)
================================================================================

Goal: rigorous lower bound on
    C_{1a} = inf_{f admissible} ||f*f||_inf
where f admissible means f >= 0, supp(f) subset [-1/4, 1/4], int f = 1.

Hoelder chain (rigorous for any nonneg g with ||g||_1 = 1):
    ||g||_inf >= ||g||_p^{p/(p-1)} = (||g||_p^p)^{1/(p-1)}.

For g = f*f: ||g||_1 = (int f)^2 = 1, so ||g||_p^p = int (f*f)^p dt and
    ||f*f||_inf >= (int (f*f)^p dt)^{1/(p-1)}.

Taking inf over f (rigorous since x -> x^{1/(p-1)} is monotone):
    C_{1a} >= (inf_f int (f*f)^p dt)^{1/(p-1)}.

For p = 2: C_{1a} >= inf_f ||f*f||_2^2.

So a rigorous LB on inf_f ||f*f||_2^2 gives a rigorous LB on C_{1a}.

================================================================================
CONNECTION TO AUTOCORRELATION
================================================================================

For real f, ||f*f||_2^2 = ||r_f||_2^2 where r_f(t) := (f * f^-)(t) = int f(s) f(s-t) ds
is the autocorrelation. Proof: by Plancherel,
    ||f*f||_2^2 = || (f*f)^^ ||_2^2 = || f^^ * f^^ ||_2^2 = || (f^)^2 ||_2^2 = int |f^|^4 d xi
    ||r_f||_2^2 = || r_f^^ ||_2^2 = || |f^|^2 ||_2^2 = int |f^|^4 d xi.
So ||f*f||_2^2 = ||r_f||_2^2 exactly.

================================================================================
RESCALED COORDINATES
================================================================================

To match the existing infrastructure in `lasserre/threepoint_full.py`, work in
rescaled coords v = 4u so v in [-1, 1], with tilde_f(v) := (1/4) f(v/4).

Then int tilde_f dv = 1, supp(tilde_f) subset [-1, 1].

Autocorrelation tilde_R(tau) := int tilde_f(s) tilde_f(s - tau) ds, supp [-2, 2].
Relation: tilde_R(tau) = (1/4) R(tau/4) where R = autocorrelation in original.
Therefore || tilde_R ||_2^2 (rescaled, on [-2, 2]) = (1/4) || R ||_2^2 (original, on [-1/2, 1/2])
                                                  = (1/4) || f*f ||_2^2 (original).
So  || f*f ||_2^2 = 4 * || tilde_R ||_2^2 (rescaled).

================================================================================
AUTOCORRELATION MOMENTS
================================================================================

Define tilde_r_a := int tau^a tilde_R(tau) d tau for a = 0, 1, 2, ....
Direct calculation (substitute u = s - tau, expand binomial):
    tilde_r_a = sum_{j=0}^a C(a, j) (-1)^j m_{a-j} m_j
where m_k := int v^k tilde_f(v) dv = k-th moment of tilde_f in rescaled coords.

In two-point lift: g_{ab} := m_a m_b in rank-1 (lifted to PSD-symmetric in Lasserre),
so tilde_r_a = sum_{j=0}^a C(a, j) (-1)^j g_{a-j, j} (linear in g).

================================================================================
LEGENDRE EXPANSION ON [-2, 2]
================================================================================

Standard Legendre P_j(x) on [-1, 1]: int P_j P_k dx = 2/(2j+1) delta_{jk}.
L^2-orthonormal Legendre on [-2, 2]:
    q_j(tau) = sqrt((2j+1)/4) * P_j(tau/2).
Verify: int_{-2}^{2} q_j^2 d tau = (2j+1)/4 * int_{-2}^{2} P_j(tau/2)^2 d tau.
        Substitute x = tau/2, d tau = 2 dx: = (2j+1)/4 * 2 * int_{-1}^{1} P_j^2 dx = (2j+1)/4 * 2 * 2/(2j+1) = 1. CHECK.

In monomial basis: q_j(tau) = sum_r Q[j, r] tau^r, where
    Q[j, r] := sqrt((2j+1)/4) * P_j_coef[r] / 2^r
and P_j_coef[r] is the coefficient of x^r in P_j(x).

================================================================================
RIGOROUS LOWER BOUND ON || tilde_R ||_2^2
================================================================================

By Bessel's inequality (truncation of orthonormal expansion):
    || tilde_R ||_2^2 >= sum_{j=0}^{N_leg} rho_j^2
where rho_j = <q_j, tilde_R> = int q_j(tau) tilde_R(tau) d tau = sum_r Q[j, r] tilde_r_r.

This is a RIGOROUS LB (with possible slack from truncation; the slack vanishes as
N_leg -> infty since the autocorrelation tilde_R is in L^2([-2, 2])).

Expanding:
    sum_j rho_j^2 = sum_{r, s} D[r, s] tilde_r_r tilde_r_s
where  D[r, s] := sum_{j=0}^{N_leg} Q[j, r] Q[j, s].

Since tilde_r_a = sum_l C(a, l) (-1)^l g_{a-l, l},
    tilde_r_r tilde_r_s = sum_{l, m} C(r, l) C(s, m) (-1)^{l+m} g_{r-l, l} g_{s-m, m}.

In FOUR-POINT LIFT, g_{a, b} g_{c, d} -> z_{a, b, c, d} where z is the lifted measure
on [-1, 1]^4 satisfying PSD-Hausdorff constraints.  Therefore:
    sum_j rho_j^2 = sum_{r, s, l, m} D[r, s] C(r, l) C(s, m) (-1)^{l+m} z_{r-l, l, s-m, m}.

This is a LINEAR functional of the 4-point moment variables z.

================================================================================
RELAXATION: WHY THIS LOWER-BOUNDS THE TRUE INFIMUM
================================================================================

For any admissible f, the moments {m_k} satisfy Hausdorff PSD constraints (Hankel
M(m) >= 0, localizer (1 - x^2) M(m) >= 0).  The 2-point moments g_{ab} = m_a m_b
form a rank-1 matrix; relaxing to general PSD-Hausdorff 2-point moments (matrix
M_2(g) >= 0, localizers >= 0) ENLARGES the feasible set.  Likewise lifting to
4-point z_{abcd} = m_a m_b m_c m_d with relaxed PSD-Hausdorff (in 4D) further
enlarges.  Minimizing a linear functional over the larger set gives a LOWER value
than over the original set:
    SDP value <= inf_{f admissible} sum_j rho_j(f)^2 <= inf_{f admissible} || tilde_R ||_2^2.

Hence:
    || f*f ||_2^2 (original) = 4 || tilde_R ||_2^2 (rescaled) >= 4 * SDP value
    C_{1a} >= 4 * SDP value.

================================================================================
SYMMETRY REDUCTIONS USED
================================================================================

S_4 exchangeability of the 4 variables: the 4-point measure z(v_1, v_2, v_3, v_4)
is invariant under any permutation of (v_1, ..., v_4).  This is a LINEARITY-
PRESERVING reduction since the objective is linear (so symmetrizing z does not
worsen the bound).

REFLECTION (Z_2): z is invariant under v_i -> -v_i for all i jointly, i.e.,
z_{abcd} = 0 if a + b + c + d odd.  This corresponds to symmetrizing the
underlying f to f_sym(x) = (f(x) + f(-x))/2, which is also admissible.
(WARNING: this is a NONTRIVIAL choice for the LB problem.  See justification
below.)

JUSTIFICATION FOR REFLECTION SYMMETRY:
For the L^2 objective ||r_f||_2^2 = int |f^|^4 d xi, replacing f with its
reflection f^- gives ||f^- * (f^-)^-||_2^2 = ||r_f||_2^2 (autocorrelation under
reflection). So  C_{1a}^{(L2)} := inf_f ||f*f||_2^2  equals
    inf_f ||r_f||_2^2 = inf_f int|f^|^4 d xi.

For any admissible f, define f_sym(x) = (f(x) + f(-x))/2.  This is admissible.
Question: is ||r_{f_sym}||_2^2 <= ||r_f||_2^2?
    f_sym^^(xi) = (f^(xi) + f^(-xi)) / 2.
    For real f: f^(-xi) = (f^(xi))*  (complex conjugate).  So
    f_sym^^(xi) = (f^(xi) + (f^(xi))*) / 2 = Re(f^(xi)).
    |f_sym^^(xi)|^2 = (Re f^)^2.
    |f^(xi)|^2 = (Re f^)^2 + (Im f^)^2.
    So |f_sym^^|^2 <= |f^|^2 pointwise.
    Hence |f_sym^^|^4 <= |f^|^4, and int |f_sym^^|^4 <= int |f^|^4.
    So ||r_{f_sym}||_2^2 <= ||r_f||_2^2.
This SHOWS that the inf over admissible f of ||r_f||_2^2 is achieved at a
symmetric f WITHOUT LOSS.  Hence WLOG f is symmetric, and we can use
reflection_zero_odd = True in the moment map.  ABSOLUTE CORRECTNESS PRESERVED.

================================================================================
COMPLEXITY
================================================================================

At Lasserre level k:
- 4D moment matrix M_k(z) of size C(k+4, 4).  k=2: 15.  k=3: 35.  k=4: 70.  k=5: 126.
- 4 localizer matrices of size C(k+3, 3).  k=2: 10.  k=3: 20.  k=4: 35.  k=5: 56.
- After S_4 x Z_2 orbit reduction, n_orbits = number of distinct sorted-decreasing
  even-total-degree multi-indices alpha with |alpha| <= 2k.

This module implements: compute Legendre coeffs, build D matrix, build SDP, solve.
"""
from __future__ import annotations

import math
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

# Add repo root to path for lasserre imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lasserre.threepoint_full import (
    enum_multi_indices,
    moment_matrix,
    localizer_matrix,
    BuildInfo,
)
from lasserre.track1_4point_lift import FourPointMap


# =====================================================================
# Legendre coefficients on [-1, 1] (standard P_j) using EXACT rationals
# =====================================================================

def standard_legendre_coeffs_exact(N: int) -> List[List[Fraction]]:
    """P_coef[j][r] = exact rational coefficient of x^r in standard Legendre P_j(x).
    Uses the recurrence (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x).
    P_0(x) = 1, P_1(x) = x.

    Returns list-of-lists where P_coef[j] has length j+1.
    """
    if N < 0:
        return []
    P: List[List[Fraction]] = [[Fraction(1)]]
    if N >= 1:
        P.append([Fraction(0), Fraction(1)])
    for n in range(1, N):
        # P_{n+1}: degree n+1.
        new = [Fraction(0)] * (n + 2)
        # (2n+1) x P_n: shift P_n up by one degree.
        for r in range(len(P[n])):
            new[r + 1] += Fraction(2 * n + 1, n + 1) * P[n][r]
        # - n P_{n-1}: same degree.
        for r in range(len(P[n - 1])):
            new[r] -= Fraction(n, n + 1) * P[n - 1][r]
        P.append(new)
    return P


def legendre_orthonormal_minus2_2_Q(N: int) -> np.ndarray:
    """Q[j, r] = coefficient of tau^r in q_j(tau), where q_j is L^2-orthonormal
    on [-2, 2] with respect to Lebesgue measure:
        q_j(tau) = sqrt((2j+1)/4) * P_j(tau/2).

    Verifies: int_{-2}^{2} q_j(tau)^2 d tau = 1.

    Returns shape (N+1, N+1) (lower-triangular: Q[j, r] = 0 for r > j).
    """
    P_coef = standard_legendre_coeffs_exact(N)
    Q = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        scale_sqrt = math.sqrt((2 * j + 1) / 4.0)
        for r in range(j + 1):
            # P_j_coef[r] is exact rational.  q_j(tau) = sqrt((2j+1)/4) * sum_r P_j_coef[r] (tau/2)^r
            # = sqrt((2j+1)/4) * sum_r P_j_coef[r] / 2^r * tau^r
            Q[j, r] = scale_sqrt * float(P_coef[j][r]) / (2 ** r)
    return Q


def verify_legendre_orthonormality(N: int, n_quad: int = 100) -> Dict[Tuple[int, int], float]:
    """Numerically verify int_{-2}^{2} q_i q_j d tau = delta_{ij} for j <= N.
    Returns dict (i, j) -> integral value.
    """
    Q = legendre_orthonormal_minus2_2_Q(N)
    # Gauss-Legendre quadrature on [-2, 2]
    # Use scipy or hand-roll
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(n_quad)
    # nodes in [-1, 1]; rescale to [-2, 2]: tau = 2*node, weight *= 2
    tau = 2.0 * nodes
    w = 2.0 * weights
    out: Dict[Tuple[int, int], float] = {}
    for i in range(N + 1):
        for j in range(i + 1):
            # q_i(tau) = sum_r Q[i, r] tau^r
            qi_at_tau = np.zeros_like(tau)
            for r in range(i + 1):
                qi_at_tau += Q[i, r] * (tau ** r)
            qj_at_tau = np.zeros_like(tau)
            for r in range(j + 1):
                qj_at_tau += Q[j, r] * (tau ** r)
            integral = float(np.sum(w * qi_at_tau * qj_at_tau))
            out[(i, j)] = integral
            out[(j, i)] = integral
    return out


# =====================================================================
# D matrix: D[r, s] = sum_{j=0}^{N_leg} Q[j, r] Q[j, s]
# =====================================================================

def build_D_matrix(N_leg: int) -> np.ndarray:
    """D[r, s] = sum_{j=0}^{N_leg} Q[j, r] Q[j, s].
    Shape (N_leg+1, N_leg+1).
    Truncated Bessel-Plancherel partial sum representation of <., .> on [-2, 2]
    via L^2-orthonormal Legendre.
    """
    Q = legendre_orthonormal_minus2_2_Q(N_leg)
    # D = Q^T Q
    D = Q.T @ Q
    return D


# =====================================================================
# Build the autocorrelation moments e_a as linear expressions in g (= z marginal)
# =====================================================================

def build_autocorr_moments(z_map: FourPointMap, k: int, N_leg: int) -> List[cp.Expression]:
    """Build  e_a := tilde_r_a = sum_l C(a, l) (-1)^l g_{a-l, l}  for a = 0, 1, ..., N_leg,
    where g_{ab} = z_{a, b, 0, 0} (2-marginal of 4-point moments).

    Returns list of length N_leg + 1 of CVXPY expressions (linear in z variables).
    """
    e: List[cp.Expression] = []
    for a in range(N_leg + 1):
        if a > 2 * k:
            # Out of range: marginal not available
            e.append(cp.Constant(0.0))
            continue
        terms: List[cp.Expression] = []
        for l in range(a + 1):
            j_idx = l
            i_idx = a - l
            # g_{i_idx, j_idx} = z_{i_idx, j_idx, 0, 0}
            if i_idx + j_idx > 2 * k:
                continue
            coef = math.comb(a, l) * ((-1) ** l)
            terms.append(coef * z_map.get((i_idx, j_idx, 0, 0)))
        if terms:
            e.append(cp.sum(cp.hstack(terms)))
        else:
            e.append(cp.Constant(0.0))
    return e


# =====================================================================
# Schur LMI for the L^2 objective: minimize u s.t. u >= ||Q e||^2
# =====================================================================

def build_schur_lmi(u_var: cp.Variable, e_exprs: List[cp.Expression], Q: np.ndarray) -> cp.Constraint:
    """Build the Schur LMI:
       [ u        rho^T   ]
       [ rho      I_{N+1} ]   >> 0
    where rho = Q e is the vector of truncated Legendre coefficients,
          I_{N+1} = identity of size (N_leg + 1).

    PSD of this block <==> u >= rho^T rho = ||Q e||^2 = e^T (Q^T Q) e = e^T D e.
    Since D = Q^T Q is PSD, this forces u >= 0.

    Returns a CVXPY PSD constraint (LMI).
    """
    N_leg_plus_1 = Q.shape[0]
    if Q.shape[1] != len(e_exprs):
        raise ValueError(f"Q has {Q.shape[1]} cols, e has {len(e_exprs)}")

    # rho_j = sum_r Q[j, r] e_r,  shape (N_leg+1,)
    rho_terms: List[List[cp.Expression]] = []
    for j in range(N_leg_plus_1):
        rj_terms: List[cp.Expression] = []
        for r in range(Q.shape[1]):
            if abs(Q[j, r]) < 1e-15:
                continue
            rj_terms.append(Q[j, r] * e_exprs[r])
        if rj_terms:
            rho_terms.append([cp.sum(cp.hstack(rj_terms))])
        else:
            rho_terms.append([cp.Constant(0.0)])

    # Build the LMI block:
    #   row 0:  [ u_var,    rho_0,     rho_1,     ..., rho_N ]
    #   row j+1: [ rho_j,    delta_{1,1}, ..., delta_{1, N+1} ]   (identity in lower-right block)
    n = N_leg_plus_1
    block_rows: List[List[cp.Expression]] = []

    # First row
    first_row: List[cp.Expression] = [u_var]
    for j in range(n):
        first_row.append(rho_terms[j][0])
    block_rows.append(first_row)

    # Identity rows
    for i in range(n):
        row: List[cp.Expression] = [rho_terms[i][0]]
        for jj in range(n):
            if i == jj:
                row.append(cp.Constant(1.0))
            else:
                row.append(cp.Constant(0.0))
        block_rows.append(row)

    M = cp.bmat(block_rows)
    return M >> 0


# =====================================================================
# Build full SDP problem
# =====================================================================

def build_4pt_l2_sdp(
    k: int,
    N_leg: int,
    *,
    reflection_zero_odd: bool = True,
    support_half_rescaled: float = 1.0,
) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """Build the 4-point Lasserre SDP minimizing the (truncated) L^2 norm of
    the autocorrelation tilde_R, in RESCALED coordinates.

    Uses the Schur-complement LMI formulation:
        minimize u
        s.t.  [ u    rho^T  ]
              [ rho   I     ] >> 0    (forces u >= ||rho||^2 = sum rho_j^2 = ||tilde_R||_2^2-truncated)
        where rho_j = (Q e)_j = sum_r Q[j, r] e_r,  e_r linear in z marginals.

    This is RIGOROUSLY a LB on inf_f sum_j rho_j(f)^2 because:
      (1) For any admissible f, the corresponding rank-1 z gives u_min = sum rho_j(f)^2 >= 0
          (with equality at the optimum). The relaxation z >= rank-1 has SDP value <=
          sum rho_j(f)^2 for that f, hence SDP value <= inf_f sum rho_j(f)^2.
      (2) sum_{j=0}^{N_leg} rho_j(f)^2 <= ||tilde_R||_2^2 by Bessel.
      (3) ||tilde_R||_2^2 (rescaled) = (1/4) ||f*f||_2^2 (original).
      (4) C_{1a} >= ||f*f||_2^2 by Hoelder.

    Therefore: C_{1a} >= 4 * SDP_value (rigorous, all chain steps verified).

    Args:
        k: Lasserre level (max moment degree = 2k).
        N_leg: number of Legendre polynomials retained.  Need N_leg <= k.
        reflection_zero_odd: enforce z_alpha = 0 for |alpha| odd (justified by symmetrization).
        support_half_rescaled: support half-width (rescaled), default 1.0.

    Returns:
        problem, info, misc dict containing {"z", "u", "e_exprs", "Q"}.
    """
    if N_leg > k:
        raise ValueError(f"N_leg ({N_leg}) must be <= k ({k}) so 2 N_leg <= 2k.")

    t0 = time.time()

    # 4-point moment variable map
    z_map = FourPointMap(max_deg=2 * k, reflection_zero_odd=reflection_zero_odd)

    constraints: List = []

    # Normalization: z_{0,0,0,0} = 1
    constraints.append(z_map.get((0, 0, 0, 0)) == 1)

    block_sizes: List[int] = []

    # 4D moment matrix PSD
    M4 = moment_matrix(z_map, k)
    constraints.append(M4 >> 0)
    block_sizes.append(M4.shape[0])

    # 4 box localizers PSD: (support_half^2 - v_i^2) M_{k-1}(z) >> 0
    if k >= 1:
        for axis in range(4):
            L = localizer_matrix(z_map, k, axis=axis, support_half=support_half_rescaled)
            constraints.append(L >> 0)
            block_sizes.append(L.shape[0])

    # Build autocorrelation moments e_a (linear in z marginals).
    # We need e_a up to degree 2 * (k_R) where k_R is the level of the autocorrelation
    # Hankel matrix.  Choose k_R = N_leg, so we need e_0, ..., e_{2 N_leg}.
    # But also need e_{2 N_leg + 2} for the localizer.  So total need: e_0, ..., e_{2 N_leg + 2}.
    max_e_deg = 2 * N_leg + 2
    if max_e_deg > 2 * k:
        max_e_deg = 2 * k  # truncate to available
    e_exprs = build_autocorr_moments(z_map, k, max_e_deg)

    # === ADDITIONAL CONSTRAINT: tilde_R is a positive measure on [-2, 2] ===
    # The autocorrelation tilde_R(tau) of any nonneg tilde_f is itself a nonneg measure.
    # So its moments e_a satisfy Hausdorff PSD on [-2, 2]:
    #   - Hankel matrix [e_{i+j}]_{i,j} >> 0  (positive measure)
    #   - Localizer [(4 - tau^2) e]_{i+j} = [4 e_{i+j} - e_{i+j+2}] >> 0  (support [-2, 2])
    # This is a TIGHTENING of the 4-point relaxation: rank-1 z always satisfies this,
    # but pseudo-z need not.

    # Build Hankel matrix of e
    k_R = N_leg  # level of the tilde_R moment hierarchy
    if 2 * k_R <= max_e_deg:
        hankel_rows: List[List[cp.Expression]] = []
        for i in range(k_R + 1):
            row: List[cp.Expression] = []
            for j in range(k_R + 1):
                if i + j <= max_e_deg:
                    row.append(e_exprs[i + j])
                else:
                    row.append(cp.Constant(0.0))
            hankel_rows.append(row)
        H = cp.bmat(hankel_rows)
        constraints.append(H >> 0)
        block_sizes.append(k_R + 1)

        # Localizer on tilde_R for support [-2, 2]: g(tau) = 4 - tau^2 >> 0.
        # Localizer matrix L_{k_R - 1}: L[i, j] = 4 * e_{i+j} - e_{i+j+2}.
        if k_R >= 1:
            loc_rows: List[List[cp.Expression]] = []
            for i in range(k_R):
                row = []
                for j in range(k_R):
                    if i + j + 2 <= max_e_deg:
                        row.append(4.0 * e_exprs[i + j] - e_exprs[i + j + 2])
                    elif i + j <= max_e_deg:
                        # Cannot enforce; skip this row (set all to 0 to keep PSD trivially)
                        row.append(cp.Constant(0.0))
                    else:
                        row.append(cp.Constant(0.0))
                loc_rows.append(row)
            if k_R >= 1:
                L_R = cp.bmat(loc_rows)
                constraints.append(L_R >> 0)
                block_sizes.append(k_R)

    # Build Legendre Q matrix on [-2, 2]
    Q = legendre_orthonormal_minus2_2_Q(N_leg)

    # Epigraph variable u (objective)
    u_var = cp.Variable(name="u_l2")

    # Schur LMI: u >= sum rho_j^2
    e_for_schur = e_exprs[: N_leg + 1]
    schur_constraint = build_schur_lmi(u_var, e_for_schur, Q)
    constraints.append(schur_constraint)
    block_sizes.append(N_leg + 2)  # size of Schur block

    problem = cp.Problem(cp.Minimize(u_var), constraints)

    info = BuildInfo(
        k=k,
        N=N_leg,
        with_3pt=True,  # using "with_3pt" as "with lift"; here it's 4-point.
        n_orbits_y=z_map.n_orbits() if hasattr(z_map, "n_orbits") else len(z_map._var_by_canon),
        n_orbits_g=0,
        n_orbits_m=0,
        block_sizes=block_sizes,
        n_constraints=len(constraints),
        build_seconds=time.time() - t0,
    )
    return problem, info, {"z": z_map, "u": u_var, "e_exprs": e_exprs, "Q": Q}


# =====================================================================
# Solver wrapper
# =====================================================================

def solve_l2_sdp(
    k: int,
    N_leg: int,
    *,
    solver: str = "MOSEK",
    verbose: bool = False,
    reflection_zero_odd: bool = True,
) -> Dict:
    """Build and solve the 4-point Lasserre SDP.

    Returns a dict containing:
      - 'sdp_value_rescaled': the SDP optimal value (rescaled, |tilde_R|_2^2 LB)
      - 'lb_C1a': the resulting rigorous LB on C_{1a} = 4 * sdp_value_rescaled.
      - 'status': solver status
      - 'wall_time_sec': total solve time
      - 'build_info': metadata
      - 'k', 'N_leg': inputs
    """
    t0 = time.time()
    problem, info, misc = build_4pt_l2_sdp(
        k=k, N_leg=N_leg, reflection_zero_odd=reflection_zero_odd
    )
    t1 = time.time()
    build_seconds = t1 - t0

    # Solve
    solver_kwargs: Dict = {"verbose": verbose}
    if solver == "MOSEK":
        # Use MOSEK with default tolerances (interior-point)
        try:
            problem.solve(solver=cp.MOSEK, **solver_kwargs)
        except Exception as e:
            return {
                "k": k,
                "N_leg": N_leg,
                "status": f"error: {type(e).__name__}: {e}",
                "sdp_value_rescaled": None,
                "lb_C1a": None,
                "wall_time_sec": time.time() - t0,
                "build_info": info,
            }
    else:
        problem.solve(solver=solver, **solver_kwargs)

    t2 = time.time()
    solve_seconds = t2 - t1

    sdp_value = problem.value
    lb_C1a = 4.0 * sdp_value if sdp_value is not None else None

    return {
        "k": k,
        "N_leg": N_leg,
        "status": problem.status,
        "sdp_value_rescaled": float(sdp_value) if sdp_value is not None else None,
        "lb_C1a": float(lb_C1a) if lb_C1a is not None else None,
        "wall_time_sec": float(t2 - t0),
        "build_seconds": float(build_seconds),
        "solve_seconds": float(solve_seconds),
        "build_info": {
            "k": info.k,
            "N_leg": info.N,
            "n_orbits": info.n_orbits_y,
            "block_sizes": info.block_sizes,
            "n_constraints": info.n_constraints,
        },
    }
