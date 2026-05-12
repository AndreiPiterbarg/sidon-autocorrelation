"""Full design of the 3-point Lasserre relaxation for the Sidon constant C_{1a}.

================================================================================
MATHEMATICAL FORMULATION
================================================================================

Goal: lower bound C_{1a} = inf_f ||f * f||_infty over f >= 0, supp f subset
[-1/4, 1/4], int f = 1.

Two-measure min-max (delsarte_dual/sdp_hierarchy_design.md):
    C_{1a} >= inf_mu sup_nu  (1/(N+1)) * J_N(mu, nu)
where  J_N(mu, nu) = int K_N(x+y-t) d(mu otimes mu)(x,y) d nu(t),
       K_N(s) = sum_{j=0}^{N} hat P_j(s)^2,  hat P_j orthonormal on [-1/2, 1/2].

Equivalent form:  J_N(mu, nu)/(N+1) = sum_j alpha_j(g) beta_j(n) / (N+1)
where alpha_j(g) = int hat P_j(u_1+u_2) F^{(2)}(u_1, u_2) du_1 du_2  (linear in 2D moments g_{ab})
      beta_j(n)  = int hat P_j(t) d nu(t)                            (linear in nu moments)

Dualizing the inner sup_nu via Putinar gives a SINGLE SDP:
    min lambda
    s.t.  lambda - tilde Q(t) = sigma_0(t) + (1/4 - t^2) sigma_1(t)
          sigma_0, sigma_1 SOS    [equiv:  X_0, X_1 PSD Gram matrices]
          mu-moment cone constraints
where tilde Q(t) = (1/(N+1)) * sum_j alpha_j(g) hat P_j(t).

================================================================================
3-POINT LIFT
================================================================================

In coordinates (u_1, u_2, u_3) all in [-1/4, 1/4], the 3D Lasserre block at
level k has variables y_{abc} for a+b+c <= 2k, with:
  - 3D moment matrix M_k(y) PSD,  basis {(a,b,c): a+b+c <= k}, size C(k+3, 3).
  - Localizers (1/16 - u_i^2) M_{k-1}(y) PSD for i = 1, 2, 3.
  - Exchangeability (3 marginals equal): y_{a b c} invariant under S_3 perms.
  - Reflection (f(-x) = f(x), WLOG by rearrangement): y_{abc} = 0 if a+b+c odd.

The 2D moment matrix on g and 1D on m are PRINCIPAL SUBMATRICES of M_k(y), so
no separate constraints needed: 2D PSD/loc and 1D PSD/loc are implied.

The objective tilde Q only uses 2D info g_{ab} = y_{a b 0}.  The 3-point block
enters only via the 3D PSD + localizer constraints, which strictly tighten the
relaxation cone over the 2-point baseline (which has only 2D PSD + localizers).

================================================================================
2-POINT BASELINE
================================================================================

Drop y_{abc}. Variables are g_{ab} (2D moments) for a+b <= 2k, plus
m_a (1D moments) for a <= 2k.  Constraints:
  - g_{ab} = g_{ba}  (S_2 exchangeability)
  - g_{ab} = 0 if a+b odd  (reflection)
  - g_{a0} = m_a  (marginal)
  - 2D moment matrix M_k(g) PSD,  basis size C(k+2, 2).
  - 2 2D localizers PSD, basis size C(k+1, 2).
  - 1D moment matrix M_k(m) PSD, size k+1.
  - 1D localizer (1/16 - u^2) M_{k-1}(m) PSD, size k.
  - m_0 = 1, g_{00} = 1.
Same SOS Gram for nu side, same polynomial identity, same lambda objective.

================================================================================
SYMMETRY REDUCTION
================================================================================

S_3 x Z/2 has order 12.  By Z/2 reflection, only even-total-degree variables
y_{abc} are nonzero.  By S_3, we represent each orbit by its sorted-decreasing
canonical form y_{a' b' c'} with a' >= b' >= c'.

For 2-point baseline: only S_2 x Z/2 (order 4) — canonical form g_{a' b'} with
a' >= b', and g_{ab} = 0 for a+b odd.

The PSD MATRIX size is unchanged by orbit reduction at the variable level —
full block-diagonalization via irrep projection would also reduce matrix size,
but adds significant complexity. We use orbit-level deduplication only here;
this still saves ~6-10x on linear-constraint count and CVXPY compile time.

================================================================================
SOLVER NOTES
================================================================================

CVXPY+MOSEK at small k (<=6).  At k=7+ on monomial basis, MOSEK's interior
point hits conditioning issues because moments span ~5 orders of magnitude
(u^k <= (1/4)^k = 4^{-k}).  We work in monomial basis here and accept the
k=6-7 ceiling on this hardware.  A shifted-Chebyshev rebuild would extend
the ceiling to k>=10 but is left for follow-up.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np


# =====================================================================
# Legendre orthonormal polynomials on [-1/2, 1/2]
# =====================================================================

def legendre_orthonormal_coeffs(N: int) -> np.ndarray:
    """alpha[j, r] = coefficient of t^r in p_j(t) where p_j is L^2-orthonormal
    on [-1/2, 1/2] with respect to Lebesgue measure.

    p_j(t) = sqrt(2j+1) * P_j(2t),  where P_j(x) is standard Legendre on [-1, 1]
    (orthonormality: int_{-1}^1 P_j P_k dx = 2/(2j+1) delta_{jk}).

    Verified: int_{-1/2}^{1/2} p_j(t)^2 dt = (2j+1) * int_{-1/2}^{1/2} P_j(2t)^2 dt
              = (2j+1) * (1/2) int_{-1}^{1} P_j(x)^2 dx
              = (2j+1) * (1/2) * 2/(2j+1) = 1.  Good.
    """
    if N < 0:
        return np.zeros((0, 0))
    # Standard Legendre on [-1, 1]: (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}.
    P = [[Fraction(0)] * (N + 1) for _ in range(N + 1)]
    P[0][0] = Fraction(1)
    if N >= 1:
        P[1][1] = Fraction(1)
    for n in range(1, N):
        for i in range(N + 1):
            term1 = P[n][i - 1] if i >= 1 else Fraction(0)
            P[n + 1][i] = (
                Fraction(2 * n + 1, n + 1) * term1
                - Fraction(n, n + 1) * P[n - 1][i]
            )
    # Convert P_j(x) coefficients to p_j(t) = sqrt(2j+1) * P_j(2t).
    # Each x^i becomes 2^i * t^i.
    alpha = np.zeros((N + 1, N + 1), dtype=np.float64)
    for j in range(N + 1):
        scale = math.sqrt(2 * j + 1)
        for r in range(j + 1):
            coef_frac = P[j][r] * Fraction(1 << r)  # P[j][r] * 2^r
            alpha[j, r] = scale * float(coef_frac)
    return alpha


# =====================================================================
# Multi-index utilities
# =====================================================================

def enum_multi_indices(d: int, max_deg: int) -> List[Tuple[int, ...]]:
    """All alpha in N^d with |alpha| <= max_deg, ordered by total degree then lex."""
    if d == 0:
        return [tuple()]
    if d == 1:
        return [(a,) for a in range(max_deg + 1)]
    out: List[Tuple[int, ...]] = []
    for a in range(max_deg + 1):
        for tail in enum_multi_indices(d - 1, max_deg - a):
            out.append((a,) + tail)
    return out


def s3_canonical(idx: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Canonical S_3 representative: sort decreasing."""
    return tuple(sorted(idx, reverse=True))


def s2_canonical(idx: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(sorted(idx, reverse=True))


# =====================================================================
# Variable creation with orbit deduplication
# =====================================================================

class MomentVarMap:
    """Maps multi-indices to CVXPY variables, deduplicated by symmetry orbit.

    For 3D (S_3 x Z_2): canonical form is sort-decreasing, and odd-total-degree
    indices map to literal scalar 0 (reflection-zero).
    For 2D (S_2 x Z_2): same with sort-decreasing in 2 entries.
    For 1D (Z_2): canonical is identity, odd-degree indices map to 0.
    """
    def __init__(self, dim: int, max_deg: int, *, reflection_zero_odd: bool = True,
                 enforce_normalization: bool = True):
        self.dim = dim
        self.max_deg = max_deg
        self.reflection_zero_odd = reflection_zero_odd
        self._var_by_canon: Dict[Tuple[int, ...], cp.Expression] = {}
        # Enumerate all multi-indices and map them to canonical orbit variables.
        for alpha in enum_multi_indices(dim, max_deg):
            if reflection_zero_odd and (sum(alpha) % 2 == 1):
                # Will be looked up as zero literal.
                continue
            canon = self._canonical(alpha)
            if canon not in self._var_by_canon:
                # Create a fresh CVXPY scalar variable for this orbit.
                self._var_by_canon[canon] = cp.Variable(name=f"y{canon}")
        # Constants
        zero_idx = tuple([0] * dim)
        if enforce_normalization and zero_idx in self._var_by_canon:
            # Will impose y_{0...0} = 1 as an equality constraint via getter.
            pass

    def _canonical(self, alpha: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.dim == 3:
            return s3_canonical(alpha)
        elif self.dim == 2:
            return s2_canonical(alpha)
        elif self.dim == 1:
            return alpha
        else:
            raise ValueError(f"Unsupported dim {self.dim}")

    def get(self, alpha: Tuple[int, ...]) -> cp.Expression:
        """Get the CVXPY expression for y_alpha (constant 0 for reflection-zero)."""
        if len(alpha) != self.dim:
            raise ValueError(f"alpha {alpha} has wrong dimension (expected {self.dim})")
        if any(a < 0 for a in alpha):
            raise ValueError(f"alpha {alpha} has negative components")
        if sum(alpha) > self.max_deg:
            raise ValueError(f"alpha {alpha} exceeds max_deg {self.max_deg}")
        if self.reflection_zero_odd and (sum(alpha) % 2 == 1):
            return cp.Constant(0.0)
        canon = self._canonical(alpha)
        return self._var_by_canon[canon]

    def free_variables(self) -> List[cp.Variable]:
        return list(self._var_by_canon.values())

    def n_orbits(self) -> int:
        return len(self._var_by_canon)


# =====================================================================
# Moment-matrix builders (CVXPY)
# =====================================================================

def moment_matrix(var_map: MomentVarMap, k: int) -> cp.Expression:
    """M_k[I, J] = y_{I + J} where I, J range over multi-indices of total degree <= k."""
    dim = var_map.dim
    basis = enum_multi_indices(dim, k)
    n = len(basis)
    rows: List[List[cp.Expression]] = []
    for i in range(n):
        row: List[cp.Expression] = []
        for j in range(n):
            alpha = tuple(basis[i][t] + basis[j][t] for t in range(dim))
            row.append(var_map.get(alpha))
        rows.append(row)
    return cp.bmat(rows)


def localizer_matrix(var_map: MomentVarMap, k: int, axis: int,
                      support_half: float) -> cp.Expression:
    """((support_half^2) - x_axis^2) * M_{k-1}(y), entrywise:
       L[I, J] = (support_half^2) * y_{I+J} - y_{I+J + 2 e_axis}.
    """
    dim = var_map.dim
    basis = enum_multi_indices(dim, k - 1)
    h2 = support_half * support_half
    n = len(basis)
    rows: List[List[cp.Expression]] = []
    for i in range(n):
        row: List[cp.Expression] = []
        for j in range(n):
            alpha = tuple(basis[i][t] + basis[j][t] for t in range(dim))
            shifted = list(alpha)
            shifted[axis] += 2
            row.append(h2 * var_map.get(alpha) - var_map.get(tuple(shifted)))
        rows.append(row)
    return cp.bmat(rows)


# =====================================================================
# 2-point baseline objective coupling
# =====================================================================

def alpha_j_of_g_coeffs(N: int) -> Dict[Tuple[int, int, int], float]:
    """For each (j, a, b) with a + b <= j <= N, the coefficient of g_{ab}
    in alpha_j(g) = int p_j(u_1 + u_2) f(u_1) f(u_2) du_1 du_2:
        alpha_j(g) = sum_{a, b: a+b <= j} A[j, a, b] * g_{ab}
    where A[j, a, b] = alpha[j, a+b] * C(a+b, a)  (alpha = legendre coeffs).

    Returned as a dict {(j, a, b): A[j, a, b]}.
    """
    leg = legendre_orthonormal_coeffs(N)
    out: Dict[Tuple[int, int, int], float] = {}
    for j in range(N + 1):
        for r in range(j + 1):
            base = leg[j, r]
            if base == 0.0:
                continue
            for a in range(r + 1):
                b = r - a
                out[(j, a, b)] = base * math.comb(r, a)
    return out


# =====================================================================
# SOS Gram + polynomial identity (nu-side dualization)
# =====================================================================

@dataclass
class NuSideDualization:
    """Encapsulates the dualization of sup_nu J_N / (N+1) into:

        sigma_0(t) + (1/4 - t^2) sigma_1(t)  with X_0, X_1 PSD Gram matrices.

    The polynomial identity, for r = 0, ..., 2 K_nu:
        lambda * 1[r=0] - [t^r] tilde Q  =  sum_{i+j=r} (X_0)_{ij}
                                          + (1/4) sum_{i+j=r} (X_1)_{ij}
                                          - sum_{i+j=r-2} (X_1)_{ij}
    where K_nu = ceil(N/2) (so 2 K_nu >= N).

    Note (1/4 - t^2) localizer corresponds to nu support [-1/2, 1/2].

    [t^r] tilde Q is a linear combination of g-moments via alpha_j_of_g_coeffs().
    """
    N: int
    K_nu: int
    lam: cp.Variable
    X0: cp.Variable
    X1: cp.Variable
    leg: np.ndarray  # shape (N+1, N+1)

    @classmethod
    def build(cls, N: int) -> "NuSideDualization":
        K_nu = (N + 1) // 2  # ceil(N/2)
        lam = cp.Variable(name="lambda")
        X0 = cp.Variable((K_nu + 1, K_nu + 1), symmetric=True, name="X0")
        X1 = cp.Variable((K_nu, K_nu), symmetric=True, name="X1") if K_nu >= 1 \
             else cp.Variable((1, 1), symmetric=True, name="X1_dummy")
        leg = legendre_orthonormal_coeffs(N)
        return cls(N=N, K_nu=K_nu, lam=lam, X0=X0, X1=X1, leg=leg)

    def constraints(self) -> List[cp.Constraint]:
        out = [self.X0 >> 0]
        if self.K_nu >= 1:
            out.append(self.X1 >> 0)
        return out

    def coeff_X0_at(self, r: int) -> cp.Expression:
        """sum_{i+j=r} (X_0)_{i,j} for 0 <= r <= 2 K_nu."""
        size = self.K_nu + 1
        terms = []
        for i in range(size):
            j = r - i
            if 0 <= j < size:
                terms.append(self.X0[i, j])
        if not terms:
            return cp.Constant(0.0)
        return cp.sum(cp.hstack(terms))

    def coeff_X1_at(self, r: int) -> cp.Expression:
        """sum_{i+j=r} (X_1)_{i,j} for 0 <= r <= 2(K_nu - 1)."""
        if self.K_nu <= 0:
            return cp.Constant(0.0)
        size = self.K_nu
        terms = []
        for i in range(size):
            j = r - i
            if 0 <= j < size:
                terms.append(self.X1[i, j])
        if not terms:
            return cp.Constant(0.0)
        return cp.sum(cp.hstack(terms))

    def rhs_at(self, r: int) -> cp.Expression:
        """RHS = sigma_0[r] + (1/4) sigma_1[r] - sigma_1[r-2].
        I.e.,  X0 sum at r  +  (1/4) X1 sum at r  -  X1 sum at r-2.
        """
        out = self.coeff_X0_at(r)
        if self.K_nu >= 1:
            out = out + 0.25 * self.coeff_X1_at(r)
            if r >= 2:
                out = out - self.coeff_X1_at(r - 2)
        return out

    def tildeQ_coeff_at(self, r: int, g_lookup, max_g_deg: int,
                          *, mu_scale: float = 1.0) -> cp.Expression:
        """[t^r] tilde Q(t) = sum_{j>=r}^{N} alpha_{j,r} * alpha_j(g)
            where alpha_j(g) = int p_j(u_1+u_2) f(u_1) f(u_2) du_1 du_2.

        With mu RESCALED to [-mu_scale, mu_scale] (so original f on [-mu_scale/4 * 4, mu_scale/4 * 4]
        gets mapped via v = u / mu_scale * SUPPORT_ORIG, etc.), the integration variable substitution
        gives:
            alpha_j(g) = sum_r alpha_{j,r} * mu_scale^r * sum_{a+b=r} C(r,a) g_{ab}
        where g_{ab} are moments in the RESCALED variable.

        For our problem: original f on [-1/4, 1/4], we work with rescaled moments
        on v in [-1, 1], so v = 4 u, and the relation is alpha_j(g_orig) =
        alpha_j(g_rescaled) under (1/4)^r factor on legendre coefficients.
        Set mu_scale = 1/4 to recover original-coord behavior.

        Interpretation: sum_j alpha_j(g) p_j(t) = (f*f)_N(t), the projection of
        f*f onto polynomials of degree <= N in Lebesgue-Legendre basis on
        [-1/2, 1/2].  Then  lambda - tilde Q(t) >= 0  on [-1/2, 1/2]  gives
        lambda >= sup_t (f*f)_N(t).
        """
        D_row = np.zeros(max_g_deg + 1)
        for s in range(max_g_deg + 1):
            jmin = max(r, s)
            for j in range(jmin, self.N + 1):
                D_row[s] += self.leg[j, r] * self.leg[j, s]
        terms = []
        for a in range(max_g_deg + 1):
            for b in range(max_g_deg + 1 - a):
                s = a + b
                if D_row[s] == 0.0:
                    continue
                coef = D_row[s] * math.comb(s, a) * (mu_scale ** s)
                terms.append(coef * g_lookup(a, b))
        if not terms:
            return cp.Constant(0.0)
        return cp.sum(cp.hstack(terms))

    def polynomial_identity_constraints(self, g_lookup, max_g_deg: int,
                                          *, mu_scale: float = 1.0) -> List[cp.Constraint]:
        """Add linear equalities encoding lambda*1[r=0] - [t^r] tilde Q = RHS_r.
        mu_scale: support half-width of the mu-side variables in their working coords.
        Pass mu_scale=1/4 if working in original coords; mu_scale=1 if rescaled to [-1, 1].
        """
        constraints = []
        for r in range(2 * self.K_nu + 1):
            lhs = (self.lam if r == 0 else cp.Constant(0.0)) - self.tildeQ_coeff_at(
                r, g_lookup, max_g_deg, mu_scale=mu_scale)
            rhs = self.rhs_at(r)
            constraints.append(lhs == rhs)
        return constraints


# =====================================================================
# Top-level builders
# =====================================================================

@dataclass
class BuildInfo:
    k: int
    N: int
    with_3pt: bool
    n_orbits_y: int = 0
    n_orbits_g: int = 0
    n_orbits_m: int = 0
    block_sizes: List[int] = field(default_factory=list)
    n_constraints: int = 0
    build_seconds: float = 0.0
    solve_seconds: float = 0.0
    peak_mem_mb: float = 0.0
    status: str = ""
    objective: Optional[float] = None
    solver_info: Dict = field(default_factory=dict)


def build_2pt_full(k: int, N: int) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """2-point baseline: 2D moments g_{ab}, 1D moments m_a, with Christoffel-Darboux
    nu-side SOS dualization.  No 3D block.
    """
    if 2 * k < N:
        raise ValueError(f"Need 2k >= N (have k={k}, N={N}) so g_{{ab}} for a+b<=N exists.")
    # Work with mu RESCALED to [-1, 1] to keep moments O(1) for MOSEK conditioning.
    # Original f on [-1/4, 1/4]; rescaled tilde f on [-1, 1] via v = 4u.
    # Localizers: 1 - v_i^2 >= 0.  In the polynomial identity, mu_scale = 1/4
    # encodes the conversion (since for original poly p(u_1+u_2), substituting
    # u_i = v_i/4 gives sum_r alpha_r (1/4)^r * sum (v_1+v_2)^r ).
    SUPPORT_HALF = 1.0
    MU_SCALE = 0.25
    t0 = time.time()

    g_map = MomentVarMap(dim=2, max_deg=2 * k)
    m_map = MomentVarMap(dim=1, max_deg=2 * k)
    nu = NuSideDualization.build(N)

    constraints: List[cp.Constraint] = []
    constraints.append(g_map.get((0, 0)) == 1)
    constraints.append(m_map.get((0,)) == 1)
    # Marginal: g_{a,0} = m_a
    for a in range(2 * k + 1):
        constraints.append(g_map.get((a, 0)) == m_map.get((a,)))

    # 2D PSD blocks
    block_sizes: List[int] = []
    M2 = moment_matrix(g_map, k)
    constraints.append(M2 >> 0)
    block_sizes.append(M2.shape[0])
    if k >= 1:
        for axis in range(2):
            L = localizer_matrix(g_map, k, axis=axis, support_half=SUPPORT_HALF)
            constraints.append(L >> 0)
            block_sizes.append(L.shape[0])

    # 1D PSD blocks
    M1 = moment_matrix(m_map, k)
    constraints.append(M1 >> 0)
    block_sizes.append(M1.shape[0])
    if k >= 1:
        L1 = localizer_matrix(m_map, k, axis=0, support_half=SUPPORT_HALF)
        constraints.append(L1 >> 0)
        block_sizes.append(L1.shape[0])

    # Nu-side SOS Gram
    constraints.extend(nu.constraints())
    block_sizes.append(nu.K_nu + 1)
    if nu.K_nu >= 1:
        block_sizes.append(nu.K_nu)

    # Polynomial identity matching: g-moment-side via g_map.get((a, b)).
    def g_lookup(a: int, b: int) -> cp.Expression:
        if a + b > 2 * k:
            return cp.Constant(0.0)
        return g_map.get((a, b))
    max_g_deg = min(2 * k, N)  # we never need a+b > N for the identity
    constraints.extend(nu.polynomial_identity_constraints(g_lookup, max_g_deg, mu_scale=MU_SCALE))

    objective = cp.Minimize(nu.lam)
    problem = cp.Problem(objective, constraints)

    info = BuildInfo(
        k=k, N=N, with_3pt=False,
        n_orbits_y=0,
        n_orbits_g=g_map.n_orbits(),
        n_orbits_m=m_map.n_orbits(),
        block_sizes=block_sizes,
        n_constraints=len(constraints),
        build_seconds=time.time() - t0,
    )
    handles = dict(g=g_map, m=m_map, nu=nu)
    return problem, info, handles


def build_3pt_full(k: int, N: int) -> Tuple[cp.Problem, BuildInfo, Dict]:
    """3-point lift: 3D moments y_{abc} in (u_1, u_2, u_3) coords, with S_3 x Z/2
    orbit reduction, 3D PSD + 3 localizers, and the same Christoffel-Darboux
    nu-side SOS dualization.

    g_{ab} := y_{a b 0}, m_a := y_{a 0 0}.  Therefore the 1D and 2D moment
    matrices are PRINCIPAL SUBMATRICES of M_k(y), and their PSD/localizer
    constraints are IMPLIED.  We only add the 3D constraints + polynomial
    identity here.
    """
    if 2 * k < N:
        raise ValueError(f"Need 2k >= N (have k={k}, N={N}) so y_{{a,b,0}} for a+b<=N exists.")
    SUPPORT_HALF = 1.0
    MU_SCALE = 0.25
    t0 = time.time()

    y_map = MomentVarMap(dim=3, max_deg=2 * k)
    nu = NuSideDualization.build(N)

    constraints: List[cp.Constraint] = []
    constraints.append(y_map.get((0, 0, 0)) == 1)

    # 3D PSD + 3 localizers
    block_sizes: List[int] = []
    M3 = moment_matrix(y_map, k)
    constraints.append(M3 >> 0)
    block_sizes.append(M3.shape[0])
    if k >= 1:
        for axis in range(3):
            L = localizer_matrix(y_map, k, axis=axis, support_half=SUPPORT_HALF)
            constraints.append(L >> 0)
            block_sizes.append(L.shape[0])

    # Nu-side SOS Gram
    constraints.extend(nu.constraints())
    block_sizes.append(nu.K_nu + 1)
    if nu.K_nu >= 1:
        block_sizes.append(nu.K_nu)

    # Polynomial identity: g_{ab} = y_{a, b, 0}
    def g_lookup(a: int, b: int) -> cp.Expression:
        if a + b > 2 * k:
            return cp.Constant(0.0)
        return y_map.get((a, b, 0))
    max_g_deg = min(2 * k, N)
    constraints.extend(nu.polynomial_identity_constraints(g_lookup, max_g_deg, mu_scale=MU_SCALE))

    objective = cp.Minimize(nu.lam)
    problem = cp.Problem(objective, constraints)

    info = BuildInfo(
        k=k, N=N, with_3pt=True,
        n_orbits_y=y_map.n_orbits(),
        n_orbits_g=0, n_orbits_m=0,
        block_sizes=block_sizes,
        n_constraints=len(constraints),
        build_seconds=time.time() - t0,
    )
    handles = dict(y=y_map, nu=nu)
    return problem, info, handles


# =====================================================================
# Solver wrapper
# =====================================================================

def solve(problem: cp.Problem, info: BuildInfo, *, solver: str = "MOSEK",
          verbose: bool = False, mosek_params: Optional[Dict] = None) -> BuildInfo:
    """Solve the SDP and populate solve_seconds, peak_mem_mb, status, objective."""
    import psutil
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t0 = time.time()
    try:
        if solver == "MOSEK" and mosek_params:
            problem.solve(solver=solver, verbose=verbose, mosek_params=mosek_params)
        else:
            problem.solve(solver=solver, verbose=verbose)
    except Exception as exc:
        info.solve_seconds = time.time() - t0
        info.status = f"ERROR: {type(exc).__name__}: {exc}"
        info.peak_mem_mb = (proc.memory_info().rss - rss_before) / 1e6
        return info
    info.solve_seconds = time.time() - t0
    info.status = problem.status
    info.objective = float(problem.value) if problem.value is not None else None
    info.peak_mem_mb = (proc.memory_info().rss - rss_before) / 1e6
    return info


__all__ = [
    "legendre_orthonormal_coeffs",
    "MomentVarMap", "moment_matrix", "localizer_matrix",
    "NuSideDualization",
    "build_2pt_full", "build_3pt_full", "solve",
    "BuildInfo",
]
