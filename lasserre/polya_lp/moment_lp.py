"""Moment-augmented LP (Approach A.1, proof-of-concept).

Adds first moments nu_i := int_{B_i} (t - x_i) f(t) dt as auxiliary LP
variables alongside bin masses mu_i := int_{B_i} f. Allows tighter
Fourier cuts that are RIGOROUSLY VALID for ANY admissible nonneg f
(not just step functions).

VARIABLES (polynomial indeterminates): mu_0, ..., mu_{d-1}, nu_0, ..., nu_{d-1}.

FEASIBLE SET S:
  mu_i >= 0,
  sum_i mu_i = 1,
  delta * mu_i + nu_i >= 0,
  delta * mu_i - nu_i >= 0,
  where delta = 1/(4d) is the bin half-width.

CUT FORMULA (per agent derivation 2026-05-04):
For each kernel K(t) >= 0 on [-1/2, 1/2] with int K > 0, the matrix triple
(A, B, C) gives the cut

  Q_K(mu, nu) := sum_{ij} A_{ij} mu_i mu_j
              + 2 sum_{ij} B_{ij} nu_i mu_j
              + sum_{ij} C_{ij} nu_i nu_j
            <=  phys_sup(f) * int K       for any admissible f.

Where, with c_{ij} = x_i + x_j and M_2^{(ij)} = min K''(eta) over
eta in [c_{ij} - 2 delta, c_{ij} + 2 delta]:

  A_{ij} = K(c_{ij}) + 1[M_2 < 0] * M_2 * delta^2
  B_{ij} = K'(c_{ij})
  C_{ij} = 1[M_2 < 0] * M_2

(Derivation: Lagrange remainder of Taylor expansion of K(s+u) around
c_{ij}, with f >= 0 substituting min K'' for K''(eta_{s,u}). The
upper-bracket E_ij <= 2 delta^2 mu_i mu_j + 2 nu_i nu_j on the
second-moment quantity gives the rigorous lower bound on R_ij.)

LP CERT (Handelman positivstellensatz):
  Q_lambda(mu, nu) - alpha
    = sum_a c_a * m_a(mu, nu)
    + q(mu, nu) * (sum_i mu_i - 1)
  where m_a runs over generator monomials prod_i mu_i^{a_mu_i}
  (delta mu_i + nu_i)^{a_plus_i} (delta mu_i - nu_i)^{a_minus_i} with
  total degree |a| <= R, c_a >= 0, and q is a free polynomial of
  degree <= R - 1 in (mu, nu).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple, Optional, Dict
import time
from math import comb

import numpy as np
from scipy import sparse as sp


# =====================================================================
# Helpers
# =====================================================================

def bin_centers(d: int) -> np.ndarray:
    return -1.0 / 4.0 + (np.arange(d) + 0.5) / (2.0 * d)


def enum_monomials_le(n: int, R: int) -> List[Tuple[int, ...]]:
    """All multi-indices alpha in N^n with |alpha| <= R."""
    if n == 0:
        return [()]
    out: List[Tuple[int, ...]] = []

    def rec(pos: int, remaining: int, cur: List[int]):
        if pos == n - 1:
            for v in range(remaining + 1):
                cur[pos] = v
                out.append(tuple(cur))
            cur[pos] = 0
            return
        for v in range(remaining + 1):
            cur[pos] = v
            rec(pos + 1, remaining - v, cur)
        cur[pos] = 0

    rec(0, R, [0] * n)
    return out


# =====================================================================
# Generator-monomial expansion in (mu, nu) basis
# =====================================================================

def expand_generator_monomial(
    a_mu: Tuple[int, ...],
    a_plus: Tuple[int, ...],
    a_minus: Tuple[int, ...],
    delta: float,
) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]:
    """Expand m_a := prod_i mu_i^{a_mu_i} (delta mu_i + nu_i)^{a_plus_i}
                    (delta mu_i - nu_i)^{a_minus_i}
    in the (mu, nu) monomial basis.

    Returns dict mapping (beta_mu, beta_nu) tuples -> coefficient.
    """
    d = len(a_mu)

    # Per-bin polynomial: P_i(mu_i, nu_i) = mu_i^{am} (delta mu_i + nu_i)^{ap}
    # (delta mu_i - nu_i)^{an}.
    # Expand each (delta mu_i + nu_i)^{ap} = sum_{k=0}^{ap} C(ap, k) (delta mu_i)^{ap-k} nu_i^k
    # and similarly for the minus version.
    # Final exponent on mu_i: am + (ap - kp) + (an - km). Final on nu_i: kp + km.
    # Coefficient: C(ap, kp) * C(an, km) * delta^{ap - kp + an - km} * (-1)^km.

    bin_polys: List[Dict[Tuple[int, int], float]] = []
    for i in range(d):
        am, ap, an = a_mu[i], a_plus[i], a_minus[i]
        d_poly: Dict[Tuple[int, int], float] = {}
        for kp in range(ap + 1):
            for km in range(an + 1):
                mu_pow = am + (ap - kp) + (an - km)
                nu_pow = kp + km
                coef = (comb(ap, kp) * comb(an, km)
                        * (delta ** (ap - kp + an - km))
                        * ((-1) ** km))
                key = (mu_pow, nu_pow)
                d_poly[key] = d_poly.get(key, 0.0) + coef
        bin_polys.append(d_poly)

    # Combine via Cartesian product
    result: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}

    def rec(i: int, mu_exp: List[int], nu_exp: List[int], coef: float):
        if i == d:
            key = (tuple(mu_exp), tuple(nu_exp))
            result[key] = result.get(key, 0.0) + coef
            return
        for (mp, np_), c in bin_polys[i].items():
            mu_exp[i] = mp
            nu_exp[i] = np_
            rec(i + 1, mu_exp, nu_exp, coef * c)
        mu_exp[i] = 0
        nu_exp[i] = 0

    rec(0, [0] * d, [0] * d, 1.0)
    return result


# =====================================================================
# Kernel cut matrices (A, B, C)
# =====================================================================

@dataclass
class KernelCut:
    label: str
    A: np.ndarray   # mu mu block: A[i, j] for sum A_ij mu_i mu_j
    B: np.ndarray   # mu nu cross: 2 sum B_ij nu_i mu_j
    C: np.ndarray   # nu nu block: sum C_ij nu_i nu_j
    integral_K: float


def krein_poisson_K_Kp_Kpp(s: float, t0: float = 0.0):
    """Return (K, K', K'') as vectorised callables for Krein-Poisson kernel."""
    one_minus_s2 = 1.0 - s * s
    one_plus_s2 = 1.0 + s * s
    twopi = 2.0 * np.pi

    def D(t):
        return one_plus_s2 - 2.0 * s * np.cos(twopi * (t - t0))

    def Dp(t):
        return 4.0 * np.pi * s * np.sin(twopi * (t - t0))

    def Dpp(t):
        return 8.0 * np.pi**2 * s * np.cos(twopi * (t - t0))

    def K(t):
        return one_minus_s2 / D(t)

    def Kp(t):
        d_ = D(t)
        return -one_minus_s2 * Dp(t) / (d_ * d_)

    def Kpp(t):
        d_ = D(t)
        return one_minus_s2 * (2.0 * Dp(t) ** 2 / d_ ** 3 - Dpp(t) / d_ ** 2)

    return K, Kp, Kpp


def kernel_cut_from_callables(
    d: int,
    K: Callable, Kp: Callable, Kpp: Callable,
    integral_K: float,
    n_grid_M2: int = 65,
    label: str = "kernel",
) -> KernelCut:
    """Compute (A, B, C) matrices for a kernel given (K, K', K'')."""
    delta = 1.0 / (4.0 * d)
    x = bin_centers(d)
    centers = x[:, None] + x[None, :]   # (d, d)

    K_vals = K(centers) / integral_K
    Kp_vals = Kp(centers) / integral_K
    A = np.array(K_vals, copy=True)
    B = np.array(Kp_vals, copy=True)
    C = np.zeros_like(A)

    # Compute M_2^{(ij)} via fine grid sampling on [-2 delta, 2 delta]
    grid = np.linspace(-2.0 * delta, 2.0 * delta, n_grid_M2)
    # Vectorise: for each (i, j), sample K'' at centers[i,j] + grid.
    eval_pts = centers[:, :, None] + grid[None, None, :]
    Kpp_vals = Kpp(eval_pts) / integral_K
    M2 = Kpp_vals.min(axis=-1)   # (d, d)

    neg_mask = M2 < 0
    A[neg_mask] += M2[neg_mask] * (delta ** 2)
    C[neg_mask] = M2[neg_mask]

    return KernelCut(label=label, A=A, B=B, C=C, integral_K=1.0)
    # (We've already divided by integral_K; the cut is now Q <= phys_sup.)


def krein_poisson_cut(d: int, s: float, t0: float = 0.0,
                      n_grid_M2: int = 65) -> KernelCut:
    K, Kp, Kpp = krein_poisson_K_Kp_Kpp(s, t0)
    return kernel_cut_from_callables(
        d, K, Kp, Kpp, integral_K=1.0, n_grid_M2=n_grid_M2,
        label=f"krein_poisson(s={s:.3f},t0={t0:+.3f})",
    )


# =====================================================================
# Z/2 projection of cut matrices
# =====================================================================
#
# For symmetric f under Z/2 (f(x) = f(-x)):
#   mu_full[i] = mu_full[d-1-i]  (symmetric)  -> mu_orbit_a, a in 0..d_eff-1
#   nu_full[i] = -nu_full[d-1-i] (antisymmetric) -> nu_orbit_a, a in 0..d_eff-1
# with eps(i) = +1 if i <= d-1-i (i is canonical), else -1.
#
# Projections:
#   A_orbit[a, b] = sum_{i: orb=a, j: orb=b} A_full[i, j]
#   B_orbit[a, b] = sum_{i: orb=a, j: orb=b} B_full[i, j] * eps(j)
#   C_orbit[a, b] = sum_{i: orb=a, j: orb=b} C_full[i, j] * eps(i) * eps(j)
#
# Standard-simplex rescale: substitute mu_orbit = mu' / orbit_size.
#   For d_orig even: orbit_size = 2 for all orbits.
#   A_eff = A_orbit / 4 (D^{-1} A D^{-1}, D = diag(2))
#   B_eff = B_orbit / 2 (only mu side rescaled)
#   C_eff = C_orbit (nu unchanged)
#
# Delta in the reduced LP: delta_eff = delta_orig / 2 = 1/(8 d_orig).
# (Because |nu_orbit| <= delta_orig mu_orbit = delta_orig (mu'/2),
# so the LP constraint in mu' is |nu_orbit| <= (delta_orig/2) mu'.)
# =====================================================================

def z2_orbit_id(d: int) -> np.ndarray:
    """orbit_id[i] = min(i, d-1-i)."""
    return np.minimum(np.arange(d), d - 1 - np.arange(d))


def z2_orbit_sign(d: int) -> np.ndarray:
    """eps(i) = +1 if i <= d-1-i, else -1."""
    return np.where(np.arange(d) <= d - 1 - np.arange(d), 1.0, -1.0)


def project_kernel_cut_to_z2(kc: KernelCut, d_orig: int) -> KernelCut:
    """Project a KernelCut from d_orig to d_eff = ceil(d_orig/2)."""
    orbit = z2_orbit_id(d_orig)
    eps = z2_orbit_sign(d_orig)
    d_eff = int(orbit.max() + 1)

    A_full, B_full, C_full = kc.A, kc.B, kc.C
    assert A_full.shape == (d_orig, d_orig)

    A_orbit = np.zeros((d_eff, d_eff))
    B_orbit = np.zeros((d_eff, d_eff))
    C_orbit = np.zeros((d_eff, d_eff))
    np.add.at(A_orbit, (orbit[:, None], orbit[None, :]), A_full)
    np.add.at(B_orbit, (orbit[:, None], orbit[None, :]),
              B_full * eps[None, :])
    np.add.at(C_orbit, (orbit[:, None], orbit[None, :]),
              C_full * eps[:, None] * eps[None, :])

    # Standard-simplex rescale (assume d_orig even -> all orbit_size = 2)
    if d_orig % 2 != 0:
        raise NotImplementedError("Z/2 projection currently only supports even d.")
    A_eff = A_orbit / 4.0
    B_eff = B_orbit / 2.0
    C_eff = C_orbit

    return KernelCut(
        label=kc.label + "[z2]", A=A_eff, B=B_eff, C=C_eff,
        integral_K=kc.integral_K,
    )


def z2_delta(d_orig: int) -> float:
    """Effective delta for the Z/2-reduced LP at d_orig.

    delta_orig = 1/(4 d_orig). After substitution mu_orbit = mu'/2:
        |nu_orbit| <= delta_orig * (mu' / 2) = (delta_orig / 2) mu'
    So the LP at d_eff sees delta_eff = delta_orig / 2 = 1/(8 d_orig).
    """
    return 1.0 / (8.0 * d_orig)


# =====================================================================
# Moment LP build
# =====================================================================

@dataclass
class MomentLPBuild:
    A_eq: sp.csr_matrix
    b_eq: np.ndarray
    c: np.ndarray
    bounds: List[Tuple[Optional[float], Optional[float]]]
    n_vars: int
    alpha_idx: int
    lambda_idx: slice
    q_idx: slice
    c_idx: slice
    n_lambda: int
    n_q: int
    n_c: int
    monos_mu_nu: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
    q_monos_mu_nu: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
    gen_monos: List[Tuple[int, ...]]
    build_wall_s: float
    n_nonzero_A: int
    d: int
    R: int


def build_moment_lp(
    d: int,
    R: int,
    M_mats: Sequence[np.ndarray],
    kernel_cuts: Sequence[KernelCut],
    delta: Optional[float] = None,
    verbose: bool = False,
) -> MomentLPBuild:
    """Build the (mu, nu)-Handelman LP at degree R."""
    t0 = time.time()
    if delta is None:
        delta = 1.0 / (4.0 * d)

    # Enumerate (mu, nu) monomials of degree <= R
    monos_2d = enum_monomials_le(2 * d, R)
    monos_mu_nu = [(tuple(b[:d]), tuple(b[d:])) for b in monos_2d]
    n_rows = len(monos_mu_nu)
    mono_to_idx = {b: i for i, b in enumerate(monos_mu_nu)}

    # q polynomial: degree <= R-1 in (mu, nu)
    q_monos_2d = enum_monomials_le(2 * d, R - 1) if R >= 1 else [tuple([0] * (2 * d))]
    q_monos_mu_nu = [(tuple(b[:d]), tuple(b[d:])) for b in q_monos_2d]
    n_q = len(q_monos_mu_nu)

    # Generator monomials in (a_mu, a_plus, a_minus) of total degree <= R
    gen_monos = enum_monomials_le(3 * d, R)
    n_c = len(gen_monos)

    n_lambda = len(M_mats) + len(kernel_cuts)
    n_vars = 1 + n_lambda + n_q + n_c

    alpha_idx = 0
    lambda_start = 1
    q_start = 1 + n_lambda
    c_start = q_start + n_q

    if verbose:
        print(f"  Moment LP: n_rows={n_rows}, n_lambda={n_lambda}, "
              f"n_q={n_q}, n_c={n_c}, n_vars={n_vars}", flush=True)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    zero_beta = (tuple([0] * d), tuple([0] * d))

    # 1. alpha contributes -1 in row beta = (0, 0)
    rows.append(mono_to_idx[zero_beta])
    cols.append(alpha_idx)
    vals.append(-1.0)

    # 2. lambda_W: M_W contributes to mu^T M_W mu (no nu).
    for w, M_W in enumerate(M_mats):
        nz = np.where(np.abs(M_W) > 0)
        for i, j in zip(*nz):
            if i > j:
                continue   # use upper triangle
            if i == j:
                val = float(M_W[i, j])
                beta_mu = tuple(2 if k == i else 0 for k in range(d))
            else:
                val = float(M_W[i, j] + M_W[j, i])
                beta_mu = tuple(1 if (k == i or k == j) else 0 for k in range(d))
            beta = (beta_mu, tuple([0] * d))
            rows.append(mono_to_idx[beta])
            cols.append(lambda_start + w)
            vals.append(val)

    # 3. lambda_K: Q_K(mu, nu) = mu^T A mu + 2 mu^T B nu + nu^T C nu
    for k_idx, kc in enumerate(kernel_cuts):
        col = lambda_start + len(M_mats) + k_idx
        A_mat, B_mat, C_mat = kc.A, kc.B, kc.C
        # mu^T A mu
        for i in range(d):
            for j in range(i, d):
                val = (float(A_mat[i, i]) if i == j
                       else float(A_mat[i, j] + A_mat[j, i]))
                if val == 0.0:
                    continue
                if i == j:
                    beta_mu = tuple(2 if k == i else 0 for k in range(d))
                else:
                    beta_mu = tuple(1 if (k == i or k == j) else 0 for k in range(d))
                beta = (beta_mu, tuple([0] * d))
                rows.append(mono_to_idx[beta])
                cols.append(col)
                vals.append(val)
        # 2 mu^T B nu (coefficient of mu_i nu_j is 2 B_{ij})
        for i in range(d):
            for j in range(d):
                val = 2.0 * float(B_mat[i, j])
                if val == 0.0:
                    continue
                beta_mu = tuple(1 if k == i else 0 for k in range(d))
                beta_nu = tuple(1 if k == j else 0 for k in range(d))
                beta = (beta_mu, beta_nu)
                rows.append(mono_to_idx[beta])
                cols.append(col)
                vals.append(val)
        # nu^T C nu
        for i in range(d):
            for j in range(i, d):
                val = (float(C_mat[i, i]) if i == j
                       else float(C_mat[i, j] + C_mat[j, i]))
                if val == 0.0:
                    continue
                if i == j:
                    beta_nu = tuple(2 if k == i else 0 for k in range(d))
                else:
                    beta_nu = tuple(1 if (k == i or k == j) else 0 for k in range(d))
                beta = (tuple([0] * d), beta_nu)
                rows.append(mono_to_idx[beta])
                cols.append(col)
                vals.append(val)

    # 4. q polynomial: -q_K in row K, +q_K in row K + e^mu_i
    for q_idx, q_K in enumerate(q_monos_mu_nu):
        K_mu, K_nu = q_K
        col = q_start + q_idx
        # -q_K in row beta = K
        if q_K in mono_to_idx:
            rows.append(mono_to_idx[q_K])
            cols.append(col)
            vals.append(-1.0)
        # +q_K in row K + e^mu_i
        for i in range(d):
            new_K_mu = list(K_mu)
            new_K_mu[i] += 1
            new_beta = (tuple(new_K_mu), K_nu)
            if new_beta in mono_to_idx:
                rows.append(mono_to_idx[new_beta])
                cols.append(col)
                vals.append(1.0)

    # 5. c_a slacks: -m_a(beta) coefficient
    for a_idx, gen_a in enumerate(gen_monos):
        col = c_start + a_idx
        a_mu = gen_a[:d]
        a_plus = gen_a[d:2 * d]
        a_minus = gen_a[2 * d:3 * d]
        expansion = expand_generator_monomial(a_mu, a_plus, a_minus, delta)
        for beta, coef in expansion.items():
            idx = mono_to_idx.get(beta)
            if idx is not None and coef != 0.0:
                rows.append(idx)
                cols.append(col)
                vals.append(-coef)

    A_eq = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64),
          np.asarray(cols, dtype=np.int64))),
        shape=(n_rows, n_vars),
    )
    b_eq = np.zeros(n_rows)

    # Add simplex constraint sum_W lambda_W = 1
    sim_row = sp.csr_matrix(
        (np.ones(n_lambda),
         (np.zeros(n_lambda, dtype=np.int64),
          np.arange(lambda_start, lambda_start + n_lambda, dtype=np.int64))),
        shape=(1, n_vars),
    )
    A_eq = sp.vstack([A_eq, sim_row], format="csr")
    b_eq = np.concatenate([b_eq, np.array([1.0])])

    # Variable bounds
    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    bounds.append((None, None))
    for _ in range(n_lambda):
        bounds.append((0.0, None))
    for _ in range(n_q):
        bounds.append((None, None))
    for _ in range(n_c):
        bounds.append((0.0, None))

    c_obj = np.zeros(n_vars)
    c_obj[alpha_idx] = -1.0

    return MomentLPBuild(
        A_eq=A_eq, b_eq=b_eq, c=c_obj, bounds=bounds, n_vars=n_vars,
        alpha_idx=alpha_idx,
        lambda_idx=slice(lambda_start, lambda_start + n_lambda),
        q_idx=slice(q_start, q_start + n_q),
        c_idx=slice(c_start, c_start + n_c),
        n_lambda=n_lambda, n_q=n_q, n_c=n_c,
        monos_mu_nu=monos_mu_nu, q_monos_mu_nu=q_monos_mu_nu,
        gen_monos=gen_monos,
        build_wall_s=time.time() - t0,
        n_nonzero_A=A_eq.nnz,
        d=d, R=R,
    )


# =====================================================================
# Solve via highspy
# =====================================================================

@dataclass
class MomentLPSolveResult:
    status: str
    alpha: Optional[float]
    x: Optional[np.ndarray]
    wall_s: float
    raw_status: object = None


def solve_moment_lp(build: MomentLPBuild, verbose: bool = False,
                    solver: str = "mosek") -> MomentLPSolveResult:
    """Solve via MOSEK (default) or HiGHS."""
    if solver == "mosek":
        return _solve_with_mosek(build, verbose)
    return _solve_with_highspy(build, verbose)


def _solve_with_mosek(build: MomentLPBuild, verbose: bool) -> MomentLPSolveResult:
    """MOSEK Optimizer API (low-level). Handles big sparse LPs much better
    than HiGHS for this Handelman moment LP.
    """
    import mosek
    t0 = time.time()
    A_eq = build.A_eq.tocsc()
    n_vars = build.n_vars
    n_eq = A_eq.shape[0]

    with mosek.Env() as env:
        with env.Task() as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log,
                                lambda msg: print(msg, end="", flush=True))
            task.appendvars(n_vars)
            task.appendcons(n_eq)

            # Variable bounds and objective
            for j, (lo, hi) in enumerate(build.bounds):
                if lo is None and hi is None:
                    bk = mosek.boundkey.fr; lb = -0.0; ub = 0.0
                elif lo is None:
                    bk = mosek.boundkey.up; lb = -0.0; ub = float(hi)
                elif hi is None:
                    bk = mosek.boundkey.lo; lb = float(lo); ub = 0.0
                elif lo == hi:
                    bk = mosek.boundkey.fx; lb = ub = float(lo)
                else:
                    bk = mosek.boundkey.ra; lb = float(lo); ub = float(hi)
                task.putvarbound(j, bk, lb, ub)
                task.putcj(j, float(build.c[j]))

            # Equality row bounds (b_eq <= row <= b_eq)
            for i in range(n_eq):
                task.putconbound(i, mosek.boundkey.fx,
                                 float(build.b_eq[i]), float(build.b_eq[i]))

            # Provide A as triplets via putaijlist
            A_coo = A_eq.tocoo()
            task.putaijlist(
                A_coo.row.astype(np.int64).tolist(),
                A_coo.col.astype(np.int64).tolist(),
                A_coo.data.astype(np.float64).tolist(),
            )

            task.putobjsense(mosek.objsense.minimize)
            task.optimize()

            wall = time.time() - t0

            sol_status = task.getsolsta(mosek.soltype.bas)
            if sol_status not in (mosek.solsta.optimal, mosek.solsta.dual_infeas_cer,
                                  mosek.solsta.prim_infeas_cer):
                # Try IPM solution (interior)
                sol_status = task.getsolsta(mosek.soltype.itr)
                soltype = mosek.soltype.itr
            else:
                soltype = mosek.soltype.bas

            if sol_status == mosek.solsta.optimal:
                xx = np.zeros(n_vars)
                task.getxx(soltype, xx)
                obj = task.getprimalobj(soltype)
                alpha = -float(obj)
                return MomentLPSolveResult("OPTIMAL", alpha, xx, wall, sol_status)
            if sol_status == mosek.solsta.prim_infeas_cer:
                return MomentLPSolveResult("INFEASIBLE", None, None, wall, sol_status)
            if sol_status == mosek.solsta.dual_infeas_cer:
                return MomentLPSolveResult("UNBOUNDED", None, None, wall, sol_status)
            return MomentLPSolveResult(f"OTHER({sol_status})", None, None, wall, sol_status)


def _solve_with_highspy(build: MomentLPBuild, verbose: bool) -> MomentLPSolveResult:
    import highspy
    t0 = time.time()
    h = highspy.Highs()
    if not verbose:
        h.silent()
    A_eq = build.A_eq.tocsc()
    inf = highspy.kHighsInf
    n_vars = build.n_vars
    n_eq = A_eq.shape[0]

    col_lo = np.empty(n_vars)
    col_hi = np.empty(n_vars)
    for i, (lo, hi) in enumerate(build.bounds):
        col_lo[i] = -inf if lo is None else lo
        col_hi[i] = inf if hi is None else hi

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_eq
    lp.col_cost_ = build.c.copy()
    lp.col_lower_ = col_lo
    lp.col_upper_ = col_hi
    lp.row_lower_ = build.b_eq.copy()
    lp.row_upper_ = build.b_eq.copy()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A_eq.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A_eq.indices.astype(np.int32)
    lp.a_matrix_.value_ = A_eq.data.astype(np.float64)
    lp.sense_ = highspy.ObjSense.kMinimize

    h.passModel(lp)
    h.run()
    status = h.getModelStatus()
    wall = time.time() - t0

    if status == highspy.HighsModelStatus.kOptimal:
        sol = h.getSolution()
        x = np.asarray(sol.col_value)
        info = h.getInfo()
        alpha = -float(info.objective_function_value)
        return MomentLPSolveResult("OPTIMAL", alpha, x, wall, status)
    if status == highspy.HighsModelStatus.kInfeasible:
        return MomentLPSolveResult("INFEASIBLE", None, None, wall, status)
    if status in (highspy.HighsModelStatus.kUnbounded,
                  highspy.HighsModelStatus.kUnboundedOrInfeasible):
        return MomentLPSolveResult("UNBOUNDED", None, None, wall, status)
    return MomentLPSolveResult(f"OTHER({status})", None, None, wall, status)
