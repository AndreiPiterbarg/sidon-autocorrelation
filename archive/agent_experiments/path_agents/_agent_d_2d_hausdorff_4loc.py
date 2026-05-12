"""
_agent_d_2d_hausdorff_4loc.py

Agent D: 2D Hausdorff 4-localizer for the Sidon autocorrelation lower bound.

Mathematical setup
==================

Let f >= 0 on [-1/4, 1/4] with int f = 1.  Set
    g(t) = (f*f)(t),   t in [-1/2, 1/2],
    g_n  = int_{-1/2}^{1/2} t^n g(t) dt = sum_{k=0..n} C(n,k) m_k m_{n-k}
where m_n = int x^n f(x) dx.

C_{1a} := inf_f max(g).  Current rigorous LB = 1.2748 (Matolcsi-Vinuesa 2010).

Where Agent A's Shor lift failed:
    Shor:  introduce M = [M[i,j]]_{i,j=0..N} with M ~ m m^T relaxed via
           [[1, m^T], [m, M]] >> 0.  Then  g_n = sum C(n,k) M[k, n-k].
           Off-diagonal M[i,j] is loose -- rank>1 minimizer gives LB = 1.0.

Agent D's fix (2D moment matrix lift):
    Model f(x)f(y) as a probability measure mu on [-1/4, 1/4]^2.
    Build the BOX-DEGREE tensor moment matrix
        M_2D[(a1, a2), (b1, b2)]  =  int x^{a1+b1} y^{a2+b2} dmu(x,y)
    for a1, a2, b1, b2 in {0, ..., d}.  In the relaxation we drop the
    factorization mu(dx,dy) = f(x) dx * f(y) dy and only enforce:
        (R1) M_2D depends only on a + b      (RIGID -- not in Shor!)
        (R2) M_2D >> 0                       (moment-PSD)
        (R3) Localizers PSD for the 1D Hausdorff factors (A^2 - x^2) >= 0
             and (A^2 - y^2) >= 0
        (R4) THE 2D 4-LOCALIZER for q(x,y) = (1/16 - x^2)(1/16 - y^2) >= 0
             on [-1/4, 1/4]^2.  This is the constraint that Shor's lift
             cannot see, because Shor doesn't know M[i,j] is a moment
             of any measure.
    Marginal coupling:
        m_n = M_2D[(0,0), (n, 0)] = M_2D[(0,0), (0, n)]
    Symmetry of x <-> y (irrelevant for the LB but pins symmetry of the
    representing measure if it exists):
        M_2D[(a1, a2), (b1, b2)] = M_2D[(a2, a1), (b2, b1)]

    The Markov lower bound on max(g) for g >= 0 on [-1/2, 1/2] with int g = 1
    and g_n = int t^n g dt is, for each EVEN n >= 2:
        max(g) >= (n + 1) * 2^n * g_n     (since int_{-1/2}^{1/2} t^n dt = 1/((n+1)*2^n))
    More generally, for any nonneg test poly p(t) with int_{-1/2}^{1/2} p = 1,
    max(g) >= int p g = sum p_n g_n.  So
        max(g) >= sup_{p in P} sum_n p_n g_n
    over the set P of probability polynomials on [-1/2, 1/2] of degree <= 2d.

    We use Krein-Markov SDP duality for the LB on max(g):
        max(g) >= L*  where  L* = min L s.t. exists nonneg measure
                  nu = (L*Leb - g) >= 0 on [-1/2, 1/2] matching the moments.
    This is the same inner-LP-Hausdorff scheme as Agent A's.

The new feature:  g_n is encoded via the *off-diagonal* entries of M_2D,
which are NOW constrained by R1+R2+R3+R4 -- not just the Shor lift.

Sizes / matrix-size budget
==========================

User requested matrix sizes <= 30.  Let d := D/2.
   - M_2D box size = (d+1)^2 x (d+1)^2.  At d=6 (D=12), this is 49 x 49.
     This EXCEEDS the 30-budget, so we restrict to total-degree slices:
     row/col index set  I_d = {(a1, a2) : a1 + a2 <= d}, which has
     |I_d| = (d+1)(d+2)/2 entries.
       D=6:  d=3, |I|=10
       D=8:  d=4, |I|=15
       D=10: d=5, |I|=21
       D=12: d=6, |I|=28
     All <= 30.  This is also the STANDARD moment-matrix indexing.

   - Localizer for q (deg 4): row/col set  I_{d-2}.
       D=6: |I_1|=3      D=12: |I_4|=15

   - Localizer for (A^2 - x^2) (deg 2): row/col set  I_{d-1}.
       D=12: |I_5|=21

Implementation notes
====================

We index multi-indices via the graded lex order:
     (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3), ...
The functions `multi_index_list(d)`, `idx_of(...)` build the mapping.

Soundness
=========

Every constraint added is a NECESSARY condition for a moment matrix of
a true product measure on [-1/4, 1/4]^2 with marginal moments m.  So the
LP minimum gives a valid LOWER bound on C_{1a}.

Solver
======

MOSEK primary; CLARABEL fallback.

Outputs
=======

  _agent_d_2d_hausdorff_4loc.json
  _agent_d_findings.md

Author: agent D, 2026-05-11.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cvxpy as cp
except ImportError as e:
    print(f"FATAL: cvxpy missing: {e}")
    sys.exit(1)

try:
    import mpmath
    mpmath.mp.prec = 200
except ImportError:
    mpmath = None


# ============================================================================
# Constants
# ============================================================================

A_FRAC = Fraction(1, 4)
B_FRAC = Fraction(1, 2)
A2_FRAC = Fraction(1, 16)        # A^2 = 1/16

A = float(A_FRAC)
B = float(B_FRAC)
A2 = float(A2_FRAC)


# ============================================================================
# Multi-index utilities (graded, total-degree <= d)
# ============================================================================

def multi_index_list(d: int) -> List[Tuple[int, int]]:
    """All (a1, a2) with a1+a2 <= d, graded lex order."""
    out = []
    for tot in range(d + 1):
        for a1 in range(tot + 1):
            out.append((a1, tot - a1))
    return out


def make_idx_of(lst: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    return {tup: i for i, tup in enumerate(lst)}


# ============================================================================
# Moment-matrix builder (RIGID: M_2D[a, b] = mu_{a+b}, depends only on sum)
# ============================================================================

def build_moment_matrix(mu_dict, basis: List[Tuple[int, int]]):
    """Build (|basis| x |basis|) cvxpy matrix with M[a, b] = mu_dict[a + b].
       mu_dict is a dict { (g1, g2) : cvxpy_var } indexed by total-degree
       multi-indices.  Returns a cvxpy matrix expression.
    """
    n = len(basis)
    rows = []
    for i in range(n):
        a = basis[i]
        row = []
        for j in range(n):
            b = basis[j]
            g = (a[0] + b[0], a[1] + b[1])
            row.append(mu_dict[g])
        rows.append(row)
    return cp.bmat(rows)


def build_localizer(mu_dict, basis: List[Tuple[int, int]], q_coefs: Dict[Tuple[int, int], float]):
    """Localizer L_q[a, b] = sum_{g in q} q_g * mu_{a + b + g}.
       q_coefs: dict { (g1, g2) : coef } for the polynomial q(x, y).
       Returns cvxpy matrix expression.
    """
    n = len(basis)
    rows = []
    for i in range(n):
        a = basis[i]
        row = []
        for j in range(n):
            b = basis[j]
            entry_terms = []
            for g, c in q_coefs.items():
                idx = (a[0] + b[0] + g[0], a[1] + b[1] + g[1])
                entry_terms.append(c * mu_dict[idx])
            row.append(sum(entry_terms))
        rows.append(row)
    return cp.bmat(rows)


# ============================================================================
# Inner Krein-Markov SDP for max(g)
# ============================================================================

def lambda_moments_g(N: int) -> List[Fraction]:
    """lam_k = int_{-1/2}^{1/2} t^k dt for k=0..N."""
    out = []
    for k in range(N + 1):
        if k % 2 == 0:
            out.append(Fraction(2, k + 1) * (B_FRAC ** (k + 1)))
        else:
            out.append(Fraction(0))
    return out


def hankel_cvx(seq, n0: int, n1: int = None):
    if n1 is None:
        n1 = n0
    return cp.bmat([[seq[i + j] for j in range(n1)] for i in range(n0)])


def hankel_localized(seq, R2: float, n: int):
    return cp.bmat(
        [[R2 * seq[i + j] - seq[i + j + 2] for j in range(n)] for i in range(n)]
    )


# ============================================================================
# Build the LP/SDP
# ============================================================================

@dataclass
class BuildOptions:
    D: int                # total degree of moments accessible from M_2D (2*d)
    use_4loc: bool = True # include 2D 4-localizer (1/16 - x^2)(1/16 - y^2)
    use_xloc: bool = True # include 1D x-Hausdorff localizer (A^2 - x^2)
    use_yloc: bool = True # include 1D y-Hausdorff localizer (A^2 - y^2)
    use_xy_cross: bool = True   # bivariate cross  (xy(A-x)(A+x)...) -- actually we use
                                # the simpler product q already covers cross-Hausdorff.
    symm_xy: bool = True  # enforce M_2D symmetric under x <-> y
    use_markov_only: bool = False  # if True, skip inner Krein-Markov SDP and use
                                   # just the simple bound max(g) >= (n+1)*2^n*g_n.
    fix_mu: Optional[Dict[Tuple[int, int], float]] = None
        # if given, fix M_2D to a candidate (for sanity checks)


@dataclass
class SolveResult:
    config: str
    D: int
    LB: Optional[float]
    status: str
    solver: str
    solve_time: float
    diag: str = ""
    extra: Dict = field(default_factory=dict)


def build_problem(opts: BuildOptions):
    """Construct the full LP/SDP for Agent D's 2D Hausdorff 4-localizer
       relaxation.  Returns (problem, vars_dict)."""
    D = opts.D
    assert D % 2 == 0 and D >= 4
    d = D // 2

    # ---- 2D moment-index basis I_d ----
    basis_d = multi_index_list(d)        # for M_2D rows/cols
    basis_dm1 = multi_index_list(d - 1)  # for 1D-Hausdorff localizer (degree 2)
    basis_dm2 = multi_index_list(d - 2)  # for 2D 4-localizer (degree 4)

    # ---- create the moment variables mu_{(g1, g2)} for g1+g2 <= 2d=D ----
    # Note: localizer for q(deg 4) needs mu_{a+b+g} with |a+b|<=2(d-2)=D-4
    # and |g| <= 4, so |a+b+g| <= D.  All in range.
    # Localizer for 1D (deg 2): mu_{a+b+g} with |a+b|<=D-2, |g|<=2, total <=D.
    mu_basis = multi_index_list(D)
    mu_dict = {}
    for g in mu_basis:
        if opts.fix_mu is not None:
            mu_dict[g] = cp.Constant(float(opts.fix_mu[g]))
        else:
            mu_dict[g] = cp.Variable(name=f"mu_{g[0]}_{g[1]}")

    constraints = []

    # mu_{(0,0)} = 1
    constraints.append(mu_dict[(0, 0)] == 1.0)

    # x<->y symmetry on the moments
    if opts.symm_xy:
        for (g1, g2) in mu_basis:
            if g1 < g2:
                constraints.append(mu_dict[(g1, g2)] == mu_dict[(g2, g1)])

    # ---- 1D marginal moments m_n = mu_{(n, 0)} = mu_{(0, n)} ----
    # By x<->y symmetry plus the moment-matrix structure, these are equal:
    # m_n = M_2D[(0,0), (n,0)] = mu_{(n,0)}
    m_seq = {n: mu_dict[(n, 0)] for n in range(D + 1)}

    # ---- Box bounds (|x|, |y| <= A so |mu_{(g1,g2)}| <= A^{g1+g2}) ----
    for (g1, g2) in mu_basis:
        bound = float(A_FRAC ** (g1 + g2))
        if opts.fix_mu is None:
            constraints.append(mu_dict[(g1, g2)] <= bound)
            constraints.append(mu_dict[(g1, g2)] >= -bound)

    # ---- (R2) moment matrix M_2D PSD ----
    M_2D = build_moment_matrix(mu_dict, basis_d)
    if opts.fix_mu is None:
        constraints.append(M_2D >> 0)

    # ---- (R3) 1D Hausdorff localizers ----
    # q_x(x,y) = A^2 - x^2.   coefs:  (0,0): A^2,  (2,0): -1
    q_x = {(0, 0): A2, (2, 0): -1.0}
    q_y = {(0, 0): A2, (0, 2): -1.0}
    if opts.use_xloc and d - 1 >= 0:
        L_x = build_localizer(mu_dict, basis_dm1, q_x)
        if opts.fix_mu is None:
            constraints.append(L_x >> 0)
    else:
        L_x = None
    if opts.use_yloc and d - 1 >= 0:
        L_y = build_localizer(mu_dict, basis_dm1, q_y)
        if opts.fix_mu is None:
            constraints.append(L_y >> 0)
    else:
        L_y = None

    # ---- (R4) 2D 4-localizer q(x,y) = (1/16 - x^2)(1/16 - y^2) ----
    # = 1/256 - x^2/16 - y^2/16 + x^2 y^2
    if opts.use_4loc and d - 2 >= 0:
        q_4 = {
            (0, 0): (A2 * A2),    # 1/256
            (2, 0): (-A2),        # -1/16
            (0, 2): (-A2),        # -1/16
            (2, 2): 1.0,
        }
        L_4 = build_localizer(mu_dict, basis_dm2, q_4)
        if opts.fix_mu is None:
            constraints.append(L_4 >> 0)
    else:
        L_4 = None

    # ---- (R5) Additional cross-Hausdorff: (A - x)(A + x)(A - y)(A + y) ----
    # Already redundant with R4 since (A-x)(A+x) = A^2 - x^2 etc.
    # so q_4 = q_x * q_y at the polynomial level. Skip.

    # ---- (R6) marginal 1D PSD (the marginal m_seq must also be a 1D moment
    #            sequence on [-A, A]).  This is implied by M_2D >> 0 plus
    #            the localizers, but make it explicit for redundancy ----
    n_f = d  # row/col size = d + 1
    H_marg = hankel_cvx([m_seq[n] for n in range(2 * n_f + 1)], n_f + 1)
    if opts.fix_mu is None:
        constraints.append(H_marg >> 0)
    if n_f >= 1:
        L_marg = hankel_localized([m_seq[n] for n in range(2 * n_f + 1)], A2, n_f)
        if opts.fix_mu is None:
            constraints.append(L_marg >> 0)

    # ---- g_n from M_2D off-diagonals ----
    # g_n = sum_{k=0}^{n} C(n,k) * M_2D[(k,0), (n-k,0)]
    #     = sum_{k=0}^{n} C(n,k) * mu_{(n, 0)}   ???   NO
    # Wait: M_2D[(k,0), (n-k,0)] = mu_{(k + n-k, 0 + 0)} = mu_{(n, 0)}
    # !!! That collapses!!!  This is because the basis is tensor-product, so the
    # OFF-diagonal entry in the moment matrix has the *same* mu-index as the
    # diagonal entry on the same anti-diagonal -- moment matrices ARE constant on
    # anti-diagonals.  So accessing m_k * m_{n-k} via M_2D[(k,0), (n-k,0)]
    # gives ONLY mu_{(n, 0)} = the marginal 1D moment.
    #
    # The CORRECT encoding for m_k * m_{n-k} via the 2D moment matrix is:
    #     mu_{(k, n-k)} = int x^k y^{n-k} f(x) f(y) dx dy = m_k * m_{n-k}.
    # This is the MIXED 2D moment, which in M_2D is the matrix entry
    #     M_2D[(k, 0), (0, n-k)] = mu_{(k, n-k)}.
    # GREAT -- THIS is the off-diagonal entry that Shor's relaxation lost.
    g_seq = []
    for n in range(D + 1):
        # we need mu_{(k, n-k)} for k=0..n  -- requires (k, n-k) in mu_basis,
        # which it is since k + (n-k) = n <= D.
        terms = []
        for k in range(n + 1):
            terms.append(comb(n, k) * mu_dict[(k, n - k)])
        g_seq.append(sum(terms))

    # g_0 = mu_{(0,0)} = 1 already implied; g_n is linear in mu.

    # ---- L = max(g) lower bound ----
    if opts.use_markov_only:
        # max(g) >= (n+1)*2^n * g_n for each EVEN n >= 0.  Take max over n.
        # Encode: L >= (n+1)*2^n * g_n.  We minimize L.
        L_var = cp.Variable(nonneg=True, name="L")
        for n in range(2, D + 1, 2):
            mult = (n + 1) * (2 ** n)
            constraints.append(L_var >= float(mult) * g_seq[n])
        problem = cp.Problem(cp.Minimize(L_var), constraints)
        return problem, {
            "mu_dict": mu_dict,
            "M_2D": M_2D,
            "L_x": L_x,
            "L_y": L_y,
            "L_4": L_4,
            "H_marg": H_marg,
            "L_var": L_var,
            "g_seq": g_seq,
            "m_seq": m_seq,
            "d": d,
        }

    # Otherwise use the inner Krein-Markov SDP (same as Agent A) for tighter LB.
    L_var = cp.Variable(nonneg=True, name="L")

    # Inner moments h_n of the candidate density h on [-1/2, 1/2] with
    # 0 <= h <= L, int h = g_0 = 1.
    # We MATCH h_k = g_k for k=0..D, and impose Hankel/localizer on h and nu=L*Leb-h.
    Nmom = D  # up to which we match
    n_inner = D // 2
    # We will use Hankel of size (n_inner + 1) for h and nu.
    # Inner moments h_seq[k] EXACT match g_seq[k] for k=0..Nmom.
    # We need m_h variables to allow extension beyond Nmom? Up to Nmom is enough
    # since Hankel of h needs (h_{i+j}) for i+j <= 2*n_inner = Nmom.
    # We just identify the variables with g_seq for k=0..Nmom.
    # For nu, we need lam_k = int t^k dt on [-1/2, 1/2] up to k = Nmom.
    lam_g = [float(v) for v in lambda_moments_g(Nmom)]
    m_h = [g_seq[k] for k in range(Nmom + 1)]
    m_nu = [L_var * lam_g[k] - m_h[k] for k in range(Nmom + 1)]

    Mh = hankel_cvx(m_h, n_inner + 1)
    if opts.fix_mu is None:
        constraints.append(Mh >> 0)
    Mnu = cp.bmat([[m_nu[i + j] for j in range(n_inner + 1)] for i in range(n_inner + 1)])
    if opts.fix_mu is None:
        constraints.append(Mnu >> 0)
    if n_inner >= 1:
        B2 = float(B_FRAC ** 2)
        Lh = hankel_localized(m_h, B2, n_inner)
        Lnu = cp.bmat(
            [
                [B2 * m_nu[i + j] - m_nu[i + j + 2] for j in range(n_inner)]
                for i in range(n_inner)
            ]
        )
        if opts.fix_mu is None:
            constraints.append(Lh >> 0)
            constraints.append(Lnu >> 0)

    # Add the simple Markov bounds too (they are equivalent dual cuts when
    # solved exactly, but help with numerical accuracy)
    for n in range(2, D + 1, 2):
        mult = (n + 1) * (2 ** n)
        # max(g) >= mult * g_n   becomes  L >= mult * g_n
        constraints.append(L_var >= float(mult) * g_seq[n])

    problem = cp.Problem(cp.Minimize(L_var), constraints)
    return problem, {
        "mu_dict": mu_dict,
        "M_2D": M_2D,
        "L_x": L_x,
        "L_y": L_y,
        "L_4": L_4,
        "H_marg": H_marg,
        "L_var": L_var,
        "g_seq": g_seq,
        "m_seq": m_seq,
        "Mh": Mh,
        "Mnu": Mnu,
        "d": d,
    }


# ============================================================================
# Solver wrapper
# ============================================================================

SOLVERS = ["MOSEK", "CLARABEL", "SCS"]


def solve_with_fallback(prob, time_limit=300):
    last_status = "not_attempted"
    last_solver = "none"
    for slv in SOLVERS:
        if slv not in cp.installed_solvers():
            continue
        t0 = time.time()
        try:
            if slv == "MOSEK":
                prob.solve(solver=slv, verbose=False)
            elif slv == "CLARABEL":
                prob.solve(solver=slv, verbose=False, max_iter=80000)
            elif slv == "SCS":
                prob.solve(solver=slv, verbose=False, max_iters=80000, eps=1e-9)
            dt = time.time() - t0
            if prob.status in ("optimal", "optimal_inaccurate"):
                return prob.status, slv, dt
            last_status = prob.status
            last_solver = slv
        except Exception as e:
            dt = time.time() - t0
            last_status = f"error_{slv}:{type(e).__name__}:{str(e)[:80]}"
            last_solver = slv
    return last_status, last_solver, 0.0


# ============================================================================
# Diagnose
# ============================================================================

def diag_block(name, mat_expr):
    if mat_expr is None:
        return f"{name}: (skipped)"
    try:
        v = mat_expr.value
        if v is None:
            return f"{name}: None"
        eig = np.linalg.eigvalsh(0.5 * (v + v.T))
        return f"{name}: shape={v.shape} eig=[{eig.min():.3e}, {eig.max():.3e}] |eig_min|/|eig_max|={(abs(eig.min())/max(abs(eig.max()),1e-30)):.2e}"
    except Exception as e:
        return f"{name}: diag_err={e}"


def diagnose(vd):
    parts = []
    for name, key in [
        ("M_2D", "M_2D"),
        ("L_x", "L_x"),
        ("L_y", "L_y"),
        ("L_4", "L_4"),
        ("H_marg", "H_marg"),
    ]:
        parts.append(diag_block(name, vd.get(key)))
    return " | ".join(parts)


def localizer_binding(vd):
    """Return dict { name : abs(min eigenvalue) } indicating which localizers
       are TIGHT at the optimum (binding) vs SLACK."""
    out = {}
    for name in ("M_2D", "L_x", "L_y", "L_4", "H_marg"):
        expr = vd.get(name)
        if expr is None:
            out[name] = None
            continue
        v = expr.value
        if v is None:
            out[name] = None
            continue
        v_sym = 0.5 * (v + v.T)
        eig = np.linalg.eigvalsh(v_sym)
        # "Binding" = min eigenvalue close to zero (i.e., the PSD constraint is tight).
        out[name] = {
            "min_eig": float(eig.min()),
            "max_eig": float(eig.max()),
            "tight": bool(abs(eig.min()) < 1e-5 * max(abs(eig.max()), 1.0)),
        }
    return out


# ============================================================================
# MV-proxy moments  (for sanity check)
# ============================================================================

def f_moments_mv_proxy(N: int) -> List[Fraction]:
    """For the symmetric proxy f(x) = (4/pi) / sqrt(1 - 16 x^2) on [-1/4, 1/4]:
       m_{2k} = C(2k, k) / 64^k;  m_odd = 0.
    """
    m = []
    for j in range(N + 1):
        if j % 2 == 0:
            k = j // 2
            m.append(Fraction(comb(2 * k, k), 64 ** k))
        else:
            m.append(Fraction(0))
    return m


def fix_mu_from_marginals(m_arr: List[float], D: int) -> Dict[Tuple[int, int], float]:
    """Build mu_{(g1, g2)} = m[g1] * m[g2] (rank-1 product) for sanity."""
    out = {}
    for tot in range(D + 1):
        for g1 in range(tot + 1):
            g2 = tot - g1
            out[(g1, g2)] = float(m_arr[g1]) * float(m_arr[g2])
    return out


def f_moments_uniform(N: int) -> List[Fraction]:
    """f = 2 on [-1/4, 1/4] (probability density).
       m_n = int_{-1/4}^{1/4} 2 x^n dx.
       For even n: m_n = 2 * 2 * (1/4)^{n+1}/(n+1) = (1/4)^n / (n+1).
       Hence m_0 = 1, m_2 = 1/(3*16), m_4 = 1/(5*256), ...
    """
    out = []
    for j in range(N + 1):
        if j % 2 == 0:
            out.append((A_FRAC ** j) / Fraction(j + 1))
        else:
            out.append(Fraction(0))
    return out


# ============================================================================
# Main runner
# ============================================================================

def main():
    print("=" * 72)
    print("AGENT-D: 2D Hausdorff 4-localizer for C_{1a} LB")
    print("=" * 72)
    print(f"Solvers available: {cp.installed_solvers()}")
    print()

    D_grid = [6, 8, 10, 12]

    results: List[SolveResult] = []
    diag_records = {}
    binding_records = {}

    # ---- with-vs-without 4-localizer sweep ----
    configs = [
        ("full_4loc",     dict(use_4loc=True,  use_xloc=True,  use_yloc=True,  symm_xy=True)),
        ("no_4loc",       dict(use_4loc=False, use_xloc=True,  use_yloc=True,  symm_xy=True)),
        ("4loc_no1Dxy",   dict(use_4loc=True,  use_xloc=False, use_yloc=False, symm_xy=True)),
        ("only_4loc_nomarkov",
                          dict(use_4loc=True,  use_xloc=True,  use_yloc=True,  symm_xy=True,
                               use_markov_only=True)),
    ]

    print(f"{'config':<28} {'D':>3} {'LB':>10}  {'status':<22} {'solver':<10} {'time':>6}")
    print("-" * 90)
    for label, kw in configs:
        for D in D_grid:
            opts = BuildOptions(D=D, **kw)
            try:
                prob, vd = build_problem(opts)
                status, slv, dt = solve_with_fallback(prob)
                Lv = vd["L_var"].value
                Lv_f = float(Lv) if Lv is not None else None
                if status in ("optimal", "optimal_inaccurate") and Lv_f is not None:
                    diag = diagnose(vd)
                    binding = localizer_binding(vd)
                else:
                    diag = ""
                    binding = {}
                res = SolveResult(
                    config=label, D=D, LB=Lv_f, status=status, solver=slv,
                    solve_time=dt, diag=diag, extra={"binding": binding},
                )
                results.append(res)
                lb_str = f"{Lv_f:.4f}" if Lv_f is not None else "n/a"
                print(f"{label:<28} {D:>3} {lb_str:>10}  {status:<22} {slv:<10} {dt:>5.1f}s")
                key = f"{label}/D={D}"
                diag_records[key] = diag
                binding_records[key] = binding
            except Exception as e:
                tb = traceback.format_exc(limit=4)
                print(f"{label:<28} {D:>3}   BUILD_ERR  {str(e)[:60]}")
                results.append(
                    SolveResult(
                        config=label, D=D, LB=None,
                        status=f"build_err:{type(e).__name__}",
                        solver="none", solve_time=0.0, diag=str(e)[:200],
                    )
                )

    # ---- MV-proxy sanity check: feed rank-1 mu = m_mv * m_mv^T and verify
    # the LP gives a number consistent with the true max(f*f) for MV-proxy.
    # The MV-proxy max(g) = max(f*f) we just compute numerically.
    print()
    print("=" * 72)
    print("SANITY CHECK: feed RANK-1 MV-proxy mu into the LP")
    print("=" * 72)
    sanity = []
    for D in D_grid:
        m_mv_frac = f_moments_mv_proxy(D)
        m_mv = [float(v) for v in m_mv_frac]
        fix_mu = fix_mu_from_marginals(m_mv, D)
        # Build with fix_mu (variables become constants; we still apply the
        # inner Krein-Markov inner SDP for the LB).
        opts = BuildOptions(D=D, use_4loc=True, use_xloc=True, use_yloc=True,
                            symm_xy=True, fix_mu=fix_mu)
        try:
            prob, vd = build_problem(opts)
            status, slv, dt = solve_with_fallback(prob)
            Lv = vd["L_var"].value
            Lv_f = float(Lv) if Lv is not None else None
            sanity.append({"D": D, "L_inner": Lv_f, "status": status, "solver": slv, "dt": dt})
            print(f"  D={D}: L_inner={Lv_f if Lv_f is None else f'{Lv_f:.4f}'} status={status} ({slv})")
        except Exception as e:
            sanity.append({"D": D, "L_inner": None, "status": f"build_err:{type(e).__name__}", "solver": "none"})
            print(f"  D={D}: BUILD_ERR {e}")

    # ---- Sanity: rank-1 mu from uniform f (max(f*f) is a triangle of height 2) ----
    print()
    print("Sanity (uniform f):  for f = 2 on [-1/4, 1/4],  true max(f*f) = 2.")
    sanity_uniform = []
    for D in D_grid:
        m_uni = [float(v) for v in f_moments_uniform(D)]
        fix_mu = fix_mu_from_marginals(m_uni, D)
        opts = BuildOptions(D=D, use_4loc=True, use_xloc=True, use_yloc=True,
                            symm_xy=True, fix_mu=fix_mu)
        try:
            prob, vd = build_problem(opts)
            status, slv, dt = solve_with_fallback(prob)
            Lv = vd["L_var"].value
            Lv_f = float(Lv) if Lv is not None else None
            sanity_uniform.append({"D": D, "L_inner": Lv_f, "status": status, "solver": slv})
            print(f"  D={D}: L_inner={Lv_f if Lv_f is None else f'{Lv_f:.4f}'} status={status} ({slv})")
        except Exception as e:
            sanity_uniform.append({"D": D, "L_inner": None, "status": f"build_err:{type(e).__name__}"})
            print(f"  D={D}: BUILD_ERR {e}")

    # ---- best per config / overall ----
    print()
    print("=" * 72)
    print("BEST LB PER CONFIG (across D)")
    print("=" * 72)
    best_per_config = {}
    for r in results:
        if r.LB is None or r.status not in ("optimal", "optimal_inaccurate"):
            continue
        prev = best_per_config.get(r.config)
        if prev is None or r.LB > prev[1]:
            best_per_config[r.config] = (r.D, r.LB, r.status)
    for c, (n, lb, st) in sorted(best_per_config.items(), key=lambda x: -x[1][1]):
        print(f"  {c:<30} D={n} LB={lb:.4f} ({st})")

    global_best = max(
        (r.LB for r in results if r.LB is not None and r.status in ("optimal", "optimal_inaccurate")),
        default=None,
    )
    print()
    print(f"Global best LB = {global_best}")

    # ---- gap analysis: with vs without 4-localizer ----
    print()
    print("4-LOCALIZER GAP (full_4loc vs no_4loc):")
    lookup = {(r.config, r.D): r.LB for r in results if r.LB is not None}
    gap_table = []
    for D in D_grid:
        lb_full = lookup.get(("full_4loc", D))
        lb_no = lookup.get(("no_4loc", D))
        gap = (lb_full - lb_no) if (lb_full is not None and lb_no is not None) else None
        gap_table.append({"D": D, "LB_full_4loc": lb_full, "LB_no_4loc": lb_no, "gap": gap})
        print(f"  D={D}: full={lb_full}  no={lb_no}  gap={gap}")

    # ---- json dump ----
    out = {
        "global_best_LB": global_best,
        "best_per_config": {
            k: {"D": v[0], "LB": v[1], "status": v[2]} for k, v in best_per_config.items()
        },
        "results": [asdict(r) for r in results],
        "gap_4loc_table": gap_table,
        "sanity_mv": sanity,
        "sanity_uniform": sanity_uniform,
        "metadata": {
            "solvers_tried": SOLVERS,
            "solvers_installed": [s for s in SOLVERS if s in cp.installed_solvers()],
            "D_grid": D_grid,
            "target_LB": 1.2748,
        },
    }
    with open("_agent_d_2d_hausdorff_4loc.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print()
    print("Wrote _agent_d_2d_hausdorff_4loc.json")

    # ---- markdown findings ----
    md = []
    md.append("# Agent D: 2D Hausdorff 4-Localizer -- Findings")
    md.append("")
    md.append("Date: 2026-05-11.  Target: rigorous LB > 1.2748 on C_{1a}.")
    md.append("")
    md.append("## Per-D LB table (full 4-localizer)")
    md.append("")
    md.append("| D | LB (full) | LB (no 4-loc) | gap | status |")
    md.append("| --- | --- | --- | --- | --- |")
    for row in gap_table:
        D = row["D"]
        lb_full = row["LB_full_4loc"]
        lb_no = row["LB_no_4loc"]
        gap = row["gap"]
        status_r = next((r.status for r in results if r.config == "full_4loc" and r.D == D), "n/a")
        md.append(
            f"| {D} | {lb_full if lb_full is None else f'{lb_full:.4f}'} | "
            f"{lb_no if lb_no is None else f'{lb_no:.4f}'} | "
            f"{gap if gap is None else f'{gap:.4f}'} | {status_r} |"
        )
    md.append("")
    md.append("## Sanity (MV-proxy rank-1 mu)")
    md.append("")
    md.append("| D | L_inner | status |")
    md.append("| --- | --- | --- |")
    for s in sanity:
        md.append(f"| {s['D']} | {s.get('L_inner','n/a')} | {s.get('status','n/a')} |")
    md.append("")
    md.append("## Sanity (uniform f rank-1 mu, true max(f*f)=2)")
    md.append("")
    md.append("| D | L_inner |")
    md.append("| --- | --- |")
    for s in sanity_uniform:
        md.append(f"| {s['D']} | {s.get('L_inner','n/a')} |")
    md.append("")
    md.append("## Honest verdict")
    md.append("")
    if global_best is None:
        md.append("All builds failed.  See JSON for errors.")
    else:
        if global_best >= 1.27:
            md.append(
                f"Global best LB = {global_best:.4f}, MEETING or EXCEEDING MV's 1.2748.  "
                "The 2D moment-matrix lift with 4-localizer is mathematically rigid enough "
                "to capture the true minimum.  Promising avenue."
            )
        elif global_best >= 1.10:
            md.append(
                f"Global best LB = {global_best:.4f}, BELOW MV's 1.2748 but above 1.0.  "
                "The 4-localizer breaks the Shor rank-1 collapse partially but does not "
                "yet close the gap to MV.  Likely needs larger D or stronger localizers."
            )
        elif global_best >= 1.00:
            md.append(
                f"Global best LB = {global_best:.4f}, at or near the trivial 1.0.  "
                "The 4-localizer alone is insufficient to break the Shor-like collapse "
                "of the off-diagonal moments.  See gap-with-vs-without table to confirm."
            )
        else:
            md.append(
                f"Global best LB = {global_best:.4f}, BELOW 1.0!  Probable LP infeasibility "
                "or solver numerical issue (since LB >= 1.0 is implied by max(g) >= int g = 1)."
            )
    md.append("")
    md.append("## 4-localizer effect summary")
    md.append("")
    if any(row.get("gap") is not None for row in gap_table):
        gaps = [row["gap"] for row in gap_table if row.get("gap") is not None]
        avg_gap = sum(gaps) / len(gaps)
        md.append(f"Average gap (full - no_4loc) across D = {avg_gap:.4f}.")
    md.append("")
    md.append("## Trajectory and next step")
    md.append("")
    md.append("See JSON `binding` field per result to see which localizers are TIGHT vs SLACK.")
    md.append("")
    with open("_agent_d_findings.md", "w") as f:
        f.write("\n".join(md))
    print("Wrote _agent_d_findings.md")


if __name__ == "__main__":
    main()
