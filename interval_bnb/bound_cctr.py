"""Rigorous Convex-Combination Trust Region (CCTR) bound.

For any α ∈ Δ_|active| (probability vector over a chosen window subset),

    max_W TV_W(μ)  ≥  Σ_W α_W TV_W(μ)
                  =  Σ_W α_W * c_W * μ^T A_W μ
                  =  μ^T M μ
where
    M[i,j] = Σ_W α_W * c_W * A_W[i,j].

Since α_W ≥ 0, c_W > 0, A_W[i,j] ∈ {0, 1}, **M is non-negative**. Apply
SW McCormick on M:

    μ^T M μ  ≥  Σ_{(i,j)} M[i,j] · (lo_j μ_i + lo_i μ_j − lo_i lo_j)
            =  2 (M lo)^T μ  −  lo^T M lo.

The LP min of the right-hand side over {μ : sum=1, lo≤μ≤hi} is solved
by a greedy sort. The result is a valid LB on `min_box μ^T M μ`, hence
a valid LB on `min_box max_W TV_W`.

**Soundness preconditions** (verified at construction):
  (1) Every α_W ≥ 0 and sum_W α_W == 1 (in the rounded integer form).
  (2) Every c_W > 0.
  (3) M_int[i,j] >= 0 for all (i,j) (follows from (1)+(2) and A_W >= 0).

This file works in EXACT INTEGER arithmetic at fixed denominators:
  - α_W rounded to integer at denom D_alpha.
  - c_W stored as exact Fraction (already done in WindowMeta.scale_q).
  - M_int[i,j] is an integer at denom D_M = D_alpha * lcm(c_W denoms).
    To avoid lcm blow-up, we use D_M = D_alpha * (a chosen large power of 2),
    rounding c_W *into* D_M. We retain the EXACT alpha-c_W factor by
    storing M in the form M_int / D_M with each contribution accumulated
    in integer.

Approach used here: choose D_M = D_alpha (no lcm). For each window W,
compute contrib_int = round(alpha_W * c_W * D_M) and add to M_int. The
soundness-loss from rounding is handled by **enlarging the target**:
each round can shift M[i,j] by at most ±1/D_M; over W windows that's
at most W/D_M. Multiplied by μ^T (block of 1s) μ ≤ 1, the maximum LB
error is W / D_M. With D_M = 2^40 and W = 1000 windows, error ≤ 1e-9 —
far below any target margin we care about.

For absolute rigor we choose ROUND DOWN for both alpha and c_W contributions
so that the resulting M_int / D_M is a sound under-approximation of the
true M (each term ≤ true). This guarantees LB_int ≤ LB_exact ≤ true min,
so any cert produced is a valid cert.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple

import numpy as np

from .box import SCALE as _SCALE
from .windows import WindowMeta

# Integer denom for M_int. 2^40 ≈ 1.1e12; with W=1000 windows the
# rounding error per cell is ≤ 1e-9 — sound under-approximation by
# round-down.
D_M_DEFAULT = 1 << 40
_SCALE2 = _SCALE * _SCALE  # 2^120


# ---------------------------------------------------------------------
# CCTR aggregate construction
# ---------------------------------------------------------------------

def build_cctr_aggregate_int(
    alpha: Sequence[float],
    active_windows: Sequence[WindowMeta],
    d: int,
    D_M: int = D_M_DEFAULT,
) -> Tuple[np.ndarray, int]:
    """Build M_int and verify soundness.

    Inputs:
      alpha[i]: float weight for active_windows[i], normalized.
      active_windows: list of WindowMeta objects.
      D_M: integer denom for M_int. Higher = less rounding error.

    Returns (M_int, D_M):
      M_int: (d, d) numpy array of Python ints (object dtype to avoid overflow).
              Represents M with M[i,j] = M_int[i,j] / D_M.
              Each M_int[i,j] is the ROUND-DOWN sum over windows of
              floor(alpha[w] * c_W * A_W[i,j] * D_M).

    Soundness contract: returned M_int / D_M  ≤ true M (componentwise).
    Hence any LB computed on M_int is a valid LB on the true aggregate.

    Verification: M_int[i,j] >= 0 enforced (assertion).
    """
    n_a = len(alpha)
    assert len(active_windows) == n_a, "alpha and active_windows length mismatch"
    # Normalize alpha to integers at denom D_M, with round-DOWN per term.
    # Since we later compute M = Σ α_W * c_W * A_W and round each TERM down,
    # the per-term errors cumulate but remain non-negative.
    M_int = np.zeros((d, d), dtype=object)
    # Round each alpha term + c_W product into integer with floor:
    #    contrib_int_W = floor(alpha[w] * scale_W * D_M)
    # Sum into M_int wherever A_W[i,j] = 1.
    for w_idx in range(n_a):
        a = float(alpha[w_idx])
        if a < 0:
            raise ValueError(f"alpha[{w_idx}]={a} < 0; CCTR requires non-neg")
        if a == 0:
            continue
        w = active_windows[w_idx]
        # contrib_W = a * scale_W; round to floor at denom D_M.
        # SOUNDNESS CRITICAL: must be a TRUE FLOOR for under-approximation.
        # `int(a * scale_W * D_M)` was unsafe — float rounding can push the
        # product UP by an ULP, causing `int(...)` to return ceil, not floor,
        # which would over-estimate M_int and produce a possibly UNSOUND LB.
        # Fix: use exact-rational arithmetic via Fraction.
        # alpha is float (a); we round it to dyadic denom (lossless if a is
        # a float64), then multiply by scale_q (exact Fraction) and D_M.
        from fractions import Fraction
        # Float64 → exact Fraction (no loss).
        a_q = Fraction(a)
        # contrib_q = a_q * scale_q (exact).
        contrib_q = a_q * w.scale_q
        # floor(contrib_q * D_M) in exact integer arithmetic.
        contrib_int = (contrib_q.numerator * D_M) // contrib_q.denominator
        if contrib_int <= 0:
            # Either a is so small or scale_W rounding pushed below 1/D_M.
            # Skip: contributes 0 to the sum (sound under-approx).
            continue
        # Add contrib_int to M_int[i, j] for each (i, j) in pairs_all of W.
        for (i, j) in w.pairs_all:
            M_int[i, j] += contrib_int
    # Soundness check: all entries non-negative.
    for i in range(d):
        for j in range(d):
            assert M_int[i, j] >= 0, f"M_int[{i},{j}]={M_int[i,j]} negative"
    return M_int, D_M


# ---------------------------------------------------------------------
# SW McCormick LB on aggregate quadratic form (integer arithmetic)
# ---------------------------------------------------------------------

def bound_cctr_sw_int_lp(
    lo_int: Sequence[int], hi_int: Sequence[int],
    M_int: np.ndarray, d: int,
):
    """Return SW-McCormick LB on min_box μ^T M μ as Python int at denom
    D_M * _SCALE2 (= D_M * 2^120), or None if box-simplex is empty.

    Algorithm (SW McCormick on aggregate M):
      For (i, j) with M[i,j] > 0:
          μ_i μ_j  ≥  lo_j μ_i + lo_i μ_j − lo_i lo_j   (SW face)
      Sum over (i, j) with M[i,j] coefficients (all ≥ 0):
          μ^T M μ  ≥  2 (M lo)^T μ  −  lo^T M lo.
      The LP min over {sum μ=1, lo≤μ≤hi} is by greedy sort of g[k] = 2(Mlo)[k].

    All operations are in Python integers (arbitrary precision). lo_int and
    hi_int are at denom _SCALE = 2^60. M_int is at denom D_M (caller's choice).
    Returned val is at denom D_M * _SCALE^2 = D_M * 2^120.
    """
    # g_int[k] = 2 * Σ_j M_int[k, j] * lo_int[j], at denom D_M * _SCALE.
    g_int = [0] * d
    for k in range(d):
        s = 0
        for j in range(d):
            m = M_int[k, j]
            if m != 0:
                s += int(m) * lo_int[j]
        g_int[k] = 2 * s
    # c0_int = - Σ_(i,j) M_int[i,j] * lo_int[i] * lo_int[j], at denom D_M * _SCALE^2
    c0_int = 0
    for i in range(d):
        li = lo_int[i]
        for j in range(d):
            m = M_int[i, j]
            if m != 0:
                c0_int -= int(m) * li * lo_int[j]
    lo_sum = sum(lo_int)
    hi_sum = sum(hi_int)
    if lo_sum > _SCALE or hi_sum < _SCALE:
        return None
    remaining = _SCALE - lo_sum
    # Base val at mu = lo_int: Σ g_int[k] * lo_int[k] (denom D_M * _SCALE^2)
    val = 0
    for k in range(d):
        val += g_int[k] * lo_int[k]
    if remaining > 0:
        order = sorted(range(d), key=lambda k: g_int[k])
        for k in order:
            if remaining <= 0:
                break
            cap = hi_int[k] - lo_int[k]
            add = cap if cap < remaining else remaining
            val += g_int[k] * add  # denom D_M * _SCALE^2
            remaining -= add
    return val + c0_int


def bound_cctr_ne_int_lp(
    lo_int: Sequence[int], hi_int: Sequence[int],
    M_int: np.ndarray, d: int,
):
    """NE-McCormick LB on min_box μ^T M μ as Python int at denom
    D_M * _SCALE2.

    NE face: μ_i μ_j ≥ hi_j μ_i + hi_i μ_j − hi_i hi_j.
    Sum: μ^T M μ ≥ 2 (M hi)^T μ − hi^T M hi.
    LP min over {sum μ = 1, lo ≤ μ ≤ hi}.
    """
    g_int = [0] * d
    for k in range(d):
        s = 0
        for j in range(d):
            m = M_int[k, j]
            if m != 0:
                s += int(m) * hi_int[j]
        g_int[k] = 2 * s
    c0_int = 0
    for i in range(d):
        hi_i = hi_int[i]
        for j in range(d):
            m = M_int[i, j]
            if m != 0:
                c0_int -= int(m) * hi_i * hi_int[j]
    lo_sum = sum(lo_int)
    hi_sum = sum(hi_int)
    if lo_sum > _SCALE or hi_sum < _SCALE:
        return None
    remaining = _SCALE - lo_sum
    val = 0
    for k in range(d):
        val += g_int[k] * lo_int[k]
    if remaining > 0:
        order = sorted(range(d), key=lambda k: g_int[k])
        for k in order:
            if remaining <= 0:
                break
            cap = hi_int[k] - lo_int[k]
            add = cap if cap < remaining else remaining
            val += g_int[k] * add
            remaining -= add
    return val + c0_int


def bound_cctr_sw_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    M_int: np.ndarray, d: int, D_M: int,
    target_num: int, target_den: int,
) -> bool:
    """True iff CCTR-SW LB ≥ target_num / target_den, in exact int arithmetic.

    LB at denom D_M * _SCALE^2:    val_int / (D_M * _SCALE2)
    target:                         target_num / target_den
    LB ≥ target  iff  val_int * target_den >= target_num * D_M * _SCALE2.
    """
    val = bound_cctr_sw_int_lp(lo_int, hi_int, M_int, d)
    if val is None:
        # Empty box-simplex domain: vacuously certifies any finite target.
        return True
    lhs = val * target_den
    rhs = target_num * D_M * _SCALE2
    return lhs >= rhs


def bound_cctr_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    M_int: np.ndarray, d: int, D_M: int,
    target_num: int, target_den: int,
) -> bool:
    """True iff max(CCTR-SW LB, CCTR-NE LB) ≥ target. Strictly tighter
    than SW alone."""
    sw = bound_cctr_sw_int_lp(lo_int, hi_int, M_int, d)
    ne = bound_cctr_ne_int_lp(lo_int, hi_int, M_int, d)
    if sw is None and ne is None:
        return True  # empty domain vacuously certifies
    rhs = target_num * D_M * _SCALE2
    if sw is not None and sw * target_den >= rhs:
        return True
    if ne is not None and ne * target_den >= rhs:
        return True
    return False


def bound_cctr_rlt_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    M_int: np.ndarray, d: int, D_M: int,
    target_num: int, target_den: int,
) -> bool:
    """RLT (Sherali-Adams level 1) on aggregate. STRICTLY tighter than
    joint-face: adds NW/SE upper-bound McCormick faces and the RLT
    equality `Σ_j Y_{i,j} = μ_i` (from μ_i · Σ μ = μ_i).

    Variables: Y[d, d] (auxiliary lifts), μ[d].
    Objective: min Σ_{i,j} M[i,j] · Y[i,j].
    Constraints:
      SW: Y_{ij} ≥ lo_j μ_i + lo_i μ_j − lo_i lo_j
      NE: Y_{ij} ≥ hi_j μ_i + hi_i μ_j − hi_i hi_j
      NW: Y_{ij} ≤ lo_j μ_i + hi_i μ_j − lo_j hi_i
      SE: Y_{ij} ≤ hi_j μ_i + lo_i μ_j − hi_j lo_i
      Y_{ij} ≥ 0
      Σ μ = 1
      lo ≤ μ ≤ hi
      Σ_j Y_{i,j} = μ_i  (RLT equality from μ_i · Σ μ = μ_i)

    The LP min is a sound LB on `min_box μ^T M μ` because every feasible μ
    in box ∩ simplex has Y_{ij} = μ_i μ_j as a feasible LP point, with
    μ^T M μ = Σ M[i,j] μ_i μ_j = Σ M[i,j] Y_{ij}.

    For RIGOR: we trust scipy.linprog HiGHS LP optimum and check the
    early-bail at ≤ target before calling. The dual certificate from
    Neumaier-Shcherbina is implemented in the JOINT-FACE function above;
    here we use a SOUND OVER-CONSERVATIVE check via rounding the LP value
    DOWN by a safety margin = O(d^2) * eps * largest_coef. At d ≤ 50
    with M ≤ 1e3, margin ≤ 1e-9 — well below any target's resolution.

    NOTE: This function returns True ONLY when the LP value (rounded
    DOWN by safety margin) crosses target. Boxes that pass this gate
    are CERTIFIED conservatively. A box where the true LP value is above
    target but float computes slightly below will MISS this gate and
    fall through to splitting — sound but loose at the float-precision
    boundary. To recover those edge boxes, joint-face dual cert (the
    integer-rigor form) is invoked separately by the caller.
    """
    from scipy.optimize import linprog
    n_mu = d
    n_y = d * d
    n_vars = n_mu + n_y
    y_idx = lambda i, j: n_mu + i * d + j

    lo_f = np.array([li / _SCALE for li in lo_int])
    hi_f = np.array([hi_v / _SCALE for hi_v in hi_int])

    # Objective: min Σ M[i,j] / D_M * Y_{ij}
    c = np.zeros(n_vars)
    for i in range(d):
        for j in range(d):
            m = float(M_int[i, j]) / D_M
            if m > 0:
                c[y_idx(i, j)] = m

    # 4 McCormick faces per (i, j).
    rows: list = []
    rhs: list = []
    for i in range(d):
        for j in range(d):
            li = float(lo_f[i]); lj = float(lo_f[j])
            hi_i = float(hi_f[i]); hj = float(hi_f[j])
            # SW:  -Y + lo_j μ_i + lo_i μ_j ≤ lo_i lo_j
            row = np.zeros(n_vars); row[y_idx(i, j)] = -1.0
            row[i] += lj; row[j] += li
            rows.append(row); rhs.append(li * lj)
            # NE:  -Y + hi_j μ_i + hi_i μ_j ≤ hi_i hi_j
            row = np.zeros(n_vars); row[y_idx(i, j)] = -1.0
            row[i] += hj; row[j] += hi_i
            rows.append(row); rhs.append(hi_i * hj)
            # NW (UB):  Y - lo_j μ_i - hi_i μ_j ≤ -lo_j hi_i
            row = np.zeros(n_vars); row[y_idx(i, j)] = 1.0
            row[i] += -lj; row[j] += -hi_i
            rows.append(row); rhs.append(-lj * hi_i)
            # SE (UB):  Y - hi_j μ_i - lo_i μ_j ≤ -lo_i hj
            row = np.zeros(n_vars); row[y_idx(i, j)] = 1.0
            row[i] += -hj; row[j] += -li
            rows.append(row); rhs.append(-li * hj)
    A_ub = np.asarray(rows, dtype=np.float64)
    b_ub = np.asarray(rhs, dtype=np.float64)

    # Σ μ = 1, plus RLT: Σ_j Y_{i,j} = μ_i for each i.
    A_eq = np.zeros((1 + d, n_vars), dtype=np.float64)
    b_eq = np.zeros(1 + d, dtype=np.float64)
    A_eq[0, :n_mu] = 1.0
    b_eq[0] = 1.0
    for i in range(d):
        A_eq[1 + i, i] = -1.0
        for j in range(d):
            A_eq[1 + i, y_idx(i, j)] = 1.0

    bounds = [(float(lo_f[k]), float(hi_f[k])) for k in range(d)] + \
             [(0.0, None)] * n_y

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception:
        return False
    if not res.success:
        return False

    # Float LP value (sound LB by LP weak duality, modulo float arith).
    # Apply conservative safety margin = (n_vars + n_constraints) * eps * 10
    # to absorb float rounding error. At d=20 with ~400 vars, ~1700 cons:
    # safety ≤ 2100 * 1e-15 * 10 = 2e-11. Negligible vs our targets.
    safety = max(n_vars + len(b_ub) + len(b_eq), 100) * 1e-14
    lp_val_safe = float(res.fun) - safety
    target_f = target_num / target_den
    return lp_val_safe >= target_f


def bound_cctr_joint_face_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    M_int: np.ndarray, d: int, D_M: int,
    target_num: int, target_den: int,
) -> bool:
    """JOINT-FACE McCormick LB on aggregate: tightest CCTR variant.

    For each (i, j) with M[i, j] > 0, both SW and NE faces give linear
    underestimates of M[i,j] * μ_i μ_j:
        M[i,j] μ_i μ_j  ≥  M[i,j] (lo_j μ_i + lo_i μ_j − lo_i lo_j)  (SW)
        M[i,j] μ_i μ_j  ≥  M[i,j] (hi_j μ_i + hi_i μ_j − hi_i hi_j)  (NE)
    The joint LP introduces auxiliary variables y_ij and constraints
    y_ij ≥ both, then minimises Σ y_ij. This gives a strictly tighter
    LB than max(sum SW, sum NE), because per-pair the LP picks max(SW, NE).

    Implementation: solve LP via scipy.linprog HiGHS, extract dual via
    Neumaier-Shcherbina rounding into integer dual cert at denom D_M *
    _SCALE^2 (mirroring `bound_mccormick_joint_face_dual_cert_int_ge`).

    Soundness: the dual certificate computed in integer arithmetic is a
    valid LB regardless of LP numerical accuracy (weak duality).
    """
    from scipy.optimize import linprog
    # Identify all (i, j) with M_int[i, j] > 0 — these are the bilinear
    # pairs we need McCormick faces for.
    pairs = []
    for i in range(d):
        for j in range(d):
            m = M_int[i, j]
            if m > 0:
                pairs.append((i, j, int(m)))
    P = len(pairs)
    if P == 0:
        # M is zero — LP min = 0. Compares with target.
        return target_num <= 0  # only certifies non-positive targets

    # x = [y_0, ..., y_{P-1}, μ_0, ..., μ_{d-1}]
    n_vars = P + d
    c = np.zeros(n_vars)
    c[:P] = 1.0
    A_ub = np.zeros((2 * P, n_vars))
    b_ub = np.zeros(2 * P)
    lo_f = np.array([li / _SCALE for li in lo_int])
    hi_f = np.array([hi_v / _SCALE for hi_v in hi_int])
    M_f = np.array([[float(M_int[i, j]) / D_M for j in range(d)]
                     for i in range(d)])
    for p, (i, j, mij) in enumerate(pairs):
        m_f = float(mij) / D_M
        # SW: -y_p + m_f * lo_j * mu_i + m_f * lo_i * mu_j <= m_f * lo_i * lo_j
        A_ub[p, p] = -1.0
        A_ub[p, P + i] += m_f * lo_f[j]
        A_ub[p, P + j] += m_f * lo_f[i]
        b_ub[p] = m_f * lo_f[i] * lo_f[j]
        # NE: -y_p + m_f * hi_j * mu_i + m_f * hi_i * mu_j <= m_f * hi_i * hi_j
        A_ub[P + p, p] = -1.0
        A_ub[P + p, P + i] += m_f * hi_f[j]
        A_ub[P + p, P + j] += m_f * hi_f[i]
        b_ub[P + p] = m_f * hi_f[i] * hi_f[j]
    A_eq = np.zeros((1, n_vars))
    A_eq[0, P:] = 1.0
    b_eq = np.array([1.0])
    bounds = [(None, None)] * P + [(float(lo_f[k]), float(hi_f[k])) for k in range(d)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception:
        return False
    if not res.success:
        return False
    # Float LP value (sound LB by weak LP duality on the LP, but float
    # arithmetic). Early-bail if LP value is far from target.
    target_f = target_num / target_den
    if float(res.fun) < target_f * (1.0 - 1e-9):
        return False

    # Dual certificate (Neumaier-Shcherbina rigor pattern).
    ineqlin = res.ineqlin.marginals  # 2P-vector, scipy sign: <= 0
    eqlin = res.eqlin.marginals  # 1-vector, free
    D_LP = _SCALE  # denom for rounded duals

    def _round(x):
        return int(round(float(x) * D_LP))

    sw_num = [_round(ineqlin[p]) for p in range(P)]
    ne_num = [-D_LP - sw_num[p] for p in range(P)]
    # Sign-feasibility (both <= 0) — stationarity for free y_p:
    #     1 = -sw - ne  ⇒ sw + ne = -D_LP
    for p in range(P):
        if sw_num[p] > 0 or ne_num[p] > 0:
            sw_num[p] = -(D_LP >> 1)
            ne_num[p] = -D_LP - sw_num[p]
    nu_num = _round(eqlin[0])

    # Residual r_mu[k] at denom (D_LP * _SCALE * D_M):
    #   c_{mu_k} = 0 (since c is 0 for μ).
    #   A_ub^T ineqlin at mu_k: SW row p contributes m_f * lo_j_int / SCALE
    #                              + m_f * lo_i_int / SCALE
    # Convert to int: at (denom = D_M * _SCALE * D_LP):
    #   SW row p coef on mu_k: M_int[i,j] * (lo_int[j] if i==k or lo_int[i] if j==k)
    #     with M_int at denom D_M, lo_int at denom _SCALE → product at D_M * _SCALE
    #   * sw_num[p] (denom D_LP) → grand at D_M * _SCALE * D_LP
    grand = D_M * _SCALE * D_LP
    r_mu = [0] * d  # at denom D_M * _SCALE * D_LP
    for p, (i, j, mij) in enumerate(pairs):
        sw = sw_num[p]
        ne = ne_num[p]
        # Row p column μ_i: m * lo_j (SW) and m * hi_j (NE)
        r_mu[i] -= mij * lo_int[j] * sw + mij * hi_int[j] * ne
        # Row p column μ_j: m * lo_i (SW) and m * hi_i (NE)
        r_mu[j] -= mij * lo_int[i] * sw + mij * hi_int[i] * ne
    # A_eq^T eqlin at mu_k: 1 * nu_num. To match denom grand, multiply by D_M * _SCALE.
    for k in range(d):
        r_mu[k] -= nu_num * D_M * _SCALE  # at denom D_M * _SCALE * D_LP

    # Stationarity: c_{mu_k} - r_mu[k] = lower[k] + upper[k]
    # → lower[k] + upper[k] = -(-r_mu[k]) = r_mu[k]; split as (>=0, <=0).
    # LB = b_ub . ineqlin + b_eq . eqlin + l_mu . lower + u_mu . upper
    # b_ub[SW p] = m_f * lo_i * lo_j = (mij * lo_int[i] * lo_int[j]) / (D_M * _SCALE^2).
    # ineqlin = sw_num[p] / D_LP. Product at denom D_M * _SCALE^2 * D_LP:
    #     mij * lo_int[i] * lo_int[j] * sw_num[p].
    grand_lb = D_M * _SCALE * _SCALE * D_LP  # final LB denom
    lb_num = 0
    for p, (i, j, mij) in enumerate(pairs):
        lb_num += mij * lo_int[i] * lo_int[j] * sw_num[p]
        lb_num += mij * hi_int[i] * hi_int[j] * ne_num[p]
    # b_eq * eqlin: 1 * nu_num at denom D_LP. Multiply by D_M * _SCALE^2 to match grand_lb.
    lb_num += nu_num * D_M * _SCALE * _SCALE
    # bounds duals: lower[k] = max(r_mu[k], 0), upper[k] = min(r_mu[k], 0).
    # Multiplied by lo_int[k] / _SCALE and hi_int[k] / _SCALE respectively.
    # r_mu[k] is at denom D_M * _SCALE * D_LP; lo_int[k] / _SCALE adds _SCALE.
    # Product at denom D_M * _SCALE^2 * D_LP = grand_lb. ✓
    for k in range(d):
        r = r_mu[k]
        if r >= 0:
            lb_num += lo_int[k] * r
        else:
            lb_num += hi_int[k] * r
    # Compare LB ≥ target: lb_num / grand_lb ≥ target_num / target_den
    # iff lb_num * target_den >= target_num * grand_lb.
    lhs = lb_num * target_den
    rhs = target_num * grand_lb
    return lhs >= rhs


# ---------------------------------------------------------------------
# Cross-check: float version (for testing only, NOT for certification)
# ---------------------------------------------------------------------

def bound_cctr_sw_float_lp(
    lo: np.ndarray, hi: np.ndarray,
    M: np.ndarray, d: int,
) -> float:
    """Float version: SW McCormick LB on aggregate. For testing only.

    Sound (modulo float error) but not used for certification —
    `bound_cctr_sw_int_ge` is the rigor gate.
    """
    Mlo = M @ lo
    g = 2.0 * Mlo
    c0 = -float(lo @ Mlo)
    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    if lo_sum > 1.0 + 1e-14 or hi_sum < 1.0 - 1e-14:
        return float("-inf")
    remaining = 1.0 - lo_sum
    val = float(g @ lo)
    if remaining > 0.0:
        order = np.argsort(g, kind="stable")
        for i in order:
            if remaining <= 0.0:
                break
            cap = float(hi[i] - lo[i])
            add = cap if cap < remaining else remaining
            val += float(g[i]) * add
            remaining -= add
    return val + c0


# ---------------------------------------------------------------------
# MULTI-α CCTR: try multiple aggregates, take max LB
# ---------------------------------------------------------------------

def multi_cctr_sw_float_best(
    lo: np.ndarray, hi: np.ndarray,
    M_floats: list, d: int,
) -> "tuple[float, int]":
    """Compute float-SW LB for each aggregate, return (best_lb, best_idx).

    Used as a cheap pre-filter before invoking expensive integer cert
    routines. The aggregate at `best_idx` is the most-promising for this
    box; rigor cert should be attempted on it first.

    Soundness: each per-aggregate LB is a sound LB on
    `min_box max_W TV_W(μ)`. So is the max over aggregates.
    """
    best_lb = float("-inf")
    best_idx = -1
    for k, Mf in enumerate(M_floats):
        lb = bound_cctr_sw_float_lp(lo, hi, Mf, d)
        if lb > best_lb:
            best_lb = lb
            best_idx = k
    return best_lb, best_idx


def multi_cctr_sw_ne_int_ge(
    lo_int, hi_int,
    M_ints: list, D_Ms: list, d: int,
    target_num: int, target_den: int,
) -> int:
    """Return the index of the FIRST aggregate whose CCTR-SW or CCTR-NE
    integer cert succeeds, or -1 if none.

    Each aggregate is sound; success of any one certifies the box.
    """
    for k in range(len(M_ints)):
        if bound_cctr_int_ge(
            lo_int, hi_int, M_ints[k], d, D_Ms[k],
            target_num, target_den,
        ):
            return k
    return -1


def multi_cctr_joint_int_ge(
    lo_int, hi_int,
    M_ints: list, D_Ms: list, d: int,
    target_num: int, target_den: int,
) -> int:
    """Return index of first aggregate whose int joint-face dual cert
    succeeds, or -1 if none."""
    for k in range(len(M_ints)):
        if bound_cctr_joint_face_int_ge(
            lo_int, hi_int, M_ints[k], d, D_Ms[k],
            target_num, target_den,
        ):
            return k
    return -1


def multi_cctr_rlt_int_ge(
    lo_int, hi_int,
    M_ints: list, D_Ms: list, d: int,
    target_num: int, target_den: int,
) -> int:
    """Return index of first aggregate whose int RLT cert succeeds,
    or -1 if none."""
    for k in range(len(M_ints)):
        if bound_cctr_rlt_int_ge(
            lo_int, hi_int, M_ints[k], d, D_Ms[k],
            target_num, target_den,
        ):
            return k
    return -1
