"""Symbolic admissibility constraints on f in A := {f >= 0, supp f subset [-1/4, 1/4], int f = 1}.

Part B.2 of the multi-moment forbidden-region pipeline.

Variables
---------
We let N be the Fourier truncation order and use TWO variable layouts:

* **Magnitude-only layout** — variables are
      r = (r_1, ..., r_N),   r_n := |hat f(n)| in [0, 1].
  (The trivial constraint r_0 = hat f(0) = 1 is implicit.)  This is the
  simplest layout; it suffices when one only knows |hat f(n)|.

* **Real-part layout** — variables are
      c = (c_1, s_1, c_2, s_2, ..., c_N, s_N),
      c_n := Re hat f(n),  s_n := Im hat f(n).
  Magnitudes are recovered by r_n = sqrt(c_n^2 + s_n^2).  This layout is
  needed to encode the STRONG MO 2.17 constraint (which uses Re hat f).

Constraints
-----------
A.  **Trivial magnitude bound**:  r_n <= 1 (since |hat f(n)| <= int f = 1).
    Encoded in ``magnitude_box_constraints`` and ``realpart_box_constraints``.

B.  **Bochner / Toeplitz PSD** (``build_bochner_toeplitz``):
    Since {hat f(n)}_n is a sequence of non-negative-definite type on the
    integers (equivalently: f >= 0 in the period-1 sense), the infinite
    Toeplitz matrix T = [hat f(i - j)]_{i, j >= 0} is Hermitian PSD.  We
    truncate to the top-left (N + 1) x (N + 1) block.

C.  **Hausdorff moment PSD** (``build_hausdorff_psd_blocks``):
    If m_k := int x^k f(x) dx with supp f subset [-1/4, 1/4], then the
    *shifted* moments mu_k := 2 int (x + 1/4)^k f(x) dx = EE[(2X + 1/2)^k]
    of the random variable (2X + 1/2) on [0, 1/2] define a Hausdorff
    moment sequence on [0, 1/2], giving two Hankel PSD conditions:
        H1_{ij} := mu_{i + j}                    >= 0  (as matrix)
        H2_{ij} := (1/2) mu_{i + j} - mu_{i+j+1} >= 0  (as matrix)
    See Akhiezer / Shohat-Tamarkin.  We take the natural untransformed
    variant:  a symmetric measure on [-1/4, 1/4] has, for the moments
    m_k = int x^k f(x) dx,
        Hankel1_{ij} := m_{i + j},
        Hankel2_{ij} := (1/16) m_{i + j} - m_{i + j + 2},
    both PSD.  This is the two-Hankel form of the Hausdorff moment
    problem on [-1/4, 1/4] (support of f).

D.  **MO 2004 Lemma 2.17** (``build_mo_constraint``):
        Re hat f(2)  <=  2 * Re hat f(1) - 1.
    Strong form: linear constraint on (c_1, c_2) in the real-part layout.
    Weak (magnitude) form: phase-eliminated version
        r_2  <=  2 * r_1 + 1
    (trivially true for any r_n in [0,1], so weak form is VACUOUS — see
    docstring of ``build_mo_constraint``).  Hence the strong form is the
    only useful variant.

E.  **Fourier <-> spatial moment map** (``fourier_to_spatial_moment_map``):
        hat f(j) = int e^{-2 pi i j x} f(x) dx
                 = sum_{k >= 0} (-2 pi i j)^k / k!  *  m_k.
    We truncate at k = K_max and bound the tail using ``sup|x|^K <= (1/4)^K``.
    Returns:
        - the (2 N + 1) x (K_max + 1) linear matrix mapping m -> hat f
          (real part + imag part),
        - an arb ball on each row representing the truncation error,
    so that callers can impose  |hat f(j) - sum_{k <= K_max} (-2 pi i j)^k m_k / k!| <= err_j.

All symbolic matrices are returned as plain Python lists-of-lists with entries
in ``flint.fmpq`` (or ``flint.arb`` for the truncation errors) so they can be
cross-compiled into CVXPY / MOSEK models downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import flint
from flint import arb, fmpq

from .kernel_data import _to_arb, _arb_pi


# -----------------------------------------------------------------------------
# Variable layouts
# -----------------------------------------------------------------------------

def magnitude_var_names(N: int) -> List[str]:
    """Variable names for the magnitude-only layout."""
    return [f"r{n}" for n in range(1, N + 1)]


def realpart_var_names(N: int) -> List[str]:
    """Variable names for the real-part layout: (c_1, s_1, ..., c_N, s_N)."""
    names = []
    for n in range(1, N + 1):
        names.append(f"c{n}")
        names.append(f"s{n}")
    return names


# -----------------------------------------------------------------------------
# Trivial box constraints
# -----------------------------------------------------------------------------

@dataclass
class BoxConstraint:
    """lo <= var_index <= hi  with lo, hi as fmpq."""
    var_index: int
    lo: fmpq
    hi: fmpq


def magnitude_box_constraints(N: int) -> List[BoxConstraint]:
    """0 <= r_n <= 1 for n = 1..N."""
    return [BoxConstraint(i, fmpq(0), fmpq(1)) for i in range(N)]


def realpart_box_constraints(N: int) -> List[BoxConstraint]:
    """-1 <= c_n, s_n <= 1 for n = 1..N (since |hat f(n)| <= 1)."""
    return [BoxConstraint(i, fmpq(-1), fmpq(1)) for i in range(2 * N)]


# -----------------------------------------------------------------------------
# Bochner / Toeplitz PSD constraint
# -----------------------------------------------------------------------------

@dataclass
class ToeplitzSymbol:
    """Symbolic (N + 1) x (N + 1) Hermitian Toeplitz matrix T with T_{ij} = hat f(i - j).

    For j - i > 0 we have T_{ij} = overline(hat f(j - i)) = c_{j-i} - i * s_{j-i}.
    For j - i < 0 we have T_{ij} = hat f(i - j) = c_{i-j} + i * s_{i-j}.
    The diagonal is hat f(0) = 1.

    Layout for downstream SDP compilation: we represent each entry as a
    ``(re_coeffs, im_coeffs)`` pair of dicts mapping variable name to
    rational coefficient.  ``const`` is the additive constant.
    """
    N: int
    size: int  # N + 1
    # entry_re[i][j] = dict(var -> fmpq coeff) for the real part;
    # entry_im likewise.
    entry_re: List[List[dict]]
    entry_im: List[List[dict]]
    const_re: List[List[fmpq]]
    const_im: List[List[fmpq]]


def build_bochner_toeplitz(N: int) -> ToeplitzSymbol:
    """Build the (N+1)x(N+1) Hermitian Toeplitz symbolic matrix.

    Diagonal = 1 (hat f(0)).  Off-diagonal at offset d > 0: c_d + i s_d
    on the upper triangle, c_d - i s_d on the lower.
    """
    size = N + 1
    re = [[{} for _ in range(size)] for _ in range(size)]
    im = [[{} for _ in range(size)] for _ in range(size)]
    cre = [[fmpq(0)] * size for _ in range(size)]
    cim = [[fmpq(0)] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            d = i - j
            if d == 0:
                cre[i][j] = fmpq(1)  # hat f(0) = 1
            elif d > 0:
                # T_{ij} = hat f(d) = c_d + i s_d
                re[i][j] = {f"c{d}": fmpq(1)}
                im[i][j] = {f"s{d}": fmpq(1)}
            else:
                d_abs = -d
                # T_{ij} = hat f(-d_abs) = overline(hat f(d_abs)) = c_{d_abs} - i s_{d_abs}
                re[i][j] = {f"c{d_abs}": fmpq(1)}
                im[i][j] = {f"s{d_abs}": fmpq(-1)}
    return ToeplitzSymbol(
        N=N, size=size,
        entry_re=re, entry_im=im,
        const_re=cre, const_im=cim,
    )


# -----------------------------------------------------------------------------
# Hausdorff moment PSD blocks
# -----------------------------------------------------------------------------

@dataclass
class HankelSymbol:
    """Symbolic Hankel matrix whose entries are LINEAR in auxiliary moment
    variables m_0, m_1, m_2, ....

    Each entry is a dict(var_name -> fmpq coeff) plus a constant.
    """
    size: int
    entries: List[List[dict]]
    const: List[List[fmpq]]
    moment_var_names: List[str]


def build_hausdorff_psd_blocks(
    N: int, K_max: int = None
) -> Tuple[HankelSymbol, HankelSymbol]:
    """Return (Hankel1, Hankel2) for f supported on [-1/4, 1/4], f >= 0.

    These are the two Hankel PSD conditions of the Hausdorff moment
    problem on [-1/4, 1/4]:

        H1_{i,j} := m_{i + j}                       (size: floor(K_max/2)+1)
        H2_{i,j} := (1/16) m_{i + j} - m_{i + j + 2}  (one size smaller)

    Both matrices are PSD iff the moment sequence {m_k}_{k=0..K_max} is the
    moment sequence of some non-negative measure on [-1/4, 1/4].  See
    e.g. Shohat-Tamarkin or Lasserre (Ch. 3).

    Parameters
    ----------
    N : int
        Fourier truncation order; K_max defaults to 2*N so that Re hat f(N)
        can be expanded to leading order 2N in the Taylor series.
    K_max : int, optional
        Maximum spatial moment order to introduce.
    """
    if K_max is None:
        K_max = 2 * N
    mom_names = [f"m{k}" for k in range(K_max + 1)]
    # H1 size: how many rows/cols do we have?  H1_{ij} = m_{i+j} with
    # i+j <= K_max, so size = floor(K_max/2) + 1.
    sz1 = K_max // 2 + 1
    entries1 = [[{mom_names[i + j]: fmpq(1)} for j in range(sz1)] for i in range(sz1)]
    const1 = [[fmpq(0)] * sz1 for _ in range(sz1)]
    # m_0 = int f = 1, so H1[0][0] has constant 1 and empty variable dict.
    entries1[0][0] = {}
    const1[0][0] = fmpq(1)
    # Enforce m_0 = 1 by substituting elsewhere as well:
    # we'll emit an auxiliary equality constraint downstream; for now keep
    # m_0 as a free variable and add an equality constraint in the compiler.

    # H2 size: (1/16) m_{i+j} - m_{i+j+2}, so we need i+j+2 <= K_max, so
    # size = (K_max - 2) // 2 + 1 = K_max // 2  (for K_max even).
    if K_max >= 2:
        sz2 = (K_max - 2) // 2 + 1
    else:
        sz2 = 0
    entries2: List[List[dict]] = []
    const2: List[List[fmpq]] = []
    for i in range(sz2):
        row = []
        crow = []
        for j in range(sz2):
            d = {
                mom_names[i + j]: fmpq(1, 16),
                mom_names[i + j + 2]: fmpq(-1),
            }
            row.append(d)
            crow.append(fmpq(0))
        entries2.append(row)
        const2.append(crow)

    return (
        HankelSymbol(size=sz1, entries=entries1, const=const1, moment_var_names=mom_names),
        HankelSymbol(size=sz2, entries=entries2, const=const2, moment_var_names=mom_names),
    )


# -----------------------------------------------------------------------------
# Fourier <-> spatial moment map
# -----------------------------------------------------------------------------

@dataclass
class FourierSpatialMap:
    """Linear map m_0, m_1, ..., m_{K_max} -> (c_n, s_n) for n = 1..N.

    ``coeffs_re[n][k]`` is the fmpq coefficient of m_k in the Taylor
    approximation of c_n := Re hat f(n); similarly ``coeffs_im[n][k]``
    for s_n := Im hat f(n) (actually -Im hat f(n) up to sign; see below).

    Identities (from hat f(j) = int e^{-2 pi i j x} f(x) dx):
        c_j = sum_{k >= 0} (-1)^k (2 pi j)^{2k} / (2k)!  * m_{2k}
        s_j = - sum_{k >= 0} (-1)^k (2 pi j)^{2k+1} / (2k+1)!  * m_{2k+1}

    Note that the coefficients are irrational (powers of pi); so we
    store them as **arb balls**, not fmpq.  Rounding to rationals for
    the final certificate is handled downstream in ``certify.py``.

    Truncation error per row is bounded by
        err_j <= (1/4)^{K_max + 1} * 1  (since |m_k| <= (1/4)^k int f)
               * e^{2 pi j / 4} / (K_max + 1)!
    """
    N: int
    K_max: int
    coeffs_re: List[List[arb]]
    coeffs_im: List[List[arb]]
    err_re: List[arb]
    err_im: List[arb]


def fourier_to_spatial_moment_map(N: int, K_max: int) -> FourierSpatialMap:
    pi = _arb_pi()
    # Precompute factorials
    fact = [arb(1)]
    for k in range(1, K_max + 2):
        fact.append(fact[-1] * arb(k))

    coeffs_re: List[List[arb]] = []
    coeffs_im: List[List[arb]] = []
    err_re: List[arb] = []
    err_im: List[arb] = []
    for n in range(1, N + 1):
        two_pi_n = arb(2) * pi * arb(n)
        row_re = [arb(0)] * (K_max + 1)
        row_im = [arb(0)] * (K_max + 1)
        # power_k = (2 pi n)^k
        power_k = arb(1)
        sign_k = 1  # (-i)^k cycles; handle below
        for k in range(0, K_max + 1):
            if k > 0:
                power_k = power_k * two_pi_n
            # (-i)^k:  k % 4: 0 -> +1 real; 1 -> -i (so Im -> -1, Re -> 0); 2 -> -1 real; 3 -> +i
            term = power_k / fact[k]
            phase = k % 4
            if phase == 0:
                row_re[k] = term
            elif phase == 1:
                row_im[k] = -term
            elif phase == 2:
                row_re[k] = -term
            else:  # phase == 3
                row_im[k] = term
        # Tail error: |sum_{k > K_max} (2 pi n)^k / k! * m_k| <=
        #   (|m_k| bound)(1/4)^(K_max+1) * exp(2 pi n / 4)
        # where |m_k| <= (1/4)^k from support.  Let R = 2 pi n / 4 = pi n / 2.
        # Dominant term: (pi n / 2)^{K_max+1} / (K_max+1)! * exp(pi n / 2)
        R = pi * arb(n) / arb(2)
        e_R = arb.exp(R) if hasattr(arb, "exp") else _arb_exp(R)
        # tail <= (R)^(K_max+1) / (K_max+1)! * e_R  (Taylor remainder)
        R_pow = R ** (K_max + 1)
        tail = R_pow / fact[K_max + 1] * e_R
        err_re.append(tail)
        err_im.append(tail)
        coeffs_re.append(row_re)
        coeffs_im.append(row_im)
    return FourierSpatialMap(
        N=N, K_max=K_max,
        coeffs_re=coeffs_re, coeffs_im=coeffs_im,
        err_re=err_re, err_im=err_im,
    )


def _arb_exp(x: arb) -> arb:
    """Fallback exp for older flint versions."""
    try:
        return x.exp()
    except AttributeError:
        return arb.exp(x)  # may not exist either; flint 0.8 has .exp()


# -----------------------------------------------------------------------------
# MO 2004 Lemma 2.17 constraint
# -----------------------------------------------------------------------------

@dataclass
class LinearConstraint:
    """LHS: sum_v coeffs[v] * var_v + const  <=  0  (default)."""
    coeffs: dict        # var_name -> fmpq
    const: fmpq
    sense: str = "<="   # "<=", ">=", "=="


def build_mo_constraint(strong: bool = True) -> List[LinearConstraint]:
    """Return the MO 2004 Lemma 2.17 constraint as linear forms.

    Strong form (real-part layout):
        Re hat f(2) <= 2 * Re hat f(1) - 1
        <=> c_2 - 2 * c_1 + 1  <=  0.

    Weak form (magnitude layout):
        r_2 <= 2 * r_1 + 1,
    which is VACUOUSLY TRUE for r_1, r_2 in [0,1] since LHS <= 1 and
    RHS >= 1.  Hence the weak form is returned as the trivial constraint
    0 <= 0 (a marker); the strong form is the only useful variant.
    """
    if strong:
        return [LinearConstraint(
            coeffs={"c2": fmpq(1), "c1": fmpq(-2)},
            const=fmpq(1),
            sense="<=",
        )]
    # weak form — vacuous
    return [LinearConstraint(coeffs={}, const=fmpq(0), sense="<=")]


# -----------------------------------------------------------------------------
# Convenience: full admissibility set builder
# -----------------------------------------------------------------------------

@dataclass
class AdmissibilitySet:
    """Aggregated admissibility constraints for use by forbidden_region.py.

    Variables (real-part layout):
        c_1, s_1, c_2, s_2, ..., c_N, s_N.

    Auxiliary variables (when ``include_hausdorff=True``):
        m_0, m_1, ..., m_{K_max}  with constraints
            H1, H2 PSD (Hausdorff),
            m_0 == 1,
            |c_n - sum_k a_{n,k} m_k| <= err_n   (Fourier<->spatial link).
    """
    N: int
    realpart_vars: List[str]
    box: List[BoxConstraint]
    toeplitz: ToeplitzSymbol
    mo_linear: List[LinearConstraint]
    use_hausdorff: bool
    hausdorff1: HankelSymbol = None
    hausdorff2: HankelSymbol = None
    fourier_spatial: FourierSpatialMap = None
    spatial_K_max: int = 0


def build_admissibility_set(
    N: int,
    use_mo: bool = True,
    mo_strong: bool = True,
    use_hausdorff: bool = False,
    spatial_K_max: int = None,
) -> AdmissibilitySet:
    """Bundle all admissibility constraints for a given (N, options) into a
    single struct that SDP / QCQP compilers can consume."""
    if spatial_K_max is None:
        spatial_K_max = 2 * N
    names = realpart_var_names(N)
    box = realpart_box_constraints(N)
    toeplitz = build_bochner_toeplitz(N)
    mo = build_mo_constraint(strong=mo_strong) if use_mo else []
    res = AdmissibilitySet(
        N=N, realpart_vars=names, box=box, toeplitz=toeplitz,
        mo_linear=mo, use_hausdorff=use_hausdorff,
    )
    if use_hausdorff:
        H1, H2 = build_hausdorff_psd_blocks(N, K_max=spatial_K_max)
        fs = fourier_to_spatial_moment_map(N, K_max=spatial_K_max)
        res.hausdorff1 = H1
        res.hausdorff2 = H2
        res.fourier_spatial = fs
        res.spatial_K_max = spatial_K_max
    return res


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    flint.ctx.prec = 212
    print("=" * 70)
    print("moment_constraints.py — self-test")
    print("=" * 70)
    S = build_admissibility_set(N=3, use_mo=True, mo_strong=True, use_hausdorff=True)
    print(f"  N = {S.N}, variables = {S.realpart_vars}")
    print(f"  Box constraints: {len(S.box)}")
    print(f"  Toeplitz size = {S.toeplitz.size}")
    print(f"  MO linear constraints: {len(S.mo_linear)}")
    if S.mo_linear:
        lc = S.mo_linear[0]
        print(f"    MO: {lc.coeffs}  const={lc.const}  sense={lc.sense}")
    print(f"  Hausdorff1 size = {S.hausdorff1.size}")
    print(f"  Hausdorff2 size = {S.hausdorff2.size}")
    print(f"  spatial K_max  = {S.spatial_K_max}")
    # Taylor map preview
    fs = S.fourier_spatial
    print(f"  c_1 Taylor row (k=0..{fs.K_max}): "
          f"{[float(v.mid()) for v in fs.coeffs_re[0]]}")
    print(f"  tail err for c_1: {fs.err_re[0]}")
