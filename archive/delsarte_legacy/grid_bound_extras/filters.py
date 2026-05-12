"""Admissibility filters on Fourier moments (a_j, b_j) of admissible f.

Variables
---------
At level N, the "candidate" is
    (a_1, b_1, a_2, b_2, ..., a_N, b_N)  in  R^{2N}
where a_j + i b_j = hat f(j) for a hypothetical admissible f.  An
admissibility filter is a predicate F((a, b)) that is TRUE for every
admissible f.  Contrapositive: if F((a, b)) = FALSE, no admissible f has
those moments.

For cell-search, each filter is evaluated on arb enclosures of (a, b).
We return a tri-state verdict:
    * REJECT -- rigorously FALSE over the entire cell (=> cell has no admissible points).
    * ACCEPT -- rigorously TRUE over the entire cell (=> filter doesn't cut here).
    * UNCLEAR -- the cell straddles the filter boundary.  (Refine or keep.)

Soundness rule: NEVER REJECT unless the evaluation proves false throughout
the cell.  Aggressive rejection = bug.  Conservative = slow but correct.

Phase 2a provides: F1, F2, F3 (degenerate), F4_MO217, F7, F8, F9.
Phase 2b will add: general Schmudgen (F4 full), Fejer-Riesz (F5), F6.

Each filter's proof is reproduced in its docstring.
"""
from __future__ import annotations

from enum import Enum
from typing import Sequence

from flint import arb, acb, fmpq, ctx


def _arb_sqr(x: arb) -> arb:
    """Dependency-aware square of an arb interval.

    Returns the tight enclosure of {x_0^2 : x_0 in x}.  For a cell straddling
    zero, flint's ``x * x`` treats the factors as independent and gives
    [-(max|x|)^2, (max|x|)^2]; this helper returns [0, (max|x|)^2] properly.
    """
    al = x.abs_lower()
    au = x.abs_upper()
    return (al * al).union(au * au)


class FilterVerdict(Enum):
    REJECT  = "REJECT"    # filter rigorously FALSE throughout the cell
    ACCEPT  = "ACCEPT"    # filter rigorously TRUE throughout the cell
    UNCLEAR = "UNCLEAR"   # cell straddles boundary; cannot decide


def _arb_nonneg(x: arb) -> FilterVerdict:
    """Tri-state 'is x >= 0?' given an arb interval.

    REJECT means x is rigorously < 0 throughout (so the 'x >= 0' predicate is false).
    ACCEPT means x is rigorously >= 0 throughout.
    UNCLEAR otherwise.
    """
    if x.upper() < 0:
        return FilterVerdict.REJECT
    if x.lower() >= 0:
        return FilterVerdict.ACCEPT
    return FilterVerdict.UNCLEAR


# =============================================================================
#  F_bathtub -- Lemma 3.4 / Chebyshev-Markov (M-dependent)
# =============================================================================
#  Lemma 1 of multi_moment_derivation.md:  if h = f*f is bounded above by M,
#  supported on [-1/2, 1/2], integrable to 1, then for every n >= 1,
#      |hat h(n)|  <=  M sin(pi/M) / pi  =:  mu(M).
#  Since hat h(n) = hat f(n)^2, this gives
#      z_n^2 = |hat f(n)|^2 = |hat h(n)|  <=  mu(M),
#  i.e.,
#      a_n^2 + b_n^2  <=  mu(M)   for every n >= 1.
# =============================================================================

def F_bathtub(ab: Sequence[arb], N: int, mu_arb: arb) -> FilterVerdict:
    """a_n^2 + b_n^2 <= mu(M)  for every n = 1..N.

    mu_arb is an arb upper bound on mu(M); we use mu_arb.upper() as a scalar
    upper bound, then check  mu.upper() - z_n^2 >= 0.
    """
    verdict = FilterVerdict.ACCEPT
    for n in range(1, N + 1):
        a = ab[2 * (n - 1)]
        b = ab[2 * (n - 1) + 1]
        z_sq = _arb_sqr(a) + _arb_sqr(b)
        v = _arb_nonneg(mu_arb - z_sq)
        if v == FilterVerdict.REJECT:
            return FilterVerdict.REJECT
        if v == FilterVerdict.UNCLEAR:
            verdict = FilterVerdict.UNCLEAR
    return verdict


# =============================================================================
#  F1 -- Moment magnitude
# =============================================================================
#  Proof: |hat f(j)| = |int f(x) exp(-2 pi i j x) dx| <= int f dx = 1
#  since f >= 0 everywhere.  Hence a_j^2 + b_j^2 <= 1 for every j >= 1.
#  QED.
# =============================================================================

def F1_magnitude(ab: Sequence[arb], N: int) -> FilterVerdict:
    """a_j^2 + b_j^2 <= 1  for every j = 1..N.

    Returns REJECT if any z_j^2 > 1 rigorously; ACCEPT if all <= 1
    rigorously; UNCLEAR otherwise.
    """
    verdict = FilterVerdict.ACCEPT
    for j in range(1, N + 1):
        a, b = ab[2 * (j - 1)], ab[2 * (j - 1) + 1]
        z_sq = _arb_sqr(a) + _arb_sqr(b)
        # The predicate "z_sq <= 1" equivalent to "1 - z_sq >= 0"
        v = _arb_nonneg(arb(1) - z_sq)
        if v == FilterVerdict.REJECT:
            return FilterVerdict.REJECT
        if v == FilterVerdict.UNCLEAR:
            verdict = FilterVerdict.UNCLEAR
    return verdict


# =============================================================================
#  F2 -- Moment Cauchy-Schwarz (spec §3.2)
# =============================================================================
#  Proof:  a_n = int f cos(2 pi n x) dx.  Cauchy-Schwarz for the finite
#  measure f dx:
#      a_n^2 = (int f cos(2 pi n x))^2 <= (int f) (int f cos^2(2 pi n x))
#            = 1 * (int f (1 + cos(4 pi n x)) / 2)
#            = (1 + a_{2n}) / 2.
#  Similarly b_n^2 <= (1 - a_{2n}) / 2 via sin^2 = (1 - cos(2 theta))/2.
#  QED.
#
#  This is only a constraint when index 2n is <= N (a_{2n} is a variable).
#  For 2n > N we can still apply using the weaker fact a_{2n} in [-1, 1].
# =============================================================================

def F2_moment_cs(ab: Sequence[arb], N: int) -> FilterVerdict:
    """For every n in 1..N: a_n^2 <= (1 + a_{2n}) / 2 and b_n^2 <= (1 - a_{2n}) / 2.

    If 2n <= N: uses the actual a_{2n} variable.
    If 2n >  N: uses the weaker bound a_{2n} in [-1, 1].
    """
    verdict = FilterVerdict.ACCEPT
    for n in range(1, N + 1):
        a_n = ab[2 * (n - 1)]
        b_n = ab[2 * (n - 1) + 1]
        two_n = 2 * n
        if two_n <= N:
            a_2n = ab[2 * (two_n - 1)]
        else:
            # a_{2n} unknown, use the worst-case interval [-1, 1]
            a_2n = arb(0, 1)   # ball centred 0, radius 1
        # a_n^2 <= (1 + a_{2n}) / 2   <=>   (1 + a_{2n}) / 2 - a_n^2 >= 0
        v1 = _arb_nonneg((arb(1) + a_2n) / arb(2) - _arb_sqr(a_n))
        if v1 == FilterVerdict.REJECT:
            return FilterVerdict.REJECT
        if v1 == FilterVerdict.UNCLEAR:
            verdict = FilterVerdict.UNCLEAR
        v2 = _arb_nonneg((arb(1) - a_2n) / arb(2) - _arb_sqr(b_n))
        if v2 == FilterVerdict.REJECT:
            return FilterVerdict.REJECT
        if v2 == FilterVerdict.UNCLEAR:
            verdict = FilterVerdict.UNCLEAR
    return verdict


# =============================================================================
#  F3 -- Autoconvolution identity
# =============================================================================
#  hat h(j) = hat f(j)^2  =>  Re hat h(j) = a_j^2 - b_j^2,  Im hat h(j) = 2 a_j b_j.
#  This is an identity, not an inequality: it just DEFINES hat h entries
#  from (a_j, b_j).  It's used implicitly by F8.
# =============================================================================

def hat_h_entry(a_n: arb, b_n: arb) -> tuple[arb, arb]:
    """Return (Re hat h(n), Im hat h(n)) given (a_n, b_n).

    By hat h(n) = hat f(n)^2, (a+ib)^2 = (a^2 - b^2) + i (2 a b).
    """
    return a_n * a_n - b_n * b_n, arb(2) * a_n * b_n


# =============================================================================
#  F4_MO217 -- Martin-O'Bryant Lemma 2.17 (spec §3.4 special case)
# =============================================================================
#  Proof: the polynomial p(y) = y(1-y) = y - y^2 >= 0 on [0, 1].
#  Apply spec §3.4 Schmudgen: int f(x) p(cos(2 pi x)) dx >= 0.
#  Expand y = cos, y^2 = (1 + cos(4 pi x))/2 = (T_0 + T_2)/2.
#  So y - y^2 = -1/2 + cos(2 pi x) - cos(4 pi x)/2 in Chebyshev.
#  Integrating against f:
#      -1/2 + a_1 - a_2/2  >=  0
#  i.e.
#      a_2 <= 2 a_1 - 1.   QED.
#
#  Note: this involves ONLY the real parts a_1 and a_2.  b_1 and b_2 are
#  unconstrained by MO 2.17.  Requires N >= 2.
# =============================================================================

def F4_MO217(ab: Sequence[arb], N: int) -> FilterVerdict:
    """MO 2.17: a_2 <= 2 a_1 - 1   <=>   2 a_1 - 1 - a_2 >= 0.  (N >= 2 only.)"""
    if N < 2:
        return FilterVerdict.ACCEPT   # vacuous at N < 2
    a_1 = ab[0]
    a_2 = ab[2]
    return _arb_nonneg(arb(2) * a_1 - arb(1) - a_2)


# =============================================================================
#  F7 -- Bochner-Herglotz: Toeplitz PSD on hat f
# =============================================================================
#  The sequence (hat f(n))_{n in Z} is the Fourier transform of the
#  non-negative finite measure f dx.  By Bochner-Herglotz, the Hermitian
#  Toeplitz matrix
#      T_f[N] := [ hat f(i - j) ]_{i, j = 0..N}
#  is positive semi-definite.  Proof (spec §3.7):
#      x^* T_f x = int f(x') | sum_i x_i exp(-2 pi i i x') |^2 dx' >= 0.
#  QED.
#
#  Rigorous check: we compute the leading principal minors up to size N+1
#  as arb intervals using acb_mat determinants (since T_f is complex Hermitian).
#  If any has .upper() < 0 strictly throughout the cell => REJECT.
#  If all have .lower() >= 0 => ACCEPT (strictly PD => PSD).
#  Note: Sylvester's criterion with leading minors gives PD not PSD; our
#  test is conservative -- cells on the PSD BOUNDARY register as UNCLEAR,
#  which is correct (we don't want to reject cells containing admissible
#  boundary points).
# =============================================================================

def _T_f_acb(ab: Sequence[arb], N: int):
    """Construct T_f[N] as a list-of-lists of acb entries at current ctx.prec."""
    # T_f[i][j] = hat f(i - j).  Row i, column j indexed 0..N.
    # hat f(0) = 1 (real, exact).
    # hat f( k) = a_k + i b_k    for k >= 1
    # hat f(-k) = a_k - i b_k
    def hat_f(k: int) -> acb:
        if k == 0:
            return acb(1)
        if k > 0:
            return acb(ab[2 * (k - 1)], ab[2 * (k - 1) + 1])
        k = -k
        return acb(ab[2 * (k - 1)], -ab[2 * (k - 1) + 1])
    M = [[hat_f(i - j) for j in range(N + 1)] for i in range(N + 1)]
    return M


def _acb_mat_det(M: list[list]) -> acb:
    """Compute det of a square list-of-lists matrix of acb entries by cofactor.

    Used only for small N+1 (<= 4 in Phase 2a).  For larger, replace with
    LU decomposition via ``flint.acb_mat``.
    """
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    # Cofactor along the first row
    total = acb(0)
    for j in range(n):
        minor = [[M[i][jj] for jj in range(n) if jj != j] for i in range(1, n)]
        cof = _acb_mat_det(minor)
        sign = acb(1) if j % 2 == 0 else acb(-1)
        total = total + sign * M[0][j] * cof
    return total


def _leading_principal_minor_real(ab: Sequence[arb], N: int, k: int) -> arb:
    """The k-th leading principal minor of T_f[N], returned as REAL arb.

    For Hermitian M, the determinants of principal submatrices are real.
    We return M_top_k.det().real; the .imag is a tiny radius by cancellation.
    """
    M = _T_f_acb(ab, N)
    top = [[M[i][j] for j in range(k)] for i in range(k)]
    det = _acb_mat_det(top)
    return det.real


def F7_bochner_f(ab: Sequence[arb], N: int) -> FilterVerdict:
    """T_f[N] >= 0 (Bochner-Herglotz).  N >= 1.

    We test leading principal minors 1..N+1.
    """
    if N < 1:
        return FilterVerdict.ACCEPT
    verdict = FilterVerdict.ACCEPT
    # k = 1: diagonal entry = 1, trivially PD; skip.
    for k in range(2, N + 2):
        minor = _leading_principal_minor_real(ab, N, k)
        v = _arb_nonneg(minor)
        if v == FilterVerdict.REJECT:
            return FilterVerdict.REJECT
        if v == FilterVerdict.UNCLEAR:
            verdict = FilterVerdict.UNCLEAR
    return verdict


# =============================================================================
#  F8 -- Autoconvolution Bochner: Toeplitz PSD on hat h
# =============================================================================
#  h = f*f >= 0 (since f >= 0 and real).  h is in L^1 ∩ L^inf; applying
#  Bochner-Herglotz to h as for F7 gives T_h[N] >= 0 PSD where
#      T_h[i, j] = hat h(i - j),   hat h(n) = hat f(n)^2.
#  The entries hat h(n) are determined by (a_j, b_j) via F3.  See spec §3.8.
# =============================================================================

def _T_h_acb(ab: Sequence[arb], N: int):
    """Construct T_h[N] via hat h(n) = hat f(n)^2."""
    def hat_h(k: int) -> acb:
        if k == 0:
            return acb(1)
        if k > 0:
            fc = acb(ab[2 * (k - 1)], ab[2 * (k - 1) + 1])
            return fc * fc
        k = -k
        fc = acb(ab[2 * (k - 1)], -ab[2 * (k - 1) + 1])
        return fc * fc
    return [[hat_h(i - j) for j in range(N + 1)] for i in range(N + 1)]


def _leading_principal_minor_h_real(ab: Sequence[arb], N: int, k: int) -> arb:
    M = _T_h_acb(ab, N)
    top = [[M[i][j] for j in range(k)] for i in range(k)]
    return _acb_mat_det(top).real


def F8_bochner_h(ab: Sequence[arb], N: int) -> FilterVerdict:
    """T_h[N] >= 0 with hat h(n) = hat f(n)^2.  N >= 1."""
    if N < 1:
        return FilterVerdict.ACCEPT
    verdict = FilterVerdict.ACCEPT
    for k in range(2, N + 2):
        minor = _leading_principal_minor_h_real(ab, N, k)
        v = _arb_nonneg(minor)
        if v == FilterVerdict.REJECT:
            return FilterVerdict.REJECT
        if v == FilterVerdict.UNCLEAR:
            verdict = FilterVerdict.UNCLEAR
    return verdict


# =============================================================================
#  Combined: run all Phase 2a filters and return the most conservative verdict.
# =============================================================================

def filter_all(ab: Sequence[arb], N: int, *,
               mu_arb: arb = None,
               enable_F4_MO217: bool = True,
               enable_F7: bool = True,
               enable_F8: bool = True) -> FilterVerdict:
    """Run F1, F2, F_bathtub (if mu provided), F4_MO217, F7, F8 (all optional).

    Returns REJECT if any filter rigorously rejects; UNCLEAR if any is
    UNCLEAR (and none rejects); else ACCEPT.
    """
    any_unclear = False
    # Order: cheap filters first (fail-fast), Bochner PSD last.
    # Bathtub is the dominant cut for N >= 2 and is very cheap.
    if mu_arb is not None:
        v = F_bathtub(ab, N, mu_arb)
        if v == FilterVerdict.REJECT:
            return FilterVerdict.REJECT
        if v == FilterVerdict.UNCLEAR:
            any_unclear = True
    for f, enabled in [
        (F1_magnitude,   True),
        (F2_moment_cs,   True),
        (F4_MO217,       enable_F4_MO217),
        (F7_bochner_f,   enable_F7),
        (F8_bochner_h,   enable_F8),
    ]:
        if not enabled:
            continue
        v = f(ab, N)
        if v == FilterVerdict.REJECT:
            return FilterVerdict.REJECT
        if v == FilterVerdict.UNCLEAR:
            any_unclear = True
    return FilterVerdict.UNCLEAR if any_unclear else FilterVerdict.ACCEPT


__all__ = [
    "FilterVerdict",
    "F1_magnitude",
    "F2_moment_cs",
    "F4_MO217",
    "F7_bochner_f",
    "F8_bochner_h",
    "F_bathtub",
    "hat_h_entry",
    "filter_all",
]
