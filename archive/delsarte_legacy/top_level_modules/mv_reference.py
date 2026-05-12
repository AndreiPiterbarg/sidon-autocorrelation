"""Matolcsi-Vinuesa (and successors) reference for the lower bound on C_{1a}.

================================================================================
IMPORTANT CLARIFICATION ABOUT THE 1.2802 CITATION
================================================================================

The figure ``1.2802`` appearing in the project CLAUDE.md is NOT due to
Matolcsi-Vinuesa.  According to Terence Tao's crowdsourced optimisation
constants repository

    https://teorth.github.io/optimizationproblems/constants/1a.html

the table of known lower bounds on C_{1a} is:

    1          (trivial)
    1.182778   [MO2004]  Martin & O'Bryant, Exp. Math. 16 (2007)
                         arXiv:math/0410004
    1.262      [MO2009]  Martin & O'Bryant, Ill. J. Math. 53 (2009)
                         arXiv:0807.5121
    1.2748     [MV2009]  Matolcsi & Vinuesa, J. Math. Anal. Appl. 372 (2010)
                         arXiv:0907.1379
    1.28       [CS2017]  Cloninger & Steinerberger, Proc. AMS 145 (2017)
                         arXiv:1403.7988
    1.2802     [XX2026]  Xie, Xinyuan. UNPUBLISHED improvement (Grok chat), 2026

So:
  * The best PUBLISHED lower bound is  C_{1a} >= 1.28  (Cloninger-Steinerberger).
  * 1.2802 is an unpublished claim by X. Xie cited on Tao's website as
    originating from a Grok chat; no paper, no proof certificate, no g is
    archived.  Treat this number as provisional.
  * Matolcsi-Vinuesa's own bound is 1.2748 (sometimes quoted 1.2749 after
    rounding), which is WEAKER than 1.28.

================================================================================
DEFINITION OF C_{1a}  (Tao's repository convention -- matches this project)
================================================================================

C_{1a} is the largest constant such that, for every non-negative
f : R -> R,

    max_{|t| <= 1/2}  (f * f)(t)  >=  C_{1a} * ( int_{-1/4}^{1/4} f )^2.

Equivalently, with the normalisation  supp f \subset [-1/4, 1/4],
int f = 1,

    max_{|t| <= 1/2} (f * f)(t)  >=  C_{1a}.

The bound is for GENERAL non-negative f -- NOT restricted to even f.
(Tao's page explicitly states "for all non-negative f : R -> R".)

================================================================================
MATOLCSI-VINUESA 2010                arXiv:0907.1379
================================================================================

Title:   "Improved bounds on the supremum of autoconvolutions"
Authors: Mate Matolcsi, Carlos Vinuesa
Journal: J. Math. Anal. Appl. 372 (2010), no. 2, 439-447.

Main theorem (paraphrased):
    For every non-negative f with supp f \subset [-1/4, 1/4] and int f = 1,

        || f * f ||_infty  >=  1.2748.

Their theoretical limit (remark at end of the paper) is approximately 1.276;
i.e. their argument cannot yield more than ~1.276 in its current form.

---  Method (Delsarte-style dual / positivity argument) ---

They use the classical "convolution-square dualization":
if B(x, y) is a positive-definite function vanishing outside the strip
|x+y| <= 1/2 in the sense required by their argument, then
sup(f*f) is controlled by a quotient involving B.

Their test kernel is NOT a single elementary function.  Instead,
a family of 1-variable trigonometric polynomials of the form

        G(x) = sum_{j=1}^{n} a_j cos(2 pi j x / u)           (*)

is optimised numerically (n = 119, u = 0.638, delta = 0.138) against the
auxiliary kernel

        K(x) = (1/delta) * (beta * beta)(x/delta)
        beta(x) = (2/pi) / sqrt(1 - 4 x^2)   on (-1/2, 1/2)

(an arcsine autoconvolution / Chebyshev-type density).
The 119 coefficients a_j are tabulated in the paper's Appendix; they are
obtained by MATHEMATICA-based numerical optimisation of the quadratic form

        minimise    sum_{j=1}^{n}  a_j^2 / |J_0(pi delta j / u)|^2

where J_0 is the Bessel function of order zero.  Hence MV's g is a
numerical trig polynomial with 119 computed coefficients -- not a closed
form in the F1 (Fejer x cosine) family.

Take-aways for our F1 search:
  * MV's g is morally a cosine-polynomial weighted by an arcsine kernel
    (beta * beta), which is NOT a Fejer kernel.  Translating MV into F1
    would require extending F1 with the arcsine-autoconvolution weight,
    and allowing ~100 cosine terms.
  * Their theoretical limit ~1.276 means even a perfectly optimised
    member of their family cannot exceed the CS2017 value of 1.28.

================================================================================
CLONINGER-STEINERBERGER 2017         arXiv:1403.7988
================================================================================

Title:   "On Suprema of Autoconvolutions with an Application to Sidon sets"
Authors: A. Cloninger, S. Steinerberger
Journal: Proc. Amer. Math. Soc. 145 (2017), 3191-3200.

Main theorem (Theorem 1.1 / corollary cited by subsequent authors):

    S  >=  1.28                       (for general non-negative f)

via a branch-and-prune argument on step-function discretisations
(the "cascade" method this project calls `cloninger-steinerberger/`).
Their construction is NOT a single dual certificate g; it is a
combinatorial exhaustion that certifies a lower bound on

    val(d) = min_{mu in Delta_d}  max_W  mu^T M_W mu

for a discretised problem and then relates val(d) to C_{1a}.

================================================================================
XIE 2026                             UNPUBLISHED
================================================================================

Claim:   C_{1a}  >=  1.2802

No archived manuscript.  The only source is a Grok conversation referenced
on Tao's optimisation-problems website.  The test function, method, and
numerical certificate are unknown to this project as of 2026-04-19.

================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

# ---------------------------------------------------------------------------
# Numerical constants (all exactly as stated in the cited sources)
# ---------------------------------------------------------------------------

# Best rigorously published lower bound (Cloninger-Steinerberger 2017).
CS2017_BOUND = mp.mpf("1.28")

# Matolcsi-Vinuesa 2010 rigorously published bound.
MV_BOUND = mp.mpf("1.2748")

# Matolcsi-Vinuesa's *theoretical ceiling* for their method.
MV_THEORETICAL_LIMIT = mp.mpf("1.276")

# Xie 2026 unpublished claim (listed on Tao's constants page but without
# an archived proof certificate).
XIE2026_CLAIM = mp.mpf("1.2802")

# The upper side of the sandwich (best published is MV2010):
MV_UPPER_PUBLISHED = mp.mpf("1.50992")

# Best unpublished upper bound on Tao's table at time of writing.
TTT_DISCOVER_UPPER = mp.mpf("1.5029")

# The bound applies to GENERAL non-negative f, NOT restricted to even f.
MV_CLASS = "general non-negative f (NOT restricted to even)"
CS2017_CLASS = "general non-negative f (NOT restricted to even)"
XIE2026_CLASS = "general non-negative f (claimed; unpublished)"


# ---------------------------------------------------------------------------
# Matolcsi-Vinuesa test function family
# ---------------------------------------------------------------------------
#
# MV do not give a closed-form g.  Their g is a trig polynomial
#     G(x) = sum_{j=1}^{119} a_j cos(2 pi j x / u),      u = 0.638,
# paired with the auxiliary arcsine-autoconvolution kernel
#     K(x) = (1/delta) * (beta * beta)(x/delta),        delta = 0.138,
#     beta(x) = (2/pi) / sqrt(1 - 4 x^2)  on (-1/2, 1/2).
# The 119 coefficients a_j are in the paper's Appendix and must be read
# off numerically (they are the output of a Mathematica QP).
#
# We expose the structural pieces so downstream code can (a) plug in the
# tabulated a_j once someone transcribes them, and (b) experiment with the
# same family of ansatz functions.

MV_U = mp.mpf("0.638")
MV_DELTA = mp.mpf("0.138")
MV_NUM_TERMS = 119


def mv_beta(x):
    """Arcsine density on (-1/2, 1/2): beta(x) = (2/pi) / sqrt(1 - 4 x^2).

    Returns 0 outside the open interval.
    """
    x = mp.mpf(x)
    if x <= mp.mpf("-0.5") or x >= mp.mpf("0.5"):
        return mp.mpf(0)
    return mp.mpf(2) / mp.pi / mp.sqrt(mp.mpf(1) - 4 * x * x)


def mv_K(x, delta=MV_DELTA):
    """Kernel K(x) = (1/delta) * (beta * beta)(x/delta).

    The convolution (beta * beta)(y) is supported on (-1, 1).  We evaluate
    it by mpmath quadrature; callers that need many evaluations should
    cache the result.
    """
    x = mp.mpf(x)
    delta = mp.mpf(delta)
    y = x / delta
    # support of beta * beta is (-1, 1)
    if y <= mp.mpf(-1) or y >= mp.mpf(1):
        return mp.mpf(0)
    # integrand beta(t) * beta(y - t); effective t-range is the intersection
    # of (-1/2, 1/2) with (y - 1/2, y + 1/2).
    a = max(mp.mpf("-0.5"), y - mp.mpf("0.5"))
    b = min(mp.mpf("0.5"),  y + mp.mpf("0.5"))
    if a >= b:
        return mp.mpf(0)
    val = mp.quad(lambda t: mv_beta(t) * mv_beta(y - t), [a, b])
    return val / delta


def mv_G(x, coeffs, u=MV_U):
    """MV trigonometric polynomial  G(x) = sum_j a_j cos(2 pi j x / u).

    Parameters
    ----------
    x       : real scalar (mpmath-compatible).
    coeffs  : sequence of mpmath-convertible a_1, ..., a_n.
    u       : MV period parameter (default 0.638).

    NOTE: The 119 coefficients from arXiv:0907.1379 Appendix must be
    supplied by the caller; we do not ship them here because they have
    not been transcribed into this project.
    """
    x = mp.mpf(x)
    u = mp.mpf(u)
    total = mp.mpf(0)
    for j, a in enumerate(coeffs, start=1):
        total += mp.mpf(a) * mp.cos(2 * mp.pi * j * x / u)
    return total


# ---------------------------------------------------------------------------
# Structured summary accessible to the rest of the codebase
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LowerBoundRecord:
    value: mp.mpf
    paper: str
    arxiv: str
    year: int
    published: bool
    class_of_f: str
    method: str
    notes: str


MV2009 = LowerBoundRecord(
    value=MV_BOUND,
    paper="Matolcsi, Vinuesa. Improved bounds on the supremum of autoconvolutions. J. Math. Anal. Appl. 372 (2010) 439-447.",
    arxiv="arXiv:0907.1379",
    year=2010,
    published=True,
    class_of_f=MV_CLASS,
    method="Delsarte-style dual; numerical trig poly G(x) = sum a_j cos(2 pi j x / u), paired with arcsine autoconvolution kernel K; 119 coefficients from Mathematica QP; theoretical ceiling ~1.276.",
    notes="Test function is NOT in closed form; coefficients are tabulated in the paper's Appendix.",
)

CS2017 = LowerBoundRecord(
    value=CS2017_BOUND,
    paper="Cloninger, Steinerberger. On suprema of autoconvolutions with an application to Sidon sets. Proc. AMS 145 (2017) 3191-3200.",
    arxiv="arXiv:1403.7988",
    year=2017,
    published=True,
    class_of_f=CS2017_CLASS,
    method="Branch-and-prune over step-function discretisations (the 'cascade' method).",
    notes="This is the current best rigorously published lower bound.",
)

XIE2026 = LowerBoundRecord(
    value=XIE2026_CLAIM,
    paper="Xie, Xinyuan. Unpublished improvement to the lower bound for C_{1a} (claiming C_{1a} >= 1.2802).",
    arxiv="(no arXiv; referenced on https://teorth.github.io/optimizationproblems/constants/1a.html via a Grok chat)",
    year=2026,
    published=False,
    class_of_f=XIE2026_CLASS,
    method="Unknown; no proof certificate archived as of 2026-04-19.",
    notes="The '1.2802' target in this project's CLAUDE.md comes from here, NOT from Matolcsi-Vinuesa.",
)


ALL_RECORDS = (MV2009, CS2017, XIE2026)


if __name__ == "__main__":
    mp.mp.dps = 30
    for rec in ALL_RECORDS:
        print(f"{rec.value}  {rec.arxiv:25s}  published={rec.published}  -- {rec.paper.split('.')[0]}")
