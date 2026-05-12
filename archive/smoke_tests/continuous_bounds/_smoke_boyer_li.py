"""
_smoke_boyer_li.py
==================

Boyer & Li (2025), arXiv:2506.16750: "An improved example for an autoconvolution
inequality."

WHAT THE PAPER PROVES
---------------------
Define the Holder-ratio constant
                  ||f*f||_{L^2}^2
    c := sup  ----------------------------       (1)
         f>=0  ||f*f||_{L^inf} ||f*f||_{L^1}

(supremum over nonneg f in L^1 cap L^2; ratio is translation/dilation invariant).

By Holder we have c <= 1 (with equality iff f*f is an indicator).  Boyer & Li
exhibit a 575-step nonneg step function realizing

    c >= 0.901564 = 9608608941040884028933976 9 / 10657701345143154524235494 4.

This improves AlphaEvolve (50 steps, 0.8962) and Matolcsi-Vinuesa (20 steps,
0.88922).  Their construction comes from simulated-annealing + gradient ascent
on the discretized objective Q_N(v) (eq. (3) in the paper).

WHAT THE PAPER DOES *NOT* PROVIDE
---------------------------------
*  No explicit lower bound on max_t (f*f)(t) over nonneg f supported on
   [-1/4, 1/4] with ||f||_{L^1}=1 (= the Cloninger-Steinerberger constant
   C_{1a}, currently bounded as 1.2802 <= C_{1a} <= 1.5029).
*  No new upper bound on C_{1a}.
*  No LP/SDP certificate that we can re-use as a per-composition filter in the
   cascade.  Their Q_N(v) is computed directly from a candidate v; it produces
   numerical *lower bounds for c*, not for C_{1a}, and only for one specific
   sequence v (a witness, not a certificate over all f).

CAN WE USE BOYER-LI'S WITNESS?
------------------------------
The paper's optimizer F_3 (575-step) is a nonneg step function on [-1/4, 1/4]
with max(F_3 * F_3) = 1 by their normalization (Section 3, "we ... normalize
the maximum of f_i*f_i to be 1").  Its L^1 norm ||F_3||_{L^1} can be read off:

    ||F_3 * F_3||_{L^inf} = 1   =>   we want max(f*f)/||f||_{L^1}^2.

If ||F_3||_{L^1} = M, then for the rescaled g = F_3 / M we have
||g||_{L^1} = 1 supported on [-1/4, 1/4], and
max(g*g) = max(F_3*F_3) / M^2 = 1 / M^2.

So the paper's witness gives an UPPER BOUND  C_{1a} <= 1 / M^2.  The question
is whether 1/M^2 is below the current 1.5029.

This script reads the integer coefficients in coeffBL.txt (if present in
github.com/zkli-math/autoconvolutionHolder) and computes 1/M^2.  Without
network access we cannot fetch that file; we instead document what would need
to be done.

CONCLUSION
----------
The Boyer-Li bound is on a *different* problem (Holder ratio c, not the sup
norm autoconvolution constant C_{1a}).  Their witness F_3 could in principle
yield an UPPER bound 1/M^2 on C_{1a}, but the user's mission is to push the
LOWER bound above 1.2802 -- so this paper is orthogonal to the cascade.

Their *algorithmic* idea (gradient ascent on a discretized step-function
objective) is the same idea Cloninger-Steinerberger and Matolcsi-Vinuesa
already use for the C_{1a} upper bound; it does not yield a lower-bound
certificate.

Verdict: NOT_DIRECTLY_USABLE for tightening the cascade lower bound.

This file is a smoke / scaffold: it (a) computes the paper's c-ratio for a
toy nonneg step function (sanity check), (b) shows the wrong-problem error
for the C_{1a} cascade.  No F-survivors are exercised because the bound is
not a per-composition filter.
"""

from __future__ import annotations

import numpy as np


def autoconv_step_eval(v: np.ndarray) -> tuple[float, float, float]:
    """For f = sum_n v_n * 1_{[n, n+1]}, return (||f*f||_inf, ||f*f||_1,
    ||f*f||_{L^2}^2).

    Implements eqs. (2) and (3) of Boyer-Li.
    """
    v = np.asarray(v, dtype=float)
    N = len(v)
    # L_j = sum_{n=max(0,j-N+1)}^{min(j,N-1)} v_n v_{j-n}, j = 0..2N-2
    # but the paper uses index shift: (f*f)(j) for j=0..2N-1, with the
    # convention (1_{[n,n+1]} * 1_{[m,m+1]})(x) is a tent peaking at m+n+1.
    # Following eq. (2):  L_j = sum_{n+m = j-1} v_n v_m, j=0..2N-1.
    L = np.zeros(2 * N)
    for j in range(2 * N):
        s = 0.0
        for n in range(max(0, j - N), min(j, N - 1) + 1):
            m = j - 1 - n
            if 0 <= m < N:
                s += v[n] * v[m]
        L[j] = s
    Linf = float(np.max(L))
    L1 = float(0.5 * np.sum(L[:-1] + L[1:]))  # 1/2 sum (L_j + L_{j+1})
    L2sq = float((1.0 / 3.0) * np.sum(L[:-1] ** 2 + L[:-1] * L[1:] + L[1:] ** 2))
    return Linf, L1, L2sq


def boyer_li_ratio(v: np.ndarray) -> float:
    """Compute  ||f*f||_{L^2}^2 / (||f*f||_inf * ||f*f||_1)  for step f."""
    Linf, L1, L2sq = autoconv_step_eval(v)
    return L2sq / (Linf * L1)


def smoke_paper_examples() -> None:
    """Sanity check: small nonneg vectors should give c-ratio in (0, 1]."""
    print("Smoke test: Boyer-Li c-ratio on toy step functions")
    print("-" * 60)
    examples = [
        ("indicator of [0,1]", np.array([1.0])),
        ("indicator of [0,3]", np.array([1.0, 1.0, 1.0])),
        ("triangle 1,2,1", np.array([1.0, 2.0, 1.0])),
        ("MV-like 4-step", np.array([0.6, 0.4, 0.5, 0.6])),
    ]
    for name, v in examples:
        c = boyer_li_ratio(v)
        print(f"  {name:30s}  c = {c:.6f}")
    print()
    print("Reference (paper Theorem 1):  c >= 0.901564  (575-step optimizer).")


def report_relation_to_C1a() -> None:
    print("=" * 70)
    print("Boyer-Li (2025) and the cascade C_{1a} >= 1.2802 lower bound")
    print("=" * 70)
    print(
        """
The paper bounds c = sup ||f*f||_2^2 / (||f*f||_inf ||f*f||_1)  (eq. (1)).

The cascade target is C_{1a} = inf over nonneg f on [-1/4,1/4], int f = 1, of
    max_{|t|<=1/2} (f*f)(t).

These constants are linked only loosely:

  * Cauchy-Schwarz:  ||f*f||_2^2 <= ||f*f||_inf * ||f*f||_1, hence c <= 1.
  * The Boyer-Li witness yields an UPPER bound on C_{1a} (witness, not
    certificate); the user's mission is the LOWER bound, so it does not help.
  * Boyer-Li's Q_N(v) (eq. (3)) evaluates a single candidate v; it is NOT
    a per-composition lower-bound filter.  No min-over-f duality is invoked.

Per-composition filter feasibility:
  In the cascade, each composition C corresponds to a polytope of admissible
  nonneg vectors (a >= 0, sum a = m, ...).  Boyer-Li does not give a lower
  bound on max(f*f) valid uniformly over such a polytope; their construction
  is purely a witness for one v.

So this paper is ORTHOGONAL to the cascade-lower-bound program.
"""
    )


if __name__ == "__main__":
    smoke_paper_examples()
    report_relation_to_C1a()
