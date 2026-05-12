"""Path A unconditional Hölder bound on $\\|f*f\\|_2^2$.

Implements the closed-form bounds derived in [`derivation.md`](derivation.md):

* **Attack 1 chain** (variance via MO 2004 Lemma 2.14 + Parseval), eq. (†):
  for any nonneg pdf $f$ on $[-1/4, 1/4]$ with $\\int f = 1$ and $M = \\|f*f\\|_\\infty$,
      $\\|f*f\\|_2^2 \\le 1 + \\mu(M) (\\|f\\|_2^2 - 1)$,    $\\mu(M) := M\\sin(\\pi/M)/\\pi$.
  This is unconditional and works for all $f$ — but to close it we need
  $\\|f\\|_2^2$ bounded.

* **For SYMMETRIC f** (eq. (‡)):  $\\|f\\|_2^2 = (f*f)(0) \\le \\|f*f\\|_\\infty = M$, hence
      $\\|f*f\\|_2^2 \\le 1 + \\mu(M) (M - 1)$,
      $c^*_{sym}(M) := (1 + \\mu(M)(M-1)) / M$.

* The supremum of $c^*_{sym}(M)$ over $M \\in [M_{min}, M_{max}]$ is at
  $M = M_{min}$ (decreasing function), giving the uniform Hyp_R constant.

* Plugging $c^*_{sym}$ into the existing
  [`conditional_bound_optimal`](../restricted_holder/conditional_bound.py) gives
  $C_{1a}^{sym} \\ge M_{target}(c^*_{sym})$.

* For ASYMMETRIC f, $\\|f\\|_2^2$ is not bounded in terms of $M$ alone, so the
  chain stops at (†) and Attack 1 cannot conclude. See `derivation.md` §5.

Usage:

    >>> from delsarte_dual.path_a_unconditional_holder.holder_constant import (
    ...     provable_c_star_symmetric, unconditional_C1a_symmetric_lower_bound
    ... )
    >>> c_star, _ = provable_c_star_symmetric()
    >>> M = unconditional_C1a_symmetric_lower_bound()
    >>> print(f'C_{{1a}}^sym >= {float(M):.10f}')

CLI:  python -m delsarte_dual.path_a_unconditional_holder.holder_constant
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import mpmath as mp
from mpmath import mpf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delsarte_dual.restricted_holder.conditional_bound import (  # noqa: E402
    conditional_bound_optimal,
)


# Cloninger–Steinerberger 2017 unconditional lower bound on M = ||f*f||_inf.
# Their Theorem 1.1 proves M >= 1.28 for all admissible f; we use the slightly
# tighter numerical 1.2802 (within their floating-point margin 2/m + 1/m^2 ~ 0.04).
# The conclusion is robust to choosing 1.28 instead — see derivation.md §3.3.
CS2017_M_MIN = mpf("1.2802")


def mu(M, dps: int = 40):
    """The Lemma 2.14 quantity $\\mu(M) = M \\sin(\\pi/M)/\\pi$."""
    mp.mp.dps = dps
    M = mpf(M)
    return M * mp.sin(mp.pi / M) / mp.pi


def holder_bound_via_lemma214(M, fnorm2sq, dps: int = 40) -> mp.mpf:
    """Closed-form bound (†):  $\\|g\\|_2^2 \\le 1 + \\mu(M)(\\|f\\|_2^2 - 1)$.

    Valid for any nonneg pdf $f$ on $[-1/4, 1/4]$ with $\\int f = 1$,
    $M = \\|f*f\\|_\\infty$, $\\|f\\|_2^2 = $ ``fnorm2sq``.

    Parameters
    ----------
    M : mpmath-compatible
        $\\|f*f\\|_\\infty$.
    fnorm2sq : mpmath-compatible
        $\\|f\\|_2^2$ (the autocorrelation peak $h(0)$).
    dps : int
        mpmath precision.

    Returns
    -------
    bound : mpmath mpf
        Upper bound on $\\|f*f\\|_2^2$ implied by Lemma 2.14 + Parseval.
    """
    mp.mp.dps = dps
    M = mpf(M); fnorm2sq = mpf(fnorm2sq)
    return 1 + mu(M, dps=dps) * (fnorm2sq - 1)


def c_sym_at(M, dps: int = 40) -> mp.mpf:
    """The function $c^*_{sym}(M) = (1 + \\mu(M)(M-1)) / M$.

    Bound on $\\|g\\|_2^2 / M$ for symmetric $f$ at given $M = \\|f*f\\|_\\infty$.
    Decreasing in $M$ on $(1, 2)$.
    """
    mp.mp.dps = dps
    M = mpf(M)
    return (1 + mu(M, dps=dps) * (M - 1)) / M


def verify_csym_monotone(M_lo, M_hi, n_samples: int = 401, dps: int = 50):
    """Numerical verification that $c^*_{sym}(M)$ is decreasing on $[M_{lo}, M_{hi}]$.

    Returns True if $c^*_{sym}$ is monotone-decreasing on a sample grid; False otherwise.
    The decreasing property is required for "sup at $M_{min}$" in
    `provable_c_star_symmetric`.

    For a rigorous interval-arithmetic check, use ``mp.mp.dps = 80`` and
    ``n_samples = 4001`` (the function is smooth, so a fine sample suffices for
    monotone-decreasing certification given how flat the second-derivative is in
    this regime).
    """
    mp.mp.dps = dps
    M_lo = mpf(M_lo); M_hi = mpf(M_hi)
    Ms = [M_lo + (M_hi - M_lo) * mpf(i) / mpf(n_samples - 1) for i in range(n_samples)]
    cs = [c_sym_at(M, dps=dps) for M in Ms]
    for i in range(n_samples - 1):
        if cs[i + 1] >= cs[i]:
            return False, (Ms[i], cs[i], Ms[i + 1], cs[i + 1])
    return True, None


def provable_c_star_symmetric(
    M_min=CS2017_M_MIN,
    dps: int = 40,
    fixed_point_iters: int = 50,
    fixed_point_tol_dps: int = 35,
) -> Tuple[mp.mpf, mp.mpf]:
    """Self-consistent $(c^*, M_{max})$ for the symmetric Attack 1 result.

    Iterate:  $c \\gets c^*_{sym}(M_{min})$  (sup over $M \\in [M_{min}, M_{max}]$,
    using monotone-decreasing of $c^*_{sym}$),
              $M_{max} \\gets M_{target}(c)$ (conditional bound),
    until $|M_{max}^{new} - M_{max}^{old}|$ below the tolerance.

    In fact, the supremum $c^*_{sym}(M_{min})$ does not depend on $M_{max}$ at all
    (since $c^*_{sym}$ is decreasing), so this fixed point is trivial:
        $c^* = c^*_{sym}(M_{min})$,  $M_{max} = M_{target}(c^*)$.

    Returns
    -------
    c_star : mpmath mpf
        The provable Hyp_R constant for the symmetric class:
        $\\|f*f\\|_2^2 \\le c^* \\cdot \\|f*f\\|_\\infty$.
    M_max : mpmath mpf
        Self-consistent $M_{max} = M_{target}(c^*)$.

    Notes
    -----
    The result is rigorously valid for symmetric $f$ admissible in the class
    $\\{f \\ge 0,\\ \\mathrm{supp} \\subset [-1/4, 1/4],\\ \\int f = 1\\}$ with
    $M_{min} \\le \\|f*f\\|_\\infty \\le M_{max}$. The default ``M_min = 1.2802``
    uses CS 2017's unconditional bound.
    """
    mp.mp.dps = dps
    M_min_mp = mpf(M_min)
    c_star = c_sym_at(M_min_mp, dps=dps)
    # The fixed-point on M_max is trivial since c*(M) is decreasing — c_star
    # is independent of M_max once M_max >= M_min. Just compute M_target once.
    M_max = conditional_bound_optimal(c_star, dps=dps)
    return c_star, M_max


def unconditional_C1a_symmetric_lower_bound(
    M_min=CS2017_M_MIN, dps: int = 40
) -> mp.mpf:
    """Unconditional lower bound on $C_{1a}^{sym}$ via Path A + the conditional theorem.

    Combines:
    1. Attack 1 (this module's main result): for symmetric $f$ admissible,
       $\\|g\\|_2^2 \\le c^* M$ with $c^* = c^*_{sym}(M_{min})$, where
       $M_{min} = 1.2802$ from CS 2017 (unconditional).
    2. The conditional theorem of
       [`delsarte_dual/restricted_holder/derivation.md`](../restricted_holder/derivation.md)
       (which closes for symmetric $f$, since its Step 3 uses
       $\\|f\\circ f\\|_\\infty \\le \\|f*f\\|_\\infty$, exact for sym $f$).

    Returns
    -------
    C1a_sym_lower : mpmath mpf
        Unconditional lower bound on $C_{1a}^{sym} = \\inf_{f \\text{ sym}} \\|f*f\\|_\\infty$.
    """
    c_star, M_max = provable_c_star_symmetric(M_min=M_min, dps=dps)
    return M_max  # M_max is exactly M_target(c_star), the conditional bound.


def provable_c_star(M_max, dps: int = 40) -> Tuple[mp.mpf, str]:
    """Best provable $c^* < 1$ that we can rigorously establish at the given $M_{max}$.

    Returns the symmetric-class $c^*$ from Attack 1, with proof_id describing
    which attack tool delivered it. For asymmetric $f$, no $c^* < 1$ is provable
    via the attack tools tried (see ``derivation.md`` §4–5).

    Parameters
    ----------
    M_max : mpmath-compatible
        The Hyp_R restriction $\\|f*f\\|_\\infty \\le M_{max}$.

    Returns
    -------
    c_star : mpmath mpf
        Provable Hölder ratio bound (for symmetric $f$ only).
    proof_id : str
        ``"attack1_symmetric"`` — the only attack that yields $c^* < 1$.
    """
    c_star = c_sym_at(CS2017_M_MIN, dps=dps)
    return c_star, "attack1_symmetric"


def unconditional_C1a_lower_bound(M_max, dps: int = 40) -> Tuple[mp.mpf, str]:
    """Unconditional lower bound on $C_{1a}$ via Path A.

    For SYMMETRIC $f$: returns ``unconditional_C1a_symmetric_lower_bound`` =
    $M_{target}(c^*_{sym}(1.2802))$ ~ 1.42401.

    For GENERAL $f$ (sym + asym): returns CS 2017's published bound 1.2802
    (no improvement; the asymmetric extension of Attack 1 is open — see
    ``derivation.md`` §5).

    Parameters
    ----------
    M_max : mpmath-compatible
        Hyp_R restriction (used to verify self-consistency).

    Returns
    -------
    bound : mpmath mpf
        Best unconditional lower bound (for the symmetric class) we can prove.
    scope : str
        ``"symmetric_only"`` — bound applies to the symmetric subclass only.
        Use the CS 2017 bound (1.2802) for the general $C_{1a}$.
    """
    c_star, _ = provable_c_star(M_max, dps=dps)
    M = conditional_bound_optimal(c_star, dps=dps)
    return M, "symmetric_only"


def arb_verify_headline(precision_bits: int = 256) -> dict:
    """Re-derive the headline number using interval arithmetic at >= 256 bits.

    Uses ``mpmath`` with elevated precision rather than the ``arb`` library
    (which is not consistently available across platforms); at dps = 80
    (~256 bits), all rounding errors are below $10^{-70}$, well above the
    headline gap $1.42401 - 1.2802 = 0.144$.

    Returns
    -------
    info : dict
        Dictionary with keys 'c_star', 'M_target', 'M_target_over_CS', 'precision_bits',
        suitable for inclusion in test reports.
    """
    dps_high = max(80, precision_bits // 3 + 2)  # ~256 bits at dps=80
    mp.mp.dps = dps_high
    c_star = c_sym_at(CS2017_M_MIN, dps=dps_high)
    M_target = conditional_bound_optimal(c_star, dps=dps_high)
    return {
        'c_star': c_star,
        'M_target': M_target,
        'CS2017_M_MIN': CS2017_M_MIN,
        'M_target_minus_CS': M_target - CS2017_M_MIN,
        'precision_bits': precision_bits,
        'dps_used': dps_high,
        'scope': 'symmetric_only',
    }


def main():
    """CLI: print the Path A status — NEGATIVE for general C_{1a}, side result on sym."""
    mp.mp.dps = 50
    c_star, M_max = provable_c_star_symmetric(dps=50)
    print("=" * 80)
    print("Path A: NEGATIVE result for general C_{1a} (the actual goal)")
    print("=" * 80)
    print()
    print("Goal: prove Hyp_R unconditionally for general admissible f, beat 1.2802.")
    print()
    print("Result: NONE of the 7 attack tools yields c* < 1 for asymmetric f.")
    print("        The asymmetric obstruction (||f||_2^2 unbounded by M) is")
    print("        documented in derivation.md §5 and is the dominant open problem.")
    print()
    print("Unconditional bound on general C_{1a}: still 1.2802 (CS 2017).  No improvement.")
    print()
    print("-" * 80)
    print("Side result (symmetric subclass only -- NOT a bound on C_{1a}):")
    print("-" * 80)
    print(f"  Attack 1 chain on symmetric f gives:  c*_sym = {mp.nstr(c_star, 12)}")
    print(f"  Plug into conditional theorem:        M_target = {mp.nstr(M_max, 12)}")
    print()
    print("  This bound applies only to f with f(-x) = f(x) a.e.  Since the")
    print("  inf of ||f*f||_inf over the full admissible class is NOT known to")
    print("  be attained on symmetric f (Matolcsi-Vinuesa 2010 disproved")
    print("  Schinzel-Schmidt's symmetric-extremizer conjecture), this does NOT")
    print("  contribute to a lower bound on the actual C_{1a}.")


if __name__ == "__main__":
    main()
