"""Rigorous sharper bathtub bound on |hat h(n)| = z_n^2 for h = f*f.

STATUS: OBSTRUCTED (see ``derivation.md``).

We did NOT derive a provable quantitative sharpening of Matolcsi-Vinuesa's
Lemma 3.4. Qualitatively, mu_sharper(M, n) < mu_MV(M) strictly for all n >= 1
(Corollary 2.3 of derivation.md: the MV bathtub extremizer M * 1_{A*} is
discontinuous, while every h = f*f with admissible f is continuous), but the
gap is not quantified.

To preserve soundness we ship the placeholder

    mu_sharper(M, n) := mu_MV(M) = M * sin(pi/M) / pi

i.e., unchanged from the MV bound. The returned value is ALWAYS a valid
(possibly non-tight) upper bound on |hat h(n)|, so downstream filters remain
sound. They also gain nothing over the MV baseline — which is the faithful
reporting of the current state of the proof.

When a rigorous quantitative sharpening is proved (see proof_paths.md Path 1
SDP route), replace the body of mu_sharper(...) with the proved formula.
"""
from __future__ import annotations

from flint import arb, ctx

from delsarte_dual.grid_bound.phi_mm import mu_of_M as _mu_MV


def mu_sharper(M: arb, n: int, prec_bits: int = 256) -> arb:
    """Rigorous (non-strict) sharper bathtub bound on |hat h(n)| = z_n^2.

    Parameters
    ----------
    M : arb
        Upper bound on ||f*f||_inf.  Must be `arb` at the requested prec_bits.
    n : int
        Fourier index (n >= 1).  Currently unused; the bound is n-independent
        because the MV bound mu_MV(M) is itself n-independent (Lemma 1 of
        multi_moment_derivation.md).
    prec_bits : int, default 256
        arb context precision for the computation.

    Returns
    -------
    arb enclosure of mu_MV(M) = M sin(pi/M) / pi.

    Notes
    -----
    CURRENTLY EQUAL TO mu_MV(M).  No strict improvement is proved; see
    derivation.md. The qualitative strict inequality mu_sharper < mu_MV is
    proved but not quantified. Shipping this as an equality preserves
    soundness at the cost of no added tightening.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        return _mu_MV(arb(M))
    finally:
        ctx.prec = old


__all__ = ["mu_sharper"]
