"""Matolcsi-Vinuesa master inequality and cell-search certifier.

This subpackage implements the rigorous Matolcsi-Vinuesa dual machinery:

  * :mod:`bessel`        -- arb wrappers for ``J_0(pi j delta / u)``.
  * :mod:`coeffs`        -- the 119 cosine coefficients of the original
                            Matolcsi-Vinuesa multiplier (kept for
                            reproducibility of the single-scale baseline).
  * :mod:`phi`           -- rigorous arb evaluation of the forbidden-
                            region function ``Phi(M, y)``.
  * :mod:`G_min`         -- Taylor branch-and-bound lower bound on
                            ``min_{[0, 1/4]} G(x)``.
  * :mod:`cell_search`   -- priority-queue cell bisection certifying
                            ``Phi(M, .) < 0`` over the admissible box.
  * :mod:`bisect`        -- bisection on ``M`` to find the largest
                            certifiable ``M_cert``.
  * :mod:`certify`       -- standalone independent verifier with no
                            internal imports.

Every transcendental output is a ``flint.arb`` interval; every
algebraic input is an exact ``flint.fmpq``.
"""
