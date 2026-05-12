"""Multi-scale arcsine kernel and production driver for the
Piterbarg-Bajaj-Vincent Bound.

This subpackage extends the single-scale Matolcsi-Vinuesa framework in
``delsarte_dual.grid_bound`` with the convex-combination kernel

    ``K = sum_i lambda_i K_arc(delta_i; .)``

and a per-kernel QP that re-optimises the cosine multiplier ``G``.  At
the writeup parameters

    ``(delta_1, delta_2, delta_3) = (138, 55, 25) / 1000``,
    ``(lambda_1, lambda_2, lambda_3) = (85, 10, 5) / 100``,
    ``N = 200``,

it certifies ``C_{1a} >= 1292/1000 = 1.292`` (the
Piterbarg-Bajaj-Vincent Bound).

Top-level entry points:

  * :class:`kernels.MultiScaleArcsineKernel` -- kernel definition.
  * :func:`optimize_G.solve_qp_for_kernel`   -- semi-infinite QP solver.
  * :func:`bisect_alt_kernel.run_single_kernel` -- full pipeline.
  * ``python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel``
    -- command-line entry point that emits a JSON certificate.
"""
