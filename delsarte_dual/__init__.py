"""The Piterbarg-Bajaj-Vincent Bound: a rigorous lower bound on the
Sidon autocorrelation constant.

This package implements the multi-scale arcsine extension of the
Matolcsi-Vinuesa dual framework that certifies

    ``C_{1a} := inf { ||f * f||_inf / (int f)^2
                       : f >= 0,  supp f subset (-1/4, 1/4),  int f > 0 }
              >= 1292 / 1000  =  1.292``.

The accompanying writeup *Improving the Bounds on the Supremum of
Autoconvolutions* is at the repository root in
``lower_bound_proof.pdf`` (LaTeX source ``lower_bound_proof.tex``); the
Lean 4 formalisation of the same bound is at ``lean/Sidon/MultiScale.lean``.

Every transcendental output of the pipeline is a rigorous ``flint.arb``
interval at 256-bit precision; every algebraic input is an exact
``flint.fmpq``.

Public entry points
-------------------

.. code-block:: python

    from delsarte_dual.grid_bound_alt_kernel.kernels import (
        ArcsineKernel,
        MultiScaleArcsineKernel,
    )
    from delsarte_dual.grid_bound_alt_kernel.optimize_G import (
        solve_qp_for_kernel,
    )
    from delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel import (
        production_kernel,
        compile_phi_params_for_kernel,
        run_single_kernel,
    )

Command-line entry points
-------------------------

    ``python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel``
        Run the full production pipeline and emit a JSON certificate of
        ``C_{1a} >= 1292/1000``.

    ``python -m delsarte_dual.grid_bound.certify <certificate.json>``
        Independent verifier: re-checks every quantitative claim of the
        certificate using only ``flint`` primitives, with no imports
        from the rest of the package.

See ``README.md`` for the full pipeline description.
"""
