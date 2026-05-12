"""Dual / moment LP reformulation of the Sidon Polya/Handelman LP.

Motivation: the primal has ~n_q+1 free variables (alpha + all q_K), which
makes PDLP-class first-order solvers structurally infeasible. The LP dual
has the SAME optimal value (LP duality) but only 1 free variable
(y_simplex), so PDLP becomes viable.

Modules:
  build_dual.py   -- construct A_eq, A_ub, c, bounds for the dual LP.
  solve_dual.py   -- MOSEK solver for the dual (soundness baseline) +
                     PDLP-on-dual via tier4.pdlp_robust (the actual
                     candidate for GPU acceleration).
"""
