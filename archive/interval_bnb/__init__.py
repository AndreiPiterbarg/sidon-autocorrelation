"""Interval branch-and-bound for rigorous lower bounds on val(d).

Certifies val(d) >= c for the discrete Sidon autocorrelation problem
    val(d) = min_{mu in Delta_d} max_{W in W_d} mu^T M_W mu
by partitioning Delta_d into closed boxes and bounding each with
interval arithmetic (natural extension + McCormick linear relaxation).

Unlike the Cloninger-Steinerberger cascade this method has no
discretisation correction: correctness relies only on interval-
arithmetic rounding, which we eliminate via exact rational replay
(fractions.Fraction) at each certified leaf.

Entry points:
    run_d14.py   -- top-level driver for d=14
    run_d10.py   -- pipeline validation at d=10
    bnb.branch_and_bound -- library API
"""
