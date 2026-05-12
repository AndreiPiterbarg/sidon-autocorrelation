"""Path B: Bochner-SOS dual certificate for C_{1a} >= 1.2805.

Modules:
  m_g_eval        : numerical evaluator for M(g) = inf_f int int g(x+y) f f / int g
  g_candidates    : candidate dual functions (CS-style, polynomial, B-spline, etc.)
  run_explore     : exploratory driver
  optimize_g      : numerical optimization of g parameters (week 2)
  build_sdp       : moment-SOS hierarchy SDP (week 2-3)
  round_rational  : rational rounding of SDP solution (week 3)
  verify_sos      : exact-arithmetic SOS verifier (week 4)

Main reference: Cloninger-Steinerberger 2017, arXiv:1403.7988 .
The dual statement is

  C_{1a} >= M(g) := inf_{f >=0, supp f c [-1/4,1/4], int f = 1}
                       [int int g(x+y) f(x) f(y) dx dy] / int g

for any nonneg g on [-1/2, 1/2] (we generally take g supported on a
sub-interval of [-1/2, 1/2]).  CS 2017 used a specific g and obtained
M(g) ~= 1.2802; we aim to beat this by enlarging the function class for g.
"""

__version__ = "0.1.0"
