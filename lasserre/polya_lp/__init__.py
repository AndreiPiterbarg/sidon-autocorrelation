"""Pólya / Krivine-Handelman LP hierarchy for the Sidon val(d) lower bound.

Replaces (or sanity-checks) the Lasserre SDP at fixed d with a sequence of
LPs of increasing degree R. The LP variables are:

  alpha            scalar lower bound on min_mu max_W mu^T M_W mu
  lambda_W >= 0    mixture over windows, sum to 1
  q_K              free polynomial multiplier of (1 - sum mu_i), |K| <= R-1
  c_beta >= 0      slack variables (one per |beta| <= R)

with equality constraints (one per |beta| <= R):

  [p_lambda(mu)]_beta - alpha * delta_{beta=0}
    + q_beta - sum_j q_{beta - e_j}        = c_beta

i.e.  mu^T M(lambda) mu - alpha  =  sum_beta c_beta mu^beta
                                    + q(mu) (1 - sum_i mu_i)

Soundness:
  Any feasible (alpha, lambda, q, c) gives val(d) >= alpha. Reason:
  on Delta_d the equality (1 - sum mu_i) = 0, so p_lambda - alpha
  equals sum c_beta mu^beta on Delta_d. This is a polynomial that is
  >= 0 on Delta_d (each mu^beta >= 0). Hence p_lambda >= alpha on
  Delta_d. Then min_mu p_lambda(mu) >= alpha, and val(d) =
  min_mu max_W mu^T M_W mu >= min_mu sum_W lambda_W mu^T M_W mu
  = min_mu p_lambda(mu) >= alpha.

Bells and whistles included:
  - Variable lambda jointly with alpha (vs fixed uniform)
  - Z/2 symmetry reduction (mu_i = mu_{d-1-i})
  - Sparse LP construction via scipy.sparse
  - HiGHS primary solver, MOSEK fallback
  - Multiple R sweeps with convergence diagnostics
"""

from lasserre.polya_lp.build import build_handelman_lp, BuildOptions, BuildResult
from lasserre.polya_lp.solve import solve_lp, SolveResult
from lasserre.polya_lp.symmetry import z2_symmetric_basis, project_M_to_z2

__all__ = [
    "build_handelman_lp",
    "BuildOptions",
    "BuildResult",
    "solve_lp",
    "SolveResult",
    "z2_symmetric_basis",
    "project_M_to_z2",
]
