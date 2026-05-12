"""Chebyshev-basis Lasserre + Fejer-Riesz periodic dual for the Sidon
autocorrelation constant.

Pipeline (see top-level prompt / PROJECT docs):

  Primal:  f parametrized by Chebyshev moments c_k = int T_k(4x) f(x) dx
           on [-1/4, 1/4].  Well-conditioned basis; Hausdorff PSD blocks
           H_1(c), H_2(c) via change-of-basis from monomial Hankel.
  Dual:    nonneg measure p on T = R/Z parametrized by cosine trig
           polynomial r, PSD Toeplitz constraint R(r) >= 0
           (Caratheodory-Toeplitz / Fejer-Riesz).
  Kernel:  <f*f, p> = c^T A(r) c + truncation_err, with A(r) linear in r
           via Jacobi-Anger expansion of the trig kernel.
  SDP:     single SDP combining primal + dual via inner-problem dualization.
  Cert:    exact-rational verification via python-flint fmpq + Arb balls.
"""
