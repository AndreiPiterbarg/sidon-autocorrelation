# Single-kernel Cohn-Elkies sweep

**Status:** DEAD (capped). The framework is what Matolcsi-Vinuesa 2010
already use; their bound $\approx 1.2748$ is essentially the ceiling
for *single* kernels.

## Framework

The Cohn-Elkies-style master inequality
$$ C_{1a} \;\ge\; \frac{\int_{\mathbb R} \widehat g(\xi)\,w(\xi)\,d\xi}{\max_{[-1/2,1/2]} g} $$
holds for any test $g \ge 0$ on $[-1/2, 1/2]$ with $\widehat g \ge 0$ on
$\mathbb R$, with the Paley-Wiener envelope $w(\xi) = \cos^2(\pi \xi/2)$
on $|\xi| \le 1$. MV's 2010 bound at $1.2748$ uses $g$ built from a
single-scale arcsine kernel together with a $119$-mode Fourier-positive
trigonometric polynomial.

## What was tried

Two independent searches inside the single-kernel family:

1. *31-kernel Bochner-admissible sweep* across alternative compactly
   supported families: Chebyshev-$\beta$, Wendland, Hertz-tilde,
   B-spline of orders $2$-$6$, Askey-$\nu$, tent / triangle, raised
   cosine, Tukey window, half-cosine, hyperbolic-secant truncations,
   plus scale variants. Each is Bochner-admissible (verified via
   Fourier-side non-negativity) and supported on $[-\delta, \delta]
   \subset [-1/4, 1/4]$.

2. *QP-reoptimised $G$ against a single-scale $\widehat K$* at the MV
   parameter $\delta = 138/1000$, with $N = 119$ modes and rationalised
   coefficients (denominator $10^{8}$). Sweep over $\delta$ in
   $\{0.025, 0.034, 0.043, 0.052, 0.060\}$ and a $5 \times 5$
   neighbourhood of MV's operating point.

## Outcome

| Search | Best $M_{\rm cert}$ (rigorous) | Improvement over MV |
|---|---:|---:|
| 31-kernel single-family sweep | $\le 1.276$ | $\le +0.001$ |
| Re-optimised $G$ at $\delta = 138/1000$ | $1.27629$ | $+0.0015$ |
| Re-optimised $G$ across $\delta$ sweep | $1.276287$ (at $\delta_2 = 0.052$, $\lambda_1 = 0.95$) | $+0.0015$ |

The single-kernel framework appears bounded by $\sim 1.276$ from
above; no candidate in either search exceeds it. MV's own remark
(p. 5, lines 232-261) computes this ceiling numerically as
$\inf_{f_s} \|f_s * \eta_\delta\|_2^2 \approx 1.276$ at $\delta
\approx 0.14$ (an informal observation, not a theorem; see
[`multiscale_arcsine.md`](multiscale_arcsine.md) for the "MV ceiling"
discussion).

## Why single-kernel is capped

The QP gain $a = (4/u) m_G^2 / S_1$ is starved by the Bessel-zero
behaviour of $\widehat K = J_0(\pi \delta \xi)^2$: $\widetilde K(j)$
touches $\sim 10^{-6}$ near $j/u \approx z_1/(\pi \delta) \approx 5.55$,
and the reciprocal in $S_1$ blows up. Other single-family kernels
exhibit similar zero or near-zero structures (sinc-squared zeros for
tent, Gibbs ringing for windowed Gaussians); none simultaneously
achieves non-vanishing $\widetilde K(j)$ across all dual frequencies
and a tight $L^2$ norm. G re-optimisation alone drops $S_1$ from
$\sim 87$ to $\sim 50$ but is partially cancelled by the Minkowski
upper bound on $K_2$ used in the rigorous certificate.

## Implication

The only way past the $\sim 1.276$ single-kernel ceiling is *convex
combinations* of admissible kernels at different scales. Each property
of admissibility ($K \ge 0$, support, unit mass, Bochner positivity)
is closed under convex combinations, but the structural identity
$\widetilde K(j) = u \,\widetilde\eta(j)^2$ MV use is not, opening the
gap. This is the multi-scale arcsine line; see
[`multiscale_arcsine.md`](multiscale_arcsine.md).

## References

- H. Cohn, N. Elkies, *New upper bounds on sphere packings I*, Annals of Mathematics 157 (2003), [arXiv:math/0110009](https://arxiv.org/abs/math/0110009).
- M. Matolcsi, C. Vinuesa (2010), [arXiv:0907.1379](https://arxiv.org/abs/0907.1379), Remark p. 5, lines 232-261.
- G. Martin, K. O'Bryant (2009), [arXiv:0807.5121](https://arxiv.org/abs/0807.5121).
- See [`multiscale_arcsine.md`](multiscale_arcsine.md), [`master_attacks.md`](master_attacks.md).
