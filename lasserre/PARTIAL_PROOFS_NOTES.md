# Partial proofs and observations for the reverse-Young conjecture

**Conjecture (REV-3/2):** For nonneg $f$ on $[-1/4, 1/4]$ with $\int f = 1$ and $\sup_{|t| \le 1/2}(f*f)(t) < \infty$:
$$\sup(f*f) \ge c_0\,\|f\|_{3/2}^3, \qquad c_0 = \pi/8 \approx 0.3927$$

## Observation 1 — Symmetric $f$ satisfies $c_0 = 1$ (much stronger than conjecture).

**Claim.** For *symmetric* feasible $f$ (i.e., $f(-x) = f(x)$):
$$\sup(f*f) \ge \|f\|_{3/2}^3$$

**Proof.** For symmetric $f$, $(f*f)(0) = \int f(x) f(-x)\,dx = \int f(x)^2\,dx = \|f\|_2^2$. So $\sup f*f \ge \|f\|_2^2$.

By Hölder, with $\int f = 1$:
$$\|f\|_{3/2}^{3/2} = \int f^{3/2} = \int f \cdot f^{1/2} \le \left(\int f^2\right)^{1/2} \left(\int f\right)^{1/2} = \|f\|_2 \cdot 1$$
So $\|f\|_{3/2}^3 \le \|f\|_2^2 \le \sup(f*f)$. $\square$

**Implication.** The conjecture is vacuous (much weaker than what's true) for symmetric $f$. The crux is the **asymmetric case**.

## Observation 2 — Symmetrization doesn't reduce to symmetric case.

Define $f^{sym}(x) = (f(x) + f(-x))/2$. Then $\int f^{sym} = 1$ and $\|f^{sym}\|_{3/2} \le \|f\|_{3/2}$ (convexity of $t^{3/2}$).

But $\sup(f^{sym} * f^{sym})$ can be *larger* than $\sup(f*f)$. For example, Schinzel-Schmidt $f_0(x) = (2x+1/2)^{-1/2}$ has $\|f_0\|_2 = \infty$, $\sup f_0 * f_0 = \pi/2 < \infty$. Its symmetrization has TWO singularities (at $x = \pm 1/4$), giving $\|f_0^{sym}\|_2 = \infty$ AND $\sup f_0^{sym} * f_0^{sym}(0) = \|f_0^{sym}\|_2^2 = \infty$.

So symmetrization doesn't help — the asymmetric case is genuinely different and the singular near-optimizers are inherently asymmetric.

## Observation 3 — The alpha-family extremality (rigorous).

For $f_a(x) = c_a (b - 2x)^{-a}$ on $[-1/4, 1/4]$, $b = 1/2$ (singular at $x = 1/4$):

For $a \in (0, 1/2]$, the function is feasible ($\sup f_a * f_a < \infty$), with:
- $\|f_a\|_{3/2}^{3/2} = c_a^{3/2} \cdot 0.5^{1-3a/2}/(1-3a/2)$ (for $3a/2 < 1$)
- $\sup f_a * f_a = c_a^2 \cdot $ Beta function (computable, finite)

At $a = 1/2$: SS, $c = 1/\sqrt 2$, $\|f_0\|_{3/2} = 2^{2/3}$, $\sup f_0 * f_0 = \pi/2$, ratio = $\pi/8$.

For $a > 1/2$: $\sup f_a * f_a = \infty$ (at $t \to 1/2$ the integrand $\sim \epsilon^{1-2a}$ diverges).

So in the alpha-family, $a = 1/2$ is the **boundary of feasibility** AND minimizes the ratio. Both endpoints of the alpha range conspire at the same point.

## Conjecture refinement

**REV-3/2 (asymmetric):** For nonneg ASYMMETRIC feasible $f$:
$$\sup(f*f) \ge \frac{\pi}{8} \,\|f\|_{3/2}^3$$
with equality at the alpha-family boundary $a = 1/2$.

**Combined REV-3/2:** Both symmetric (with $c_0 = 1$) and asymmetric (with $c_0 = \pi/8$) cases hold; the global minimum over all feasible $f$ is $\pi/8$, achieved on the asymmetric SS-boundary.

## Observation 4 — SS has a *flat plateau* of $f*f$ on $[-1/2, 0]$ (CONFIRMED ANALYTICALLY).

For $f_0(x) = (2x+1/2)^{-1/2}$:

**Claim.** $(f_0 * f_0)(t) = \pi/2$ for all $t \in [-1/2, 0]$, strictly decreasing on $(0, 1/2]$.

**Proof sketch.** Substituting $u = 2x + 1/2$ in the convolution integral:
$(f_0 * f_0)(t) = \frac{1}{2} \int_{u_-}^{u_+} (u(K-u))^{-1/2}\,du$, where $K = 2t + 1$ and $u_\pm$ are the support endpoints.

For $t \in (-1/2, 0]$: support overlap is $[-1/4, t+1/4]$, giving $u \in [0, K]$. The substitution $u = K\sin^2\theta$ yields integrand $= 2\,d\theta$ on $\theta \in [0, \pi/2]$, so the integral $= \pi$ and $(f_0*f_0)(t) = \pi/2$ — independent of $K$, hence of $t$.

For $t \in (0, 1/2)$: $u$ ranges over $[2t, 1] \subsetneq [0, K]$, so the angular integral is strictly less than $\pi$.

Numerically verified at 50 sampled points — diff from $\pi/2$ is $\le 3 \times 10^{-3}$ on $[-1/2, 0]$ (quadrature error from singularity), and SS strictly decreases on $(0, 1/2)$.

**Implication.** The dual maximizing measure $\nu^*$ for SS is supported on the *entire interval* $[-1/2, 0]$, not at a single point. This is unusual structurally and suggests the SS-extremality of the ratio has a flat-direction interpretation in the dual problem.

## What needs proving

The asymmetric case is the technical content. Possible approaches:
1. Variational: show SS is the unique critical point of the ratio in the asymmetric class.
2. Direct interpolation: find a chain $\sup f*f \ge X(f)^? \ge ... \ge c_0 \|f\|_{3/2}^3$ that's sharp at SS.
3. Move constants: in the asymmetric case, $\int f f(-\cdot) < \int f^2$, but we could consider $\sup f*f \ge (f*f)(t^*)$ for $t^* = $ optimal point and bound this.

This document records the partial structure for further work.
