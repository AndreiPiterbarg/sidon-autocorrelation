# Calculus-of-Variations View of $C_{1a}$

## The variational problem

Find $f \ge 0$, $\operatorname{supp} f \subset [-1/4,1/4]$, $\int f = 1$ minimizing
$\Phi(f) = \|f*f\|_\infty = \max_{|t|\le 1/2} \int f(x)\,f(t-x)\,dx.$
This is a **convex minimax** problem: $\Phi$ is convex (sup of linear-in-$f$ maps once we quadratize), the feasible set is convex and weak-* compact, and Madrid-Ramos (arXiv:2003.06962) and de Dios Pont-Madrid (arXiv:2001.02326) proved **existence of extremizers** (for suitable weighted versions) via duality + measure-theoretic compactness.

## Stationarity / KKT

Let $T^\star(f) = \{t : (f*f)(t) = \Phi(f)\}$ be the "active" set of maxima. For any probability measure $\nu$ on $T^\star(f)$, the Lagrangian is
$\mathcal{L}(f;\nu,\lambda,\mu) = \int\!\!\int f(x)f(t-x)\,dx\,d\nu(t) - \lambda\!\left(\!\int f - 1\right) - \int \mu(x)\,f(x)\,dx$
with $\mu\ge 0$, $\mu f \equiv 0$ (complementary slackness). The first-order condition $\delta\mathcal{L}/\delta f = 0$ gives the **non-local Euler-Lagrange identity**:
$(f * \check{\nu})(x) + (\check f * \nu)(x) \;=\; \lambda \qquad \text{on } \operatorname{supp} f,$
where $\check\nu(t) = \nu(-t)$. By symmetrizing (replace $\nu$ by $(\nu+\check\nu)/2$) and using that $f*f$ is even when the problem is symmetrized, one obtains the clean **saddle identity**

$(f * \sigma)(x) = \lambda/2 \quad \text{for a.e. } x \in \operatorname{supp} f, \qquad \sigma = (\nu+\check\nu)/2.$

Equivalently: there exists a probability measure $\sigma$ supported on the active set such that **the convolution $f*\sigma$ is constant on $\operatorname{supp} f$**, and $\ge \lambda/2$ elsewhere on $\operatorname{supp} f$'s interior. This is a **double variational principle**: inner $\max_\nu$ over active maxima, outer $\min_f$ over densities.

## Structural consequences

1. **Banzhaf/equi-level structure.** If $\sigma$ has $k$ atoms $\{t_1,\dots,t_k\}$ with weights $w_i$, then on $\operatorname{supp} f$ one has $\sum_i w_i\,f(t_i - x) = \lambda/2$. This is a **linear integral equation** with finite-rank kernel on $\operatorname{supp} f$; generically it forces $f$ to be **piecewise analytic** in the translate structure of $\{t_i\}$.

2. **Piecewise-polynomial / step conjecture.** The AlphaEvolve→Haar-refine sequence (arXiv:2506.16750) found extremizers that are **step functions with 50 then 575 pieces**; arXiv:2508.02803 notes a **"comb" motif**: sharp drop near $x\approx -0.24$, followed by small-peak comb. This is consistent with $\sigma$ being a finite sum of Diracs (because the active set $T^\star$ is expected to be finite or a Cantor-like set). Schinzel-Schmidt conjectured a smooth extremizer giving $c=\pi/2$; Matolcsi-Vinuesa (0907.1379) disproved it — **the true extremizer is not smooth**.

3. **Equioscillation analogy.** In Chebyshev best approximation, optimality is characterized by error equioscillation across $n+2$ points. Here the analogue: $(f*f)(t)$ should **equal $\Phi(f)$ on a "large enough" set** $T^\star$ to pin down $f$. The number of active points must match the degrees of freedom of $\operatorname{supp} f$'s parameterization.

## Can this give a rigorous lower bound?

**Yes, via the dual.** The min-over-$f$ max-over-$\nu$ has **dual**:
$C_{1a} = \sup_\nu \inf_{f} \frac{\int\!\!\int f(x)f(t-x)\,dx\,d\nu(t)}{(\int f)^2}$
subject to $\operatorname{supp}f \subset [-1/4,1/4]$, $f\ge 0$. Fix any symmetric probability measure $\nu$ on $[-1/2,1/2]$. The inner infimum is the **smallest eigenvalue of the (non-negative) quadratic form $Q_\nu(f,f) = \langle f, f*\check\nu\rangle$ restricted to non-negative $f$ with support in $[-1/4,1/4]$**. This is a **Perron / copositive eigenvalue**:
$\lambda_{\min}^{\ge 0}(Q_\nu) \;:=\; \inf\{Q_\nu(f,f): f\ge 0, \int f = 1, \operatorname{supp} f\subset[-1/4,1/4]\}.$

**Any feasible $\nu$ yields a valid lower bound** $C_{1a} \ge \lambda_{\min}^{\ge 0}(Q_\nu)$. Key observation:

- If we choose $\nu$ as a **finite atomic measure** $\nu = \sum w_i \delta_{t_i}$, the form $Q_\nu(f,f) = \sum w_i (f*f)(t_i)$ is a finite linear combination of point evaluations of $f*f$.
- The copositive eigenvalue is a **polynomial optimization problem** in the moments of $f$, solvable rigorously by **Lasserre / copositive SDP** — exactly the framework we already run in `lasserre/`.
- This gives a new family of Lasserre LPs: **moment Lasserre for $\inf_f \sum w_i (f*f)(t_i)$**, which by weak duality is $\le C_{1a}$.

**Concrete suggested probe.** Take $\nu$ supported on $\{0, \pm t^\star\}$ where $t^\star$ is the Cloninger-Steinerberger near-minimum of the current best $f*f$. Solve the Lasserre relaxation of $\inf_{f\ge 0, \int f=1, \text{supp}\subset[-1/4,1/4]} w_0(f*f)(0) + 2w_1(f*f)(t^\star)$. Optimize over $(w_0,w_1,t^\star)$ via bilevel outer loop.

## Connection to existing infrastructure

- Our `lasserre/preelim.py` already builds moment matrices for polynomial forms in $\mu$ (our discretized $f$). Replacing the inner `max_W M_W` with a **fixed nonnegative combination** $\sum w_i M_{t_i}$ (one $t_i$ per active point) gives a **single SDP per choice of $\nu$** — cheaper than the current min-max.
- The `certified_lasserre/` Farkas pipeline can then **certify** each dual $\nu$-bound rigorously, avoiding the min-max rigor gap noted in `project_rigor_parity_barrier.md`.

## 100-word summary

The $C_{1a}$ extremizer $f$ obeys a non-local Euler-Lagrange identity: there exists a probability measure $\sigma$ on the active set $T^\star = \{t:(f*f)(t)=\Phi(f)\}$ such that $f * \sigma \equiv \lambda/2$ on $\operatorname{supp} f$ (complementary slackness on $\{f=0\}$). Recent computational extremizers (Haar/comb step functions, arXiv:2506.16750, 2508.02803) are consistent with $\sigma$ being finitely atomic; Schinzel-Schmidt's smooth conjecture is false. **Rigorous exploit:** for any finite atomic $\nu=\sum w_i\delta_{t_i}$, $C_{1a}\ge \inf_{f\ge 0}\sum w_i(f*f)(t_i)$ is a **single copositive SDP** — solvable by our Lasserre+Farkas pipeline, bypassing the current min-max rigor barrier.

## Sources

- [Cloninger-Steinerberger, arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- [Matolcsi-Vinuesa, arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [Madrid-Ramos, arXiv:2003.06962](https://arxiv.org/abs/2003.06962)
- [de Dios Pont-Madrid, arXiv:2001.02326](https://arxiv.org/abs/2001.02326)
- [Barnard-Steinerberger, arXiv:2106.13873](https://arxiv.org/abs/2106.13873)
- [AlphaEvolve-style improvement, arXiv:2506.16750](https://arxiv.org/abs/2506.16750)
- [Haar-refine 575-step extremizer, arXiv:2508.02803](https://arxiv.org/abs/2508.02803)
- [Equioscillation theorem (Wikipedia)](https://en.wikipedia.org/wiki/Equioscillation_theorem)
