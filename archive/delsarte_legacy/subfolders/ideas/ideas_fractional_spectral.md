# Fractional / Linear-Fractional / Spectral Reformulations of C_{1a}

**Date:** 2026-04-20
**Status:** Creative / exploratory. Budget: 10 fetches, 15 min.
**Context:** Current bounds 1.2802 <= C_{1a} <= 1.5029; we want to push the lower bound.

## 1. The ratio structure

Recall
$$ C_{1a} \;=\; \inf_{\substack{f \ge 0,\; \mathrm{supp}\, f \subset [-1/4,1/4]\\ \int f = 1}} \; \|f*f\|_{L^\infty[-1/2,1/2]}. $$

Because the constraint $\int f = 1$ is *linear* and the objective $\|f*f\|_\infty$ is **quadratic in $f$** (the convolution squares the mass), the natural homogenisation is
$$ C_{1a} \;=\; \inf_{f \ge 0,\; f \neq 0} \; \frac{\|f*f\|_{L^\infty}}{(\int f)^2}. $$

This is a **ratio of a convex quadratic functional (numerator) to a squared linear functional (denominator)**, subject to the cone constraint $f \ge 0$ (and support). That places it squarely in the territory of **fractional quadratic programming**.

## 2. Known fractional-programming transforms and what they give us

| Transform | Typical use | Applicability to $C_{1a}$? |
|---|---|---|
| **Charnes–Cooper** (1962): $y = f/(\int f)$, $t = 1/\int f$ | linear / linear ratio -> LP | *Directly applicable*: sets $\int f = 1$ and minimises $\|y*y\|_\infty$ with $y \ge 0$. This is exactly the normalisation already used in the codebase -- Charnes–Cooper gives no new information here because the ratio is already homogenised. |
| **Dinkelbach** (1967): solve $\lambda^* = \sup\{\lambda : \exists f,\; N(f) - \lambda D(f) \le 0\}$ | single-ratio nonlinear FP | *Trivially equivalent* to the parametric form $\inf_f \|f*f\|_\infty - \lambda(\int f)^2$ with $\lambda = C_{1a}^2$ (up to squaring); each inner problem is a convex-quadratic minimisation over a cone -- a **sign-definite QP** that's still $L^\infty$-hard, so Dinkelbach buys nothing by itself. But *combined with SDP relaxation* of the inner problem it yields a **parametric SDP** whose root is a certified lower bound on $C_{1a}$. This is a fresh angle: instead of one SDP for val(d), iterate $\lambda \uparrow$. |
| **Isbell–Marlow** (1956): Frank–Wolfe linearisation | linear/linear FP | Subsumed by Charnes–Cooper, no gain. |
| **Schaible / complex FP** (2005) | ratio of generalised-convex functions | Gives conditions under which Dinkelbach converges superlinearly; our ratio is **not jointly convex** (numerator is convex in $f$, denominator convex but squared), but it is a *ratio of a convex quadratic to a concave/linear* -- a case where the **S-lemma + SDP** is known to be tight (Ben-Tal/Teboulle, J. Convex Anal. 2010). |

## 3. Eigenvalue characterisation: does $C_{1a}$ admit one?

Fix a kernel $W(t) = \mathbf{1}_{[-1/2,1/2]}(t)$ and the convolution operator $T_W : f \mapsto (f*f)(t_0)$ restricted to a test point $t_0$. Then
$$ (f*f)(t_0) \;=\; \langle f,\; M_{t_0} f \rangle, \qquad (M_{t_0})(x,y) = \mathbf{1}_{x+y=t_0}. $$
For each $t_0$, $M_{t_0}$ is a **rank-1 Hankel-type** positive operator on $L^2([-1/4,1/4])$.

Then $\|f*f\|_\infty = \sup_{t_0} \langle f, M_{t_0} f\rangle$, and
$$ C_{1a} \;=\; \inf_{f \ge 0,\; \int f = 1} \; \sup_{t_0} \langle f, M_{t_0} f\rangle \;=\; \inf_{f \ge 0,\; \int f = 1} \; \lambda_{\max}\bigl(\text{convex hull of }\{M_{t_0}\}\bigr). $$
This is a **min–max eigenvalue problem** over the simplex of nonneg functions. Without the $f \ge 0$ constraint, the inner-max over $t_0$ in $L^2$ would collapse to a single generalised eigenvalue against $\mathbf{1}\mathbf{1}^\top$ (the "mean mass" operator), producing an elegant closed form -- but the positivity cone is essential and destroys that collapse.

However, two partial spectral reformulations *are* tractable:

- **Fourier / Toeplitz side.** In Fourier, $(f*f)^{\widehat{}}(\xi) = \hat f(\xi)^2$. By Plancherel, $\|f*f\|_2^2 = \int |\hat f|^4$, so the *$L^2$ version* (Matolcsi–Vinuesa) becomes a genuine eigenvalue problem for a **positive Toeplitz-type integral operator**. The $L^\infty$ problem only weakens this; any lower bound on the $L^\infty \to L^2$ ratio (a **Rayleigh-quotient-style** inequality on the support $[-1/2,1/2]$) immediately gives $C_{1a} \ge (\text{something in terms of the }L^2\text{ constant})$.
- **Generalised eigenvalue of two pencils.** Writing the Lasserre moment matrix $M(y)$ and the localising $L_{t_0}(y)$, the tight SDP lower bound at level $d$ satisfies $\mathrm{val}(d) = \min_y \max_{t_0}\;$ (generalised eigenvalue of $L_{t_0}(y)$ against $M(y)$), a *matrix pencil* whose smallest eigenvalue is monotone in $d$. This is implicit in the current `lasserre/` pipeline; making it explicit may reveal a **pencil-deflation** speedup.

## 4. Concrete ideas worth prototyping

1. **Parametric SDP via Dinkelbach.** Instead of minimising $\|f*f\|_\infty - \lambda\cdot 1$ subject to $\int f =1$, iterate: for each trial $\lambda$, certify $\inf_f (\|f*f\|_\infty - \lambda(\int f)^2) \ge 0$ via a *single* Lasserre SDP (no normalisation constraint -- $f$ is only constrained by nonnegativity/support). Homogeneity means each SDP is **smaller** than the current moment-matrix formulation (one linear equality removed), and a bisection in $\lambda$ converges quadratically once inside the basin.
2. **Generalised-eigenvalue dual.** Re-derive val(d) as $\min_y \max_{t_0} \lambda_{\max}(A(t_0,y), B(y))$ and apply the **Overton–Womersley** subdifferential of max-eigenvalue to build a nonsmooth-descent solver that outperforms generic interior-point on large $d$.
3. **Complex FP / Boyer–Li lift.** Boyer–Li's 1.2802 came from simulated annealing on $f$ directly. Embed the search in the **Charnes–Cooper homogenised cone** and replace SA with a *convex FP subgradient method* (Benson 2007 efficient QCQP algorithm). This is the cheapest win.
4. **$L^2$ surrogate + monotone lift.** The $L^2$ analogue has an exact Toeplitz eigenvalue characterisation (Cilleruelo–Ruzsa–Vinuesa). Combine with a quantitative $L^2 \to L^\infty$ embedding on $[-1/2,1/2]$ (Bernstein-type, since $f*f$ is supported on $[-1/2,1/2]$ and has controlled Fourier tail) to get a *new, spectrally clean* lower bound -- separate from Lasserre, hence potentially additive.

## 5. What I could not find

- No paper in the last 20 years explicitly applies Dinkelbach or Charnes–Cooper to **autoconvolution extremal problems**. This gap is suggestive: the ratio structure has been invisible in the CS / MV / White / Boyer–Li line.
- No eigenvalue characterisation of $C_{1a}$ is known; the $L^2$ analogue does admit one (Matolcsi–Vinuesa 2010) and may be the right stepping stone.

## 6. Recommended next step

Prototype **idea (1)**: a Dinkelbach-parametric Lasserre SDP at $d = 8$. Compare val against the current normalised pipeline. If the unnormalised moment matrix is strictly smaller and the bisection in $\lambda$ terminates in $\sim 6$ iterations, net wall-clock drops enough to reach $d = 10$--$12$ rigorously, which is where the current $1.2802$ lower bound lives.

---

## 100-word summary

$C_{1a}$ is the infimum of a quadratic-over-linear ratio, so fractional-programming transforms apply. Charnes–Cooper just restates the usual $\int f = 1$ normalisation and gives nothing new. Dinkelbach, however, turns $C_{1a}$ into a parametric family of *homogeneous* SDPs — one linear constraint fewer than the current Lasserre formulation — opening a bisection-in-$\lambda$ route at $d = 10$--$12$. A clean eigenvalue characterisation of $C_{1a}$ does **not** exist; the $L^2$ autoconvolution analogue does (via positive Toeplitz operators), and combining it with a quantitative $L^2 \to L^\infty$ embedding is a promising, spectrally independent lower-bound route.

## Sources

- [Linear-fractional programming (Wikipedia)](https://en.wikipedia.org/wiki/Linear-fractional_programming)
- [Charnes & Cooper 1962, *Programming with Linear Fractional Functionals*](http://iiif.library.cmu.edu/file/Cooper_box00010_fld00009_bdl0001_doc0001/Cooper_box00010_fld00009_bdl0001_doc0001.pdf)
- [Schaible, *Complex Fractional Programming and the Charnes-Cooper Transformation*, JOTA 2005](https://link.springer.com/article/10.1007/s10957-005-2669-y)
- [*Fractional Programming II: On Dinkelbach's Algorithm*, Management Science](https://pubsonline.informs.org/doi/10.1287/mnsc.22.8.868)
- [Benson 2007, *An Efficient Algorithm for Solving Convex–Convex Quadratic Fractional Programs*, JOTA](https://link.springer.com/article/10.1007/s10957-007-9188-y)
- [Fractional quadratic optimization with two quadratic constraints (NACO 2020)](https://www.aimsciences.org/article/doi/10.3934/naco.2020003)
- [Ben-Tal & Teboulle, *Fractional Generalisations*, J. Convex Anal. 17 (2010) 789–804](https://www.tau.ac.il/~becka/FracGeneral.pdf)
- [Amaral & Bomze, *Copositivity and constrained fractional quadratic problems*, Math. Prog. 2013](https://link.springer.com/article/10.1007/s10107-013-0690-8)
- [Boyer & Li 2025, *An improved example for an autoconvolution inequality*, arXiv:2506.16750](https://arxiv.org/abs/2506.16750)
- [White 2022, *An almost-tight $L^2$ autoconvolution inequality*, arXiv:2210.16437](https://arxiv.org/abs/2210.16437)
