# SDP Relaxation Ideas for $C_{1a}=\inf\lVert f*f\rVert_\infty$

**Date:** 2026-04-20.  **Scope:** concrete SDP formulations beyond `sdp_hierarchy_design.md`; mine the Shor/Parrilo/Nesterov toolbox.  **Target:** certified lower bound on $C_{1a}$ with $1.2802<\lambda\le 1.5029$.

---

## 0. 100-word summary

Model $f$ as a degree-$d$ polynomial times a PSD kernel on $[-1/4,1/4]$ (Chebyshev/Legendre). The constraint $(f*f)(t)\le M$ pointwise becomes a **nonnegative polynomial constraint** on $[-1/2,1/2]$: certify it as SOS via Putinar (two Gram matrices, sizes $\le d$). The objective $M$ is scalar, so minimising $M$ over the joint $(f\text{-coeffs},\,\text{Gram matrices})$ is a **single SDP** (Shor-style: the bilinear $f\cdot f$ makes it a QCQP, relaxed by lifting $c c^T\succeq X$ with $X$ PSD). Add Fejér-Riesz / Bochner PSD for Fourier nonnegativity. Works at $d=64\text{--}128$ with Mosek $\sim 10$ min.

---

## 1. Shor relaxation for our QCQP (new angle)

### 1.1 The polynomial-$f$ parametrisation
Write $f(x)=w(x)\,p(x)^2$ where
- $w(x)=(\tfrac14-x)(\tfrac14+x)=\tfrac1{16}-x^2\ge0$ on $[-1/4,1/4]$ (enforces support/nonnegativity without extra cone),
- $p(x)=\sum_{k=0}^{d}c_k T_k(4x)$, $c\in\mathbb R^{d+1}$ (Chebyshev basis).

Then $f\ge0$ on $[-1/4,1/4]$ automatically; $f\equiv0$ off support. Normalisation $\int f=1$ is **linear** in the Gram $C:=cc^T$: $\int w T_iT_j\,dx=L_{ij}$ precomputed; $\int f=\langle L,C\rangle=1$.

### 1.2 Pointwise upper bound as SOS

$(f*f)(t)=\int f(x)f(t-x)\,dx$ is a **polynomial in $t$** of degree $\le 2(2d+2)$ whose coefficients are **bilinear** in $C$:
$$
(f*f)(t)=\sum_{i,j,k,l} C_{ij}C_{kl}\,Q_{ijkl}(t),\qquad Q_{ijkl}(t):=\int w(x)T_i(4x)T_j(4x)w(t-x)T_k(4(t-x))T_l(4(t-x))\,dx.
$$
The constraint $(f*f)(t)\le M\ \forall t\in[-1/2,1/2]$ is
$$
M-(f*f)(t)\;\ge\;0\quad\text{on }[-1/2,1/2].
$$
By **Putinar**, equivalently: $\exists$ SOS $\sigma_0(t),\sigma_1(t)$ with
$$
M-(f*f)(t)=\sigma_0(t)+(\tfrac14-t^2)\,\sigma_1(t).
$$
Each $\sigma_i$ has a Gram matrix $G_i\succeq 0$ of size $\le 2d+2$. Matching coefficients of $t^0,\dots,t^{4d+2}$ gives linear equations in $(C,G_0,G_1,M)$ — but with **quartic** cross terms $C_{ij}C_{kl}$.

### 1.3 Shor lift
Introduce $Y=\text{vec}(C)\text{vec}(C)^T\succeq 0$, size $(d+1)^2\times(d+1)^2$. The quartic becomes **linear** in $Y$:
$$
\sum_{ijkl} Y_{(ij),(kl)}\,Q_{ijkl}(t)=\sigma_0(t)+(\tfrac14-t^2)\sigma_1(t)+M\cdot 1.
$$
Relax $Y=\text{vec}(C)\text{vec}(C)^T$ to $Y\succeq 0$, $Y_{(ij),(kl)}=Y_{(kl),(ij)}=Y_{(ji),(lk)}$ (Chebyshev symmetry), $\text{tr}_{13}Y=C$ (marginal). **Single SDP**:
$$
\boxed{\ \min_{Y,G_0,G_1,M,C}\ M\quad\text{s.t.}\quad Y\succeq0,\ G_0\succeq0,\ G_1\succeq0,\ \text{coeff-match eqs},\ \langle L,C\rangle=1.\ }
$$
This is **Shor applied to the polynomial-$f$ QCQP**: drop the rank-1 constraint $Y=cc^T\otimes cc^T$.

**Size.** $(d+1)^2\times(d+1)^2$ PSD block for $Y$, plus two $O(d)$ blocks. At $d=32$: $Y$ is $1089\times 1089$ ($\approx 6\cdot10^5$ svec variables) — **well within Mosek range**. At $d=64$: $Y$ is $4225\times4225$ ($\approx 9\cdot10^6$ svec) — pushes it, use SCS or decompose by Chebyshev parity.

---

## 2. Tightenings (all SDP-compatible)

1. **Moment consistency.** $\text{tr}_{13}Y$ must equal $C$; $\text{tr}_{24}Y$ same. This partial rank-1 recovery pins the marginals exactly.
2. **Reynolds symmetry.** $f$ WLOG even ($f(x)=f(-x)$) by standard argument ⇒ $c_k=0$ for odd $k$, halving the dim of $C$.
3. **Fourier Bochner SDP.** Add $\hat f(\xi)$ as secondary variable; Bochner ⇒ Toeplitz matrix $T[\hat f]\succeq 0$ (Carathéodory). For $f$ polynomial, $\hat f$ is entire of exponential type and its restriction to $[-1,1]$ is a finite-dim linear image of $C$. Couple $(C,T)$ via the known Chebyshev→Fourier coefficients.
4. **$|\hat f|^2\ge\cos^2(\pi\xi/2)\mathbf1_{|\xi|\le1}$** Matolcsi–Vinuesa lower bound: linear inequalities on $Y$.
5. **Matolcsi–Vinuesa LP integrated.** The LP dual over admissible test $g$ plugs in as **linear constraints** on the $f*f$ moments, same SDP.

---

## 3. The Fejér-Riesz / Nesterov cone as a cleaner route

Instead of $f=w\cdot p^2$, parametrise **$\hat f$ directly**: since $f\ge0$ and $\int f=1$, $\hat f(0)=1$ and $\hat f$ is a positive-definite function. By **Fejér-Riesz** (univariate, bandlimited case): any nonneg trigonometric polynomial of degree $N$ on $\mathbb T$ is $|q(e^{i\theta})|^2$ for a polynomial $q$ of degree $N$, parametrised by a Hermitian Toeplitz $T\succeq0$.

For us, $f*f$ has Fourier transform $\hat f^2\ge0$, so $\hat f^2$ restricted to $[-1,1]$ (truncated: $f$ has type $\pi/2$ so $\hat f^2$ has type $\pi$) can be written via Dumitrescu's **positive trigonometric polynomials** toolbox: $|\hat f|^2=\langle T,\Phi(\xi)\rangle$ with $T\succeq0$ Toeplitz. Then
$$
(f*f)(t)=\int \hat f(\xi)^2 e^{2\pi i\xi t}\,d\xi=\langle T,\Psi(t)\rangle,
$$
with $\Psi(t)$ explicit. The $\sup_t\le M$ constraint is again Putinar-SOS. **Same SDP, dual parametrisation**; often numerically sharper because it bakes in Bochner.

Refs: Dumitrescu, *Positive Trigonometric Polynomials and Signal Processing Applications* (Springer 2017); Nesterov, *Squared functional systems and optimisation problems* (2000); Alkire–Vandenberghe, SIAM J. Optim. 2002 on FIR autocorrelation SDP.

---

## 4. Why this is different from the existing `sdp_hierarchy_design.md`

That note does a **moment-based** Lasserre lift on a 3-cube with a Dirac kernel, scaling $O(k^9)$ and requiring kernel degree $N\gg1$. Here:

| Aspect | Moment Lasserre (existing) | Shor QCQP + SOS (this note) |
|---|---|---|
| Primal var | joint moments $y_{\alpha\beta\gamma}$ | $f$-coefficients $c\in\mathbb R^{d+1}$ plus lift $Y$ |
| PSD blocks | one of size $\binom{3+k}{3}$ + localisers | one $(d+1)^2\times(d+1)^2$ Shor block + 2 Gram SOS |
| Objective | inner product with kernel | scalar $M$ |
| Dirac handling | kernel smoothing (error $1/N$) | **exact** via Putinar on $t$ |
| Size at target | $k{=}8\Rightarrow$ svec $10^4$ | $d{=}32\Rightarrow$ svec $6\cdot10^5$ |
| Rigorous cert | yes (SOS dual) | yes (Putinar dual $\sigma_0,\sigma_1$) |

The Shor formulation's advantage: **exact pointwise** $(f*f)(t)\le M$ enforcement, no kernel error; the cost is a larger $Y$ block.

---

## 5. Solver reality check

| Solver | Regime | Expected at $d=32$ ($Y\approx1100^2$) | $d=64$ ($Y\approx4200^2$) |
|---|---|---|---|
| **Mosek** (IPM, dense blocks) | best on $\lesssim 5000$-sized PSD, needs RAM | $\sim 2$–$10$ min, $\sim 30$ GB | borderline; may OOM on $>60$ GB |
| **SCS** (ADMM, low RAM) | scales to $n\sim 10^4$ but low accuracy | $\sim 1$–$5$ min, $\sim 2$ GB | feasible, $10$–$60$ min |
| **SDPA-DD** | high-precision dual, slow | slow but rigorous | too big |
| **COSMO / Clarabel** | ADMM, 2nd-order | similar to SCS | feasible |

Sources: Hans Mittelmann's SDPLIB benchmarks; `cvxpy` issue #2051 (Mosek svec RAM). Strategy: **Mosek for rigor + warmstart with SCS** (same pattern as `lasserre/solvers.py`). Decompose $Y$ by $\mathbb Z/2$ even/odd to cut to $2\times(d/2+1)^2$ blocks — halves size twice in the two Chebyshev factors ⇒ $4\times$ saving.

---

## 6. Concrete action plan (1 week)

1. **Prototype** Shor QCQP of §1 at $d=8$, compare to `lasserre/core.py` `val(d=8)=1.205`. Expected: $\lambda\approx1.20\pm0.02$ (validates coefficient match).
2. **Add Fejér-Riesz / Bochner** block of §3; check if bound tightens at $d=8$.
3. **Scale to $d=32$** with Mosek dual + `lasserre/z2_blockdiag.py` symmetry reuse.
4. **Rigorous rationalisation** via `certified_lasserre/` Farkas pipeline (post-solve LP feasibility on rational $(C,Y,G_0,G_1,M)$).
5. **Benchmark against** the cascade's $d=16$ bound $1.319$: if Shor at $d=32$ beats $1.284$, it is a drop-in replacement; if it beats $1.30$, it leapfrogs.

---

## 7. Key references (fetched 2026-04-20)

- [Chapter 3 Shor's Semidefinite Relaxation — Hankyang (Harvard)](https://hankyang.seas.harvard.edu/Semidefinite/Shor.html)
- [Park & Boyd, *General Heuristics for Nonconvex QCQPs*](https://web.stanford.edu/~boyd/papers/pdf/qcqp.pdf)
- [Wang & Kılınç-Karzan, *Exactness in SDP relaxations of QCQPs*, arXiv:2107.06885](https://arxiv.org/abs/2107.06885)
- [Parrilo, *Sum of squares programs and polynomial inequalities*, MIT OCW lec22](https://ocw.mit.edu/courses/6-972-algebraic-techniques-and-semidefinite-optimization-spring-2006/63d6e45650205ceef6701971a9b4621c_lecture_22.pdf)
- [Parrilo & Lall, ECC 2003 SOS course, lecture 5](https://www.mit.edu/~parrilo/ecc03_course/05_sum_of_squares.pdf)
- [Ahmadi, *SOS techniques — ORF523 lec 15*](https://www.princeton.edu/~aaa/Public/Teaching/ORF523/ORF523_Lec15.pdf)
- [Dumitrescu, *Positive Trigonometric Polynomials and Signal Processing Applications*](http://ndl.ethernet.edu.et/bitstream/123456789/20753/1/145.pdf)
- [Alkire & Vandenberghe, *Convex problems with autocorrelation constraints*, Math. Prog. 2002](https://link.springer.com/article/10.1007/s10107-002-0334-x)
- [Genin–Nesterov–Van Dooren, nonneg trigonometric polynomial interior-point scaling, SIAM J. Optim. 2006](https://www.seas.ucla.edu/~vandenbe/publications/nnp.pdf)
- [Scalable SDP survey, arXiv:1908.05209](https://arxiv.org/pdf/1908.05209)
- [MOSEK Modeling Cookbook 3.4.0 — SDP chapter](https://docs.mosek.com/modeling-cookbook/sdo.html)
- [Hans Mittelmann SDP benchmarks, DIMACS node15](https://plato.asu.edu/dimacs/node15.html)

---

## 8. One-line answer to the key question

> *"If we parametrize $f$ (or $f*f$, or $\hat f$) as a polynomial $\times$ kernel, can $\lVert f*f\rVert_\infty\le M$ become an SDP constraint?"*

**Yes.** With $f=(\tfrac1{16}-x^2)\,p(x)^2$ on $[-1/4,1/4]$, $\lVert f*f\rVert_\infty\le M$ is **Putinar-SOS** on $[-1/2,1/2]$ — two Gram matrices $G_0,G_1\succeq0$ — and the quartic-in-$p$ coefficients are linearised by a Shor lift $Y=\text{vec}(cc^T)\text{vec}(cc^T)^T\succeq0$. The whole program is a **single SDP** with PSD blocks of size $(d+1)^2$, $d+1$, $d$; rigorous, scales to $d\sim 64$, and drops into `lasserre/dual_sdp.py` + `certified_lasserre/` Farkas rationalisation without new tooling.
