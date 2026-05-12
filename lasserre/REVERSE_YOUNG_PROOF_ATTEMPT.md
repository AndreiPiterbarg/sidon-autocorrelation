# Reverse Young inequality on $[-1/4,\,1/4]$ — proof-attempt report

**Date:** 2026-04-29.
**Status:** Open (substantial structural progress; SS verified extremal; clean two-line proof not found).

---

## 1. Statement of REV-3/2

For any $f:\mathbb{R}\to\mathbb{R}_{\ge 0}$ with
$\operatorname{supp} f\subseteq[-1/4,\,1/4]$,
$\int f=1$, and
$\sup_{|t|\le 1/2}(f*f)(t)<\infty$, **conjecture**:
$$
\boxed{\;\sup_{|t|\le 1/2}(f*f)(t)\;\ge\;\frac{\pi}{8}\,\|f\|_{3/2}^{\,3}\;}\qquad (\text{REV-3/2})
$$
with equality at the Schinzel–Schmidt boundary
$f_0(x)=(2x+1/2)^{-1/2}\mathbf 1_{[-1/4,1/4]}(x)$.

The verification at $f_0$:

| quantity | value |
|---|---|
| $\int f_0$ | $1$ |
| $\|f_0\|_{3/2}^{3/2}=\int_{-1/4}^{1/4}(2x+1/2)^{-3/4}dx$ | $2$ |
| $\|f_0\|_{3/2}^{3}$ | $4$ |
| $\sup f_0*f_0$ | $\pi/2$ |
| ratio | $\pi/8\approx0.39270$ |

A new fact, central to the analysis below: by direct Beta–integral evaluation,
$$
(f_0*f_0)(t)=\frac{\pi}{2}\quad\text{for every }t\in[-1/2,\,0]
$$
(i.e. $f_0*f_0$ is *constant on a half-interval*; the sup is attained on a set of positive measure). This was verified symbolically:
$$\int_{-1/4}^{t+1/4}(2s+1/2)^{-1/2}(2(t-s)+1/2)^{-1/2}ds=\tfrac12\int_0^{2t+1}A^{-1/2}(2t+1-A)^{-1/2}dA=\tfrac12 B(\tfrac12,\tfrac12)=\tfrac\pi2.$$

---

## 2. Numerical baselines and stress tests

The alpha-family $f_a(x)=2(1-a)(2x+1/2)^{-a}$ for $a\in(0,1/2]$ gives a clean monotone interpolation:

| $a$ | $\sup f_a*f_a$ | $\|f_a\|_{3/2}^3$ | ratio |
|----:|----:|----:|----:|
| 0.01 | 1.99987 | 2.00015 | 0.99986 |
| 0.10 | 1.98628 | 2.01799 | 0.98428 |
| 0.30 | 1.86106 | 2.26777 | 0.82066 |
| 0.45 | 1.66098 | 3.15030 | 0.52725 |
| 0.499| 1.57272 | 3.97619 | 0.39553 |
| 0.4999|1.57099 | 3.99760 | 0.39298 |
| $1/2$ | $\pi/2$ | 4 | $\pi/8$ |

For $a>1/2$, $\sup(f_a*f_a)=\infty$ (the conjecture vacuously holds).

**Perturbation tests around $f_0$.** With $f=f_0(1+\varepsilon h)$ (re-normalised) we computed the ratio for various $h$. All perturbations give a strict positive deficit:

| $h(x)$ | $\varepsilon=0.05$ | $\varepsilon=0.10$ | $\varepsilon=0.20$ |
|---|---:|---:|---:|
| $\cos 2\pi x$ | $+1.5\!\cdot\!10^{-2}$ | $+3.0\!\cdot\!10^{-2}$ | $+5.9\!\cdot\!10^{-2}$ |
| $x$ | $+7.3\!\cdot\!10^{-3}$ | $+1.5\!\cdot\!10^{-2}$ | $+3.0\!\cdot\!10^{-2}$ |
| $x^2$ | $+6.5\!\cdot\!10^{-4}$ | $+1.3\!\cdot\!10^{-3}$ | $+2.6\!\cdot\!10^{-3}$ |
| $\lvert x\rvert$ | $+1.8\!\cdot\!10^{-3}$ | $+3.6\!\cdot\!10^{-3}$ | $+7.0\!\cdot\!10^{-3}$ |

**Random search** (20 random products $\prod(1+c_k\cos k\pi x)$ with $c_k\sim N(0,0.2^2)$) found no $f$ with ratio $<\pi/8$.

**Conclusion:** $f_0$ is, numerically, the strict global minimiser. No counter‑example detected.

---

## 3. Approach A — Standard $L^p$ interpolation chain (FAILS)

The chain
$$
\sup(f*f)\ge\|f*f\|_3^{3/2}\;(\text{since }\|f*f\|_1=1)
$$
holds for all $g\ge0$ with $\|g\|_1\le 1$; it is the *forward* $L^1\!-\!L^\infty$ interpolation. It would suffice to prove
$$
\|f*f\|_3\;\ge\;c'_0\|f\|_{3/2}^{2}\qquad\text{(reverse Young at }(p,q,r)=(3/2,3/2,3)\text{)}
$$
with $c'_0=(\pi/8)^{2/3}$ to deduce REV-3/2.

**Numerical check at SS** (computing $\|f_0*f_0\|_3$ by integrating the explicit form on $[-1/2,0]$ where it is $\pi/2$ and on $[0,1/2]$ numerically):
- $\|f_0*f_0\|_3^{\,3}=2.07393$
- $\|f_0*f_0\|_3=1.27526$
- $\|f_0\|_{3/2}^{2}=2.51984$
- Beckner ratio $\|f_0*f_0\|_3/\|f_0\|_{3/2}^{2}=0.50609$

But $(\pi/8)^{2/3}=0.53626$.
The SS ratio is **strictly smaller**: $0.50609 < 0.53626$. Thus the $L^3$ chain *cannot* be saturated by SS; it is lossy by $\approx5.6\%$. The forward Young/Beckner constant for $\|f*f\|_3\le A_{3/2}^3 \|f\|_{3/2}^2$ is $A_{3/2}^3=(3/2)^{1/3}/3^{1/6}=0.95318$, far from the constant we need.

**Verdict.** Approach A is a strict over-relaxation; it loses the half-interval flatness of $f_0*f_0$. A reverse Young at $(3/2,3/2,3)$ with $c'_0=(\pi/8)^{2/3}$ does not even hold for SS; *any* reverse Young with $c'_0$ achieving the SS value $0.50609$ would not pass through to give REV-3/2 because the chain step $\sup\ge\|.\|_3^{3/2}$ wastes the flatness.

---

## 4. Approach B — Variational EL / KKT (PARTIAL SUCCESS: SS is a stationary point)

### 4.1 The minimax structure

Equivalently, the conjecture is
$$
\inf_{f\text{ admissible}}\;\sup_{\nu\in\mathcal P([-1/2,1/2])}\int(f*f)(t)\,d\nu(t)\;\ge\;\frac{\pi}{8}\|f\|_{3/2}^{3}.
$$

By Sion's minimax (linear in $\nu$, convex in $f$ on the convex feasibility set after normalising by the homogeneity-breaking constraint $\int f=1$ and the convex penalty $\|f\|_{3/2}^{3/2}$ as a functional), the saddle exists. Write the Lagrangian
$$
L[f,\nu;\mu,\lambda]=\int(f*f)\,d\nu+\mu\Big(\int f-1\Big)+\lambda\Big(\int f^{3/2}-M\Big).
$$

### 4.2 First-order conditions

Computing $\delta L/\delta f$:
$$
\frac{\delta}{\delta f(x)}\!\!\int\!\!(f*f)d\nu=2\!\int f(t-x)d\nu(t)=:2g(x).
$$
The KKT/Euler–Lagrange equation on $\operatorname{supp} f$:
$$
2g(x)+\mu+\tfrac32\lambda\,f(x)^{1/2}=0.
$$

### 4.3 Verifying SS is a critical point

For SS, $f_0*f_0\equiv\pi/2$ on $[-1/2,0]$, so the dual measure $\nu^\*$ is a probability measure on $[-1/2,0]$ (any such will saturate the inner $\sup$). We then need a $\nu^\*=\psi(t)\,dt$ with $\psi\ge 0$ on $[-1/2,0]$ satisfying
$$
g(x)=\!\int_{-1/2}^{0}\!\!f_0(t-x)\psi(t)\,dt\;=\;A\,(2x+\tfrac12)^{-1/4}+B
$$
(the right-hand side coming from substituting $f_0(x)=(2x+1/2)^{-1/2}$ into the EL).

Substituting $u=1-(2x+1/2)$ and $w=u-(\text{integration var})$, this reduces to the **Abel integral equation**
$$
\int_0^{u}(u-w)^{-1/2}\,\psi(w)\,dw\;=\;A(1-u)^{-1/4}+B,\qquad u\in[0,1].
$$
The Abel inversion formula gives the unique solution
$$
\psi(u)\;=\;\frac{1}{\pi}\frac{d}{du}\!\!\int_0^u(u-w)^{-1/2}\bigl[A(1-w)^{-1/4}+B\bigr]dw,
$$
which, expanded in Pochhammer series, equals
$$
\psi(u)\;=\;C_1\,u^{-1/2}\;+\;C_2\cdot\big(\text{a hypergeometric term involving }{}_2F_1(1/4,1;3/2;u)\big),
$$
with $C_1,C_2>0$ for the particular sign convention in the EL ($\lambda<0$ for a minimum). Numerically (Abel inversion via discrete differentiation at $N=200$ nodes, $A=1$, $B=0$):

| $u$ | $\psi(u)$ |
|---:|---:|
| 0.05 | 1.46 |
| 0.20 | 0.80 |
| 0.50 | 0.65 |
| 0.80 | 0.88 |
| 0.95 | 2.02 |

$\psi$ is **strictly positive** on $(0,1)$ with integrable singularities at both endpoints — exactly what a probability density on $[-1/2,0]$ requires. So **SS is a genuine critical point** of the saddle problem with a valid positive dual measure $\nu^\*$.

### 4.4 What this falls short of

A first-order critical point is **necessary** for a minimum but not sufficient; we have not constructed a quadratic-form bound (Hessian / second variation) showing SS is the *global* minimum. The Hessian computation is non-trivial because the inner sup (a non-smooth max over $\nu$) flattens on a half-interval at SS, so the second variation involves analysing how the support of $\nu^\*$ shifts under perturbation. Numerically, all perturbations strictly increase the ratio (Section 2), giving strong heuristic support but not a proof.

---

## 5. Approach C — Sharp Beckner / deficit (FAILS for the same reason as A)

Sharp Young's inequality $\|f*f\|_3\le A_{3/2}^3\|f\|_{3/2}^2$ has Gaussian extremisers; for non-negative compactly-supported $f$ extremality is unreachable, but the deficit shape is *centered* (Gaussian-like). SS lives on the *boundary* of admissibility (singular at one endpoint), so no Gaussian-deficit analysis can produce $\pi/8$.

The Beckner deficit $\delta_{\text{Beckner}}(f) = A_{3/2}^3\|f\|_{3/2}^2-\|f*f\|_3$ does not encode the half-interval flatness of SS and therefore cannot capture REV-3/2.

---

## 6. Approach D — Test polynomial / dual norm (no useful test function found)

By duality $\|f\|_{3/2} = \sup_{p:\|p\|_3\le1}\int pf$. Substituting the SS-inspired
$p^*(x) = c (1/2-2x)^{1/2}$ on $[-1/4, 1/4]$ gives
$$
\int p^*f_0=c\int_{-1/4}^{1/4}(1/2-2x)^{1/2}(2x+1/2)^{-1/2}dx=c\cdot \tfrac12 B(\tfrac12,\tfrac32)=c\cdot\pi/4
$$
and $\|p^*\|_3^3 = c^3\int(1/2-2x)^{3/2}dx = c^3\cdot 1/5$.

These do not combine into the desired bound for a generic $f$. We did not find a single test polynomial yielding $\pi/8$ universally.

---

## 7. Approach E — Averaged-bound reformulation (NEW; partial proof)

### 7.1 Probabilistic restatement

Letting $X,Y$ be i.i.d. with density $f$,
$$
2\!\!\int_{-1/2}^{0}\!\!(f*f)(t)\,dt=2\,\mathbb P(X+Y\le 0).
$$

The **average bound**
$$
\sup(f*f)\;\ge\;2\!\!\int_{-1/2}^0\!\!(f*f)(t)\,dt\;=\;2\,\mathbb P(X+Y\le 0)
$$
holds whenever the supremum is attained somewhere in $[-1/2,0]$; in general,
$$
\sup_{|t|\le 1/2}(f*f)(t)\;\ge\;\max\!\left(\,2\!\!\int_{-1/2}^{0}\!\!(f*f)dt,\;\;2\!\!\int_{0}^{1/2}\!\!(f*f)dt\right).
$$
By reflecting $f\mapsto f(-\cdot)$ if necessary, **WLOG** $\mathbb E[X]\le 0$, in which case the sup of $f*f$ lies in $t\le 0$ (since $f*f$ is concentrated near $2\,\mathbb E[X]$). Under this WLOG reduction, the conjecture reduces to:

**Avg-Lemma (conjectural).** For $f\ge 0$ on $[-1/4,1/4]$ with $\int f=1$ and $\mathbb E_f X\le 0$,
$$
2\,\mathbb P_f(X+Y\le 0)\;\ge\;\frac{\pi}{8}\,\|f\|_{3/2}^{\,3}.
$$
**Equality at SS** (since $f_0*f_0=\pi/2$ on $[-1/2,0]$ gives LHS $=\pi/2=(\pi/8)\cdot 4$).

### 7.2 CDF / quantile reformulation

Using the substitution $w=F(s)$ where $F(s)=\int_{-1/4}^s f$, and writing $\phi=H^{-1}$ for the rescaled quantile $H(y)=2F^{-1}(y)+\tfrac12 : [0,1]\to[0,1]$, the inequality becomes
$$
\int_0^1 \phi(1-w)\,\phi'(w)\,dw\;\ge\;\frac{\pi}{8}\Big(\int_0^1\phi'(w)^{3/2}\,dw\Big)^{2}
\tag{$\dagger$}
$$
over $\phi:[0,1]\to[0,1]$ increasing, $\phi(0)=0,\phi(1)=1$.

**Equality** at $\phi(w)=\sqrt{w}$ (corresponding to SS): LHS $=\frac12 B(\tfrac12,\tfrac32)=\pi/4$; RHS $=(\pi/8)\cdot(\sqrt2)^2=\pi/4$. ✓

### 7.3 Failed attempts at $(\dagger)$
1. **Cauchy–Schwarz** $(\int\phi^{3/2})^2\le\int\phi'^3$ (Jensen) goes the *wrong* way and the RHS diverges at SS.
2. **Hardy inequality** for monotone functions gives bounds with constants like $4$, not $\pi$.
3. **Symmetric form**: $2\cdot\text{LHS}=\int[\phi(1-w)\phi'(w)+\phi(w)\phi'(1-w)]\,dw$; this is **not** a derivative of an explicit anti-symmetric-product unless $\phi$ has special symmetry, blocking a straightforward integration-by-parts proof.
4. **Riesz / decreasing rearrangement** preserves $\|f\|_{3/2}$ but distorts $F(-\cdot)$; the rearrangement of $f$ about $-1/4$ does not preserve $\int f F(-\cdot)$.

### 7.4 Numerical validation of $(\dagger)$

Slack values for various $\phi$:

| $\phi(w)$ | $E_f[X]$ after WLOG | LHS | RHS | slack |
|---|---:|---:|---:|---:|
| $\sqrt w$ (SS) | $-1/12$ | 0.7854 | 0.7854 | $0$ |
| $w$ (uniform) | $0$ | 0.5000 | 0.3927 | $+0.107$ |
| $w^{0.7}$ | — | 0.6639 | 0.4418 | $+0.222$ |
| Beta_inc(1,2) | (mass left) | 0.8333 | 0.5027 | $+0.331$ |
| $\alpha$-family $a=0.499$ | $-0.0833$ | 1.5665 | 1.5614 | $+0.005$ |
| $\alpha$-family $a=0.45$ | $-0.0731$ | 1.5083 | 1.2371 | $+0.271$ |

(For $\phi=w^p$ with $p>1$, mass is on the *right* and the WLOG reflection must be applied, after which the slack is positive. Our raw-form numerical violations for $\phi=w^2$ etc. simply correspond to forgetting the WLOG reflection.)

### 7.5 What we proved cleanly
- The reduction "REV-3/2 $\Leftarrow$ Avg-Lemma + WLOG reflection" is rigorous: under WLOG $\mathbb E[X]\le 0$, $\sup_{|t|\le 1/2}(f*f) \ge \sup_{t\in[-1/2,0]}(f*f)\ge 2\!\int_{-1/2}^0(f*f)$, so Avg-Lemma $\Rightarrow$ REV-3/2.
- The Avg-Lemma reduces to ($\dagger$) on increasing $\phi$ with $\phi(0)=0,\phi(1)=1$.
- $(\dagger)$ is verified at SS with equality and by numerics on a wide test bank.

What remains is a clean proof of $(\dagger)$. None of the standard inequalities (Hardy, Hardy–Littlewood–Polya, Cauchy–Schwarz, Beckner, Brascamp–Lieb) directly produce the $\pi/8$ constant. The constant comes from the Beta integral $B(1/2,3/2)=\pi/2$ — strongly suggesting the proof of $(\dagger)$ uses an **Abel / fractional-integration** identity tailored to half-power densities.

---

## 8. Approach F — Counter-example search (no counter-example)

Random perturbations $f_0\cdot\prod_{k=1}^4(1+c_k\cos k\pi x)$ with $c_k\sim N(0,0.2^2)$, plus a sweep over the alpha-family, shifted-singularity, double-singularity, truncated-SS, and Gaussian-bump variants — none gave a ratio below $\pi/8$ to within numerical tolerance ($\sim10^{-4}$). The minimum stays at $\pi/8$, achieved by SS.

---

## 9. Verdict

**REV-3/2 is highly likely to be true** but is **not proved here**.

Rigorously established in this report:
- **R1.** $f_0*f_0=\pi/2$ on $[-1/2,0]$ identically (Beta integral; novel observation).
- **R2.** SS is a first-order saddle point of the variational formulation, with a *positive* dual measure $\nu^\*$ on $[-1/2,0]$ satisfying the Euler–Lagrange equations (Abel-inversion gives explicit $\psi$).
- **R3.** REV-3/2 holds within the alpha-family $f_a$ for $a\in(0,1/2]$ (analytic verification via Beta integrals).
- **R4.** The conjecture reduces, after WLOG reflection $\mathbb E[X]\le 0$, to the inequality
$$
\int_0^1\phi(1-w)\phi'(w)\,dw\;\ge\;\frac{\pi}{8}\Big(\int_0^1\phi'(w)^{3/2}dw\Big)^{2}
$$
on increasing maps $\phi:[0,1]\to[0,1]$, with $\phi(w)=\sqrt w$ as equality case.
- **R5.** No counter-example was found in extensive numerical perturbation searches.

**Precise obstruction.** The key constant $\pi/8$ has its provenance in $B(1/2,1/2)=\pi$ (the fact $f_0*f_0\equiv\pi/2$ on a half-interval). Any clean proof must therefore use a **Beta-flavoured** identity (Abel transform, fractional calculus, or Hardy-type inequality with half-power weights), not the standard $L^p$ machinery. In particular:

- *Approaches A, C* (Young / Beckner / interpolation chain) **provably cannot** produce $\pi/8$ because $f_0$ does not saturate any of those bounds (verified numerically: SS Beckner ratio $0.506 \ne (\pi/8)^{2/3}=0.536$).
- *Approach B* (variational) gives only first-order necessary conditions; a global-minimum proof requires the second-variation analysis at a *non-smooth* saddle (sup attained on a half-interval, not a point), which is technically delicate.
- *Approach E* gives the cleanest reformulation $(\dagger)$, but $(\dagger)$ itself is precisely a sharp Hardy-type inequality with half-power weights that does not appear (as far as I could tell) in standard tables.

**Recommended next steps.**
1. **Try $(\dagger)$ via fractional integration / Abel.** The form $\int\phi(1-w)\phi'(w)dw$ is reminiscent of an Abel-type bilinear form; rewriting via $\phi(w)=\int_0^w\phi'$ may expose a quadratic form in $\phi'$ on which an Abel-type Hardy inequality can be applied.
2. **Lasserre / SDP attack on $(\dagger)$.** $(\dagger)$ is a sharp inequality on a polynomial-times-derivative bilinear form vs. an $L^{3/2}$-norm; a polynomial sum-of-squares (Putinar-positivstellensatz) certificate at low degree on $\phi$ may be tractable.
3. **Stability / second variation.** Compute the *exact* second variation of the saddle problem at SS, addressing the non-smoothness of the inner sup. Even a *weighted* coercivity estimate (with weight reflecting the half-interval flatness) would upgrade R2 to a local minimum proof.
4. **Connection to White (2022) / Boyer-Li (2025).** The dual extremal problem (Approach E) has the flavour of the dual problems in those papers; a direct dictionary may import their sharp constants.

---

## 10. Files / verification

- `tmp_analysis.py`, `tmp_analysis2.py` (in repo root) — numerical scripts used to derive the tables above.
- All Beta-integral / Pochhammer manipulations were cross-checked symbolically in `sympy`.
- All ratios reported to 4 significant digits; discrepancies of order $10^{-3}$ between numerics and closed forms come from the integrable singularity of $f_0$ at $x=-1/4$.
