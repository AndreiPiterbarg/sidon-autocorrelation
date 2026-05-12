# Creative Fourier-Analytic Inequalities for the $C_{1a}$ Dual

## 100-word summary

The MV dual uses Cauchy-Schwarz with arcsine kernel on `(f*f)(0)`. Two under-used
Fourier-analytic tools may break the 1.276 ceiling: **(1) Siegel-type Turan duality**,
which replaces the Cauchy-Schwarz step with a *sharp* positive-definite extremal
identity tailored to the support `[-1/4, 1/4]` of `f`, inherently encoding
Paley-Wiener structure; and **(2) Logvinenko-Sereda / bandlimited concentration
bounds** (via prolate spheroidal wave functions), which bound the fraction of
`|f̂|^2` mass escaping any finite window `[-T,T]`, tightening the rigorous dual
integrand `w(ξ)=cos^2(πξ/2)` currently used on `[-1,1]`. Both plug into the
existing admissible-`g` framework without modifying the primal.

---

## Problem setup (reminder)

Let $f \ge 0$, $\operatorname{supp} f \subset [-\tfrac14,\tfrac14]$, $\int f = 1$.
Then $f*f$ is supported on $[-\tfrac12,\tfrac12]$ and $\hat f$ is entire of
exponential type $\pi/2$ with $|\hat f(0)|=1$, $|\hat f| \le 1$ (Paley-Wiener).

Goal: lift $\|f*f\|_\infty \ge L$, currently $L = 1.2802$ (CS17) / $1.2748$ (MV dual).

---

## Inequality 1: Siegel's sharp Turan identity (for self-convolutions)

**Classical fact** (Siegel 1935, via Kolountzakis-Revesz, Gorbachev-Tikhonov
2024 "Turan problem and its dual"): among positive-definite functions
$g$ supported on a symmetric convex body $K$ with $g(0)=1$,
$$
\sup_g \int_K g \;=\; \frac{|K|}{2^d},
$$
achieved **uniquely** by $g_* = \mathbb 1_{K/2} * \mathbb 1_{K/2} / |K/2|$.

**Application to $C_{1a}$.** Our problem is dual to the Turan problem on
$K=[-\tfrac12,\tfrac12]$:
- $f*f$ is supported on $K$.
- $f*f$ **is already positive definite** (since $\widehat{f*f} = |\hat f|^2 \ge 0$).
- $(f*f)(0) = \int f^2$.

So for any admissible $g$ with $\operatorname{supp} g \subset K$ and
$\hat g \ge 0$ (i.e. $g$ **positive definite** by Bochner):
$$
\int_{-1/2}^{1/2} g(t) (f*f)(t)\,dt \;=\; \int_{\mathbb R} \hat g(\xi) |\hat f(\xi)|^2\,d\xi.
$$
Pick $g = g_*$ *the Siegel extremizer* rescaled to live on $[-\tfrac12,\tfrac12]$:
$g_*(t) = (1 - 2|t|)_+$ (the triangle / Fejer-in-space). Then
$g_*(0) = 1 = M_{g_*}$, $\hat g_*(\xi) = \tfrac12 \operatorname{sinc}^2(\pi\xi/2) \ge 0$,
$\int g_* = \tfrac12$.

**Concrete lower bound.** With $w(\xi) = \cos^2(\pi\xi/2) \mathbb 1_{[-1,1]}$ the
rigorous minorant for $|\hat f|^2$:
$$
\|f*f\|_\infty \;\ge\; \int_{-1}^{1} \tfrac12 \operatorname{sinc}^2(\pi\xi/2) \cos^2(\pi\xi/2)\,d\xi.
$$
This has a closed form and sits below 1.276, but it is a *starting triangle*
that can then be **convex-combined** with MV's arcsine $g$, or **perturbed** in the
Kolountzakis-Revesz class of "spectral convex" Turan-admissible `g`'s
(`g = mathbb 1_{[-1/4,1/4]} * h` for positive definite `h` on `[-1/4,1/4]`).
The resulting family lives in the same dual admissibility class
(A)-(D) from `theory.md` but is **not** exhausted by the arcsine/cosine ansatz
of MV — so it escapes the 1.276 ceiling argument.

**Why this is new for us**: MV use cosine $g$ with arcsine weight $K$. The
Turan-extremizer family $g = (\mathbb 1_{K/2}*\mathbb 1_{K/2}/|K/2|) \cdot P(t^2)$
with $P$ a non-neg polynomial has $\hat g \ge 0$ by Schur product (since the
Gaussian-poly F2 family already uses this trick) AND has **sharp** Turan mass
properties on $K$. Implementing as F5 in `delsarte_dual/` would test whether the
sharp Turan extremizer's narrower spectral support beats MV's arcsine transform.

---

## Inequality 2: Prolate / Logvinenko-Sereda concentration on $|\hat f|^2$

The rigorous dual in `theory.md` uses
$|\hat f(\xi)|^2 \ge w(\xi) = \cos^2(\pi\xi/2) \mathbb 1_{[-1,1]}$,
which decays to 0 at $|\xi|=1$ but is 0 for $|\xi|>1$. However, Paley-Wiener
gives that $\hat f$ is entire of exponential type $\pi/2$, so $\hat f$ is
**never** compactly supported — it extends globally.

**Logvinenko-Sereda / Kovrijkine (sharp constant, Reznikov 2010
"Sharp constants in the Paneyah-Logvinenko-Sereda theorem"):**
for $\hat f$ in the Paley-Wiener class $PW_\sigma$ (type $\sigma$),
$$
\|\hat f\|_{L^2(\mathbb R)} \le C(\gamma, \sigma) \|\hat f\|_{L^2(E)}
\quad \text{whenever } E \subset \mathbb R \text{ is } \gamma\text{-thick}.
$$

For our problem, $\sigma = \pi/2$, so
$$
\int_{\mathbb R} |\hat f|^2 = \int_{-1/4}^{1/4} f^2 \ge 1
$$
(Cauchy-Schwarz on $\int f = 1$, support in interval of length $1/2$).
Applied with $E = [-T,T]$:
$$
\int_{-T}^{T} |\hat f(\xi)|^2\,d\xi \;\ge\; c(T) \int_{\mathbb R} |\hat f|^2 \;\ge\; c(T),
$$
where $c(T)$ is the **sharp prolate eigenvalue** $\lambda_0(T,\pi/2)$ of the
operator $P_{[-T,T]} \mathcal F P_{[-1/4,1/4]} \mathcal F^{-1} P_{[-T,T]}$.
These are tabulated (Slepian 1978; Osipov-Rokhlin 2013) and satisfy
$\lambda_0 \to 1$ exponentially fast as $T$ grows beyond $1/\pi$.

**Application.** For admissible $g$ with $\hat g$ supported in $[-T,T]$
(the Selberg-Cohn-Elkies regime of Section 3 of `theory.md`):
$$
\int_{\mathbb R} \hat g(\xi) |\hat f(\xi)|^2\,d\xi
\;\ge\; \bigl(\min_{[-T,T]} \hat g\bigr) \cdot \lambda_0(T, \pi/2).
$$
For $T$ where $\lambda_0$ is close to 1 but $\min \hat g$ stays positive, this
beats the pointwise $\cos^2$ minorant integrated against $\hat g$ because it
captures the **concentration** (not just pointwise lower bound) of $|\hat f|^2$.
This is a genuinely new input: $\cos^2(\pi\xi/2)$ alone integrates to $1/2$
on $[-1,1]$ and loses a factor of 2; the prolate bound is *tight* on the
dominant eigenspace.

**Concrete first step**: pick $T = 1$ (the length beyond which $\cos^2$ vanishes).
The prolate eigenvalue $\lambda_0(1, \pi/2) = \lambda_0(c)$ at Slepian parameter
$c = \pi/2 \cdot 1 = \pi/2 \approx 1.5708$. From Slepian's tables,
$\lambda_0(\pi/2) \approx 0.9567$. Then any admissible positive-definite $g$ with
$\hat g$ supported in $[-1,1]$ and $\min_{[-1,1]} \hat g = m > 0$ gives
$$
\|f*f\|_\infty \ge \frac{m \cdot 0.9567}{M_g}.
$$
Optimizing the ratio over the Selberg-Fejer family `F1` with $T=1$ should recover
MV-like bounds but **without** the arcsine/cosine compatibility constraint.

---

## Concrete next steps (if implemented)

1. **F5 module**: Siegel-Turan extremizer family
   $g(t) = (1-2|t|)_+ \cdot P(t^2)$ with $P$ positive-poly. PD certificate
   comes from convolving $\mathbb 1$ with $P$-weighted triangles; `numpy` / `sympy`
   closed forms analogous to F1.
2. **Prolate-weighted dual**: replace the minorant `w(ξ)=cos²(πξ/2)` in
   `delsarte_dual/optimise.py` by a *linear combination* `α·w(ξ) + β·λ₀·δ_{[-T,T]}`
   where the prolate mass-lower-bound contributes a rectangular pulse of height
   $\lambda_0 / 2T$. Tune $(α,β,T)$. Expected lift: 0.002 to 0.01 over 1.2748.
3. **Sanity**: both collapse to the MV bound when constrained to the
   arcsine/cosine ansatz, so they are strict generalisations.

---

## Sources

- [Kolountzakis-Revesz, *On a problem of Turan about positive definite functions*, arXiv:math/0204086](https://arxiv.org/abs/math/0204086)
- [Gorbachev-Tikhonov, *The Turan Problem and Its Dual for Positive Definite Functions Supported on a Ball*, J. Fourier Anal. Appl. 2024](https://link.springer.com/content/pdf/10.1007/s00041-024-10068-0.pdf)
- [Reznikov, *Sharp constants in the Paneyah-Logvinenko-Sereda theorem*, Numdam 2009](http://www.numdam.org/articles/10.1016/j.crma.2009.10.029/)
- [Slepian, *Prolate spheroidal wave functions, Fourier analysis and uncertainty -- IV* (Bell Labs 1978)](https://ieeexplore.ieee.org/abstract/document/6773659/)
- [Jaming-Iosevich, *Uncertainty Principle, annihilating pairs and Fourier restriction*, arXiv:2502.13786](https://arxiv.org/abs/2502.13786)
- [Barnard-Steinerberger / follow-up, *On optimal autocorrelation inequalities on the real line*, CPAA 2021](https://www.aimsciences.org/article/doi/10.3934/cpaa.2020271)
- [Cohn-Elkies, *New upper bounds on sphere packings I*, Annals 2003](https://annals.math.princeton.edu/wp-content/uploads/annals-v157-n2-p09.pdf)
- [Matolcsi-Vinuesa, arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
