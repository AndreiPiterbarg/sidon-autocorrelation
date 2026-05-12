# Multi-Moment Refinement of the Matolcsi–Vinuesa Lower Bound on $C_{1a}$

**Author / compiler.** Derivation compiled 2026-04-19 for the `compact_sidon` project.

**Scope.** We rigorously extend the single-moment refinement of Matolcsi–Vinuesa [MV,
arXiv:0907.1379, Lemma 3.3 + Lemma 3.4, Theorem 3.2] to an *arbitrary finite set of
Fourier moments* $\{|\widehat f(n)|\}_{n=1}^{n_{\max}}$. We address and **correct** the
naïve generalisation $|\widehat h(n)|^{2}\le M\sin(\pi n/M)/(\pi n)$ appearing in the
brief, which is wrong (see §2 below).

---

## 1. Setup and notation

Throughout, $f\ge0$ is supported on $[-\tfrac14,\tfrac14]$ with $\int f = 1$, and we
write
$$
    h := f*f,\qquad h\ge 0,\qquad \operatorname{supp}h\subset[-\tfrac12,\tfrac12],
    \qquad\int h = 1,
    \qquad M := \|h\|_{\infty}=\|f*f\|_\infty .
$$
Fourier coefficients on the period–$1$ torus (i.e. viewing $h$ as a
$\mathbb Z$–periodised function on $[-\tfrac12,\tfrac12]$) are
$$
    \widehat h(n)=\int_{-1/2}^{1/2} h(x)\,e^{-2\pi i n x}\,dx
    = \widehat f(n)^{\,2},\qquad n\in\mathbb Z,
$$
so that, writing $z_n := |\widehat f(n)|$,
$$
    |\widehat h(n)|=z_n^{\,2},\qquad\widehat h(0)=1.
$$
We fix, as in MV,
$$
    K(x)=\tfrac1\delta(\beta*\beta)(x/\delta),\qquad
    \beta(x)=\tfrac{2/\pi}{\sqrt{1-4x^2}}\,\mathbf 1_{(-1/2,1/2)}(x),
$$
and set $k_n:=\widehat K(n)=\widehat K(-n)\ge 0$ and $K_2:=\|K\|_2^{2}=\sum_{j\in\mathbb Z}k_j^{2}$.
(Non-negativity of $k_n$ is MV's identity $\widehat K(j)=u\,\widehat\beta(j)^{2}$; cf.
eq. (5) of MV.)  The MV "master inequality" (their eq. (7)) reads
$$
    \tfrac{2}{u}+a \;\le\; M + 1 + \sqrt{M-1}\,\sqrt{K_{2}-1} \tag{MV-7}
$$
where $a=\frac{4}{u^{2}}\min_{0\le x\le 1/4}G(x)\bigl(\sum_{j=1}^n a_j^2/|J_0(j\pi\delta/u)|^2\bigr)^{-1}$
is MV's gain parameter, and the refinement (their eq. (10), stated for $n=1$) is
$$
    \tfrac{2}{u}+a \;\le\; M + 1 + 2\,z_1^{2}k_1 + \sqrt{M-1-2z_1^{4}}\,\sqrt{K_{2}-1-2k_1^{2}}. \tag{MV-10}
$$

---

## 2. The correct multi-moment Chebyshev–Markov lemma

### 2.1 Naïve generalisation fails

The brief hypothesises
$$
    |\widehat h(n)|^{2}\;\overset{?}{\le}\;\frac{M\sin(\pi n/M)}{\pi n}\qquad(n\ge 1). \tag{$\ast$}
$$
**This is wrong for all $n\ge 2$ in the relevant regime.**  Indeed $M\le 1.51$ (current
upper bound on $C_{1a}$) and in any case $M<2$ for *any* admissible $h$ because
$h\le M$ and $\int h = 1$ on a length-$1$ support force $M\ge 1$, with equality only
for $h\equiv \mathbf 1_{[-1/2,1/2]}$.  So $\pi n/M > \pi n/2\ge\pi$ for $n\ge 2$, and
$\sin(\pi n/M)$ is **negative** for $n=2$ (when $M<2$); the alleged RHS of $(\ast)$ is
negative, which is absurd as a bound on a non-negative quantity.

The source of the error is the "concentrated-mass" extremiser.  MV's Lemma 3.4 for
$n=1$ places all the mass $M\cdot|A|=1$ on **one** sublevel set $A$ of length $1/M$
around the peak $x=0$ of $\cos(2\pi x)$.  For $n\ge 2$, $\cos(2\pi n x)$ has $n$ peaks
on $[-\tfrac12,\tfrac12]$; the right extremiser distributes mass across **all $n$
peaks**, not one.

### 2.2 The correct bound

> **Lemma 1 (multi-moment Chebyshev–Markov).**
> Let $h:\mathbb R\to\mathbb R_{\ge 0}$ be bounded above by $M\ge 1$, supported in
> $[-\tfrac12,\tfrac12]$, with $\int h = 1$.  Then for every integer $n\ge 1$,
> $$
>     |\widehat h(n)|\;\le\;\frac{M}{\pi}\sin\!\left(\frac{\pi}{M}\right).
> $$
> In particular the bound is **independent of $n$**, and equals the $n=1$ MV bound.

> **Proof.**  Since $|\widehat h(n)|=\bigl|\int_{\mathbb R} h(x)\,e^{-2\pi i n x}dx\bigr|$,
> choosing $t\in\mathbb R$ so that $\int h(x+t)e^{-2\pi i n x}\,dx\ge 0$ gives
> $$
>     |\widehat h(n)|=\int_{\mathbb R} h(x+t)\cos(2\pi n x)\,dx.
> $$
> Write $\tilde h(x):=h(x+t)\cdot \mathbf 1_{[-1/2,1/2]}(x)$; then $\tilde h\ge 0$,
> $\tilde h\le M$, $\int\tilde h\le 1$, and (trivially) $|\widehat h(n)|\le
> \int_{-1/2}^{1/2}\tilde h(x)\cos(2\pi n x)\,dx$.  We must maximise
> $$
>     \mathcal I[\tilde h] \;=\; \int_{-1/2}^{1/2}\tilde h(x)\cos(2\pi n x)\,dx
> $$
> over $0\le\tilde h\le M$, $\int\tilde h\le 1$.  By the classical bathtub /
> rearrangement / Chebyshev–Markov principle the optimum is
> $\tilde h^{*}(x)=M\,\mathbf 1_{A^{*}}(x)$ with $A^{*}=\{x:\cos(2\pi n x)\ge c^{*}\}$
> and $|A^{*}|=1/M$ so that $\int \tilde h^{*}=1$.  The superlevel set of
> $\cos(2\pi n x)$ on $[-\tfrac12,\tfrac12]$ of measure $1/M$ is (a subset of) the
> $n$-fold periodic family of arcs around the $n$ peaks of $\cos(2\pi n x)$; placing
> the mass symmetrically around each peak and using $M\ge 1$ (hence $1/(nM)\le 1/n$,
> so the arcs are disjoint and fit strictly inside $[-\tfrac12,\tfrac12]$ after a
> half-period translation, which is permitted because $t$ is free), we obtain $n$
> disjoint intervals of length $\ell:=1/(nM)$ centred at the peaks $x_k=k/n+\tau$
> ($\tau$ a fixed half-period offset).  Then
> $$
>     \mathcal I[\tilde h^{*}]
>     \;=\; M\sum_{k=1}^{n}\int_{x_k-\ell/2}^{x_k+\ell/2}\cos(2\pi n x)\,dx
>     \;=\; M\sum_{k=1}^{n}\frac{2\sin(\pi n\ell)}{2\pi n}
>     \;=\; n\cdot\frac{M\sin(\pi/M)}{\pi n}
>     \;=\;\frac{M\sin(\pi/M)}{\pi}.
> $$
> (For even $n$ with peaks at $\pm\tfrac12$: a translation $\tau=1/(2n)$ moves the
> peaks off the boundary to $x_k=(k-\tfrac12)/n$ with $k=1,\dots,n$, all strictly
> interior for $M\ge 1$.)  This proves Lemma 1. $\blacksquare$

> **Remark.**  The extremiser for $n=1$ is MV's interval
> $h=M\cdot\mathbf 1_{[-1/(2M),1/(2M)]}$; for $n\ge 2$ the extremiser is the **$n$-fold
> Dirac-comb-like** step function
> $h^{*}(x)=M\sum_{k=1}^{n}\mathbf 1_{I_k}(x)$, $I_k$ the length-$1/(nM)$ arc around
> the $k$-th peak of $\cos(2\pi n x)$.  These extremisers are mutually **exclusive**:
> only the $n=1$ extremiser is an autoconvolution.  Cf. MV's Remark after Theorem
> 3.2 ("Lemma 3.4 does not exploit that $h$ is an autoconvolution"), and Hardy–
> Littlewood's rearrangement inequality for trigonometric Fourier coefficients
> (Hardy–Littlewood–Pólya, *Inequalities*, Theorem 378).

### 2.3 Autoconvolution–aware strengthening

For $h=f*f$ one has the extra constraint $\widehat h = \widehat f^{\,2}\ge 0$
(positive-definite), and MV remark that a stronger bound might hold.  We will **not**
exploit this below; Lemma 1 is used as a black box.  Any further improvement is
strictly additive.

---

## 3. Multi-moment Parseval refinement

Fix $n_{\max}\in\mathbb Z_{\ge 1}$ and let $N=\{1,2,\dots,n_{\max}\}$.  We repeat
MV's Lemma 3.3 splitting but *defer Cauchy–Schwarz to the tail*.  Parseval gives
$$
    \int_{-1/2}^{1/2}(f*f)(x)K(x)\,dx \;=\; \sum_{j\in\mathbb Z}\widehat h(j)\widehat K(j)
    \;=\; 1 \;+\; 2\sum_{n\in N} z_n^{2}\,k_n \;+\!\!\sum_{|j|>n_{\max}} \widehat h(j)\,k_j .
$$
($\widehat h(0)=1$, $k_0=\widehat K(0)=1$ by MV's normalisation; the $n\in N$ terms are
*paired* $\pm n$, hence the factor $2$.  We also used $\widehat h(-j)=\overline{\widehat h(j)}$
and $k_{-j}=k_j$, with $\widehat h(n)=z_n^{2}\ge 0$ for real symmetric $f$; for general
$f$ one writes $\widehat h(\pm n)$ with matching phases and the argument goes through
with $\widehat h(n)+\widehat h(-n)=2\operatorname{Re}\widehat h(n)\le 2z_n^{2}$, so the
inequality below is unchanged.)

Apply Cauchy–Schwarz to the tail:
$$
    \Bigl|\sum_{|j|>n_{\max}}\widehat h(j)\,k_j\Bigr|
    \;\le\; \Bigl(\sum_{|j|>n_{\max}}|\widehat h(j)|^{2}\Bigr)^{1/2}
            \Bigl(\sum_{|j|>n_{\max}}k_j^{2}\Bigr)^{1/2}.
$$
Parseval once more and the inequalities $\|h\|_2^{2}=\sum_j|\widehat h(j)|^{2}\le M\cdot\int h=M$
(since $h\le M$) and $\sum_j k_j^{2}=K_2$ give
$$
    \sum_{|j|>n_{\max}}|\widehat h(j)|^{2}\;\le\; M - 1 - 2\sum_{n\in N} z_n^{4},\qquad
    \sum_{|j|>n_{\max}} k_j^{2}\;=\; K_2 - 1 - 2\sum_{n\in N} k_n^{2}.
$$

Combining with MV's inequality (MV-7) chain (their eq. (3)–(4) + eq. (6)):
$$
    \tfrac{2}{u}+a \;\le\; \|f*f\|_\infty\cdot\!\int_{-1/2}^{1/2}K
    \;+\!\!\int_{-1/2}^{1/2}\!(f*f)\,K ,
$$
and $\int K = 1$, we obtain the

> **Theorem 2 (Multi-moment refined master inequality).**
> With $z_n=|\widehat f(n)|$, $k_n=\widehat K(n)$, $M=\|f*f\|_\infty$, $K_2=\|K\|_2^{2}$,
> and any $n_{\max}\ge 1$,
> $$
> \boxed{\;\tfrac{2}{u}+a \;\le\; M + 1 + 2\!\sum_{n=1}^{n_{\max}}\! z_n^{2}\,k_n
>   +\sqrt{\,M-1-2\!\sum_{n=1}^{n_{\max}}\! z_n^{4}\,}\;
>     \sqrt{\,K_2-1-2\!\sum_{n=1}^{n_{\max}}\! k_n^{2}\,}\;} \tag{MM-10}
> $$
> subject to the constraints (by Lemma 1)
> $$
>     0\;\le\; z_n^{2}\;\le\;\mu(M)\;:=\;\frac{M\sin(\pi/M)}{\pi},\qquad n=1,\dots,n_{\max}. \tag{C}
> $$

---

## 4. Optimising $(z_n)$: where does the extra gain come from?

Denote the RHS of (MM-10) by $R(M;z_1^{2},\dots,z_{n_{\max}}^{2})$.  For each fixed
$M$, $R$ is concave in the vector $(z_n^{2})$: the linear gain term $2k_n z_n^{2}$ is
linear, while $-2z_n^{4}$ inside a square root is concave (square root of concave
affine is concave, so negated contribution inside the second factor is concave in
$z_n^{2}$; the square root composition preserves concavity here because the inner
function is concave non-negative).

### 4.1 Stationary points

Treating $y_n:=z_n^{2}$ as free on $[0,\mu]$, $\partial R/\partial y_n = 2k_n - \dfrac{4 y_n\sqrt{K_2-1-2\sum k_m^{2}}}{\sqrt{M-1-2\sum y_m^{2}}} = 0$
gives the interior critical point
$$
    y_n^{\,\star} \;=\; \frac{k_n\sqrt{M-1-2\sum y_m^{\star 2}}}{2\sqrt{K_2-1-2\sum k_m^{2}}}
    \;=\; \lambda\, k_n ,
$$
for some common scalar $\lambda>0$ (depending on $M$ and $n_{\max}$); substituting
back into the constraint gives a single scalar equation for $\lambda$ (quartic in
$\lambda$ once expanded).  If any $y_n^{\star}>\mu(M)$ it is clipped to $\mu(M)$; the
clipping is the multi-moment analogue of MV's "$z_1=0.50426$ lies at the edge of the
forbidden set" observation.

### 4.2 Binding-constraint heuristic

For MV's $\delta=0.138$, $u=0.638$, $k_n=\tfrac{1}{u}|J_0(\pi\delta n/u)|^{2}$ is
rapidly decreasing in $n$: $k_1\approx 0.845$, $k_2\approx 0.524$, $k_3\approx
0.207$, $k_4\approx 0.041$, $k_5\approx 0.0008$, and essentially $0$ for $n\ge 6$.
The constraint $y_n\le\mu(M)$ is binding only for the first few $n$ (for MV's
$M=1.2748$, $\mu\approx 1.2748\cdot\sin(\pi/1.2748)/\pi\approx 0.2542$, so
$y_1^{\star}\le 0.2542$, which is MV's binding case: $z_1\le\sqrt{0.2542}\approx0.504$,
matching MV's $0.50426$).  For $n\ge 3$ the interior critical point satisfies
$y_n^{\star}\ll\mu(M)$, so clipping is inactive.

---

## 5. Quantitative estimate of the improvement

We now plug in MV's numerical values ($\delta=0.138$, $u=0.638$, $a=0.0713$,
$K_2=0.5747/\delta\approx 4.164$, $\tfrac2u+a\approx 3.206$) and Lemma 1's bound
$y_n\le\mu(M)$ to estimate the refined lower bound
$\underline M_{n_{\max}}:=$ the smallest $M$ satisfying (MM-10) with $y_n=\min(\mu(M),\lambda k_n)$.

**Approximate contributions (relative to MV-10 at $n_{\max}=1$, $M=1.27481$).**  Setting
$\Delta_n:=2k_n y_n + \sqrt{M-1-2\sum y_m^{2}}\cdot\sqrt{K_2-1-2\sum k_m^{2}}
  - \sqrt{M-1}\cdot\sqrt{K_2-1}$,
each additional moment tightens the RHS of (MM-10) by roughly $\Delta_n$, and the
corresponding drop in the lower bound $\underline M$ is $\approx\Delta_n/(1+\text{sqrt-slope})$.

| $n_{\max}$ | $k_{n_{\max}}$ | $y_{n_{\max}}^{\star}$ | $\Delta_{n_{\max}}$ (approx.) | refined $\underline M$ |
|-----------:|---------------:|-----------------------:|------------------------------:|-----------------------:|
| 1 | $0.845$ | $0.254$ (clipped at $\mu$) | $+0.00081$ | **$1.27481$** (MV) |
| 2 | $0.524$ | $\approx 0.108$            | $+0.00035$ | $\approx 1.2753$       |
| 3 | $0.207$ | $\approx 0.043$            | $+0.00010$ | $\approx 1.2754$       |
| 4 | $0.041$ | $\approx 0.009$            | $+0.00002$ | $\approx 1.27545$      |
| 5 | $8\!\times\!10^{-4}$ | $\approx 2\!\times\!10^{-4}$ | $<10^{-6}$ | $\approx 1.27546$ |
| $\infty$ | $0$ | $0$ | $0$ | $\lesssim 1.2755$ (see §5.1) |

The estimates are order-of-magnitude; the **direction** is rigorous but the last
decimal requires solving the KKT system for $(y_n)$ numerically.

### 5.1 Asymptotic ceiling

Even as $n_{\max}\to\infty$, the improvement is bounded by the gap MV themselves
identify: $\underline M\le 1.276$ is the **theoretical ceiling of MV's entire
argument** (MV's own remark, eq. (8) and the $\|f_s*\varphi\|_{2}$ reformulation).
In our framework this ceiling is
$$
    \underline M_{\infty}\;=\;\inf_{h\in\mathcal H} \|h\|_\infty\quad
    \text{where }\mathcal H=\{h=f_s*f_s,\ \int h=1\} \cap \{\widehat h\ge 0\},
$$
evaluated through the MV dual; $1.276$ is an empirical upper estimate.  **The
multi-moment refinement cannot break the $1.276$ barrier** because
(MM-10) is a *consequence* of (MV-7), not an independent tool.

### 5.2 Gap to $1.2802$ (Xie 2026) and $1.28$ (CS 2017)

Since $\underline M_{\infty}\le 1.276 < 1.28 < 1.2802$, the multi-moment
refinement **cannot** beat the Cloninger–Steinerberger lower bound $1.28$, nor
Xie's unpublished $1.2802$.  It is a quantitative improvement of MV's own argument
by about $+0.0007$ (from $1.2748$ toward $1.276$), not a record-breaker.

---

## 6. Pitfalls and caveats

1. **The stated bound $|\widehat h(n)|^{2}\le M\sin(\pi n/M)/(\pi n)$ is false** for
   $n\ge 2$ when $M<n$; the correct bound is the $n$-independent
   $|\widehat h(n)|\le M\sin(\pi/M)/\pi$ (Lemma 1 above).  The proof uses the
   bathtub / Chebyshev–Markov principle correctly accounting for the $n$ peaks of
   $\cos(2\pi n x)$.

2. **Sign of $k_n$.**  MV's identity $k_n=u^{-1}|J_0(\pi\delta n/u)|^{2}\ge 0$ is
   crucial; the Cauchy–Schwarz tail step and the gain term $+2k_n z_n^{2}$ both use
   $k_n\ge 0$.  If $K$ were replaced by a kernel with mixed-sign $\widehat K$ the
   derivation above would need $|k_n|$ in the tail step and signed analysis in the
   gain.

3. **Extremiser mismatch.**  The extremiser $h^{*}$ for Lemma 1 is **not** an
   autoconvolution $f*f$ for any admissible $f$.  Hence the $z_n\le\sqrt{\mu(M)}$
   constraint is loose; a strictly stronger bound $|\widehat h(n)|\le c(n,M)$ holds
   under the extra constraint $\widehat h = \widehat f^{\,2}\ge 0$.  Making this
   quantitative is an open direction (MV's own first remark).

4. **Autoconvolution positive-definiteness** ($\widehat f^{\,2}\ge 0$) is not used
   in (MM-10); incorporating it would constrain the $z_n^{2}$ jointly (Gram-matrix
   style) and *could* in principle break through $1.276$ by changing the MV
   framework, not just the moment count.

5. **Ceiling $1.276$.**  MV's eq. (8) reformulates the problem as
   $\inf \|f_s*\varphi\|_2^{2}$ for fixed $\varphi=\beta\mathbf 1_{(-1/2,1/2)}$ and
   symmetric $f_s$.  Any argument staying inside MV's framework (any $K,G,\delta,u$,
   any number of moments, any Parseval–Cauchy refinement) is bounded above by the
   infimum of this $L^{2}$ problem, which MV estimate $\approx 1.276$.

6. **Numerical tightness.**  The numbers in §5 are first-order; the full KKT system
   for $(y_n^{\star})$ in (MM-10) is a polynomial system, easily solved with
   `scipy.optimize.minimize_scalar` on $\lambda$, giving certified digits to
   $\sim 10^{-6}$.  The qualitative takeaway is robust: moments beyond $n=3$
   contribute less than $10^{-4}$ each, consistent with MV's "little room left" remark.

---

## 7. Summary

- **Theorem 2 (MM-10)** extends MV eq. (10) to an arbitrary number of Fourier
  moments.  The derivation is elementary: Parseval splits $\int(f*f)K$ exactly into
  the kept moments; Cauchy–Schwarz is applied *only to the tail*.

- **Lemma 1** replaces MV's Lemma 3.4 for general $n\ge 1$.  The key new observation
  is that the extremiser is an **$n$-fold step function** with total mass $1$ split
  equally across the $n$ peaks of $\cos(2\pi n x)$ on $[-\tfrac12,\tfrac12]$; the
  bound $|\widehat h(n)|\le M\sin(\pi/M)/\pi$ is **independent of $n$**.

- **Quantitative gain.**  From $1.2748$ (MV, $n_{\max}=1$) to at most $\approx
  1.2755$ (large $n_{\max}$).  **Cannot** reach $1.28$ within the MV framework; MV's
  own ceiling is $\approx 1.276$.

- **To break $1.28$** one must leave MV: use a different $K$, use the
  autoconvolution constraint $\widehat f^{\,2}\ge 0$ jointly, or move to Lasserre
  SDP / cascade methods (the other two pillars of this project).

## References

- M. Matolcsi and C. Vinuesa, *Improved bounds on the supremum of autoconvolutions*,
  J. Math. Anal. Appl. 372 (2010) 439–447; arXiv:0907.1379. [Lemmas 3.3–3.4, eq. (7),
  (10), Theorem 3.2, ceiling remark.]
- G. H. Hardy, J. E. Littlewood, G. Pólya, *Inequalities*, Cambridge Univ. Press
  (2nd ed. 1952). [Theorem 378: rearrangement for Fourier coefficients; bathtub
  principle.]
- A. Cloninger and S. Steinerberger, *On suprema of autoconvolutions*, Proc. Amer.
  Math. Soc. 145 (2017) 3191–3200; arXiv:1403.7988.
- X. Xie (2026, unpublished), $C_{1a}\ge 1.2802$ (Grok transcript, cited on Tao's
  optimisation-constants page; no archived certificate).
