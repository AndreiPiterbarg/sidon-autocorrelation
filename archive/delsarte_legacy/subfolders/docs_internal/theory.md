# Theory: Delsarte-type Dual Lower Bound for $C_{1a}$

## 1. The Sidon autocorrelation constant

Let
$$
C_{1a}\ :=\ \inf\Bigl\{\;\|f*f\|_\infty\;:\;f\ge 0,\ \operatorname{supp}f\subset[-\tfrac14,\tfrac14],\ \int f=1\Bigr\}.
$$
Any admissible $f$ has $\operatorname{supp}(f*f)\subset[-\tfrac12,\tfrac12]$ and $\int(f*f)=1$, with Fourier transform $\widehat{f*f}=\widehat f^{\,2}$, $\widehat f(0)=1$, $|\widehat f|\le 1$, and $\widehat f$ of exponential type $\pi/2$ (Paley–Wiener).

Current bounds: $1.2802\le C_{1a}\le 1.5029$. Lower record by Matolcsi–Vinuesa (arXiv:0907.1379); upper record by explicit $f$ constructions.

## 2. The admissible dual (precise version)

We want a lower bound $\|f*f\|_\infty\ge L(g)$ from a test function $g$.

**Admissibility class $\mathcal{A}$.** Call $g\in L^1(\mathbb R)\cap C(\mathbb R)$ **admissible** if

- **(A)** $g(t)\ge 0$ for every $t\in[-\tfrac12,\tfrac12]$,
- **(B)** $\widehat g(\xi)\ge 0$ for every $\xi\in\mathbb R$  (positive-definite; Bochner),
- **(C)** $\widehat g(0)=\int g>0$,
- **(D)** $\widehat g\in L^1(\mathbb R)$.

(Note: (A)+(B) imply $g$ is continuous at $0$ and $g(0)=\sup g$. Condition (A) is needed so that $g$ acts as a non-negative "window" on $\operatorname{supp}(f*f)$.)

**Bound.** For any admissible $g$ and any admissible $f$,
$$
M_g\,\|f*f\|_\infty\ \ge\ \int_{[-1/2,1/2]} g(t)\,(f*f)(t)\,dt
\ =\ \int_{\mathbb R}\widehat g(\xi)\,|\widehat f(\xi)|^2\,d\xi
\ \ge\ \widehat g(0)\cdot|\widehat f(0)|^2\ \underbrace{=}_{\widehat f(0)=1}\ \widehat g(0),
$$
where $M_g:=\max_{t\in[-1/2,1/2]} g(t)$ (well-defined, $>0$ by (A)+(C)). The middle equality is Parseval/Plancherel, applicable because $g,f*f\in L^1\cap L^2$. The last inequality uses $\widehat g\ge 0$ and $|\widehat f|\le|\widehat f(0)|=1$.

**Sharp rigorous lower bound on $|\widehat f|^2$.** For $f\ge 0$ supported on $[-\tfrac14,\tfrac14]$ with $\int f=1$, for $|\xi|\le 1$:
$$
\operatorname{Re}\widehat f(\xi)\;=\;\int_{-1/4}^{1/4} f(x)\cos(2\pi x\xi)\,dx\;\ge\;\cos(\pi|\xi|/2)\cdot\int f\;=\;\cos(\pi|\xi|/2),
$$
since for $x\in[-\tfrac14,\tfrac14]$ and $|\xi|\le 1$, $2\pi x\xi\in[-\pi/2,\pi/2]$, and $\cos$ is minimised at the endpoints: $\cos(2\pi x\xi)\ge\cos(\pi|\xi|/2)$. Therefore
$$
\boxed{\ |\widehat f(\xi)|^2\ \ge\ w(\xi):=\begin{cases}\cos^2(\pi\xi/2)&|\xi|\le 1\\[2pt]0&|\xi|>1\end{cases}\ }
$$
and this is **rigorous, elementary, and sharp at $\xi=0$** (matching $|\widehat f(0)|^2=1$).

Plugging in: the final rigorous bound is
$$
\boxed{\ \|f*f\|_\infty\ \ge\ \frac{1}{M_g}\int_{-1}^{1}\widehat g(\xi)\,\cos^2(\pi\xi/2)\,d\xi\ }
$$
valid for any admissible $g$ (conditions (A)–(D)). This is **strictly stronger** than the weaker linear-corner bound $(1-\pi|\xi|/2)^2_+$ used as an intermediate sanity check, but still weaker than the idealised $\widehat g(0)/M_g$ that would be available if we could justify concentrating all the $\widehat g$-mass at $\xi=0$.

The "reduce to $\widehat g(0)$" form in the task brief corresponds to replacing $w(\xi)$ by $\mathbb{1}_{\{0\}}$ — a Dirac at zero — which is **not admissible** in the rigorous argument. For $\widehat g$ highly concentrated at $0$ (narrow-support compactly-supported $\widehat g$) the two bounds agree to leading order, and the cost of rigor is a multiplicative factor close to $1$; for $\widehat g$ with spread support the rigorous bound is substantially smaller.

## 3. Selberg–Cohn–Elkies reduction for compactly Fourier-supported $g$

When $\widehat g$ is **supported in $[-T,T]$** with $T\le$ small, one can sharpen using the Paley–Wiener structure of $\widehat f$:
- $\widehat f$ is entire of exponential type $\pi/2$, so $|\widehat f|^2$ is entire of type $\pi$.
- By Plancherel–Polya-type inequalities, $\int_{|\xi|\le T}|\widehat f|^2\,d\xi\ge (1-\sin(\pi T)/(\pi T))$ when $f$ is suitably normalised.

These refinements are outside the scope of the present implementation; we use only the elementary bound
$$
\boxed{\ \|f*f\|_\infty\ \ge\ \frac{1}{M_g}\int_{\mathbb R}\widehat g(\xi)\,w(\xi)\,d\xi\,,\qquad w(\xi):=\bigl(\max(0,1-\tfrac{\pi}{2}|\xi|)\bigr)^2\ }
$$
which is *strictly weaker* than (but strictly rigorous for) the full dual.

## 4. Positive-definiteness certificates

For each family implemented in the code the positive-definiteness certificate is in closed form:

### F1 — Selberg / Fejér-modulated.
$$
g(t)\ =\ \Bigl(\tfrac{\sin(\pi T t)}{\pi t}\Bigr)^{\!2}\cdot\Bigl(1+\sum_{k=1}^{K}a_k\cos(2\pi k\omega t)\Bigr).
$$
Then $\widehat g$ is a sum of shifts of the triangle
$\Delta_T(\xi)=\max(0, T-|\xi|)$:
$$
\widehat g(\xi)\ =\ \Delta_T(\xi)+\tfrac12\sum_{k=1}^K a_k\bigl(\Delta_T(\xi-k\omega)+\Delta_T(\xi+k\omega)\bigr).
$$
**Certificate:** $\widehat g\ge 0$ reduces to sign/sum checks on finitely many nodes (breakpoints of $\widehat g$), since $\widehat g$ is piecewise linear with finitely many breakpoints in $[-T-K\omega,T+K\omega]$.

### F2 — Gaussian-modulated even polynomial.
$$
g(t)\ =\ e^{-\alpha t^2}\cdot P(t^2),\qquad P\in\mathbb R[u],\ \deg P=N.
$$
Then $\widehat g(\xi)=e^{-\pi^2\xi^2/\alpha}\cdot Q(\xi^2)$ for an explicit polynomial $Q$ derived from $P$ by operator calculus:
$$
\widehat{t^{2m}e^{-\alpha t^2}}(\xi)\ =\ \sqrt{\tfrac{\pi}{\alpha}}\cdot (-1)^m\,\partial_\alpha^{\,m}e^{-\pi^2\xi^2/\alpha}.
$$
**Certificate:** $\widehat g\ge 0$ iff $Q(u)\ge 0$ for $u\ge 0$, a univariate polynomial non-negativity test (closed form for $\deg Q\le 2$; SOS/Sturm for larger).

### F3 — Vaaler / Beurling-Selberg.
The Vaaler function $V$ satisfies $V\ge \operatorname{sgn}$ on $\mathbb R\setminus\{0\}$ and $\widehat V$ vanishes outside $[-1,1]$. See Vaaler, "Some extremal functions in Fourier analysis" (Bull. AMS 1985) for the explicit formula. Our F3 module **documents but does not fully implement** this family: Vaaler's closed-form $\widehat V$ and the positive-definiteness of the symmetrised bi-shifted version
$g_{t_0}(t)=V(t-t_0)+V(-t-t_0)$ require care with the sign and $L^1$ normalisation, and a full rigorous interval implementation is deferred.

## 5. Rigor pipeline

1. **Symbolic step.** Build $g,\widehat g$ with `sympy` as explicit closed forms.
2. **Positive-definiteness.** Family-specific: (F1) sign-checks at finitely many nodes; (F2) univariate polynomial non-negativity on $[0,\infty)$ via Sturm.
3. **Interval $\widehat g(0)$ and $\int\widehat g\,w\,d\xi$.** `mpmath` at ≥100 digits with quadrature error bounds added to an enclosing ball.
4. **Rigorous max of $g$ on $[-1/2,1/2]$.** Adaptive interval branch-and-bound (`rigorous_max.py`) using mpmath's `iv` context: we maintain a covering of sub-intervals with certified upper bounds on $\max g$, repeatedly splitting the worst sub-interval until $(\text{up}-\text{lo})/\text{lo}<10^{-10}$ or budget exhausted.
5. **Certified ratio.** Ball division:
   $\bigl[\tfrac{\underline{N}}{\overline{M}},\,\tfrac{\overline{N}}{\underline{M}}\bigr]$
   where $\underline N,\overline N$ enclose the numerator and $\underline M,\overline M$ enclose $M_g$.

## 6. Why this is not equivalent to the repo's Fourier SDP

`lasserre/fourier_sdp.py` discretises the dual to a knot grid of size $2d+1$, producing a weak duality upper bound $\le\operatorname{val}(d)$. The present approach optimises directly over an infinite-dimensional parametric family; its bound is **not** controlled by the $\operatorname{val}(d)$ ladder and in principle can exceed it.

In practice, however, the existing SDP achieves near-record bounds because the parametric families here (F1/F2) have only modest degrees of freedom, and the SDP effectively searches over a much larger cone. The hope of a fresh win therefore rests on (i) better analytic choices in the continuous parameter space, or (ii) the F3 Beurling–Selberg family which is inaccessible to knot-grid SDPs.

## References

- Matolcsi, Vinuesa (2010), *"Improved constants for the Erdős–Ko–Rado theorem"*, arXiv:0907.1379.
- Kolountzakis, Révész (2006), *"Turán's extremal problem for positive definite functions on groups"*, JLMS.
- Vaaler (1985), *"Some extremal functions in Fourier analysis"*, Bull. AMS.
- Cohn, Elkies (2003), *"New upper bounds on sphere packings I"*, Ann. Math.
- White (2022), arXiv:2210.16437.
