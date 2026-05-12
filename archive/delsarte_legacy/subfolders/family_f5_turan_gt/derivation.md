# F5 (Turán/Gorbachev–Tikhonov) — Derivation, Status, and Verdict

This document records the rigorous derivation, implementation, and **honest
verdict** for the F5 family

$$
g(t) \;=\; (1-2|t|)_{+}\cdot P(t^{2}),\qquad P\in\mathbb{R}[u],\ P\ge 0\ \text{on }[0,1/4],
$$

supported on $[-1/2,1/2]$. The conclusion of Phases 0–5 is that **F5 is
structurally dominated** by the existing rigorous Delsarte pipeline
(`delsarte_dual/theory.md` Section 2). It is *not* a viable route to a new
unconditional rigorous lower bound $>1.27481$ on $C_{1a}$.

---

## Phase 0 — five-bullet status report

(a) **Dual formulation actually used.** Not the bare ratio $\hat g(0)/M_g$.
 `delsarte_dual/theory.md` Section 2 derives the rigorous
 $$
 \|f*f\|_\infty \;\ge\; \frac{1}{M_g}\int_{\mathbb R}\widehat g(\xi)\,w(\xi)\,d\xi,
 $$
 with the **sharp Paley–Wiener weight**
 $w(\xi)=\cos^{2}(\pi\xi/2)\cdot\mathbf{1}_{[-1,1]}(\xi)$ at line 40, or the
 even-sharper signed minorant $L^{\sharp}(\xi)=\cos(\pi|\xi|)$ on $[-1,1]$
 and $-1$ outside, used in code (`family_f1_selberg.py:240–281`).

(b) **What cone does F1 span?** $g(t)=T^{2}\operatorname{sinc}^{2}(\pi T t)\cdot
 (1+\sum_{k=1}^{K}a_{k}\cos(2\pi k\omega t))$. The prefix is
 $T^{2}\operatorname{sinc}^{2}$ (a triangle in *spectrum*, height $T$ at $\xi=0$,
 width $T$). It is **distinct** from the Turán triangle prefix
 $(1-2|t|)_{+}$ (a triangle in *time*).

(c) **Is F5 a strict superset of F1? No, it is disjoint.** F1 has a triangle
 in the Fourier domain; F5 has a triangle in the time domain. Neither
 contains the other as a special case. They can be related by the duality
 $g\leftrightarrow\hat g$, but only if the modulation factor is moved to
 the spectral side, which changes the family.

(d) **Existing rigorous-verification entry point.** `delsarte_dual/verify.py`
 `_verify_family(family_module, params, …)`. It calls
 `family_module.ghat_iv`, `family_module.g_iv`, `family_module.weight_iv`,
 `family_module.positive_definite_certificate`,
 `family_module.integral_g_iv`, `family_module.ghat_support`. Returns a
 `VerifiedBound` with `lb_low ≤ C_{1a}`.

(e) **Pre-existing reverse-direction obstacle (CRITICAL).** As recorded in
 `delsarte_dual/postmortem.md` and reproduced numerically in this work
 (Phase 0 sanity below):
 * The repo's *Delsarte ratio* on F1 maxes out near $L\approx 0.7$ (LP
 sweep over $T,\omega$, $K\le 3$).
 * **MV's 1.27481 is NOT this ratio.** MV (eq. (10), p. 7) solves a
 quadratic in $\|f*f\|_{\infty}$ obtained from
 $1+\sqrt{\|f*f\|_\infty-1}\sqrt{\|K\|_2^2-1}\le \int(f*f)K$ combined
 with a sharper Cauchy–Schwarz on $|\hat f(1)|$. It uses a *different*
 algebraic structure (`mv_construction_detailed.md`, lines 52–118).
 * Putting $g_{*}(t)=(1-2|t|)_{+}$ alone (the bare F5 prefix) in the
 Delsarte ratio gives $\hat g_{*}(0)=1/2$, $M_{g_{*}}=1$, idealised ratio
 $=0.5$; with the rigorous $\cos^{2}(\pi\xi/2)$-weighted integral it is
 $\approx 0.451$; with the sharper signed weight $L^{\sharp}$ even
 *negative* (because $\hat g_{*}=\tfrac12\operatorname{sinc}^{2}(\pi\xi/2)$ has
 *no compact support* and the $-1$ outer-tail penalty dominates).

The brief's premise that the F5 family in this Delsarte pipeline can produce
a bound $>1.27481$ is therefore based on a category error (mixing MV's
quadratic framework with the Delsarte ratio framework). This is documented
explicitly in `delsarte_dual/postmortem.md` lines 26–30 ("Anatomy of the
gap") and corroborated by the numerical evidence below.

---

## Phase 1 — rigorous derivation of the F5 dual lower bound

We re-derive in full from `delsarte_dual/theory.md` Section 2.

**Primal.** $C_{1a}=\inf\{\|f*f\|_{\infty}:f\ge 0,\ \operatorname{supp}f\subset[-\tfrac14,\tfrac14],\
\int f=1\}$.

**Admissibility class $\mathcal A$ for the test function $g\in L^{1}(\mathbb R)\cap C(\mathbb R)$.**
- (A) $g(t)\ge 0$ for every $t\in[-\tfrac12,\tfrac12]$ — **time-domain nonnegativity**.
- (B) $\hat g(\xi)\ge 0$ for every $\xi\in\mathbb R$ — **Bochner positive-definiteness**.
- (C) $\hat g(0)=\int g>0$.
- (D) $\hat g\in L^{1}(\mathbb R)$.

**Master inequality (theory.md eq. between lines 26–32).** For any admissible $g$ and any feasible $f$,
$$
M_g\,\|f*f\|_{\infty}\;\ge\;\int_{[-1/2,1/2]} g(t)(f*f)(t)\,dt
\;=\;\int_{\mathbb R}\hat g(\xi)|\hat f(\xi)|^{2}\,d\xi,
$$
where $M_{g}:=\max_{t\in[-1/2,1/2]}g(t)$. Step 1 uses (A) and the support of
$f*f\subset[-\tfrac12,\tfrac12]$; step 2 is Plancherel, valid because $g,f*f\in
L^{1}\cap L^{2}$.

**Rigorous lower bound on $|\hat f|^{2}$ (theory.md eqs (40)–(42)).**
For $|\xi|\le 1$, $\operatorname{Re}\hat f(\xi)=\int f(x)\cos(2\pi x\xi)dx\ge\cos(\pi|\xi|/2)$
since $2\pi x\xi\in[-\pi/2,\pi/2]$ on the support. Hence
$|\hat f(\xi)|^{2}\ge w(\xi):=\cos^{2}(\pi\xi/2)\mathbf 1_{[-1,1]}(\xi)$.
This is **the load-bearing step** that replaces the Dirac at zero in any
"idealised" $\hat g(0)/M_{g}$ formula. The repo also implements the *sharper*
signed minorant $L^{\sharp}(\xi)=\cos(\pi|\xi|)\mathbf 1_{|\xi|\le 1}-\mathbf 1_{|\xi|>1}$
(`family_f1_selberg.weight_iv`), which uses $|\hat f|^{2}\le 1$ globally to
penalise tail-mass of $\hat g$.

**Final rigorous bound (theory.md line 46).**
$$
\boxed{\;\|f*f\|_{\infty}\;\ge\;\frac{1}{M_{g}}\int_{\mathbb R}\hat g(\xi)\,w(\xi)\,d\xi.\;}
$$
**The ratio $\hat g(0)/M_{g}$ is NOT a valid rigorous bound** unless an
*additional* concentration inequality (Logvinenko–Sereda / prolate; `ideas_fourier_ineqs.md`
§Inequality 2) supplies the missing factor; that machinery is not implemented in this repo.

### F5 form: time-domain admissibility (A)

$g_{F5}(t)=(1-2|t|)_{+}P(t^{2})$. $(1-2|t|)_{+}\ge 0$ on $\mathbb R$, vanishes
outside $[-\tfrac12,\tfrac12]$. So (A) holds iff $P(t^{2})\ge 0$ for
$t\in[-\tfrac12,\tfrac12]$, i.e., $P(u)\ge 0$ on $[0,1/4]$. By Markov–Lukács,
on the bounded interval $[0,1/4]$ this is exact for any polynomial degree
(SOS plus weighted SOS).

### F5 Fourier transform — closed form

Write $g_{F5}=h\cdot P(t^{2})$ with $h(t)=(1-2|t|)_{+}$, supported on $[-\tfrac12,\tfrac12]$.

The Fourier transform of the triangle is exactly
$$
\hat h(\xi)\;=\;\int_{-1/2}^{1/2}(1-2|t|)e^{-2\pi i t\xi}\,dt
\;=\;\frac{\sin^{2}(\pi\xi/2)}{(\pi\xi/2)^{2}/4}\cdot\frac14
\;=\;\frac12\operatorname{sinc}^{2}(\pi\xi/2),
$$
where $\operatorname{sinc}(x)=\sin(x)/x$. Reference: Gradshteyn–Ryzhik 3.823 (Fourier
of $(1-|t|)_{+}\cdot\mathbf 1_{[-1,1]}$, scaled by 2 in $t$).

Multiplication by $t^{2k}$ in time corresponds to differentiation $(-1/(2\pi i))^{2k}\partial_{\xi}^{2k}=\frac{(-1)^{k}}{(2\pi)^{2k}}\partial_{\xi}^{2k}$ in frequency:
$$
\widehat{t^{2k}h(t)}(\xi)\;=\;\frac{(-1)^{k}}{(2\pi)^{2k}}\,\frac{d^{2k}}{d\xi^{2k}}\hat h(\xi).
$$
Reference: standard distributional derivative-Fourier identity, e.g., Stein–Weiss
"Introduction to Fourier Analysis on Euclidean Spaces" Ch. I §1.

For $P(u)=\sum_{k=0}^{K}c_{k}u^{k}$,
$$
\hat g_{F5}(\xi)
\;=\;\sum_{k=0}^{K}c_{k}\,\widehat{t^{2k}h(t)}(\xi)
\;=\;\frac12\sum_{k=0}^{K}c_{k}\frac{(-1)^{k}}{(2\pi)^{2k}}\,\frac{d^{2k}}{d\xi^{2k}}\operatorname{sinc}^{2}(\pi\xi/2).
$$

In particular,
$$
\hat g_{F5}(0)=\int g_{F5}(t)\,dt
=\sum_{k=0}^{K}c_{k}\int_{-1/2}^{1/2}(1-2|t|)t^{2k}\,dt
=\sum_{k=0}^{K}c_{k}\cdot 2\int_{0}^{1/2}(1-2t)t^{2k}\,dt.
$$
Using $\int_{0}^{1/2}(1-2t)t^{2k}dt=\frac{1}{2k+1}\cdot\frac{1}{2^{2k+1}}-\frac{2}{2k+2}\cdot\frac{1}{2^{2k+2}}=\frac{1}{(2k+1)(2k+2)2^{2k+1}}$,
$$
\boxed{\;\hat g_{F5}(0)=\sum_{k=0}^{K}\frac{c_{k}}{(2k+1)(2k+2)\,2^{2k}}.\;}
$$

### F5 spectral-positivity check (B) — STRUCTURAL OBSTRUCTION

For $P\equiv 1$: $\hat g_{F5}(\xi)=\tfrac12\operatorname{sinc}^{2}(\pi\xi/2)\ge 0$ .

For $P(u)=u$: $\hat g_{F5}(\xi)=-\frac{1}{(2\pi)^{2}}\partial_{\xi}^{2}\bigl[\tfrac12\operatorname{sinc}^{2}(\pi\xi/2)\bigr]$.
This second derivative changes sign — concretely, $\operatorname{sinc}^{2}$ is concave at 0 but
oscillates with sign-changing concavity for $|\xi|>1$. Hence (B) **fails for $P=u$
alone**. To restore (B), one must add a nonnegative multiple of the $P=1$ baseline.
But adding $\alpha\cdot 1$ to $P$ then *multiplies* the time-domain triangle by
$\alpha+u$, which raises $M_{g}$ proportionally, and the gain $\hat g(0)$ scales with
the same coefficients — net change in the ratio is small.

This is the central tension: $P(t^{2})\ge 0$ on $[0,1/4]$ for time admissibility (A),
but $\hat g_{F5}(\xi)\ge 0$ on $\mathbb R$ for spectral admissibility (B), and the two
constraints together restrict $P$ to a narrow cone (small perturbation of $P=1$).

---

## Phase 2 — implementation

`f5.py` provides:

* `g_value(t, c)` — $g_{F5}(t)=(1-2|t|)_{+}P(t^{2})$ with $P(u)=\sum c_{k}u^{k}$.
* `g_iv(t_iv, c)` — interval-arithmetic enclosure on $t\in[-1/2,1/2]$.
* `g_hat_zero(c)` — closed form $\hat g_{F5}(0)=\sum c_{k}/((2k+1)(2k+2)2^{2k})$.
* `g_hat_value(xi, c)` — closed form via $\hat h$ and its derivatives.
* `M_g(c)` — interval B&B max via `delsarte_dual.rigorous_max.rigorous_max`.
* `is_pd_admissible(c, …)` — verifies $\hat g\ge 0$ on a fine grid plus a tail
 bound; returns a certificate dict (grid, values, tail bound, decision).
* `f5_lower_bound(c)` — returns $\int\hat g(\xi)w(\xi)d\xi/M_{g}$ if PD-certified,
 else raises `NotPDAdmissible` with the certificate.

The "ratio $\hat g(0)/M_{g}$" is **not** the lower bound; we expose it only as
`f5_idealised_ratio(c)` and document that it is unsound.

## Phase 3–5 — numerical results, sanity, verdict

See `RESULTS.md` and `results.json`. Headline:

| $K$ | $P$ | rigorous $L$ | comments |
|---|---|---|---|
| 0 | 1 | $\approx 0.451$ (cos² weight) | bare F5; $\hat g(0)/M_{g}=0.5$ |
| 1 | $1+0.0\cdot u$ | same | $u$-coeff drives PD violation |
| 2 | small SOS perturbation | within $10^{-3}$ of K=0 | no improvement |

F1 (existing module) at the same precision and same weight reaches $L\approx 0.668$
(Selberg–Fejér with $T=2,\omega=2,a_{1}=0.5$, certified by `verify.py`).
F5 is therefore **dominated by F1 in the same Delsarte pipeline**, and **both are
far below MV's 1.27481 because the Delsarte ratio framework is fundamentally weaker
than MV's quadratic framework** — see `postmortem.md` lines 22–30.

**Verdict: (c) DOES NOT BEAT — F5 is dominated.**

This is consistent with the warning in `ideas_fourier_ineqs.md` line 56:
"This has a closed form and sits below 1.276."

---

## What was verified rigorously (30-line proof sketch)

Let $c=(c_{0},\dots,c_{K})$ with $P(u)=\sum c_{k}u^{k}$, and define
$g_{F5}$, $\hat g_{F5}$ by the closed forms above. Then:

1. (A) holds iff $P(u)\ge 0$ for $u\in[0,1/4]$. We verified this by
 evaluating $P$ on a uniform 1001-point grid in $[0,1/4]$ at mpmath dps=50
 *and* checking the SOS certificate (Markov–Lukács) for the chosen $c$.
2. (B) was verified numerically via `is_pd_admissible(c, xi_grid_density=20001)`
 on $|\xi|\le R=20$ at dps=50, with tail bound $|\hat g_{F5}(\xi)|\le
 C(c)/\xi^{2}$ valid for $|\xi|\ge R$. (See `f5.py:is_pd_admissible`.)
3. The Plancherel identity $\int_{[-1/2,1/2]}g_{F5}(f*f)dt=\int\hat g_{F5}|\hat f|^{2}d\xi$
 follows from $g_{F5},f*f\in L^{1}\cap L^{2}$.
4. $|\hat f|^{2}\ge w$ on $[-1,1]$ is the elementary Paley–Wiener step (theory.md).
5. $\int\hat g_{F5}\cdot w\,d\xi$ is computed in mpmath at dps=50 with rigorous
 midpoint-quadrature error bounds via `rigorous_max.rigorous_integral_with_weight`
 and `n_subdiv=8192`.
6. $M_{g_{F5}}$ is enclosed via interval B&B (`rigorous_max.rigorous_max`) with
 `rel_tol=1e-8`.
7. The certified ratio $L\le C_{1a}$ is the ball $[\underline N/\overline M,\overline N/\underline M]$.

The numerical certificate, parameter values, and sanity checks for each $K$
are in `results.json`.
