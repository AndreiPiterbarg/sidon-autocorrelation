# Optimisation review — improvements to `delsarte_dual/`

Expert review after initial implementation. Items ordered by expected impact on the **certified lower bound `lb_low`**, not by code effort. Many of these compound.

---

## A. Analytical / mathematical improvements (big bound lifts)

### A1. **Use $\int_{[-1/2,1/2]} g_+$ as the denominator, not $\max g$.** *[Major win. Single biggest bound lift.]*

The rigorous dual inequality, derived from $\int g(f*f)\le \|f*f\|_\infty \int g_+$:
$$
\|f*f\|_\infty\ \ge\ \frac{\int \widehat g(\xi)\,|\widehat f(\xi)|^2\,d\xi}{\int_{[-1/2,1/2]} g_+(t)\,dt}
\ \ge\ \frac{\int \widehat g(\xi)\,w(\xi)\,d\xi}{\int_{[-1/2,1/2]} g_+(t)\,dt}.
$$
Since $\int_{[-1/2,1/2]} g_+\le \max_{[-1/2,1/2]} g$ (on an interval of length 1), the denominator shrinks and the bound tightens.

**Expected lift**: for the plain Gaussian with $\alpha=4$, $g(t)=e^{-4t^2}$:
- $\max g=1$ (at $t{=}0$)
- $\int_{-1/2}^{1/2}e^{-4t^2}dt=\tfrac{\sqrt\pi}{2}\operatorname{erf}(2)\approx 0.8820$.

So the denominator drops by ~12% ⇒ bound rises $0.668 \to 0.758$ **for the same $g$**. A similar lift applies to F1.

Cost: need a rigorous enclosure of $\int_{[-1/2,1/2]} g$ (easy: closed form for F2, piecewise analytic for F1) AND need to handle $g_+$ if $g$ takes negative values on $[-1/2,1/2]$. Simpler: **impose the extra constraint $g\ge 0$ on $[-1/2,1/2]$** and use $\int g$ directly.

### A2. **Sharper weight $w(\xi)$ using higher moments.** *[Medium-big win.]*

Current: $w(\xi)=\cos^2(\pi\xi/2)\cdot \mathbb 1_{|\xi|\le 1}$. Derivation: one-term Taylor of $\cos(2\pi x\xi)$ with $|x|\le 1/4$.

Sharper: note that for $f\ge 0$ with $\int f=1$, $\int xf\,dx=:\mu\in[-\tfrac14,\tfrac14]$ and $\int x^2 f\,dx=:\sigma^2\in[0,\tfrac{1}{16}]$. Then
$$
\operatorname{Re}\widehat f(\xi)=\cos(2\pi\mu\xi)-\tfrac{(2\pi\sigma\xi)^2}{2!}\cos(2\pi\tilde\mu\xi)+\cdots
$$
Taking the worst-case $\mu=\pm 1/4$ and using second-order Taylor with remainder gives a tighter $w$ on $|\xi|\le 1$ **and a nontrivial $w$ on $|\xi|>1$**. This is a genuine improvement, not a "guess".

Alternative: **Chebyshev extremal problem.** Find
$$
w^\star(\xi):=\inf\{|\widehat f(\xi)|^2\;:\;f\ge 0,\ \operatorname{supp} f\subset[-\tfrac14,\tfrac14],\ \int f=1\}.
$$
This is a linear-fractional extremal problem; its solution can be computed numerically by Chebyshev-Markov-Krein quadrature for each $\xi$ and fed into the outer optimisation as a *rigorous* sharp weight.

**Expected lift**: 10–30% on the bound; lifts the "cost of rigor" from ~30% down toward ~10%.

### A3. **Use $\widehat g(\xi)|\widehat f(\xi)|^2 \ge \widehat g(\xi) - |\widehat g(\xi)|(1-|\widehat f|^2)$.** *[Medium win when combined with A2.]*

Since $|\widehat f|^2\le 1$ (not just $\ge w$), we also have an upper bound on $(1-|\widehat f|^2)$. Specifically, $(1-|\widehat f(\xi)|^2)\le 1-w(\xi)$ with the sharp $w^\star$ of A2. This gives a **two-sided** enclosure on $\int \widehat g|\widehat f|^2$ in the dual, which is useful when $\widehat g$ changes sign (it cannot under (B), but combinations that relax (B) become possible).

### A4. **Drop PD on all of $\mathbb R$; require PD on $[-1, 1]$ only.** *[Conceptual win; enables new families.]*

Only $\widehat g$ values **at points where $|\widehat f|^2>0$** matter. Since $|\widehat f|^2$ may be non-zero on a sparse set outside $[-1,1]$, the full $\widehat g\ge 0$ constraint is overkill. Krein's theory of "positive-definite functions on an interval" (Krein–Milman) would give a tighter constraint cone.

### A5. **Drop even-symmetry assumption.** *[Small win.]*

Current F1 and F2 are even in $t$, giving even $\widehat g$. Asymmetric $g$ (with $g(-t)\ne g(t)$) is also admissible if $\widehat g\ge 0$ (but $\widehat g$ then complex-valued, requires more care). Little-studied territory, probably small lift.

---

## B. Algorithmic / formulation improvements (enables bigger families)

### B1. **Recast the optimisation as an LP in $a$ for fixed $(T,\omega)$ — F1.** *[Huge efficiency win, enables $K=10$+.]*

For fixed $(T,\omega)$:
- $\widehat g(\xi)=\Delta_T(\xi)+\sum (a_k/2)(\Delta_T(\xi-k\omega)+\Delta_T(\xi+k\omega))$ is **linear** in $a$.
- The PD constraints $\widehat g(\xi)\ge 0$ at the finite breakpoint set are **linear** in $a$.
- $g(t)=\text{sinc}^2(Tt)\cdot(1+\sum a_k\cos(2\pi k\omega t))$ is **linear** in $a$.
- $g(t_j)\ge 0$ on a dense $t$-grid $\{t_j\}\subset[-1/2,1/2]$ are **linear** in $a$.
- $\int_{[-1/2,1/2]} g$ is **linear** in $a$ (with coefficients $\beta_k=\int \text{sinc}^2(Tt)\cos(2\pi k\omega t)\,dt$ computable in closed form).
- Numerator $\int\widehat g\cdot w$ is **linear** in $a$ (coefficients are the integrals of shifted-triangle × $w$, also closed form).

So: **LP** over $a$. Solve in milliseconds with `scipy.optimize.linprog`. Then do a 2-D search over $(T,\omega)$ externally. Eliminates all issues with Nelder-Mead getting stuck, and enables $K=10$, $K=20$ within budget.

### B2. **Recast F2 as an SDP for fixed $\alpha$.** *[Medium efficiency win.]*

For fixed $\alpha$:
- $\widehat g(\xi)=\sqrt{\pi/\alpha}\,e^{-\pi^2\xi^2/\alpha}Q(\xi^2)$ with $Q(u)=\sum q_m u^m$, $q_m$ linear in $c$.
- PD: $Q(u)\ge 0$ on $[0,\infty)$, expressible as $Q=s_0+u s_1$ with $s_0, s_1$ SOS ⇒ **SDP** constraints on $(c, s_0, s_1)$.
- $g(t)\ge 0$ on $[-1/2,1/2]$: $P(u)\ge 0$ on $[0, 1/4]$, same SOS-on-interval form ⇒ SDP.
- Numerator: $\int e^{-\pi^2\xi^2/\alpha}\cos^2(\pi\xi/2)\xi^{2m}d\xi$ — closed-form Gaussian-moment integrals.
- Denominator: $\int_{-1/2}^{1/2} e^{-\alpha t^2}P(t^2)dt$ — closed form in erf.

SDP over $(c, s_0, s_1)$ for fixed $\alpha$; outer loop over $\alpha$.

### B3. **Use continuous-Fourier SDP with a trig-polynomial $g$.** *[Big win if done right.]*

Parametrise $g(t)=\sum_{k=-N}^N c_k e^{2\pi i k t/L}$ (trig polynomial of degree $N$ on period $L$). Then:
- $\widehat g$ is a comb of Dirac-type values at $\xi=k/L$; replace by a smoothed version or use a compactly supported $g$ via convolution with a fixed Fejér kernel (same trick as `lasserre/fourier_sdp.py`).
- All constraints become finite.
- PD becomes Toeplitz-PSD (Bochner discrete).

This is exactly what the existing `lasserre/fourier_sdp.py` does, but **continuous-support** version. The continuous version is *not* bounded above by `val(d)`.

---

## C. Rigorous-verification improvements (tighter ball widths, 100x speedup)

### C1. **Tight interval enclosure for $\text{sinc}^2(Tt)$ near $t=0$.** *[100x B&B speedup.]*

Current F1 `g_iv` uses the crude bound $\text{sinc}^2\in[0,T^2]$ on any interval containing $0$. This makes the interval B&B split **thousands of times** near $t=0$ (observed: 13k splits for rel-tol $10^{-4}$).

Fix: Taylor-expand $\text{sinc}^2(Tt)=(\sin(\pi Tt)/(\pi t))^2$ with explicit remainder:
$$
\text{sinc}^2(x)=1-\tfrac{x^2}{3}+\tfrac{2x^4}{45}-\cdots,\qquad\text{sinc}^2(x)\in\Bigl[\bigl(1-\tfrac{x^2}{6}\bigr)^2,\,1\Bigr]\quad\text{for }|x|\le 1.
$$
So $\text{sinc}^2(Tt)\in[T^2\,(1-(T\pi t)^2/6)^2,\,T^2]$ for $|Tt|\le 1/\pi$. Use this when $t_\text{iv}$ is close to 0; fall back to the direct formula otherwise.

**Expected**: B&B splits $13\,000\to 100$ for F1, $0.5\text{s}\to 5\text{ms}$ per verify. This makes high-precision verification (rel-tol $10^{-10}$, G4 gate) free.

### C2. **Analytic closed-form numerator for F1.** *[Eliminates quadrature error entirely.]*

$$
\int_{-1}^{1}\Delta_T(\xi-s)\cos^2(\pi\xi/2)\,d\xi\;=\;\text{closed form}
$$
via $\cos^2(x)=(1+\cos(2x))/2$ and piecewise-linear $\Delta_T$. Compute once per $(T, s)$; $\widehat g$ contributions sum linearly over $k$.

**Expected**: numerator ball width drops from $\sim 10^{-2}$ (current, n_subdiv=1024) to $\sim 10^{-200}$ (mpmath precision). Zero quadrature error.

### C3. **Replace B&B for $\max g$ with analytic critical-point finding.** *[For F1, F2: 100x speedup.]*

$g$ is a closed-form smooth function. $g'=0$ has finitely many roots on $[-1/2,1/2]$. Use:
- Sturm sequence or Newton with certification for F1/F2,
- Evaluate $g$ at all critical points + endpoints,
- Max is certified to mpmath precision.

No interval B&B needed. Works because our $g$ is an explicit closed form.

### C4. **Rigorous Gaussian quadrature for F2.** *[Medium.]*

For F2 the numerator is $\int e^{-\pi^2\xi^2/\alpha}Q(\xi^2)\cos^2(\pi\xi/2)d\xi$. Use Gauss-Hermite (for the Gaussian factor) with known error bounds, much higher accuracy than midpoint.

### C5. **Interval contraction for PD certificate (F1).** *[Tightens ball.]*

Currently PD check returns a scalar `bool`. Track the minimum value $\widehat g_{\min}$ and include it in the ball: if $\widehat g_{\min}<0$, the ball is invalid; else the gap $\widehat g_{\min}\ge 0$ is the PD margin.

### C6. **Cache breakpoints and $\widehat g$ at them.** Simple perf.

---

## D. Computational / engineering polish

- **D1.** `iv.mpf(repr(x))` for floats — use `Fraction(x).limit_denominator(...)` + `iv.mpf([num, den])` for exact rational → exact-ball conversion. Saves repeated rounding.
- **D2.** Vectorise float64 scoring in `optimise.py` with numpy broadcasting (currently Python-level loops).
- **D3.** Replace Nelder-Mead with SLSQP with analytic gradients (once we use the LP reformulation of B1, this becomes moot).
- **D4.** Parallelise the outer $(T,\omega)$ grid search with `joblib` (already in `requirements.txt`).
- **D5.** Add `--profile` flag for `cProfile`.
- **D6.** `optimise.py._f1_score_f64` has a nested loop over grid × $K$ — vectorise.
- **D7.** F2 `Q_coefs` is recomputed each scoring call — cache by hashable params.
- **D8.** B&B priority queue: currently `float(-G.b)` for ordering loses precision at high mpmath prec; use two-level priority (float hint + exact mpf tiebreaker).
- **D9.** Interval B&B uses best-first by upper bound; try **best-first by `(upper - lower)`** (largest uncertainty first) for faster convergence when near the optimum.

---

## E. Test-suite gaps

- **E1.** No test reproduces a known value of $\widehat g(0)$ for F1 when triangles overlap ($k\omega<T$). Add.
- **E2.** No test for the conservation property: `lo <= samples <= hi` for the B&B output over a grid of 1000 sample points. Add.
- **E3.** No regression test locking in the best-known bound per family. Add after we have a stable baseline.
- **E4.** The F2 PD certificate is only tested for $N\le 1$; test $N=2$ with a Q that has a positive discriminant interior minimum.

---

## F. Correctness audit — actual bugs

- **F1.** In `rigorous_max.py`, the update `global_hi = max(e.g_hi for e in heap)` after every split is $O(|\text{heap}|)$; over 13k splits this is $O(10^8)$ work. Fix: update `global_hi` lazily, or maintain a max-heap indexed by `g_hi` as well.
- **F2.** In `family_f1_selberg.g_iv`, when `t_iv.a <= 0 <= t_iv.b`, the enclosure `[0, T^2]` is correct but loose (see C1). **Confirmed: this is the dominant loss in the B&B.**
- **F3.** `_weight_f64` is re-evaluated inside the `for xi in xs:` loop in F1 scoring — vectorise.
- **F4.** In F2 `ghat_iv`, `xi_sq` from `xi_iv * xi_iv` for an interval containing 0 gives `[0, max]` but loses information about monotonicity of the Gaussian ⇒ looser bound. Cleanest: evaluate at `abs(xi_iv)` first.
- **F5.** F1 `breakpoints` returns symmetric breakpoints even if $K=0$ — trivial but wasteful.

---

## Suggested implementation order (max bound lift for minimum code)

1. **A1** (switch denominator to $\int g$) — ~12% bound lift, 1 hour code.
2. **C1** (tight sinc² near 0) — enables 10⁻¹⁰ verification cheaply, 1 hour.
3. **C2** (analytic numerator for F1) — zero quadrature error, 2 hours.
4. **B1** (F1 as LP) — enables $K=10$, 4 hours.
5. **A2** (sharper weight $w$ via second moment) — ~15% bound lift, 1 day.
6. **B2** (F2 as SDP) — 1 day.
7. **B3** (continuous-Fourier SDP) — big, several days.

After 1–4, expected certified bound rises from **0.668 → ~0.95** for F2 alone, and from **0.464 → ~0.90** for F1 ($K=10$). Still under 1.2802 but closing.

After 5–7, we get to the analytical frontier and the gap becomes research-hard.
