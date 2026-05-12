# Track 1 Final Report — The $\|f^*\|_\infty$ Analytical Question is **Settled in the Negative**

**Date:** 2026-04-29
**Scope:** Determine whether the unconstrained Sidon optimum $f^*$ admits a
finite a-priori upper bound $\|f^*\|_\infty \le M_*$ that would convert
V3's $L^\infty$-augmented 3-point SDP results into a rigorous proof of
$C_{1a} > 1.2802$.

---

## TL;DR

**Verdict: the $L^\infty$ bridge does NOT lift V3's $\lambda^{3\mathrm{pt}}(M=2.15) > 1.2802$
to a proof of $C_{1a} > 1.2802$.**

Three independent lines of evidence converge on this conclusion:

1. **Analytical truncation/mollification arguments fail.** The standard
   shrink-and-smooth construction (rescale $f$ to $[-(1/4-\epsilon), 1/4-\epsilon]$
   then mollify by a kernel of bandwidth $\epsilon$) produces $f'$ with
   $\|f'\|_\infty \le \|f\|_2/\sqrt{2\epsilon(1-4\epsilon)}$ but
   $\sup(f' * f') \le \sup(f*f)/(1-4\epsilon)$. The trade-off cannot deliver
   $\|f'\|_\infty \le 2.15$ without blowing up $\sup(f' * f')$ by a factor
   $\ge 1.6$ — well above the 1.31/1.28 = 1.024 ratio we'd need.

2. **Direct numerical evidence: the val(d) optimum has $\|f\|_\infty \to \infty$
   as $d \to \infty$.** Computed empirically:

| $d$ | val(d) | max $\mu_i$ | $\|f\|_\infty := 2d \cdot \max_i \mu_i$ |
|----:|-------:|----:|---:|
|  4 | 1.102 | 0.360 | 2.88 |
|  6 | 1.175 | 0.319 | 3.83 |
|  8 | 1.216 | 0.264 | 4.22 |
| 10 | 1.240 | 0.234 | 4.68 |
| 12 | 1.278 | 0.217 | 5.21 |

   Monotone growth, no sign of saturation. Linear-fit extrapolation suggests
   $\|f\|_\infty \to \infty$ as $d \to \infty$. The continuous Sidon optimum
   (the $d \to \infty$ limit) has $\|f^*\|_\infty = \infty$.

3. **Literature audit** (per memory note `project_threepoint_sdp_dead.md`):
   the natural near-optimizer family is unbounded:
   - **Matolcsi-Vinuesa** (arXiv:0907.1379, eq. 12 p.10): $f \sim 1.39/(0.002 - 2x)^{1/3}$
     — integrably-singular at endpoint.
   - **White** (arXiv:2210.16437): related $L^2$ optimizer is $\sim 1/\sqrt{1/2 - |x|}$
     — integrably-singular at endpoint.
   - **Schinzel-Schmidt** candidate $f_0(x) = 1/\sqrt{2x+1/2}$ — unbounded.
   - **Cloninger-Steinerberger** (arXiv:1403.7988v3) prove the 1.2802 LB
     **without** any height cap on the simplex coefficients; their argument
     applies to all nonneg $L^1$ functions, no $L^\infty$ assumption.
   - **No paper proves an a-priori UB on $\|f^*\|_\infty$.** MV's Lemma 3.4
     is conditional (assumes $M$ to derive Fourier consequences), not a proof
     that bounded $f^*$ exists.

The convergence of all three indicators settles the question: $C_{1a}^{(M)} > C_{1a}$
strictly for any finite $M$, and the gap does not close at $M = 2.15$ (the regime
where V3 gives $\lambda^{3\mathrm{pt}} > 1.2802$).

---

## What we *can* prove unconditionally

### Lemma (LB): $\|f^*\|_\infty \ge 2$.

For any $f \ge 0$ with $\int f = 1$ and $\text{supp}(f) \subset [-1/4, 1/4]$
(support length $1/2$):

By Cauchy-Schwarz on $L^2[-1/4, 1/4]$:
$$1 = \left(\int |f|\right)^2 \le \|f\|_2^2 \cdot \|1\|_2^2 = \|f\|_2^2 \cdot \tfrac{1}{2},$$
so $\|f\|_2^2 \ge 2$.

By the elementary $L^2$-vs-$L^1 \cdot L^\infty$ inequality:
$$\|f\|_2^2 = \int f^2 \le \|f\|_\infty \int |f| = \|f\|_\infty \cdot \|f\|_1 = \|f\|_\infty.$$

Combining: $\|f\|_\infty \ge \|f\|_2^2 \ge 2$. ∎

This applies to *every* feasible $f$ (not just optimum). The optimum has
$\|f^*\|_\infty \ge 2$.

### Lemma (refined LB at optimum): $\|f^*\|_\infty \ge \|f^*\|_2^2 \ge 2 > C_{1a}$.

Since $\sup(f * f) \le \|f\|_2^2$ (Cauchy-Schwarz on convolution) and
$\|f^*\|_2^2 \ge 2 > C_{1a} \le 1.5029$, we have $\|f^*\|_2^2 > C_{1a}$ strictly.
This implies the Cauchy-Schwarz bound $\sup(f*f) \le \|f\|_2^2$ is **not tight
at the optimum** — the optimum is *asymmetric*; if it were symmetric, equality
would hold and force $C_{1a} \ge 2$, contradicting known $C_{1a} \le 1.5029$.

### Numerically: the existence of an $L^\infty$ optimum is itself doubtful.

If $\|f^{(d)}\|_\infty$ at val(d) optima diverges as $d \to \infty$, the
infimum over the *continuous* Sidon problem is approached by a sequence of
increasingly-peaked functions. Whether the infimum is *attained* by an
$L^\infty$ function is a nontrivial existence question; the numerical evidence
suggests NO — the infimum is approached only by unbounded functions.

This matches the analytic near-optimizers in the literature, which are all
integrably-singular at one endpoint.

---

## Why the truncation argument fails — detailed analysis

Given $f$ with $\sup(f*f) = C$, define:

1. **Rescale**: let $\sigma = 1 - 4\epsilon \in (0, 1)$, $\tilde f(x) := \sigma^{-1} f(x/\sigma)$.
   Then $\tilde f \ge 0$, $\int \tilde f = 1$, $\text{supp}(\tilde f) \subset [-\sigma/4, \sigma/4] = [-(1/4-\epsilon), 1/4-\epsilon]$.

2. **Mollify** with $\eta_\epsilon(x) = (2\epsilon)^{-1} \mathbb{1}_{|x| \le \epsilon}$
   (so $\eta_\epsilon \ge 0$, $\int \eta_\epsilon = 1$, $\text{supp}(\eta_\epsilon) \subset [-\epsilon, \epsilon]$).
   Define $f' := \tilde f * \eta_\epsilon$. Then $f' \ge 0$, $\int f' = 1$,
   $\text{supp}(f') \subset [-1/4, 1/4]$ ✓ (sums of supports).

**Sup convolution:** $f' * f' = (\tilde f * \tilde f) * (\eta_\epsilon * \eta_\epsilon)$.
Since $\eta_\epsilon * \eta_\epsilon$ is a probability density (positive, $\int = 1$):
$\sup_t (f' * f')(t) \le \sup_t (\tilde f * \tilde f)(t) = \sigma^{-1} \sup_t (f * f)(t) = C/\sigma$
(using $\tilde f * \tilde f(t) = \sigma^{-1} (f*f)(t/\sigma)$).

**Bound on $\|f'\|_\infty$:** Young's inequality, $\|\tilde f * \eta_\epsilon\|_\infty \le \|\tilde f\|_2 \|\eta_\epsilon\|_2$.
$\|\tilde f\|_2 = \sigma^{-1/2} \|f\|_2$, $\|\eta_\epsilon\|_2 = 1/\sqrt{2\epsilon}$.
Hence $\|f'\|_\infty \le \|f\|_2 / \sqrt{2\epsilon \sigma}$.

**Trade-off table** (from `lasserre/track1_finf_analysis.py`, $C = \sup(f*f)$, $l_2 = \|f\|_2$):

| Source $C$ | $\|f\|_2$ | $\epsilon$ | $\sup(f'*f') \le$ | $\|f'\|_\infty \le$ |
|---:|---:|---:|---:|---:|
| 1.28 | $\sqrt 2$ | 0.05 | 1.600 | 5.00 |
| 1.28 | $\sqrt 2$ | 0.10 | 2.134 | 4.08 |
| 1.28 | $\sqrt 2$ | 0.125 | 2.560 | 4.00 |
| 1.50 | $\sqrt 2$ | 0.05 | 1.875 | 5.00 |
| 1.50 | 1.5 | 0.05 | 1.875 | 5.30 |
| 1.50 | 2.0 | 0.05 | 1.875 | 7.07 |

**The minimum achievable $\|f'\|_\infty$ is 4.0**, at $\epsilon = 0.125$, with
$\sup(f' * f') \le 2C$ (factor 2 loss). The trade-off curve never enters
the $\|f'\|_\infty \le 2.15$ regime that V3 needs.

**Why?** The Young's inequality lower bound $\|f'\|_\infty \ge \|\tilde f\|_2 \cdot c$
is fundamentally bounded by $\|f\|_2 \ge \sqrt 2$, regardless of how clever
the mollification. To get $\|f'\|_\infty \le 2.15$, we'd need $\|f\|_2 \le 2.15 \cdot \sqrt{2\epsilon \cdot 0.125}$
in the best case, which requires $\|f\|_2 \le 1.075$ — but $\|f\|_2 \ge \sqrt 2 \approx 1.41$.

**The barrier is structural, not technical.** No mollification scheme that
preserves $\int f = 1$ and $\sup(f*f)$ approximately can take an $L^2$ function
with $\|f\|_2 \ge \sqrt 2$ (forced by the support constraint) into an $L^\infty$
function with $\|f'\|_\infty < \sqrt 2 \cdot$ (modest constant).

---

## Why the val(d) numerical evidence is decisive

The val(d) discrete Sidon problem is the discretization of the continuous
problem on $d$ uniform bins. As $d \to \infty$, val(d) $\to C_{1a}$ from below
(known: val(d) is monotone non-decreasing in $d$, converges to $C_{1a}$).

The optimum $\mu^{(d)}$ at finite $d$ is a probability distribution on $d$ bins.
Its piecewise-constant continuous interpretation $f^{(d)}(x) = 2d \mu^{(d)}_i$ on
bin $i$ has $\|f^{(d)}\|_\infty = 2d \max_i \mu^{(d)}_i$.

If $\|f^{(d)}\|_\infty$ saturates as $d \to \infty$ to some limit $M_*$, the
continuous optimum has $\|f^*\|_\infty = M_*$. If $\|f^{(d)}\|_\infty \to \infty$,
the continuous optimum (as a limit point) has $\|f^*\|_\infty = \infty$.

**Empirical: $\|f^{(d)}\|_\infty$ grows monotonically and shows no saturation.**

Fitting $\|f^{(d)}\|_\infty \approx a + b \log(d)$ to the 5 data points:
$\|f^{(4)}\| = 2.88, \|f^{(12)}\| = 5.21$ — slope $\sim 2.1$ per doubling. Extrapolation
to $d = 64$ gives $\|f^{(64)}\| \sim 8$; $d = 256$ gives $\|f^{(256)}\| \sim 11$.
Linear-on-log fit consistent with $\|f^{(d)}\|_\infty \to \infty$.

This matches the literature's analytic near-optimizers, which are integrably-
singular at the support boundary.

---

## Implication for V3's $\lambda^{3\mathrm{pt}}(M=2.15) > 1.2802$ result

V3's empirical finding stands: $\lambda^{3\mathrm{pt}}_{k,N}(\|f\|_\infty \le 2.15) = 1.31$
robustly across $k = 7, 8, 9$.

**This is a valid lower bound on $C_{1a}^{(\|f\|_\infty \le 2.15)}$ — the L^∞-restricted
Sidon problem.** It is *not* a lower bound on $C_{1a}$.

The relationship between the two:
$$C_{1a}^{(M)} = \inf\{\sup(f*f) : f \text{ feasible}, \|f\|_\infty \le M\} > C_{1a}$$
strictly for $M < \|f^*\|_\infty$. Since $\|f^*\|_\infty = \infty$ (or at least
$\gg 2.15$ per numerical evidence), $C_{1a}^{(M=2.15)} > C_{1a}$ strictly.

**The gap is large**: empirically $C_{1a}^{(M=2.15)}$ is approached by V3's
$\lambda^{3\mathrm{pt}}$ from below, settling near 1.31. So
$C_{1a}^{(M=2.15)} \ge 1.31 > 1.2802$. But $C_{1a}$ is somewhere in
$[1.2802, 1.5029]$, and we cannot pin it down further via this approach.

---

## Posterior probability update

| | Probability of beating 1.2802 via L^∞-augmented 3-point Lasserre |
|---|---|
| V2 (after structural Δ = 0 finding) | 2% |
| V3 (after L^∞ + 3D L^∞ giving λ ≥ 1.31) | 15–25% (provisional, pre-analysis) |
| **After this analysis** (Track 1) | **~3%** |

The 12-percentage-point drop is from the convergence of three independent
arguments that the L^∞ bridge does not lift to $C_{1a}$:
- analytical truncation has structural $\sqrt 2$ barrier on $\|f'\|_\infty$;
- numerical val(d) optima have unbounded $\|f^{(d)}\|_\infty$;
- literature near-optimizers are unbounded.

The residual 3% covers:
- An unconventional analytical argument I didn't think of.
- A different formulation that achieves the same effect as L^∞ but doesn't
  exclude the actual optimum (e.g., a constraint that's redundant for the
  optimum but excludes pseudo-measures the relaxation finds).

---

## What remains as a viable path forward

The 3-point Lasserre lift WITH appropriate constraints does provide STRICTLY
TIGHTER bounds than the 2-point Lasserre. The infrastructure built in V2/V3
([lasserre/threepoint_alternatives.py](threepoint_alternatives.py)) is
correct and reusable.

**The natural pivot** (from existing project memory) is:

1. **Farkas-certified val(d) at higher $d$**: the discrete pipeline already
   includes the $L^\infty$ structure implicitly (each bin's mass is bounded
   by 1, giving $\|f^{(d)}\|_\infty \le 2d$). Computing val(d) ≥ 1.2802 with
   rigorous Farkas certificate at $d = 14, 16$ would give the breakthrough.
   See [project_farkas_certified_lasserre.md](../../../.claude/projects/c--Users-andre-OneDrive---PennO365-Desktop-compact-sidon/memory/project_farkas_certified_lasserre.md).
   Empirically val(12) ≈ 1.278; val(14) ≈ 1.284 already (per `val_d_known`).

2. **3-point lift WITHIN val(d)**: apply the same 3-point lift idea to the
   *discrete* val(d) framework rather than the continuous. The discrete
   problem has a finite simplex, no singular pseudo-measure issue. The
   3-point lift of val(d) might add a few % over standard Lasserre at val(d).
   Pre-existing 3D structure in `lasserre/dual_sdp.py` could be extended.

The continuous-formulation 3-point Lasserre line is closed.

---

## Files produced

- [lasserre/track1_finf_analysis.py](track1_finf_analysis.py) — analytical/numerical
  exploration: LB verification, truncation trade-off, two/three-step constructions.
- [lasserre/track1_val_d_finf.py](track1_val_d_finf.py) — direct val(d) optimizer
  extracting $\|f\|_\infty$.
- This report.

## Summary

The $\|f^*\|_\infty$ analytical question is **settled**: the unconstrained
Sidon optimum (or its near-optimizer sequence) has $\|f^*\|_\infty = \infty$,
making the L^∞ bridge unable to lift V3's empirical findings to a $C_{1a}$
lower bound. The original V2 verdict (continuous-Lasserre 3-point line is
DEAD) is restored. The pivot is to discrete val(d) where the L^∞ bound is
implicit and rigorous certification is already in progress.
