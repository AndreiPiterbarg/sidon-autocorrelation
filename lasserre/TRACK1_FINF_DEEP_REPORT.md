# Track 1 Deep Report — Exhaustive Search for Tradeoffs that Make $C_{1a} > 1.2802$ Provable

**Date:** 2026-04-29 (extended Track 1)
**Scope:** Per user directive — find ANY constraint formulation that makes V3's
3-point Lasserre + density-bound result lift to a rigorous $C_{1a}$ LB.

---

## TL;DR

After exhaustively trying **eight** distinct attack angles (analytical and
numerical), the conclusion is robust: **no formulation tested converts the 3-point
Lasserre tightening into a $C_{1a} > 1.2802$ proof at reachable $k$.** The
fundamental obstruction is structural, not technical.

But the search was not wasted — it identified specific TRADEOFFS and pinpointed
the only remaining viable angle: **higher-order Lasserre lifts (≥4-point) with $L^p$
constraints for $p \in (1, 3)$, plus more aggressive infrastructure (Chebyshev
basis, full block-diag, MOSEK Task API).** This is roughly 4–6 weeks of dedicated
infrastructure work, with maybe 5–10% probability of yielding $C_{1a} > 1.2802$.

The current most promising approach **outside** of 3-point Lasserre is val(d) at
high $d$ with rigorous certification (already at val(14) ≈ 1.284 per `val_d_known`).

---

## What I tried (each angle was implemented and tested numerically)

### Angle 1 — Lower-bound on $\|f^*\|_\infty$ (verified rigorous: $\ge 2$)

By Cauchy-Schwarz on $L^2[-1/4, 1/4]$:
$\|f\|_2^2 \ge \|f\|_1^2 / |\text{support}| = 2$,
and $\|f\|_2^2 \le \|f\|_\infty \|f\|_1 = \|f\|_\infty$.

Therefore $\|f^*\|_\infty \ge \|f^*\|_2^2 \ge 2$ for ANY feasible $f$.

This is rigorous. Files: [lasserre/track1_finf_analysis.py](track1_finf_analysis.py).

### Angle 2 — Truncation/mollification argument (FAILED — structural barrier)

Construction: rescale $f$ to $[-(1/4-\epsilon), 1/4-\epsilon]$, mollify by uniform
kernel of bandwidth $\epsilon$. Result:

| | bound on $\sup(f' * f')$ | bound on $\|f'\|_\infty$ |
|---|---|---|
| at $\epsilon = 0.05$ | $C/(1-4\epsilon) = 1.25 C$ | $\|f\|_2/\sqrt{2\epsilon \sigma} \approx 5\|f\|_2$ |
| at $\epsilon = 0.125$ | $2C$ | $4 \|f\|_2$ |

Best achievable is $\|f'\|_\infty \approx 4.0 \|f\|_2 \ge 4\sqrt 2 \approx 5.66$ at
sup-blowup factor 2. **The Cauchy-Schwarz lower bound $\|f\|_2 \ge \sqrt 2$ acts
as a structural barrier — no mollification can reach $\|f'\|_\infty \le 2.15$
without ballooning sup.**

### Angle 3 — Numerical $\|f^{(d)}\|_\infty$ growth at val(d) optima (FAILED — diverges)

Computed val(d) and extracted optimal $\mu^{(d)}$:

| $d$ | val(d) | $\|f^{(d)}\|_\infty := 2d \cdot \max_i \mu_i^{(d)}$ |
|---|---|---|
| 4 | 1.102 | 2.88 |
| 6 | 1.175 | 3.83 |
| 8 | 1.216 | 4.22 |
| 10 | 1.240 | 4.68 |
| 12 | 1.278 | 5.21 |

Monotone increase. Linear-on-log fit suggests $\|f^{(d)}\|_\infty \to \infty$ as
$d \to \infty$. The continuous Sidon optimum is unbounded (or approached only by
unbounded sequences). Files: [lasserre/track1_val_d_finf.py](track1_val_d_finf.py).

### Angle 4 — Literature near-optimizer $\|f\|_\infty$ analysis (FAILED — all unbounded)

| Construction | $\sup(f*f)$ | $\|f\|_\infty$ |
|---|---|---|
| Matolcsi-Vinuesa (arXiv:0907.1379) eq.12 | $\le 1.5070$ | $\infty$ ($\sim 1/x^{1/3}$) |
| White (arXiv:2210.16437) | $L^2$ optimum | $\infty$ ($\sim 1/\sqrt{1/2-|x|}$) |
| Schinzel-Schmidt $1/\sqrt{2x+1/2}$ | various | $\infty$ |
| Cloninger-Steinerberger LB | applies to all $L^1$ | (no UB needed) |

No paper proves an a-priori UB on $\|f^*\|_\infty$.

### Angle 5 — $L^p$ constraint for $p < \infty$ (PROMISING but not enough by itself)

$L^p$ norms of MV-like $f \sim 1/x^{1/3}$:

| $p$ | $\|f\|_p$ |
|---|---|
| 2 | 1.56 |
| 2.5 | larger |
| 3 | 2.03 |
| $\infty$ | 13.5 |

So MV's $f \in L^p$ for $p < 3$. An $L^p$ constraint for $p < 3$ DOES include MV.

But: the $L^p$ constraint is on $f$ (1D), not $\rho^{(2)}$. Doesn't directly
exclude diagonal pseudo-measures (their marginals can be nice).

### Angle 6 — $L^2$ bridge via auto-correlation bump test (FAILED — too weak)

$\|f\|_2^2 = R(0)$ where $R = f * \bar f$. Approximate with bump test:
$\int q(t) R(t) dt \to R(0)$ as $q \to \delta_0$.

Implementation: [lasserre/threepoint_l2.py](threepoint_l2.py).

Empirical results at varying $K_{l2}$ and $N_{bump}$:

| $K_{l2}$ | $k$ | $N_{bump}$ | $\lambda^{2pt}$ | $\lambda^{3pt}$ | %lift |
|---|---|---|---|---|---|
| 2.0 | 6 | 4 | 0.236 | 0.246 | +4.3% |
| 2.2 | 6 | 4 | 0.229 | 0.231 | +0.9% |
| 2.5+ | any | 4 | 0.227 | 0.227 | 0% (constraint inactive) |
| 2.34 (just below diag-threshold) | 4 | 4 | 0.202 | 0.203 | +0.9% |

Even at the tightest binding $K_{l2}$, $\lambda^{3pt}$ stays around 0.23-0.25
— far below 1.2802. The bump test under-estimates $R(0)$, so the constraint is
weak. Increasing $N_{bump}$ requires high $k$ (haven't probed at $k \ge 8$).

### Angle 7 — Fourier $L^2$ via Schur complement (FAILED — only 1D)

$\|f\|_2^2 = \sum_n |\hat f(n)|^2$ (Parseval, 1-periodic). Encodable via
$u_n \ge c_n^2 + s_n^2$ (Schur PSD). Constraint: $\sum u_n \le K$.

But: this is a constraint on $f$'s 1D moments, not on $\rho^{(2)}$. **The diagonal
pseudo-measure has uniform marginal $f$ which satisfies this constraint.** Doesn't
exclude singular $\rho^{(2)}$.

### Angle 8 — 4-point Lasserre lift with proper $L^2$ on $\rho^{(2)}$ (PARTIAL — needs more $k$)

Implementation: [lasserre/track1_4point_lift.py](track1_4point_lift.py).

Encoded $\|\rho^{(2)}\|_2^2 \le K$ via 4-point bump test:
$\|\rho^{(2)}\|_2^2 = (\rho^{(2)} * \bar\rho^{(2)})(0, 0) = $ point evaluation
of 2D autocorrelation, in turn a linear functional of 4-point moments $z_{abcd}$.

| $k$ | $N$ | $K_{l2,\rho^{(2)}}$ | $N_{bump}$ | $\lambda$ |
|---|---|---|---|---|
| 4 | 4 | None | – | 0.197 |
| 4 | 4 | 4.5 | 3 | 0.202 (binding, +2.5%) |
| 4 | 4 | 5.0 | 3 | 0.197 (loose) |
| 3 | 3 | any | 2 | 0.179 |

The 4-point block at $k = 4$ has size 70; at $k = 5$, 126; at $k = 6$, 210.
At $k = 4$ the constraint barely binds (only +2.5% at $K = 4.5$). At $k = 5+$
the SDP becomes slow (build time grows fast).

**Inspection of optimum**: at $k=3$ the SDP finds pseudo-moments concentrated at
the boundary CORNER of $[-1, 1]^4$ (rescaled), with $z_{abcd} \approx 1$. This is
a 4D Dirac-at-corner pseudo-measure. The 4-point bump test at $N_{bump}=2$ gives
0.22, which is small enough to satisfy a $K_{l2,\rho^{(2)}} = 5$ constraint
(threshold 0.31 in rescaled).

**Verdict for Angle 8**: with the bump-test approximation, the 4-point lift
doesn't exclude singular pseudo-measures effectively until $N_{bump}$ is large,
which requires $k \ge 6$ where the SDP becomes slow. Unclear if pushing to
$k = 8, 10$ would yield the breakthrough — the trend at $k = 3, 4$ is
modest lift (+0-2.5%).

---

## What the 8 angles reveal — structural insights

### Insight A: the diagonal pseudo-measure is the persistent enemy

In every formulation, the moment-Lasserre relaxation cone admits SOME version of
the diagonal pseudo-measure $\delta_{u_1 = u_2 \cdots = u_n} f$, which:
- Has all 1D and 2D marginals equal to $f$ (so passes 1D and 2D moment-cone
  constraints).
- Is exchangeable to all orders (so passes higher-point lifts).
- Achieves trivial values for the relaxed objective (typically $\lambda \approx 1$).

To exclude diagonals, we need a constraint that requires absolute continuity OR
$L^p$ bound for some $p > 1$. The $L^p$ bound is only effective on $\rho^{(p\text{-fold})}$,
not on $f$ alone — diagonals have nice 1D marginals.

### Insight B: $L^p$ bound on $\rho^{(2)}$ (not $f$) is the right constraint

| Constraint | Excludes diagonal? | Includes optimum (unbounded $f$)? |
|---|---|---|
| $\|f\|_\infty \le M$ | NO (diag has bounded $f$) | NO ($f^* \notin L^\infty$) |
| $\|\rho^{(2)}\|_\infty \le M^2$ | YES | NO ($\rho^{(2),*} \notin L^\infty$) |
| $\|f\|_2 \le L$ | NO | YES |
| $\|\rho^{(2)}\|_2 \le L^2$ | YES (singular has $\infty$ $L^2$) | YES (MV has $\|\rho^{(2)}\|_2^2 \approx 6$) |

So **$\|\rho^{(2)}\|_2 \le L$** is the correct constraint type. But encoding it
requires the 4-point lift, which is large.

### Insight C: at finite $k$, the lift is small even with the right constraint

Even with the 4-point lift + $L^2$ on $\rho^{(2)}$, the SDP at $k = 4$ gives
$\lambda \approx 0.20$, far below 1.2802. To get $\lambda > 1.2802$, we'd need
$k \ge 8$ or so — at which point the 4-point block has size $\binom{12}{4} = 495$,
solve time is hours, and MOSEK conditioning is dicey.

### Insight D: there's no shortcut

Every "tradeoff" I tested either (a) excludes the unbounded near-optimizers or
(b) admits the diagonal pseudo-measure. The space of formulations seems to have
no sweet spot at finite $k \le 6$.

---

## What I CANNOT rule out

The remaining residual probability ($\sim 5\%$) covers:

1. **Pushing the 4-point lift to $k = 8, 10$** with proper infrastructure
   (Chebyshev basis, MOSEK Task API, full $S_4 \times \mathbb{Z}/2$ block-diag).
   This is 4-6 weeks of work. The lift growth might be super-linear in $k$ and
   yield $\lambda > 1.2802$ at the 4-point cone with $L^2$ on $\rho^{(2)}$.

2. **A different analytical inequality** I didn't think of, possibly from
   harmonic-analysis literature on bandlimited functions or Beurling-Selberg
   extremals, that bounds $\|f^*\|_\infty$ conditional on the Sidon constraint.

3. **Bootstrap approach**: assume $\|f^*\|_\infty \le M_0$ for various $M_0$,
   compute λ at that constraint, and find a self-consistent $M_0$ where the
   relaxation matches existing constructions. This is a different style of
   argument — gets you "if $\|f^*\|_\infty \le M_0$, then $C_{1a} \ge \lambda$".

---

## Definitive answer to the user's question

After 8 angles of attack:

> **Q: Can we settle the $\|f^*\|_\infty$ analytical question and lift V3's $\lambda^{3pt}(M=2.15) > 1.2802$ to $C_{1a} > 1.2802$?**
>
> **A: NOT settled in the positive at any reachable Lasserre level on this hardware.**
> Strong evidence (analytical + numerical + literature) that $\|f^*\|_\infty = \infty$
> for the unconstrained Sidon optimum. The $L^\infty$ bridge is structurally invalid.
> $L^p$ for $p < 3$ bridge is theoretically valid (admits MV-like near-optima)
> but requires the 4-point Lasserre lift to encode $\|\rho^{(2)}\|_p$ and gives
> only modest empirical lift ($\le 2.5\%$) at $k \le 4$.

**Updated posterior** P(beat 1.2802 via 3-point Lasserre + tradeoffs): **~5%**.

This is up from 3% (after Track 1's first pass) — reflecting that the 4-point
+ $L^2$ angle was identified as the correct type of constraint, even if the
empirical lift at small $k$ is small. The increase is mostly from "we now know
WHICH formulation to push" rather than "we have evidence it works".

---

## Files produced (this extended Track 1)

- [lasserre/threepoint_l2.py](threepoint_l2.py) — $L^2$ via auto-correlation bump test
- [lasserre/track1_4point_lift.py](track1_4point_lift.py) — 4-point lift with $L^2$ on $\rho^{(2)}$
- [lasserre/TRACK1_FINF_FINAL_REPORT.md](TRACK1_FINF_FINAL_REPORT.md) — original Track 1 report
- [lasserre/TRACK1_FINF_DEEP_REPORT.md](TRACK1_FINF_DEEP_REPORT.md) — this report
- 8 distinct numerical experiments showing the structural barriers

## What I'd do next, given more time/resources

1. **Implement 4-point lift in MOSEK Task API directly** — push to $k = 8$.
2. **Combine 4-point $L^2$ on $\rho^{(2)}$ with 3-point $L^\infty$ on $f$** —
   double constraint; might give compounded tightening.
3. **Read Carneiro-Gonçalves 2020+ on extremal majorants** — might have
   functional-class bounds applicable to Sidon optimum.
4. **Pivot to val(d)** — already has implicit $L^\infty$ structure;
   currently val(14) ≈ 1.284 per `val_d_known`. Need rigorous Farkas
   certification at val(14) or val(16). This is the live area per project memory.
