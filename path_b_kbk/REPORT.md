# Path B — Krein-Boas-Kac SDP for Hyp_R Closure: REPORT

## Goal

Prove the restricted-Hölder hypothesis Hyp_R unconditionally, where Hyp_R is:

> For every nonneg pdf $f$ on $[-1/4, 1/4]$ with $\int f = 1$ and
> $\|f*f\|_\infty \le M_{\max} = 1.378$, we have
> $\|f*f\|_2^2 \le (\log 16/\pi)\,\|f*f\|_\infty \approx 0.88254\,M$.

If proved, gives $C_{1a} \ge 1.378$ unconditionally — a 0.10 jump over Cloninger–Steinerberger 2017's 1.2802.

## Theoretical setup

Variables: $\hat f(n) = a_n + i b_n$ for $n = 1, \ldots, N$ (with $\hat f(0) = 1$, $\hat f(-n) = \overline{\hat f(n)}$).
Set $y_n := |\hat f(n)|^2 = a_n^2 + b_n^2$.

Constraints:
1. **MO 2.14**: $y_n \le \mu(M) := M\sin(\pi/M)/\pi$ for $n \ge 1$.
2. **Bochner on f** (encodes $f \ge 0$): Hermitian Toeplitz $T = [\hat f(i-j)]$ PSD.
3. **Krein-Boas-Kac** (encodes $\text{supp}\,f \subset [-1/4, 1/4]$): localizing Hermitian Toeplitz $L = [(p \cdot \hat f)(i-j)]$ PSD with $p(x) = 1/16 - x^2$.
4. **Parseval**: $\sum y_n = (K-1)/2$ where $K = \|f\|_2^2 \ge 2$.

Goal: bound $\max \sum y_n^2$ from above. If $\max \sum y_n^2 \le 0.108$, then $c_{\mathrm{emp}} \le 0.88254$ (Hyp_R closes).

## Implementation

`kbk_sdp.py` implements the order-1 Lasserre / Schur-lift SDP relaxation:
- Real-form encoding of Hermitian Toeplitz (size $2(N+1) \times 2(N+1)$).
- Localizing real-form encoding via $\hat p(k)$ Fourier coefficients of $p(x)$.
- Schur lift $V \succeq y y^T$ for the quartic objective.
- Linear cut $V_{nn} \le \mu y_n$ (from $y_n(\mu - y_n) \ge 0$).

## Numerical results

At $M = 1.378$, $\mu(M) = 0.333$, target = 0.108:

| K_upper | Path A loose | + phase-Bochner | + KBK localizing |
|---|---|---|---|
| 2.0 | 0.967 | 0.967 | 0.967 |
| 2.5 | 1.088 | 1.088 | 1.088 |
| 3.0 | 1.209 | 1.209 | 1.209 |
| 4.0 | 1.451 | 1.451 | 1.451 |

**KBK adds nothing at this Lasserre order.** The SDP just returns $\mu \cdot S = \mu \cdot (K-1)/2$ regardless of phase-Bochner or KBK.

## Why it doesn't close — root cause

The SDP relaxation uses the linear cut $V_{nn} \le \mu \cdot y_n$, giving
$$\sum V_{nn} \le \mu \sum y_n.$$
This is the **loose Path A bound** $\sum y_n^2 \le \mu \cdot (K-1)/2$.

For the relaxation to give a tighter bound, additional constraints must bind. The candidates:
1. **Bochner on f** (Hermitian Toeplitz): doesn't reduce $\sum y_n$.
2. **KBK localizing**: doesn't reduce $\sum y_n$ either — empirically max $\sum y_n = N \cdot \mu$ even with KBK enforced. Phases can be chosen to satisfy KBK with all $y_n = \mu$.
3. **Quartic objective structure**: the constraint $V_{nn} \ge y_n^2$ alone doesn't tighten $\sum V_{nn}$ from above.

**Verified by direct computation:** at $N=16$, $K_\text{trunc}=10$, max $\sum y_n$ with KBK + Bochner + MO 2.14 is **5.33** (corresponding to $K = 11.65$), well above the Cauchy-Schwarz minimum 0.5.

## The fundamental obstacle

For asymmetric admissible $f$, $K = \|f\|_2^2$ has **no analytical upper bound** in terms of $M = \|f*f\|_\infty$. This is the same K-unboundedness obstruction that defeats Path A (CLAUDE.md F4).

The order-1 Lasserre / Schur-lift SDP relaxation is too weak: it only uses linear cuts on $V_{nn}$ and PSD constraints on the moment matrix at order 1. These don't capture the deep structural relationship that admissible $f$ at small $M$ has constrained $\sum y_n^2$.

## What would be needed

**Order-2 Lasserre relaxation** with quartic-moment matrix $M_2$ of size $\sim 2N^2 \times 2N^2$:
- For $N = 8$: $\sim 153 \times 153$ moment matrix. Tractable.
- For $N = 15$: $\sim 511 \times 511$. Borderline.
- For $N = 30$: $\sim 1830 \times 1830$. Heavy SDP.

This is "moderate" computation by 2026 standards but requires substantial SOS-package infrastructure (SumOfSquares.jl, ncpol2sdpa, or hand-coded Lasserre lifts). Implementation effort: 1-2 weeks of focused coding + significant solver time per evaluation.

**Whether order-2 Lasserre actually closes Hyp_R is unknown.** Plausibly:
- If yes: Hyp_R proved, $C_{1a} \ge 1.378$ unconditionally.
- If no: the KBK + Bochner + MO 2.14 chain at any finite Lasserre order is fundamentally insufficient, and the problem requires deeper structural inputs (e.g., decay rates of $\hat f$, sharper MO 2.14 variants, etc.).

## Verdict on Path B

**Order-1 Lasserre / Schur lift (this implementation): does NOT close Hyp_R.**
The SDP relaxation is too loose. KBK localizing PSD adds zero in this formulation
because it doesn't restrict $\sum y_n$.

**Order-2 Lasserre: not implemented; would require 1-2 weeks of code + substantial compute.**
Unknown if it closes Hyp_R; theoretical analysis suggests partial closure at best.

**Honest recommendation:** the most promising remaining path for an unconditional
improvement on $C_{1a} = 1.2802$ is the **Lasserre val(d) certificate at $d=32$ or $d=64$**
(separate track in `lasserre/`), which directly bounds $M$ rather than going through
Hyp_R. The val(d) certificate has existing infrastructure (`d64_farkas_cert.py`) and
predicted value 1.28-1.32 unconditionally.

The Hyp_R-via-KBK route (this Path B) is **not feasible at order-1 Lasserre** and
**unverified at order-2 Lasserre**, with the K-unboundedness obstruction being the
fundamental barrier even with full KBK localization.

## Files

- `kbk_sdp.py` — the SDP implementation (with Schur lift relaxation)
- `run_kbk.py` — driver for parameter sweeps
- `REPORT.md` — this file

## Inputs from the existing codebase

- `delsarte_dual/grid_bound_sharper_bathtub/path1_sdp.py` — Fejér-Riesz LMI idiom for trig poly nonnegativity (used as basis for real-form Hermitian PSD encoding)
- `bochner_sos/toeplitz_psd_probe.py` — Toeplitz-PSD on $|\hat f|^2$ sequence
- `delsarte_dual/restricted_holder/derivation.md` — the Hyp_R conditional theorem
- `delsarte_dual/path_a_unconditional_holder/derivation.md` — Path A failure analysis
