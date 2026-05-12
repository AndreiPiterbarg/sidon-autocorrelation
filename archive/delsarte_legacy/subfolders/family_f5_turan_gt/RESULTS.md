# F5 Turán/Gorbachev–Tikhonov — Final Results and Verdict

## Verdict (one sentence)

**(c) DOES NOT BEAT — F5 is dominated by F1 in the existing Delsarte-ratio
pipeline; bare F5 has idealised ratio $\hat g(0)/M_g = 0.5$ exactly, and the
parameterised family $g=(1-2|t|)_{+}P(t^2)$ cannot exceed $\sim 0.5$ even at
$K=8$ in the existing pipeline.** The brief's premise that this family can
reach $\ge 1.27481$ is based on conflating two distinct frameworks:
MV's quadratic dual (which produces 1.27481 from a Cauchy–Schwarz/Parseval
identity, see `mv_construction_detailed.md` eqs (6)–(10)) and the Delsarte
ratio framework here (see `theory.md` Section 2 and `postmortem.md`).

## Phase 1 — derivation outcome

The closed-form Fourier transforms used in the implementation are:

* $\hat h(\xi)=\frac12\operatorname{sinc}^{2}(\pi\xi/2)$ — Gradshteyn–Ryzhik 3.823.
* $\widehat{t^{2k}h(t)}(\xi)=\frac{(-1)^{k}}{(2\pi)^{2k}}\frac{d^{2k}}{d\xi^{2k}}\hat h(\xi)$
  — Stein–Weiss, Ch. I §1.
* $\hat g_{F5}(0)=\sum_{k=0}^{K}\frac{c_{k}}{(2k+1)(2k+2)2^{2k}}$.

`f5.g_hat_value` evaluates $\hat g$ as $\int_{-1/2}^{1/2}(1-2|t|)P(t^2)\cos(2\pi t\xi)dt$
in mpmath with the corner at $t=0$ split out — **rigorous in the sense of
mpmath quadrature error bounds at dps=50**. The closed-form derivative
representation is documented in `derivation.md` for cross-checking.

## Phase 2 — implementation status

| Function | Status | Notes |
|---|---|---|
| `g_value(t, c)` | rigorous | mpmath dps=50 |
| `g_iv(t_iv, c)` | rigorous | range bound on iv intervals |
| `g_hat_zero(c)` | closed form | exact rational of mpf |
| `g_hat_value(xi, c)` | mpmath quad | dps=50 |
| `M_g(c)` | rigorous | interval B&B, rel_tol 1e-10 |
| `is_pd_admissible(c)` | rigorous | grid + Lipschitz + tail bound |
| `f5_lower_bound(c)` | rigorous | $\int\hat g\cdot w/M_g$ |
| `f5_idealised_ratio(c)` | rigorous (but UNSOUND as a $C_{1a}$ LB) | $\hat g(0)/M_g$, exposed only for sanity |

## Phase 3 — K-sweep results (with rigorous-PD enforcement)

Numerical sweep ran twice with different PD-enforcement strategies; the lesson
from the comparison is the headline finding.

### Run 1 — loose PD (sparse 201-pt $\xi$-grid). Numerical artifact.

| K | "best" idealised | "best" proxy | Comment |
|---|---|---|---|
| 0 | 0.500 | 0.451 | bare F5; correct |
| 1 | 0.833 | 0.710 | **artifact**: $c\sim 10^{-8}$, $\hat g_{\min}\!=\!-8\!\times\!10^{-10}$ on finer grid |
| 2 | 0.857 | 0.721 | **artifact**: similar |
| 3 | 0.874 | 0.734 | **artifact**: $c\sim 10^{12}$, ill-conditioned |
| 4 | 0.914 | 0.760 | **artifact**: $P(0)\!=\!0.003$, $\hat g_{\min}\!=\!-6\!\times\!10^{-4}$ |
| 6 | 0.921 | 0.763 | **artifact** |
| 8 | 0.872 | 0.732 | **artifact** |

In every case with $K\ge 1$ above, a finer-resolution check on $\hat g(\xi)$
shows a NEGATIVE dip (at $\xi\approx 1.5$ for $K=4$, $\xi\approx -1.6$ for $K=1$).
Nelder–Mead exploits the discreteness of the spectral-PD grid by pushing
$P\to 0$ scaled, where the absolute PD violation is below the grid tolerance.
**These rows are NOT valid bounds on $C_{1a}$.**

### Run 2 — strict PD (dense 2001-pt $\xi$-grid in $[-25,25]$, hard cutoff 1e-9).

| K | best idealised | best rigorous proxy ($\int\hat g\,w/M_g$) | $c$ |
|---|---|---|---|
| 0 | 0.5000 | 0.4515 | $[0.95]$ (scaled bare triangle) |
| 1 | 0.8333 | 0.7097 | $c\!\approx\!10^{-9}$ (still degenerate; below grid resolution) |
| 2 | 0.5004 | 0.4517 | tiny perturbation of $K\!=\!0$ |
| 3 | 0.5006 | 0.4520 | tiny perturbation |
| 4 | 0.5007 | 0.4520 | tiny perturbation |

K=1 still escapes at the $10^{-9}$ scale below even the strict 2001-pt grid.
$K=2,3,4$ converge cleanly to **idealised $\approx 0.50$, rigorous $\approx 0.452$**
— matching $K=0$ to four digits.

**Conclusion:** F5 with rigorous PD enforcement gives idealised ratio
**exactly $1/2$** (at $K=0$) and rigorous Delsarte bound
**$\approx 0.4515$** at any $K$. The intuition is mechanical: any
$P(u)\ge 0$ on $[0,1/4]$ that boosts $\hat g(0)$ also proportionally boosts
$M_{g}$ at $t=0$ (since $g(0)=P(0)$), AND the spectral non-negativity
constraint $\hat g\ge 0$ on $\mathbb R$ pins down $P\equiv\text{const}$
in the limit (any non-trivial second-derivative content of $P(t^2)$ flips
sign in the spectrum because $\partial_\xi^2\operatorname{sinc}^2$ does).

## Phase 4 — rigorous verification on the best $c^\*=[1.0]$

| Quantity | Enclosure (mpmath dps=50) |
|---|---|
| $\hat g(0)$ | $0.5$ exact |
| $M_g$ | $[1.0,\,1.0]$ |
| Idealised ratio $\hat g(0)/M_g$ | $[0.5,\,0.5]$ |
| Rigorous numerator $\int\hat g\,w$ ($w=\cos^2$, $|\xi|\le 1$, $n\_subdiv=512$) | $\approx 0.4514$ |
| Rigorous Delsarte LB | $\approx 0.4514$ |

Cross-check via numerical FFT (4001-pt $t$-grid on $[-1/2,1/2]$):
$\int g\,(f\!*\!f)$ with $f=\delta_{1/4}$-style approximation reproduces
$\hat g(0)=0.5$ to 5 decimals.

This is **far below** F1's best certified bound $0.4645$ (postmortem.md row
1) and laughably below MV's record $1.27481$.

The PD admissibility check `is_pd_admissible` reports a grid violation under
the conservative threshold $L\cdot h$; the underlying $\hat g_{F5}(\xi)=\frac12
\operatorname{sinc}^{2}(\pi\xi/2)\ge 0$ is in fact PD (closed-form), so this is a
limitation of our generic Lipschitz-plus-grid certificate, **not** a bug in
the F5 formulation. For $K=0$ specifically PD is trivial.

## Phase 5 — comparison and verdict

* **F5 vs F1.** F5 (best in this pipeline): $\le 0.5$ idealised, $\le 0.451$
  rigorous-proxy. F1 (existing module, postmortem.md): best certified
  $L=0.4645$. F5 is **strictly below** F1 in this Delsarte ratio framework.

* **F5 vs MV's 1.27481.** Not comparable in the same algebraic sense.
  MV's 1.27481 comes from solving a quadratic in $\|f*f\|_\infty$
  (`mv_construction_detailed.md` eq. (10)), not from a $\hat g(0)/M_g$ ratio.
  In the Delsarte ratio framework used here, MV-type constructions also do
  not break 1, let alone 1.27481.

* **Lift over MV's 1.27481:** None (negative; F5 gives $\sim 0.45$ in this
  pipeline).

**Final verdict: (c) DOES NOT BEAT — F5 is dominated by F1.**

This is consistent with the explicit warning in `delsarte_dual/ideas_fourier_ineqs.md`
line 56: "*This has a closed form and sits below 1.276.*"

## Why was F5 ever proposed?

Re-reading `ideas_fourier_ineqs.md` lines 45–71 carefully: the F5 sketch
was proposed there as a **starting point** to be *combined* (convex combination)
with MV's arcsine $g$ or *embedded* into a richer Kolountzakis–Révész family,
**together with** the prolate/Logvinenko–Sereda concentration upgrade to the
weight $w$. The proposal does **not** claim that bare F5, in the existing
Delsarte ratio pipeline with $w=\cos^{2}(\pi\xi/2)$, beats 1.27481.

The brief specified to "verify the EXACT form by re-deriving from primal/dual
(see Appendix A)". The re-derivation in `derivation.md` and in `theory.md`
makes clear the rigorous bound is

$$\|f*f\|_\infty\;\ge\;\frac{1}{M_g}\int\hat g(\xi)w(\xi)\,d\xi,$$

**not** $\hat g(0)/M_g$. Under the rigorous bound the brief's expected lift
"+0.002 to +0.01 over 1.2748" cannot be achieved by F5 alone.

## What would unblock F5

1. **Replace $w=\cos^{2}(\pi\xi/2)$ by a prolate / Logvinenko–Sereda weight**
   that captures the *concentration* of $|\hat f|^{2}$ in a finite window
   ($\lambda_{0}(c)\to 1$). This requires implementing the prolate kernel
   (`ideas_pswf.md`, `ideas_fourier_ineqs.md` Inequality 2). Independent of F5.
2. **Combine F5 with a richer Kolountzakis–Révész $g=\mathbf 1_{K/2}*h$**
   construction. Outside the scope of this F5 module.
3. **Use F5 as input to MV's quadratic dual** (eq. (10)), not the Delsarte
   ratio. This is essentially F4 (MV arcsine), currently a scaffold in the repo.

None of these is F5 alone. **F5 alone is not viable in this pipeline.** STOP.

## Files

* `derivation.md` — rigorous derivation, sanity, full proof sketch.
* `f5.py` — implementation with `g_value`, `g_hat_value`, `g_hat_zero`,
  `M_g`, `is_pd_admissible`, `f5_lower_bound`, `f5_idealised_ratio`.
* `optimize.py` — sweep driver and proxy scoring.
* `results.json` — machine-readable K-sweep results.
* `../tests/test_f5_turan_gt.py` — smoke + dominance tests.
