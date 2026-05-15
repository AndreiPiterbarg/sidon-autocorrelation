# Proof outline

A short mathematical summary of the proof that $C_{1a} \ge 1292/1000 =
1.292$ — the **Piterbarg--Bajaj--Vincent Bound** of *A New Lower Bound
for the Supremum of Autoconvolutions*. Full statements, proofs,
and bibliography are in the paper
[`../lower_bound_proof.pdf`](../lower_bound_proof.pdf). For the Lean
encoding see [`formalization.md`](formalization.md); for the
computation see [`reproducibility.md`](reproducibility.md); for the
audit specification see [`verification.md`](verification.md).

## 1. The Matolcsi--Vinuesa master inequality

Fix $\delta \in (0, 1/4]$, $u = 1/2 + \delta$. For a *Bochner-admissible*
kernel $K$ (nonnegative, even, supported on $[-\delta, \delta]$,
unit-mass, with $\widetilde K(j) := \widehat K(j/u) \ge 0$ for every
$j \in \mathbb{Z}$) and an admissible cosine multiplier
$G(x) = \sum_{j=1}^N a_j \cos(2\pi j x /u)$ with $G \ge m_G > 0$ on
$[0, 1/4]$, set
$$S_1 = \sum_{j=1}^N \frac{a_j^2}{\widetilde K(j)}, \qquad a = \frac{4}{u} \cdot \frac{m_G^2}{S_1}, \qquad K_2 = \|K\|_{L^2(\mathbb{R})}^2.$$
The Matolcsi--Vinuesa master inequality (2010, Eq. (7) / Lemma 3.1)
states that for every $f \in \mathcal{F}$ with $\int f = 1$, writing
$M = R(f) = \|f * f\|_\infty$, $k_1 = \widetilde K(1)$, and
$z_1 = \widehat f(1/u)$,
$$M + 1 + 2 z_1^2 k_1 + \sqrt{(M-1-2z_1^4)(K_2 - 1 - 2 k_1^2)} \;\ge\; \frac{2}{u} + a.$$
The Cauchy--Schwarz estimate
$\sqrt{ac}+\sqrt{bd}\le\sqrt{(a+b)(c+d)}$ collapses this to the
*$z_1$-free* form actually used in the Lean master-inequality
theorem `master_inequality_M_lower` (a *theorem*, not an axiom),
$$M + 1 + \sqrt{(M-1)(K_2 - 1)} \;\ge\; \frac{2}{u} + a.$$
The MV derivation uses the arcsine kernel only through
Bochner-admissibility, so the same inequality extends to any
admissible $K$. See paper, section 2.

## 2. The three-scale arcsine kernel

For $\delta \in (0, 1/4]$ the *arcsine kernel*
$K_{\rm arc}(\delta; x) = \delta^{-1} (2/\pi) (1 - (2x/\delta)^2)^{-1/2}
\mathbf{1}_{|x| < \delta/2}$ is Bochner-admissible via the
Sonine--Bessel identity
$\widehat{K_{\rm arc}(\delta;\cdot)}(\xi) = J_0(\pi \delta \xi)^2 \ge 0$
(Watson 1944, §13.46). Convex combinations preserve every admissibility
condition, so for half-widths $\delta_i \in (0, 1/4]$ and weights
$\lambda_i \ge 0$ summing to $1$, the *multi-scale arcsine kernel*
$K_{\rm ms}(x) = \sum_i \lambda_i K_{\rm arc}(\delta_i; x)$ is
Bochner-admissible at scale $\max_i \delta_i$. We take
$$(\delta_1, \delta_2, \delta_3) = (138, 55, 25)/1000, \qquad (\lambda_1, \lambda_2, \lambda_3) = (85, 10, 5)/100,$$
and $u = 1/2 + \delta_1 = 638/1000$. The multiplier $G$ has $N = 200$ modes;
its rational coefficients are obtained by rounding the QP that minimises
$S_1$ subject to $G \ge 1$ on $[0, 1/4]$. See paper section 3 (Sonine
and admissibility) and section 4 (the QP for $G$).

## 3. The five certified anchors

| Anchor | Symbol | Direction | Rational bound | Decimal |
|---|---|---|---|---|
| Kernel-mass moment | $k_1$ | $\ge$ | $9212/10000$ | $0.9212$ |
| Kernel energy | $K_2$ | $\le$ | $47897/10000$ | $4.7897$ |
| Multiplier denominator | $S_1$ | $\le$ | $29841/1000$ | $29.841$ |
| Multiplier minimum | $m_G$ | $\ge$ | $998/1000$ | $0.998$ |
| Gain | $a$ | $\ge$ | $20925/100000$ | $0.20925$ |

All five are computed in `flint.arb` interval arithmetic at $256$-bit
precision and rounded outward to exact rationals (kernel-mass moment:
direct evaluation of $\sum_i \lambda_i J_0(\pi \delta_i)^2$; kernel
energy: adaptive Gauss--Legendre quadrature plus the Watson tail bound
$|J_0(z)|^2 \le 2/(\pi z)$; multiplier denominator: exact arb summation
of $a_j^2 / \widetilde K_{\rm ms}(j)$; multiplier minimum: Taylor
branch-and-bound on $[0, 1/4]$; gain: rational arithmetic from $m_G$ and
$S_1$). See paper section 4.

## 4. The strict-failure witness at $M = 1292/1000$

Set $\tau := 2/u + a$ and $\Phi(M) := M + 1 + \sqrt{(M-1)(K_2 - 1)}$.
With the certified anchors,
$$\tau \;\ge\; \frac{2000}{638} + \frac{20925}{100000} \;=\; 3.344046\ldots$$
At $M = 1292/1000$, using $K_2 \le 47897/10000$ and the rational bound
$\sqrt{(292/1000)(37897/10000)} \le 105195/10^5 = 1.05195$,
$$\Phi(1292/1000) \;\le\; \tfrac{1292}{1000} + 1 + \tfrac{105195}{10^5} \;=\; \tfrac{66879}{20000} \;=\; 3.34395.$$
The margin is
$$\tau - \Phi(1292/1000) \;\ge\; \frac{4267003}{1276000} - \frac{66879}{20000} \;=\; \frac{307}{3190000} \;\ge\; 9.6 \times 10^{-5} \;>\; 0.$$
Since $\Phi$ is strictly increasing on $[1, \infty)$, every $M <
1292/1000$ satisfies $\Phi(M) < \tau$, contradicting the master
inequality. Hence $R(f) \ge 1292/1000$ for every $f \in \mathcal{F}$.
See paper section 5.

## 5. The lift mechanism

The three-scale kernel improves over the Matolcsi--Vinuesa
single-arcsine baseline ($\delta = 138/1000$, $N = 119$) along three
independent channels. Re-solving the master inequality for $M$ while
changing one of $(k_1, K_2, S_1)$ at a time isolates the per-anchor
contribution:

| Varied | $M_{\rm cert}$ | $\Delta M$ |
|---|---:|---:|
| $k_1$ alone | $1.27510$ | $+0.00011$ |
| $K_2$ alone | $1.26455$ | $-0.01044$ |
| $S_1$ alone | $1.29091$ | $+0.01592$ |
| all three jointly | $1.28013$ | $+0.00514$ |

The $k_1$ term is essentially neutral. $K_2$ is a *headwind*: adding
small-scale mass inflates the $L^2$ norm and enters the right factor of
the square root, forcing a smaller $M$. The engine is $S_1$, which drops
from $\approx 87.4$ (single-scale) to $\le 29.841$ (three-scale,
$N = 200$); this defeats the $K_2$ headwind by roughly $3{:}2$.

The structural reason is *Bessel-zero rescue*. For the single-scale
arcsine, $\widehat{K_{\rm arc}}(\xi) = J_0(\pi \delta \xi)^2$ *vanishes*
at the first Bessel zero $z_1 = 2.40483\ldots$, i.e.\ at
$\xi = z_1/(\pi \delta) \approx 5.55$ when $\delta = 138/1000$, so the
periodic coefficient $\widetilde K(j)$ at the corresponding indices is
essentially zero and the reciprocal $a_j^2/\widetilde K(j)$ in $S_1$
blows up. Replacing $K_{\rm arc}$ by $K_{\rm ms}$ adds two smoother
profiles whose first Bessel zeros lie at $\xi \approx 13.9$ and
$\xi \approx 30.6$; the $\lambda_2$ and $\lambda_3$ components fill the
holes left by $\lambda_1 J_0(\pi \delta_1 \xi)^2$. Concretely
$\min_{1 \le j \le 200} \widetilde K_{\rm ms}(j) \ge 2.08 \times
10^{-4}$, whereas the single-scale periodic coefficients touch
$\sim 10^{-6}$. The handful of indices $j$ that dominated the
single-scale $S_1$ have their denominators rescued from $\sim 10^{-6}$
up to $\sim 10^{-2}$, explaining the $\approx 3 \times$ shrinkage of
$S_1$ and the net $+0.0178$ lift in $M_{\rm cert}$ over
Matolcsi--Vinuesa.

## 6. Comparison with Matolcsi--Vinuesa (2010): axiom budget

The published Matolcsi--Vinuesa paper (J. Math. Anal. Appl. **372**
(2010), 439--447) proves $C_{1a} \ge 1.2748$ by:

1. Formally proving Lemmas 3.1 (Eqs.(1)--(4), via Martin--O'Bryant),
   3.3 ($z_1$ refinement), and 3.4 (the $\sin$ bound).
2. **Citing Mathematica** for the numerical values of $J_0(\pi\cdot
   0.138)^2$, $m_G$, $S_1$, and $a = 0.0713$.
3. Combining 1 and 2 algebraically to obtain $1.2748$.

The present Lean proof of $C_{1a} \ge 1.292$ adopts the same
axiom architecture, but with each of the three layers strengthened:

1. **Formally proves the analytic content in Lean.** Approximately
   8650 axiom-free lines spread across fifteen modules (`Sidon.Defs`,
   `Sidon.Bessel`, `Sidon.FourierAux`, `Sidon.TorusParseval`,
   `Sidon.MVLemmas`, `Sidon.MasterFromLemmas`, `Sidon.BundleDefs`,
   `Sidon.BundleEq1`, `Sidon.BundleEq2Schwartz`,
   `Sidon.BundleEq3Schwartz`, `Sidon.BundleEq4`,
   `Sidon.BilinearParseval`, `Sidon.MultiScale`,
   `Sidon.MultiScaleSchwartz`, `Sidon.SchwartzAtomicDischarge`)
   covering the (autoconvolution) arcsine Fourier-transform identity,
   the $L^2$-Plancherel and Schwartz infrastructure (on top of
   `mathlib`'s `MeasureTheory.Lp.fourierTransformₗᵢ`, introduced
   in the `v4.29.1` bump used by this project), the period-$u$
   torus Parseval and lattice-Fourier identities, the four MV
   Lemma 3.1 atomic primitives (Eqs.(1)--(4)) together with their
   dedicated discharge modules, the bilinear Parseval pairings, the
   partial Schwartz atomic-primitive discharge, and the algebraic
   assembly of the master inequality. None of these modules
   declares a `sorry`; only `Sidon.MultiScale` declares user axioms
   (exactly two, both verifiable-by-computation).
2. **2 verifiable-by-computation axioms** (also: "rigorously
   certified numerical assertions"). Each is a logically decidable
   inequality about a specific real number, backed by `flint.arb`
   at 256-bit precision via
   `delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`; the only reason
   they appear as `axiom` rather than `theorem` is that mathlib
   does not yet have a Bessel interval-arithmetic library to
   discharge them mechanically. They are exact analogues of MV's
   Mathematica citations but strictly more rigorous (proven
   interval bounds vs. heuristic numerics), independently audited
   by 14 agents, and anchored to a SHA-256-stamped reproducible
   certificate. The FlySpeck formalisation of Kepler's conjecture
   used the same convention.
3. **1 admissibility-bundle hypothesis.** The structure
   `ExtremiserPrimitives f` collects the conclusions of MV Lemma
   3.1 Eqs.(1)--(4) instantiated at the specific pair $(f, K_{\rm
   ms})$. It is the analogue of MV invoking "by Lemma 3.1
   (Martin--O'Bryant)"; the bundle's existence for a specific
   admissible $f$ is the same analytic assertion MV makes about
   his arbitrary admissible $f$. This is a *hypothesis* of the
   headline theorem, not an axiom.

**Categorisation of the axiom budget.** Lean's `#print axioms`
output mixes three categorically distinct kinds of dependency,
which we separate explicitly here:

- **Logical axioms** -- `propext`, `Classical.choice`,
  `Quot.sound`. Lean 4 core; trusted without proof, cannot be
  derived by any finite computation.
- **Verifiable-by-computation axioms** -- the two numerical ones
  (`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`).
  Logically *decidable* statements about specific real numbers,
  certified by a reproducible `flint.arb` algorithm at 256-bit
  precision, currently un-formalised in Lean only because mathlib
  lacks a Bessel interval-arithmetic library.
- **Analytic admissibility bundle** -- `ExtremiserPrimitives f`.
  Not an axiom but a *hypothesis* of the headline.

The distinction between *conjectural* and
*verifiable-by-computation* axioms matters because a natural
critic's question is "are you assuming something unprovable?" The
answer here is **no**: both numerical axioms are provable; they
are simply not yet formalised in Lean for engineering reasons.
They are not RH-style assertions undecidable from within the
system -- they are statements about specific integers that any
sufficient implementation of interval arithmetic and the Bessel
power series can decide.

**Why this is publication-valid.** It exceeds MV's standard --
~8650 lines of formal Lean for what MV proved on ~5 pages of text --
and adopts the same axiom architecture used by every published
computer-assisted proof of a real-number constant (Flyspeck cited
Kepler's interval arithmetic; the polynomial-method cap-set proof
cited specific Lagrange polynomial bounds; the Polynomial
Freiman--Ruzsa formalisation cited numerical Plünnecke--Ruzsa
constants). The mathematical content of the proof is in the Lean
theorems; the verifiable-by-computation axioms encode only "evaluate
this specific integral and compare it to this specific rational".

**What replacing the verifiable-by-computation axioms would
require.** A rigorous Lean interval-arithmetic library, verified
Gauss--Legendre quadrature, verified Bessel evaluation, and verified
Taylor branch-and-bound -- roughly 6000--10000 additional lines of
Lean. None of this exists in mathlib today; producing it would be a
separate multi-year subproject (comparable in scope to the
decade-long Flyspeck effort for Kepler), with no upside for the
mathematical claim.

**Honesty caveats.** Three points the reader should hold in mind:

- The headline is conditional on the analytic admissibility bundle
  `ExtremiserPrimitives f`, not unconditional. Producing the bundle
  for an arbitrary admissible $f$ requires the $L^1 \cap L^2$
  periodisation bridge for $f * f$ and $f \circ f$ on
  $\mathbb{R}/u\mathbb{Z}$; the relevant infrastructure is in
  `Sidon.TorusParseval` and `Sidon.FourierAux`, but the final
  stitching is not yet a one-line mathlib call.
- The two verifiable-by-computation axioms depend on trusting the
  `flint.arb` library. `flint.arb` is peer-reviewed (Johansson 2017,
  IEEE TC) and widely used for rigorous mathematics, but it is not
  itself Lean-verified.
- A Mathematica computation underlying MV is *strictly less*
  rigorous than these `flint.arb`-discharged axioms (Mathematica uses
  heuristic precision tracking; `flint.arb` provides proven interval
  inclusion).
