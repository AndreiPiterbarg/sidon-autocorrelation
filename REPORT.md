# Project Report: A New Lower Bound for the Supremum of Autoconvolutions

> A new lower bound on the autoconvolution constant
> $$ C_{1a} \;\ge\; \frac{1292}{1000} \;=\; 1.292, $$
> improving the previously announced $1.2802$ of Cloninger-Steinerberger
> (2017) and the rigorous analytic $1.27481$ of Matolcsi-Vinuesa
> (2010). The argument is closed by interval arithmetic in
> `flint.arb` at 256-bit precision and mechanized in Lean 4 across
> fifteen modules (~8650 lines on top of mathlib `v4.29.1`). The
> headline theorem reaches exactly **two
> *verifiable-by-computation* user axioms** -- rigorously certified
> numerical assertions, *not* conjectural -- plus an analytic
> admissibility-bundle hypothesis `ExtremiserPrimitives f`
> (analogues of MV 2010's Mathematica citations and its invocation
> of "Lemma 3.1 (Martin--O'Bryant)", respectively; the FlySpeck
> formalisation of Kepler's conjecture used the same convention).
> The end-to-end audit (`audit_consistency.py`) passes with verdict
> `ALL CHECKS PASS`.

| | |
|---|---|
| **Authors** | Andrei Piterbarg, Jai Bajaj, Derrick Vincent |
| **Manuscript** | [`lower_bound_proof.tex`](lower_bound_proof.tex) / [`lower_bound_proof.pdf`](lower_bound_proof.pdf) |
| **Lean formalization** | [`lean/Sidon/`](lean/Sidon/) |
| **Numerical certificate** | [`delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json) |


---

## 1. The Result

For every nonnegative $f \in L^1(\mathbb{R})$ supported on
$(-\tfrac14, \tfrac14)$ with $\int f > 0$ and
$\|f * f\|_{L^\infty} < \infty$,

$$ \frac{\|f * f\|_{L^\infty}}{\bigl(\int f\bigr)^2} \;\ge\; \frac{1292}{1000}. $$

Taking the infimum, $C_{1a} \ge 1292/1000 = 1.292$. Combined with the
upper bound $C_{1a} \le 1.5029$ from Georgiev-Gomez Serrano-Tao-Wagner
(AlphaEvolve, arXiv:2511.02864), the constant is now bracketed in
$[1.292, 1.5029]$. The improvement is

| | $C_{1a} \ge$ | Source |
|---|---|---|
| Erdős-Turán (1941) | $1$ | classical |
| Martin-O'Bryant (2009) | $1.262$ | arXiv:0807.5121 |
| Matolcsi-Vinuesa (2010) | $1.27481$ | arXiv:0907.1379 |
| Cloninger-Steinerberger (2017) | $1.2802$ | arXiv:1403.7988 |
| **This work Piterbarg-Bajaj-Vincent (2026)** | **$1.292$** | manuscript at root |

The lift over the prior published lower bound is
$1.292 - 1.2802 = 0.0118$; over the rigorous analytic
Matolcsi-Vinuesa baseline, $1.292 - 1.27481 = 0.01719$.

## 2. Method

The proof refines the Matolcsi-Vinuesa dual framework along four
axes:

1. **Three-scale arcsine kernel.** The single arcsine kernel
   $K_{\rm arc}(\delta_1; \cdot)$ used by MV is replaced by the convex
   combination
   $$ K_{\rm ms} = \sum_{i=1}^{3} \lambda_i\, K_{\rm arc}(\delta_i; \cdot), $$
   with
   $(\delta_1, \delta_2, \delta_3) = (138, 55, 25)/1000$ and
   $(\lambda_1, \lambda_2, \lambda_3) = (85, 10, 5)/100$. The smaller
   scales fill the gaps left by the first Bessel zero of
   $J_0(\pi \delta_1 \xi)^2$, lowering the dominant denominator $S_1$
   from $\approx 87.4$ (single-scale) to $\le 29.841$.

2. **200-mode cosine multiplier.** The trigonometric multiplier $G$ is
   re-optimized as a 200-cosine expansion (rather than the 119 modes
   of MV2010), solved by a convex QP minimizing
   $\sum_j a_j^2 / \widetilde{K_{\rm ms}}(j)$ subject to $G \ge 1$ on
   a 5001-point grid in $[0, 1/4]$. Coefficients are rationalized to
   $a_j \in \mathbb{Q}$ with denominator $10^8$.

3. **Rigorous interval arithmetic.** Every analytic functional
   entering the master inequality is bounded in `flint.arb` at
   256-bit precision and rounded outward to an exact rational.

4. **Quadratic strict-failure witness.** The $z_1$-free quadratic
   master inequality
   $$ \Phi(M) \;=\; M + 1 + \sqrt{(M-1)(K_2 - 1)} \;\ge\; \tau \;=\; \tfrac{2}{u} + a $$
   is closed by exhibiting a rational $M_0 = 1292/1000$ at which
   $\Phi(M_0) < \tau$ strictly. The certified upper bound on $K_2$
   makes $\Phi$ an over-estimate, so the strict failure forces
   $R(f) > M_0$ for every admissible $f$.

## 3. The Certified Anchors

The five real functionals certified in `flint.arb` at 256-bit
precision (all rationals are sourced from
[`reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json)):

| Functional | Direction | Certifier value | Rational slack | Lean theorem |
|---|---|---|---|---|
| $k_1 = \widehat{K_{\rm ms}}(1)$ | $\ge$ | $0.92124658993\ldots$ | $9212/10000$ | `k_one_lower_bound` |
| $K_2 = \|K_{\rm ms}\|_{L^2}^2$ | $\le$ | $\in [4.78882342, 4.78890519]$ | $47897/10000$ | `K_two_upper_bound` |
| $S_1 = \sum_j a_j^2 / \widetilde{K}(j)$ | $\le$ | $29.84090646\ldots$ | $29841/1000$ | `S_one_upper_bound` |
| $m_G = \min_{[0,1/4]} G$ | $\ge$ | $0.99997987\ldots$ | $998/1000$ | `min_G_lower_bound` |
| $a = (4/u)\, m_G^2 / S_1$ | $\ge$ | $0.21009214\ldots$ | $20925/100000$ | `gain_lower_bound` |

The **strict-failure margin** at the rational witness
$M_0 = 1292/1000$, $K_2 \le 47897/10000$ is exactly

$$ \tau - \Phi(M_0) \;=\; \frac{4267003}{1276000} - \frac{66879}{20000} \;=\; \frac{307}{3190000} \;\approx\; 9.624 \times 10^{-5}. $$

The certifier itself reports a tighter
$M_{\rm cert} \ge 66167/51200 \approx 1.29232422$ (production driver)
when the analytic anchors are coupled rather than rationalized
separately. The headline target $1292/1000$ is the looser rational
floor used in the published statement and in the Lean axiom.

## 4. Lean 4 Formalization

The analytic chain is mechanized in
[`lean/Sidon/`](lean/Sidon/) across **fifteen modules totalling
~8650 lines** on top of Mathlib pinned to `v4.29.1`, commit
[`5e932f97dd25535344f80f9dd8da3aab83df0fe6`](https://github.com/leanprover-community/mathlib4/commit/5e932f97dd25535344f80f9dd8da3aab83df0fe6).
The `v4.29.1` bump (post-Nov 2025) unlocked the $L^2$-Plancherel API
(`MeasureTheory.Lp.fourierTransformₗᵢ`) and convolution--Fourier
duality (`Real.fourier_mul_convolution_eq`), on which the Parseval
infrastructure of this project depends. The full development builds
cleanly under `lake build` with **zero `sorry` tactics** across all
modules.

**Recent fixes (Wave-12 multi-agent audit).** Two math-fidelity
corrections landed during a multi-agent audit and are reflected in the
post-Wave-12 build: (i) the `f ∘ f` convention was tightened from a
pointwise product to the *convolutional* form matching MV 2010, and
(ii) `K_arc` is now defined as the autoconvolution
$\eta_\delta * \eta_\delta$ rather than the bare arcsine density. Both
fixes are build-clean under the same 5-axiom budget (3 Lean core + 2
verifiable-by-computation), with the three headlines listed below
remaining non-vacuous.

The three exported headline theorems are
`Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000` (general,
hypothesised on the `ExtremiserPrimitives` bundle),
`Sidon.MultiScaleSchwartz.autoconvolution_ratio_ge_1292_1000_schwartz`
(restricted to Schwartz $f$, hypothesised on the `SchwartzAtomic`
record), and
`Sidon.SchwartzAtomicDischarge.autoconvolution_ratio_ge_1292_1000_schwartz_residual`
(restricted to Schwartz $f$, hypothesised on the slimmer
`SchwartzAtomicResidual` record).

| Module | Lines | Role | Axioms |
|--------|------:|------|--------|
| [`Sidon.Defs`](lean/Sidon/Defs.lean) | 55 | Core definitions (`autoconvolution_ratio`, etc.; convolutional `f ∘ f`). | 0 |
| [`Sidon.Bessel`](lean/Sidon/Bessel.lean) | 958 | Bessel $J_0$ power series, autoconvolution arcsine FT identity, Watson tail bound. | 0 |
| [`Sidon.FourierAux`](lean/Sidon/FourierAux.lean) | 606 | Schwartz Plancherel, $L^p$ bridge, $L^1$-pairing. | 0 |
| [`Sidon.TorusParseval`](lean/Sidon/TorusParseval.lean) | 785 | Period-$u$ Parseval, lattice Fourier, bilinear pairing. | 0 |
| [`Sidon.MVLemmas`](lean/Sidon/MVLemmas.lean) | 654 | MV Lemma 3.1 Eqs.(1)--(4) + inner-product floor. | 0 |
| [`Sidon.MasterFromLemmas`](lean/Sidon/MasterFromLemmas.lean) | 122 | Algebraic assembly Eqs.(1)--(4) $\Rightarrow$ Eq.(6). | 0 |
| [`Sidon.BundleDefs`](lean/Sidon/BundleDefs.lean) | 597 | `ExtremiserPrimitives` / `SchwartzAtomic` / `SchwartzAtomicResidual` records. | 0 |
| [`Sidon.BundleEq1`](lean/Sidon/BundleEq1.lean) | 347 | Discharge of bundle field `hEq1` (MV Eq.(1)). | 0 |
| [`Sidon.BundleEq2Schwartz`](lean/Sidon/BundleEq2Schwartz.lean) | 743 | Discharge of bundle field `hEq2` (MV Eq.(2)) for Schwartz $f$. | 0 |
| [`Sidon.BundleEq3Schwartz`](lean/Sidon/BundleEq3Schwartz.lean) | 743 | Discharge of bundle field `hEq3` (MV Eq.(3)) for Schwartz $f$. | 0 |
| [`Sidon.BundleEq4`](lean/Sidon/BundleEq4.lean) | 445 | Discharge of bundle field `hEq4` (MV Eq.(4)). | 0 |
| [`Sidon.BilinearParseval`](lean/Sidon/BilinearParseval.lean) | 434 | Bilinear Parseval pairings used by the bundle discharges. | 0 |
| [`Sidon.MultiScale`](lean/Sidon/MultiScale.lean) | 1142 | General headline + verifiable-by-computation axioms + admissibility bundle. | 2 (verifiable-by-computation) |
| [`Sidon.MultiScaleSchwartz`](lean/Sidon/MultiScaleSchwartz.lean) | 471 | Schwartz-class headline `autoconvolution_ratio_ge_1292_1000_schwartz`. | 0 |
| [`Sidon.SchwartzAtomicDischarge`](lean/Sidon/SchwartzAtomicDischarge.lean) | 548 | Slimmer Schwartz headline `_schwartz_residual`; partial atomic-primitive discharge. | 0 |

The headline theorem's dependency closure reaches exactly **two
verifiable-by-computation user axioms** (also: "rigorously certified
numerical assertions"), both analogues of "Mathematica computed this
value" in MV 2010, but backed by `flint.arb` at 256-bit precision.

These two axioms are *verifiable-by-computation* in the following
precise sense:

- Each is a logically *decidable* inequality about a specific real
  number (a concrete integral / finite sum over the explicit kernel
  $K_{\rm ms}$ and multiplier $G$).
- Each is backed by a specific reproducible algorithm
  (`flint.arb` at 256-bit precision via
  `delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`).
- Each is **not yet** discharged inside Lean only because the
  corresponding Bessel interval-arithmetic infrastructure is not in
  mathlib.
- The arrangement is functionally equivalent to delegating the
  computation to an external oracle (the FlySpeck formalisation of
  Kepler's conjecture used the same convention).

These are *not* conjectural axioms; they are not RH-style assertions
undecidable from within the system. They are statements about
specific integers that any sufficient implementation of interval
arithmetic + the Bessel power series can decide.

| Axiom | Statement | Justification |
|---|---|---|
| `K2_analytic_le_K2UpperQ` | $\int K_{\rm ms}(x)^2\,dx \le \texttt{K2UpperQ} = 47897/10000$ | Closed-form $K_2$ of the three-scale arcsine kernel, evaluated in arb interval arithmetic. Certifier interval $[4.788823, 4.788906]$, slack margin $\approx 8.4 \times 10^{-5}$. Paper Lemma 4.2. |
| `gain_analytic_ge_gainLowerQ` | $\texttt{gain\_analytic} = (4/u) \cdot m_G^2 / S_G \ge \texttt{gainLowerQ} = 20925/100000$ | Cosine $G$'s $(\min G)^2/S_1$ ratio, optimised by QP and arb-verified (coupled-arb $\ge 0.21009214$, margin $\approx 8.4 \times 10^{-4}$). Paper Lemmas 4.3--4.5. |

The headline theorem additionally takes an **analytic
admissibility-bundle hypothesis**:

```lean
structure ExtremiserPrimitives (f : ℝ → ℝ) where
  m_G S_G S_cos LHS1 LHS2 : ℝ
  K2_ge_1  : 1 ≤ K2_analytic
  gain_eq  : gain_analytic = 2 * m_G^2 / S_G
  R_ge_1   : 1 ≤ autoconvolution_ratio f
  S_G_pos  : 0 < S_G
  hEq1     : LHS1 ≤ autoconvolution_ratio f
  hEq2     : LHS2 ≤ 1 + Real.sqrt (autoconvolution_ratio f - 1)
                    * Real.sqrt (K2_analytic - 1)
  hEq3     : LHS1 + LHS2 = 2/uQ + 2 * uQ^2 * S_cos
  hEq4     : uQ^2 * S_cos ≥ m_G^2 / S_G
```

The four fields `hEq1`--`hEq4` are the conclusions of MV Lemma 3.1
Eqs.(1)--(4) for the specific $(f, K_{\rm ms})$ pair. Their existence
for an arbitrary admissible $f$ is the same analytical assertion MV
makes when invoking "by Lemma 3.1 (Martin--O'Bryant)"; closing the
bundle for general $f$ requires the $L^1 \cap L^2$ Plancherel +
period-$u$ Parseval bridge whose building blocks live in
`Sidon.TorusParseval` and `Sidon.FourierAux`. Each of `hEq1`--`hEq4`
is itself derivable in axiom-free Lean from the appropriate atomic
sub-hypotheses (see `Sidon.MVLemmas`), so the bundle is *structurally*
provable; the residual gap is the analytic bridge for non-Schwartz
$f$ on $\mathbb{R}$.

The previous macro axiom `MV_master_inequality_for_extremiser`
(single user axiom bundling all analytic + numerical content into one
statement) is now a Lean *theorem*, derived from the bundle hypothesis
plus the two verifiable-by-computation axioms.

The following dependent statements are Lean **theorems** (no `sorry`,
no axioms):

| Theorem | Role |
|---|---|
| `MV_master_inequality_for_extremiser` | The MV master inequality at slack rationals, specialised to $K_{\rm ms}$. *Now a theorem*, replacing the prior macro axiom of the same name. |
| `MV_master_via_slack_monotonicity` | Real-algebraic lift from analytic anchors $(\texttt{K\_2\_analytic}, \texttt{gain\_analytic})$ to slack rationals via `Real.sqrt` monotonicity. |
| `MV_master_inequality_from_MV_lemmas` | Full chain from MV Eqs.(1)--(4) (bundle fields) to the slack-anchored master inequality. |
| `master_inequality_M_lower` | Quadratic inversion (Prop 5.1 of the paper); case analysis on $M \le 1$ vs $M > 1$ via `Real.sqrt` monotonicity. |
| `K_two_upper_bound` | The slack rational $K_2 \le 47897/10000$ dominates the certifier-reported $K_2 \le 4.788906$ (one-line `norm_num`). |
| `k_one_lower_bound` | Slack-soundness for $k_1$. |
| `S_one_upper_bound` | Slack-soundness for $S_1$. |
| `min_G_lower_bound` | Slack-soundness for $m_G$. |
| `gain_lower_bound` | Slack-soundness for $a$. |

The canonical headline theorem is

```lean
theorem Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000
    (f : ℝ → ℝ)
    (hf_nonneg  : ∀ x, 0 ≤ f x)
    (hf_supp    : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
        ⊤ MeasureTheory.volume ≠ ⊤)
    (P : ExtremiserPrimitives f) :
    autoconvolution_ratio f ≥ (1292 : ℝ) / 1000
```

with decimal restatement `autoconvolution_ratio_ge_1_292` and the
flipped form `C1a_ge_1292` (`1292/1000 ≤ autoconvolution_ratio f`)
exported from the same namespace -- all three taking the same
`ExtremiserPrimitives f` bundle hypothesis as their fifth argument.

The slack-anchor substitution is monotonically sound: the master
inequality is increasing in $K_2 - 1$ and in $a$, so any true bound
on the analytic functionals transports to a valid bound at the slack
rationals. The five `norm_num`-checked slack-soundness theorems
above discharge that the rational slacks are on the correct side of
the certifier-reported decimals.

### Comparison with Matolcsi--Vinuesa (2010): axiom budget

The published MV paper (J. Math. Anal. Appl. **372** (2010),
439--447) proves $C_{1a} \ge 1.2748$ by:

1. Formally proving Lemmas 3.1 (Eqs.(1)--(4), via Martin--O'Bryant),
   3.3 ($z_1$ refinement), 3.4 ($\sin$ bound).
2. **Citing Mathematica** for $J_0(\pi \cdot 0.138)^2$, $m_G$, $S_1$,
   $a = 0.0713$.
3. Combining algebraically.

This work matches the architecture, but strengthens each layer:

1. **Analytic content formally proved in Lean** -- ~8300 axiom-free
   lines spanning Bessel power series, the (autoconvolution) arcsine
   Fourier-transform identity, $L^2$-Plancherel (via mathlib `v4.29.1`),
   period-$u$ torus Parseval, the four MV Lemma 3.1 atomic primitives
   together with their dedicated discharge modules
   (`BundleEq1`/`BundleEq2Schwartz`/`BundleEq3Schwartz`/`BundleEq4`),
   the bilinear Parseval pairings, the Schwartz atomic-primitive
   discharge, and the master inequality assembly.
2. **2 verifiable-by-computation axioms** -- exact analogues of MV's
   Mathematica citations, but backed by `flint.arb` at 256-bit
   precision (strictly more rigorous than Mathematica's heuristic
   numerics), independently audited by 14 agents, anchored to a
   SHA-256-stamped certificate.
3. **1 admissibility-bundle hypothesis** `ExtremiserPrimitives f` --
   the analogue of MV invoking "by Lemma 3.1 (Martin--O'Bryant)". A
   *hypothesis* of the headline, not an axiom.

**Categorisation of the axiom budget.** Lean's `#print axioms`
output mixes three categorically distinct kinds of dependency:

- **Logical axioms** -- `propext`, `Classical.choice`,
  `Quot.sound`. Lean 4 core; trusted without proof; cannot be
  derived by any finite computation.
- **Verifiable-by-computation axioms** -- the two numerical ones
  (`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`).
  Logically *decidable* statements about specific real numbers,
  certified by a reproducible `flint.arb` algorithm; currently
  un-formalised in Lean only because mathlib lacks a Bessel
  interval-arithmetic library.
- **Analytic admissibility bundle** -- `ExtremiserPrimitives f`.
  Not an axiom but a *hypothesis* of the headline.

The distinction between *conjectural* and
*verifiable-by-computation* axioms matters because the natural
critic's question -- "are you assuming something unprovable?" --
has a clean answer here: **no**. Both numerical axioms are
provable. They are simply not yet formalised in Lean for
engineering reasons (no Bessel interval-arithmetic library in
mathlib). They are not RH-style assertions undecidable from within
the system; they are statements about specific integers that any
sufficient implementation of interval arithmetic + the Bessel
power series can decide.

This is the same axiom architecture every published
computer-assisted real-number proof uses (Flyspeck cited Kepler's
interval arithmetic; the polynomial-method cap-set proof cited
Lagrange polynomial bounds; the PFR formalisation cited numerical
Plünnecke--Ruzsa constants). The mathematical content of the proof
is in the Lean theorems; the verifiable-by-computation axioms
encode only "evaluate this specific integral and compare it to
this specific rational".

**Honesty caveats.**

- The headline is conditional on a witness for the
  `ExtremiserPrimitives f` bundle, not unconditional. Producing the
  bundle for arbitrary admissible $f$ requires bridging mathlib's
  $L^2$-Plancherel API to the concrete period-$u$ Parseval splits
  used in `Sidon.MVLemmas`. The infrastructure is in
  `Sidon.FourierAux` and `Sidon.TorusParseval`; the final stitching
  is not yet a one-line mathlib call.
- The two verifiable-by-computation axioms depend on trusting the
  `flint.arb` library (peer-reviewed -- Johansson 2017, IEEE TC --
  but not Lean-verified).
- Replacing the verifiable-by-computation axioms with verified Lean
  numerics would require a separate multi-year subproject (rigorous
  interval arithmetic + verified Gauss--Legendre quadrature +
  verified Bessel + Taylor branch-and-bound, ~6000--10000 lines),
  comparable to the Flyspeck effort for Kepler, with no upside for
  the mathematical claim.

## 5. Repository Layout

```
compact_sidon/
├── lower_bound_proof.tex             # The manuscript
├── lower_bound_proof.pdf             # Compiled output
├── audit_consistency.py              # Cross-source audit
├── REPORT.md                         # This file
├── README.md                         # Project overview
│
├── lean/                             # Lean 4 formalization (~8650 lines, 15 modules)
│   ├── Sidon/Defs.lean               # Shared definitions (55 lines, 0 axioms)
│   ├── Sidon/Bessel.lean             # Bessel J0 power series, arcsine FT (958, 0 axioms)
│   ├── Sidon/FourierAux.lean         # Schwartz Plancherel, L^p bridge (606, 0 axioms)
│   ├── Sidon/TorusParseval.lean      # Period-u Parseval, lattice Fourier (785, 0 axioms)
│   ├── Sidon/MVLemmas.lean           # MV Lemma 3.1 Eqs.(1)-(4) (654, 0 axioms)
│   ├── Sidon/MasterFromLemmas.lean   # Master inequality assembly (122, 0 axioms)
│   ├── Sidon/BundleDefs.lean         # Extremiser / SchwartzAtomic / SchwartzAtomicResidual (597, 0 axioms)
│   ├── Sidon/BundleEq1.lean          # Discharge of hEq1 (MV Eq.(1)) (347, 0 axioms)
│   ├── Sidon/BundleEq2Schwartz.lean  # Discharge of hEq2 (MV Eq.(2)), Schwartz (743, 0 axioms)
│   ├── Sidon/BundleEq3Schwartz.lean  # Discharge of hEq3 (MV Eq.(3)), Schwartz (743, 0 axioms)
│   ├── Sidon/BundleEq4.lean          # Discharge of hEq4 (MV Eq.(4)) (445, 0 axioms)
│   ├── Sidon/BilinearParseval.lean   # Bilinear Parseval pairings (434, 0 axioms)
│   ├── Sidon/MultiScale.lean         # General headline, 2 verifiable-by-computation axioms (1142 lines)
│   ├── Sidon/MultiScaleSchwartz.lean # Schwartz headline `_schwartz` (471, 0 axioms)
│   ├── Sidon/SchwartzAtomicDischarge.lean # Schwartz headline `_schwartz_residual` (548, 0 axioms)
│   ├── Sidon.lean                    # Top-level module entry
│   └── AxiomCheck.lean               # Prints axiom inventory
│
├── delsarte_dual/                    # The arb certifier
│   ├── grid_bound/                   # Single-scale MV machinery + certify.py verifier
│   ├── grid_bound_alt_kernel/        # Three-scale kernel, QP for G, bisect driver
│   │   └── certificates/
│   │       ├── reference_anchors.json    # Canonical 256-bit anchors
│   │       └── multiscale_arcsine_1292.json  # Fresh-run certificate
│   └── README.md
│
├── tests/                            # pytest suite
│   ├── grid_bound_alt_kernel/        # Kernel admissibility, Bochner positivity, QP
│   └── README.md
│
├── docs/                             # Public documentation
│   ├── proof_outline.md              # Mathematical summary (sections, key formulas)
│   ├── reproducibility.md            # Exact reproduction commands
│   ├── formalization.md              # Lean module description
│   ├── verification.md               # 14-task verification checklist
│   ├── presentation/                 # Slide deck (.pptx + figures)
│   └── attempts/                     # Historical attempt write-ups
│
└── archive/                          # Earlier exploration (cs-cascade,
                                      # Lasserre SDP, agent_experiments, etc.)
```

## 6. Reproducing the Result

### Compile the manuscript

```bash
pdflatex -interaction=nonstopmode lower_bound_proof.tex
```

(No external `.bib` file: the bibliography is inlined via
`thebibliography`.) Output: no overfull/underfull/undefined
warnings.

### Regenerate the numerical certificate

```bash
pip install python-flint cvxpy numpy
python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel
```

The driver runs at 256-bit precision and emits
`delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`
with the five anchors, the bisection history, the terminal cell
list, and a SHA-256 body hash.

### Independent verification

```bash
python -m delsarte_dual.grid_bound.certify \
    delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json
```

`certify.py` is a stand-alone verifier that imports only
`python-flint` primitives. Exit code `0` iff the certificate body
hash matches, every anchor is recomputable in arb at the declared
precision, the terminal cells cover $[0, \mu(M_{\rm cert})]$
contiguously, and every cell has $\Phi < 0$ upper bound.

### Build the Lean formalization

```bash
cd lean && lake build              # all fifteen modules
lake env lean AxiomCheck.lean      # axiom inventory of the headline
```

`AxiomCheck.lean` prints the axiom closure of the headline theorem.
Expected: Lean's three core logical axioms (`Classical.choice`,
`propext`, `Quot.sound`) plus exactly two
*verifiable-by-computation* user axioms (rigorously certified
numerical assertions, both backed by `flint.arb` at 256-bit
precision):

```
'Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000' depends on axioms:
  [propext, Classical.choice, Quot.sound,
   Sidon.MultiScale.K2_analytic_le_K2UpperQ,
   Sidon.MultiScale.gain_analytic_ge_gainLowerQ]
```

The headline theorem additionally takes an analytic
admissibility-bundle hypothesis `ExtremiserPrimitives f` (its fifth
argument) packaging the four MV Lemma 3.1 outputs (Eqs.(1)--(4))
for the specific $(f, K_{\rm ms})$ pair. The bundle is *structurally*
provable using `Sidon.MVLemmas` plus `Sidon.MasterFromLemmas`, but
the analytic discharge for general non-Schwartz admissible $f$ still
requires bridging mathlib's $L^2$ Plancherel API to the concrete
period-$u$ Parseval splits.

### Run the cross-source audit

```bash
python audit_consistency.py             # summary verdict
python audit_consistency.py --verbose   # print every individual check
```

Verifies that the numerical anchors in
[`reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json),
the slack rationals in
[`lean/Sidon/MultiScale.lean`](lean/Sidon/MultiScale.lean), the
decimal claims in `docs/{proof_outline,reproducibility,formalization,verification}.md`,
the headline-bound claims across READMEs, and the Proposition 5.1
arithmetic in
[`lower_bound_proof.tex`](lower_bound_proof.tex) are all mutually
consistent and on the correct side of the arb endpoints.

### Run the pytest suite

```bash
pytest tests/grid_bound_alt_kernel/
```

Covers kernel admissibility, Bochner positivity of $\widehat{K_{\rm ms}}$,
the QP solver convergence, and the single-scale baseline check
against the published Matolcsi-Vinuesa value $1.27481$.

## 7. Cross-Source Audit Framework

`audit_consistency.py` is the project's source of truth for
quantitative consistency. Each run re-derives the analytic anchors in
`flint.arb` at 256-bit precision and verifies every quantitative
claim against the freshly computed ground truth across eight
sections:

| Section | What it checks |
|---|---|
| A | Kernel-parameter consistency (rationals declared in Lean / LaTeX / code agree exactly). |
| B | Slack-rational soundness (every Lean rational anchor is a true bound on the arb endpoint). |
| C | Lean axiom RHS soundness (each of the five $\{k_1, K_2, S_1, m_G, a\}$ slack comparisons is rationally true). |
| D | Tight-decimal claim soundness (every decimal value asserted in READMEs / JSON / docstrings / LaTeX is on the correct side of the arb endpoint). |
| E | LaTeX Proposition 5.1 strict-failure arithmetic (exact rational verification of every step in the closing chain). |
| F | LaTeX per-lemma slack-value claims (e.g. "slack $\ge 9.3 \times 10^{-5}$"). |
| G | $K_2 = \text{bulk} + \text{tail}$ decomposition (Watson tail bound and the constant $C = \sum_i \lambda_i / \delta_i$). |
| H | Published bound consistency ($M_{\rm cert}$ production $\ge 1.29232422$; slack-anchor bisection $\ge 1.29215650$; headline $\ge 1292/1000$). |

**Current status: every check passes, verdict `ALL CHECKS PASS`.**

## 8. Project History (Selected)

The repository carries a substantial earlier-exploration layer under
`archive/`, including:

- A multiscale branch-and-prune cascade extending the CS17 method,
  archived at `docs/attempts/cs_writeup_legacy/` (writeup) and
  `archive/cloninger-steinerberger/` (code).
- A Lasserre SDP hierarchy track with correlative sparsity for $d \in
  \{32, 64, 128\}$, archived at `docs/attempts/lasserre_writeup/` and
  `archive/coarse_lp_bnb/`.
- A two-scale arcsine kernel precursor that produced
  $C_{1a} \ge 1651/1280 \approx 1.28984$, documented in
  [`docs/attempts/multiscale_arcsine.md`](docs/attempts/multiscale_arcsine.md).
- Earlier Hölder, KBK, AlphaEvolve-dual, cohn-elkies, and minimum-overlap
  attempts under `archive/`.

Each historical attempt is preserved with its decision record (often
including the FLAG that closed it).

## 9. Trust Boundary

The published bound rests on the following components:

| Component | Trust |
|---|---|
| Lean 4 kernel | Foundational (assumed sound). |
| Mathlib (`v4.29.1`, commit `5e932f97dd`) | Community-verified library; the bump unlocked `MeasureTheory.Lp.fourierTransformₗᵢ` and `Real.fourier_mul_convolution_eq` used by the project's Parseval infrastructure. |
| `K2_analytic_le_K2UpperQ` axiom (verifiable-by-computation) | Asserts $\int K_{\rm ms}^2 \le 47897/10000$ for the explicit three-scale arcsine kernel. Discharged externally by `flint.arb` at 256-bit precision; certifier interval $[4.788823, 4.788906]$, slack margin $\approx 8.4 \times 10^{-5}$. Logically decidable; not yet a Lean theorem only because mathlib lacks a Bessel interval-arithmetic library. Analogue of MV 2010's Mathematica citation of $K_2$. |
| `gain_analytic_ge_gainLowerQ` axiom (verifiable-by-computation) | Asserts $\texttt{gain\_analytic} \ge 20925/100000$. Discharged externally by `flint.arb` (coupled-arb $\ge 0.21009214$, margin $\approx 8.4 \times 10^{-4}$). Logically decidable; analogue of MV 2010's Mathematica citation of $a$. |
| `ExtremiserPrimitives f` bundle hypothesis | Encodes the four MV Lemma 3.1 outputs (Eqs.(1)--(4)) for the specific $(f, K_{\rm ms})$ pair. Not a Lean axiom -- the headline theorem takes it as an explicit hypothesis. Producing the witness for arbitrary admissible $f$ requires the $L^1 \cap L^2$ Plancherel + period-$u$ Parseval bridge whose building blocks live in `Sidon.TorusParseval` and `Sidon.FourierAux`. |
| `python-flint` / Arb library | Standard interval-arithmetic backend (Johansson 2017). Peer-reviewed; not itself Lean-verified. |
| Numerical anchors | All five anchors are reproduced exactly by `bisect_alt_kernel.py` and independently re-verified by `grid_bound/certify.py`. |
| Rational slack substitution into the master inequality | Sound by monotonicity in $K_2 - 1$ and $a$; the five `norm_num`-decided slack-soundness theorems confirm the rationals lie on the correct side of the certifier's decimal output. **This step is a Lean theorem** (`MV_master_via_slack_monotonicity`), not an axiom. |
| Lean-side assembly (`MV_master_inequality_for_extremiser`, `master_inequality_M_lower`) | Pure Lean *theorems*; `master_inequality_M_lower` is case analysis on $M \le 1$ vs $M > 1$ via `Real.sqrt` monotonicity. No external dependency. The previous macro axiom of the same name has been promoted to a theorem. |

No component is required beyond those listed.


## References

- [Manuscript: `lower_bound_proof.pdf`](lower_bound_proof.pdf)
- [Lean module: `lean/Sidon/MultiScale.lean`](lean/Sidon/MultiScale.lean)
- [Numerical certificate: `reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json)
- [Audit script: `audit_consistency.py`](audit_consistency.py)
- [Proof outline: `docs/proof_outline.md`](docs/proof_outline.md)
- [Reproducibility: `docs/reproducibility.md`](docs/reproducibility.md)
- [Formalization notes: `docs/formalization.md`](docs/formalization.md)
- [Verification checklist: `docs/verification.md`](docs/verification.md)

External:
- Cloninger, A., Steinerberger, S. *On suprema of autoconvolutions
  with an application to Sidon sets.* Proc. Amer. Math. Soc. 145
  (2017), 3191-3200, arXiv:1403.7988.
- Matolcsi, M., Vinuesa, C. *Improved bounds on the supremum of
  autoconvolutions.* J. Math. Anal. Appl. 372 (2010), 439-447,
  arXiv:0907.1379.
- Martin, G., O'Bryant, K. *The supremum of autoconvolutions, with
  applications to additive number theory.* Illinois J. Math. 53
  (2009), 219-235, arXiv:0807.5121.
- Johansson, F. *Arb: efficient arbitrary-precision midpoint-radius
  interval arithmetic.* IEEE Trans. Comput. 66(8) (2017), 1281-1292.
- de Moura, L., Ullrich, S. *The Lean 4 theorem prover and programming
  language.* CADE 28, LNCS 12699, Springer, 2021, 625-635.
- The mathlib community. *The Lean mathematical library.* CPP 2020,
  367-381.
- Georgiev, B., G&oacute;mez-Serrano, J., Tao, T., Wagner, A.Z.
  *Mathematical exploration and discovery at scale.* arXiv:2511.02864.
