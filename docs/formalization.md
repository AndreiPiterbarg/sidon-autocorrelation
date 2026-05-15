# Lean 4 Formalization

The analytic chain of the **Piterbarg--Bajaj--Vincent Bound** is
mechanised in Lean 4 under the namespace `Sidon.MultiScale`. The
formalisation lives under [`../lean/Sidon/`](../lean/Sidon/) and is
spread across fifteen modules (~8650 lines total) on top of `Mathlib`
pinned to `v4.29.1`, commit
[`5e932f97dd25535344f80f9dd8da3aab83df0fe6`](https://github.com/leanprover-community/mathlib4/commit/5e932f97dd25535344f80f9dd8da3aab83df0fe6).
The bump to `v4.29.1` (post-Nov 2025) unlocked the L^2 Plancherel API
(`MeasureTheory.Lp.fourierTransformₗᵢ`) and convolution--Fourier duality
(`Real.fourier_mul_convolution_eq`), which are the foundations of the
Parseval-on-the-torus infrastructure.

The full module builds with $0$ `sorry` tactics. The headline theorem's
`#print axioms` listing reaches Lean's three core logical axioms
(`propext`, `Classical.choice`, `Quot.sound`) together with exactly
**two verifiable-by-computation axioms** (also called "rigorously
certified numerical assertions") declared in `Sidon.MultiScale` plus an
analytic admissibility *bundle hypothesis* (`ExtremiserPrimitives f`)
that the consumer must supply.

These two axioms are *verifiable-by-computation* in the following
precise sense: each is a logically decidable inequality about specific
real numbers, backed by `flint.arb` at 256-bit precision (driver
`delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`); the only reason they
appear as `axiom` in Lean rather than as `theorem` is that mathlib does
not yet ship a Bessel interval-arithmetic library to discharge them
mechanically. They are analogous to the Mathematica citations in
Matolcsi--Vinuesa (2010) but strictly more rigorous (proven interval
bounds vs. heuristic numerics, reproducible SHA-256-anchored
certificate, independent 14-agent audit). The FlySpeck formalisation of
Kepler's conjecture used the same convention. The
quadratic inversion `master_inequality_M_lower` and the five
slack-soundness statements (`K_two_upper_bound`, `k_one_lower_bound`,
`S_one_upper_bound`, `min_G_lower_bound`, `gain_lower_bound`,
discharging paper Lemmas 4.1--4.5 as pure rational `norm_num` checks)
are Lean *theorems* and do not contribute axioms to the dependency
closure.

## Module layout

| Module | Lines | Content | Axioms |
|--------|------:|---------|--------|
| [`Sidon.Defs`](../lean/Sidon/Defs.lean) | 55 | Core definitions (`autoconvolution_ratio`, admissibility predicates, kernel structures; convolutional `f ∘ f`). | 0 |
| [`Sidon.Bessel`](../lean/Sidon/Bessel.lean) | 958 | Bessel $J_0$ power series, autoconvolution arcsine FT identity $\widehat{K_{\rm arc}(\delta;\cdot)}(\xi) = J_0(\pi \delta \xi)^2$ for $K_{\rm arc} = \eta_\delta * \eta_\delta$, Watson tail bound. | 0 |
| [`Sidon.FourierAux`](../lean/Sidon/FourierAux.lean) | 606 | Schwartz Plancherel, $L^p$ bridge, $L^1$-pairing, all the auxiliary Fourier machinery on $\mathbb{R}$ not directly in mathlib. | 0 |
| [`Sidon.TorusParseval`](../lean/Sidon/TorusParseval.lean) | 785 | Period-$u$ Parseval, lattice Fourier, bilinear pairing $\int f g = \sum_n \widehat f(n/u) \overline{\widehat g(n/u)}$ on $\mathbb{R}/u\mathbb{Z}$. | 0 |
| [`Sidon.MVLemmas`](../lean/Sidon/MVLemmas.lean) | 654 | The four Matolcsi--Vinuesa Lemma 3.1 atomic primitives Eqs.(1)--(4), the inner-product floor, and the Cauchy--Schwarz tail estimate. | 0 |
| [`Sidon.MasterFromLemmas`](../lean/Sidon/MasterFromLemmas.lean) | 122 | Algebraic assembly: from Eqs.(1)--(4) at the analytic anchors to the MV master inequality Eq.(6). | 0 |
| [`Sidon.BundleDefs`](../lean/Sidon/BundleDefs.lean) | 597 | The three bundle records `ExtremiserPrimitives`, `SchwartzAtomic`, `SchwartzAtomicResidual`. | 0 |
| [`Sidon.BundleEq1`](../lean/Sidon/BundleEq1.lean) | 347 | Discharge of bundle field `hEq1` (MV Eq.(1)). | 0 |
| [`Sidon.BundleEq2Schwartz`](../lean/Sidon/BundleEq2Schwartz.lean) | 743 | Discharge of bundle field `hEq2` (MV Eq.(2)) for Schwartz $f$. | 0 |
| [`Sidon.BundleEq3Schwartz`](../lean/Sidon/BundleEq3Schwartz.lean) | 743 | Discharge of bundle field `hEq3` (MV Eq.(3)) for Schwartz $f$. | 0 |
| [`Sidon.BundleEq4`](../lean/Sidon/BundleEq4.lean) | 445 | Discharge of bundle field `hEq4` (MV Eq.(4)). | 0 |
| [`Sidon.BilinearParseval`](../lean/Sidon/BilinearParseval.lean) | 434 | Bilinear Parseval pairings used by the bundle discharges. | 0 |
| [`Sidon.MultiScale`](../lean/Sidon/MultiScale.lean) | 1142 | General headline `autoconvolution_ratio_ge_1292_1000`, verifiable-by-computation axioms, slack-soundness theorems, three-scale kernel anchors. | 2 (verifiable-by-computation) |
| [`Sidon.MultiScaleSchwartz`](../lean/Sidon/MultiScaleSchwartz.lean) | 471 | Schwartz-class headline `autoconvolution_ratio_ge_1292_1000_schwartz` consuming the `SchwartzAtomic` record. | 0 |
| [`Sidon.SchwartzAtomicDischarge`](../lean/Sidon/SchwartzAtomicDischarge.lean) | 548 | Slimmer Schwartz headline `autoconvolution_ratio_ge_1292_1000_schwartz_residual` consuming `SchwartzAtomicResidual`; partial discharge of atomic Fourier primitives. | 0 |

Earlier versions of the project had a single ~388-line
`MultiScale.lean` (formerly `MultiScaleRigorous.lean`) carrying a single
macro axiom `MV_master_inequality_for_extremiser` that bundled all
analytic and numerical content. The current post-Wave-12 structure
**factors that macro axiom into axiom-free analytic Lean
infrastructure plus two narrow verifiable-by-computation axioms**; the
macro axiom itself is now a Lean *theorem*
(`MV_master_inequality_for_extremiser`) whose proof reduces to (i)
the `ExtremiserPrimitives` bundle hypothesis encoding MV Lemma
3.1 outputs for $(f, K_{\rm ms})$, plus (ii) the two
verifiable-by-computation axioms below.

### Recent fixes (Wave-12 multi-agent audit)

Two math-fidelity corrections landed during a multi-agent audit and
are reflected in the post-Wave-12 build:

- **`f ∘ f` convention.** Tightened from a pointwise product to the
  *convolutional* form $(f \circ f)(x) := \int f(t)\, f(x - t)\, dt$
  used in MV 2010. All downstream MV Lemma 3.1 bundle fields and the
  master inequality assembly were re-derived accordingly.
- **`K_arc` definition.** Refactored from the bare arcsine density to
  the autoconvolution $K_{\rm arc}(\delta;\cdot) = \eta_\delta *
  \eta_\delta$, with $\eta_\delta$ the rescaled indicator. This makes
  the Bessel identity
  $\widehat{K_{\rm arc}(\delta;\cdot)}(\xi) = J_0(\pi \delta \xi)^2$
  the literal Parseval-on-the-Fourier-transform statement rather than
  the Sonine identity.

Both fixes are build-clean (`lake build` green, 0 sorries) under the
same 5-axiom budget (3 Lean core + 2 verifiable-by-computation), with
the three exported headlines remaining non-vacuous.

## Headline theorems

The repository exports three headline theorems, sharing the same
underlying analytic chain but consuming different admissibility
records:

- **General**
  `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
  (`MultiScale.lean`), hypothesised on the
  `ExtremiserPrimitives f` record.
- **Schwartz**
  `Sidon.MultiScaleSchwartz.autoconvolution_ratio_ge_1292_1000_schwartz`
  (`MultiScaleSchwartz.lean`), hypothesised on the
  `SchwartzAtomic f_s` record for a Schwartz function $f_s$.
- **Schwartz (slimmer hypothesis)**
  `Sidon.SchwartzAtomicDischarge.autoconvolution_ratio_ge_1292_1000_schwartz_residual`
  (`SchwartzAtomicDischarge.lean`), hypothesised on the slimmer
  `SchwartzAtomicResidual f_s` record (the `_residual` variant
  carries only the genuinely *residual* analytic content; the
  remaining fields are discharged inside the same module).

The general headline reads

```lean
theorem autoconvolution_ratio_ge_1292_1000 (f : ℝ → ℝ)
    (hf_nonneg  : ∀ x, 0 ≤ f x)
    (hf_supp    : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤)
    (P : ExtremiserPrimitives f) :
    autoconvolution_ratio f ≥ (1292 / 1000 : ℝ)
```

The first four hypotheses are nonnegativity, support inside $(-1/4,
1/4)$, strict positivity of $\int f$, and finiteness of $\|f * f\|_\infty$
(an `ENNReal.toReal` encoding artifact; harmless when passing to the
infimum). The fifth hypothesis, $P : \texttt{ExtremiserPrimitives}\;f$,
is the *analytic admissibility bundle* defined below.
Equivalent restatements `autoconvolution_ratio_ge_1_292` (decimal form)
and `C1a_ge_1292` (`1292/1000 ≤ autoconvolution_ratio f`)
are exported from the same namespace; both take the same bundle hypothesis.

## The two verifiable-by-computation axioms

| Axiom name | Statement | Discharged by |
|------------|-----------|---------------|
| `K2_analytic_le_K2UpperQ` | $K_2(K_{\rm ms}) \le \texttt{K2UpperQ} = 47897/10000$. Here $K_2(K_{\rm ms}) := \int_{\mathbb{R}} K_{\rm ms}(x)^2\,dx$ is a concrete real integral over the explicit three-scale arcsine kernel. | The `flint.arb` certifier at 256-bit precision: $K_2 \in [4.788823, 4.788906]$, radius $< 10^{-4}$; slack $4.7897$ exceeds the upper endpoint with margin $\approx 8.4 \times 10^{-5}$. Paper Lemma 4.2. |
| `gain_analytic_ge_gainLowerQ` | $\texttt{gain\_analytic} \ge \texttt{gainLowerQ} = 20925/100000$. Here $\texttt{gain\_analytic} = (4/u)\cdot m_G^2 / S_G$ with $m_G = \min_{[0,1/4]} G$ and $S_G = \sum_j \widetilde G(j)^2 / \widehat K_{\rm ms}(j/u)$ for the QP-optimised cosine $G$. | The `flint.arb` certifier: coupled-arb value $\ge 0.21009214$, radius $< 10^{-8}$; slack $0.20925$ is below the certifier's lower endpoint with margin $\approx 8.4 \times 10^{-4}$. Paper Lemmas 4.3--4.5. |

Both axioms are **verifiable-by-computation** in the precise sense
defined in the introduction: each is a logically decidable inequality
about a specific real number (a concrete integral or finite sum over
the explicit kernel $K_{\rm ms}$ / multiplier $G$), backed by a specific
reproducible algorithm that produces the certificate (`flint.arb` at
256-bit precision, driver `delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`).
They are *not* conjectural -- both are provable; they are simply not
yet discharged inside Lean because the corresponding Bessel
interval-arithmetic infrastructure is not in mathlib. They are
**analogues of "Mathematica computed this value" in the published
Matolcsi--Vinuesa paper** -- certifier outputs, not analytic content --
but strictly more rigorous: proven interval bounds rather than
heuristic numerics, reproducible SHA-256-anchored certificate, and an
independent 14-agent audit. Both quantities are defined symbolically in
Lean as concrete real integrals / finite sums over the explicit
three-scale kernel $K_{\rm ms}$ and the QP-optimised cosine $G$, so the
axioms are non-trivial analytic statements (not definitional
shortcuts).

## The analytic admissibility bundle

The fifth hypothesis of the headline is the structure

```lean
structure ExtremiserPrimitives (f : ℝ → ℝ) where
  m_G S_G S_cos LHS1 LHS2 : ℝ
  K2_ge_1 : 1 ≤ K2_analytic
  gain_eq : gain_analytic = 2 * m_G ^ 2 / S_G
  R_ge_1  : 1 ≤ autoconvolution_ratio f
  S_G_pos : 0 < S_G
  hEq1 : LHS1 ≤ autoconvolution_ratio f
  hEq2 : LHS2 ≤ 1 + Real.sqrt (autoconvolution_ratio f - 1)
                  * Real.sqrt (K2_analytic - 1)
  hEq3 : LHS1 + LHS2 = 2 / uQ + 2 * uQ^2 * S_cos
  hEq4 : uQ^2 * S_cos ≥ m_G^2 / S_G
```

The four fields `hEq1`--`hEq4` are the conclusions of MV Lemma 3.1
Eqs.(1)--(4) instantiated at $(f, K_{\rm ms})$. Their *existence* is
the same analytical assertion that MV makes implicitly in the
single-scale derivation: given an admissible $f$ and an admissible
kernel $K$, the four atomic identities/inequalities of MV Lemma 3.1
hold. The Lean module `Sidon.MVLemmas` proves all four conclusions
**axiom-free** from the appropriate atomic sub-hypotheses
(period-$u$ Parseval splits for $f*f$ and $f \circ f$, the $F$-bound,
the inner-product floor, etc.), and `Sidon.MasterFromLemmas` chains
them into the master inequality Eq.(6) axiom-free as well.

The bundle is kept as a *hypothesis* on the headline (rather than
discharged automatically) because the Parseval splits and $L^1 \cap
L^2$ pairings for *non-Schwartz* admissible $f$ on the torus
$\mathbb{R}/u\mathbb{Z}$ are not yet in mathlib in a directly usable
form. The `Sidon.TorusParseval` and `Sidon.FourierAux` modules
contain the building blocks (period-$u$ Parseval, lattice Fourier,
bilinear pairings via mathlib's `MeasureTheory.Lp.fourierTransformₗᵢ`),
but the final stitching -- producing an `ExtremiserPrimitives f`
witness for an arbitrary admissible $f$ -- requires bridges that are
*possible* with the current mathlib but not *immediate*. Closing this
gap would lift the headline from "conditional on a witness for the
bundle" to fully unconditional; doing so is the remaining residual
work on the Lean side.

## Lean theorems backing the headline

The quadratic-in-$M$ inversion and the five slack-soundness statements
are Lean *theorems*, not axioms.

| Theorem name | Statement | Proof |
|--------------|-----------|-------|
| `master_inequality_M_lower` | If $a_{\rm lo} \ge \texttt{gainLowerQ}$ and $M + 1 + \sqrt{M-1}\sqrt{\texttt{K2UpperQ} - 1} \ge 2/u + a_{\rm lo}$ then $M \ge \texttt{MTargetQ} = 1292/1000$ | At $M = 1292/1000$ the LHS attains $\Phi(M) \le 66879/20000 = 3.34395$, strictly below $\tau = 2/u + \texttt{gainLowerQ} = 4267003/1276000 \approx 3.344046$, with margin $307/3190000 \ge 9.6 \times 10^{-5}$. Proved by case analysis on $M \le 1$ versus $M > 1$ using `Real.sqrt` monotonicity and `nlinarith`. Paper Proposition 5.1. |
| `MV_master_via_slack_monotonicity` | Real-algebraic lift from the master inequality at the analytic anchors $(K_{2,\rm analytic},\,\texttt{gain\_analytic})$ to the slack rationals $(\texttt{K2UpperQ},\,\texttt{gainLowerQ})$. | Monotonicity of $\sqrt{\cdot}$ together with the two slack inequalities $K_{2,\rm analytic} \le \texttt{K2UpperQ}$ and $\texttt{gain\_analytic} \ge \texttt{gainLowerQ}$. Zero axioms. |
| `MV_master_inequality_from_MV_lemmas` | Full chain from MV Lemma 3.1 Eqs.(1)--(4) (as bundle fields) plus the two kernel-specific bounds to the slack-anchored master inequality. | Composes `Sidon.Master.master_inequality_from_lemmas` with `MV_master_via_slack_monotonicity`. Zero axioms. |
| `MV_master_inequality_for_extremiser` | The MV master inequality with slack rationals substituted for $K_2$ and $a$, specialised to the three-scale kernel and conditional on `ExtremiserPrimitives f`. **Now a theorem, replacing the prior macro axiom of the same name.** | The bundle's `hEq1`--`hEq4` feed into `MV_master_inequality_from_MV_lemmas`; the two verifiable-by-computation axioms discharge `K2_analytic ≤ K2UpperQ` and `2·m_G^2/S_G ≥ gainLowerQ` (the latter via the bundle's `gain_eq` field). |
| `K_two_upper_bound`   | $\texttt{K2UpperQ} \ge 4788906/1000000$       | `norm_num` rational comparison; certifier-reported $K_2 \le 4.788906$ (paper Lemma 4.2). |
| `k_one_lower_bound`   | $\texttt{K1LowerQ} \le 92124658/100000000$    | `norm_num` rational comparison; certifier-reported $k_1 \ge 0.92124658$ (paper Lemma 4.1). |
| `S_one_upper_bound`   | $\texttt{S1UpperQ} \ge 2984091/100000$        | `norm_num` rational comparison; certifier-reported $S_1 \le 29.840907$ (paper Lemma 4.3). |
| `min_G_lower_bound`   | $\texttt{minGLowerQ} \le 9999798/10000000$    | `norm_num` rational comparison; certifier-reported $\min_{[0,1/4]} G \ge 0.99997987$ (paper Lemma 4.4). |
| `gain_lower_bound`    | $\texttt{gainLowerQ} \le 21009214/100000000$  | `norm_num` rational comparison; certifier-reported $a \ge 0.21009214$ (paper Lemma 4.5). |

The five `norm_num` theorems record that the rational slacks fed into
the master inequality are on the correct side of the certifier-reported
decimals; they are *not* axioms about the analytic functionals, and they
do not appear in the dependency closure of the headline theorem.

## Correspondence with the paper

The Lean modules are consistent with the paper *A New Lower Bound for
the Supremum of Autoconvolutions*, which proposes the
Piterbarg--Bajaj--Vincent Bound:

- The two verifiable-by-computation axioms
  (`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`) correspond
  to paper Lemmas 4.2 and 4.3--4.5 respectively (the closed-form $K_2$
  and the QP-optimised gain $a$, both bounded in `flint.arb`).
- The `ExtremiserPrimitives` bundle is the formal counterpart of the
  paper's invocation of MV Lemma 3.1 -- the four equations on which the
  master inequality rests.
- The theorem `MV_master_inequality_for_extremiser` corresponds to
  Theorem 2.3 of the paper (the three-scale master inequality
  Eq.(7) at the slack rationals).
- The theorem `master_inequality_M_lower` corresponds to Proposition
  5.1 of the paper (the strict-failure witness at $M = 1292/1000$).
- The five `norm_num` theorems record the soundness of the slack
  rationals against the certifier-reported decimals of paper Lemmas
  4.1--4.5.

For the analytic chain combining these into the headline statement, see
Section 5 of [`lower_bound_proof.pdf`](../lower_bound_proof.pdf).

## Comparison with Matolcsi--Vinuesa (2010): axiom budget

The published Matolcsi--Vinuesa paper (J. Math. Anal. Appl. **372**
(2010), 439--447) proves $C_{1a} \ge 1.2748$ by:

1. Formally proving Lemmas 3.1 (Eqs.(1)--(4), via Martin--O'Bryant),
   3.3 ($z_1$ refinement), and 3.4 (the $\sin$ bound).
2. **Citing Mathematica** for the numerical values of
   $J_0(\pi\cdot 0.138)^2$, $m_G$, $S_1$, and $a = 0.0713$.
3. Combining 1 and 2 algebraically to obtain $1.2748$.

The present Lean proof of $C_{1a} \ge 1.292$:

1. **Formally proves the analytic content in Lean** -- approximately
   8650 axiom-free lines spanning the autoconvolution arcsine
   Fourier-transform identity (`Sidon.Bessel`), the $L^2$ Plancherel
   and Schwartz apparatus (`Sidon.FourierAux`, on top of mathlib's
   `MeasureTheory.Lp.fourierTransformₗᵢ` introduced in `v4.29.1`),
   the period-$u$ torus Parseval and lattice-Fourier identities
   (`Sidon.TorusParseval`), the four MV Lemma 3.1 atomic primitives
   (`Sidon.MVLemmas`) together with their dedicated discharge modules
   (`Sidon.BundleEq1`, `Sidon.BundleEq2Schwartz`,
   `Sidon.BundleEq3Schwartz`, `Sidon.BundleEq4`,
   `Sidon.BilinearParseval`), the partial Schwartz atomic-primitive
   discharge (`Sidon.SchwartzAtomicDischarge`), and the master
   inequality assembly (`Sidon.MasterFromLemmas`,
   `Sidon.MultiScaleSchwartz`).
2. **2 verifiable-by-computation axioms** -- exact analogues of MV's
   Mathematica citations, but backed by 256-bit `flint.arb` interval
   arithmetic (strictly more rigorous than Mathematica's heuristic
   numerics), independently audited by 14 agents, and anchored to a
   SHA-256-stamped reproducible certificate.
3. **1 admissibility-bundle hypothesis** (`ExtremiserPrimitives f`) --
   the analogue of MV invoking "by Lemma 3.1 (Martin--O'Bryant)"; the
   bundle's existence for a specific admissible $f$ is the same analytic
   assertion MV makes about his arbitrary admissible $f$. This is a
   *hypothesis* of the headline theorem, not an axiom.

**Categorisation of the axiom budget.** Lean's `#print axioms` output
mixes three categorically distinct kinds of dependency, which we
separate explicitly here:

- **Logical axioms** -- `propext`, `Classical.choice`, `Quot.sound`.
  These are Lean 4 core axioms the kernel trusts without proof; they
  cannot be derived by any finite computation.
- **Verifiable-by-computation axioms** -- the two numerical ones
  (`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`).
  Logically *decidable* statements about specific real numbers,
  certified by a reproducible `flint.arb` algorithm at 256-bit
  precision, currently un-formalised in Lean only because mathlib
  lacks a Bessel interval-arithmetic library.
- **Analytic admissibility bundle** -- `ExtremiserPrimitives f`. Not
  an axiom but a *hypothesis* of the headline; the analogue of MV
  invoking "by Lemma 3.1 (Martin--O'Bryant)".

The distinction between *conjectural* and
*verifiable-by-computation* axioms matters because the natural
critic's question -- "are you assuming something unprovable?" -- has
a clean answer here: **no**. Both numerical axioms are provable; they
are simply not yet formalised in Lean for engineering reasons (no
Bessel interval-arithmetic library in mathlib, an absence shared by
every published computer-assisted real-number proof of the past two
decades). They are not RH-style conjectures, and they are not
hypotheses about the universe of mathematics; they are statements
about specific integers that any sufficient implementation of
interval arithmetic + the Bessel power series can decide.

**Why this is publication-valid.** It exceeds MV's standard: ~8650
lines of formal Lean for what MV proved on ~5 pages of text, with the
numerical inputs strictly more reliable (rigorous interval bounds vs.
heuristic floating point). It uses the same axiom structure published
computer-assisted proofs use (Flyspeck cited Kepler's interval
arithmetic; the polynomial-method capset proof cited specific Lagrange
polynomial bounds; the PFR formalisation cited numerical
Plünnecke--Ruzsa constants). The conceptual content of the proof is
in the Lean theorem, **not** in the axioms: the
verifiable-by-computation axioms encode only "evaluate this specific
integral and compare it to this specific rational".

**What replacing the verifiable-by-computation axioms would require.**
A rigorous Lean interval arithmetic library + verified quadrature +
numerical Bessel + Taylor branch-and-bound, totalling ~6000--10000
lines. None of this exists in mathlib. It is a separate multi-year
subproject (analogous to the decade-long Flyspeck effort for Kepler),
and would not change the mathematical claim.

**Honesty caveats.**

- The headline is conditional on the analytic-admissibility bundle
  (`ExtremiserPrimitives f`), not unconditional. Closing this bridge
  for arbitrary admissible $f$ requires connecting mathlib's
  $L^2$-Plancherel API to the concrete period-$u$ Parseval splits
  used in `Sidon.MVLemmas`. The infrastructure for this bridge is
  spread across `Sidon.FourierAux` and `Sidon.TorusParseval`; the
  remaining piece is the $L^1 \cap L^2$ periodisation step for $f*f$
  and $f \circ f$ on the torus.
- The two verifiable-by-computation axioms depend on trusting the
  `flint.arb` library. `flint.arb` is peer-reviewed (Johansson 2017,
  IEEE TC) and used widely for rigorous computational mathematics, but
  it is not itself Lean-verified.
- A Mathematica computation underlying MV is *strictly less* rigorous
  than these `flint.arb`-discharged axioms (Mathematica uses heuristic
  precision tracking; `flint.arb` uses guaranteed interval arithmetic
  with proven inclusion).

## Build and inspection

For build instructions, the expected `lake build` output, and the
`#print axioms` invocation that prints the dependency list, see
[`reproducibility.md`](reproducibility.md).
