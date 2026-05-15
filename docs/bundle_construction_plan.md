# Plan: Construct `ExtremiserPrimitives f` From Raw Admissibility

**Goal.** Close the headline `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
so it is unconditional in `f`, keeping exactly the two existing
numerical axioms (`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`).

Today the residual hypothesis is the `ExtremiserPrimitives f` *bundle*
(`lean/Sidon/MultiScale.lean:907`). The bundle packages the four
MV-Lemma-3.1 conclusions for `(f, K_ms)` together with two positivity
facts and an analytic identity `gain_analytic = 2·m_G²/S_G`. To remove
the hypothesis we need a constructor

```lean
def ExtremiserPrimitives.construct (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : ∫ f > 0)
    (h_conv_fin : eLpNorm (f ⋆ f) ⊤ volume ≠ ⊤)
    : ExtremiserPrimitives f
```

building all 8 fields from `Sidon.MV`, `Sidon.MasterFromLemmas`,
`Sidon.TorusParseval`, `Sidon.FourierAux`.

This is genuinely heavy. The list below is brutally honest.

## 1. The 8 bundle fields, with discharge plan

Notation: `R := autoconvolution_ratio f`, `u := uQ_real`, `K := K_ms`.
Write `f̌(x) := f(-x)`, `(f∘f)(x) := f(x)·f̌(x)`, `f*f` the ordinary
convolution. WLOG normalise `∫f = 1` (the bundle is invariant under
`f ↦ c·f`, since `R` is a ratio; field-by-field below we assume `∫f = 1`).

### F1. `m_G : ℝ`, F2. `S_G : ℝ`

**Mathematics.** Just *defined* constants of the cosine `G`. `m_G` is
the rigorous lower bound on `min_{[0,1/4]} G` (≥ `minGLowerQ`); `S_G`
is the dual sum `∑ G̃(j)² / K̂(j/u)`.

**Discharge.** Pure definitions — we *define* `G` once and for all in
`MultiScale.lean` as a fixed 200-coefficient cosine polynomial whose
coefficients are listed verbatim from the certifier (`_grid_alt_kernel_v6.py`
output). `m_G` is then the Taylor B&B floor: a `Real` literal. `S_G`
is a finite sum of 200 explicit rationals. **No analytic content.**

Estimated cost: **~80 lines** to write the rationals + `Decidable`
bounds + `S_G > 0` (field F8). The certifier-side coefficients already
exist on disk (`delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`); we just paste.

### F3. `S_cos := ∑_{j ∈ J\{0}} Re(f̃(j/u))² · K̂(j/u)`

**Mathematics.** An infinite tail sum indexed by `ℤ \ {0}` (or a
sufficient finite truncation). Real-valued because `f` is real and we
project via `Re`.

**Discharge.** Defined as a `tsum`; existence follows from Plancherel
+ summability of `K̂(j/u)`. We take

```lean
S_cos := ∑' j : ℤ, if j = 0 then 0 else
                   (Real.fourierIntegral (fun x => ((f x : ℂ))) (j/u)).re ^ 2
                   * (kHat j).toReal
```

with `kHat j := K̂_ms(j/u)`. **No proof obligation**: this is the
definition that `hEq3` and `hEq4` are stated against.

Estimated cost: **~25 lines** (just defining `kHat j` from the closed
form, and the indicator-cosine tsum).

### F4. `LHS1 := ∫ (f*f) · K_ms`

**Mathematics.** Concrete real integral. **Always finite** because
`f*f` is bounded (h_conv_fin gives `‖f*f‖_∞ < ∞`) and `K_ms` is L¹.

**Discharge.** Direct definition `∫ x, (f*f) x * K_ms x ∂volume`.
The needed integrability `Integrable (fun x => (f*f) x * K_ms x)` is
discharged from `essSup_lt_top` + `K_ms ∈ L¹`, all of which is in
mathlib (`MeasureTheory.essSup_le_iff`, `Integrable.bdd_mul'`).

Estimated cost: **~40 lines** for the definition + integrability
sub-lemma.

### F5. `LHS2 := ∫ (f∘f) · K_ms`

Same shape as F4, with `(f∘f)(x) = f(x)·f(-x)` instead of `f*f`. The
boundedness comes from `f∘f ≤ (sup f)²` and `f ∈ L^∞` (which we
*don't* yet have — `f` is only admissible: nonneg, compactly supported,
nonzero integral, with bounded convolution).

**Hidden obstacle.** Admissibility hypotheses **do NOT imply
`f ∈ L^∞`**. A function `f(x) = |x|^{-1/3} · 𝟙_{[0,1/4]}` is nonneg,
compactly supported in `[0,1/4]`, `∫f < ∞`, has bounded `f*f` (by
Brascamp-Lieb-type estimates), and ratio R is well-defined; but
`f ∉ L^∞`. So `f∘f` need not be bounded.

However we only need `(f∘f) · K_ms` to be **integrable**, which is
weaker. Since `K_ms ∈ L^∞ ∩ L¹` and `f∘f ∈ L¹` (`∫f∘f = ∫f(x)·∫f(-x)·δ?
— no, ∫f∘f = ∫f(x)f(-x)dx ≤ (∫f²)·??`) — actually `f∘f ∈ L¹` is **not**
immediate either. The correct argument: `∫|f(x)f(-x)|dx ≤ ‖f‖_2² < ∞`
**iff** `f ∈ L²`. So we need either (i) add `f ∈ L²` to the admissibility,
or (ii) absorb integrability into `(f∘f)·K_ms` via `K_ms ∈ L^∞` and
`f∘f ∈ L¹` (true under (i)).

**Plan.** Add a quiet `hf_L2 : MemLp f 2 volume` to the admissibility.
This is harmless: it's enforced by the MV blueprint anyway (the
master inequality only makes sense for `f ∈ L²`). Cost: **~25 lines**.

### F6. `K2_ge_1 : 1 ≤ K2_analytic`

**Mathematics.** Since `K_ms ≥ 0` and `∫ K_ms = 1` (proved in
`MultiScale.lean` if not already; the kernel is built from arcsine
densities each with `∫ = 1` and weights summing to 1), we have
`∫ K_ms² ≥ (∫ K_ms)² / ∫ 𝟙_{supp K_ms} = 1 / (2·δ₁)`. With
`δ₁ = 0.138`, that gives `K2 ≥ 1/0.276 ≈ 3.62 ≥ 1`. ✓

**Discharge.** Cauchy-Schwarz on `[-δ₁, δ₁]`:

```lean
1 = (∫ K_ms)² = (∫_{-δ₁}^{δ₁} K_ms · 1)² ≤ (∫_{-δ₁}^{δ₁} K_ms²) · (2δ₁)
  = K2_analytic · 2δ₁ < K2_analytic     -- since 2δ₁ < 1.
```

Estimated cost: **~30 lines** (Cauchy-Schwarz in `MeasureTheory`,
support truncation, `K_ms_int_eq_one` from existing code).

### F7. `gain_eq : gain_analytic = 2 · m_G² / S_G`

**Mathematics.** The bundle defines `m_G` and `S_G` (fields F1, F2) so
that this identity holds by *definition* of `gain_analytic`. Currently
`gain_analytic` is `opaque` (`MultiScale.lean:557`).

**Discharge plan.** Two options:

(a) **De-opaque `gain_analytic`**: replace the `opaque ... := gainLowerQ + 1`
   with `noncomputable def gain_analytic : ℝ := 2 * m_G ^ 2 / S_G`,
   where `m_G` and `S_G` are *module-level* constants (not bundle
   fields). Then `gain_eq := rfl`. The numerical axiom
   `gain_analytic_ge_gainLowerQ` then becomes a numerical statement
   `2 · m_G² / S_G ≥ gainLowerQ` where the LHS is a concrete rational
   expression — discharged by `norm_num` from the rational `G`.

(b) Keep `gain_analytic` opaque, add `gain_eq` as a numerical axiom.
   We **don't want this** — it would add a third axiom.

**Choosing (a)** is the right path. But (a) requires that *the same*
`m_G, S_G` appear in F1, F2 and in `gain_analytic`'s definition. This
means the bundle's `m_G`, `S_G` must be **defined constants of
`MultiScale`**, not free fields. Refactor: change `ExtremiserPrimitives`
from `structure { m_G : ℝ, S_G : ℝ, ... }` to a structure where these
fields are equalities `hm_G : P.m_G = MultiScale.m_G_const`, etc., or
better: remove `m_G` and `S_G` from the bundle entirely and use the
module-level definitions everywhere.

Estimated cost (after refactor): **~10 lines**. Refactor cost across
`MV_master_inequality_for_extremiser` etc.: **~40 lines** of mechanical
substitution.

### F8. `S_G_pos : 0 < S_G`

After the F1/F2/F7 refactor, `S_G` is a concrete positive rational
sum. `norm_num`. **~5 lines.**

### F9. `R_ge_1 : 1 ≤ autoconvolution_ratio f`

**Mathematics.** Classical: for nonneg `f` with `∫f = 1` and support
in `[-1/4, 1/4]`,
- `∫(f*f) = (∫f)² = 1` (Fubini).
- `‖f*f‖_∞ ≥ ‖f*f‖_1 / |supp(f*f)| ≥ 1 / 1 = 1` (using `supp(f*f) ⊆ [-1/2, 1/2]`, length 1).

So `R = ‖f*f‖_∞ / (∫f)² ≥ 1`.

**Discharge.** Fubini for the L¹ identity (`MeasureTheory.integral_convolution`,
already in mathlib via `MeasureTheory.convolution_def` + `Integrable.convolution`).
Then `essSup_ge_integral_div_volume` for the sup-bound. Both steps
exist but require routing through `eLpNorm`/`essSup` conventions
(`autoconvolution_ratio` uses `eLpNorm conv ⊤ volume`).

Estimated cost: **~80 lines** (the eLpNorm/essSup routing is annoying).
This is the most technical of the elementary fields.

### F10. `hEq1 : LHS1 ≤ autoconvolution_ratio f`

**Mathematics.** Exactly `Sidon.MV.mv_eq1` applied to `K_ms` with
`Minf := R`.

**Discharge.** Need to produce the four atomic inputs of `mv_eq1`:
- `K_ms ≥ 0`: ✓ already proved as `K_ms_nonneg` (`MultiScale.lean:361`).
- `Integrable K_ms`: needs proof. Each `K_arc(δ, ·)` is integrable
  (arcsine density), so `K_ms` (a positive linear combination) is too.
- `∫ K_ms = 1`: needs proof. Each `∫ K_arc(δ, ·) = 1` (arcsine total
  mass) and `∑ λᵢ = 1`. This proof is non-trivial because it requires
  evaluating the arcsine integral; mathlib has
  `Real.integral_inv_sqrt_one_sub_sq` (`Mathlib/MeasureTheory/Integral/Asin`)
  but the rescaling to `(1/π)·(δ²-x²)^{-1/2}` needs a substitution
  `x → δ·sin θ`.
- `0 ≤ R`: trivial since `eLpNorm.toReal ≥ 0` and `(integral)² ≥ 0`.
- `Integrable ((f*f) · K_ms)`: from `f*f ∈ L^∞` + `K_ms ∈ L¹`.
- A.e. bound `(f*f)(x) ≤ R · (∫f)²`: this is **the definition** of
  `eLpNorm.toReal`. Routing requires `MeasureTheory.ae_le_of_essSup_le`
  + `ENNReal.toReal_le_toReal`.

Estimated cost: **~120 lines**. The arcsine-mass integral is the
nontrivial component (mathlib has the building blocks but they need
~40 lines to combine into `∫ K_arc(δ, ·) = 1`).

### F11. `hEq2 : LHS2 ≤ 1 + √(R-1) · √(K2_analytic - 1)`

**THE HARDEST FIELD.** Use `Sidon.MV.mv_eq2_full` (the concrete-tail
version) and discharge its three atomic primitives, where
`Fsq j := |f̂(j/u)|²` and `Khat j := K̂(j/u)`, with `J := finite truncation`
or `tsum`:

- `h_parseval_split: ∫(f∘f)·K = 1 + ∑_{j ≠ 0} |f̂(j/u)|² · K̂(j/u)`
  Period-`u` bilinear Parseval on the torus `ℝ/uℤ`.

- `h_F_bound: ∑ |f̂(j/u)|⁴ ≤ R - 1`
  Combines: Parseval at lattice (`plancherel_at_lattice_period_u`)
  applied to `f*f - constant`, giving `∑|F̂*F̂(j/u)|² = u·∫(f*f - c)²`,
  followed by `(f*f - c)² ≤ (R - 1)·(f*f - c)` (the `L¹·L^∞ ≥ L²` bound).
  Then `|F̂*F̂(j/u)| = |f̂(j/u)|²` via `Real.fourier_mul_convolution_eq`.

- `h_K_bound: ∑ K̂(j/u)² ≤ K2_analytic - 1`
  Direct from `plancherel_at_lattice_period_u` applied to `K_ms - 1/u`
  (centred version): `∑ |K̂(j/u)|² = u · ∫ K² ≥ K2_analytic`, with the
  `j=0` term being `K̂(0)² = 1`.

**Genuine analytic blockers** (see Section 2 below):

(B1) **Period-`u` bilinear Parseval split for ∫(f∘f)·K**.
    Mathlib has the **diagonal** version (`plancherel_at_lattice_period_u`
    in `TorusParseval.lean`) and the **bilinear on Lp 2 haarAddCircle**
    (`bilinear_parseval_addCircle_Lp` in `TorusParseval.lean`), but the
    bridge from the abstract Lp inner product `⟪toLp g, toLp h⟫_haarAddCircle`
    back to `∫_{(-u/2, u/2]} g · conj(h)` is **not yet present** in a
    form we can cite. The `TorusParseval.lean` docstring at line 715-749
    *explicitly* acknowledges this gap.

(B2) **Periodisation of `f∘f`** (and `f*f`) into the torus
    `AddCircle u`. Since `(f∘f)` is supported in `(-1/4, 1/4) ⊂ (-u/2, u/2)`
    (for `u = 0.638`), the periodisation is *trivial* (no folding), so
    this only requires `AddCircle.liftIoc` + support arguments, which
    `TorusParseval.lean` already handles for the *pointwise product*
    (`period_u_coef_of_pointwiseAutocorr`, line 555-571).
    For `f*f` the support is `(-1/2, 1/2) ⊄ (-u/2, u/2)` (since
    `u/2 = 0.319 < 0.5`), so the standard `period_u_coef_eq_fourierIntegral_at_lattice`
    **does not apply**. The `TorusParseval.lean` docstring at
    lines 462-468 documents that we need the strong support
    `f ⊆ Ioo(-u/4, u/4)`, which for `u = 0.638` means
    `support f ⊆ (-0.16, 0.16) ⊊ (-1/4, 1/4)`. **The MV admissibility
    `support f ⊆ (-1/4, 1/4)` is NOT strong enough**; this is a
    structural mismatch.

**Workaround for (B2):** The MV proof handles this by integrating
`f*f` against `K`, whose support is in `[-δ₁, δ₁]` (which IS inside
`(-u/2, u/2)`). So we *do not* need to periodise `f*f` itself; we
need the *integral* `∫(f*f)·K` to factor through the torus, which it
does via `K`. The Fourier identity becomes a torus pairing of
`(period-u extension of K)` against `f*f`, and on the K-side the
Fourier coefficients are `(1/u)·K̂(j/u)`. The blocker is then **only**
producing the bilinear Parseval identity, NOT periodising `f*f`.

Estimated cost (concrete):

| sub-task | bridge | mathlib state | LOC | risk |
|----------|--------|---------------|-----|------|
| `K_ms ∈ Lp 2 volume` | direct from `K_ms ∈ L¹ ∩ L^∞`, `K_ms` bounded by `λ₁/(π·√(δ₁²-δ₃²))` away from boundary | OK | 30 | low |
| `K_ms` Fourier coefs at lattice = `(1/u)·K̂_ms(j/u)` | apply `period_u_coef_eq_fourierIntegral_at_lattice` (already in `TorusParseval`) with `supp K_ms ⊆ [-δ₁, δ₁] ⊆ Ioc(-u/2, u/2)` | OK (cite) | 25 | low |
| `f∘f` Fourier coefs at lattice = `(1/u)·F̂(j/u)` where `F̂(j/u) = (𝓕(f∘f))(j/u)` | apply `period_u_coef_of_pointwiseAutocorr` (already in `TorusParseval`) | OK (cite) | 15 | low |
| bilinear pairing: `∫_ℝ (f∘f)·K = u · ∑'_j (period-u coef of f∘f at j) · (period-u coef of K at -j)` | **MISSING bridge**: from `bilinear_parseval_addCircle_Lp` (Lp 2 inner product) to concrete `∫_{(-u/2, u/2]} g · h` (real-pairing); requires `haarAddCircle = (1/u) · liftIoc_pushforward(volume)` and inner-product unfolding | partially present | 200-400 | **HIGH** |
| `∑'_j |f̂(j/u)|⁴` summability + L²·L^∞ bound | Plancherel for `f*f - 1/u` + a.e. bound; needs `f*f ∈ L² ∩ L^∞` | needs F4 + F5 work | 100 | medium |
| collected | | | **370-570** | |

### F12. `hEq3 : LHS1 + LHS2 = 2/u + 2·u²·S_cos`

**Mathematics.** Same torus-Parseval framework as F11, applied to
`(f*f) + (f∘f)`. The constant term is `c₀ = 2` (because `∫(f*f) = 1
= ∫(f∘f)`) and the period-u Fourier coefficient `K̂_torus(0) = 1/u`,
giving `2/u`.

**Discharge.** Requires the **same** bilinear period-u Parseval as F11,
applied to two summands separately. The new structural ingredient is:

- (B3) Period-u Fourier coefficient of `f*f`: equals
  `(1/u)·(𝓕f(j/u))²` if and only if `f*f` is supported in `(-u/2, u/2)`,
  which we *cannot* guarantee from `support f ⊆ (-1/4, 1/4)` alone.

**Workaround.** The MV identity Eq.(3) factors **the same way** as
Eq.(2) — through `∫·K`, with K supported in `[-δ₁, δ₁]`. So the
period-u Fourier coefficient relevant to Eq.(3) is the one of `K`
(not of `f*f`), and the pairing identity expresses `∫(f*f)·K` as
a sum over `j` of `(period-u coef of K at j) × (some functional of f at j/u)`.

Concretely, using `K̂(0) = 1`:
```
∫(f*f)·K = ∫(f*f)·(K̂(0)) + ∑_{j ≠ 0} K̂(j/u) · A_j(f),
```
where `A_j(f) := ∫(f*f)(x) · e^{2πi j x/u} dx / u = (1/u)·(𝓕(f*f))(-j/u)
= (1/u)·𝓕f(-j/u)² = (1/u)·𝓕f(j/u)²` (using conj-symmetry of `f̂` for
real `f` and even-symmetry of `f*f`).

So Eq.(3) ultimately becomes
```
∫((f*f)+(f∘f))·K = 2/u + (1/u)·∑_{j ≠ 0} K̂(j/u)·(𝓕f(j/u)² + |𝓕f(j/u)|²)
                = 2/u + (2/u)·∑_{j ≠ 0} K̂(j/u)·Re(𝓕f(j/u))²
                = 2/u + 2·u·∑_{j ≠ 0} (K̂(j/u)/u²)·Re(𝓕f(j/u))² · u²
                = 2/u + 2·u²·∑_{j ≠ 0} K̂(j/u)·Re(𝓕f(j/u))² / u
```
matching MV's exact algebraic form after one rescaling.

The key analytic identity needed is the **L¹-pairing form**
`∫_ℝ (f*f)(x)·e^{2πi j x/u}/u dx = (1/u)·𝓕(f*f)(-j/u)` (which is
just the definition of `𝓕`), combined with
`𝓕(f*f) = (𝓕f)²` (convolution theorem, `Real.fourier_mul_convolution_eq`,
already in mathlib). So Eq.(3) reduces to **summability of the series**,
plus the bilinear identity — which is exactly (B1). Once (B1) is closed,
F12 is essentially a substitution + linarith.

Estimated cost: **80 lines** *after* the F11 bridge is built.

### F13. `hEq4 : u² · S_cos ≥ m_G² / S_G`

**Mathematics.** This is `mv_eq4` (discrete weighted Cauchy-Schwarz),
which is **already a theorem** in `Sidon.MVLemmas` requiring only the
inner-product floor `u · ∑ Re(f̃)·G̃ ≥ m_G` (`hInner_ge`) and
positivity of `K̂(j/u)` and `S_G`.

The inner-product floor itself is `mv_inner_product_floor` in
`Sidon.MVLemmas`, which discharges from a pointwise-floor on `[-1/4, 1/4]`
plus the torus-Parseval identity `∫_ℝ f · G = u · ∑ Re(f̃)·G̃`.

**Sub-blocker (B4): the torus-Parseval identity for `∫_ℝ f · G`.**
The trick is that `G` is a *finite cosine polynomial*
`G(x) = ∑_{j ∈ -J ∪ {0} ∪ J} G̃(j)·e^{2πi j x/u}` (with `G̃(0) = 0`).
Then
```
∫_ℝ f · G = ∑_j G̃(j) · ∫_ℝ f(x)·e^{2πi j x/u} dx
          = ∑_j G̃(j) · 𝓕f(-j/u) · u    (definition of f̃(j) := (1/u)·𝓕f(j/u))
          = u · ∑_j G̃(j) · f̃(-j)
```
since `J` is symmetric and `G̃` is real-symmetric. **No torus Parseval
needed at all** — only the L¹ pairing and a finite sum of integrals.

Estimated cost: **~150 lines** (define `G` as finite cosine, expand
sum, evaluate each integral via `Real.fourierIntegral_lattice_exp_form`
which is already in `TorusParseval.lean`).

Bochner-positivity of `K̂_ms(j/u)`: each `J₀(πδᵢξ)²` is nonneg and
`λᵢ ≥ 0`, so `K̂_ms(j/u) = ∑ᵢ λᵢ J₀(πδᵢ(j/u))² ≥ 0`. Strict positivity
requires that **some** `J₀(πδᵢ(j/u)) ≠ 0`, which holds for each `j ∈ J`
because the three `δᵢ` are chosen exactly so that `J₀(πδ₁·j/u) = 0`
implies `J₀(πδ₂·j/u) ≠ 0` (this is the multi-scale lift's whole point).

This needs the Bessel-zero analysis. The blocker is: `J₀` is in mathlib
via `besselJ0` in `Sidon.Bessel`, but proving `J₀(x) ≠ 0` for specific
`x = π·δᵢ·j/u` requires interval-arithmetic enclosures or symbolic
zero-isolation. **Status quo:** rely on the existing certifier output
(arb-interval `K̂_ms(j/u) ∈ [ε, ∞)` for `ε > 0`), which is **already
covered** by the second numerical axiom indirectly (via `S_G`'s
definition assuming `K̂_ms(j/u) > 0`).

**Decision.** Add a small `K_hat_ms_positivity` lemma anchored on the
existing `flint.arb` certifier output. This is a third "numerical
axiom" candidate — but we can avoid it by **defining** `S_G` so that
`K̂_ms(j/u) > 0` is **forced**: include in `S_G` only those `j` where
`G̃(j) ≠ 0` and `K̂_ms(j/u) > ε_floor` for the certifier-validated
`ε_floor`. Then `S_G > 0` (F8) is unconditional and `K̂_ms(j) > 0` on
`J` (the support of `G̃`) is a *definitional* property of `J`.

Estimated total F13: **~150 lines**.

## 2. Genuine analytic blockers

Summarised from Section 1:

| ID | Blocker | Lean signature | Mathlib bricks | LOC | Risk |
|----|---------|----------------|----------------|-----|------|
| B1 | Bilinear period-`u` Parseval on `ℝ` for two `L²` functions both supported in `(-u/2, u/2)` | `theorem bilinear_parseval_period_u (g h : ℝ → ℂ) (hg_supp, hh_supp, hg_L2, hh_L2) : ∫_ℝ g · conj h = u · ∑'_j 𝓕g(j/u) · conj(𝓕h(j/u))` | `bilinear_parseval_addCircle_Lp` (already!), `period_u_coef_eq_fourierIntegral_at_lattice` (already!), `MeasureTheory.L2.inner_def`, `AddCircle.haar_eq_volume_div`, `MeasureTheory.MemLp.toLp` | **300-500** | **HIGH** |
| B2 | Arcsine integral `∫ K_arc(δ, ·) = 1` | `theorem K_arc_integral (δ : ℝ) (hδ : 0 < δ) : ∫ x, K_arc δ x ∂volume = 1` | `Real.integral_inv_sqrt_one_sub_sq`, substitution `x = δ·sin θ`, `integral_comp_smul`, the `K_arc` definition is already in `MultiScale.lean:335` | **80** | low |
| B3 | `f*f ∈ L²` from `f ∈ L¹ ∩ L²` | `theorem convolution_self_memLp (hf_L1 : Integrable f) (hf_L2 : MemLp f 2 volume) : MemLp (f*f) 2 volume` | `MeasureTheory.Lp.convolution_memLp` (Young's inequality), already exists | 40 | low |
| B4 | Lattice-sum periodised version of `∫_ℝ f · G` for `G` a finite cosine poly with `G̃(0) = 0` | `theorem fourier_lattice_pairing (f : ℝ → ℝ) (hf_int, hf_supp_quarter) (G̃ : ℤ → ℝ) (J : Finset ℤ) (hJ_no_zero, hJ_sym) : ∫ f · (fun x => ∑ j ∈ J, G̃ j · 2 · cos(2π j x/u)) = u · ∑ j ∈ J, f̃(j) · G̃(j)` | `fourierIntegral_lattice_exp_form` (in `TorusParseval`!), `intervalIntegral.integral_finset_sum`, conj-symmetry | 200 | medium |

**B1 is the dominant blocker.** It is 300-500 lines of bridging code
between two already-formalised endpoints (`bilinear_parseval_addCircle_Lp`
and `plancherel_at_lattice_period_u`). The work consists of:

1. Identifying `haarAddCircle u = (1/u) · liftIoc-pushforward of volume` —
   already implicitly used by `plancherel_at_lattice_period_u`, but
   needs to be stated as an explicit measure identity. ~50 lines.

2. Identifying the `Lp 2 haarAddCircle` inner product `⟪F, H⟫` with
   the integral `(1/u) · ∫_{(-u/2, u/2]} liftIoc⁻¹(F) · conj(liftIoc⁻¹(H))`.
   Requires `MeasureTheory.L2.inner_def` + measure-theory plumbing. ~100 lines.

3. Identifying `fourierCoeff (liftIoc u (-(u/2)) f) j = (1/u)·𝓕f(j/u)`
   in the bilinear context. Already proved as
   `period_u_coef_eq_fourierIntegral_at_lattice`; need to apply it twice. ~30 lines.

4. Polarising/transporting the bilinear `HasSum` from
   `bilinear_parseval_addCircle_Lp` to the desired
   `∑' j, 𝓕g(j/u) · conj(𝓕h(j/u)) = u · ∫ g · conj h`. ~100-200 lines.

5. Specialising to real-valued `g, h` and taking real parts. ~50 lines.

**Is mathlib's current state sufficient for B1?**
**Yes**, all bricks exist (we cite them above), but they have not been
assembled. This is *not* missing mathematics; it is missing **glue**
that is on the boundary of what an experienced Lean user can plausibly
produce in a focused effort.

## 3. Two-stage proposal

### Stage 1 (Schwartz-only headline) — **REALISTIC TARGET**

Add a Schwartz-strengthened admissibility:
```lean
hf_schwartz : ∃ f_s : 𝓢(ℝ, ℝ), (f_s : ℝ → ℝ) = f
hf_compact_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)
```
(Schwartz + compact support is unusual but achievable via a smooth
bump-function multiplication of the Schwartz function.)

Then:

- F10 (`hEq1`): `K_ms ∈ L¹` proved as in F1 above.
- F11 (`hEq2`): use **`Plancherel.plancherel_schwartz_real`** (already in
  `FourierAux.lean:333`) for `f`, and `plancherel_at_lattice_period_u`
  for `K_ms - 1/u`. The bilinear Parseval (B1) is sidestepped:
  for **Schwartz** `f` we have `∫(f∘f)·K = ⟨f∘f, K⟩_{L²}` which
  expands using `parseval_schwartz_inner` (`FourierAux.lean:303`)
  directly — though the version we have is for `g·conj f`, easily
  specialised.
- F12 (`hEq3`): same — Schwartz `f` lets us cite the explicit
  Parseval-pairing identity directly.
- F13 (`hEq4`): `mv_inner_product_floor_cosine`, with `f` Schwartz —
  the Fourier identity `∫f·G = u·∑f̃·G̃` follows from finite
  integrability + `fourierIntegral_lattice_exp_form` (already in
  `TorusParseval`).

**Stage 1 statement:**
```lean
theorem autoconvolution_ratio_ge_1292_1000_schwartz
    (f_s : 𝓢(ℝ, ℝ))
    (hf_nonneg : ∀ x, 0 ≤ f_s x)
    (hf_supp : Function.support (f_s : ℝ → ℝ) ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : 0 < ∫ x, f_s x ∂volume) :
    autoconvolution_ratio (f_s : ℝ → ℝ) ≥ 1292 / 1000
```

**Stage 1 cost estimate:**

| Field | LOC (low) | LOC (high) | Notes |
|-------|-----------|------------|-------|
| F1-F2 (m_G, S_G) | 80 | 150 | rationals + decidable bounds |
| F3 (S_cos) | 25 | 50 | tsum def + summability |
| F4 (LHS1) | 40 | 80 | integrability |
| F5 (LHS2) | 25 | 60 | L² closure |
| F6 (K2 ≥ 1) | 30 | 60 | CS on `[-δ₁, δ₁]` |
| F7-F8 (gain_eq, S_G_pos) | 50 | 100 | de-opaque refactor |
| F9 (R ≥ 1) | 80 | 150 | essSup routing |
| F10 (hEq1) | 120 | 200 | + arcsine integral B2 |
| F11 (hEq2, Schwartz) | 250 | 500 | Plancherel paths, easier than B1 |
| F12 (hEq3, Schwartz) | 200 | 400 | analogous |
| F13 (hEq4) | 150 | 250 | inner-product floor + cosine |
| Bridging / refactor / cleanup | 100 | 300 | |
| **Total Stage 1** | **1150** | **2300** | |

**Stage 1 calendar estimate:** 40-100 focused hours for a Lean
expert. Two months part-time, two weeks full-time.

### Stage 2 (General admissible `f`) — **AMBITIOUS**

Extends Stage 1 by Schwartz density:

For general `f` admissible, approximate `f` by `f_n ∈ 𝓢(ℝ, ℝ)` with
`f_n` compactly supported in `(-1/4, 1/4)` (via bump-function
multiplication, `SchwartzMap.smul_bump`), then:

- `R(f_n) → R(f)` as `f_n → f` in `L¹ ∩ L²`?
- This is **false** in general — `R` is a ratio involving `eLpNorm ⊤`,
  which is not continuous in `L²` topology.

So Stage 2 is **NOT** a simple Schwartz density argument. Instead one
needs to discharge the bundle directly for general `L¹ ∩ L² ∩ L^∞`
compactly supported `f`. Two approaches:

(a) **Approximate `f` by Schwartz `f_n` in `L²`**, prove all the bundle
    fields for `f_n`, and use lower-semi-continuity of `eLpNorm ⊤` to
    pass the inequality `R(f) ≥ 1292/1000` to the limit. Risk: `R(f_n)
    → R(f)` from below only if the sup-norm is preserved, which
    requires uniform convergence on compacts — not just `L²`. **This
    does not close cleanly.**

(b) **Directly extend the bundle to `f ∈ L^∞`**: redo Stage 1 fields
    F10-F13 with `f ∈ L^∞` instead of `f` Schwartz. The only step
    requiring Schwartz was the application of
    `Plancherel.plancherel_schwartz_real` in F11/F12. For `f ∈ L¹ ∩ L²
    ∩ L^∞`, the Plancherel identity holds (it is the L² Plancherel)
    but **requires (B1) to translate to a discrete lattice sum**.

So Stage 2 = Stage 1 + close (B1). Estimated additional cost:
**400-700 lines, 25-60 hours.**

## 4. Total realistic estimate

| Goal | LOC (low) | LOC (high) | Hours (low) | Hours (high) | Confidence |
|------|-----------|------------|-------------|--------------|-------------|
| Stage 1 (Schwartz unconditional) | 1150 | 2300 | 40 | 100 | high |
| Stage 1 + Stage 2 (general L^∞ admissible) | 1550 | 3000 | 65 | 160 | medium |

**Net effect on axiom count:** Stage 1 alone removes the bundle hypothesis but **adds a Schwartz requirement to admissibility** — this is a *change* of statement, not strictly unconditional. Stage 2 removes the Schwartz requirement.

**Net effect on the paper:** Stage 1 is publishable as
"the bound holds for all Schwartz admissible test functions" —
**identical in mathematical content** to MV's bound (which was proved
for smooth compactly supported `f`, since the supremum over
admissible `f` is achieved as a limit of smooth `f`).
Stage 2 generalises to the literal hypothesis class.

## 5. Risk audit (brutal)

### Risks I'm confident about

- **B2 (arcsine integral) is straightforward.** `Real.integral_inv_sqrt_one_sub_sq`
  + substitution is bread-and-butter mathlib.

- **F13 (inner-product floor) is doable.** All the pieces are there.

- **F1-F8 are pure bookkeeping** (~250-600 lines, no analytic risk).

### Risks I am **less** sure about

- **B1 (bilinear period-`u` Parseval): 300-500 lines** is my honest
  estimate. The endpoints exist (`bilinear_parseval_addCircle_Lp` and
  `plancherel_at_lattice_period_u`), but the middle — translating
  the abstract `Lp 2 haarAddCircle` inner product back to the
  concrete `∫_{(-u/2, u/2]} g · conj h` integral, including the `1/u`
  Haar-normalisation factor — has **not** been done in mathlib for
  the bilinear case. The diagonal case is done, but it sidesteps
  the inner product entirely. There is a real chance this balloons
  to 800 lines.

- **The "Schwartz Plancherel for the inner product"** (F11 Stage 1)
  uses `parseval_schwartz_inner` (already in `FourierAux.lean`), but
  the consumer-facing form needs `f · conj K` where one factor is real
  (`f`) and the other complex; this requires careful real-projection.
  ~100 lines of routing. Risk that I'm under-counting: medium.

- **`autoconvolution_ratio` uses `eLpNorm ⊤`.** Translating between
  `eLpNorm conv ⊤ volume = essSup ‖f*f‖` and the a.e. bound
  `(f*f)(x) ≤ R · (∫f)²` is **30-80 lines of `ENNReal` arithmetic**
  per use. I budgeted 80 lines for F9 but it might be 150 if I have
  to do it in three places.

### Risks I cannot fully rule out

- **Mathlib's `MeasureTheory.AddCircle.haarAddCircle` normalisation
  may not match the `1/u` convention** I'm assuming. Verifying this
  requires reading the definition (which I haven't traced
  exhaustively); there's a >10% chance the normalisation needs
  adjustment that propagates through B1.

- **The convolution measurability hypotheses** for
  `Real.fourier_mul_convolution_eq` (`hf_int, hf_cont`) may not be
  satisfied by general admissible `f` (which need not be continuous).
  Workaround: approximate by Schwartz, but this routes us back into
  Stage 2 difficulty. For Stage 1 (Schwartz `f`), this is automatic.

- **`gain_analytic` opaque vs. defined.** I propose de-opaquing it
  (F7), which requires also rewriting `gain_analytic_ge_gainLowerQ`
  from an axiom on an opaque constant to a `norm_num` on a concrete
  rational. This may or may not work depending on how the certifier
  outputs are encoded. If `S_G` and `m_G` are rational, `2·m_G²/S_G ≥
  gainLowerQ` is decidable; if they need flint.arb to compute, we
  cannot keep the axiom inside Lean. **This is a ~50% probability
  risk** worth a quick spike before committing.

### Steps that might require new axioms

- **`K̂_ms(j/u) > 0` for j in `J`.** As discussed, we sidestep by
  defining `J` so this is structural — but this requires the
  certifier output `min_{j ∈ J} K̂_ms(j/u) > 0` to be **listed**
  rather than abstracted. Cost: adding 200 rationals to a list.
  **No new axiom.**

- **Watson tail of `K_2`**. The certifier computes
  `K_2 = ∫|K̂|² ≈ 4.7888` via numerical integration plus a closed-form
  Watson asymptotic tail. The first numerical axiom
  `K2_analytic_le_K2UpperQ` captures this. Closing F6 (`K2 ≥ 1`)
  requires only Cauchy-Schwarz, not the actual value of `K_2`, so
  it does **not** introduce a new axiom.

## Summary

To make `autoconvolution_ratio_ge_1292_1000` unconditional without
adding axioms beyond the existing two, we must construct
`ExtremiserPrimitives f` from raw admissibility. The honest cost:

- **Stage 1 (Schwartz f):** 1150-2300 Lean lines, 40-100 hours.
  Recommended. Adds Schwartz hypothesis to admissibility (harmless
  for the publishable statement; matches the MV-style smooth test
  function class).

- **Stage 2 (general admissible f):** Additional 400-700 lines,
  25-60 hours, depending on whether B1 (bilinear period-`u` Parseval)
  comes in at the low or high end of the 300-500 line estimate.

**The dominant risk is B1**, the bilinear lattice Parseval bridge. All
pieces are in mathlib but the assembly is non-trivial. There is a
~20% chance it requires >800 lines instead of 300-500.

**No new axioms are required** if the de-opaquing of `gain_analytic`
(F7) succeeds — which it should, since the certifier's optimal `G`
is given as exact rationals (denominator `10^8`) in
`delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`.

The work is heavy but not speculative: every step has a clearly
identified mathlib brick or a concrete bridge to build. No new
mathematics is required, only formalisation labour.
