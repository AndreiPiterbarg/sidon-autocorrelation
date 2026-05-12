# Audit Report: Continuous Analysis Core (Cauchy-Schwarz, Asymmetry Bound, Essential Supremum)

> **OUTDATED (2026-04-07):** This audit was conducted against the old coarse-grid
> parameterization (compositions summing to m). The Lean definitions have been updated
> to the C&S fine grid (compositions summing to 4nm, heights = c_i/m). Re-audit needed.

**Files audited:**
1. `lean/Sidon/CauchySchwarz.lean` — Cauchy-Schwarz single-bin bound (Claims 4.5, 4.7, 4.8)
2. `lean/Sidon/AsymmetryBound.lean` — Asymmetry bound (Claim 2.1)
3. `lean/Sidon/EssSup.lean` — Essential supremum bounds

**Date:** 2026-03-24

**Verdict: ALL 30 DEFINITIONS/THEOREMS CORRECT. No errors, no suspicious claims.**

---

## File 1: CauchySchwarz.lean (264 lines, Claims 4.5, 4.7, 4.8)

### Definitions

| # | Definition | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `bin_interval` | 35–39 | **CORRECT** | Partitions `[-1/4, 1/4)` into `2n` half-open bins of width `δ = 1/(4n)` via `Set.Ico`. Standard. |
| 2 | `f_bin` | 42–43 | **CORRECT** | `Set.indicator (bin_interval n i) f`. Consistent with `bin_interval`. |

### Lemmas and Theorems

| # | Theorem | Lines | Verdict | Analysis |
|---|---|---|---|---|
| 3 | `f_bin_le_f` | 45–51 | **CORRECT** | For `f ≥ 0`: indicator is `f` on the set, `0` off it, so `≤ f` pointwise. Trivial. |
| 4 | `f_bin_nonneg` | 53–55 | **CORRECT** | `Set.indicator_nonneg` with `hf`. |
| 5 | `integral_f_bin` | 57–61 | **CORRECT** | Definitional unfolding: `f_bin f n i = indicator(Ico a b) f`, and `bin_masses f n i = ∫ indicator(Ico a b) f` (Defs.lean:68–73). These are identical. |
| 6 | `f_bin_integrable` | 63–69 | **CORRECT** | Uses `Integrable.mono'` with dominator `\|f\|`. AEStronglyMeasurability of `indicator(Ico)` follows from `measurableSet_Ico`. Pointwise bound `\|indicator_S f x\| ≤ \|f x\|` is clear. |
| 7 | `convolution_mono_pointwise` | 71–85 | **CORRECT** | For `0 ≤ g ≤ f`: when `g·g` integrand is integrable, uses `integral_mono` with `g(t)·g(x-t) ≤ f(t)·f(x-t)` (via `mul_le_mul`). When not integrable, `∫ g·g = 0` (Lean convention for undefined integrals) and `∫ f·f ≥ 0` by `integral_nonneg`. The second branch is technically vacuous (dominated by integrable `f·f` implies `g·g` also integrable), but logically sound. |
| 8 | `support_f_bin` | 87–89 | **CORRECT** | Support of indicator ⊆ the indicator set. |
| 9 | `measure_support_convolution_f_bin` | 91–102 | **CORRECT** | Chain: `support(f_bin_i ⋆ f_bin_i) ⊆ bin_interval + bin_interval ⊆ Ico(2a, 2b)`. Minkowski sum of `[a,b) + [a,b) = [2a, 2b)` with length `2δ = 1/(2n)`. The `linarith` at line 101 verifies the containment, and the measure computation at line 102 uses `norm_num`/`ring`. |
| 10 | `integral_convolution_f_bin` | 104–113 | **CORRECT** | Uses `MeasureTheory.integral_convolution` (Fubini): `∫(g⋆g) = (∫g)²`. Applied with `g = f_bin_i` which is integrable. Then `(∫ f_bin_i)² = (bin_masses f n i)²` by `integral_f_bin`. |
| 11 | `convolution_mono_ae_fbin` | 115–135 | **CORRECT** | Establishes a.e. integrability of `f(t)·f(x-t)` via product integrability + measure-preserving map `(p₁,p₂) ↦ (p₁, p₂-p₁)` + Fubini fiberization (`integrable_prod_iff'`). Then applies `convolution_mono_pointwise`. Variable shadowing (`h_int_f` reused 3 times) is confusing but valid in Lean. |
| 12 | `lintegral_convolution_f_bin` | 137–152 | **CORRECT** | Converts via `ofReal_integral_eq_lintegral_ofReal` (requires integrability + a.e. nonnegativity, both provided). Nonnegativity of convolution of nonneg functions at lines 148–152 is correct. |
| 13 | `lintegral_le_norm_mul_vol` | 154–180 | **CORRECT** | Key averaging inequality. **Finite measure case:** `∫⁻ ofReal(g) = ∫⁻_S ofReal(g) ≤ ∫⁻_S ‖g‖_∞ = ‖g‖_∞ · μ(S)`. Uses `enorm_ae_le_eLpNormEssSup` with `Real.enorm_eq_ofReal(hg x)` for the conversion. **Infinite measure case:** either `‖g‖_∞ = 0` (so `g = 0` a.e., integral vanishes) or `‖g‖_∞ · ∞ = ∞` (trivial bound). |
| 14 | `single_bin_bound_ennreal` | 182–220 | **CORRECT** | Core argument: (1) `∫⁻(f_bin_i ⋆ f_bin_i) = ofReal(M_i²)` (from `lintegral_convolution_f_bin`), (2) `∫⁻(f_bin_i ⋆ f_bin_i) ≤ ‖f_bin_i ⋆ f_bin_i‖_∞ · ofReal(1/(2n))` (from `lintegral_le_norm_mul_vol` + support bound), (3) `‖f ⋆ f‖_∞ ≥ ‖f_bin_i ⋆ f_bin_i‖_∞` (from `eLpNorm_mono_ae` + convolution monotonicity). Combining: `‖f⋆f‖_∞ · ofReal(1/(2n)) ≥ ofReal(M_i²)`, then multiply by `ofReal(2n)` and cancel. ENNReal arithmetic at lines 218–220 is correct. |
| 15 | **`single_bin_bound`** (Claim 4.5) | 222–236 | **CORRECT** | Converts from ENNReal via `ENNReal.toReal_mono` (requires `‖f⋆f‖_∞ < ⊤`, given as `h_fin`) and `toReal_ofReal` (nonneg by `positivity`). **Mathematical statement verified: ‖f⋆f‖_∞ ≥ 2n · M_i².** |
| 16 | **`conv_entry_le_total`** (Claim 4.8) | 240–244 | **CORRECT** | Drops the `if i+j=k` filter: `∑∑(if ... then c_i·c_j else 0) ≤ ∑∑ c_i·c_j = (∑c_i)² = m²`. |
| 17 | **`conv_total`** (Claim 4.7) | 248–257 | **CORRECT** | Fubini sum exchange, then for each `(i,j)`, exactly one `k = i+j` lies in `range(2d-1)` (verified by `h_filter`: `i+j < 2d-1` since `i,j < d`). Result: `∑∑ c_i·c_j = m²`. |
| 18 | `int32_safe` | 260–262 | **CORRECT** | `200² = 40,000 ≤ 2³¹ - 1 = 2,147,483,647`. |

### Mathematical verification of single_bin_bound chain

The full chain for `single_bin_bound` (the Cauchy-Schwarz argument):

1. Restrict `f` to bin `i`: `f_bin_i = indicator(bin_interval n i) · f`
2. **Integral of self-convolution equals square of integral:** `∫(f_bin_i ⋆ f_bin_i) = M_i²` where `M_i = ∫ f_bin_i = bin_masses(f, n, i)`. This is Fubini: `∫∫ f_bin_i(t) · f_bin_i(x-t) dt dx = (∫ f_bin_i)²`.
3. **Support of self-convolution is small:** `support(f_bin_i ⋆ f_bin_i) ⊆ bin_interval + bin_interval`, which has Minkowski sum measure `2 · (1/(4n)) = 1/(2n)`.
4. **Averaging principle:** `‖f_bin_i ⋆ f_bin_i‖_∞ ≥ M_i² / (1/(2n)) = 2n · M_i²`.
5. **Monotonicity lift:** `f_bin_i ≤ f` pointwise (since `f ≥ 0`), so `(f_bin_i ⋆ f_bin_i)(x) ≤ (f ⋆ f)(x)` for all `x`, hence `‖f ⋆ f‖_∞ ≥ ‖f_bin_i ⋆ f_bin_i‖_∞ ≥ 2n · M_i²`.

This is the standard Cauchy-Schwarz / pigeonhole averaging argument. Each step is correctly formalized.

---

## File 2: AsymmetryBound.lean (165 lines, Claim 2.1)

### Definitions

| # | Definition | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `f_L` | 27 | **CORRECT** | `indicator(Ioo(-1/4, 0)) · f`. Left-half restriction. |

### Theorems

| # | Theorem | Lines | Verdict | Analysis |
|---|---|---|---|---|
| 2 | `f_L_le_f` | 29–33 | **CORRECT** | Indicator ≤ original for nonneg functions. |
| 3 | `f_L_supp` | 35–37 | **CORRECT** | Support of indicator ⊆ indicator set. |
| 4 | `f_L_conv_supp` | 39–48 | **CORRECT** | Minkowski sum: `Ioo(-1/4, 0) + Ioo(-1/4, 0) = Ioo(-1/2, 0)`. If `x ∉ Ioo(-1/2, 0)`, then for all `t`, either `f_L(t) = 0` or `f_L(x-t) = 0`, so the convolution integrand vanishes. The `linarith` at line 48 closes the arithmetic from the support constraints `{-1/4 < t < 0, -1/4 < x-t < 0}` ⟹ `-1/2 < x < 0`. |
| 5 | `convolution_mono_ae` | 51–75 | **CORRECT** | For `0 ≤ f ≤ g` integrable: a.e. `(f⋆f)(x) ≤ (g⋆g)(x)`. Establishes a.e. integrability of `g(y)·g(x-y)` via product integrability + `measurePreserving_prod_sub` change of variables + `integrable_prod_iff'` Fubini fiberization. Then `integral_mono_of_nonneg` with `f(t)·f(x-t) ≤ g(t)·g(x-t)`. |
| 6 | **`averaging_principle`** | 78–100 | **CORRECT** | For `g ≥ 0` integrable with `support(g) ⊆ S` and `μ(S) = ofReal(v)`, `v > 0`: `‖g‖_∞ ≥ ofReal((∫g)/v)`. Chain: `∫g = ∫_S g` (g vanishes outside S), convert to lintegral via `ofReal_integral_eq_lintegral_ofReal`, bound `∫⁻_S ofReal(g) ≤ ‖g‖_∞ · μ(S)` (each integrand ≤ essSup), rearrange via `div_le_iff_le_mul`. Does not require measurability of S (uses outer measure). |
| 7 | `integral_convolution_square` | 102–110 | **CORRECT** | `∫(f⋆f) = (∫f)²`. Direct application of `MeasureTheory.integral_convolution`. |
| 8 | `f_L_integrable` | 112–114 | **CORRECT** | Indicator of measurable set preserves integrability. |
| 9 | `convolution_nonneg` | 116–120 | **CORRECT** | `f,g ≥ 0 ⟹ (f⋆g)(x) = ∫ f(t)·g(x-t) dt ≥ 0`. |
| 10 | `f_L_nonneg` | 122–124 | **CORRECT** | Indicator nonneg from `f` nonneg. |
| 11 | `volume_Ioo_half` | 126–128 | **CORRECT** | `μ((-1/2, 0)) = 1/2`. |
| 12 | **`asymmetry_bound`** (Claim 2.1) | 133–163 | **CORRECT** | See detailed analysis below. |

### Mathematical verification of asymmetry_bound chain

The full chain for `asymmetry_bound`:

1. **Restrict to left half:** `f_L = indicator((-1/4, 0)) · f`, with `0 ≤ f_L ≤ f`.
2. **Convolution monotonicity:** `‖f_L ⋆ f_L‖_∞ ≤ ‖f ⋆ f‖_∞` (via `eLpNorm_mono_ae` applied to the pointwise bound `(f_L ⋆ f_L)(x) ≤ (f ⋆ f)(x)`).
3. **Integral of self-convolution:** `∫(f_L ⋆ f_L) = L²` where `L = ∫ f_L` (Fubini).
4. **Support of left-half self-convolution:** `support(f_L ⋆ f_L) ⊆ Ioo(-1/2, 0)` with `μ(Ioo(-1/2, 0)) = 1/2`.
5. **Averaging principle:** `‖f_L ⋆ f_L‖_∞ ≥ ofReal(L² / (1/2))`.
6. **Combine:** `‖f ⋆ f‖_∞ ≥ ‖f_L ⋆ f_L‖_∞ ≥ ofReal(L² / (1/2))`.
7. **Convert to reals:** `‖f ⋆ f‖_∞.toReal ≥ L²/(1/2) = 2L²`. The `ring!` at line 163 verifies the algebraic identity.

**Note on hypotheses:** Uses `hf_supp : support f ⊆ Set.Icc(-1/4, 1/4)` (closed interval), while `autoconvolution_constant` in Defs.lean uses `Set.Ioo` (open). This is correct and more general — `Ioo ⊆ Icc`, so any function with open-interval support also satisfies the closed-interval hypothesis.

---

## File 3: EssSup.lean (74 lines)

| # | Theorem | Lines | Verdict | Analysis |
|---|---|---|---|---|
| 1 | `eLpNorm_eq_essSup_ofReal` | 27–32 | **CORRECT** | For `f ≥ 0`: `‖f‖_∞ = eLpNormEssSup f = essSup(‖f(·)‖ₑ) = essSup(ofReal(f(·)))`, using `Real.enorm_eq_ofReal(hf x)` for the last step (valid since `f(x) ≥ 0`). |
| 2 | `lintegral_le_essSup_mul_measure_ennreal` | 35–51 | **CORRECT** | For ENNReal-valued `f` with `support(f) ⊆ S`: `∫⁻ f ≤ essSup(f) · μ(support(f)) ≤ essSup(f) · μ(S)`. Inner bound: restrict integral to support via `setLIntegral_eq_of_support_subset`, apply `lintegral_mono_ae` with `ae_le_essSup`, then `setLIntegral_const`. Outer bound: `mul_le_mul_left'` with `measure_mono`. |
| 3 | **`eLpNorm_ge_integral_div_measure_real`** | 54–72 | **CORRECT** | For `f ≥ 0`, `support(f) ⊆ S`, `0 < μ(S) < ∞`, `f` integrable, `‖f‖_∞ < ⊤`: conclusion `‖f‖_∞.toReal ≥ (∫f)/μ(S).toReal`. Uses `div_le_iff₀` with `ENNReal.toReal_pos`, then `integral_eq_lintegral_of_nonneg_ae` to convert Bochner integral to lintegral, applies `lintegral_le_essSup_mul_measure_ennreal`, and converts back via `toReal_mono`. Finiteness of the product `essSup · μ(S)` established via `mul_ne_top`. |

---

## Cross-cutting Concerns

### ENNReal ↔ Real Conversions

All conversions are handled correctly:

| Conversion | Where used | Validity condition | Verified? |
|---|---|---|---|
| `ofReal_integral_eq_lintegral_ofReal` | `lintegral_convolution_f_bin`, `averaging_principle`, `eLpNorm_ge_integral_div_measure_real` | Integrability + a.e. nonnegativity | Yes, in each case |
| `ENNReal.toReal_mono` | `single_bin_bound`, `asymmetry_bound` | Upper bound is finite | Yes (`h_fin`, `h_bdd`) |
| `ENNReal.toReal_ofReal` | `single_bin_bound` | Argument ≥ 0 | Yes (`positivity`) |
| `ENNReal.ofReal_mul` | `single_bin_bound_ennreal` L220 | First factor ≥ 0 | Yes (`M_i² ≥ 0` always) |
| `Real.enorm_eq_ofReal(hf x)` | `lintegral_le_norm_mul_vol`, `averaging_principle`, `eLpNorm_ge_integral_div_measure_real` | `f(x) ≥ 0` | Yes (from `hf` hypothesis) |

**No ENNReal conversion bugs found.**

### Measure-Theoretic Arguments

1. **Fubini for convolution integrals:** `MeasureTheory.integral_convolution` correctly applied in `integral_convolution_f_bin` and `integral_convolution_square`, with integrability hypotheses provided.

2. **Product integrability → fiberwise integrability:** The `integrable_prod_iff'` applications in both `convolution_mono_ae_fbin` (CauchySchwarz) and `convolution_mono_ae` (AsymmetryBound) correctly extract fiberwise integrability from product integrability.

3. **Measure-preserving change of variables:** `measurePreserving_prod_sub` for `(p₁,p₂) ↦ (p₁, p₂-p₁)` is standard and correctly applied in both files.

4. **Support containment arguments:** All `support ⊆ S` claims are geometrically correct:
   - `support(f_bin_i) ⊆ bin_interval(n, i)` (indicator support)
   - `support(f_bin_i ⋆ f_bin_i) ⊆ bin_interval + bin_interval` (Minkowski sum of support)
   - `support(f_L ⋆ f_L) ⊆ Ioo(-1/2, 0)` (Minkowski sum of `Ioo(-1/4, 0)`)

5. **No unnecessary measurability hypotheses:** The `averaging_principle` does not require `S` to be measurable — it only needs `μ(S) = ofReal(v)`. This is fine because the lintegral/setLIntegral APIs work with outer measure in Mathlib.

### Hypothesis Verification

Every hypothesis in the key theorems is necessary:

- **`single_bin_bound`** requires `f ≥ 0` (for monotonicity + nonnegativity of convolution), `support ⊆ Ioo(-1/4, 1/4)` (for bin structure), `n > 0` (for well-defined bins), integrability (for Fubini), and `‖f⋆f‖_∞ < ⊤` (for ENNReal → Real conversion).

- **`asymmetry_bound`** requires `f ≥ 0`, support containment (for the left-half restriction to make sense), `∫f = 1` (normalization, used via `integrable_of_integral_eq_one`), and `‖f⋆f‖_∞ ≠ ⊤` (for toReal conversion).

- **`eLpNorm_ge_integral_div_measure_real`** requires `f ≥ 0` (for enorm conversion), `support ⊆ S` (for the averaging argument), `μ(S) ∈ (0, ∞)` (for division), integrability (for integral conversion), and `‖f‖_∞ < ⊤` (for toReal).

---

## Summary

| File | Declarations | Correct | Suspicious | Errors |
|---|---|---|---|---|
| CauchySchwarz.lean | 18 | 18 | 0 | 0 |
| AsymmetryBound.lean | 12 | 12 | 0 | 0 |
| EssSup.lean | 3 | 3 | 0 | 0 |
| **Total** | **33** | **33** | **0** | **0** |

**All 33 definitions and theorems are mathematically correct.** The proof chains faithfully implement the standard Cauchy-Schwarz averaging argument (single-bin bound), the left-half restriction asymmetry argument, and the essential supremum / averaging principle from the Cloninger-Steinerberger framework. ENNReal ↔ Real conversions are handled carefully with explicit finiteness hypotheses throughout. No measure-theoretic gaps found.
