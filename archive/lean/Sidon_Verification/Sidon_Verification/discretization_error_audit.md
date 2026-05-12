# Correctness Audit: `DiscretizationError.lean`

> **OUTDATED (2026-04-07):** This audit was conducted against the old coarse-grid
> parameterization (compositions summing to m). The Lean definitions have been updated
> to the C&S fine grid (compositions summing to 4nm, heights = c_i/m). Re-audit needed.

**Date**: 2026-03-25
**File**: `lean/Sidon/DiscretizationError.lean` (742 lines)
**Status**: BLOCKED (upstream errors in `TestValueBounds.lean`; logical content audited below is sound)

## Overview

This file proves the discretization error bound, contributing bins characterization,
correction term bound, and dynamic threshold soundness — the chain connecting discrete
test values to continuous autoconvolution ratios. It imports from `Defs.lean`,
`Foundational.lean`, `StepFunction.lean`, and `TestValueBounds.lean`.

## Overall Assessment

**The file is logically sound.** Every theorem statement correctly encodes its intended
mathematical claim, all proofs follow valid reasoning chains, there are no `sorry`/`admit`/
`native_decide`/`Axiom` declarations, and all dependencies are invoked with correct
signatures. One minor issue (unused hypothesis) is noted below.

---

## 1. Theorem Statement Correctness

### `discretization_error_bound` (line 38)

**Statement**: `|c_i/m - μ_i| ≤ 1/m` for each bin `i`.

**Verification**: Writing `ε_k = T_k·m - ⌊T_k·m⌋` (fractional part), we have
`c_i/m - μ_i = (ε_i - ε_{i+1})/m`. Since `0 ≤ ε_k < 1`, we get `|c_i/m - μ_i| < 1/m`,
so the `≤` bound is correct (weaker but valid). Types, quantifiers, and the `Fin (2*n)`
index are all correct. **No issues found.**

### `contributing_bins_iff` (line 121)

**Statement**: `i ∈ CB(n, ℓ, s_lo) ↔ max(0, s_lo - (2n-1)) ≤ i ∧ i ≤ min(2n-1, s_lo + ℓ - 2)`.

**Verification**: Bin `i` contributes iff `∃ j ∈ Fin(2n), s_lo ≤ i+j ≤ s_lo+ℓ-2`. Such `j`
exists iff the range `[max(0, s_lo - i), min(2n-1, s_lo+ℓ-2-i)]` is non-empty, which
simplifies to the stated interval characterization. Natural-number subtraction edge cases
are handled by `hn : n > 0` and `hℓ : 2 ≤ ℓ`. **No issues found.**

### `discretization_autoconv_error` (line 338)

**Statement**: `TV(c) - TV_cont ≤ (4n/ℓ)·(1/m² + 2W/m)`.

**Verification**: The two-term decomposition `w_i·w_j - μ_i·μ_j = δ_i·w_j + μ_i·δ_j` yields
`Q = Part_A + Part_B`. Part_A ≤ W/m (via range sum bounds and CB restriction), Part_B ≤
W/m + 1/m² (via CB mass bound `Σ_{CB} μ_i ≤ W + 1/m`). Combined: `Q ≤ 1/m² + 2W/m`. The
prefactor `(4n/ℓ)` from factoring `(4n)²/(4nℓ)` is correct. Matches the Python
`correction()` formula in `pruning.py`. **No issues found.**

### `correction_term_bound` (line 666)

**Statement**: `R(f) ≥ TV(c, ℓ, s_lo) - (4n/ℓ)·(1/m² + 2W/m)`.

**Verification**: Chains `R(f) ≥ TV_cont` (from `continuous_test_value_le_ratio`) with
`TV(c) - TV_cont ≤ correction` (from `discretization_autoconv_error`).
**No issues found.**

### `correction_term` (line 684)

**Statement**: `R(f) ≥ max_TV(c) - 2n·(2/m + 1/m²)`.

**Verification**: Takes the maximizing window `(ℓ*, s_lo*)` from `max_test_value_le_max`,
applies `correction_term_bound`, then bounds `(4n/ℓ*) ≤ 2n` (since `ℓ* ≥ 2`) and
`2W/m ≤ 2/m` (since `W ≤ 1`). Matches the Python global correction `2n*(2/m + 1/m²)`.
**No issues found.**

### `dynamic_threshold_sound` (line 738)

**Statement**: If `TV(c, ℓ, s_lo) > c_target + correction`, then for all `f` with
`canonical_discretization f n m = c`, we have `R(f) ≥ c_target`.

**Verification**: Direct application of `correction_term_bound` plus `linarith`. The strict
`>` in the hypothesis yields `R(f) > c_target`, which implies `R(f) ≥ c_target`. Correct
use of `≥` in the conclusion. **No issues found.**

---

## 2. Proof Completeness

Grep for `sorry`, `admit`, `native_decide`, `Axiom` returned **zero matches** in
`DiscretizationError.lean`. **No issues found.**

---

## 3. Logical Soundness

### `simp_all` / `aesop` usage

All uses of `simp_all +decide` and `aesop` appear in contexts where the goal is a routine
simplification (e.g., unfolding definitions, closing trivial equalities, applying
positivity). No instance was found where `False` could be in the context to allow vacuous
closure.

### Hypothesis usage

**Minor issue**: In `dynamic_threshold_sound` (line 740), the hypothesis
`hc : ∑ i, c i = m` is stated but **never used** in the proof body. The proof works
purely through `correction_term_bound` and the `hdisc` rewrite. This is a **redundant
hypothesis** — it makes the theorem slightly harder to apply (callers must supply it) but
does **not** affect correctness. The downstream use in `FinalResult.lean` (line 152) does
supply `hc_sum`, so this causes no practical problem.

### `linarith` / `omega` / `norm_num` dependencies

All automated arithmetic closures are invoked in contexts where the needed inequalities
are present in the hypotheses. In particular:

- Line 109: `linarith` depends on `Int.floor_le` and `Int.lt_floor_add_one` applied to
  partial sums — correct.
- Line 266: `linarith` depends on `Int.sub_one_lt_floor` — correct.
- Line 305: `linarith` chains `cumulative_delta_upper` and `cumulative_delta_lower` —
  correct.

**No soundness issues found** (except the redundant hypothesis noted above).

---

## 4. Definition Consistency

| Definition | Source | Usage in DiscretizationError.lean | Correct? |
|---|---|---|---|
| `test_value(n, m, c, ℓ, s_lo)` | Defs.lean:48 | Lines 344, 674, 694, 743 | Yes |
| `test_value_continuous(n, f, ℓ, s_lo)` | StepFunction.lean:34 | Lines 344, 676 | Yes |
| `discrete_autoconvolution(a, k)` | Defs.lean:44 | Via `test_value`/`test_value_continuous` | Yes |
| `canonical_discretization(f, n, m)` | Defs.lean:76 | Lines 42, 46, 169, etc. | Yes |
| `contributing_bins(n, ℓ, s_lo)` | Defs.lean:87 | Lines 343, 512, 625, etc. | Yes |
| `bin_masses(f, n)` | Defs.lean:68 | Lines 42, 103, 139, etc. | Yes |
| `autoconvolution_ratio(f)` | Defs.lean:33 | Lines 674, 690 | Yes |
| `canonical_cumulative_distribution(f, n, m, k)` | Defs.lean:92 | Lines 153, 158, 171 | Yes |
| `max_test_value(n, m, c)` | Defs.lean:56 | Lines 691, 694 | Yes |

All definitions used with correct argument count, order, and mathematical intent.
**No issues found.**

---

## 5. Numeric Constants

The file is **fully general** — it proves theorems for arbitrary `n, m : ℕ` with `n > 0`,
`m > 0`. No hard-coded values of `n = 64`, `m = 20`, or `c_target = 7/5`. The specific
instantiation happens only in `FinalResult.lean`.

The Python cross-reference confirms:

- `correction = max(1, 4n/ℓ_min) * (2/m + 1/m²)` from `pruning.py` matches
  `correction_term` with `ℓ_min = 2`
- Per-window correction `(4n/ℓ) * (1/m² + 2W/m)` matches `dynamic_threshold_sound`

**No issues found.**

---

## 6. Dependency Correctness

| Imported lemma | Source | Invocation lines | Correct args? |
|---|---|---|---|
| `sum_bin_masses_eq_one` | TestValueBounds:68 | 104, 172, 227, 247, 368, 703 | Yes |
| `continuous_test_value_le_ratio` | TestValueBounds:347 | 676 | Yes |
| `max_test_value_le_max` | TestValueBounds:91 | 694 | Yes |
| `bin_masses_nonneg` | Foundational:49 | 149, 367 | Yes |
| `canonical_discretization_sum_eq_m` | Foundational:112 | 371, 705 | Yes |
| `canonical_cumulative_distribution_2n` | Foundational:43 | 174 | Yes |
| `canonical_cumulative_distribution_mono` | Foundational:76 | 175 | Yes |
| `canonical_discretization_eq_diff` | Foundational:63 | 179 | Yes |
| `canonical_cumulative_distribution_zero` | Foundational:38 | 210 | Yes |

**No issues found.**

---

## 7. Edge Cases

- **Bin i = 0**: The floor rounding gives `D(1) - D(0) = D(1) - 0 = D(1)`, handled by
  the general case. Correct.
- **Bin i = 2n-1** (last bin): The `split_ifs` at line 48 enters the `else` branch
  (`c_i = m - D(i)`). The proof handles `Nat.cast_sub` with monotonicity. Correct.
- **Empty contributing bins**: When CB is empty, sums over CB are 0 and bounds hold
  trivially (lines 634-636 handle this with `Finset.sum_empty`). Correct.
- **Window extending beyond valid range**: The `range_sum_delta_le` proof clips `b` to
  `2*n` when `b > 2*n` (lines 501-508), ensuring the Fin-indexed sums are well-defined.
  Correct.
- **Natural subtraction truncation**: `s_lo + ℓ - 2` with `ℓ ≥ 2` ensures no truncation.
  `2*n - 1` with `n > 0` ensures no truncation. Correct.

**No issues found.**

---

## 8. Proof Structure Detail

### Helper lemmas (private)

The file defines several private helper lemmas that build up the main results:

| Helper | Lines | Purpose |
|---|---|---|
| `target_cum_mass_eq` | 138-142 | Simplifies `target_cum_mass` when `Σ μ_i = 1` |
| `target_cum_mass_nonneg` | 144-150 | Nonnegativity of target cumulative mass |
| `ccd_eq_floor_natAbs` | 152-154 | `D(k) = ⌊target_cum_mass(k)⌋.natAbs` |
| `ccd_cast_eq` | 156-162 | Real cast: `(D(k) : ℝ) = ⌊target_cum_mass(k)⌋` |
| `partial_sum_discretization` | 164-211 | `Σ_{i<k} c_i = D(k)` via telescoping |
| `partial_sum_mu` | 213-216 | Filter-to-ite rewrite for bin mass sums |
| `cumulative_delta_upper` | 218-237 | `Σ_{i<k} (c_i/m - μ_i) ≤ 0` |
| `cumulative_delta_lower` | 239-266 | `-1/m ≤ Σ_{i<k} (c_i/m - μ_i)` |
| `range_sum_delta_le` | 268-305 | `Σ_{a≤i<b} δ_i ≤ 1/m` |
| `range_sum_delta_ge` | 307-334 | `-1/m ≤ Σ_{a≤i<b} δ_i` |

The helper lemma chain is: `ccd_cast_eq` + `target_cum_mass_eq` enable
`partial_sum_discretization` (telescoping), which feeds into `cumulative_delta_upper/lower`
(cumulative error bounds), which combine into `range_sum_delta_le/ge` (range error bounds),
which are the workhorse for the main `discretization_autoconv_error` proof.

### Main proof chain

```
discretization_error_bound  (|δ_i| ≤ 1/m per bin)
         |
    range_sum_delta_le/ge   (|Σ_{range} δ_i| ≤ 1/m)
         |
discretization_autoconv_error  (TV(c) - TV_cont ≤ (4n/ℓ)(1/m² + 2W/m))
         |
    + continuous_test_value_le_ratio  (R(f) ≥ TV_cont)
         |
correction_term_bound  (R(f) ≥ TV(c) - correction, per window)
         |
correction_term  (R(f) ≥ max_TV(c) - global correction)
         |
dynamic_threshold_sound  (if TV > c_target + correction, then R(f) ≥ c_target)
```

### Key algebraic decomposition (lines 389-662)

The core of `discretization_autoconv_error` uses the identity:

```
w_i · w_j - μ_i · μ_j = δ_i · w_j + μ_i · δ_j
```

to split Q into Part_A and Part_B, then bounds each part independently:

- **Part_A** (lines 451-534): Exchange summation order to get
  `Part_A = Σ_j w_j · g(j)` where `g(j) = Σ_{i in range} δ_i ≤ 1/m` and `g(j) = 0`
  outside CB. So `Part_A ≤ (1/m) · Σ_{CB} w_j = W/m`.

- **Part_B** (lines 535-656): Symmetrically,
  `Part_B = Σ_i μ_i · h(i)` where `h(i) ≤ 1/m` and `h(i) = 0` outside CB.
  Using `Σ_{CB} μ_i ≤ W + 1/m`, get `Part_B ≤ (1/m)(W + 1/m) = W/m + 1/m²`.

- **Combined**: `Q ≤ W/m + W/m + 1/m² = 1/m² + 2W/m`.

---

## 9. Cross-Reference with Python Implementation

| Lean concept | Python equivalent | Match? |
|---|---|---|
| `test_value` | `test_values.py:85-101` (`inv_norm * ws`) | Yes |
| `contributing_bins` | Implicit in window/conv logic | Yes |
| `discretization_autoconv_error` bound | `pruning.py:11-25` (`correction()`) | Yes |
| `dynamic_threshold_sound` | `solvers.py:1038-1039` (`c_target = prune_target - correction_max`) | Yes |
| Per-window vs global correction | Python uses global for speed; Lean proves per-window | Compatible |

---

## Summary Table

| Category | Verdict |
|---|---|
| 1. Theorem statement correctness | **No issues found** |
| 2. Proof completeness (sorry/admit/axiom) | **No issues found** |
| 3. Logical soundness | **1 minor issue**: unused `hc` hypothesis in `dynamic_threshold_sound` (line 740) |
| 4. Definition consistency | **No issues found** |
| 5. Numeric constants | **No issues found** |
| 6. Dependency correctness | **No issues found** |
| 7. Edge cases | **No issues found** |

**Conclusion**: The file `DiscretizationError.lean` is mathematically correct and logically
sound. Once the upstream errors in `TestValueBounds.lean` are resolved, it should compile
without modification.
