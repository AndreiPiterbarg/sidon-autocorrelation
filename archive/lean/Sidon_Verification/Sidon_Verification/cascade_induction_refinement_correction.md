# Audit Report: Cascade Induction, Refinement Support & Correction Terms

> **OUTDATED (2026-04-07):** This audit was conducted against the old coarse-grid
> parameterization (compositions summing to m). The Lean definitions have been updated
> to the C&S fine grid (compositions summing to 4nm, heights = c_i/m). Re-audit needed.

**Files audited:**
1. `lean/Sidon/CascadeInduction.lean` — Cascade completeness (Claim 3.4)
2. `lean/Sidon/RefinementSupport.lean` — Refinement & support properties (Claims 2.2, 2.3)
3. `lean/Sidon/CorrectionSupport.lean` — Correction term support lemmas

**Date:** 2026-03-24

**Verdict: ALL 11 DEFINITIONS/THEOREMS CORRECT. No errors or suspicious items.**

---

## File 1: CascadeInduction.lean (Claim 3.4)

| # | Definition/Theorem | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `merge_pairs` | 32 | **CORRECT** | Merges consecutive child bin pairs: `child[2i] + child[2i+1]`. Correctly indexed with `by omega` bounds proofs. Inverse of the refinement operation in `run_cascade.py`. |
| 2 | `merge_pairs_sum` | 37-42 | **CORRECT** | Mass preservation under merging by induction on `d`. Uses `Fin.sum_univ_castSucc` for inductive step, `ring!` closes the algebraic goal. |
| 3 | `cascade_completeness_step` | 46-67 | **CORRECT** | The inductive backbone of the proof. Logical structure verified in detail below. |

### Detailed analysis: `cascade_completeness_step`

**Hypotheses:**
- `h_discretize_sum`: Every continuous function discretizes to a composition summing to `m`
- `h_pruning_sound`: If test value exceeds `c_target + 2/m + 1/m²`, then `autoconvolution_ratio f ≥ c_target`
- `h_all_pruned`: Every composition (summing to `m`) at the final resolution has some window where the test value exceeds the threshold

**Proof trace:**
1. Takes any `f` (nonneg, supported on `(-1/4, 1/4)`, nonzero integral)
2. Defines `c := discretize f (2*(2^L*n)) m` — line 64
3. `h_discretize_sum` gives `∑ c_i = m` — line 65
4. `h_all_pruned c hc_sum` gives killing window `(ℓ, s_lo)` with `h_val : tv > threshold` — line 66
5. `h_pruning_sound n m c_target L c ℓ s_lo f h_val rfl` concludes — line 67

The `rfl` at line 67 works because `c` is definitionally `discretize f (2*(2^L*n)) m`.

**Shadowing note:** `h_pruning_sound` universally quantifies over `n, m, c_target, L`, shadowing the outer parameters. When applied on line 67, these are instantiated with the outer values. The `m` in `2/(m:ℝ)` inside `h_pruning_sound` consistently refers to the same value throughout. No type-theoretic issue.

### Python comparison

- **`merge_pairs`** matches the inverse of the refinement in `run_cascade.py`'s `process_parent_fused`: each parent bin `parent[q]` splits into child bins `child[2q], child[2q+1]` with `child[2q] + child[2q+1] = parent[q]`.
- **`cascade_completeness_step`** captures the single-resolution argument: if ALL compositions at resolution `d_L = 2*(2^L*n)` are pruned, then `c ≥ c_target`. The multi-level cascade is an optimization (pruning at coarser levels avoids testing descendants), justified by `SubtreePruning.lean`.

### Critical path observation

`cascade_completeness_step` is **NOT on the critical path** of `FinalResult.lean`. The main theorem `autoconvolution_ratio_ge_7_5` uses the axiom `cascade_all_pruned` + `dynamic_threshold_sound` directly, with a per-window dynamic threshold:

```
7/5 + (4·64/ℓ) × (1/m² + 2·W/(m·m))
```

rather than the flat correction `2/m + 1/m²` used in `cascade_completeness_step`. Both are valid; the dynamic version is tighter and is what the computation actually uses.

### Threshold consistency verification

The axiom `cascade_all_pruned` uses threshold (in test-value space):
```
1.4 + (256/ℓ) × (1/400 + W_int/200) = 1.4 + (0.64 + 1.28·W_int)/ℓ
```

The Python code's integer-space threshold `c_target·m²·ℓ/(4n) + 1 + 2·W_int` converts to test-value space as:
```
1.4 + (4n/(m²ℓ)) × (1 + 2·W_int) = 1.4 + (0.64 + 1.28·W_int)/ℓ
```

**Exact match.** The Python code additionally includes `eps_margin` (= 1e-9·m²) and `one_minus_4eps` (= 1 - 4·2.22e-16) safety margins, making it slightly MORE conservative than the axiom — an additional safety buffer.

---

## File 2: RefinementSupport.lean (Claims 2.2, 2.3)

| # | Definition/Theorem | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `support_convolution_subset_add` | 27-33 | **CORRECT** | Standard Minkowski sum result for convolution support. |
| 2 | `left_frac_exact` | 36-41 | **CORRECT** | Trivial arithmetic: `(-1/4) + n·(1/(4n)) = 0`. |
| 3 | `asymmetry_no_margin` | 44-49 | **CORRECT** | Clean contrapositive proof for threshold comparison. |
| 4 | `convolution_integrand_le` | 52-54 | **CORRECT** | Pointwise monotonicity via `mul_le_mul`. |
| 5 | `integral_convolution_le` | 57-62 | **CORRECT** | Direct `integral_mono` application. |
| 6 | `measure_support_convolution_bound` | 65-71 | **CORRECT** | `μ(supp(g*g)) ≤ 2δ` via Minkowski sum inclusion. |

### Mathematical verification

**`support_convolution_subset_add`:** Proves `supp(f*f) ⊆ s + s` when `supp(f) ⊆ s`.

Proof by contrapositive: if no `y` has both `f(y) ≠ 0` and `f(x-y) ≠ 0`, then the convolution integrand `f(t)·f(x-t) = 0` for all `t`, so `(f*f)(x) = 0` via `integral_eq_zero_of_ae` with `of_forall`. The `by_cases h : f t = 0` correctly splits: if `f(t) = 0`, the product is zero; if `f(t) ≠ 0`, then by hypothesis `f(x-t) = 0`, so the product is zero. The existential witness `⟨y, hf hy1, x-y, hf hy2, by ring⟩` correctly constructs the Minkowski sum element. `aesop` handles the `ContinuousLinearMap.mul ℝ ℝ` unfolding.

**`asymmetry_no_margin`:** Proves `2L² ≥ c_target` from `L ≥ √(c_target/2)` and an oracle hypothesis. After `contrapose! h_bound`, the goal becomes `∃ L', 2L'² ≤ c_target ∧ L ≤ L'`. Witness `L` with `by linarith` (from `2L² < c_target`) and `le_rfl`. Matches `pruning.py:asymmetry_threshold(c_target) = sqrt(c_target/2)`.

**`measure_support_convolution_bound`:** For `supp(g) ⊆ (a, a+δ)`:
- Minkowski sum: `(a, a+δ) + (a, a+δ) = (2a, 2a+2δ)` — four `linarith` calls close the bounds from `y.1 > a, y.2 < a+δ, z.1 > a, z.2 < a+δ`.
- `μ((2a, 2a+2δ)) = 2δ` — closed by `simp`.

### Python comparison

- **`asymmetry_no_margin`** matches the asymmetry pruning in `pruning.py:asymmetry_prune_mask`: compositions with `left_frac ≥ threshold` (or `right_frac ≥ threshold`) are pruned because `‖f*f‖_∞ ≥ 2·threshold² ≥ c_target`. The "no margin" result confirms no discretization margin is needed for the asymmetry comparison.
- **`measure_support_convolution_bound`** underpins the support-based L∞ lower bound used throughout the discretization error analysis.

---

## File 3: CorrectionSupport.lean

| # | Definition/Theorem | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `nat_floor_approx` | 27-30 | **CORRECT** | Floor division approximation bound. |
| 2 | `product_approx_error` | 33-36 | **CORRECT** | Product approximation error bound. |

### Mathematical verification

**`nat_floor_approx`:** Proves `|x/m - ⌊x⌋/m| ≤ 1/m` for `x ≥ 0`, `m > 0`.

After `field_simp`, equivalent to `|x - ⌊x⌋| ≤ 1` (divided by `m`). By `Nat.floor_le h`: `⌊x⌋ ≤ x`, and by `Nat.lt_floor_add_one x`: `x < ⌊x⌋ + 1`. Hence `0 ≤ x - ⌊x⌋ < 1 ≤ 1`. `abs_cases` splits into positive/negative; `nlinarith` closes both branches using these two Mathlib lemmas.

**`product_approx_error`:** Proves `|x₁x₂ - y₁y₂| ≤ y₁ + y₂ + 1` given `|xᵢ - yᵢ| ≤ 1` and all values non-negative.

Algebraic verification: write `xᵢ = yᵢ + dᵢ` with `|dᵢ| ≤ 1`:
```
x₁x₂ - y₁y₂ = y₁d₂ + d₁y₂ + d₁d₂
```
- **Upper bound** (d₁ = d₂ = +1): `y₁ + y₂ + 1` — equality at extremes
- **Lower bound** (d₁ = d₂ = -1): `-y₁ - y₂ + 1 ≥ -(y₁ + y₂ + 1)` since `2 ≥ 0`

Proof uses `abs_le.mpr` to split into two goals, each closed by `nlinarith [abs_le.mp h1, abs_le.mp h2]`. The `nlinarith` calls derive the bounds by multiplying linear inequalities `xᵢ ≤ yᵢ + 1` with non-negativity hypotheses (e.g., `x₁x₂ ≤ (y₁+1)(y₂+1) = y₁y₂ + y₁ + y₂ + 1` for the upper bound).

### Role in the proof

These lemmas support the discretization error analysis in `DiscretizationError.lean`. `nat_floor_approx` bounds the error from floor-rounding in `canonical_discretization`, and `product_approx_error` bounds the error when approximating products of continuous bin masses with discrete integer compositions.

---

## Cross-File Structural Analysis

### Cascade completeness and the axiom

The cascade algorithm's correctness chain:

1. **L0**: All `C(23, 3) = 1771` compositions at `d=4` enumerated (891 canonical). Survivors → L1.
2. **L1–L4**: For each surviving parent, ALL possible refinements generated and tested.
3. **L5**: 76,829 parents × children → 0 survivors.

The axiom `cascade_all_pruned` (in `FinalResult.lean`) asserts all `Fin 128 → ℕ` compositions summing to 20 have a killing window. This covers:
- **(a) Directly tested compositions** (children of L4 survivors): pruned at L5, test value exceeds dynamic threshold.
- **(b) Untested compositions** (ancestors pruned at L0–L4): justified by subtree pruning (`SubtreePruning.lean`, Claim 4.4) — if a parent is pruned, all descendants in its subtree are also prunable.

### Correction term consistency

| Context | Correction formula | Value (m=20, ℓ=2, W=20, n=64) |
|---|---|---|
| `cascade_completeness_step` (flat) | `2/m + 1/m²` | 0.1025 |
| `cascade_all_pruned` (dynamic) | `(256/ℓ)·(1/400 + W_int/200)` | 13.12 |
| Python `_prune_dynamic_int32` | `(4n/(m²ℓ))·(1 + 2·W_int)` | 13.12 |

The dynamic and Python corrections match exactly. The flat correction in `cascade_completeness_step` is an independent (smaller) bound not used on the critical path.

---

## Summary

All 11 definitions and theorems across the three files are **mathematically correct**. The cascade induction structure is sound, the support and refinement properties are standard, and the correction term bounds are tight. The axiom `cascade_all_pruned` in `FinalResult.lean` exactly matches the Python computation's threshold (with additional floating-point safety margins in the Python code).
