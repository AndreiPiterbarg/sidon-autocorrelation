# Audit Report: Discretization & Composition Properties

> **OUTDATED (2026-04-07):** This audit was conducted against the old coarse-grid
> parameterization (compositions summing to m). The Lean definitions have been updated
> to the C&S fine grid (compositions summing to 4nm, heights = c_i/m). Re-audit needed.

**Files audited:**
1. `lean/Sidon/ReversalSymmetry.lean` — Reversal symmetry (Claims 3.3a, 3.3e)
2. `lean/Sidon/RefinementMass.lean` — Refinement mass preservation (Claims 3.2c, 4.6)
3. `lean/Sidon/CompositionEnum.lean` — Composition enumeration (Claims 3.1, 3.2a)

**Date:** 2026-03-24

**Verdict: ALL 11 DEFINITIONS/THEOREMS CORRECT. 0 LEAN-PYTHON MISMATCHES.**

---

## File 1: ReversalSymmetry.lean

| # | Definition/Theorem | Verdict | Notes |
|---|---|---|---|
| 1 | `rev_vector` | **CORRECT** | Standard reversal `i -> c(d-1-i)` on `Fin d -> N` |
| 2 | `rev_vector_real` | **CORRECT** | Same reversal for R-valued vectors |
| 3 | `autoconv_reversal_symmetry` (Claim 3.3a) | **CORRECT** | Proves `conv[k](a) = conv[2d-2-k](rev(a))`. Proof re-indexes both sums via bijection `i -> d-1-i`, then `omega` closes the index arithmetic `(d-1-i)+(d-1-j) = k <=> i+j = 2d-2-k`. |
| 4 | `left_sum_reversal` (Claim 3.3e helper) | **CORRECT** | Shows `{i : i<n}` and `{2n-1-i : i<n}` partition `Fin(2n)`, so left-sum + reversed-left-sum = total = m. |
| 5 | `asymmetry_reversal_symmetric` (Claim 3.3e) | **CORRECT** | Since `L + L_rev = 1`, the condition `L >= t or 1-L >= t` is symmetric under `L <-> 1-L`. |

### Python comparison

- **`autoconv_reversal_symmetry`** matches `test_values.py:72-77` where `conv[i+j] += a[i]*a[j]` computes the same `discrete_autoconvolution`. The reversal symmetry `conv[k] = conv_rev[2d-2-k]` is what justifies `_canonical_mask` in `pruning.py:68-81` (filtering to `b <= rev(b)` lexicographically).
- **`asymmetry_reversal_symmetric`** matches `pruning.py:44-64`. The Python checks `left_frac in (1-threshold, threshold)` for "needs checking" — the complement of the asymmetry prune condition `L >= t or 1-L >= t`. The Lean theorem guarantees this condition is invariant under reversal, so no prunable configs are lost when canonical filtering is applied first.

### Mathematical verification

**`autoconv_reversal_symmetry`:** Let `b_i = a_{d-1-i}` (reversal). Then:
- `conv_b[k'] = sum_{i+j=k'} b_i * b_j = sum_{i+j=k'} a_{d-1-i} * a_{d-1-j}`
- Substituting `i' = d-1-i`, `j' = d-1-j`: `i'+j' = 2(d-1) - (i+j) = 2d-2-k'`
- So `conv_b[2d-2-k] = conv_a[k]`.

**`left_sum_reversal`:** The sets `{i : i < n}` = `{0,...,n-1}` and `{2n-1-i : i < n}` = `{n,...,2n-1}` are complementary halves of `{0,...,2n-1}`, so their sums over `c` partition the total `m`.

**`asymmetry_reversal_symmetric`:** Since `L + L_rev = 1` (from `left_sum_reversal`), we have `L_rev = 1-L`. The condition `L >= t or 1-L >= t` trivially equals `1-L >= t or L >= t` = `L_rev >= t or 1-L_rev >= t`.

---

## File 2: RefinementMass.lean

| # | Definition/Theorem | Verdict | Notes |
|---|---|---|---|
| 1 | `child_bin_pair_sum` | **CORRECT** | Trivial: `a_i + (parent_i - a_i) = parent_i` given `a_i <= parent_i`. |
| 2 | `child_preserves_total_mass` (Claim 3.2c) | **CORRECT** | Splits `sum_{j<2d} child_j` into `sum_{i<d}(child[2i] + child[2i+1])` = `sum_i parent_i = m`. Proof establishes even/odd partition via `Nat.even_or_odd'`. |
| 3 | `left_half_sum_invariant` (Claim 4.6) | **CORRECT** | The first `2n` child bins (indices 0..2n-1) are the even/odd splits of parent bins 0..n-1, so `sum_{j<2n} child_j = sum_{i<n} parent_i`. |
| 4 | `left_half_sum_same_for_all_children` | **CORRECT** | Immediate corollary: both children share the same left-half sum (= parent's left-half sum). |

### Python comparison

- **`child_preserves_total_mass`** matches the composition generators in `compositions.py` — all generators produce vectors summing to `S` (=m), and refinement preserves this invariant.
- **`left_half_sum_invariant`** matches the comment in `pruning.py:53-56`: *"is preserved exactly under refinement (child bins sum to parent bins)"*. The Python asymmetry pruning relies on this: if a parent's left-mass fraction is in the prunable range, all its children inherit the same left-mass fraction.

### Mathematical verification

**`child_preserves_total_mass`:** The child vector has `2d` entries constructed as even/odd pairs from `d` parent bins. The even/odd indices partition `{0,...,2d-1}`, so:
```
sum_{j=0}^{2d-1} child_j = sum_{i=0}^{d-1} (child[2i] + child[2i+1])
                          = sum_{i=0}^{d-1} (a_i + (parent_i - a_i))
                          = sum_{i=0}^{d-1} parent_i = m
```

**`left_half_sum_invariant`:** The first `2n` child indices `{0,...,2n-1}` correspond to even/odd splits of parent indices `{0,...,n-1}`. Each pair sums to the parent value:
```
sum_{j=0}^{2n-1} child_j = sum_{i=0}^{n-1} (child[2i] + child[2i+1])
                          = sum_{i=0}^{n-1} parent_i
```

This is the key invariant that makes asymmetry pruning inheritable across refinement levels.

---

## File 3: CompositionEnum.lean

| # | Definition/Theorem | Verdict | Notes |
|---|---|---|---|
| 1 | `composition_count` (Claim 3.1) | **CORRECT** | Stars-and-bars: the number of compositions of `m` into `d` non-negative integer parts equals `C(m+d-1, d-1)`. Proof by induction on `d`, conditioning on first element `c_0 = k`, reducing to compositions of `m-k` into `d-1` parts. Base cases and bijection between `Fin(m+1)` and bounded-N representations verified. |
| 2 | `per_bin_choices` (Claim 3.2a) | **CORRECT** | For a parent bin with mass `c_i <= 2*x_cap`, the valid split range `[max(0, c_i - x_cap), min(c_i, x_cap)]` has cardinality `min(c_i, x_cap) - max(0, c_i - x_cap) + 1`. Standard integer interval counting. |

### Python comparison

- **`composition_count`** matches `pruning.py:37-41`: `count_compositions(d, S) = comb(S + d - 1, d - 1)` — identical formula. Exact match.
- **`per_bin_choices`** corresponds to the per-bin loop bounds in `compositions.py` child generators (e.g., `_fill_batch_d4`, line 58: `while c2 <= r1` where `r1 = S - c0 - c1`). The Lean theorem gives the count of valid values per bin; the Python generators enumerate them directly. The bound `c_i <= 2*x_cap` ensures the interval is non-empty.

### Mathematical verification

**`composition_count`:** The classical stars-and-bars identity. We place `m` identical stars into `d` bins separated by `d-1` bars, choosing positions for the bars from `m+d-1` total symbols: `C(m+d-1, d-1)`.

Inductive proof structure:
- Base case `d=1`: exactly one composition `(m)`, and `C(m, 0) = 1`.
- Inductive step: fix `c_0 = k` for `k = 0,...,m`. The remaining `d-1` parts form a composition of `m-k`. By IH, the count is `C(m-k+d-2, d-2)`. Summing over k: `sum_{k=0}^{m} C(m-k+d-2, d-2) = C(m+d-1, d-1)` (hockey stick identity).

**`per_bin_choices`:** For a parent bin of mass `c_i`, the child's first sub-bin takes value `a_i` with `0 <= a_i <= c_i` and `0 <= c_i - a_i <= x_cap`, giving `a_i in [max(0, c_i - x_cap), min(c_i, x_cap)]`. The constraint `c_i <= 2*x_cap` ensures `c_i - x_cap <= x_cap`, so the interval is non-empty.

---

## Overall Summary

| File | Theorems | All Correct | Lean-Python Mismatches |
|---|---|---|---|
| ReversalSymmetry.lean | 5 | **YES** | 0 |
| RefinementMass.lean | 4 | **YES** | 0 |
| CompositionEnum.lean | 2 | **YES** | 0 |
| **Total** | **11** | **YES** | **0** |

### Mathematical concerns: None.

All 11 definitions/theorems are mathematically correct, the proofs are logically sound (confirmed by zero-error compilation), and every theorem statement faithfully captures the corresponding algorithmic property used in the Python implementation.
