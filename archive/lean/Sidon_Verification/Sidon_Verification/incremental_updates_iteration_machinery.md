# Audit Report: Incremental Updates & Iteration Machinery

> **OUTDATED (2026-04-07):** This audit was conducted against the old coarse-grid
> parameterization (compositions summing to m). The Lean definitions have been updated
> to the C&S fine grid (compositions summing to 4nm, heights = c_i/m). Re-audit needed.

**Files audited:**
1. `lean/Sidon/IncrementalAutoconv.lean` — Incremental autoconvolution (Claim 4.2)
2. `lean/Sidon/FusedKernel.lean` — Fused kernel and quick-check (Claims 4.1, 4.3)
3. `lean/Sidon/GrayCode.lean` — Gray code kernel (Claims 4.9, 4.10, 4.11)

**Date:** 2026-03-24

**Verdict: ALL 15 DEFINITIONS/THEOREMS CORRECT. 0 LEAN-PYTHON MISMATCHES.**

---

## File 1: IncrementalAutoconv.lean (Claim 4.2)

| # | Definition/Theorem | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `int_autoconvolution` | 27-28 | **CORRECT** | Standard `sum_i sum_j [i+j=t] c_i * c_j`. Matches Python's `conv[k] = sum_{i+j=k} a_i * a_j` (Python uses the `2*a_i*a_j` symmetry optimization but computes the same value). |
| 2 | `autoconv_delta` | 31-32 | **CORRECT** | `new - old`, trivially well-defined. |
| 3 | `delta_eq_sum` | 35-45 | **CORRECT** | Distributes subtraction over double sum. Proof via `Finset.sum_sub_distrib` + `aesop`. |
| 4 | `unchanged_terms_zero` | 48-52 | **CORRECT** | If `c'_i = c_i` and `c'_j = c_j`, then `c'_i * c'_j - c_i * c_j = 0`. Trivial. |
| 5 | `delta_three_way_split` | 55-67 | **CORRECT** | Decomposes delta into (S x S) + (S x S^c) + (S^c x S), omitting the zero (S^c x S^c) group. Captures the refinement structure: when parent bin p splits into child bins {2p, 2p+1}, only pairs touching those bins contribute delta. |
| 6 | `cross_term_simplify` | 70-75 | **CORRECT** | `c'_i * c'_j - c_i * c_j = (c'_i - c_i) * c_j` when `c'_j = c_j`. Verified by `ring`. |
| 7 | `incremental_update_correct` | 78-81 | **CORRECT** | `old + (new - old) = new`. This is Claim 4.2: incremental update is bit-exact. Proved by `ring`. |
| 8 | `groups_exhaustive` | 84-86 | **CORRECT** | The 4 membership groups cover all (i,j). By `tauto`. |
| 9 | `groups_disjoint` | 89-96 | **CORRECT** | The 4 groups are pairwise disjoint. By `tauto`. |

### Python comparison

- **`int_autoconvolution`** matches `solvers.py:727-733` where the Python kernel computes `conv[k] = sum_{i+j=k} a_i * a_j` using the symmetry optimization `conv[i+j] += 2.0 * ai * (c[jj] * scale)` for `i < j` and `conv[2*ii] += ai * ai` for diagonal terms.
- **`delta_three_way_split`** matches the refinement structure in `_prove_target_generic` (`solvers.py:674-678`): when transitioning from parent to child, the changed set S = {2p, 2p+1} partitions the index pairs into exactly these three non-zero groups.
- **`incremental_update_correct`** validates the mathematical correctness of the incremental approach. The Python kernel recomputes convolution from scratch per child for performance, but this theorem guarantees an incremental approach would give identical results.

### Mathematical verification

**`delta_three_way_split`:** The four groups {(S,S), (S,S^c), (S^c,S), (S^c,S^c)} partition all index pairs. Since `unchanged_terms_zero` shows the (S^c,S^c) group contributes zero, the delta is exactly the sum of the other three groups. This is the standard decomposition for incremental convolution updates.

**`cross_term_simplify`:** When j is unchanged (`c'_j = c_j`):
- `c'_i * c'_j - c_i * c_j = c'_i * c_j - c_i * c_j = (c'_i - c_i) * c_j`

This factorization is the key algebraic identity enabling efficient incremental updates: cross terms depend only on the *delta* at the changed index.

**`incremental_update_correct`:** `old + delta = old + (new - old) = new`. Trivially correct by ring arithmetic.

---

## File 2: FusedKernel.lean (Claims 4.1, 4.3)

| # | Definition/Theorem | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `odometer_bijection` (Claim 4.1) | 27-32 | **CORRECT** | Existence of bijection `Fin(prod_i r_i) <-> forall i, Fin(r_i)`. Proved via `Fintype.equivOfCardEq` (cardinalities match by `Fintype.card_pi`). |
| 2 | `quickcheck_sound` (Claim 4.3) | 35-38 | **CORRECT** | Existential introduction: given witness `(l_star, s_star)` exceeding threshold, conclude `exists l s, ws l s > dyn l s`. Trivially proved by `<l_star, s_star, h>`. |
| 3 | `w_int_fast_update` | 41-53 | **CORRECT** | Sum of `c'` over `Icc` = `W_old` + conditional deltas at `2p` and `2p+1`. Proof splits `c'_i = c_i + delta_i` via `grind`, then applies `Finset.sum_add_distrib`. |

### Python comparison

- **`odometer_bijection`** matches the nested-loop Cartesian product iteration in `compositions.py:_fill_batch_generic` (stack-based odometer visiting every d-tuple summing to S exactly once).
- **`quickcheck_sound`** matches the early-exit logic in the Python kernel (`solvers.py:765-767`): when any window `(l, s)` has `tv > dyn_thresh`, the child is prunable without checking remaining windows.
- **`w_int_fast_update`** matches the `W_int = prefix_c_arr[hi_bin + 1] - prefix_c_arr[lo_bin]` incremental update in `solvers.py:763,995`. When bins 2p and 2p+1 change, `W_int_new = W_int_old + delta_{2p} * [2p in window] + delta_{2p+1} * [2p+1 in window]`.

### Mathematical verification

**`odometer_bijection`:** The Cartesian product `prod_i {lo_i, ..., hi_i}` has cardinality `prod_i (hi_i - lo_i + 1)`. The bijection from `Fin(prod)` to the product type is constructed via `Fintype.equivOfCardEq`, which uses Lean's `Fintype.card_pi` lemma to verify equal cardinalities. The `h_valid` hypothesis (`lo i <= hi i`) is technically unused in the proof (in N arithmetic, `hi - lo = 0` when `lo > hi`, so the bijection would still exist trivially), but is mathematically reasonable context.

**`quickcheck_sound`:** Given `ws l_star s_star > dyn l_star s_star`, the conclusion `exists l s, ws l s > dyn l s` follows by existential introduction with witnesses `l_star, s_star`. This isolates the logical step "finding one killing window suffices"; the full pruning soundness (that exceeding the dynamic threshold means the child is prunable) lives in `DynamicThreshold.lean` and `SubtreePruning.lean`.

**`w_int_fast_update`:** Since `c'` agrees with `c` at all positions except possibly `2p` and `2p+1`:
- `sum_{i in Icc} c'_i = sum_{i in Icc} c_i + [2p in Icc] * (c'_{2p} - c_{2p}) + [2p+1 in Icc] * (c'_{2p+1} - c_{2p+1})`

This follows because `c'_i - c_i = 0` for `i != 2p` and `i != 2p+1`.

---

## File 3: GrayCode.lean (Claims 4.9, 4.10, 4.11)

| # | Definition/Theorem | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `gray_code_bijection` (Claim 4.9) | 28-33 | **CORRECT** | Identical structure to `odometer_bijection`: existence of bijection `Fin(prod_i r_i) <-> forall i, Fin(r_i)`. Proved via `Fintype.equivOfCardEq`. |
| 2 | `cross_term_split` (Claim 4.10) | 37-53 | **CORRECT** | Splits `sum {q != 2p and q != 2p+1} f(q)` into `(sum_{q<2p} f(q)) + (sum_{q>2p+1} f(q))`. Proved via `Finset.sum_union` + disjointness. |
| 3 | `w_int_gray_update` (Claim 4.11) | 57-67 | **CORRECT** | Same statement as `w_int_fast_update`: sum of `c'` over `Icc` = `W_old` + conditional deltas at `2p` and `2p+1`. Cleaner proof via `Finset.sum_congr` with `by_cases` + `simp_all`. |

### Python comparison

- **`gray_code_bijection`** formalizes the mathematical property that Gray code enumeration visits every element of the Cartesian product exactly once. The Python `_fill_batch_generic` uses a stack-based odometer (equivalent traversal).
- **`cross_term_split`** matches the structure of cross-term computation in the fused kernel: when changing position p (bins 2p and 2p+1), the contributions from all other bins decompose into left (`q < 2p`) and right (`q > 2p+1`) parts.
- **`w_int_gray_update`** matches `solvers.py:763,995` — the same `W_int` incremental update formalized in `w_int_fast_update`.

### Mathematical verification

**`gray_code_bijection`:** Same argument as `odometer_bijection`. The hypothesis `hr : forall i, 0 < r i` is unused (when `r i = 0`, `Fin 0` is empty, the product is 0, and the bijection `Fin 0 -> (forall i, Fin(r_i))` is the unique empty function — still valid). Present for mathematical context.

**`cross_term_split`:** The indices `{0, ..., d-1} \ {2p, 2p+1}` = `{0, ..., 2p-1} union {2p+2, ..., d-1}` (given `2p+1 < d`). These two sets are disjoint (`q < 2p` vs `q > 2p+1`), so the sum splits by `Finset.sum_union`. The proof constructs explicit bijections between the attached finsets and the univ-filter finsets.

**`w_int_gray_update`:** Identical mathematical content to `w_int_fast_update`. The proof decomposes each `c'_i` as `c_i + [i=2p]*(c'_{2p}-c_{2p}) + [i=2p+1]*(c'_{2p+1}-c_{2p+1})` using `by_cases`, then distributes the sum.

---

## Cross-file: Lean vs Python Correspondence

| Lean concept | Python location | Match? |
|---|---|---|
| `int_autoconvolution` (double-sum form) | `solvers.py:727-733` (symmetry-optimized loop) | **Yes** — same value, different computation order |
| `incremental_update_correct` (old + delta = new) | Conceptual; Python recomputes from scratch per child | **Yes** — validates the incremental approach |
| `delta_three_way_split` (S = changed bins) | `solvers.py:674-678` (stack machine changes one position at a time) | **Yes** — S = {2p, 2p+1} for parent-to-child refinement |
| `cross_term_simplify` | Cross terms in incremental kernel | **Yes** — `(c'_i - c_i) * c_j` factorization |
| `odometer_bijection` / `gray_code_bijection` | `compositions.py:_fill_batch_generic` (stack-based odometer) | **Yes** — bijection guarantees every composition visited exactly once |
| `quickcheck_sound` | `solvers.py:765-767` (early exit when `tv > dyn_thresh`) | **Yes** — logical backbone of quick-check optimization |
| `w_int_fast_update` / `w_int_gray_update` | `solvers.py:763,995` (`W_int = prefix_c[hi+1] - prefix_c[lo]`) | **Yes** — incremental W_int update when bins 2p and 2p+1 change |

---

## Minor Observations (not errors)

1. **Unused hypotheses:** `odometer_bijection.h_valid` and `gray_code_bijection.hr` are not used in their proofs. The theorems are valid but slightly weaker than necessary (they hold unconditionally). These hypotheses are present for mathematical context and do not affect correctness.

2. **Duplication:** `w_int_fast_update` (FusedKernel.lean) and `w_int_gray_update` (GrayCode.lean) prove nearly identical statements with different proof strategies (`grind` vs `Finset.sum_congr` + `by_cases`). Not an error, but could be deduplicated.

3. **Thin theorem:** `quickcheck_sound` is logically trivial (existential introduction). The actual pruning soundness — that exceeding the dynamic threshold means the child is prunable — lives in `DynamicThreshold.lean` and `SubtreePruning.lean`. This theorem isolates one logical step in the chain.
