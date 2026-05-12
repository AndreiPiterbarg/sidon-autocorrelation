# Audit Report: Subtree Pruning, Sliding Window & Dynamic Threshold

> **OUTDATED (2026-04-07):** This audit was conducted against the old coarse-grid
> parameterization (compositions summing to m). The Lean definitions have been updated
> to the C&S fine grid (compositions summing to 4nm, heights = c_i/m). Re-audit needed.

**Files audited:**
1. `lean/Sidon/SubtreePruning.lean` — Subtree pruning (Claim 4.4)
2. `lean/Sidon/SlidingWindow.lean` — Sliding window and zero-bin skip (Claims 4.12, 4.13)
3. `lean/Sidon/DynamicThreshold.lean` — Integer dynamic threshold (Claims 2.4, 5.1, 5.2)

**Date:** 2026-03-24

**Verdict: 16/17 DEFINITIONS/THEOREMS CORRECT. 1 SUSPICIOUS (`dyn_it` formula mismatch — does NOT affect proof soundness).**

---

## File 1: SubtreePruning.lean (Claim 4.4)

| # | Definition/Theorem | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `partial_conv_le_full_conv` | 28-35 | **CORRECT** | Restricting to `i,j < 2p` gives a subset of nonneg terms. Each summand is either the same (both conditions met) or 0 vs a nonneg product. Proof via `Finset.sum_le_sum` + `mul_nonneg`. |
| 2 | `w_int_bounded_unfixed` | 39-62 | **CORRECT** | Standard refinement grouping: child bins `{2q, 2q+1}` sum to parent bin `q` for `q >= p`. Groups indices into disjoint pairs via `Finset.sum_biUnion`, then uses `h_split` to bound each pair. |
| 3 | `w_int_bounded_corrected` | 64-81 | **CORRECT** | Decomposes `[lo, hi]` into `[lo, min(hi, 2p-1)]` (individual child bins) + `[max(lo, 2p), hi]` (parent-bounded). Sound interval splitting with disjointness proof. |
| 4 | `w_int_bounded` | 83-93 | **CORRECT** | Thin wrapper for `w_int_bounded_corrected`. |
| 5 | `dyn_it_mono` | 96-101 | **CORRECT** | Monotonicity of `floor((base + 2*W) * s)` in `W` when `s > 0`. Uses `Int.floor_le_floor` + `mul_le_mul_of_nonneg_right`. |
| 6 | `subtree_pruning_chain` | 104-109 | **CORRECT** | Chain `ws_full >= ws_partial > dyn_max >= dyn_actual ==> ws_full > dyn_actual`. Pure transitivity, closed by `omega`. |

### Python comparison

- **`partial_conv_le_full_conv`** matches the Python zero-skip optimization in `run_cascade.py:89-95` where `if ci != 0` / `if cj != 0` guards skip zero-mass bins. The theorem justifies that restricting to a subset of nonneg terms gives a lower bound.
- **`w_int_bounded`** matches the subtree pruning in the cascade (`run_cascade.py`): when refining from parent resolution `d/2` to child resolution `d`, child bins beyond index `2p` are bounded by parent sums. The `h_split` hypothesis encodes the refinement invariant `child[2q] + child[2q+1] = parent[q]`.
- **`subtree_pruning_chain`** matches the pruning soundness chain: full window sum >= partial (restricted to first 2p bins) > conservative threshold >= exact threshold. This justifies pruning at the parent level using partial information.

### Mathematical verification

**`partial_conv_le_full_conv`:** The restricted sum adds conditions `i < 2p` and `j < 2p` to each `if` branch. When these conditions fail, the restricted sum contributes 0 while the full sum contributes `c_i * c_j >= 0` (since all `c_i >= 0`). When conditions hold, both sums contribute the same product. Hence restricted <= full, term by term.

**`w_int_bounded_unfixed`:** For even `d`, each index `i >= 2p` maps to parent index `q = i/2 >= p`. The child pair `{2q, 2q+1}` sums to `parent[q]` by `h_split`. The proof:
1. Shows `S = Icc(max(lo,2p), hi)` is covered by `bigcup_{q in Q} Icc(2q, 2q+1)`
2. Uses disjointness of the pairs (if `a != b`, then `Icc(2a, 2a+1)` and `Icc(2b, 2b+1)` are disjoint — proved by `linarith` on the endpoints)
3. Bounds each pair sum by the parent value via `h_split`

**`w_int_bounded_corrected`:** Splits `Icc(lo, hi)` into low part `Icc(lo, min(hi, 2p-1))` and high part `Icc(max(lo, 2p), hi)`. The low part passes through unchanged (these bins have individual child values). The high part is bounded by `w_int_bounded_unfixed`. The union covers the original interval, and disjointness follows from `min(hi, 2p-1) < max(lo, 2p)` (when both parts are nonempty).

**`subtree_pruning_chain`:** `ws_full >= ws_partial > dyn_max >= dyn_actual` implies `ws_full > dyn_actual` by transitivity of `>=` and `>` on integers. This is the chain that makes subtree pruning sound: even using a partial window sum and a conservative threshold, pruning is justified.

---

## File 2: SlidingWindow.lean (Claims 4.12, 4.13)

| # | Definition/Theorem | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `sliding_window_step` (Claim 4.12) | 28-35 | **CORRECT** | Standard sliding window identity: `W_{s+1} = W_s + A[s+n_cv] - A[s]`. |
| 2 | `zero_term_vanishes` | 38-39 | **CORRECT** | Trivial: `a * 0 = 0`. |
| 3 | `sum_filter_zero` | 42-49 | **CORRECT** | Filtering out `c_j = 0` terms doesn't change `sum c_j * f_j`. Uses `Finset.sum_subset`. |
| 4 | `autoconv_zero_skip` (Claim 4.13) | 53-60 | **CORRECT** | Restricting autoconvolution double sum to `{i | c_i != 0} x {j | c_j != 0}` is exact. |
| 5 | `cross_term_zero_skip` | 63-73 | **CORRECT** | Same zero-filtering principle for cross-terms `sum delta * c_q`. |

### Python comparison

- **`sliding_window_step`** matches `run_cascade.py:112-114`: `ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])`. The Lean version uses `Finset.Ico` sums and shows the telescoping identity `sum[s+1, s+1+n_cv) = sum[s, s+n_cv) + A[s+n_cv] - A[s]`.
- **`autoconv_zero_skip`** matches `run_cascade.py:89-95` where the `if ci != 0` / `if cj != 0` guards skip zero-mass bins in the inner convolution loop. The theorem proves this optimization is exact (not just a bound).
- **`cross_term_zero_skip`** matches the same pattern applied to the cross-term computation when updating W_int incrementally.

### Mathematical verification

**`sliding_window_step`:** The window `[s, s+n_cv)` contains indices `{s, s+1, ..., s+n_cv-1}`. Shifting right by 1 gives `[s+1, s+1+n_cv) = {s+1, ..., s+n_cv}`. The difference is: remove `A[s]`, add `A[s+n_cv]`. So `W_{s+1} = W_s - A[s] + A[s+n_cv]`. The proof uses `Finset.sum_Ico_eq_sub` to decompose the sums and `Finset.sum_range_succ` to peel off the boundary terms.

**`autoconv_zero_skip`:** For any term where `c_i = 0` or `c_j = 0`, the product `c_i * c_j = 0` regardless of the `i+j = t` condition. So filtering the outer sum to `{i | c_i != 0}` and the inner sum to `{j | c_j != 0}` drops only zero summands. The proof uses `Finset.sum_filter` with contextual simplification.

**`cross_term_zero_skip`:** `delta * c_q = delta * 0 = 0` when `c_q = 0`, so filtering to `{q | c_q != 0}` is exact.

---

## File 3: DynamicThreshold.lean (Claims 2.4, 5.1, 5.2)

| # | Definition/Theorem | Lines | Verdict | Notes |
|---|---|---|---|---|
| 1 | `conv` | 27-28 | **CORRECT** | Standard integer autoconvolution `sum_{i+j=k} c_i * c_j`. |
| 2 | `window_sum` | 31-32 | **CORRECT** | `sum_{k in [s_lo, s_lo+l-2]} conv(k)`, which is `l-1` terms. Matches Python window range. |
| 3 | `dyn_it` | 35-37 | **SUSPICIOUS** | Formula mismatch with Python. See detailed analysis below. |
| 4 | `dyn_it_conservative` (Claim 2.4) | 40-53 | **CORRECT** (internally) | Proves `floor(A) <= floor(B)` where `B = dyn_it`. Valid: `1e-9*m^2` margin dominates `4*2.22e-16` by ~10^6 within stated bounds (`m <= 200`, `c_target <= 2`). |
| 5 | `pruning_condition` (Claim 5.1) | 56-57 | **CORRECT** | Simple predicate: `(ws : Z) > threshold`. |
| 6 | `pruning_soundness` (Claim 5.2) | 60-70 | **CORRECT** (internally) | `ws > dyn_it >= floor(A) ==> ws > floor(A)`. Transitivity via `lt.trans_le'`. |

### Python comparison

- **`conv`** matches `run_cascade.py:87-95` where the integer convolution is computed as `conv[i+j] += 2 * ci * cj` (symmetry optimization) and `conv[2*i] += ci * ci` (diagonal).
- **`window_sum`** matches the sliding window sum `ws` in `run_cascade.py:109-114`, which sums `l-1` consecutive convolution entries.

### Critical analysis: `dyn_it` formula mismatch

**Lean formula** (`dyn_it`, line 36-37):
```
floor((c_target * m^2 + 1 + 1e-9 * m^2 + 2 * W_int) * (l / (4*n)) * (1 - 4 * 2.22e-16))
```
ALL terms (including correction `1 + 2*W_int`) are multiplied by `l/(4n)`.

**Python formula** (`run_cascade.py:82-84,123-124`):
```
ct_base_ell = c_target * m^2 * ell / (4*n)
dyn_x = ct_base_ell + 1.0 + 1e-9 * m^2 + 2.0 * W_int
dyn_it = int64(dyn_x * (1 - 4*DBL_EPS))
```
Expanding: `floor((c_target * m^2 * l/(4n) + 1 + 1e-9*m^2 + 2*W_int) * (1 - 4*DBL_EPS))`.
Only `c_target * m^2` is multiplied by `l/(4n)`. The correction terms `1 + 1e-9*m^2 + 2*W_int` are added UNSCALED.

**The Python code comments explicitly state** (line 77-79): *"CORRECTED: the correction term (1 + 2\*W_int) is NOT scaled by ell/(4n). Correct threshold: floor(c_target\*m^2\*ell/(4n) + 1 + eps + 2\*W_int). Previously the (1 + 2\*W_int) was incorrectly multiplied by ell/(4n)."*

**Why the Python is correct:** The `cascade_all_pruned` axiom in `FinalResult.lean` (line 63-64) encodes the correction as:
```
(4 * n / l) * (1/m^2 + 2 * W_sum / (m * m))
```
Converting this test-value-space correction to integer ws-space (multiply by `m^2 * l / (4n)`):
```
(4n/l) * (1 + 2*W_int)/m^2 * m^2 * l/(4n) = 1 + 2*W_int
```
The `l/(4n)` factor cancels, confirming the correction in ws-space is `1 + 2*W_int` (unscaled). This matches Python, not Lean's `dyn_it`.

**Impact on proof soundness: NONE.** `FinalResult.lean` does NOT import `DynamicThreshold.lean`. The main theorem uses `cascade_all_pruned` (axiom with correct formula) + `dynamic_threshold_sound` (from `DiscretizationError.lean`), neither of which references `dyn_it`. The `DynamicThreshold.lean` definitions are compiled as part of `lake build Sidon` but are dead code in the proof dependency chain.

**Numerical example** (m=20, n=2, c_target=1.4, l=2, W_int=10):
- **Python**: `1.4*400*0.25 + 1 + 4e-7 + 20 = 161.0` -> `dyn_it = 161`
- **Lean**: `(560 + 1 + 4e-7 + 20) * 0.25 * (1-eps) = 145.25` -> `dyn_it = 145`
- **Cascade axiom** (correct): threshold = `1.4*400*0.25 + 1 + 20 = 161`

For `l/(4n) < 1` (small windows), the Lean formula is MORE aggressive (smaller threshold). For `l/(4n) > 1` (large windows), the Lean formula is LESS aggressive. Neither direction matters since `dyn_it` is unused.

### Epsilon/rounding guard analysis

| Guard | Lean value | Python value | Comparison | Impact |
|---|---|---|---|---|
| FP margin | `1e-9 * m^2` | `1e-9 * m^2` | Identical | Provides ~10^6x headroom over the `(1-4eps)` reduction |
| Machine epsilon | `2.22e-16` | `2.220446049250313e-16` | Lean rounds DOWN | Lean's `(1 - 4*2.22e-16)` is slightly LARGER than Python's `(1 - 4*DBL_EPS)`, making Lean marginally more conservative (safe direction). Difference ~1.78e-19 relative, negligible. |
| `int64()` truncation | `floor` | `np.int64()` (truncates toward zero) | Equivalent for positive values | Both round down. |

The `1e-9 * m^2` margin is adequate: for worst case (m=200, c_target=2, W_int=200), the margin is `4e-5` while the FP error absorbed is at most `~7.14e-11`. Safety factor > 500,000x.

### Pruning chain soundness

The full pruning chain verified across `SubtreePruning.lean` and `DynamicThreshold.lean`:

```
ws_full >= ws_partial > dyn_computed >= dyn_exact
   |            |            |              |
   |            |            |              +-- exact integer threshold
   |            |            +-- dyn_it (with epsilon guards)
   |            +-- restricted to first 2p bins (partial_conv_le_full_conv)
   +-- full window sum over all bins
```

Each link:
1. `ws_full >= ws_partial`: `partial_conv_le_full_conv` — nonneg subset gives lower bound
2. `ws_partial > dyn_computed`: the pruning test (runtime comparison)
3. `dyn_computed >= dyn_exact`: `dyn_it_conservative` — margin absorbs FP error

Transitivity: `subtree_pruning_chain` chains these into `ws_full > dyn_exact`.

The chain is logically complete. The connection from `dyn_exact` to the continuous autoconvolution ratio (`R(f) >= c_target`) is established in `DiscretizationError.lean` via `dynamic_threshold_sound`, using the correct (unscaled) correction formula from the cascade axiom.

---

## Summary

| File | Theorems | Correct | Suspicious | Errors |
|---|---|---|---|---|
| SubtreePruning.lean | 6 | 6 | 0 | 0 |
| SlidingWindow.lean | 5 | 5 | 0 | 0 |
| DynamicThreshold.lean | 6 | 5 | 1 | 0 |
| **Total** | **17** | **16** | **1** | **0** |

**The `dyn_it` formula mismatch is a real discrepancy between the Lean definition and the Python implementation, but has zero impact on proof soundness** because:
1. `dyn_it` is not imported or referenced by `FinalResult.lean` or any file in its dependency chain
2. The cascade axiom (`cascade_all_pruned`) directly encodes the correct threshold formula
3. `dynamic_threshold_sound` (in `DiscretizationError.lean`) connects the axiom to the main theorem independently of `dyn_it`

**Recommendation:** The `dyn_it` definition could be corrected for cleanliness (move the `l/(4n)` factor to only multiply `c_target * m^2`), but this is cosmetic — the proof does not depend on it.
