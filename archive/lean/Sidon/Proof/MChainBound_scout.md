# Scout: Existing Lean Infrastructure for the M-Chain Proof

This document surveys the available definitions and lemmas in the existing
Sidon Lean project that are reusable for proving the M-chain per-conv-position
pruning theorem (`proof/m_chain_proof.md`).

## 1. Sidon/Defs.lean

Reusable definitions:

- `discrete_autoconvolution {d : ℕ} (a : Fin d → ℝ) (k : ℕ) : ℝ`
  = `Σ_i Σ_j (if i+j = k then a_i a_j else 0)`. Lines 47-48.
  This is exactly the `conv[k](a)` we need.
- `test_value (n m : ℕ) (c : Fin (2*n) → ℕ) (ℓ s_lo : ℕ) : ℝ` (lines 57-62) —
  TV using `c_i / m` heights; not directly needed but related.
- `bin_masses` (lines 65-71), `canonical_discretization` (lines 77-86),
  `canonical_cumulative_distribution` (90-96).
- `autoconvolution_ratio (f : ℝ → ℝ) : ℝ` (40-44) — `‖f*f‖_∞ / (∫f)²`.
- `convolution_nonneg` (131-135).

## 2. Sidon/Proof/Foundational.lean

Lemmas about discretization / cumulative distributions (F1-F15). Not directly
needed for M-chain but `canonical_discretization_sum_eq_m` shows `Σ c_i = 4nm`
under the hypothesis.

## 3. Sidon/Proof/StepFunction.lean

Critical reusable content:

- `step_function (n m : ℕ) (c : Fin (2*n) → ℕ) : ℝ → ℝ` (lines 46-54) — exactly
  the `f_a = (c_i / m) · 1_{bin_i}` we need.
- `step_function_nonneg`, `step_function_support`, `step_function_integrable`,
  `integral_step_function` (= 1).
- **`convolution_at_grid_point`** (lines 183-395) — THE KEY LEMMA:
  At grid point `y_k = -1/2 + (k+1) · (1/(4n))`,
  `(g * g)(y_k) = (1 / (4n) / m²) · discrete_autoconvolution((c_i : ℝ), k)`.

  In our M-chain notation with `d = 2n`, `t_k = y_k`, this gives
  `(f_a * f_a)(t_k) = conv[k](c) / (2d · m²)` (the integer-c version), which
  matches Lemma A part (A4) when `a_i = c_i / m`.

  Note: this lemma uses integer `c` and divides by `m²`. For our M-chain
  Lemma A statement, we need it for arbitrary real `a` (no `c/m` denominator).
  We'll need a real-valued version of `convolution_at_grid_point`.

## 4. Sidon/Proof/PostFilterF.lean

Critical reusable content:

- `BB_W (n : ℕ) (c : Fin (2*n) → ℕ) (s_lo ℓ : ℕ) (j : Fin (2*n)) : ℝ` —
  the row-mass `BB^j = Σ_{i: i+j ∈ window} c_i`. Lines 70-73.
  For M-chain with single-conv-position (no window-averaging), we want
  the analog `BB_single(c, k, j) = c_{k-j} if 0 ≤ k-j < d else 0`. The
  existing `BB_W` reduces to this when the window is a singleton; we
  define a fresh `M_b` for clarity.

- `Delta_BB (n : ℕ) (c : Fin (2*n) → ℕ) (s_lo ℓ : ℕ) : ℝ` —
  top-half-sorted minus bot-half-sorted. Lines 191-193.
  For M-chain, define `Delta_M_b k = sum_top - sum_bot` of the sorted
  vector `b(k)`. We can either reuse `Delta_BB` with a singleton window
  or define a parallel `Delta_M_b`.

- `Delta_BB_nonneg`, `Delta_BB_le_total`.

- **`lp_closed_form_le`** (lines 357-523) — THE LP CLOSED-FORM:
  For `δ : Fin (2n) → ℝ`, `h ≥ 0`, `|δ_j| ≤ h`, `Σ δ_j = 0`:
  `Σ_j δ_j BB_W^j ≤ h · Delta_BB`.

  This is exactly Lemma B specialized to a single window (any window).
  We will reuse this lemma.

- `linear_window_bound_F`, `tv_linear_bound_F` — TV-normalized linear bound.
- `tight_discretization_bound_F`, `tight_cascade_prune_sound_F` — full
  pruning theorem in window-averaged form.

The `lp_closed_form_le` is the main reusable engine for our Lemma B.

Note: the M-chain uses `|δ_j| ≤ 1` (with c integer, S = 2dm, etc.) — so we
apply `lp_closed_form_le` with `h = 1`. Specifically: parametrise
`a = c + δ` so `δ_j = a_j - c_j`; the cell condition `|a_j - c_j| ≤ 1`
matches `|δ_j| ≤ 1`. The LP bound gives `Σ δ_j BB^j ≤ 1 · Delta_BB`.

## 5. Sidon/Proof/TightDiscretizationBound.lean

Reusable content:

- `ell_int_arr (n k : ℕ) : ℕ` (line 118) — `n_pairs(k) = #{(i,j) ∈ [0,2n-1]² : i+j = k}`.
  Closed form: `if k+1 ≤ 2n then k+1 else if k+1 < 4n then 4n-1-k else 0`.
- `ell_int_arr_eq_card` (lines 137-...) — proven equality with the cardinality.
- `ell_int_sum`, `ell_int_sum_eq_card`.
- `window_pair_set (n s_lo ℓ : ℕ) : Finset (Fin (2*n) × Fin (2*n))` —
  `{(i,j) : i+j ∈ [s_lo, s_lo+ℓ-2]}`.
- `N_row`, `N_row_le_ell_minus_1`, `sum_n_i_eq_ell_int_sum`.
- `W_int_overlap` — sum of c over rows that touch the window.
- `delta_sq_window_bound` (lines 550-586) — `|Σ_W δ_i δ_j| ≤ ell_int_sum / m²`
  given `|δ_i| ≤ 1/m`. This is exactly Lemma C in window-summed form.
  For single-conv-position: same proof works with `n_pairs(k)` in place of
  `ell_int_sum` (cardinality of a single conv position = `n_pairs(k)`).
- `tv_delta_sq_bound` — TV-normalized version.
- `test_value_real` (line 938-942), `test_value_real_eq_window_sum`.
- `tight_cascade_prune_sound`.

## 6. Sidon/Proof/PostFilterQ.lean

Multi-window joint LP. Not directly needed for M-chain (which is per-conv-
position), but the same `lp_closed_form_le` generalizes there.

## 7. Sidon/Proof/WRefinedDefs.lean

`W_int_for_window`, `w_refined_correction`. Not directly used.

## Plan for the M-Chain Lean File

The clean approach: **specialise** the existing `BB_W`/`Delta_BB`/`lp_closed_form_le`
machinery to the **single-conv-position** case `k`. We model the M-chain
window as `[k, k]` (single integer, ℓ = 2), so:

  `BB_W n c k 2 j = Σ_i (if i + j ∈ [k, k] then c_i else 0) = c_{k-j}` (clipped).

This is exactly the `b(k)` vector of the M-chain. With this trick,
`Delta_BB n c k 2` already equals `Delta_b(k)` (the sort-extremes value).

For the quadratic part (Lemma C), we use a custom version with `h = 1`
(not `1/m`); the proof template from `delta_sq_window_bound` is reused.

For the main theorem, we combine:
- Lemma A (`convolution_at_grid_point`) to reduce `(f_a * f_a)(t_k)` to
  `(1 / (4n)) / m² · Σ_{i+j=k} a_i a_j`.
- Decomposition `a = c/m + δ/m` (heights), re-cast as `(c + δ_int)/m` with
  `|δ_int| ≤ 1`.

### Choice of "M-chain window" specialisation

We use `s_lo = k` and `ℓ = 2`. Then `s_lo + ℓ - 2 = k`. The window
becomes `{i+j = k}`, exactly the single conv position.

`ell_int_sum n k 2 = ell_int_arr n k = n_pairs(k)`.
`window_pair_set n k 2 = {(i,j) : i+j = k}`.
`BB_W n c k 2 j = c_{k-j}` clipped — exactly `b(k)_j`.
`Delta_BB n c k 2` = `Delta_b(k)` (top-bot-sorted).

This lets us reuse `lp_closed_form_le` directly.

### Sketch of the file

1. Define `M_pruned (n m c c_target)` as `max_k LB(k) > c_target` where
   `LB(k) = (Σ_{i+j=k} (c_i:ℝ) · c_j - 2 · Delta_BB n c k 2 - n_pairs(k)) / (2d·m²)`.
2. Prove the per-cell soundness theorem: for `a` in the cell (heights
   `c_i/m + δ_i/m` with `|δ_i| ≤ 1`, `Σ δ = 0`, `a ≥ 0`),
   `(f_a*f_a)(t_k) ≥ LB(k)`.
3. Conclude `M(f_a) ≥ max_k LB(k)`.

### Potential complications

- The `c_target` comparison is in absolute units (≈ 1.281). The Cell
  containing `a` works in scaled heights `c/m`, while the LB bound is in
  the same scale (per-conv-position) as `(f_a*f_a)(t_k)`. We must verify
  that `(f_a*f_a)(t_k)` and `LB(k)` are both computed in heights where
  `Σ a_i = S = 4nm` and the integral of `f_a` equals 1.
- Verify the `2·d·m²` factor: at `d = 2n`, `2·d·m² = 4·n·m²`, matching
  Lemma A's denominator `(1/(4n)) / m² = 1 / (4n·m²)`. So `LB(k)` and
  `(f_a*f_a)(t_k)` are in the same units.
- Asymmetric `δ_i ≥ 0` constraint when `c_i = 0` (Lemma D). The LP
  closed-form gives a sound *upper bound* on `δ·b` (the linear term)
  over a *symmetric* polytope; restricting to `δ_i ≥ 0` only tightens
  the polytope. So the bound `δ·b ≥ -Delta_b` carries through.

## Conclusion

Strategy: **set the M-chain window to `(s_lo=k, ℓ=2)`** and reuse
`Delta_BB`, `lp_closed_form_le`, `delta_sq_window_bound`, etc. by direct
instantiation. We need:
- A clean Lemma A (convolution-at-grid-point in real heights, no `c/m`).
- A clean Lemma B (instantiate `lp_closed_form_le` with `h = 1`).
- A clean Lemma C (specialise `delta_sq_window_bound`, also with `h = 1`).
- Lemma D (subset polytope ⇒ `min_S f ≥ min_T f` for `S ⊆ T`).
- Main theorem combining these.
