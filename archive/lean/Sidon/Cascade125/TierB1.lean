/-
Sidon Cascade-125 — Tier B1: μ-Space Corner Lower Bound

Mirrors `tier_B1_mu_corner` in `_coarse_bnb_v4.py`:

  def tier_B1_mu_corner(cell, W, c_target) -> float:
      """μ^T A_W μ ≥ Σ_{(i,j) in W} lo_i · lo_j  (since μ ≥ lo ≥ 0).
      Sound for ANY A_W (uses μ ≥ lo ≥ 0 only).  No PSD assumption."""
      lo = cell.lo
      corner_val = float(np.sum(W.A * np.outer(lo, lo)))
      return W.Q_coef * corner_val - c_target

Mathematical content: for any μ in the cell (which has lo ≥ 0),
  μ_i · μ_j ≥ lo_i · lo_j  for every (i, j)
because nonneg-monotone-product.  Summing over (i, j) with weight A_W[i,j] ≥ 0
and multiplying by Q_coef = 2d/ℓ > 0 gives the inequality.

We prove the corner bound is a sound lower bound on `mass_test_value`
(which is exactly Q_coef · μ^T A_W μ; see `CoarseCascade.lean`).

No axioms, no sorries.
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.CoarseCascade
import Sidon.Cascade125.Cell

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

namespace Sidon.Cascade125

/-- Corner value (Python's `corner_val`):
    `∑_{k ∈ [s, s+ℓ-2]} ∑_{i + j = k} lo_i · lo_j`. -/
noncomputable def cornerConvSum {d : ℕ} (lo : Fin d → ℝ) (ℓ s : ℕ) : ℝ :=
  ∑ k ∈ Finset.Icc s (s + ℓ - 2), discrete_autoconvolution lo k

/-- B1 lower bound on `mass_test_value` over a cell:
    `(2d/ℓ) · cornerConvSum lo`. -/
noncomputable def cellB1Bound {d : ℕ} (cell : Cell d) (ℓ s : ℕ) : ℝ :=
  (2 * (d : ℝ) / (ℓ : ℝ)) * cornerConvSum cell.lo ℓ s

/-- Pointwise monotonicity: if `0 ≤ a i ≤ b i` for all `i`, then
    `discrete_autoconvolution a k ≤ discrete_autoconvolution b k`. -/
theorem discrete_autoconvolution_mono {d : ℕ} (a b : Fin d → ℝ)
    (ha_nn : ∀ i, 0 ≤ a i) (h_le : ∀ i, a i ≤ b i) (k : ℕ) :
    discrete_autoconvolution a k ≤ discrete_autoconvolution b k := by
  unfold discrete_autoconvolution
  apply Finset.sum_le_sum
  intro i _
  apply Finset.sum_le_sum
  intro j _
  by_cases hk : i.1 + j.1 = k
  · simp [hk]
    have hb_nn_j : 0 ≤ b j := le_trans (ha_nn j) (h_le j)
    have hb_nn_i : 0 ≤ b i := le_trans (ha_nn i) (h_le i)
    exact mul_le_mul (h_le i) (h_le j) (ha_nn j) hb_nn_i
  · simp [hk]

/-- Sum monotonicity over a window. -/
theorem cornerConvSum_le_of_mem {d : ℕ} (cell : Cell d)
    {μ : Fin d → ℝ} (hμ : cell.Mem μ) (ℓ s : ℕ) :
    cornerConvSum cell.lo ℓ s ≤
    ∑ k ∈ Finset.Icc s (s + ℓ - 2), discrete_autoconvolution μ k := by
  unfold cornerConvSum
  apply Finset.sum_le_sum
  intro k _
  apply discrete_autoconvolution_mono
  · exact cell.lo_nonneg
  · intro i; exact (hμ i).1

/-- **B1 soundness**: for any `μ` in the cell, `mass_test_value μ ℓ s` is at
    least the B1 corner bound.  Provided `d > 0` and `ℓ > 0` (so the scaling
    coefficient `(2d/ℓ)` is nonneg). -/
theorem cellB1Bound_le_mass_test_value {d : ℕ} (cell : Cell d)
    {μ : Fin d → ℝ} (hμ : cell.Mem μ) (ℓ s : ℕ) (_hℓ : 0 < ℓ) :
    cellB1Bound cell ℓ s ≤ mass_test_value d μ ℓ s := by
  unfold cellB1Bound mass_test_value
  -- `(2d/ℓ)` is nonneg, so multiplying preserves `≤`.
  have h_coef_nn : 0 ≤ 2 * (d : ℝ) / (ℓ : ℝ) := by positivity
  exact mul_le_mul_of_nonneg_left (cornerConvSum_le_of_mem cell hμ ℓ s) h_coef_nn

/-- The cascade-level conclusion: a window for which the B1 bound exceeds
    `c_target` directly certifies that every `μ` in the cell satisfies
    `mass_test_value ≥ c_target`. -/
theorem cell_certified_by_B1 {d : ℕ} (cell : Cell d) (ℓ s : ℕ) (hℓ : 0 < ℓ)
    (c_target : ℝ) (h_bound : cellB1Bound cell ℓ s ≥ c_target) :
    ∀ μ : Fin d → ℝ, cell.Mem μ → mass_test_value d μ ℓ s ≥ c_target := by
  intro μ hμ
  exact le_trans h_bound (cellB1Bound_le_mass_test_value cell hμ ℓ s hℓ)

end Sidon.Cascade125
