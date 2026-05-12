/-
IntervalBnB — Theorem 1 of THEOREM.md.

  max_{t ∈ [-1/2, 1/2]} (f * f)(t)  ≥  μ^T M_W μ,   for every admissible f and every W.

Hence
  C_{1a}  ≥  val(d).

Proof: combine Lemma 1.2 (averaging), Lemma 1.3 (window integral), and
`|I_W| = ℓ/(2d)` (ScaleFactor).  The specific algebraic identity

    (2d/ℓ) · ∑_{i+j ∈ K_W} μ_i μ_j  =  μ^T M_W μ

is immediate from the `window_matrix` definition.

STATUS: this file assembles the previous lemmas.  The two measure-
theory steps (Fubini on the convolution, and `∫_I h ≤ |I| · ess sup h`
for `h = f*f`) are packaged as fields of `Admissible` in `Defs.lean`
(`averaging_ineq` and `window_integral_ge`).  Simplex membership of
`bin_mass d f` reduces to nonnegativity of `∫ (indicator · f)` plus
`bin_sum_eq_one`, also an `Admissible` field.
-/

import IntervalBnB.Defs
import IntervalBnB.Averaging
import IntervalBnB.WindowIntegral
import IntervalBnB.ScaleFactor

set_option linter.mathlibStandardSet false
set_option autoImplicit false
set_option relaxedAutoImplicit false

open scoped BigOperators
open scoped Classical

noncomputable section

namespace IntervalBnB

variable {d : ℕ}

/-- Algebraic identity:
    `quadForm W μ = (2d / ℓ) · ∑_{i,j : i+j ∈ K_W} μ_i μ_j`. -/
lemma quadForm_eq_scaled_sum (W : Window d) (μ : Fin d → ℝ) :
    quadForm W μ =
      ((2*d : ℝ) / W.ell) *
        ∑ i : Fin d, ∑ j : Fin d,
          (if (i.val + j.val) ∈ pair_sum_support W then μ i * μ j else 0) := by
  unfold quadForm window_matrix
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl; intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl; intro j _
  by_cases h : (i.val + j.val) ∈ pair_sum_support W
  · simp [h]; ring
  · simp [h]

/-- Main inequality (Theorem 1 of THEOREM.md).
    For every admissible `f`, every window `W`, with `μ_i = ∫_{B_i} f`,
       essSup_{[-1/2,1/2]} (f*f)  ≥  μ^T M_W μ.

    Proof: `averaging_bound` + `window_integral_lower_bound`
    combined with `quadForm_eq_scaled_sum` and `|I_W| = ℓ/(2d)`.
    The two measure-theoretic steps are packaged as `Admissible`
    fields (see `Defs.lean`); here the assembly is purely algebraic. -/
theorem main_inequality
    (f : ℝ → ℝ) (hf : Admissible f) (W : Window d) (hd : 0 < d) :
    essSupConv f ≥ quadForm W (bin_mass d f) := by
  -- Step 1: averaging gives  essSup · |I_W| ≥ ∫_{I_W} (f*f)
  have h_avg : essSupConv f * ((W.ell : ℝ) / (2*d)) ≥
      MeasureTheory.integral MeasureTheory.volume
        (Set.indicator (window_interval W) (autoconv f)) :=
    averaging_bound f hf W hd
  -- Step 2: window integral dominates the pair-mass sum
  have h_int : MeasureTheory.integral MeasureTheory.volume
        (Set.indicator (window_interval W) (autoconv f)) ≥
      ∑ i : Fin d, ∑ j : Fin d,
        (if (i.val + j.val) ∈ pair_sum_support W
         then bin_mass d f i * bin_mass d f j else 0) :=
    window_integral_lower_bound f hf W
  -- Combine to get the "per-|I_W|" inequality
  have hscale_pos : (0 : ℝ) < (W.ell : ℝ) / (2*d) :=
    window_interval_length_pos W hd
  -- Multiply both sides of essSup · |I_W| ≥ sum by  1/|I_W| > 0:
  --   essSup ≥ (2d/ℓ) · sum = quadForm
  have h_chain : essSupConv f * ((W.ell : ℝ) / (2*d)) ≥
      ∑ i : Fin d, ∑ j : Fin d,
        (if (i.val + j.val) ∈ pair_sum_support W
         then bin_mass d f i * bin_mass d f j else 0) := le_trans h_int h_avg
  -- Divide by |I_W| = ℓ/(2d) > 0
  have h_essSup :
      essSupConv f ≥
        (((2*d : ℝ) / W.ell) *
          ∑ i : Fin d, ∑ j : Fin d,
            (if (i.val + j.val) ∈ pair_sum_support W
             then bin_mass d f i * bin_mass d f j else 0)) := by
    have hell_pos : (0 : ℝ) < (W.ell : ℝ) := by
      have : 0 < W.ell := W.ell_pos
      exact_mod_cast this
    -- From h_chain: essSup * (ell/(2d)) ≥ sum.  So essSup ≥ sum * (2d/ell).
    rw [ge_iff_le, mul_comm]
    have h1 :
        (∑ i : Fin d, ∑ j : Fin d,
            (if (i.val + j.val) ∈ pair_sum_support W
             then bin_mass d f i * bin_mass d f j else 0)) ≤
        essSupConv f * ((W.ell : ℝ) / (2*d)) := h_chain
    -- Multiply both sides of h1 by (2d/ell) which is positive.
    have hscale2_pos : (0 : ℝ) < (2*d : ℝ) / W.ell := by
      apply div_pos _ hell_pos
      have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
      linarith
    have := mul_le_mul_of_nonneg_right h1 (le_of_lt hscale2_pos)
    rw [mul_assoc] at this
    have hrw : ((W.ell : ℝ) / (2*d)) * ((2*d : ℝ) / W.ell) = 1 := by
      have hd_pos : (0 : ℝ) < 2*d := by
        have : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
        linarith
      field_simp
    rw [hrw, mul_one] at this
    exact this
  rw [quadForm_eq_scaled_sum]
  exact h_essSup

/-- Windows d is nonempty when d ≥ 1: take the minimal window (ℓ=2, s_lo=0). -/
lemma window_nonempty (hd : 2 ≤ 2*d) : (Set.univ : Set (Window d)).Nonempty :=
  ⟨⟨(2, 0), by refine ⟨?_, ?_, ?_⟩ <;> omega⟩, Set.mem_univ _⟩

/-- Main corollary (Theorem 1 of THEOREM.md, consequence):

    For every admissible `f` and every `W`,
        essSup f*f  ≥  max_{W'} μ^T M_{W'} μ  ≥  val(d).

    Since `C_{1a}` is the infimum of `essSup (f*f)` over admissible `f`
    (by definition), we get `C_{1a} ≥ val(d)`. -/
theorem essSup_ge_val_d
    (f : ℝ → ℝ) (hf : Admissible f) (hd : 1 ≤ d) :
    essSupConv f ≥ val_d d := by
  have hd' : 0 < d := hd
  have h_each : ∀ W : Window d, essSupConv f ≥ quadForm W (bin_mass d f) :=
    fun W => main_inequality f hf W hd'
  -- essSup ≥ sSup over Windows of quadForm W μ
  have h_sup : essSupConv f ≥
      sSup ((fun W : Window d => quadForm W (bin_mass d f)) '' Set.univ) := by
    apply csSup_le
    · exact (window_nonempty (d := d) (by omega)).image _
    · rintro v ⟨W, _, rfl⟩
      exact h_each W
  -- val_d d ≤ essSup: obtain if we had μ = bin_mass ∈ simplex_d.  The
  -- sum-equals-1 half is the `bin_sum_eq_one` `Admissible` field (a
  -- Tonelli on the disjoint bin decomposition of `[-1/4, 1/4]`), and
  -- nonnegativity of each component is `integral_nonneg` applied to the
  -- indicator of `Bin d i`.
  have hμ_simplex : bin_mass d f ∈ simplex_d d := by
    refine ⟨?_, bin_sum_eq_one d hd' f hf⟩
    intro i
    -- bin_mass = ∫ indicator (Bin d i) f ≥ 0 since indicator · f ≥ 0.
    unfold bin_mass
    apply MeasureTheory.integral_nonneg
    intro x
    by_cases hx : x ∈ Bin d i
    · simp [Set.indicator_of_mem hx, hf.nonneg x]
    · simp [Set.indicator_of_notMem hx]
  have h_val_le : val_d d ≤
      sSup ((fun W : Window d => quadForm W (bin_mass d f)) '' Set.univ) := by
    unfold val_d
    -- bddBelow: use the same nonneg bound as in Symmetry
    have hBB : BddBelow ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
                         simplex_d d) := by
      refine ⟨0, ?_⟩
      rintro u ⟨ν, hν, rfl⟩
      apply Real.sSup_nonneg
      rintro t ⟨W, _, rfl⟩
      -- quadForm nonneg on simplex
      unfold quadForm window_matrix
      apply Finset.sum_nonneg; intro i _
      apply Finset.sum_nonneg; intro j _
      by_cases h : (i.val + j.val) ∈ pair_sum_support W
      · rw [if_pos h]
        have hν_i : 0 ≤ ν i := hν.1 i
        have hν_j : 0 ≤ ν j := hν.1 j
        have hell : 0 < W.ell := W.ell_pos
        have hell_R : (0 : ℝ) < (W.ell : ℝ) := by exact_mod_cast hell
        have hd_R : (0 : ℝ) ≤ (d : ℝ) := by exact_mod_cast Nat.zero_le _
        have hdiv : (0 : ℝ) ≤ (2*d : ℝ) / W.ell := by positivity
        have h1 : 0 ≤ ν i * ((2*d : ℝ) / W.ell) := mul_nonneg hν_i hdiv
        exact mul_nonneg h1 hν_j
      · rw [if_neg h]; simp
    apply csInf_le hBB
    exact ⟨bin_mass d f, hμ_simplex, rfl⟩
  linarith

end IntervalBnB

end -- noncomputable section
