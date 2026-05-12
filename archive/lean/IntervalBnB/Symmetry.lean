/-
IntervalBnB — Section 3 of THEOREM.md.  Z/2 symmetry reduction.

σ : {0, …, d-1} → {0, …, d-1},  σ(i) = d - 1 - i.

Three facts:
  * `window_reversal` : (ℓ, s_lo) ↦ (ℓ, 2d - ℓ - s_lo) is a self-bijection
    on `Window d`.
  * `window_matrix_reversal` : M_{σ(W)}[i, j] = M_W[σ(i), σ(j)].
  * `sigma_invariance`       : max_W μ^T M_W μ = max_W (σμ)^T M_W (σμ).
  * `half_simplex_cover`     : val(d) = inf_{μ ∈ H_d} max_W μ^T M_W μ.

All four are pure combinatorics on Finset/Fin with arithmetic sprinkled
in; no measure theory.
-/

import IntervalBnB.Defs

set_option linter.mathlibStandardSet false
set_option autoImplicit false
set_option relaxedAutoImplicit false

open scoped BigOperators
open scoped Classical

noncomputable section

namespace IntervalBnB

variable {d : ℕ}

/-!
## 3.1 The window reversal bijection.
-/

/-- The reversal of a window: `(ℓ, s_lo) ↦ (ℓ, 2d - ℓ - s_lo)`.
    Matches THEOREM.md Lemma 3.1. -/
def reverseWindow (W : Window d) : Window d :=
  ⟨(W.ell, 2*d - W.ell - W.sLo), by
    refine ⟨W.ell_ge_two, W.ell_le, ?_⟩
    have h1 := W.sLo_add_ell_le
    have h2 := W.ell_le
    omega⟩

@[simp] lemma reverseWindow_ell (W : Window d) : (reverseWindow W).ell = W.ell := rfl
@[simp] lemma reverseWindow_sLo (W : Window d) :
    (reverseWindow W).sLo = 2*d - W.ell - W.sLo := rfl

/-- Double reversal is the identity. -/
lemma reverseWindow_involutive (W : Window d) :
    reverseWindow (reverseWindow W) = W := by
  obtain ⟨⟨ℓ, s⟩, h1, h2, h3⟩ := W
  apply Subtype.ext
  simp only [reverseWindow, Window.ell, Window.sLo]
  refine Prod.mk.injEq _ _ _ _ |>.mpr ⟨rfl, ?_⟩
  omega

/-- Reversal as an `Equiv`. -/
def windowReversalEquiv : Equiv (Window d) (Window d) where
  toFun := reverseWindow
  invFun := reverseWindow
  left_inv := reverseWindow_involutive
  right_inv := reverseWindow_involutive

/-!
## 3.2 The matrix reversal identity.
-/

/-- `σ(i) := d - 1 - i`. -/
def sigmaFin {d : ℕ} (i : Fin d) : Fin d :=
  ⟨d - 1 - i.val, by
    have hi : i.val < d := i.isLt
    omega⟩

@[simp] lemma sigmaFin_val {d : ℕ} (i : Fin d) : (sigmaFin i).val = d - 1 - i.val := rfl

lemma sigmaFin_involutive {d : ℕ} (i : Fin d) : sigmaFin (sigmaFin i) = i := by
  apply Fin.ext
  simp only [sigmaFin_val]
  have hi : i.val < d := i.isLt
  omega

/-- Key arithmetic identity for the matrix reversal:

    `(i.val + j.val) ∈ K_{σW}  ↔  (σi.val + σj.val) ∈ K_W`. -/
lemma pair_sum_support_reverseWindow
    (W : Window d) (i j : Fin d) :
    (i.val + j.val) ∈ pair_sum_support (reverseWindow W) ↔
      (((sigmaFin i).val) + ((sigmaFin j).val)) ∈ pair_sum_support W := by
  have hi : i.val < d := i.isLt
  have hj : j.val < d := j.isLt
  have hd_pos : 0 < d := by
    have := W.ell_pos; have := W.ell_le; omega
  have hell_le : W.ell ≤ 2*d := W.ell_le
  have hsum_le : W.sLo + W.ell ≤ 2*d := W.sLo_add_ell_le
  have hell_ge2 : 2 ≤ W.ell := W.ell_ge_two
  simp only [pair_sum_support, Finset.mem_Icc, reverseWindow_ell,
             reverseWindow_sLo, sigmaFin_val]
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨?_, ?_⟩
    · omega
    · omega
  · rintro ⟨h1, h2⟩
    refine ⟨?_, ?_⟩
    · omega
    · omega

/-- Matrix reversal: `M_{σW}[i, j] = M_W[σi, σj]`. -/
lemma window_matrix_reversal (W : Window d) (i j : Fin d) :
    window_matrix (reverseWindow W) i j =
      window_matrix W (sigmaFin i) (sigmaFin j) := by
  unfold window_matrix
  have hell : (reverseWindow W).ell = W.ell := rfl
  rw [hell]
  have hequiv :
      ((i.val + j.val) ∈ pair_sum_support (reverseWindow W)) ↔
        (((sigmaFin i).val + (sigmaFin j).val) ∈ pair_sum_support W) :=
    pair_sum_support_reverseWindow W i j
  by_cases h : (i.val + j.val) ∈ pair_sum_support (reverseWindow W)
  · have h' := hequiv.mp h
    rw [if_pos h, if_pos h']
  · have h' : ((sigmaFin i).val + (sigmaFin j).val) ∉ pair_sum_support W :=
      fun hh => h (hequiv.mpr hh)
    rw [if_neg h, if_neg h']

/-!
## 3.3 σ-invariance of the objective.
-/

/-- σ acting on distributions: `(σμ)_i = μ_{d-1-i}`. -/
def sigmaDist {d : ℕ} (μ : Fin d → ℝ) : Fin d → ℝ :=
  fun i => μ (sigmaFin i)

/-- σ as a bijection `Fin d ≃ Fin d`. -/
def sigmaEquiv {d : ℕ} : Fin d ≃ Fin d where
  toFun := sigmaFin
  invFun := sigmaFin
  left_inv := sigmaFin_involutive
  right_inv := sigmaFin_involutive

/-- `(σμ)^T M_W (σμ) = μ^T M_{σW} μ`. -/
lemma quadForm_sigmaDist_eq
    (W : Window d) (μ : Fin d → ℝ) :
    quadForm W (sigmaDist μ) = quadForm (reverseWindow W) μ := by
  unfold quadForm sigmaDist
  -- Reindexing lemma: `∑ i j, g i j = ∑ i j, g (σi) (σj)`
  -- via σ-bijection applied twice.
  have key : ∀ g : Fin d → Fin d → ℝ,
      (∑ i, ∑ j, g i j) = ∑ i, ∑ j, g (sigmaFin i) (sigmaFin j) := by
    intro g
    calc (∑ i, ∑ j, g i j)
        = ∑ i, ∑ j, g (sigmaEquiv i) j := (Equiv.sum_comp sigmaEquiv _).symm
      _ = ∑ i, ∑ j, g (sigmaEquiv i) (sigmaEquiv j) := by
            apply Finset.sum_congr rfl; intro i _
            exact (Equiv.sum_comp sigmaEquiv _).symm
      _ = ∑ i, ∑ j, g (sigmaFin i) (sigmaFin j) := rfl
  -- Apply `key` to the RHS `(i, j) ↦ μ i * M_{σW} i j * μ j`.
  rw [show (∑ i, ∑ j, μ i * window_matrix (reverseWindow W) i j * μ j)
        = ∑ i, ∑ j, μ (sigmaFin i) * window_matrix (reverseWindow W) (sigmaFin i)
                                                    (sigmaFin j) * μ (sigmaFin j)
        from key _]
  apply Finset.sum_congr rfl; intro i _
  apply Finset.sum_congr rfl; intro j _
  rw [window_matrix_reversal, sigmaFin_involutive, sigmaFin_involutive]

/-- σ-invariance of the max objective (Lemma 3.3). -/
lemma sigma_invariance
    (μ : Fin d → ℝ) :
    sSup ((fun W : Window d => quadForm W (sigmaDist μ)) '' Set.univ) =
      sSup ((fun W : Window d => quadForm W μ) '' Set.univ) := by
  have h_image :
      (fun W : Window d => quadForm W (sigmaDist μ)) '' Set.univ
        = (fun W : Window d => quadForm W μ) '' Set.univ := by
    apply Set.eq_of_subset_of_subset
    · rintro v ⟨W, _, rfl⟩
      refine ⟨reverseWindow W, Set.mem_univ _, ?_⟩
      exact (quadForm_sigmaDist_eq W μ).symm
    · rintro v ⟨W, _, rfl⟩
      refine ⟨reverseWindow W, Set.mem_univ _, ?_⟩
      have := quadForm_sigmaDist_eq (reverseWindow W) μ
      rw [reverseWindow_involutive] at this
      exact this
  rw [h_image]

/-!
## 3.4 Soundness of the half-simplex cut.
-/

/-- The half-simplex `H_d := {μ ∈ Δ_d : μ₀ ≤ μ_{d-1}}`. -/
def halfSimplex_d (d : ℕ) : Set (Fin d → ℝ) :=
  {μ ∈ simplex_d d | ∀ h0 : 0 < d, ∀ hd : d - 1 < d,
      μ ⟨0, h0⟩ ≤ μ ⟨d - 1, hd⟩}

/-- σμ is in Δ_d if μ is. -/
lemma sigmaDist_mem_simplex (μ : Fin d → ℝ) (hμ : μ ∈ simplex_d d) :
    sigmaDist μ ∈ simplex_d d := by
  obtain ⟨hnn, hsum⟩ := hμ
  refine ⟨?_, ?_⟩
  · intro i; exact hnn _
  · unfold sigmaDist
    calc (∑ i, μ (sigmaFin i))
        = ∑ i, μ (sigmaEquiv i) := rfl
      _ = ∑ i, μ i := Equiv.sum_comp sigmaEquiv μ
      _ = 1 := hsum

/-- quadForm on the simplex is nonneg. -/
lemma quadForm_nonneg_of_simplex (W : Window d) {μ : Fin d → ℝ}
    (hμ : μ ∈ simplex_d d) : 0 ≤ quadForm W μ := by
  unfold quadForm window_matrix
  apply Finset.sum_nonneg; intro i _
  apply Finset.sum_nonneg; intro j _
  by_cases h : (i.val + j.val) ∈ pair_sum_support W
  · rw [if_pos h]
    have hμ_i : 0 ≤ μ i := hμ.1 i
    have hμ_j : 0 ≤ μ j := hμ.1 j
    have hell : 0 < W.ell := W.ell_pos
    have hell_R : (0 : ℝ) < (W.ell : ℝ) := by exact_mod_cast hell
    have hd_R : (0 : ℝ) ≤ (d : ℝ) := by exact_mod_cast Nat.zero_le _
    have hdiv : (0 : ℝ) ≤ (2*d : ℝ) / W.ell := by positivity
    have h1 : 0 ≤ μ i * ((2*d : ℝ) / W.ell) := mul_nonneg hμ_i hdiv
    exact mul_nonneg h1 hμ_j
  · rw [if_neg h]
    simp

/-- Every element of the Δ_d image is nonneg, so 0 is a lower bound. -/
lemma bddBelow_quadFormSup_image :
    BddBelow ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
               simplex_d d) := by
  refine ⟨0, ?_⟩
  rintro u ⟨ν, hν, rfl⟩
  apply Real.sSup_nonneg
  rintro t ⟨W, _, rfl⟩
  exact quadForm_nonneg_of_simplex W hν

/-- Every element of the H_d image is nonneg, so 0 is a lower bound. -/
lemma bddBelow_quadFormSup_image_half :
    BddBelow ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
               halfSimplex_d d) := by
  refine ⟨0, ?_⟩
  rintro u ⟨ν, hν, rfl⟩
  apply Real.sSup_nonneg
  rintro t ⟨W, _, rfl⟩
  exact quadForm_nonneg_of_simplex W hν.1

/-- Reformulating: `val_d d ≤ sInf (image over H_d)` using containment `H_d ⊆ Δ_d`.

    Note: on ℝ `sInf` of an empty set is 0, not `-∞`, so we use the classical
    `sInf` monotonicity argument that works for non-empty, bounded-below sets.
    Our sets contain values ≥ 0 (quadratic form on a nonneg simplex), so this
    is fine — we bound via a direct explicit argument. -/
lemma val_d_le_halfSimplex_inf :
    val_d d ≤ sInf ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
                    halfSimplex_d d) := by
  unfold val_d
  by_cases h_halfEmpty :
      ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
        halfSimplex_d d) = ∅
  · -- H_d image empty: sInf (empty) = 0 in Mathlib; but then Δ_d image must
    -- also be empty (a rep of every Δ_d orbit is in H_d), so both are 0.
    rw [h_halfEmpty, Real.sInf_empty]
    by_cases h_deltaEmpty :
        ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
          simplex_d d) = ∅
    · rw [h_deltaEmpty, Real.sInf_empty]
    · -- Δ_d image nonempty: pick μ ∈ Δ_d; then μ or σμ is in H_d, contradiction.
      rw [Set.eq_empty_iff_forall_notMem, not_forall] at h_deltaEmpty
      obtain ⟨_, hv⟩ := h_deltaEmpty
      push_neg at hv
      rcases hv with ⟨μ, hμΔ, _⟩
      exfalso
      by_cases hcase : ∀ h0 : 0 < d, ∀ hd : d - 1 < d,
                         μ ⟨0, h0⟩ ≤ μ ⟨d - 1, hd⟩
      · have : sSup ((fun W : Window d => quadForm W μ) '' Set.univ) ∈
               ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
                halfSimplex_d d) := ⟨μ, ⟨hμΔ, hcase⟩, rfl⟩
        rw [h_halfEmpty] at this; exact this.elim
      · push_neg at hcase
        obtain ⟨h0, hd, hlt⟩ := hcase
        have hσμH : sigmaDist μ ∈ halfSimplex_d d := by
          refine ⟨sigmaDist_mem_simplex μ hμΔ, ?_⟩
          intro h0' hd'
          unfold sigmaDist
          have e1 : sigmaFin ⟨0, h0'⟩ = ⟨d - 1, hd⟩ := by
            apply Fin.ext; simp [sigmaFin_val]
          have e2 : sigmaFin ⟨d - 1, hd'⟩ = ⟨0, h0⟩ := by
            apply Fin.ext; simp [sigmaFin_val]
          rw [e1, e2]; exact le_of_lt hlt
        have : sSup ((fun W : Window d => quadForm W (sigmaDist μ)) '' Set.univ) ∈
               ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
                halfSimplex_d d) := ⟨sigmaDist μ, hσμH, rfl⟩
        rw [h_halfEmpty] at this; exact this.elim
  · -- H_d image nonempty: apply `le_csInf` with bddBelow and forall-ge.
    rw [Set.eq_empty_iff_forall_notMem, not_forall] at h_halfEmpty
    obtain ⟨_, hv⟩ := h_halfEmpty
    push_neg at hv
    rcases hv with ⟨μ₀, hμ₀H, _⟩
    have h_ne : ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
                  halfSimplex_d d).Nonempty := ⟨_, μ₀, hμ₀H, rfl⟩
    apply le_csInf h_ne
    rintro w ⟨μ', hμ'H, rfl⟩
    -- val_d d ≤ w: use that μ' ∈ Δ_d via H_d ⊆ Δ_d.
    apply csInf_le bddBelow_quadFormSup_image
    exact ⟨μ', hμ'H.1, rfl⟩

/-- The reverse inequality: `sInf (H_d image) ≤ val_d d`, using σ-invariance.
    For every μ ∈ Δ_d, either μ ∈ H_d or σμ ∈ H_d. In the latter case
    `sigma_invariance` says the max-value is preserved, so the H_d
    representative reaches the same objective. -/
lemma halfSimplex_inf_le_val_d :
    sInf ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
           halfSimplex_d d) ≤ val_d d := by
  unfold val_d
  by_cases h_deltaEmpty :
      ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
        simplex_d d) = ∅
  · -- Δ_d image empty: `sInf` of empty is 0; and since H_d ⊆ Δ_d, H_d is empty too.
    rw [h_deltaEmpty, Real.sInf_empty]
    have h_halfEmpty :
        ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
          halfSimplex_d d) = ∅ := by
      apply Set.eq_empty_iff_forall_notMem.mpr
      rintro v ⟨μ, hμH, rfl⟩
      have : sSup ((fun W : Window d => quadForm W μ) '' Set.univ) ∈
             ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
              simplex_d d) := ⟨μ, hμH.1, rfl⟩
      rw [h_deltaEmpty] at this; exact this
    rw [h_halfEmpty, Real.sInf_empty]
  · rw [Set.eq_empty_iff_forall_notMem, not_forall] at h_deltaEmpty
    obtain ⟨_, hv⟩ := h_deltaEmpty
    push_neg at hv
    rcases hv with ⟨μ₀, hμ₀Δ, _⟩
    have h_ne : ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
                  simplex_d d).Nonempty := ⟨_, μ₀, hμ₀Δ, rfl⟩
    apply le_csInf h_ne
    rintro v ⟨μ, hμΔ, rfl⟩
    by_cases hcase : ∀ h0 : 0 < d, ∀ hd : d - 1 < d,
                       μ ⟨0, h0⟩ ≤ μ ⟨d - 1, hd⟩
    · -- μ ∈ H_d: direct.
      apply csInf_le bddBelow_quadFormSup_image_half
      exact ⟨μ, ⟨hμΔ, hcase⟩, rfl⟩
    · -- σμ ∈ H_d, and the max value is the same.
      push_neg at hcase
      obtain ⟨h0, hd, hlt⟩ := hcase
      have hσμH : sigmaDist μ ∈ halfSimplex_d d := by
        refine ⟨sigmaDist_mem_simplex μ hμΔ, ?_⟩
        intro h0' hd'
        unfold sigmaDist
        have e1 : sigmaFin ⟨0, h0'⟩ = ⟨d - 1, hd⟩ := by
          apply Fin.ext; simp [sigmaFin_val]
        have e2 : sigmaFin ⟨d - 1, hd'⟩ = ⟨0, h0⟩ := by
          apply Fin.ext; simp [sigmaFin_val]
        rw [e1, e2]; exact le_of_lt hlt
      have hσinv : sSup ((fun W : Window d => quadForm W (sigmaDist μ)) '' Set.univ) =
                   sSup ((fun W : Window d => quadForm W μ) '' Set.univ) :=
        sigma_invariance μ
      calc sInf ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
                  halfSimplex_d d)
          ≤ sSup ((fun W : Window d => quadForm W (sigmaDist μ)) '' Set.univ) := by
              apply csInf_le bddBelow_quadFormSup_image_half
              exact ⟨sigmaDist μ, hσμH, rfl⟩
        _ = sSup ((fun W : Window d => quadForm W μ) '' Set.univ) := hσinv

/-- Lemma 3.4: `val(d) = inf_{μ ∈ H_d} max_W μ^T M_W μ`. -/
theorem half_simplex_cover :
    val_d d = sInf ((fun μ => sSup ((fun W : Window d => quadForm W μ) '' Set.univ)) ''
                    halfSimplex_d d) :=
  le_antisymm val_d_le_halfSimplex_inf halfSimplex_inf_le_val_d

end IntervalBnB

end -- noncomputable section
