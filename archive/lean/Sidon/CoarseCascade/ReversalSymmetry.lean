/-
Sidon Autocorrelation Project — Reversal Symmetry for Coarse Grid (Proof Stubs)

For the coarse cascade, canonical form c <= rev(c) lexicographically halves
the search space. This is sound because TV_W is invariant under reversal
of the mass vector (with corresponding window index reflection).

Source: run_cascade_coarse.py L0 uses generate_canonical_compositions_batched.
-/

import Sidon.Proof.CoarseCascade

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-- Reversal of a vector: rev(mu)_i = mu_{d-1-i}. -/
def vec_reverse {d : ℕ} (μ : Fin d → ℝ) : Fin d → ℝ :=
  fun i => μ ⟨d - 1 - i.val, by omega⟩

/-- Reversal is involutive: rev(rev(μ)) = μ. -/
lemma vec_reverse_involutive {d : ℕ} (μ : Fin d → ℝ) :
    vec_reverse (vec_reverse μ) = μ := by
  funext i
  unfold vec_reverse
  congr 1
  apply Fin.ext
  have hi : i.val < d := i.isLt
  show d - 1 - (d - 1 - i.val) = i.val
  omega

-- =============================================================================
-- PART 1: Autoconvolution Under Reversal
-- =============================================================================

/-- Discrete autoconvolution is symmetric under reversal with index reflection:
    DA(rev(mu), k) = DA(mu, 2(d-1) - k).

    This follows from the substitution i' = (d-1-i), j' = (d-1-j),
    so i'+j' = 2(d-1) - (i+j).

    HYPOTHESIS: We need k ≤ 2(d-1) for the index reflection to be well-defined
    in ℕ. (Outside this range LHS is 0 but RHS is the convolution at 0
    via Nat truncation, generally nonzero.) The d=0 case is vacuous (Fin 0 empty). -/
theorem autoconv_reverse {d : ℕ} (μ : Fin d → ℝ) (k : ℕ)
    (hk : k ≤ 2 * (d - 1)) :
    discrete_autoconvolution (vec_reverse μ) k =
    discrete_autoconvolution μ (2 * (d - 1) - k) := by
  -- d = 0 case: Fin 0 is empty so both sides vanish.
  rcases Nat.eq_zero_or_pos d with hd0 | hd
  · subst hd0
    unfold discrete_autoconvolution
    simp
  -- d ≥ 1: bijection σ : i ↦ ⟨d-1-i.val, _⟩.
  unfold discrete_autoconvolution vec_reverse
  -- Define the involutive bijection σ on Fin d.
  set σ : Fin d → Fin d := fun i => ⟨d - 1 - i.val, by have := i.isLt; omega⟩ with hσ_def
  have hσ_inv : ∀ i : Fin d, σ (σ i) = i := by
    intro i
    apply Fin.ext
    show d - 1 - (d - 1 - i.val) = i.val
    have := i.isLt
    omega
  -- Re-index outer sum via σ, then inner sum via σ.
  -- Strategy: show the LHS sum equals the canonical form ∑ μ_i μ_j · 1[i+j = 2(d-1)-k]
  -- by reindexing both i and j via σ.
  refine Finset.sum_nbij' σ σ ?_ ?_ ?_ ?_ ?_
  · intro a _; exact Finset.mem_univ _
  · intro a _; exact Finset.mem_univ _
  · intro a _; exact hσ_inv a
  · intro a _; exact hσ_inv a
  · intro i _
    refine Finset.sum_nbij' σ σ ?_ ?_ ?_ ?_ ?_
    · intro a _; exact Finset.mem_univ _
    · intro a _; exact Finset.mem_univ _
    · intro a _; exact hσ_inv a
    · intro a _; exact hσ_inv a
    · intro j _
      -- Original goal (after reindexing):
      --   (if i.val + j.val = k then μ ⟨d-1-i.val, _⟩ * μ ⟨d-1-j.val, _⟩ else 0)
      -- =
      --   (if (σ i).val + (σ j).val = 2(d-1)-k then μ (σ i) * μ (σ j) else 0)
      -- Using (σ i).val = d-1-i.val:
      --   (σ i).val + (σ j).val = (d-1-i.val) + (d-1-j.val).
      -- For i.val ≤ d-1 and j.val ≤ d-1:
      --   (d-1-i.val) + (d-1-j.val) = 2(d-1) - k ↔ i.val + j.val = k.
      -- And μ (σ i) = μ ⟨d-1-i.val, _⟩ = LHS body (similarly for j).
      have hi : i.val ≤ d - 1 := by have := i.isLt; omega
      have hj : j.val ≤ d - 1 := by have := j.isLt; omega
      -- σ i = ⟨d - 1 - i.val, _⟩, so (σ i).val = d - 1 - i.val.
      have hσ_i : (σ i).val = d - 1 - i.val := rfl
      have hσ_j : (σ j).val = d - 1 - j.val := rfl
      have h_iff : (i.val + j.val = k) ↔
                   ((σ i).val + (σ j).val = 2 * (d - 1) - k) := by
        rw [hσ_i, hσ_j]
        omega
      -- The body μ (σ i) * μ (σ j) on the RHS equals μ ⟨d-1-i.val, _⟩ * μ ⟨d-1-j.val, _⟩
      -- by definition of σ.
      by_cases hcond : i.val + j.val = k
      · rw [if_pos hcond, if_pos (h_iff.mp hcond)]
      · rw [if_neg hcond, if_neg (fun h => hcond (h_iff.mpr h))]

-- =============================================================================
-- PART 2: TV Under Reversal
-- =============================================================================

/-- TV_W is invariant under reversal (with reflected window):
    TV(rev(mu), ell, s) = TV(mu, ell, 2(d-1) - (s + ell - 2))

    HYPOTHESES:
    - hd : 0 < d  (else d-1 underflows in ℕ; vacuously true for d=0)
    - hℓ : 2 ≤ ell  (so the window is well-formed: s + ell - 2 ≥ s)
    - hs : s + ell ≤ 2 * d  (else the index reflection underflows in ℕ;
      indices outside this range give zero contributions on both sides anyway) -/
theorem tv_reverse_invariant {d : ℕ} (μ : Fin d → ℝ) (ell s : ℕ)
    (hd : 0 < d) (hℓ : 2 ≤ ell) (hs : s + ell ≤ 2 * d) :
    mass_test_value d (vec_reverse μ) ell s =
    mass_test_value d μ ell (2 * (d - 1) - (s + ell - 2)) := by
  unfold mass_test_value
  -- The new window endpoint: (2*(d-1) - (s+ell-2)) + ell - 2 = 2*(d-1) - s.
  have h_window_lo : 2 * (d - 1) - (s + ell - 2) + ell - 2 = 2 * (d - 1) - s := by
    omega
  rw [h_window_lo]
  congr 1
  -- Show ∑_{k ∈ Icc s (s+ell-2)} DA(vec_reverse μ) k
  --    = ∑_{k' ∈ Icc (2*(d-1) - (s+ell-2)) (2*(d-1) - s)} DA μ k'.
  -- Step 1: Pointwise apply autoconv_reverse to convert LHS into ∑ DA μ (2(d-1)-k).
  have h_step1 :
      (∑ k ∈ Finset.Icc s (s + ell - 2), discrete_autoconvolution (vec_reverse μ) k)
      = (∑ k ∈ Finset.Icc s (s + ell - 2), discrete_autoconvolution μ (2 * (d - 1) - k)) := by
    apply Finset.sum_congr rfl
    intro k hk
    rw [Finset.mem_Icc] at hk
    have hk_le : k ≤ 2 * (d - 1) := by omega
    exact autoconv_reverse μ k hk_le
  rw [h_step1]
  -- Step 2: Re-index k ↦ k' := 2(d-1) - k. Bijection between
  -- Icc s (s+ell-2) and Icc (2(d-1)-(s+ell-2)) (2(d-1)-s).
  -- Bijection φ : k ↦ 2(d-1) - k.
  refine Finset.sum_nbij' (fun k => 2 * (d - 1) - k) (fun k => 2 * (d - 1) - k)
    ?_ ?_ ?_ ?_ ?_
  · intro k hk
    simp only [Finset.mem_Icc] at hk ⊢
    refine ⟨?_, ?_⟩ <;> omega
  · intro k hk
    simp only [Finset.mem_Icc] at hk ⊢
    refine ⟨?_, ?_⟩ <;> omega
  · intro k hk
    simp only [Finset.mem_Icc] at hk
    show 2 * (d - 1) - (2 * (d - 1) - k) = k
    omega
  · intro k hk
    simp only [Finset.mem_Icc] at hk
    show 2 * (d - 1) - (2 * (d - 1) - k) = k
    omega
  · intro k _
    -- discrete_autoconvolution μ (2(d-1) - k) on both sides — definitionally equal.
    rfl

/-- The maximum over all windows is the same for mu and rev(mu).

    HYPOTHESIS: 0 < d (else Fin 0 is empty and `vec_reverse` is trivial).

    Both directions use the equation `tv_reverse_invariant`. The existential
    constrains `s + ell ≤ 2 * d` so the index reflection is well-defined.
    For `s + ell > 2 * d`, the convolution at indices > 2(d-1) is zero;
    the cascade pruning logic only ever queries `s + ell ≤ 2 * d`. -/
theorem max_tv_reverse_eq {d : ℕ} (μ : Fin d → ℝ) (c_target : ℝ) (hd : 0 < d) :
    (∃ ell s, 2 ≤ ell ∧ s + ell ≤ 2 * d ∧
      mass_test_value d μ ell s ≥ c_target) ↔
    (∃ ell s, 2 ≤ ell ∧ s + ell ≤ 2 * d ∧
      mass_test_value d (vec_reverse μ) ell s ≥ c_target) := by
  constructor
  · rintro ⟨ell, s, hℓ, hs, h_tv⟩
    refine ⟨ell, 2 * (d - 1) - (s + ell - 2), hℓ, ?_, ?_⟩
    · omega
    · -- Apply tv_reverse_invariant with vec_reverse μ in place of μ
      -- and use vec_reverse_involutive: vec_reverse (vec_reverse μ) = μ.
      have h_eq := tv_reverse_invariant (vec_reverse μ) ell s hd hℓ hs
      rw [vec_reverse_involutive] at h_eq
      -- h_eq : mass_test_value d μ ell s
      --        = mass_test_value d (vec_reverse μ) ell (2*(d-1) - (s+ell-2))
      linarith
  · rintro ⟨ell, s, hℓ, hs, h_tv⟩
    refine ⟨ell, 2 * (d - 1) - (s + ell - 2), hℓ, ?_, ?_⟩
    · omega
    · -- Apply tv_reverse_invariant directly:
      -- mass_test_value d (vec_reverse μ) ell s
      --   = mass_test_value d μ ell (2*(d-1) - (s+ell-2))
      have h_eq := tv_reverse_invariant μ ell s hd hℓ hs
      linarith

-- =============================================================================
-- PART 3: Canonical Form Soundness
-- =============================================================================

/-- **Canonical form:** restricting enumeration to c <= rev(c) lexicographically
    does not miss any unpruned compositions. Every composition is either
    canonical or its reversal is canonical (or both, if palindromic).

    Source: run_cascade_coarse.py uses _canonical_mask from pruning.py.

    HYPOTHESIS: 0 < d (vec_reverse needs d-1 well-defined). The existentials
    are constrained to `s + ell ≤ 2 * d`. -/
theorem canonical_form_complete {d : ℕ} (S : ℕ) (hd : 0 < d)
    (c : Fin d → ℕ) (_hc_sum : ∑ i, c i = S) (c_target : ℝ) :
    (∃ ell s, 2 ≤ ell ∧ s + ell ≤ 2 * d ∧
      mass_test_value d (fun i => (c i : ℝ) / (S : ℝ)) ell s ≥ c_target) →
    (∃ ell s, 2 ≤ ell ∧ s + ell ≤ 2 * d ∧
      mass_test_value d (fun i => (vec_reverse (fun j => (c j : ℝ) / (S : ℝ))) i) ell s ≥
        c_target) := by
  intro h
  exact (max_tv_reverse_eq (fun i => (c i : ℝ) / (S : ℝ)) c_target hd).mp h

end -- noncomputable section
