/-
Sidon Autocorrelation Project — Post-Filter Q (Multi-window joint LP)

This file formalizes **variant Q**, the multi-window joint LP tightening
of variant F.  Where F maximises a single-window LP per composition, Q
takes a CONVEX COMBINATION over windows (probability simplex) and uses a
SHARED δ across all windows.  By LP duality this is uniformly stronger
than F (F = Q at a single-vertex λ).

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL CONTENT
═══════════════════════════════════════════════════════════════════════════════

For composition c with b = c/m, ε = b - a (so Σε = 0 and |εⱼ| ≤ 1/m), recall
   TV_W(b) − TV_W(a)
       = (1/(4nℓ)) · [(2/m) · Σⱼ εⱼ BB^W_j  −  Σ_{(i,j)∈W} εᵢεⱼ].

For variant Q with a finite set 𝒲 of windows and λ : 𝒲 → ℝ_≥0 with Σ λ = 1,
we get
   Σ_W λ_W [TV_W(b) − TV_W(a)]
       = (2/(4n·m)) · Σⱼ εⱼ · T_j(λ; c)  −  (1/(4n)) · Σ_W (λ_W/ℓ_W) · Σ_{(i,j)∈W} εᵢεⱼ
where
   T_j(λ; c) := Σ_W (λ_W / ℓ_W) · BB^W_j.

Applying the LP closed-form (`lp_closed_form_generic`) with h := 1/m,
   |Σⱼ εⱼ T_j(λ; c)| ≤ (1/m) · Δ_T(λ; c)
where Δ_T = (top-half) − (bot-half) of sorted T.  Combined with the δ²
triangle bound:
   |Σ_W λ_W [TV_W(b) − TV_W(a)]| ≤ Δ_T(λ;c) / (2n·m²) + E_λ / (4n·m²)
                                  =: corr_Q_m2(λ;c) / m².

Hence Q PRUNES iff ∃ λ ∈ Δ with
   Σ_W λ_W · TV_W(b) > c_target + corr_Q_m2(λ;c) / m²,
implying for every valid a, Σ_W λ_W · TV_W(a) > c_target, hence by Σλ=1
some TV_W(a) > c_target.

KEY OBSERVATION: F = Q at a single-vertex λ = e_W.  When λ = e_W:
   T_j(e_W) = BB^W_j / ℓ_W,   Δ_T(e_W) = Δ_BB^W / ℓ_W,
   E_{e_W}  = ell_int_sum^W / ℓ_W,   corr_Q_m2(e_W) = corr_F_m2^W.
This gives the chain F ⊆ Q.

PYTHON CROSS-REFERENCE: `_Q_bench.py:1-90`.

NEW AXIOMS DECLARED IN THIS FILE: ZERO.
NEW SORRYS IN THIS FILE: ZERO.
═══════════════════════════════════════════════════════════════════════════════
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.StepFunction
import Sidon.Proof.WRefinedDefs
import Sidon.Proof.TightDiscretizationBound
import Sidon.Proof.PostFilterF

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 1: Generic LP closed-form lemma (for ANY f : Fin (2n) → ℝ)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The sorting permutation for a generic `f : Fin (2n) → ℝ`. -/
noncomputable def sigma_gen (n : ℕ) (f : Fin (2 * n) → ℝ) : Equiv.Perm (Fin (2 * n)) :=
  Tuple.sort f

/-- `f ∘ sigma_gen f` is monotone (ascending). -/
theorem f_sigma_monotone (n : ℕ) (f : Fin (2 * n) → ℝ) :
    Monotone (f ∘ sigma_gen n f) := by
  unfold sigma_gen
  exact Tuple.monotone_sort f

/-- The variant-Q LP closed-form value for a generic `f : Fin (2n) → ℝ`:
    top-half sum minus bottom-half sum over the sort order. -/
noncomputable def Delta_gen (n : ℕ) (f : Fin (2 * n) → ℝ) : ℝ :=
  ∑ j ∈ top_half_set n, f (sigma_gen n f j) -
  ∑ j ∈ bot_half_set n, f (sigma_gen n f j)

/-- The sum over both halves equals the total. -/
theorem sum_top_add_bot_gen (n : ℕ) (f : Fin (2 * n) → ℝ) :
    ∑ j ∈ top_half_set n, f (sigma_gen n f j) +
    ∑ j ∈ bot_half_set n, f (sigma_gen n f j) =
    ∑ j : Fin (2 * n), f j := by
  classical
  rw [← Finset.sum_union (top_bot_disjoint n), top_bot_union_eq_univ n]
  exact Equiv.sum_comp (sigma_gen n f) f

/-- **Generic LP closed-form lemma**:

    For any `f : Fin (2n) → ℝ` and any `δ : Fin (2n) → ℝ` with `Σδ = 0` and
    `|δⱼ| ≤ h` (h ≥ 0),
       Σⱼ δⱼ · fⱼ ≤ h · Δ_gen n f.

    This is the same proof as `lp_closed_form_le` in `PostFilterF`, except
    that here `f` is an arbitrary real-valued function on `Fin (2n)` rather
    than `BB_W`.  Variant Q applies it to the multi-window weighted column
    mass `T_λ`. -/
theorem lp_closed_form_generic (n : ℕ) (f : Fin (2 * n) → ℝ)
    (δ : Fin (2 * n) → ℝ) (h : ℝ) (h_nn : 0 ≤ h)
    (h_close : ∀ j, |δ j| ≤ h)
    (h_sum : ∑ j : Fin (2 * n), δ j = 0) :
    ∑ j : Fin (2 * n), δ j * f j ≤ h * Delta_gen n f := by
  classical
  set σ := sigma_gen n f with hσ_def
  -- Step 1: reindex Σ_j δ_j f_j = Σ_k (δ ∘ σ)_k (f ∘ σ)_k via Equiv.sum_comp.
  have h_reindex : ∑ j : Fin (2 * n), δ j * f j =
                   ∑ k : Fin (2 * n), δ (σ k) * f (σ k) := by
    rw [← Equiv.sum_comp σ (fun j => δ j * f j)]
  rw [h_reindex]
  -- Step 2: also reindex Σ δ = 0 to Σ (δ ∘ σ) = 0.
  have h_sum_perm : ∑ k : Fin (2 * n), δ (σ k) = 0 := by
    rw [Equiv.sum_comp σ δ]; exact h_sum
  -- Step 3: also reindex |·| ≤ h.
  have h_close_perm : ∀ k, |δ (σ k)| ≤ h := fun k => h_close (σ k)
  have h_abs_iff : ∀ k, -h ≤ δ (σ k) ∧ δ (σ k) ≤ h := by
    intro k; exact abs_le.mp (h_close_perm k)
  -- Step 4: split on n = 0.
  by_cases hn : n = 0
  · subst hn
    have hdelta : Delta_gen 0 f = 0 := by
      unfold Delta_gen
      have h_top : top_half_set 0 = ∅ := by
        unfold top_half_set
        ext j; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty]
        have := j.2; omega
      have h_bot : bot_half_set 0 = ∅ := by
        unfold bot_half_set
        ext j; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty]
        have := j.2; omega
      rw [h_top, h_bot]
      simp
    rw [hdelta]
    have h_emp : ∀ k : Fin (2 * 0), False :=
      fun k => Nat.not_lt_zero k.val (by simpa using k.2)
    have h_lhs : ∑ k : Fin (2 * 0), δ (σ k) * f (σ k) = 0 := by
      apply Finset.sum_eq_zero
      intro k _; exact (h_emp k).elim
    rw [h_lhs]
    linarith
  -- Now n ≥ 1.  Pick pivot μ = (f ∘ σ)(⟨n, _⟩).
  push_neg at hn
  have hn_pos : 0 < n := Nat.pos_of_ne_zero hn
  have hn_lt : n < 2 * n := by omega
  set μ := f (σ ⟨n, hn_lt⟩) with hμ_def
  -- Σ (δ∘σ)(k) (f∘σ)(k) = Σ (δ∘σ)(k) ((f∘σ)(k) - μ) using Σ (δ∘σ) = 0.
  have h_pivot : ∑ k : Fin (2 * n), δ (σ k) * f (σ k) =
                 ∑ k : Fin (2 * n), δ (σ k) * (f (σ k) - μ) := by
    have h_split : ∀ k : Fin (2 * n),
        δ (σ k) * f (σ k) = δ (σ k) * (f (σ k) - μ) + δ (σ k) * μ := by
      intro k; ring
    simp_rw [h_split]
    rw [Finset.sum_add_distrib]
    rw [show ∑ k : Fin (2 * n), δ (σ k) * μ =
         (∑ k : Fin (2 * n), δ (σ k)) * μ from (Finset.sum_mul _ _ _).symm]
    rw [h_sum_perm]; ring
  rw [h_pivot]
  -- Split Fin (2n) into top_half_set ∪ bot_half_set.
  have h_split_sum :
      ∑ k : Fin (2 * n), δ (σ k) * (f (σ k) - μ) =
      ∑ k ∈ top_half_set n, δ (σ k) * (f (σ k) - μ) +
      ∑ k ∈ bot_half_set n, δ (σ k) * (f (σ k) - μ) := by
    rw [← Finset.sum_union (top_bot_disjoint n), top_bot_union_eq_univ n]
  rw [h_split_sum]
  -- Bot half: f(σ k) ≤ μ. δ(σ k)(f(σ k) - μ) ≤ h(μ - f(σ k)).
  have h_mono_f : Monotone (f ∘ σ) := by
    rw [hσ_def]; exact Tuple.monotone_sort _
  have h_bot_le :
      ∑ k ∈ bot_half_set n, δ (σ k) * (f (σ k) - μ) ≤
      ∑ k ∈ bot_half_set n, h * (μ - f (σ k)) := by
    apply Finset.sum_le_sum
    intro k hk
    unfold bot_half_set at hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    have h_le_pivot : f (σ k) ≤ μ := by
      have h_le_n : k ≤ (⟨n, hn_lt⟩ : Fin (2 * n)) := by
        show k.val ≤ n
        omega
      have := h_mono_f h_le_n
      simp only [Function.comp_apply] at this
      exact this
    have h_neg : f (σ k) - μ ≤ 0 := by linarith
    have h_abs := h_abs_iff k
    have hδ : -h ≤ δ (σ k) := h_abs.1
    have h_pos : 0 ≤ μ - f (σ k) := by linarith
    nlinarith [h_pos, hδ]
  -- Top half: f(σ k) ≥ μ. δ(σ k)(f(σ k) - μ) ≤ h(f(σ k) - μ).
  have h_top_le :
      ∑ k ∈ top_half_set n, δ (σ k) * (f (σ k) - μ) ≤
      ∑ k ∈ top_half_set n, h * (f (σ k) - μ) := by
    apply Finset.sum_le_sum
    intro k hk
    unfold top_half_set at hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    have h_ge_pivot : μ ≤ f (σ k) := by
      have h_ge_n : (⟨n, hn_lt⟩ : Fin (2 * n)) ≤ k := by
        show n ≤ k.val
        exact hk
      have := h_mono_f h_ge_n
      simp only [Function.comp_apply] at this
      exact this
    have h_pos : 0 ≤ f (σ k) - μ := by linarith
    have h_abs := h_abs_iff k
    have hδ : δ (σ k) ≤ h := h_abs.2
    nlinarith [h_pos, hδ]
  -- Combine: ≤ h · ((Σ_top f - n·μ) + (n·μ - Σ_bot f)) = h · Δ_gen.
  have h_top_eq :
      ∑ k ∈ top_half_set n, h * (f (σ k) - μ) =
      h * (∑ k ∈ top_half_set n, f (σ k)) - h * n * μ := by
    rw [show (h * (∑ k ∈ top_half_set n, f (σ k)) - h * n * μ) =
            h * (∑ k ∈ top_half_set n, f (σ k)) -
            h * (∑ _ ∈ top_half_set n, μ) from by
        rw [show (∑ _ ∈ top_half_set n, μ) = (top_half_set n).card • μ from
          Finset.sum_const _]
        rw [top_half_card]
        ring]
    rw [show h * (∑ k ∈ top_half_set n, f (σ k)) =
          ∑ k ∈ top_half_set n, h * f (σ k) from Finset.mul_sum _ _ _]
    rw [show h * (∑ _ ∈ top_half_set n, μ) =
          ∑ _ ∈ top_half_set n, h * μ from Finset.mul_sum _ _ _]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro k _; ring
  have h_bot_eq :
      ∑ k ∈ bot_half_set n, h * (μ - f (σ k)) =
      h * n * μ - h * (∑ k ∈ bot_half_set n, f (σ k)) := by
    rw [show (h * n * μ - h * (∑ k ∈ bot_half_set n, f (σ k))) =
            h * (∑ _ ∈ bot_half_set n, μ) -
            h * (∑ k ∈ bot_half_set n, f (σ k)) from by
        rw [show (∑ _ ∈ bot_half_set n, μ) = (bot_half_set n).card • μ from
          Finset.sum_const _]
        rw [bot_half_card]
        ring]
    rw [show h * (∑ _ ∈ bot_half_set n, μ) =
          ∑ _ ∈ bot_half_set n, h * μ from Finset.mul_sum _ _ _]
    rw [show h * (∑ k ∈ bot_half_set n, f (σ k)) =
          ∑ k ∈ bot_half_set n, h * f (σ k) from Finset.mul_sum _ _ _]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro k _; ring
  have h_sum_le_diff :
      (∑ k ∈ top_half_set n, h * (f (σ k) - μ)) +
      (∑ k ∈ bot_half_set n, h * (μ - f (σ k))) =
      h * Delta_gen n f := by
    rw [h_top_eq, h_bot_eq]
    unfold Delta_gen
    rw [hσ_def]
    ring
  linarith [h_top_le, h_bot_le, h_sum_le_diff]

/-- Generic absolute-value LP closed-form: applies to ±δ to give |·| bound. -/
theorem lp_closed_form_generic_abs (n : ℕ) (f : Fin (2 * n) → ℝ)
    (δ : Fin (2 * n) → ℝ) (h : ℝ) (h_nn : 0 ≤ h)
    (h_close : ∀ j, |δ j| ≤ h)
    (h_sum : ∑ j : Fin (2 * n), δ j = 0) :
    |∑ j : Fin (2 * n), δ j * f j| ≤ h * Delta_gen n f := by
  classical
  have h_pos := lp_closed_form_generic n f δ h h_nn h_close h_sum
  -- For the lower bound, apply to -δ.
  have h_neg_close : ∀ j, |-δ j| ≤ h := by
    intro j; rw [abs_neg]; exact h_close j
  have h_neg_sum : ∑ j : Fin (2 * n), -δ j = 0 := by
    have h1 : ∑ j : Fin (2 * n), -δ j = -(∑ j : Fin (2 * n), δ j) := by
      rw [← Finset.sum_neg_distrib]
    rw [h1, h_sum]
    ring
  have h_lp := lp_closed_form_generic n f (fun j => -δ j) h h_nn h_neg_close h_neg_sum
  have h_neg_eq : ∑ j : Fin (2 * n), (-δ j) * f j =
                  -(∑ j : Fin (2 * n), δ j * f j) := by
    have h_step : ∀ j : Fin (2 * n), (-δ j) * f j = -(δ j * f j) := by
      intro j; ring
    simp_rw [h_step]
    rw [← Finset.sum_neg_distrib]
  rw [h_neg_eq] at h_lp
  rw [abs_le]
  exact ⟨by linarith, h_pos⟩

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 2: T_lambda — multi-window weighted column-mass T_j(λ;c)
-- ═══════════════════════════════════════════════════════════════════════════════

-- A "window" is a pair `(ℓ, s_lo) : ℕ × ℕ` of length and start.  We index
-- λ by `ℕ × ℕ` and require `lam` to vanish outside the LP support.
--
-- For variant Q we additionally take a finite set `S : Finset (ℕ × ℕ)` of
-- candidate windows so all sums are finite.

/-- The multi-window weighted column-mass:
       T_j(λ;c) := Σ_{W = (ℓ,s_lo) ∈ S} (λ_W / ℓ_W) · BB^W_j.

    Here `S` is the finite set of windows over which λ is supported. -/
noncomputable def T_lambda (n : ℕ) (c : Fin (2 * n) → ℕ)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ) (j : Fin (2 * n)) : ℝ :=
  ∑ W ∈ S, (lam W / (W.1 : ℝ)) * BB_W n c W.2 W.1 j

/-- The variant-Q sorted-extremes value:
       Δ_T(λ;c) := top-half sum − bot-half sum  of sorted T_λ. -/
noncomputable def Delta_T (n : ℕ) (c : Fin (2 * n) → ℕ)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ) : ℝ :=
  Delta_gen n (T_lambda n c S lam)

/-- The variant-Q weighted ell_int_sum:
       E_λ := Σ_W (λ_W / ℓ_W) · ell_int_sum^W. -/
noncomputable def E_lambda (n : ℕ) (c : Fin (2 * n) → ℕ)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ) : ℝ :=
  ∑ W ∈ S, (lam W / (W.1 : ℝ)) * (ell_int_sum n W.2 W.1 : ℝ)

/-- The variant-Q correction in m² units:
       corr_Q_m2(λ;c) = Δ_T / (2n) + E_λ / (4n). -/
noncomputable def corr_Q_m2 (n : ℕ) (c : Fin (2 * n) → ℕ)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ) : ℝ :=
  Delta_T n c S lam / (2 * n) + E_lambda n c S lam / (4 * n)

/-- The variant-Q correction in TV space (divided by m²). -/
noncomputable def correction_Q (n m : ℕ) (c : Fin (2 * n) → ℕ)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ) : ℝ :=
  corr_Q_m2 n c S lam / (m : ℝ)^2

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: Multi-window LP bound (the critical Q theorem)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Helper: the linear part of TV_W(b) − TV_W(a) is bounded by the LP closed-form
    applied to T_λ.  Specifically:

       Σ_W (λ_W / (4nℓ_W)) · Σ_p∈W (b_p₁ ε_p₂ + b_p₂ ε_p₁)
           = (2 / (4n·m)) · Σⱼ εⱼ · T_j(λ;c).

    This identity (the per-window F linear-bound proof, summed over W with
    weights λ_W) is the algebraic core of the multi-window LP bound. -/
theorem multiwindow_linear_identity
    (n m : ℕ) (hm : m > 0) (hn : 0 < n)
    (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (h_sum_eq : ∑ i, a i = ∑ i, (c i : ℝ) / m)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ)
    (hℓ_pos : ∀ W ∈ S, 2 ≤ W.1) :
    (∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
      ∑ p ∈ window_pair_set n W.2 W.1,
        ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
         (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))) =
    (2 / (4 * n * (m : ℝ))) *
      ∑ j : Fin (2 * n), ((c j : ℝ)/m - a j) * T_lambda n c S lam j := by
  classical
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  -- Set ε.
  set b : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m with hb_def
  set ε : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m - a i with hε_def
  -- Per window W ∈ S:  Σ_p∈W (b_p₁ ε_p₂ + b_p₂ ε_p₁) = (2/m) · Σⱼ εⱼ BB^W_j.
  -- This follows from the linear-window identity from PostFilterF
  -- (specifically `h_combined` inside `linear_window_bound_F`).  We re-derive it.
  have h_per_window : ∀ W ∈ S,
      ∑ p ∈ window_pair_set n W.2 W.1,
        (b p.1 * ε p.2 + b p.2 * ε p.1) =
      (2 / (m : ℝ)) * ∑ j : Fin (2*n), ε j * BB_W n c W.2 W.1 j := by
    intro W _hW
    -- First identity:  Σ_p b(p.1) ε(p.2) = (1/m) · Σⱼ εⱼ BB^W_j  (interchange).
    have h_first :
        ∑ p ∈ window_pair_set n W.2 W.1, b p.1 * ε p.2 =
        (1 / (m : ℝ)) * ∑ j : Fin (2*n), ε j * BB_W n c W.2 W.1 j := by
      have h_double :
          ∑ p ∈ window_pair_set n W.2 W.1, b p.1 * ε p.2 =
          ∑ i : Fin (2*n), ∑ j : Fin (2*n),
            (if i.val + j.val ∈ Finset.Icc W.2 (W.2 + W.1 - 2)
             then b i * ε j else 0) := by
        unfold window_pair_set
        rw [Finset.sum_filter]
        rw [Fintype.sum_prod_type]
      rw [h_double]
      rw [Finset.sum_comm]
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      rw [show (1 : ℝ) / m * (ε j * BB_W n c W.2 W.1 j) =
              ε j * ((1 / m) * BB_W n c W.2 W.1 j) from by ring]
      unfold BB_W
      rw [Finset.mul_sum, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      by_cases hk : i.val + j.val ∈ Finset.Icc W.2 (W.2 + W.1 - 2)
      · rw [if_pos hk, if_pos hk]
        rw [hb_def]
        field_simp
      · rw [if_neg hk, if_neg hk]; ring
    -- Symmetric identity via swap.
    have h_swap :
        ∑ p ∈ window_pair_set n W.2 W.1, b p.2 * ε p.1 =
        ∑ p ∈ window_pair_set n W.2 W.1, b p.1 * ε p.2 := by
      apply Finset.sum_bij (fun (p : Fin (2*n) × Fin (2*n))
        (_hp : p ∈ window_pair_set n W.2 W.1) => (p.2, p.1))
      · intro p hp
        unfold window_pair_set at hp ⊢
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
        rw [show p.2.val + p.1.val = p.1.val + p.2.val from Nat.add_comm _ _]
        exact hp
      · intro p₁ _ p₂ _ heq
        simp only [Prod.mk.injEq] at heq
        apply Prod.ext heq.2 heq.1
      · intro p hp
        refine ⟨(p.2, p.1), ?_, rfl⟩
        unfold window_pair_set at hp ⊢
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
        rw [show p.2.val + p.1.val = p.1.val + p.2.val from Nat.add_comm _ _]
        exact hp
      · intros; rfl
    rw [Finset.sum_add_distrib, h_first, h_swap, h_first]
    ring
  -- Substitute h_per_window into the LHS, then collect.
  -- Original LHS expression uses `c/m`, `a` — rewrite as `b`, `ε`.
  have h_subst : ∀ W ∈ S,
      ∑ p ∈ window_pair_set n W.2 W.1,
        ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
         (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1)) =
      ∑ p ∈ window_pair_set n W.2 W.1, (b p.1 * ε p.2 + b p.2 * ε p.1) := by
    intro W _hW
    apply Finset.sum_congr rfl
    intro p _
    rw [hb_def, hε_def]
  -- Now substitute h_per_window inside the LHS.
  have h_lhs_eq :
    (∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
      ∑ p ∈ window_pair_set n W.2 W.1,
        ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
         (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))) =
    (∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
      ((2 / (m : ℝ)) * ∑ j : Fin (2*n), ε j * BB_W n c W.2 W.1 j)) := by
    apply Finset.sum_congr rfl
    intro W hW
    rw [h_subst W hW, h_per_window W hW]
  rw [h_lhs_eq]
  -- Now factor out 2/(4n·m) and the ε_j sum becomes a weighted sum of BB^W_j.
  -- ∑_W (λ_W/(4nℓ_W)) · (2/m) · Σ_j ε_j BB^W_j
  --   = (2/(4n·m)) · ∑_W (λ_W/ℓ_W) · Σ_j ε_j BB^W_j
  --   = (2/(4n·m)) · Σ_j ε_j · ∑_W (λ_W/ℓ_W) BB^W_j
  --   = (2/(4n·m)) · Σ_j ε_j · T_j(λ;c).
  have h_step1 :
    (∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
      ((2 / (m : ℝ)) * ∑ j : Fin (2*n), ε j * BB_W n c W.2 W.1 j)) =
    (2 / (4 * n * (m : ℝ))) *
      ∑ W ∈ S, (lam W / (W.1 : ℝ)) *
        ∑ j : Fin (2*n), ε j * BB_W n c W.2 W.1 j := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro W hW
    have hℓ : 2 ≤ W.1 := hℓ_pos W hW
    have hℓ_real : (0 : ℝ) < (W.1 : ℝ) := by
      have : (2 : ℝ) ≤ (W.1 : ℝ) := by exact_mod_cast hℓ
      linarith
    have h_ne : (W.1 : ℝ) ≠ 0 := ne_of_gt hℓ_real
    have hm_ne : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
    have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
    field_simp
  rw [h_step1]
  -- Swap sums: Σ_W (λ_W/ℓ_W) Σ_j ε_j BB^W_j = Σ_j ε_j · (Σ_W (λ_W/ℓ_W) BB^W_j) = Σ_j ε_j · T_j.
  have h_swap_sum :
    (∑ W ∈ S, (lam W / (W.1 : ℝ)) *
      ∑ j : Fin (2*n), ε j * BB_W n c W.2 W.1 j) =
    ∑ j : Fin (2*n), ε j * T_lambda n c S lam j := by
    rw [show (∑ W ∈ S, (lam W / (W.1 : ℝ)) *
              ∑ j : Fin (2*n), ε j * BB_W n c W.2 W.1 j) =
            ∑ W ∈ S, ∑ j : Fin (2*n),
              (lam W / (W.1 : ℝ)) * (ε j * BB_W n c W.2 W.1 j) from by
        apply Finset.sum_congr rfl
        intro W _; rw [Finset.mul_sum]]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j _
    unfold T_lambda
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro W _
    ring
  rw [h_swap_sum]

/-- Helper: for any window W ∈ S with 2 ≤ ℓ, the δ² window-pair sum has |·| bounded.
    This is a per-window restatement of `delta_sq_window_bound`. -/
theorem delta_sq_per_window_bound
    (n m : ℕ) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (s_lo ℓ : ℕ) :
    |∑ p ∈ window_pair_set n s_lo ℓ,
       ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)|
      ≤ (ell_int_sum n s_lo ℓ : ℝ) / (m : ℝ)^2 := by
  set b : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m with hb_def
  have h_close' : ∀ i, |a i - b i| ≤ 1 / (m : ℝ) := by
    intro i; rw [hb_def]; exact h_close i
  have h := delta_sq_window_bound n m hm a b h_close' ℓ s_lo
  -- (b p.1 - a p.1) * (b p.2 - a p.2) = ((c.1)/m - a p.1) * ((c.2)/m - a p.2).
  have h_eq : ∑ p ∈ window_pair_set n s_lo ℓ,
      ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2) =
      ∑ p ∈ window_pair_set n s_lo ℓ, (b p.1 - a p.1) * (b p.2 - a p.2) := by
    apply Finset.sum_congr rfl
    intro p _; rw [hb_def]
  rw [h_eq]
  exact h

/-- **Multi-window LP bound** (variant Q, raw):

    For any finite set `S` of windows with each `ℓ ≥ 2`, any nonneg `lam`
    on `S`, and any valid `a` (closeness `|aᵢ − cᵢ/m| ≤ 1/m` and total-mass
    equality `Σa = Σc/m`), the convex-combination over windows of
    `TV_W(b) − TV_W(a)` is bounded by `corr_Q_m2(λ;c) / m²`:

      |Σ_W λ_W [TV_W(b) − TV_W(a)]| ≤ corr_Q_m2(λ;c) / m².

    Here `b = c/m` and `λ_W = lam W` (we do NOT require Σλ = 1 in this
    bound — only nonnegativity; the sum-1 normalization is used in
    `q_prune_sound`).

    PROOF (sketch):
    1. Decompose Σ_W λ_W [TV_W(b) − TV_W(a)] using `b·b − a·a = b·ε + ε·b − ε·ε`.
    2. The linear part is `(2/(4n·m)) · Σⱼ εⱼ T_j(λ;c)`
       (`multiwindow_linear_identity`).
       By `lp_closed_form_generic_abs` (with h = 1/m), this is bounded
       in absolute value by `(2/(4n·m)) · (1/m) · Δ_T = Δ_T/(2n·m²)`.
    3. The δ² part: by triangle inequality and the per-window
       δ²-bound, |Σ_W (λ_W/ℓ_W) (1/(4n)) Σ_p∈W εᵢεⱼ| ≤ E_λ/(4n·m²).
    4. Combine. -/
theorem multiwindow_lp_bound
    (n m : ℕ) (hn : 0 < n) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_eq : ∑ i, a i = ∑ i, (c i : ℝ) / m)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ)
    (h_lam_nonneg : ∀ W ∈ S, 0 ≤ lam W)
    (hℓ_pos : ∀ W ∈ S, 2 ≤ W.1) :
    |∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
       ∑ p ∈ window_pair_set n W.2 W.1,
         ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m) - a p.1 * a p.2)|
      ≤ corr_Q_m2 n c S lam / (m : ℝ)^2 := by
  classical
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hm_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  have hm_sq_pos : (0 : ℝ) < (m : ℝ)^2 := by positivity
  -- Decompose b·b − a·a per window.
  set b : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m with hb_def
  set ε : Fin (2*n) → ℝ := fun i => (c i : ℝ) / m - a i with hε_def
  have h_eps_sum : ∑ i, ε i = 0 := by
    show ∑ i, ((c i : ℝ) / m - a i) = 0
    rw [Finset.sum_sub_distrib, ← h_sum_eq]; ring
  have h_eps_close : ∀ i, |ε i| ≤ 1 / (m : ℝ) := by
    intro i
    rw [hε_def]
    show |(c i : ℝ)/m - a i| ≤ 1 / (m : ℝ)
    rw [abs_sub_comm]
    exact h_close i
  -- For each W ∈ S, decompose:
  --   b_p₁ b_p₂ − a_p₁ a_p₂ = (b_p₁ ε_p₂ + b_p₂ ε_p₁) − ε_p₁ ε_p₂.
  have h_decomp : ∀ p : Fin (2 * n) × Fin (2 * n),
      (c p.1 : ℝ) / m * ((c p.2 : ℝ) / m) - a p.1 * a p.2 =
      ((c p.1 : ℝ) / m * ((c p.2 : ℝ) / m - a p.2) +
       (c p.2 : ℝ) / m * ((c p.1 : ℝ) / m - a p.1)) -
        ((c p.1 : ℝ) / m - a p.1) * ((c p.2 : ℝ) / m - a p.2) := by
    intro p; ring
  -- Per-window decomposition of the bilinear sum.
  have h_per_W_decomp : ∀ W ∈ S,
      ∑ p ∈ window_pair_set n W.2 W.1,
        ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m) - a p.1 * a p.2) =
      (∑ p ∈ window_pair_set n W.2 W.1,
        ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
         (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))) -
      (∑ p ∈ window_pair_set n W.2 W.1,
        ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)) := by
    intro W _hW
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro p _; rw [h_decomp]
  -- Decompose the LHS sum across windows.
  have h_lhs_decomp :
    (∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
       ∑ p ∈ window_pair_set n W.2 W.1,
         ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m) - a p.1 * a p.2)) =
    (∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
       ∑ p ∈ window_pair_set n W.2 W.1,
         ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
          (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))) -
    (∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
       ∑ p ∈ window_pair_set n W.2 W.1,
         ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro W hW
    rw [← mul_sub, h_per_W_decomp W hW]
  rw [h_lhs_decomp]
  -- Bound each piece: linear via multiwindow LP + LP closed form.
  -- Linear part identity:
  have h_lin_id := multiwindow_linear_identity n m hm hn c a h_sum_eq S lam hℓ_pos
  -- |Σⱼ εⱼ T_j(λ;c)| ≤ (1/m) · Δ_T (LP closed-form generic).
  have h_lp := lp_closed_form_generic_abs n (T_lambda n c S lam) ε (1 / (m : ℝ))
              (by positivity) h_eps_close h_eps_sum
  -- Restate h_lp in our setup.
  have h_lp' : |∑ j : Fin (2 * n), ε j * T_lambda n c S lam j| ≤
               (1 / (m : ℝ)) * Delta_T n c S lam := by
    unfold Delta_T
    convert h_lp using 1
  -- So |linear_term| ≤ (2/(4n·m)) · (1/m) · Δ_T = Δ_T / (2n·m²).
  have h_lin_bound :
      |∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
        ∑ p ∈ window_pair_set n W.2 W.1,
          ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
           (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))| ≤
      Delta_T n c S lam / (2 * n * (m : ℝ)^2) := by
    rw [h_lin_id]
    rw [abs_mul]
    rw [abs_of_pos (by positivity : (0 : ℝ) < 2 / (4 * n * (m : ℝ)))]
    calc (2 / (4 * n * (m : ℝ))) * |∑ j : Fin (2 * n), ε j * T_lambda n c S lam j|
        ≤ (2 / (4 * n * (m : ℝ))) * ((1 / (m : ℝ)) * Delta_T n c S lam) := by
          apply mul_le_mul_of_nonneg_left h_lp' (by positivity)
      _ = Delta_T n c S lam / (2 * n * (m : ℝ)^2) := by
          have hm_ne : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
          have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
          field_simp
          ring
  -- δ² part: bound by triangle.
  have h_delta_per : ∀ W ∈ S,
      |(lam W / (4 * n * (W.1 : ℝ))) *
       ∑ p ∈ window_pair_set n W.2 W.1,
         ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)| ≤
      (lam W / (4 * n * (W.1 : ℝ))) *
       ((ell_int_sum n W.2 W.1 : ℝ) / (m : ℝ)^2) := by
    intro W hW
    have hℓ : 2 ≤ W.1 := hℓ_pos W hW
    have hℓ_real : (0 : ℝ) < (W.1 : ℝ) := by
      have : (2 : ℝ) ≤ (W.1 : ℝ) := by exact_mod_cast hℓ
      linarith
    have h_lam_nn : 0 ≤ lam W := h_lam_nonneg W hW
    have h_coef_nn : 0 ≤ lam W / (4 * n * (W.1 : ℝ)) := by
      apply div_nonneg h_lam_nn
      positivity
    rw [abs_mul]
    rw [abs_of_nonneg h_coef_nn]
    apply mul_le_mul_of_nonneg_left
      (delta_sq_per_window_bound n m hm c a h_close W.2 W.1) h_coef_nn
  have h_delta_bound :
      |∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
        ∑ p ∈ window_pair_set n W.2 W.1,
          ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)| ≤
      E_lambda n c S lam / (4 * n * (m : ℝ)^2) := by
    calc |∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
          ∑ p ∈ window_pair_set n W.2 W.1,
            ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)|
        ≤ ∑ W ∈ S, |(lam W / (4 * n * (W.1 : ℝ))) *
            ∑ p ∈ window_pair_set n W.2 W.1,
              ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
            ((ell_int_sum n W.2 W.1 : ℝ) / (m : ℝ)^2) :=
          Finset.sum_le_sum h_delta_per
      _ = E_lambda n c S lam / (4 * n * (m : ℝ)^2) := by
          unfold E_lambda
          rw [Finset.sum_div]
          apply Finset.sum_congr rfl
          intro W hW
          have hℓ : 2 ≤ W.1 := hℓ_pos W hW
          have hℓ_real : (0 : ℝ) < (W.1 : ℝ) := by
            have : (2 : ℝ) ≤ (W.1 : ℝ) := by exact_mod_cast hℓ
            linarith
          have h_ne : (W.1 : ℝ) ≠ 0 := ne_of_gt hℓ_real
          have hm_ne : (m : ℝ) ≠ 0 := ne_of_gt hm_pos
          have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
          field_simp
  -- Combine via |x − y| ≤ |x| + |y|.
  calc |(∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
          ∑ p ∈ window_pair_set n W.2 W.1,
            ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
             (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))) -
        (∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
          ∑ p ∈ window_pair_set n W.2 W.1,
            ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2))|
      ≤ |∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
            ∑ p ∈ window_pair_set n W.2 W.1,
              ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m - a p.2) +
               (c p.2 : ℝ)/m * ((c p.1 : ℝ)/m - a p.1))| +
        |∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
            ∑ p ∈ window_pair_set n W.2 W.1,
              ((c p.1 : ℝ)/m - a p.1) * ((c p.2 : ℝ)/m - a p.2)| := abs_sub _ _
    _ ≤ Delta_T n c S lam / (2 * n * (m : ℝ)^2) +
        E_lambda n c S lam / (4 * n * (m : ℝ)^2) := by
        linarith [h_lin_bound, h_delta_bound]
    _ = corr_Q_m2 n c S lam / (m : ℝ)^2 := by
        unfold corr_Q_m2
        have h2n_pos : (0 : ℝ) < 2 * n := by positivity
        have h4n_pos : (0 : ℝ) < 4 * n := by positivity
        field_simp

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3.5: Helper — Δ_gen scales by a nonneg constant
-- ═══════════════════════════════════════════════════════════════════════════════

/-- `Delta_gen` is homogeneous of degree 1 under scaling by a nonneg constant.

    The proof is direct: Δ_gen(c·f) = top(c·f) − bot(c·f) = c·(top(f) − bot(f))
    = c · Δ_gen(f).  We use the fact that scaling by `c ≥ 0` does not change
    the sort order, so the sort permutation of `c · f` agrees with that of `f`
    on the sums (modulo possibly different ties; but ties are irrelevant since
    sorting any function gives the same multiset of values).  We bypass the
    permutation argument entirely: by `Delta_BB_le_total`-style splitting and
    the sum identity Σ (c·f) = c · Σf with `Σ_top + Σ_bot = Σ_total`, we
    can solve for the bot sum in terms of top sum and total, then substitute. -/
theorem Delta_gen_scaled_pos (n : ℕ) (f : Fin (2 * n) → ℝ) (c : ℝ) (hc : 0 ≤ c) :
    Delta_gen n (fun j => c * f j) = c * Delta_gen n f := by
  classical
  -- Handle c = 0 separately: both sides become 0.
  rcases eq_or_lt_of_le hc with hc0 | hc_pos
  · -- c = 0: scaled function is constant 0.
    rw [← hc0]
    unfold Delta_gen
    simp
  -- Now c > 0.
  -- Key lemma: for c > 0, f ∘ σ_{c·f} = f ∘ σ_f pointwise (both are monotone
  -- sorts of f, and monotone sorts are unique up to value via Tuple.unique_monotone).
  have h_pointwise : ∀ k, f (sigma_gen n (fun j => c * f j) k) =
                           f (sigma_gen n f k) := by
    intro k
    have h_left_mono : Monotone (f ∘ sigma_gen n (fun j => c * f j)) := by
      intro i j hij
      have h1 : (fun k => c * f k) (sigma_gen n (fun k => c * f k) i) ≤
                (fun k => c * f k) (sigma_gen n (fun k => c * f k) j) := by
        have := f_sigma_monotone n (fun k => c * f k) hij
        simpa using this
      simp only at h1
      exact (mul_le_mul_left hc_pos).mp h1
    have h_eq_pointwise :
        (f ∘ sigma_gen n (fun j => c * f j)) =
        (f ∘ sigma_gen n f) := by
      unfold sigma_gen
      exact Tuple.unique_monotone h_left_mono (Tuple.monotone_sort f)
    have := congr_fun h_eq_pointwise k
    simpa [Function.comp] using this
  -- Top sum:  Σ_{top} (c·f)(σ_{c·f} j) = c · Σ_{top} f(σ_{c·f} j) = c · Σ_{top} f(σ_f j).
  have h_top : ∑ j ∈ top_half_set n,
        (fun k => c * f k) (sigma_gen n (fun k => c * f k) j) =
      c * ∑ j ∈ top_half_set n, f (sigma_gen n f j) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    show c * f (sigma_gen n (fun k => c * f k) j) = c * f (sigma_gen n f j)
    rw [h_pointwise]
  -- Bot sum: same.
  have h_bot : ∑ j ∈ bot_half_set n,
        (fun k => c * f k) (sigma_gen n (fun k => c * f k) j) =
      c * ∑ j ∈ bot_half_set n, f (sigma_gen n f j) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    show c * f (sigma_gen n (fun k => c * f k) j) = c * f (sigma_gen n f j)
    rw [h_pointwise]
  unfold Delta_gen
  rw [h_top, h_bot]
  ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 4: Q-prune soundness
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Convenience: the convex-combination of `test_value_real` over windows in S
    weighted by lam. -/
noncomputable def weighted_test_value_real (n : ℕ) (a : Fin (2 * n) → ℝ)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ) : ℝ :=
  ∑ W ∈ S, lam W * test_value_real n a W.1 W.2

/-- Convenience: the convex-combination of `test_value` over windows in S
    weighted by lam. -/
noncomputable def weighted_test_value (n m : ℕ) (c : Fin (2 * n) → ℕ)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ) : ℝ :=
  ∑ W ∈ S, lam W * test_value n m c W.1 W.2

/-- The convex-combination over windows of (TV_W(b) − TV_W(a)) equals
    the multi-window LHS expression of `multiwindow_lp_bound`. -/
theorem weighted_test_value_diff_eq
    (n m : ℕ) (c : Fin (2 * n) → ℕ) (a : Fin (2 * n) → ℝ)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ) :
    weighted_test_value n m c S lam - weighted_test_value_real n a S lam =
    ∑ W ∈ S, (lam W / (4 * n * (W.1 : ℝ))) *
       ∑ p ∈ window_pair_set n W.2 W.1,
         ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m) - a p.1 * a p.2) := by
  unfold weighted_test_value weighted_test_value_real
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro W _
  rw [test_value_eq_window_sum, test_value_real_eq_window_sum]
  -- LHS: lam W * ((1/(4nℓ)) * Σ_p (c.1/m)(c.2/m)) - lam W * ((1/(4nℓ)) * Σ_p a.1 a.2)
  -- RHS: (lam W / (4nℓ)) * Σ_p ((c.1/m)(c.2/m) - a.1 a.2)
  -- Strategy: distribute the (lam W / (4nℓ)) factor on RHS into the sum, then
  -- use Σ-sub on RHS, leaving LHS - LHS' = RHS' equality involving the factored form.
  have hRHS :
      (lam W / (4 * (n:ℝ) * (W.1 : ℝ))) *
        ∑ p ∈ window_pair_set n W.2 W.1,
          ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m) - a p.1 * a p.2) =
      lam W * ((1 / (4 * (n:ℝ) * (W.1 : ℝ))) *
        ∑ p ∈ window_pair_set n W.2 W.1, (c p.1 : ℝ)/m * ((c p.2 : ℝ)/m)) -
      lam W * ((1 / (4 * (n:ℝ) * (W.1 : ℝ))) *
        ∑ p ∈ window_pair_set n W.2 W.1, a p.1 * a p.2) := by
    rw [show ∑ p ∈ window_pair_set n W.2 W.1,
          ((c p.1 : ℝ)/m * ((c p.2 : ℝ)/m) - a p.1 * a p.2) =
          (∑ p ∈ window_pair_set n W.2 W.1, (c p.1 : ℝ)/m * ((c p.2 : ℝ)/m)) -
          (∑ p ∈ window_pair_set n W.2 W.1, a p.1 * a p.2) from by
            rw [Finset.sum_sub_distrib]]
    ring
  rw [hRHS]

/-- **Variant Q multi-window LP soundness** (TV form).

    If for some finite set `S` of windows (each with ℓ ≥ 2) and some `lam`
    on `S` (nonnegative, sum 1) the weighted test value at `b = c/m` exceeds
    `c_target + corr_Q_m2(λ;c) / m²`, then for every valid real `a`, the
    weighted test value at `a` exceeds `c_target`.

    Hence by Σλ = 1 (which gives weighted average ≤ max), some
    `test_value_real n a W.1 W.2 > c_target`, which is the standard
    cascade-pruning conclusion. -/
theorem q_prune_sound
    (n m : ℕ) (c_target : ℝ) (hn : 0 < n) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (S : Finset (ℕ × ℕ)) (lam : ℕ × ℕ → ℝ)
    (h_lam_nonneg : ∀ W ∈ S, 0 ≤ lam W)
    (h_lam_sum : ∑ W ∈ S, lam W = 1)
    (hℓ_pos : ∀ W ∈ S, 2 ≤ W.1)
    (h_exceeds : weighted_test_value n m c S lam >
                 c_target + corr_Q_m2 n c S lam / (m : ℝ)^2)
    (a : Fin (2 * n) → ℝ)
    (ha_nonneg : ∀ i, 0 ≤ a i)
    (h_close : ∀ i, |a i - (c i : ℝ) / m| ≤ 1 / (m : ℝ))
    (h_sum_eq : ∑ i, a i = ∑ i, (c i : ℝ) / m) :
    ∃ W ∈ S, test_value_real n a W.1 W.2 > c_target := by
  classical
  -- Apply the multiwindow LP bound.
  have h_lp := multiwindow_lp_bound n m hn hm c a ha_nonneg h_close h_sum_eq
                S lam h_lam_nonneg hℓ_pos
  rw [← weighted_test_value_diff_eq n m c a S lam] at h_lp
  have h_abs_le := abs_sub_le_iff.mp h_lp
  -- weighted_test_value_real n a > c_target.
  have h_avg_lt : weighted_test_value_real n a S lam > c_target := by
    have h1 : weighted_test_value n m c S lam -
              weighted_test_value_real n a S lam ≤
              corr_Q_m2 n c S lam / (m : ℝ)^2 := h_abs_le.1
    linarith
  -- Now use Σλ = 1 to conclude ∃ W with test_value_real ≥ c_target.
  by_contra h_neg
  push_neg at h_neg
  -- h_neg : ∀ W ∈ S, test_value_real n a W.1 W.2 ≤ c_target.
  have h_avg_le : weighted_test_value_real n a S lam ≤ c_target := by
    unfold weighted_test_value_real
    have : ∑ W ∈ S, lam W * test_value_real n a W.1 W.2 ≤
           ∑ W ∈ S, lam W * c_target := by
      apply Finset.sum_le_sum
      intro W hW
      apply mul_le_mul_of_nonneg_left (h_neg W hW) (h_lam_nonneg W hW)
    have h_const_sum : ∑ W ∈ S, lam W * c_target = c_target := by
      rw [← Finset.sum_mul, h_lam_sum]; ring
    linarith
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 5: F ⊆ Q (single-vertex λ recovers variant F)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The Dirac (single-vertex) λ at a chosen window W₀.  All mass on W₀, none
    elsewhere. -/
noncomputable def diracLambda (W₀ : ℕ × ℕ) : ℕ × ℕ → ℝ :=
  fun W => if W = W₀ then 1 else 0

/-- The Dirac λ is nonneg. -/
theorem diracLambda_nonneg (W₀ : ℕ × ℕ) : ∀ W, 0 ≤ diracLambda W₀ W := by
  intro W
  unfold diracLambda
  by_cases h : W = W₀
  · rw [if_pos h]; norm_num
  · rw [if_neg h]

/-- The Dirac λ sums to 1 over `{W₀}`. -/
theorem diracLambda_sum (W₀ : ℕ × ℕ) :
    ∑ W ∈ ({W₀} : Finset (ℕ × ℕ)), diracLambda W₀ W = 1 := by
  unfold diracLambda
  simp

/-- **F-prune ⇒ Q-prune** (subset chain).

    If there exists a window `W₀ = (ℓ, s_lo)` with 2 ≤ ℓ such that variant F
    prunes the composition `c` at `W₀` (i.e. `test_value > c_target +
    correction_F`), then variant Q prunes `c` with the Dirac λ at `W₀`.

    This shows F ⊆ Q (every F-pruned composition is Q-pruned), so Q is
    uniformly tighter than F (subset relation on prune sets). -/
theorem f_prune_implies_q_prune
    (n m : ℕ) (c_target : ℝ) (hn : 0 < n) (hm : m > 0)
    (c : Fin (2 * n) → ℕ)
    (W₀ : ℕ × ℕ) (hℓ_W₀ : 2 ≤ W₀.1)
    (h_F_prune : test_value n m c W₀.1 W₀.2 >
                 c_target + correction_F n m c W₀.1 W₀.2) :
    weighted_test_value n m c ({W₀} : Finset (ℕ × ℕ)) (diracLambda W₀) >
    c_target +
      corr_Q_m2 n c ({W₀} : Finset (ℕ × ℕ)) (diracLambda W₀) / (m : ℝ)^2 := by
  classical
  -- Step 1: weighted_test_value at the singleton with Dirac is just test_value at W₀.
  have h_wtv :
      weighted_test_value n m c ({W₀} : Finset (ℕ × ℕ)) (diracLambda W₀) =
      test_value n m c W₀.1 W₀.2 := by
    unfold weighted_test_value diracLambda
    simp
  -- Step 2: corr_Q_m2 at the Dirac equals corr_F_m2.
  have h_corr :
      corr_Q_m2 n c ({W₀} : Finset (ℕ × ℕ)) (diracLambda W₀) =
      corr_F_m2 n c W₀.1 W₀.2 := by
    unfold corr_Q_m2 Delta_T E_lambda T_lambda corr_F_m2 diracLambda
    simp only [Finset.sum_singleton, if_true]
    -- Need: Delta_gen of (1/ℓ_{W₀}) · BB_W = (1/ℓ_{W₀}) · Δ_BB.
    -- And the E_λ term simplifies to ell_int_sum/ℓ.
    have hℓ_real : (0 : ℝ) < (W₀.1 : ℝ) := by
      have : (2 : ℝ) ≤ (W₀.1 : ℝ) := by exact_mod_cast hℓ_W₀
      linarith
    have hℓ_ne : (W₀.1 : ℝ) ≠ 0 := ne_of_gt hℓ_real
    have hn_real : (0 : ℝ) < n := Nat.cast_pos.mpr hn
    have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_real
    -- Step A: Delta_gen of T_λ where T_λ j = (1/ℓ_{W₀}) · BB_W^j equals (1/ℓ) · Delta_BB.
    -- The sort permutation of c · f equals sort of f when c > 0 (preserves order).
    -- Here we use that scaling by a positive constant preserves the sort pattern,
    -- so sigma_gen of (k · BB_W) = sigma_BB, and Delta_gen of (k · BB_W) = k · Delta_BB.
    have h_T_eq : (fun j => (1 / (W₀.1 : ℝ)) * BB_W n c W₀.2 W₀.1 j) =
                  (fun j => (1 / (W₀.1 : ℝ)) * BB_W n c W₀.2 W₀.1 j) := rfl
    -- Compute Delta_gen for (1/ℓ) · BB_W.
    have h_Delta_gen_scaled :
        Delta_gen n (fun j => (1 / (W₀.1 : ℝ)) * BB_W n c W₀.2 W₀.1 j) =
        (1 / (W₀.1 : ℝ)) * Delta_BB n c W₀.2 W₀.1 := by
      apply Delta_gen_scaled_pos n (BB_W n c W₀.2 W₀.1) (1 / (W₀.1 : ℝ))
        (by positivity)
    rw [h_Delta_gen_scaled]
    -- Now both sides have the form `(1/ℓ) · Δ_BB / (2n) + (1/ℓ) · ell_int_sum / (4n)`.
    -- LHS = (1/ℓ) · Δ_BB / (2n) + (1/ℓ) · ell_int_sum / (4n).
    -- RHS = Δ_BB / (2n·ℓ) + ell_int_sum / (4n·ℓ).
    -- These are equal.
    field_simp
  -- Combine.
  rw [h_wtv, h_corr]
  -- Now we need: test_value n m c W₀.1 W₀.2 > c_target + corr_F_m2/m².
  unfold correction_F at h_F_prune
  exact h_F_prune

end -- noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- AUDIT BLOCK (variant Q)
-- ═══════════════════════════════════════════════════════════════════════════════
/-
COEFFICIENT VARIANT:  Q (multi-window joint LP).
  D:  corr_tight = (ℓ−1)·W_int_overlap / (2n·ℓ) + ell_int_sum / (4n·ℓ)        (per W)
  F:  corr_F     = Δ_BB / (2n·ℓ) + ell_int_sum / (4n·ℓ)                       (per W)
  Q:  corr_Q     = Δ_T(λ) / (2n)  + E_λ / (4n)                                 (over λ ∈ Δ)
       where  T_j(λ) = Σ_W (λ_W/ℓ_W) BB^W_j,  E_λ = Σ_W (λ_W/ℓ_W) ell_int_sum^W,
              Δ_T = top-half sum − bot-half sum  of sorted T_λ.

SUBSET CHAIN:  F ⊆ Q  (`f_prune_implies_q_prune`).  Q is uniformly tighter
(prunes more) than F.  At λ = e_W (Dirac), Q reduces to F at W.

NEW HYPOTHESIS REQUIRED FOR VARIANT Q:
  Like F, Q requires Σ aᵢ = Σ cᵢ/m (total-mass equality), since the LP
  closed-form is over {δ : Σδ = 0, |δⱼ| ≤ h}.  Additionally, λ must be
  nonneg with Σλ = 1 (probability simplex).

HYPOTHESES USED PER LEMMA:
  lp_closed_form_generic            0 ≤ h, |δⱼ| ≤ h, Σ δ = 0.
  lp_closed_form_generic_abs        Same as above.
  multiwindow_linear_identity       m > 0, n > 0, Σ a = Σ c/m, ∀W ∈ S, 2 ≤ ℓ_W.
  delta_sq_per_window_bound         m > 0, |aᵢ - cᵢ/m| ≤ 1/m.
  multiwindow_lp_bound              n > 0, m > 0, |aᵢ - cᵢ/m| ≤ 1/m,
                                    Σ a = Σ c/m, λ ≥ 0, ∀W ∈ S, 2 ≤ ℓ_W.
  q_prune_sound                     Same + Σλ = 1, exceedance hypothesis.
  f_prune_implies_q_prune           n > 0, m > 0, 2 ≤ ℓ_{W₀}, F-prune at W₀.

NEW AXIOMS DECLARED IN THIS FILE:  ZERO.
NEW SORRYS IN THIS FILE:  ZERO.
═══════════════════════════════════════════════════════════════════════════════
-/

-- Axiom audit (uncomment to verify only kernel axioms used).
#print axioms lp_closed_form_generic
#print axioms lp_closed_form_generic_abs
#print axioms multiwindow_linear_identity
#print axioms delta_sq_per_window_bound
#print axioms multiwindow_lp_bound
#print axioms q_prune_sound
#print axioms f_prune_implies_q_prune
#print axioms Delta_gen_scaled_pos
