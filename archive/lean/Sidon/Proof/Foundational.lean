import Mathlib
import Sidon.Defs

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
-- Foundational Lemmas (F1–F15)
-- Fine grid: S = 4nm, heights a_i = c_i/m.
-- ═══════════════════════════════════════════════════════════════════════════════

-- F1: c_i = D(i+1) - D(i) rewrite
theorem canonical_discretization_eq (f : ℝ → ℝ) (n m : ℕ) (i : Fin (2 * n)) :
    canonical_discretization f n m i =
    if i.1 + 1 < 2 * n then
      canonical_cumulative_distribution f n m (i.1 + 1) - canonical_cumulative_distribution f n m i.1
    else
      4 * n * m - canonical_cumulative_distribution f n m i.1 := by
        unfold canonical_discretization canonical_cumulative_distribution;
        simp +zetaDelta at *

-- F2: D(0) = 0
theorem canonical_cumulative_distribution_zero (f : ℝ → ℝ) (n m : ℕ) :
    canonical_cumulative_distribution f n m 0 = 0 := by
      unfold canonical_cumulative_distribution; aesop;

-- F3: D(2n) = S = 4nm (fine-grid boundary condition)
theorem canonical_cumulative_distribution_2n (f : ℝ → ℝ) (n m : ℕ) (_hn : n > 0) (_hm : m > 0)
    (h_mass_pos : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0) :
    canonical_cumulative_distribution f n m (2 * n) = 4 * n * m := by
      unfold canonical_cumulative_distribution; dsimp only
      have h1 : (∑ j : Fin (2 * n), if j.1 < 2 * n then bin_masses f n j else 0) =
                ∑ j : Fin (2 * n), bin_masses f n j :=
        Finset.sum_congr rfl fun j _ => if_pos j.isLt
      simp only [h1, div_self h_mass_pos, one_mul]
      rw [Int.floor_natCast, Int.natAbs_natCast]

-- F4: Bin masses ≥ 0 for f ≥ 0
theorem bin_masses_nonneg (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x) (n : ℕ) (i : Fin (2 * n)) :
    0 ≤ bin_masses f n i := by
      apply_rules [ MeasureTheory.integral_nonneg, Set.indicator_nonneg ] ; aesop

-- F6: c_i = D(i+1) - D(i) (alt hypothesis, given D(2n) = 4nm)
theorem canonical_discretization_eq_diff (f : ℝ → ℝ) (n m : ℕ)
    (h_D_2n : canonical_cumulative_distribution f n m (2 * n) = 4 * n * m) (i : Fin (2 * n)) :
    canonical_discretization f n m i = canonical_cumulative_distribution f n m (i.1 + 1) - canonical_cumulative_distribution f n m i.1 := by
      convert canonical_discretization_eq f n m i using 1;
      grind

-- F8: D is monotone for f ≥ 0
theorem canonical_cumulative_distribution_mono (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x) (n m : ℕ) :
    Monotone (canonical_cumulative_distribution f n m) := by
      have h_floor_nonneg : ∀ k l : ℕ, k ≤ l → ⌊(∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0) / (∑ j : Fin (2 * n), bin_masses f n j) * (4 * n * m)⌋ ≤ ⌊(∑ j : Fin (2 * n), if j.1 < l then bin_masses f n j else 0) / (∑ j : Fin (2 * n), bin_masses f n j) * (4 * n * m)⌋ := by
        intros k l hkl
        have h_sum_le : (∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0) ≤ (∑ j : Fin (2 * n), if j.1 < l then bin_masses f n j else 0) := by
          exact Finset.sum_le_sum fun i _ => by split_ifs <;> linarith [ show 0 ≤ bin_masses f n i from by exact MeasureTheory.integral_nonneg fun x => by exact Set.indicator_nonneg ( fun x hx => hf_nonneg x ) _ ] ;
        gcongr; exact Finset.sum_nonneg fun _ _ => bin_masses_nonneg f hf_nonneg n _
      intro k l hkl; specialize h_floor_nonneg k l hkl; simp_all +decide [ canonical_cumulative_distribution ] ;
      rw [ ← Int.ofNat_le, Int.natAbs_of_nonneg ( Int.floor_nonneg.mpr _ ), Int.natAbs_of_nonneg ( Int.floor_nonneg.mpr _ ) ];
      · convert h_floor_nonneg using 1;
      · exact mul_nonneg ( div_nonneg ( Finset.sum_nonneg fun _ _ => by split_ifs <;> [ exact bin_masses_nonneg f hf_nonneg n _ ; norm_num ] ) ( Finset.sum_nonneg fun _ _ => bin_masses_nonneg f hf_nonneg n _ ) ) ( by positivity );
      · exact mul_nonneg ( div_nonneg ( Finset.sum_nonneg fun _ _ => by split_ifs <;> [ exact bin_masses_nonneg f hf_nonneg n _ ; exact le_rfl ] ) ( Finset.sum_nonneg fun _ _ => bin_masses_nonneg f hf_nonneg n _ ) ) ( by positivity )

-- F9: ∑ c_i = telescope form
theorem canonical_discretization_sum_eq_telescope (f : ℝ → ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (h_mass_pos : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0) :
    ∑ i : Fin (2 * n), canonical_discretization f n m i =
    ∑ i : Fin (2 * n), (canonical_cumulative_distribution f n m (i + 1) - canonical_cumulative_distribution f n m i) := by
      convert Finset.sum_congr rfl fun i _ => canonical_discretization_eq_diff f n m (canonical_cumulative_distribution_2n f n m hn hm h_mass_pos) i

-- Nat telescoping sum (fixed from exact? gap)
theorem sum_fin_telescope_nat (f : ℕ → ℕ) (n : ℕ) (h_mono : Monotone f) :
    ∑ i : Fin n, (f (i + 1) - f i) = f n - f 0 := by
      have h_telescope : ∀ (n : ℕ), ∑ i ∈ Finset.range n, (f (i + 1) - f i) = f n - f 0 := by
        intro n
        induction n with
        | zero => simp
        | succ k ih =>
          rw [Finset.sum_range_succ, ih]
          have h1 : f 0 ≤ f k := h_mono (Nat.zero_le k)
          have h2 : f k ≤ f (k + 1) := h_mono (Nat.le_succ k)
          omega
      rw [ ← h_telescope, Finset.sum_range ]

-- F15: ∑ c_i = 4nm (full proof, positive mass, fine grid)
-- canonical_discretization rounds to S = 4nm quanta (fine grid).
theorem canonical_discretization_sum_eq_m (f : ℝ → ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (h_mass_pos : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0)
    (hf_nonneg : ∀ x, 0 ≤ f x) :
    ∑ i : Fin (2 * n), canonical_discretization f n m i = 4 * n * m := by
      rw [canonical_discretization_sum_eq_telescope f n m hn hm h_mass_pos]
      rw [sum_fin_telescope_nat (canonical_cumulative_distribution f n m) (2 * n) (canonical_cumulative_distribution_mono f hf_nonneg n m)]
      rw [canonical_cumulative_distribution_2n f n m hn hm h_mass_pos]
      rw [canonical_cumulative_distribution_zero]
      simp

end -- noncomputable section
