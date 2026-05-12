/-
Sidon Autocorrelation Project — Step Function and Grid Convolution

Step function definition, basic properties (nonneg, support, integrability, integral),
and the key lemma that convolution at grid points equals scaled discrete autoconvolution.

Fine grid (C&S B_{n,m}): d = 2n bins, heights a_i = c_i/m, ∑c_i = 4nm.
-/

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
-- Step Function and Grid Convolution
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Test value computed from continuous bin masses (for comparison). -/
noncomputable def test_value_continuous (n : ℕ) (f : ℝ → ℝ) (ℓ s_lo : ℕ) : ℝ :=
  let d := 2 * n
  let a : Fin d → ℝ := fun i => (4 * n : ℝ) * bin_masses f n i
  let conv := discrete_autoconvolution a
  let sum_conv := ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), conv k
  (1 / (4 * n * ℓ : ℝ)) * sum_conv

/-- Step function on the 2n-bin grid (fine grid, C&S B_{n,m}).
    Height in bin i = c_i / m, where c_i are integer coordinates summing to 4nm.
    Integral: ∑ (c_i/m)·(1/(4n)) = (∑c_i)/(4nm) = 4nm/(4nm) = 1. -/
noncomputable def step_function (n m : ℕ) (c : Fin (2 * n) → ℕ) : ℝ → ℝ :=
  fun x =>
    let d := 2 * n
    let δ := 1 / (4 * n : ℝ)
    if x < -1/4 ∨ x ≥ 1/4 then 0
    else
      let i := ⌊(x + 1/4) / δ⌋.toNat
      if h : i < d then (c ⟨i, h⟩ : ℝ) / m
      else 0

-- Helper: step function is nonneg
lemma step_function_nonneg (n m : ℕ) (hm : m > 0) (c : Fin (2 * n) → ℕ) :
    ∀ x, 0 ≤ step_function n m c x := by
  intros x
  simp [step_function];
  split_ifs <;> positivity

-- Helper: step function support ⊆ Ico(-1/4, 1/4)
lemma step_function_support (n m : ℕ) (c : Fin (2 * n) → ℕ) :
    Function.support (step_function n m c) ⊆ Set.Ico (-1/4 : ℝ) (1/4) := by
  intro x hx; unfold step_function at hx; aesop;

-- Helper: step function is integrable
lemma step_function_integrable (n m : ℕ) (c : Fin (2 * n) → ℕ) :
    MeasureTheory.Integrable (step_function n m c) MeasureTheory.volume := by
  -- step_function is bounded and supported on [-1/4, 1/4), hence integrable
  -- Strategy: bound ‖step_function‖ ≤ C · 1_{[-1/4, 1/4)}, integrate the bound
  set C := (∑ i : Fin (2 * n), (c i : ℝ)) / m with hC_def
  have hC_nn : 0 ≤ C := div_nonneg (Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have h_bounded : ∀ x, ‖step_function n m c x‖ ≤ C := by
    intro x
    rw [Real.norm_eq_abs]; simp only [step_function]
    split_ifs with h1 h2
    · simp; exact hC_nn
    · rw [abs_of_nonneg (by positivity)]
      exact div_le_div_of_nonneg_right
        (by exact_mod_cast Finset.single_le_sum (fun a _ => Nat.zero_le (c a)) (Finset.mem_univ _))
        (Nat.cast_nonneg _)
    · simp; exact hC_nn
  have h_vanish : ∀ x, x ∉ Set.Ico (-1/4 : ℝ) (1/4) → step_function n m c x = 0 := by
    intro x hx; simp only [Set.mem_Ico, not_and_or, not_le, not_lt] at hx
    simp only [step_function]; rcases hx with hx | hx
    · exact if_pos (Or.inl (by linarith))
    · exact if_pos (Or.inr (by linarith))
  apply MeasureTheory.Integrable.mono' (g := fun x => C * Set.indicator (Set.Ico (-1/4 : ℝ) (1/4)) (fun _ => 1) x)
  · exact (MeasureTheory.integrable_indicator_iff measurableSet_Ico |>.2 (by norm_num)).const_mul _
  · exact (measurable_step_function n m c).aestronglyMeasurable
  · exact MeasureTheory.ae_of_all _ fun x => by
      by_cases hx : x ∈ Set.Ico (-1/4 : ℝ) (1/4)
      · rw [Set.indicator_of_mem hx, mul_one]; exact h_bounded x
      · rw [h_vanish x hx, norm_zero]
        exact mul_nonneg hC_nn (Set.indicator_nonneg (fun _ _ => zero_le_one) x)
where
  measurable_step_function (n m : ℕ) (c : Fin (2 * n) → ℕ) :
      Measurable (step_function n m c) := by
    unfold step_function; dsimp only []
    refine Measurable.ite ?_ measurable_const ?_
    · exact measurableSet_Iio.union measurableSet_Ici
    · -- else branch = (lookup : ℕ → ℝ) ∘ (Int.toNat : ℤ → ℕ) ∘ (⌊·⌋ : ℝ → ℤ) ∘ (linear : ℝ → ℝ)
      exact (measurable_from_top (f := fun i : ℕ =>
          if h : i < 2 * n then (↑(c ⟨i, h⟩) : ℝ) / ↑m else 0)).comp
        ((measurable_from_top (f := Int.toNat)).comp
          ((Int.measurable_floor (R := ℝ)).comp
            (Measurable.div_const (Measurable.add_const measurable_id (1 / 4 : ℝ))
              (1 / (4 * (↑n : ℝ))))))

-- Helper: integral of step function = 1
lemma integral_step_function (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = 4 * n * m) :
    ∫ x, step_function n m c x = 1 := by
  -- The integral over ℝ equals the integral over [-1/4, 1/4) since step_function = 0 outside
  have h_restrict : ∫ x, step_function n m c x = ∫ x in Set.Ico (-1 / 4 : ℝ) (1 / 4), step_function n m c x := by
    rw [ MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero ] ; unfold step_function ; aesop;
  -- Each bin contributes c_i / m * (1/(4n))
  have h_const : ∀ i : Fin (2 * n), ∫ x in Set.Ico (-1 / 4 + (i : ℝ) / (4 * n)) (-1 / 4 + (i + 1) / (4 * n)), step_function n m c x = (c i : ℝ) / m * (1 / (4 * n)) := by
    intro i
    have h_const_interval : ∀ x ∈ Set.Ico (-1 / 4 + (i : ℝ) / (4 * n)) (-1 / 4 + (i + 1) / (4 * n)), step_function n m c x = (c i : ℝ) / m := by
      unfold step_function;
      field_simp;
      intro x hx; split_ifs <;> simp_all +decide ;
      · cases ‹_› <;> nlinarith [ show ( i : ℝ ) + 1 ≤ 2 * n by norm_cast; linarith [ Fin.is_lt i ], div_mul_cancel₀ ( -n + ( i : ℝ ) ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), div_mul_cancel₀ ( -n + ( i + 1 : ℝ ) ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ) ] ;
      · have h4n_pos : (4 * n : ℝ) > 0 := by positivity
        rw [ div_le_iff₀ h4n_pos, lt_div_iff₀ h4n_pos ] at hx
        have h_floor : ⌊(↑n * (4 * x + 1) : ℝ)⌋ = (i : ℤ) := by
          rw [Int.floor_eq_iff]
          constructor <;> · push_cast; nlinarith [hx.1, hx.2]
        have h_idx : ⌊(↑n * (4 * x + 1) : ℝ)⌋.toNat = i.val := by
          rw [h_floor]; exact Int.toNat_natCast _
        have h_c : ∀ h_lt, c (⟨⌊(↑n * (4 * x + 1) : ℝ)⌋.toNat, h_lt⟩ : Fin (2 * n)) = c i :=
          fun _ => congrArg c (Fin.ext h_idx)
        simp only [h_c]; field_simp
      · rw [ Int.le_floor ] at * ; norm_num at * ; nlinarith [ ( by norm_cast : ( 1 :ℝ ) ≤ n ), mul_div_cancel₀ ( -n + ( i + 1 ) :ℝ ) ( by positivity : ( 4 * n :ℝ ) ≠ 0 ) ] ;
    rw [ MeasureTheory.setIntegral_congr_fun measurableSet_Ico h_const_interval ]
    rw [MeasureTheory.setIntegral_const]
    simp only [MeasureTheory.Measure.real, Real.volume_Ico, smul_eq_mul]
    have h4n_pos : (0 : ℝ) < 4 * n := by positivity
    have h_width : -1 / 4 + ((i : ℝ) + 1) / (4 * n) - (-1 / 4 + (i : ℝ) / (4 * n)) = 1 / (4 * n) := by
      field_simp; ring
    have h_nn : (0 : ℝ) ≤ -1 / 4 + ((i : ℝ) + 1) / (4 * ↑n) - (-1 / 4 + (i : ℝ) / (4 * ↑n)) := by
      have h_pos : (0 : ℝ) < 1 / (4 * ↑n) := by positivity
      linarith [h_width]
    rw [ENNReal.toReal_ofReal h_nn, h_width]
    ring
  -- Split integral into sum over bins
  have h_split : ∫ x in Set.Ico (-1 / 4 : ℝ) (1 / 4), step_function n m c x = ∑ i : Fin (2 * n), ∫ x in Set.Ico (-1 / 4 + (i : ℝ) / (4 * n)) (-1 / 4 + (i + 1) / (4 * n)), step_function n m c x := by
    rw [ ← MeasureTheory.integral_biUnion_finset ];
    · congr with x ; norm_num [ Finset.mem_univ ];
      constructor <;> intro hx;
      · refine' ⟨ ⟨ ⌊ ( 1 / 4 + x ) * ( 4 * n ) ⌋₊, _ ⟩, _, _ ⟩ <;> norm_num;
        · rw [ Nat.floor_lt ] <;> norm_num <;> nlinarith [ show ( n : ℝ ) ≥ 1 by norm_cast ];
        · rw [ div_le_iff₀ ] <;> nlinarith [ Nat.floor_le ( show 0 ≤ ( 1 / 4 + x ) * ( 4 * n ) by nlinarith [ show ( n : ℝ ) ≥ 1 by norm_cast ] ), show ( n : ℝ ) ≥ 1 by norm_cast ];
        · rw [ lt_div_iff₀ ] <;> first | positivity | linarith [ Nat.lt_floor_add_one ( ( 1 / 4 + x ) * ( 4 * n ) ) ] ;
      · obtain ⟨ i, hi₁, hi₂ ⟩ := hx; constructor <;> nlinarith [ show ( i : ℝ ) + 1 ≤ 2 * n by norm_cast; linarith [ Fin.is_lt i ], div_mul_cancel₀ ( ( i : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), div_mul_cancel₀ ( ( i + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ) ] ;
    · exact fun _ _ => measurableSet_Ico;
    · intros i hi j hj hij; exact Set.disjoint_left.mpr fun x hx₁ hx₂ => hij <| Fin.ext <| Nat.le_antisymm ( Nat.le_of_lt_succ <| by { rw [ ← @Nat.cast_lt ℝ ] ; push_cast; nlinarith [ hx₁.1, hx₁.2, hx₂.1, hx₂.2, show ( n : ℝ ) > 0 by positivity, mul_div_cancel₀ ( ( i : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( j : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( i + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( j + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ) ] } ) ( Nat.le_of_lt_succ <| by { rw [ ← @Nat.cast_lt ℝ ] ; push_cast; nlinarith [ hx₁.1, hx₁.2, hx₂.1, hx₂.2, show ( n : ℝ ) > 0 by positivity, mul_div_cancel₀ ( ( i : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( j : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( i + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( j + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ) ] } ) ;
    · intro i hi; specialize h_const i; contrapose! h_const; rw [ MeasureTheory.integral_undef h_const ] ; ring; norm_num [ hn.ne', hm.ne' ] ;
      intro H; simp_all +decide;
      exact h_const <| MeasureTheory.Integrable.integrableOn <| step_function_integrable n m c;
  -- Now sum the bin contributions: ∑ (c_i / m) * (1/(4n)) = (∑ c_i) / (4nm) = 1
  rw [h_restrict, h_split]
  simp only [fun i => h_const i]
  have hm_ne : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hn_ne : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have h4n_ne : (4 * (n : ℝ)) ≠ 0 := by positivity
  -- ∑ (c_i/m * (1/(4n))) = (∑ c_i) / (4nm) = 1
  have h_term : ∀ i : Fin (2 * n), (c i : ℝ) / ↑m * (1 / (4 * ↑n)) = (c i : ℝ) / (↑m * (4 * ↑n)) := by
    intro i; ring
  simp_rw [h_term, ← Finset.sum_div]
  rw [show (∑ i : Fin (2 * n), (c i : ℝ)) = (4 * ↑n * ↑m : ℝ) from by exact_mod_cast hc]
  field_simp

/-- The convolution of the step function with itself, evaluated at
    grid point y_k, equals (1/(4n) · (1/m)²) · conv_int[k].
    Grid point: y_k = -1/2 + (k+1)·δ where δ = 1/(4n).
    With fine-grid heights a_i = c_i/m: (g*g)(y_k) = δ · ∑_{i+j=k} a_i·a_j.
    So (g*g)(y_k) = (1/(4n)) · discrete_autoconvolution(a, k)
                   = (1/(4nm²)) · ∑_{i+j=k} c_i·c_j. -/
lemma convolution_at_grid_point (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = 4 * n * m) (k : ℕ) :
    MeasureTheory.convolution (step_function n m c) (step_function n m c)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
      (-1/2 + (↑k + 1) * (1 / (4 * ↑n))) =
    (1 / (4 * (n : ℝ)) / (m : ℝ)^2) *
      discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ)) k := by
  set δ := (1 : ℝ) / (4 * ↑n) with hδ_def
  set y := (-1 : ℝ) / 2 + (↑k + 1) * δ with hy_def
  simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
  have h_const : ∀ i : Fin (2 * n), ∀ x ∈ Set.Ico (-(1/4:ℝ) + (i : ℝ) / (4 * n)) (-(1/4:ℝ) + ((i : ℝ) + 1) / (4 * n)), step_function n m c x = (c i : ℝ) / m := by
    unfold step_function
    field_simp
    intro i x hx; split_ifs <;> simp_all +decide
    · cases ‹_› <;> nlinarith [show (i : ℝ) + 1 ≤ 2 * n by norm_cast; linarith [Fin.is_lt i], div_mul_cancel₀ (-n + (i : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0), div_mul_cancel₀ (-n + (i + 1 : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0)]
    · have h4n_pos : (4 * n : ℝ) > 0 := by positivity
      rw [ div_le_iff₀ h4n_pos, lt_div_iff₀ h4n_pos ] at hx
      have h_floor : ⌊(↑n * (4 * x + 1) : ℝ)⌋ = (i : ℤ) := by
          rw [Int.floor_eq_iff]
          constructor <;> · push_cast; nlinarith [hx.1, hx.2]
      have h_idx : ⌊(↑n * (4 * x + 1) : ℝ)⌋.toNat = i.val := by
        rw [h_floor]; exact Int.toNat_natCast _
      have h_c : ∀ h_lt, c (⟨⌊(↑n * (4 * x + 1) : ℝ)⌋.toNat, h_lt⟩ : Fin (2 * n)) = c i :=
        fun _ => congrArg c (Fin.ext h_idx)
      simp only [h_c]; field_simp
    · rw [Int.le_floor] at *; norm_num at *; nlinarith [(by norm_cast : (1 : ℝ) ≤ n), mul_div_cancel₀ (-n + (i + 1) : ℝ) (by positivity : (4 * n : ℝ) ≠ 0)]
  have h_n_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have h_m_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have h_δ_pos : (0 : ℝ) < δ := by positivity
  have h_const_rev : ∀ (i : Fin (2 * n)),
      ∀ t ∈ Set.Ioo (-(1/4:ℝ) + (i : ℝ) / (4 * n)) (-(1/4:ℝ) + ((i : ℝ) + 1) / (4 * n)),
      step_function n m c (y - t) =
      if h : i.val ≤ k ∧ k - i.val < 2 * n then
        (c ⟨k - i.val, h.2⟩ : ℝ) / m
      else 0 := by
    intro i t ht
    have ht1 := ht.1
    have ht2 := ht.2
    have h4n_pos : (4 * n : ℝ) > 0 := by positivity
    have h4n_ne : (4 * n : ℝ) ≠ 0 := ne_of_gt h4n_pos
    have hyt_upper : y - t < -(1/4 : ℝ) + ((↑k - ↑i : ℝ) + 1) / (4 * n) := by
      have h1 : t > -(1/4 : ℝ) + (↑i : ℝ) / (4 * n) := ht1
      have hkey : -(1/4 : ℝ) + ((↑k - ↑i : ℝ) + 1) / (4 * n) - (y - t) =
                  t - (-(1/4 : ℝ) + (↑i : ℝ) / (4 * n)) := by
        simp only [hy_def, hδ_def]; field_simp; ring
      linarith
    have hyt_lower : y - t > -(1/4 : ℝ) + (↑k - ↑i : ℝ) / (4 * n) := by
      have h2 : t < -(1/4 : ℝ) + ((↑i : ℝ) + 1) / (4 * n) := ht2
      have hkey : (y - t) - (-(1/4 : ℝ) + (↑k - ↑i : ℝ) / (4 * n)) =
                  (-(1/4 : ℝ) + ((↑i : ℝ) + 1) / (4 * n)) - t := by
        simp only [hy_def, hδ_def]; field_simp; ring
      linarith
    split_ifs with hik
    · obtain ⟨hle, hlt⟩ := hik
      have hki_nat : (k - i.val : ℝ) = (↑(k - i.val) : ℝ) := by
        rw [Nat.cast_sub hle]
      apply h_const ⟨k - i.val, hlt⟩ (y - t)
      simp only []
      constructor
      · rw [hki_nat] at hyt_lower; linarith
      · rw [hki_nat] at hyt_upper
        have : ((↑(k - i.val) : ℝ) + 1) = (↑(k - i.val) + 1 : ℝ) := by ring
        linarith
    · push_neg at hik
      simp only [step_function]
      by_cases hle : i.val ≤ k
      · have h2n_le : 2 * n ≤ k - i.val := hik hle
        have : y - t ≥ 1/4 := by
          have hcast : (↑k - ↑i.val : ℝ) ≥ 2 * n := by
            rw [← Nat.cast_sub hle]; exact_mod_cast h2n_le
          have hdiv : (↑k - ↑i.val : ℝ) / (4 * n) ≥ 1 / 2 := by
            rw [ge_iff_le, le_div_iff₀ h4n_pos]
            linarith
          have : y - t > -(1/4 : ℝ) + (↑k - ↑i.val : ℝ) / (4 * n) :=
            hyt_lower
          linarith
        exact if_pos (Or.inr (by linarith))
      · push_neg at hle
        have : (↑k : ℝ) - (↑i.val : ℝ) + 1 ≤ 0 := by
          have : (↑i.val : ℝ) ≥ ↑k + 1 := by exact_mod_cast hle
          linarith
        have : y - t < -(1/4 : ℝ) := by
          have : (↑k - ↑i.val : ℝ) + 1 ≤ 0 := this
          have hbound : -(1/4 : ℝ) + ((↑k - ↑i.val : ℝ) + 1) / (4 * n) ≤ -(1/4 : ℝ) := by
            have : ((↑k - ↑i.val : ℝ) + 1) / (4 * n) ≤ 0 := div_nonpos_of_nonpos_of_nonneg (by linarith) (by positivity)
            linarith
          linarith
        exact if_pos (Or.inl (by linarith))
  have h_prod_on_Ioo : ∀ (i : Fin (2 * n)),
      ∀ t ∈ Set.Ioo (-(1/4:ℝ) + (i : ℝ) / (4 * n)) (-(1/4:ℝ) + ((i : ℝ) + 1) / (4 * n)),
      step_function n m c t * step_function n m c (y - t) =
      if h : i.val ≤ k ∧ k - i.val < 2 * n then
        (c i : ℝ) / m * ((c ⟨k - i.val, h.2⟩ : ℝ) / m)
      else 0 := by
    intro i t ht
    rw [h_const i t (Set.Ioo_subset_Ico_self ht), h_const_rev i t ht]
    split_ifs <;> ring
  have h_bin_contrib : ∀ (i : Fin (2 * n)),
      ∫ t in Set.Ico (-(1/4:ℝ) + (i : ℝ) / (4 * n)) (-(1/4:ℝ) + ((i : ℝ) + 1) / (4 * n)),
        step_function n m c t * step_function n m c (y - t) =
      if h : i.val ≤ k ∧ k - i.val < 2 * n then
        (c i : ℝ) / m * ((c ⟨k - i.val, h.2⟩ : ℝ) / m) * δ
      else 0 := by
    intro i
    set a := -(1/4:ℝ) + (i : ℝ) / (4 * n) with ha_def
    set b := -(1/4:ℝ) + ((i : ℝ) + 1) / (4 * n) with hb_def
    have hab : a < b := by
      simp only [ha_def, hb_def]
      have h2 : (1 : ℝ) / (4 * n) > 0 := by positivity
      linarith [show ((↑i : ℝ) + 1) / (4 * n) = (↑i : ℝ) / (4 * n) + 1 / (4 * n) by ring]
    have hab_le : a ≤ b := le_of_lt hab
    set val := if h : i.val ≤ k ∧ k - i.val < 2 * n then
        (c i : ℝ) / m * ((c ⟨k - i.val, h.2⟩ : ℝ) / m)
      else 0 with hval_def
    have h_ae : ∀ᵐ t ∂(MeasureTheory.volume.restrict (Set.Ico a b)), step_function n m c t * step_function n m c (y - t) = val := by
      rw [MeasureTheory.ae_restrict_iff' measurableSet_Ico]
      have h_singleton_null : MeasureTheory.volume ({a} : Set ℝ) = 0 := Real.volume_singleton
      have h_aenull : ∀ᵐ t ∂MeasureTheory.volume, t ≠ a := by
        rw [MeasureTheory.ae_iff]
        convert h_singleton_null using 2
        ext t; simp
      filter_upwards [h_aenull] with t hta
      intro ht_Ico
      have ht_ioo : t ∈ Set.Ioo a b :=
        ⟨lt_of_le_of_ne ht_Ico.1 (Ne.symm hta), ht_Ico.2⟩
      rw [hval_def]
      exact h_prod_on_Ioo i t ht_ioo
    have h_int_eq : ∫ t in Set.Ico a b, step_function n m c t * step_function n m c (y - t) = val * (b - a) := by
      have h_ae_unres := (MeasureTheory.ae_restrict_iff' measurableSet_Ico).mp h_ae
      calc ∫ t in Set.Ico a b, step_function n m c t * step_function n m c (y - t)
          = ∫ _ in Set.Ico a b, val := MeasureTheory.setIntegral_congr_ae measurableSet_Ico h_ae_unres
        _ = val * (b - a) := by
            rw [MeasureTheory.setIntegral_const]
            simp [MeasureTheory.Measure.real, Real.volume_Ico, ENNReal.toReal_ofReal (by linarith : 0 ≤ b - a), smul_eq_mul, mul_comm]
    rw [h_int_eq]
    have hba : b - a = δ := by simp [ha_def, hb_def, hδ_def]; field_simp; ring
    rw [hba]
    simp only [hval_def]
    split_ifs <;> ring
  have h_restrict : ∫ t, step_function n m c t * step_function n m c (y - t) =
      ∫ t in Set.Ico (-(1/4:ℝ)) (1/4), step_function n m c t * step_function n m c (y - t) := by
    rw [MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero]
    intro t ht; simp only [Set.mem_Ico, not_and_or, not_le, not_lt] at ht
    have : step_function n m c t = 0 := by
      simp only [step_function]
      rcases ht with ht | ht
      · exact if_pos (Or.inl (by linarith))
      · exact if_pos (Or.inr (by linarith))
    simp [this]
  have h_split : ∫ t in Set.Ico (-(1/4:ℝ)) (1/4), step_function n m c t * step_function n m c (y - t) =
      ∑ i : Fin (2 * n), ∫ t in Set.Ico (-(1/4:ℝ) + (i : ℝ) / (4 * n)) (-(1/4:ℝ) + ((i : ℝ) + 1) / (4 * n)),
        step_function n m c t * step_function n m c (y - t) := by
    rw [← MeasureTheory.integral_biUnion_finset]
    · congr with x; norm_num [Finset.mem_univ]
      constructor <;> intro hx
      · refine' ⟨⟨⌊(1/4 + x) * (4 * n)⌋₊, _⟩, _, _⟩ <;> norm_num
        · rw [Nat.floor_lt] <;> norm_num <;> nlinarith [show (n : ℝ) ≥ 1 by norm_cast]
        · rw [div_le_iff₀] <;> nlinarith [Nat.floor_le (show 0 ≤ (1/4 + x) * (4 * n) by nlinarith [show (n : ℝ) ≥ 1 by norm_cast]), show (n : ℝ) ≥ 1 by norm_cast]
        · rw [lt_div_iff₀] <;> first | positivity | linarith [Nat.lt_floor_add_one ((1/4 + x) * (4 * n))]
      · obtain ⟨i, hi₁, hi₂⟩ := hx; constructor <;> nlinarith [show (i : ℝ) + 1 ≤ 2 * n by norm_cast; linarith [Fin.is_lt i], div_mul_cancel₀ ((i : ℝ) : ℝ) (by positivity : (4 * n : ℝ) ≠ 0), div_mul_cancel₀ ((i + 1 : ℝ) : ℝ) (by positivity : (4 * n : ℝ) ≠ 0)]
    · exact fun _ _ => measurableSet_Ico
    · intros i hi j hj hij; exact Set.disjoint_left.mpr fun x hx₁ hx₂ => hij <| Fin.ext <| Nat.le_antisymm (Nat.le_of_lt_succ <| by { rw [← @Nat.cast_lt ℝ]; push_cast; nlinarith [hx₁.1, hx₁.2, hx₂.1, hx₂.2, show (n : ℝ) > 0 by positivity, mul_div_cancel₀ ((i : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0), mul_div_cancel₀ ((j : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0), mul_div_cancel₀ ((i + 1 : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0), mul_div_cancel₀ ((j + 1 : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0)] }) (Nat.le_of_lt_succ <| by { rw [← @Nat.cast_lt ℝ]; push_cast; nlinarith [hx₁.1, hx₁.2, hx₂.1, hx₂.2, show (n : ℝ) > 0 by positivity, mul_div_cancel₀ ((i : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0), mul_div_cancel₀ ((j : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0), mul_div_cancel₀ ((i + 1 : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0), mul_div_cancel₀ ((j + 1 : ℝ)) (by positivity : (4 * n : ℝ) ≠ 0)] })
    · intro i _
      have h1 := step_function_integrable n m c
      have h2 : MeasureTheory.Integrable (fun t => step_function n m c (y - t)) MeasureTheory.volume :=
        h1.comp_sub_left y
      have h_bdd : ∀ x, ‖step_function n m c x‖ ≤ ↑(4 * n * m) / ↑m := by
        intro x
        rw [Real.norm_eq_abs, abs_of_nonneg (step_function_nonneg n m hm c x)]
        simp only [step_function]
        split_ifs with h1 h2
        · positivity
        · apply div_le_div_of_nonneg_right _ (by positivity : (0 : ℝ) ≤ ↑m)
          exact_mod_cast (hc ▸ Finset.single_le_sum (fun a _ => Nat.zero_le (c a)) (Finset.mem_univ _) : c _ ≤ 4 * n * m)
        · positivity
      exact (h2.bdd_mul' h1.aestronglyMeasurable
        (MeasureTheory.ae_of_all _ (fun x => h_bdd x))).integrableOn
  rw [h_restrict, h_split, Finset.sum_congr rfl fun i _ => h_bin_contrib i]
  unfold discrete_autoconvolution
  simp only [hδ_def]
  have h_inner : ∀ i : Fin (2 * n),
      ∑ j : Fin (2 * n), (if i.val + j.val = k then (c i : ℝ) * (c j : ℝ) else 0) =
      if h : i.val ≤ k ∧ k - i.val < 2 * n then (c i : ℝ) * (c ⟨k - i.val, h.2⟩ : ℝ) else 0 := by
    intro i
    split_ifs with hik
    · obtain ⟨hle, hlt⟩ := hik
      have : ∑ j : Fin (2 * n), (if i.val + j.val = k then (c i : ℝ) * (c j : ℝ) else 0) =
        ∑ j : Fin (2 * n), (if j = ⟨k - i.val, hlt⟩ then (c i : ℝ) * (c j : ℝ) else 0) := by
        congr 1; ext j
        have : (i.val + j.val = k) ↔ (j = ⟨k - i.val, hlt⟩) := by
          constructor
          · intro h; exact Fin.ext (by simp; omega)
          · intro h; have := congr_arg Fin.val h; simp at this; omega
        simp [this]
      rw [this]
      simp [Finset.sum_ite_eq']
    · push_neg at hik
      apply Finset.sum_eq_zero
      intro j _
      split_ifs with hij
      · exfalso
        by_cases hle : i.val ≤ k
        · have := hik hle
          have : j.val = k - i.val := by omega
          exact absurd (by omega : k - i.val < 2 * n) (by omega)
        · omega
      · rfl
  simp only [h_inner]
  rw [Finset.mul_sum]
  congr 1; ext i; split_ifs with h
  · -- Goal: (c i / m) * ((c ⟨k-i, _⟩) / m) * (1/(4*n)) = 1/(4*n) / m^2 * ((c i) * (c ⟨k-i, _⟩))
    field_simp
  · simp

end -- noncomputable section
