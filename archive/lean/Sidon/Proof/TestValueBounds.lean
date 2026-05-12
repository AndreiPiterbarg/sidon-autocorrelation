/-
Sidon Autocorrelation Project — Test Value Bounds

eLpNorm bounds at grid points, test value ≤ autoconvolution ratio,
and continuous test value ≤ ratio (Fubini set-integral chain).
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.CauchySchwarz
import Sidon.Proof.AsymmetryBound
import Sidon.Proof.StepFunction

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
-- Test Value Bounds (Section 18b)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Sub-lemma: For any nonneg integrable function g on ℝ, the L∞ norm
    (essential supremum) is ≥ g(x) for any x where g is continuous. -/
lemma eLpNorm_top_ge_of_continuous_at (g : ℝ → ℝ)
    (hg_nn : ∀ x, 0 ≤ g x) (_hg_int : MeasureTheory.Integrable g)
    (x₀ : ℝ) (hg_cont : ContinuousAt g x₀)
    (h_fin : MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume ≠ ⊤) :
    (MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume).toReal ≥ g x₀ := by
  by_contra h_lt; push_neg at h_lt
  set M := (MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume).toReal with hM_def
  have h_fin' : MeasureTheory.eLpNormEssSup g MeasureTheory.volume ≠ ⊤ := by
    rwa [← MeasureTheory.eLpNorm_exponent_top]
  have h_ae_le : ∀ᵐ x ∂MeasureTheory.volume, g x ≤ M := by
    filter_upwards [MeasureTheory.enorm_ae_le_eLpNormEssSup g MeasureTheory.volume] with x hx
    rw [← ENNReal.toReal_ofReal (hg_nn x)]
    rw [hM_def, MeasureTheory.eLpNorm_exponent_top]
    exact ENNReal.toReal_mono h_fin' (le_trans (by rw [Real.enorm_eq_ofReal (hg_nn x)]) hx)
  have h_zero : MeasureTheory.volume {x | M < g x} = 0 := by
    have h_ae_neg : ∀ᵐ x ∂MeasureTheory.volume, ¬(M < g x) :=
      h_ae_le.mono fun x hx h => not_le.mpr h hx
    rw [MeasureTheory.ae_iff] at h_ae_neg
    convert h_ae_neg using 2
    ext x; simp
  obtain ⟨δ, hδ_pos, hball⟩ := Metric.continuousAt_iff.mp hg_cont (g x₀ - M) (by linarith)
  have h_sub : Metric.ball x₀ δ ⊆ {x | M < g x} := by
    intro x hx; simp only [Set.mem_setOf_eq]
    have := hball hx; rw [Real.dist_eq] at this; linarith [(abs_lt.mp this).1]
  have h_pos : 0 < MeasureTheory.volume (Metric.ball x₀ δ) :=
    Metric.isOpen_ball.measure_pos MeasureTheory.volume ⟨x₀, Metric.mem_ball_self hδ_pos⟩
  exact absurd (le_antisymm (le_of_le_of_eq (MeasureTheory.measure_mono h_sub) h_zero) (zero_le _)) (ne_of_gt h_pos)

/-- The sum of all bin masses equals 1 (for normalized f). -/
lemma sum_bin_masses_eq_one (n : ℕ) (hn : n > 0) (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    ∑ i : Fin (2 * n), bin_masses f n i = 1 := by
  convert hf_int using 1;
  have h_sum_eq_integral : ∑ i : Fin (2 * n), MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ico (-(1 / 4 : ℝ) + i.val * (1 / (4 * n : ℝ))) (-(1 / 4 : ℝ) + (i.val + 1) * (1 / (4 * n : ℝ)))) f) = ∫ x in Set.Ico (-(1 / 4 : ℝ)) (-(1 / 4 : ℝ) + 2 * n * (1 / (4 * n : ℝ))), f x := by
    have h_sum_eq_integral : ∑ i : Fin (2 * n), ∫ x in Set.Ico (-(1 / 4 : ℝ) + i.val * (1 / (4 * n : ℝ))) (-(1 / 4 : ℝ) + (i.val + 1) * (1 / (4 * n : ℝ))), f x = ∫ x in Set.Ico (-(1 / 4 : ℝ)) (-(1 / 4 : ℝ) + 2 * n * (1 / (4 * n : ℝ))), f x := by
      have h_sum_eq_integral : ∀ m : ℕ, ∑ i ∈ Finset.range m, ∫ x in Set.Ico (-(1 / 4 : ℝ) + i * (1 / (4 * n : ℝ))) (-(1 / 4 : ℝ) + (i + 1) * (1 / (4 * n : ℝ))), f x = ∫ x in Set.Ico (-(1 / 4 : ℝ)) (-(1 / 4 : ℝ) + m * (1 / (4 * n : ℝ))), f x := by
        intro m
        induction' m with m ih;
        · norm_num;
        · rw [ Finset.sum_range_succ, ih, Nat.cast_succ, add_mul, one_mul, ← MeasureTheory.setIntegral_union ] <;> norm_num;
          · rw [ Set.Ico_union_Ico_eq_Ico ] <;> ring <;> norm_num [ hn ];
          · exact MeasureTheory.Integrable.integrableOn ( MeasureTheory.integrable_of_integral_eq_one hf_int );
          · exact MeasureTheory.Integrable.integrableOn ( MeasureTheory.integrable_of_integral_eq_one hf_int );
      simpa [ Finset.sum_range ] using h_sum_eq_integral ( 2 * n );
    convert h_sum_eq_integral using 2;
    rw [ MeasureTheory.integral_indicator ( measurableSet_Ico ) ];
  convert h_sum_eq_integral using 1;
  rw [ MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero ] ; ring_nf ; norm_num [ hn.ne' ];
  exact fun x hx => Classical.not_not.1 fun hx' => by have := hf_supp hx'; exact this.2.not_ge <| hx <| by linarith [ this.1 ] ;

/-- The step function is ContinuousAt at every point not equal to a bin boundary. -/
private lemma step_function_continuousAt (n m : ℕ) (hn : n > 0)
    (c : Fin (2 * n) → ℕ) (x : ℝ)
    (hx : ∀ k : Fin (2 * n + 1), x ≠ -(1/4 : ℝ) + ↑k.val / (4 * ↑n)) :
    ContinuousAt (step_function n m c) x := by
  have h_n_pos : (0 : ℝ) < ↑n := Nat.cast_pos.mpr hn
  suffices h : ∀ᶠ y in nhds x, step_function n m c y = step_function n m c x from
    continuousAt_const.congr (h.mono fun _ hy => hy.symm)
  by_cases hx_lt : x < -(1/4 : ℝ)
  · exact Filter.eventually_of_mem (IsOpen.mem_nhds isOpen_Iio hx_lt) fun y hy => by
      show step_function n m c y = step_function n m c x
      have h1 : y < -(1/4 : ℝ) ∨ y ≥ 1/4 := Or.inl hy
      have h2 : x < -(1/4 : ℝ) ∨ x ≥ 1/4 := Or.inl hx_lt
      simp only [step_function, neg_div, h1, h2, ↓reduceIte]
  · push_neg at hx_lt
    by_cases hx_ge : x ≥ (1/4 : ℝ)
    · have hx_gt : x > 1/4 := by
        rcases eq_or_lt_of_le hx_ge with h_eq | h_gt
        · exfalso
          have h_bnd := hx ⟨2 * n, by omega⟩
          apply h_bnd
          rw [← h_eq]; push_cast
          have : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.ne_zero_of_lt (Nat.zero_lt_of_lt hn))
          field_simp; norm_num
        · exact h_gt
      exact Filter.eventually_of_mem (IsOpen.mem_nhds isOpen_Ioi hx_gt) fun y hy => by
        show step_function n m c y = step_function n m c x
        have h1 : y < -(1/4 : ℝ) ∨ y ≥ 1/4 := Or.inr (le_of_lt hy)
        have h2 : x < -(1/4 : ℝ) ∨ x ≥ 1/4 := Or.inr hx_ge
        simp only [step_function, neg_div, h1, h2, ↓reduceIte]
    · push_neg at hx_ge
      have hx_lo : -(1/4 : ℝ) < x := by
        rcases eq_or_lt_of_le hx_lt with h_eq | h_lt
        · exfalso; have := hx ⟨0, by omega⟩; apply this; simp [← h_eq]
        · exact h_lt
      set δ := (1 : ℝ) / (4 * ↑n) with hδ_def
      have hδ_pos : (0 : ℝ) < δ := by rw [hδ_def]; positivity
      set α := (x + 1/4) / δ with hα_def
      have hα_pos : 0 < α := div_pos (by linarith) hδ_pos
      have h4n_ne : (4 * (↑n : ℝ)) ≠ 0 := by positivity
      have hα_ne_int : ∀ z : ℤ, α ≠ ↑z := by
        intro z hz
        have hx_eq : x = -(1/4 : ℝ) + (z : ℝ) / (4 * ↑n) := by
          rw [hα_def, hδ_def] at hz
          field_simp [h4n_ne] at hz ⊢
          linarith
        have hz_nn : (0 : ℤ) ≤ z := by
          by_contra h; push_neg at h
          have : (z : ℝ) < 0 := Int.cast_lt_zero.mpr h
          linarith [hx_eq]
        have hz_le : z ≤ 2 * ↑n := by
          by_contra h; push_neg at h
          have hα_lt : α < 2 * ↑n := by
            rw [hα_def, hδ_def, div_lt_iff₀ hδ_pos]
            have hδ_val : 2 * (↑n : ℝ) * δ = 1/2 := by rw [hδ_def]; field_simp; ring
            linarith
          have : (↑z : ℝ) > 2 * (↑n : ℝ) := by exact_mod_cast h
          linarith [hz]
        have h := hx ⟨z.toNat, by omega⟩
        apply h; rw [hx_eq]
        push_cast
        rw [show (z.toNat : ℝ) = (z : ℝ) from by exact_mod_cast Int.toNat_of_nonneg hz_nn]
      set z := ⌊α⌋
      have hz1 : (↑z : ℝ) < α := lt_of_le_of_ne (Int.floor_le α) (fun h => hα_ne_int z h.symm)
      have hz2 : α < ↑z + 1 := Int.lt_floor_add_one α
      filter_upwards [
        (isOpen_Ioo.preimage (by fun_prop : Continuous fun y => (y + 1/4) / δ)).mem_nhds
          (show (x + 1/4) / δ ∈ Set.Ioo (↑z : ℝ) (↑z + 1) from ⟨hz1, hz2⟩),
        isOpen_Ioo.mem_nhds (show x ∈ Set.Ioo (-(1/4 : ℝ)) (1/4) from ⟨hx_lo, hx_ge⟩)
      ] with y hy_floor hy_range
      have h_floor_eq : ⌊(y + 1/4) / δ⌋ = z := by
        apply le_antisymm
        · have h2 : (y + 1/4) / δ < ↑z + 1 := hy_floor.2
          have h3 : (y + 1/4) / δ < (↑(z + 1) : ℝ) := by push_cast; exact h2
          have := Int.floor_lt.mpr h3; omega
        · exact Int.le_floor.mpr (le_of_lt hy_floor.1)
      have hy_cond : ¬(y < (-1:ℝ)/4 ∨ y ≥ 1/4) := by push_neg; constructor <;> linarith [hy_range.1, hy_range.2]
      have hx_cond : ¬(x < (-1:ℝ)/4 ∨ x ≥ 1/4) := by push_neg; constructor <;> linarith
      simp only [step_function, hy_cond, hx_cond, ↓reduceIte]
      simp only [show (1 : ℝ) / (4 * (↑n : ℝ)) = δ from rfl, h_floor_eq]
      rfl

lemma eLpNorm_conv_ge_discrete (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = 4 * n * m) (k : ℕ) :
    (MeasureTheory.eLpNorm
      (MeasureTheory.convolution (step_function n m c) (step_function n m c)
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume).toReal ≥
    (1 / (4 * (n : ℝ)) / (m : ℝ)^2) *
      discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ)) k := by
  have h_grid := convolution_at_grid_point n m hn hm c hc k
  rw [← h_grid]
  set S := step_function n m c
  have h_S_int : MeasureTheory.Integrable S MeasureTheory.volume := step_function_integrable n m c
  have h_S_nn : ∀ x, 0 ≤ S x := step_function_nonneg n m hm c
  have h_conv_nn : ∀ x, 0 ≤ MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x :=
    convolution_nonneg h_S_nn h_S_nn
  have h_S_compact : HasCompactSupport S := by
    apply CompactIccSpace.isCompact_Icc.of_isClosed_subset isClosed_closure
    exact (closure_mono ((step_function_support n m c).trans Set.Ico_subset_Icc_self)).trans
      (by rw [closure_Icc])
  have h_S_le_bound : ∀ x, S x ≤ 4 * n := by
    intro x; simp only [S, step_function]
    split_ifs with h1 h2
    · positivity
    · rw [div_le_iff₀ (by positivity : (0 : ℝ) < m)]
      exact_mod_cast (hc ▸ Finset.single_le_sum (fun a _ => Nat.zero_le (c a)) (Finset.mem_univ _) : c _ ≤ 4 * n * m)
    · positivity
  have h_bound : ∀ y, MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume y ≤ (4 * n) * ∫ t, S t := by
    intro y
    simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
    calc ∫ t, S t * S (y - t) ≤ ∫ t, S t * (4 * n) := by
          apply MeasureTheory.integral_mono
          · exact (h_S_int.comp_sub_left y).bdd_mul' h_S_int.aestronglyMeasurable
              (MeasureTheory.ae_of_all _ (fun x => by
                rw [Real.norm_eq_abs, abs_of_nonneg (h_S_nn x)]
                exact h_S_le_bound x))
          · exact h_S_int.mul_const _
          · exact fun t => mul_le_mul_of_nonneg_left (h_S_le_bound _) (h_S_nn t)
      _ = (4 * n) * ∫ t, S t := by rw [MeasureTheory.integral_mul_const]; ring
  have h_conv_int : MeasureTheory.Integrable (MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) MeasureTheory.volume :=
    h_S_int.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) h_S_int
  have h_memLp := MeasureTheory.memLp_top_of_bound
    h_conv_int.aestronglyMeasurable ((4 * n) * ∫ t, S t)
    (MeasureTheory.ae_of_all _ (fun y => by
      rw [Real.norm_eq_abs, abs_of_nonneg (h_conv_nn y)]
      exact h_bound y))
  have h_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ :=
    h_memLp.2.ne
  -- ContinuousAt of the convolution at the grid point via dominated convergence
  set z₀ := (-1 : ℝ) / 2 + (↑k + 1) * (1 / (4 * ↑n))
  have h_conv_ca : ContinuousAt (fun z => ∫ t, S t * S (z - t)) z₀ := by
    apply MeasureTheory.continuousAt_of_dominated
    · -- AEStronglyMeasurable for each z near z₀
      apply Filter.Eventually.of_forall; intro z
      exact ((h_S_int.comp_sub_left z).bdd_mul' h_S_int.aestronglyMeasurable
        (MeasureTheory.ae_of_all _ fun x => by
          rw [Real.norm_eq_abs, abs_of_nonneg (h_S_nn x)]
          exact h_S_le_bound x)).aestronglyMeasurable
    · -- Dominated by (4*n) * S (integrable bound)
      apply Filter.Eventually.of_forall; intro z
      exact MeasureTheory.ae_of_all _ fun t => by
        rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (h_S_nn t) (h_S_nn (z - t)))]
        calc S t * S (z - t) ≤ S t * (4 * n) :=
              mul_le_mul_of_nonneg_left (h_S_le_bound _) (h_S_nn t)
          _ = (4 * n) * S t := by ring
    · exact h_S_int.const_mul _
    · -- ContinuousAt for ae-every t
      -- The boundary set {t : z₀ - t is a bin boundary} is finite, hence null
      -- Bad set: where z₀ - t is a bin boundary
      set B := Set.range (fun i : Fin (2 * n + 1) => z₀ - (-(1/4 : ℝ) + ↑i.val / (4 * ↑n)))
      have hB_finite : Set.Finite B := Set.finite_range _
      have hB_null : MeasureTheory.volume B = 0 := hB_finite.measure_zero _
      rw [Filter.eventually_iff]
      apply MeasureTheory.measure_mono_null (fun t (ht : ¬ContinuousAt (fun z => S t * S (z - t)) z₀) => _)
        hB_null
      intro t ht
      show t ∈ B
      by_contra ht_not_in
      apply ht
      apply ContinuousAt.mul continuousAt_const
      have h_ca_S := step_function_continuousAt n m hn c (z₀ - t) (fun i h_eq =>
        ht_not_in (Set.mem_range.mpr ⟨i, by linarith⟩))
      change ContinuousAt (S ∘ (· - t)) z₀
      exact ContinuousAt.comp h_ca_S (continuous_sub_right t).continuousAt
  have h_conv_ca' : ContinuousAt (MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) z₀ := by
    have h_eq : (fun z => ∫ t, S t * S (z - t)) = MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume := by
      ext z; simp [MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
    rw [← h_eq]; exact h_conv_ca
  exact eLpNorm_top_ge_of_continuous_at
    (MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
    h_conv_nn h_conv_int _ h_conv_ca' h_fin

-- Helper: window sum bound implies test_value ≤ autoconvolution_ratio
-- Fine grid: test_value = (1/(4nℓ)) · ∑_window DA(a,k) where a_i = c_i/m.
-- eLpNorm_conv_ge_discrete gives DA(a,k) ≤ 4n · ‖g*g‖_∞ (via convolution_at_grid_point).
-- So test_value ≤ ((ℓ-1)/ℓ) · ‖g*g‖_∞ ≤ ‖g*g‖_∞ = R(g).
lemma window_sum_le_max_times (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = 4 * n * m) (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    test_value n m c ℓ s_lo ≤
      autoconvolution_ratio (step_function n m c) := by
  set N := (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal
  -- Each discrete_autoconvolution(a, k) where a_i = c_i/m is bounded:
  -- eLpNorm_conv_ge_discrete: N ≥ (1/(4n·m²)) · conv_int[k]
  -- Since DA(a, k) = (1/m²) · conv_int[k], we get DA(a, k) ≤ 4n · N
  have h_DA_bound : ∀ k, discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ) / m) k ≤ 4 * n * N := by
    intro k
    have h_ge := eLpNorm_conv_ge_discrete n m hn hm c hc k
    -- h_ge : N ≥ (1/(4n) / m²) · ∑_{i+j=k} c_i·c_j
    -- We need: DA(c/m, k) = (1/m²) · ∑_{i+j=k} c_i·c_j ≤ 4n · N
    have h_da_eq : discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ) / m) k =
        (1 / (m : ℝ)^2) * discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ)) k := by
      unfold discrete_autoconvolution; simp_rw [Finset.mul_sum]; congr 1; ext i
      congr 1; ext j; split_ifs <;> ring
    rw [h_da_eq]
    have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
    have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
    rw [ge_iff_le] at h_ge
    -- h_ge: (1/(4n) / m²) · conv_int ≤ N, so conv_int ≤ 4n·m² · N
    have h_conv_int_le : discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ)) k ≤ 4 * n * (m : ℝ)^2 * N := by
      rw [div_div, div_mul_eq_mul_div] at h_ge
      rw [div_le_iff₀ (by positivity : (0 : ℝ) < 4 * n * m^2)] at h_ge; linarith
    calc (1 / (m : ℝ)^2) * discrete_autoconvolution (fun i => (c i : ℝ)) k
        ≤ (1 / (m : ℝ)^2) * (4 * n * (m : ℝ)^2 * N) :=
          mul_le_mul_of_nonneg_left h_conv_int_le (by positivity)
      _ = 4 * n * N := by field_simp
  -- test_value = (1/(4nℓ)) · ∑_window DA(a, k)
  -- window has ≤ ℓ-1 terms, each ≤ 4n·N
  have h_card : (Finset.Icc s_lo (s_lo + ℓ - 2)).card ≤ ℓ - 1 := by
    rw [Nat.card_Icc]; omega
  have h_window_sum : ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ) / m) k ≤ (ℓ - 1) * (4 * n * N) := by
    calc ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
          discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ) / m) k
        ≤ ∑ _ ∈ Finset.Icc s_lo (s_lo + ℓ - 2), (4 * n * N) :=
          Finset.sum_le_sum fun k _ => h_DA_bound k
      _ = (Finset.Icc s_lo (s_lo + ℓ - 2)).card * (4 * n * N) := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ (ℓ - 1) * (4 * n * N) := by
          apply mul_le_mul_of_nonneg_right _ (by positivity)
          have : (↑(Finset.Icc s_lo (s_lo + ℓ - 2)).card : ℝ) ≤ ↑(ℓ - 1) := by exact_mod_cast h_card
          have hℓ1 : (1 : ℕ) ≤ ℓ := by omega
          rw [Nat.cast_sub hℓ1] at this; push_cast at this ⊢; linarith
  -- test_value ≤ (1/(4nℓ)) · (ℓ-1) · 4n · N = ((ℓ-1)/ℓ) · N ≤ N
  have h_tv_le : test_value n m c ℓ s_lo ≤ N := by
    unfold test_value; dsimp only []
    calc (1 / (4 * ↑n * ↑ℓ)) * ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
          discrete_autoconvolution (fun i : Fin (2 * n) => (↑(c i) : ℝ) / ↑m) k
        ≤ (1 / (4 * ↑n * ↑ℓ)) * ((↑ℓ - 1) * (4 * ↑n * N)) :=
          mul_le_mul_of_nonneg_left h_window_sum (by positivity)
      _ = ((↑ℓ - 1) / ↑ℓ) * N := by
          have hℓ_pos : (0 : ℝ) < ℓ := by exact_mod_cast Nat.lt_of_lt_of_le (by norm_num) hℓ
          have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
          field_simp
      _ ≤ N := by
          have hN_nn : 0 ≤ N := ENNReal.toReal_nonneg
          have : (↑ℓ - 1 : ℝ) / ↑ℓ ≤ 1 := by
            rw [div_le_one₀ (by positivity : (0 : ℝ) < ℓ)]; linarith
          exact mul_le_of_le_one_left hN_nn this
  -- autoconvolution_ratio = N / (∫g)² = N / 1 = N
  have h_ratio : autoconvolution_ratio (step_function n m c) = N := by
    unfold autoconvolution_ratio; dsimp only []
    rw [integral_step_function n m hn hm c hc, one_pow, div_one]
  linarith

/-- Bins are disjoint: for any point t, sum_i f_bin(f,n,i)(t) <= f(t). -/
private lemma sum_f_bin_le (n : ℕ) (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x) (t : ℝ) :
    ∑ i : Fin (2 * n), f_bin f n i t ≤ f t := by
  simp only [f_bin]
  by_cases ht : ∃ i : Fin (2 * n), t ∈ bin_interval n i
  · obtain ⟨i₀, hi₀⟩ := ht
    have h_eq : ∀ i : Fin (2 * n), Set.indicator (bin_interval n i) f t =
        if i = i₀ then f t else 0 := by
      intro i; simp only [Set.indicator_apply]
      split_ifs with h1 h2
      · rfl
      · exfalso; apply h2; simp only [bin_interval] at h1 hi₀
        have hδ : (0 : ℝ) < 1 / (4 * (n : ℝ)) := by
          have : 0 < n := by linarith [i₀.2]
          positivity
        have h1l := (Set.mem_Ico.mp h1).1; have h1r := (Set.mem_Ico.mp h1).2
        have h0l := (Set.mem_Ico.mp hi₀).1; have h0r := (Set.mem_Ico.mp hi₀).2
        have : i.1 = i₀.1 := by
          by_contra h_ne; rcases Nat.lt_or_gt_of_ne h_ne with h | h
          · linarith [mul_le_mul_of_nonneg_right (show (↑i + 1 : ℝ) ≤ ↑↑i₀ from by exact_mod_cast h) (le_of_lt hδ)]
          · linarith [mul_le_mul_of_nonneg_right (show (↑↑i₀ + 1 : ℝ) ≤ ↑↑i from by exact_mod_cast h) (le_of_lt hδ)]
        exact Fin.ext this
      · simp_all
      · rfl
    simp_rw [h_eq]; simp
  · have : ∀ i : Fin (2 * n), Set.indicator (bin_interval n i) f t = 0 := fun i => by
      simp only [Set.indicator_apply]
      split_ifs with h
      · exact absurd ⟨i, h⟩ ht
      · rfl
    simp only [this, Finset.sum_const_zero]; exact hf_nonneg t

/-- **Theorem**: Continuous test value lower bound.
    R(f) >= TV_continuous for admissible f with f*f ∈ L∞, ell >= 2.

    The hypothesis h_conv_fin (finiteness of ‖f*f‖_∞) is necessary: for f ∈ L¹ \ L²
    (e.g., f(x) = C|x|^{-3/4}·1_{(0,ε)}), the self-convolution f*f may be unbounded,
    and autoconvolution_ratio returns 0 via ENNReal.toReal(⊤) = 0 rather than ∞.
    This hypothesis is satisfied by all bounded functions, all L² functions,
    and in particular all step functions used in the cascade computation. -/
theorem continuous_test_value_le_ratio (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    autoconvolution_ratio f ≥ test_value_continuous n f ℓ s_lo := by
  set μ := bin_masses f n
  set conv_ff := MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
  set N := (MeasureTheory.eLpNorm conv_ff ⊤ MeasureTheory.volume).toReal
  have hf_int' := MeasureTheory.integrable_of_integral_eq_one hf_int
  have hμ_nn : ∀ i, 0 ≤ μ i := fun i => bin_masses_nonneg f hf_nonneg n i
  have hμ_sum := sum_bin_masses_eq_one n hn f hf_supp hf_int
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hℓ_pos : (0 : ℝ) < ℓ := by exact_mod_cast Nat.lt_of_lt_of_le (by norm_num) hℓ
  have hR : autoconvolution_ratio f = N := by
    unfold autoconvolution_ratio; dsimp only []
    rw [hf_int, one_pow, div_one]
  rw [ge_iff_le, hR]; unfold test_value_continuous; simp only [discrete_autoconvolution]
  have h_fac : ∀ k, (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
      if i.1 + j.1 = k then (4 * ↑n * μ i) * (4 * ↑n * μ j) else 0) =
    (4 * ↑n) ^ 2 * (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
      if i.1 + j.1 = k then μ i * μ j else 0) := by
    intro k; rw [Finset.mul_sum]; congr 1; ext i; rw [Finset.mul_sum]; congr 1; ext j
    split_ifs <;> ring
  have h_fac' : ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then ((4 : ℝ) * ↑n * μ i) * ((4 : ℝ) * ↑n * μ j) else (0 : ℝ)) =
      ((4 : ℝ) * ↑n) ^ 2 * ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
        ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
          if i.1 + j.1 = k then μ i * μ j else 0 := by
    rw [Finset.mul_sum]; congr 1; ext k; exact h_fac k
  rw [h_fac']
  set ws := ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
    ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
      if i.1 + j.1 = k then μ i * μ j else 0
  have h_simp : (1 / (4 * ↑n * ↑ℓ)) * ((4 * ↑n) ^ 2 * ws) = (4 * ↑n / ↑ℓ) * ws := by
    field_simp
  rw [h_simp]
  have h_ws_le : ws ≤ 1 := by
    have h_rearrange : ws = ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        if i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then μ i * μ j else 0 := by
      simp only [ws]; rw [Finset.sum_comm]; congr 1; ext i; rw [Finset.sum_comm]
      congr 1; ext j
      simp [Finset.sum_ite_eq]
    calc ws = ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
          if i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then μ i * μ j else 0 := h_rearrange
      _ ≤ ∑ i : Fin (2 * n), ∑ j : Fin (2 * n), μ i * μ j := by
          apply Finset.sum_le_sum; intro i _; apply Finset.sum_le_sum; intro j _
          split_ifs <;> [exact le_refl _; exact mul_nonneg (hμ_nn i) (hμ_nn j)]
      _ = (∑ i : Fin (2 * n), μ i) ^ 2 := by rw [sq, Finset.sum_mul_sum]
      _ = 1 := by rw [hμ_sum]; ring
  have hws_nn : 0 ≤ ws := Finset.sum_nonneg fun k _ => Finset.sum_nonneg fun i _ =>
    Finset.sum_nonneg fun j _ => by
      split_ifs <;> [exact mul_nonneg (hμ_nn i) (hμ_nn j); exact le_refl 0]
  have hN_nn : 0 ≤ N := ENNReal.toReal_nonneg
  suffices h_key : ws ≤ N * (↑ℓ / (4 * ↑n)) by
    calc (4 * ↑n / ↑ℓ) * ws ≤ (4 * ↑n / ↑ℓ) * (N * (↑ℓ / (4 * ↑n))) :=
          mul_le_mul_of_nonneg_left h_key (div_nonneg (by positivity) (by positivity))
      _ = N := by field_simp
  by_cases h_easy : 1 ≤ N * (↑ℓ / (4 * ↑n))
  · linarith
  · push_neg at h_easy
    set δ := (1 : ℝ) / (4 * n)
    set Z := Set.Ico (-(1/2 : ℝ) + s_lo * δ) (-(1/2 : ℝ) + (↑s_lo + ↑ℓ) * δ) with hZ_def
    have hδ_pos : 0 < δ := by positivity
    have h_supp : ∀ (i j : Fin (2 * n)),
        i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2) → ∀ z, z ∉ Z →
        MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume z = 0 := by
      intro i j hij z hz
      simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
      apply MeasureTheory.integral_eq_zero_of_ae
      filter_upwards [] with t
      show f_bin f n i t * f_bin f n j (z - t) = 0
      simp only [f_bin, Set.indicator_apply, bin_interval]
      split_ifs with h1 h2
      · exfalso; apply hz; simp only [Z, Set.mem_Ico, δ]
        have h1l := (Set.mem_Ico.mp h1).1; have h1r := (Set.mem_Ico.mp h1).2
        have h2l := (Set.mem_Ico.mp h2).1; have h2r := (Set.mem_Ico.mp h2).2
        have hij1 := (Finset.mem_Icc.mp hij).1; have hij2 := (Finset.mem_Icc.mp hij).2
        have h_ij_le : (↑i.val + ↑j.val : ℝ) + 2 ≤ (↑s_lo + ↑ℓ : ℝ) := by
          have : i.val + j.val + 2 ≤ s_lo + ℓ := by omega
          exact_mod_cast this
        have h_slo_le : (↑s_lo : ℝ) ≤ ↑i.val + ↑j.val := by exact_mod_cast hij1
        constructor <;> nlinarith
      · simp
      · simp
      · simp
    have hconv_nn : ∀ x, 0 ≤ conv_ff x := convolution_nonneg hf_nonneg hf_nonneg
    have hconv_int : MeasureTheory.Integrable conv_ff MeasureTheory.volume :=
      hf_int'.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) hf_int'
    -- Helper: each bin convolution is nonneg
    have h_conv_bin_nn : ∀ (i j : Fin (2 * n)) (z : ℝ),
        0 ≤ MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume z :=
      fun i j z => MeasureTheory.integral_nonneg fun t =>
        mul_nonneg (f_bin_nonneg f hf_nonneg n i t) (f_bin_nonneg f hf_nonneg n j (z - t))
    -- Helper: each f_bin integrable
    have h_fbin_int : ∀ i : Fin (2 * n),
        MeasureTheory.Integrable (f_bin f n i) MeasureTheory.volume :=
      fun i => f_bin_integrable f hf_int' n i
    -- Helper: each bin convolution integrable
    have h_conv_integrable : ∀ i j : Fin (2 * n),
        MeasureTheory.Integrable (MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) MeasureTheory.volume :=
      fun i j => (h_fbin_int i).integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) (h_fbin_int j)
    -- Helper: if-then-else integrable
    have h_ite_integrable : ∀ (k : ℕ) (i j : Fin (2 * n)),
        MeasureTheory.Integrable (fun z =>
          if i.1 + j.1 = k then
            MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
              (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume z
          else 0) MeasureTheory.volume := by
      intro k i j; split_ifs
      · exact h_conv_integrable i j
      · exact MeasureTheory.integrable_zero _ _ _
    -- Helper: cross integrals equal products of bin masses
    have h_cross_int : ∀ (i j : Fin (2 * n)),
        ∫ x, MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x = μ i * μ j := by
      intro i j
      rw [MeasureTheory.integral_convolution (ContinuousLinearMap.mul ℝ ℝ)
        (h_fbin_int i) (h_fbin_int j)]
      simp only [ContinuousLinearMap.mul_apply', integral_f_bin, μ]
    -- Filtered sum of bin convolutions ≤ conv_ff, ae.
    -- Proof sketch: (1) drop filter (nonneg terms), (2) by Fubini, for ae z the
    -- convolution integrand is integrable, so we swap ∑ past ∫, getting
    -- ∫ (∑ f_bin_i(t))*(∑ f_bin_j(z-t)) dt, (3) bound by ∫ f(t)*f(z-t) dt = conv_ff(z)
    -- via sum_f_bin_le and integral_mono.
    have h_pw : ∀ᵐ z ∂MeasureTheory.volume, (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
        ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
          if i.1 + j.1 = k then
            MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
              (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume z
          else 0) ≤ conv_ff z := by
      -- Use Fubini to get ae integrability of the convolution integrand
      have h_prod := hf_int'.convolution_integrand (ContinuousLinearMap.mul ℝ ℝ) hf_int'
      rw [MeasureTheory.integrable_prod_iff h_prod.aestronglyMeasurable] at h_prod
      filter_upwards [h_prod.1] with z hz_int
      -- At z where the convolution integrand is integrable:
      -- hz_int : Integrable (fun t => f(t) * f(z - t))
      simp only [ContinuousLinearMap.mul_apply'] at hz_int
      -- The proof proceeds in two steps:
      -- (1) The filtered triple sum ≤ the unfiltered double sum (nonneg terms)
      -- (2) The unfiltered double sum ≤ conv_ff(z) (via integral_mono + sum_f_bin_le)
      -- Each convolution term at z equals ∫ f_bin_i(t) * f_bin_j(z-t) dt
      set cij := fun (i j : Fin (2 * n)) =>
        MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume z
      -- Step 1: filtered sum ≤ unfiltered double sum
      -- Key identity: ∑_k∈W ∑_i ∑_j (if i+j=k then cij else 0) = ∑_i ∑_j (if i+j∈W then cij else 0)
      have h_rewrite : (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
          ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
            if i.1 + j.1 = k then cij i j else 0) =
          ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
            if i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then cij i j else 0 := by
        rw [Finset.sum_comm]; congr 1; ext i; rw [Finset.sum_comm]; congr 1; ext j
        simp [Finset.sum_ite_eq]
      rw [h_rewrite]
      have h_le_full : (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
            if i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2) then cij i j else 0) ≤
          ∑ i : Fin (2 * n), ∑ j : Fin (2 * n), cij i j := by
        apply Finset.sum_le_sum; intro i _; apply Finset.sum_le_sum; intro j _
        split_ifs
        · exact le_refl _
        · exact h_conv_bin_nn i j z
      -- Step 2: unfiltered double sum ≤ conv_ff(z)
      -- Each cij = ∫ f_bin_i(t) * f_bin_j(z-t) dt
      -- ∑_i ∑_j cij ≤ ∫ f(t)*f(z-t) dt = conv_ff(z) by integral_mono + sum_f_bin_le
      have h_full_le : (∑ i : Fin (2 * n), ∑ j : Fin (2 * n), cij i j) ≤ conv_ff z := by
        -- Each summand is ∫ f_bin_i(t) * f_bin_j(z-t) dt, integrable since ≤ f(t)*f(z-t) ae
        have h_bin_prod_int : ∀ i j : Fin (2 * n),
            MeasureTheory.Integrable (fun t => f_bin f n i t * f_bin f n j (z - t))
              MeasureTheory.volume := by
          intro i j
          apply MeasureTheory.Integrable.mono' hz_int
          · exact (h_fbin_int i).aestronglyMeasurable.mul
              ((h_fbin_int j).comp_sub_left z).aestronglyMeasurable
          · apply MeasureTheory.ae_of_all; intro t
            rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (f_bin_nonneg f hf_nonneg n i t)
              (f_bin_nonneg f hf_nonneg n j (z - t)))]
            exact mul_le_mul
              (sum_f_bin_le n f hf_nonneg t |>.trans' <| Finset.single_le_sum
                (fun a _ => f_bin_nonneg f hf_nonneg n a t) (Finset.mem_univ i))
              (sum_f_bin_le n f hf_nonneg (z - t) |>.trans' <| Finset.single_le_sum
                (fun a _ => f_bin_nonneg f hf_nonneg n a (z - t)) (Finset.mem_univ j))
              (f_bin_nonneg f hf_nonneg n j (z - t))
              (hf_nonneg t)
        -- Strategy: ∑_i ∑_j cij ≤ ∫ f(t)*f(z-t) dt = conv_ff(z)
        -- We rewrite cij as integrals, then bound the double sum directly.
        -- For each fixed t: ∑_i ∑_j f_bin_i(t) * f_bin_j(z-t) ≤ f(t) * f(z-t)
        -- So: ∑_i ∑_j (∫ f_bin_i * f_bin_j) ≤ ∫ (∑_i ∑_j f_bin_i * f_bin_j) ≤ ∫ f * f(z-·)
        have h_inner_int : ∀ i : Fin (2 * n),
            MeasureTheory.Integrable (fun t => ∑ j : Fin (2 * n), f_bin f n i t * f_bin f n j (z - t))
              MeasureTheory.volume :=
          fun i => MeasureTheory.integrable_finset_sum _ (fun j _ => h_bin_prod_int i j)
        -- Key step: push both sums inside the integral
        -- ∑_i ∑_j cij = ∑_i ∑_j ∫ f_bin_i * f_bin_j(z-·) = ∫ ∑_i ∑_j f_bin_i * f_bin_j(z-·)
        have h_eq : ∑ i : Fin (2 * n), ∑ j : Fin (2 * n), cij i j =
            ∫ t, ∑ i : Fin (2 * n), ∑ j : Fin (2 * n), f_bin f n i t * f_bin f n j (z - t) := by
          simp only [cij, MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
          -- Push outer sum past integral: ∑_i (∫ ...) = ∫ (∑_i ...)
          rw [show (∑ i : Fin (2 * n), ∑ j : Fin (2 * n), ∫ t, f_bin f n i t * f_bin f n j (z - t)) =
              (∑ i : Fin (2 * n), ∫ t, ∑ j : Fin (2 * n), f_bin f n i t * f_bin f n j (z - t)) from by
            congr 1; ext i
            exact (MeasureTheory.integral_finset_sum Finset.univ (fun j _ => h_bin_prod_int i j)).symm]
          exact (MeasureTheory.integral_finset_sum Finset.univ (fun i _ => h_inner_int i)).symm
        rw [h_eq]
        -- Now bound: ∫ ∑∑ f_bin * f_bin ≤ ∫ f * f(z-·)
        simp only [conv_ff, MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
        apply MeasureTheory.integral_mono
        · exact MeasureTheory.integrable_finset_sum _ (fun i _ => h_inner_int i)
        · exact hz_int
        · intro t
          calc ∑ i : Fin (2 * n), ∑ j : Fin (2 * n), f_bin f n i t * f_bin f n j (z - t)
              = (∑ i : Fin (2 * n), f_bin f n i t) * (∑ j : Fin (2 * n), f_bin f n j (z - t)) := by
                rw [Finset.sum_mul_sum]
            _ ≤ f t * f (z - t) := by
                apply mul_le_mul (sum_f_bin_le n f hf_nonneg t) (sum_f_bin_le n f hf_nonneg (z - t))
                  (Finset.sum_nonneg fun j _ => f_bin_nonneg f hf_nonneg n j (z - t)) (hf_nonneg t)
      linarith [h_le_full, h_full_le]
    have h_ae_N : ∀ᵐ z ∂MeasureTheory.volume, conv_ff z ≤ N := by
      filter_upwards [MeasureTheory.enorm_ae_le_eLpNormEssSup conv_ff MeasureTheory.volume] with z hz
      rw [← ENNReal.toReal_ofReal (hconv_nn z)]
      rw [show N = (MeasureTheory.eLpNormEssSup conv_ff MeasureTheory.volume).toReal from by
        simp [N, MeasureTheory.eLpNorm_exponent_top]]
      exact ENNReal.toReal_mono
        (by rwa [← MeasureTheory.eLpNorm_exponent_top])
        (le_trans (by rw [Real.enorm_eq_ofReal (hconv_nn z)]) hz)
    have h_meas_Z : MeasureTheory.volume Z ≠ ⊤ := by
      simp only [Z]; rw [Real.volume_Ico]; exact ENNReal.ofReal_ne_top
    have h_intZ : ∫ z in Z, conv_ff z ≤ N * (↑ℓ * δ) := by
      calc ∫ z in Z, conv_ff z ≤ ∫ z in Z, N :=
            MeasureTheory.setIntegral_mono_ae hconv_int.integrableOn
              (MeasureTheory.integrableOn_const (hs := h_meas_Z)) h_ae_N
        _ = N * (↑ℓ * δ) := by
            rw [MeasureTheory.setIntegral_const, smul_eq_mul, mul_comm]
            congr 1
            simp only [Z, hZ_def, δ]
            rw [Real.volume_real_Ico]
            have : (0 : ℝ) < ↑ℓ * (1 / (4 * ↑n)) := by positivity
            rw [max_eq_left (by linarith)]; ring
    set g : ℝ → ℝ := fun z => ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then
          MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
            (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume z
        else 0
    have hg_vanish : ∀ z, z ∉ Z → g z = 0 := by
      intro z hz; simp only [g]
      apply Finset.sum_eq_zero; intro k hk
      apply Finset.sum_eq_zero; intro i _
      apply Finset.sum_eq_zero; intro j _
      split_ifs with h
      · rw [← h] at hk; exact h_supp i j hk z hz
      · rfl
    have hg_le_ae : ∀ᵐ z ∂MeasureTheory.volume, g z ≤ conv_ff z := h_pw
    have hg_nn : ∀ z, 0 ≤ g z := by
      intro z; simp only [g]
      apply Finset.sum_nonneg; intro k _; apply Finset.sum_nonneg; intro i _
      apply Finset.sum_nonneg; intro j _; split_ifs
      · exact h_conv_bin_nn i j z
      · exact le_refl 0
    have hg_int : MeasureTheory.Integrable g MeasureTheory.volume := by
      apply MeasureTheory.Integrable.mono' hconv_int
      · apply Finset.aestronglyMeasurable_fun_sum; intro k _
        apply Finset.aestronglyMeasurable_fun_sum; intro i _
        apply Finset.aestronglyMeasurable_fun_sum; intro j _
        split_ifs
        · exact (h_conv_integrable i j).aestronglyMeasurable
        · exact MeasureTheory.aestronglyMeasurable_zero
      · filter_upwards [hg_le_ae] with z hz
        simp only [Real.norm_eq_abs, abs_of_nonneg (hg_nn z)]
        exact hz
    -- ═══════ Sorry 3: Integral of g equals ws ═══════
    have hg_integral : ∫ z, g z = ws := by
      -- We compute ∫ g by swapping ∫ past the triple finite sum, then evaluating
      -- each integral using h_cross_int: ∫ conv(f_bin_i, f_bin_j) = μ_i * μ_j.
      -- Since g = ∑_k ∑_i ∑_j h_kij where each h_kij is integrable:
      -- ∫ g = ∫ ∑_k ∑_i ∑_j h_kij = ∑_k ∑_i ∑_j ∫ h_kij
      --     = ∑_k ∑_i ∑_j (if i+j=k then μ_i*μ_j else 0) = ws.
      -- Direct proof using MeasureTheory.integral_finset_sum at each level:
      -- Linearity of integral past 3 levels of finite sums, then h_cross_int.
      -- Each summand is integrable (h_ite_integrable). Uses integral_finset_sum 3x.
      simp only [g]
      -- Level 1: swap ∫ past ∑_k
      rw [MeasureTheory.integral_finset_sum _ (fun k _ =>
        MeasureTheory.integrable_finset_sum _ (fun i _ =>
          MeasureTheory.integrable_finset_sum _ (fun j _ => h_ite_integrable k i j)))]
      congr 1; ext k
      -- Level 2: swap ∫ past ∑_i
      rw [MeasureTheory.integral_finset_sum _ (fun i _ =>
        MeasureTheory.integrable_finset_sum _ (fun j _ => h_ite_integrable k i j))]
      congr 1; ext i
      -- Level 3: swap ∫ past ∑_j
      rw [MeasureTheory.integral_finset_sum _ (fun j _ => h_ite_integrable k i j)]
      congr 1; ext j
      -- Evaluate each integral
      split_ifs with h
      · exact h_cross_int i j
      · simp
    have hg_setint : ∫ z, g z = ∫ z in Z, g z := by
      rw [MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero (fun z hz => hg_vanish z hz)]
    have hg_Z_le : ∫ z in Z, g z ≤ ∫ z in Z, conv_ff z :=
      MeasureTheory.setIntegral_mono_ae hg_int.integrableOn hconv_int.integrableOn hg_le_ae
    calc ws = ∫ z, g z := hg_integral.symm
      _ = ∫ z in Z, g z := hg_setint
      _ ≤ ∫ z in Z, conv_ff z := hg_Z_le
      _ ≤ N * (↑ℓ * δ) := h_intZ
      _ = N * (↑ℓ / (4 * ↑n)) := by simp only [δ]; ring

end -- noncomputable section
