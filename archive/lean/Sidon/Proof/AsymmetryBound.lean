import Mathlib

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
-- Asymmetry Bound (Claim 2.1)
-- Source: output (7).lean (UUID: f31f701e), prompt04_asymmetry_pruning.lean (PROVED)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Left-half restriction of f (indicator on (-1/4, 0)). -/
def f_L (f : ℝ → ℝ) : ℝ → ℝ := Set.indicator (Set.Ioo (-1/4 : ℝ) 0) f

theorem f_L_le_f (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x) :
    ∀ x, f_L f x ≤ f x := by
  intros x
  simp [f_L];
  by_cases hx : x ∈ Set.Ioo (-1 / 4 : ℝ) 0 <;> simp [hx, hf]

theorem f_L_supp (f : ℝ → ℝ) :
    Function.support (f_L f) ⊆ Set.Ioo (-1/4 : ℝ) 0 := by
  simp [f_L]

theorem f_L_conv_supp (f : ℝ → ℝ) :
    Function.support (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊆
    Set.Ioo (-1/2 : ℝ) 0 := by
  intro x hx; simp_all +decide [ MeasureTheory.convolution ] ;
  have h_support : ∀ t, f_L f t ≠ 0 → -1 / 4 < t ∧ t < 0 := by
    unfold f_L; aesop;
  contrapose! hx;
  rw [ MeasureTheory.integral_eq_zero_of_ae ];
  filter_upwards [ ] with t ; by_cases ht : f_L f t = 0 <;> by_cases ht' : f_L f ( x - t ) = 0 <;> simp_all +decide [ sub_eq_add_neg ];
  linarith [ h_support t ht, h_support ( x + -t ) ht', hx ( by linarith [ h_support t ht, h_support ( x + -t ) ht' ] ) ]

/-- Monotonicity of convolution for nonneg functions. -/
theorem convolution_mono_ae (f g : ℝ → ℝ)
    (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) (hfg : ∀ x, f x ≤ g x)
    (_hf_int : MeasureTheory.Integrable f MeasureTheory.volume)
    (hg_int : MeasureTheory.Integrable g MeasureTheory.volume) :
    ∀ᵐ x ∂MeasureTheory.volume,
      MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ≤
      MeasureTheory.convolution g g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  have h_convol_g_exists : ∀ᵐ x ∂MeasureTheory.volume, MeasureTheory.Integrable (fun y => g y * g (x - y)) MeasureTheory.volume := by
    have h_ae_conv : MeasureTheory.Integrable (fun p : ℝ × ℝ => g p.1 * g p.2) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume) := by
      exact MeasureTheory.Integrable.mul_prod hg_int hg_int;
    have h_ae_conv : MeasureTheory.Integrable (fun p : ℝ × ℝ => g p.1 * g (p.2 - p.1)) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume) := by
      have h_ae_conv : MeasureTheory.MeasurePreserving (fun p : ℝ × ℝ => (p.1, p.2 - p.1)) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume) := by
        exact MeasureTheory.measurePreserving_prod_sub MeasureTheory.volume MeasureTheory.volume;
      have h_ae_conv : MeasureTheory.Integrable (fun p : ℝ × ℝ => g p.1 * g p.2) (MeasureTheory.Measure.map (fun p : ℝ × ℝ => (p.1, p.2 - p.1)) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume)) := by
        rw [ h_ae_conv.map_eq ] ; assumption;
      rw [ MeasureTheory.integrable_map_measure ] at h_ae_conv ; aesop;
      · exact h_ae_conv.1;
      · exact AEMeasurable.prodMk ( measurable_fst.aemeasurable ) ( measurable_snd.sub measurable_fst |> Measurable.aemeasurable );
    rw [ MeasureTheory.integrable_prod_iff' ] at h_ae_conv ; aesop;
    exact h_ae_conv.1;
  filter_upwards [ h_convol_g_exists ] with x hx;
  refine' MeasureTheory.integral_mono_of_nonneg _ _ _;
  · exact Filter.Eventually.of_forall fun y => mul_nonneg ( hf _ ) ( hf _ );
  · exact hx;
  · filter_upwards [ ] with t using mul_le_mul ( hfg t ) ( hfg ( x - t ) ) ( hf _ ) ( hg _ )

/-- Averaging principle: ‖g‖∞ ≥ (∫g) / measure(support). -/
theorem averaging_principle (g : ℝ → ℝ) (hg : ∀ x, 0 ≤ g x)
    (hg_int : MeasureTheory.Integrable g MeasureTheory.volume)
    (S : Set ℝ) (hS : Function.support g ⊆ S)
    (v : ℝ) (hS_meas : MeasureTheory.volume S = ENNReal.ofReal v)
    (hv : 0 < v) :
    MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume ≥
      ENNReal.ofReal (MeasureTheory.integral MeasureTheory.volume g / v) := by
  have h_integral_restrict : ∫ x, g x ∂MeasureTheory.volume = ∫ x in S, g x ∂MeasureTheory.volume := by
    rw [ MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero fun x hx => by_contra fun hx' => hx <| hS <| by aesop ];
  have h_integral_bound : (∫⁻ x in S, ENNReal.ofReal (g x) ∂MeasureTheory.volume) ≤ (MeasureTheory.eLpNorm g ⊤ MeasureTheory.MeasureSpace.volume) * (MeasureTheory.MeasureSpace.volume S) := by
    have h_integral_bound : ∀ᵐ x ∂MeasureTheory.Measure.restrict MeasureTheory.volume S, ENNReal.ofReal (g x) ≤ MeasureTheory.eLpNorm g ⊤ MeasureTheory.MeasureSpace.volume := by
      have h_integral_bound : ∀ᵐ x ∂MeasureTheory.MeasureSpace.volume, ENNReal.ofReal (g x) ≤ MeasureTheory.eLpNorm g ⊤ MeasureTheory.MeasureSpace.volume := by
        have h_integral_bound : ∀ᵐ x ∂MeasureTheory.MeasureSpace.volume, ‖g x‖ₑ ≤ essSup (fun x => ‖g x‖ₑ) MeasureTheory.MeasureSpace.volume := by
          exact MeasureTheory.enorm_ae_le_eLpNormEssSup g MeasureTheory.MeasureSpace.volume;
        filter_upwards [ h_integral_bound ] with x hx using le_trans ( by simp +decide [ Real.enorm_eq_ofReal ( hg x ) ] ) hx;
      exact MeasureTheory.ae_restrict_of_ae h_integral_bound;
    refine' le_trans ( MeasureTheory.lintegral_mono_ae h_integral_bound ) _ ; aesop;
  simp_all +decide [ ENNReal.ofReal_div_of_pos hv ];
  rw [ ENNReal.div_le_iff_le_mul ] <;> norm_num [ hv ];
  refine' le_trans _ h_integral_bound;
  rw [ MeasureTheory.ofReal_integral_eq_lintegral_ofReal ];
  · exact hg_int.integrableOn;
  · exact Filter.Eventually.of_forall hg

/-- Integral of convolution = (integral)². -/
theorem integral_convolution_square (f : ℝ → ℝ)
    (hf : MeasureTheory.Integrable f MeasureTheory.volume) :
    MeasureTheory.integral MeasureTheory.volume (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) =
    (MeasureTheory.integral MeasureTheory.volume f) ^ 2 := by
  rw [ sq ];
  apply MeasureTheory.integral_convolution;
  · exact hf;
  · exact hf

theorem f_L_integrable (f : ℝ → ℝ) (hf : MeasureTheory.Integrable f MeasureTheory.volume) :
    MeasureTheory.Integrable (f_L f) MeasureTheory.volume := by
  convert hf.indicator measurableSet_Ioo using 1

theorem convolution_nonneg {f g : ℝ → ℝ} (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) :
    ∀ x, 0 ≤ MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  intro x
  simp [MeasureTheory.convolution];
  exact MeasureTheory.integral_nonneg fun t => mul_nonneg ( hf t ) ( hg ( x - t ) )

theorem f_L_nonneg (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x) :
    ∀ x, 0 ≤ f_L f x := by
  exact fun x => Set.indicator_nonneg ( fun _ _ => hf _ ) _

theorem volume_Ioo_half :
    MeasureTheory.volume (Set.Ioo (-1/2 : ℝ) 0) = ENNReal.ofReal (1/2) := by
  norm_num

/-- Asymmetry bound: ‖f*f‖∞ ≥ 2L² where L = ∫_{-1/4}^0 f.
    Proof chain: restrict f to left half, use convolution monotonicity,
    then averaging principle on the support of f_L * f_L ⊆ (-1/2, 0). -/
theorem asymmetry_bound (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (_hf_supp : Function.support f ⊆ Set.Icc (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_bdd : MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    let L := MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ioo (-1/4 : ℝ) 0) f)
    (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥ 2 * L ^ 2 := by
  have h_conv : MeasureTheory.eLpNorm (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≤ MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume := by
    have h_conv_le : ∀ᵐ x ∂MeasureTheory.volume, MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ≤ MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
      apply convolution_mono_ae (f_L f) f (fun x => f_L_nonneg f hf_nonneg x) hf_nonneg (fun x => f_L_le_f f hf_nonneg x) (f_L_integrable f (MeasureTheory.integrable_of_integral_eq_one hf_int)) (MeasureTheory.integrable_of_integral_eq_one hf_int);
    apply_rules [ MeasureTheory.eLpNorm_mono_ae ];
    filter_upwards [ h_conv_le ] with x hx using by rw [ Real.norm_of_nonneg ( convolution_nonneg ( f_L_nonneg f hf_nonneg ) ( f_L_nonneg f hf_nonneg ) x ), Real.norm_of_nonneg ( convolution_nonneg ( hf_nonneg ) ( hf_nonneg ) x ) ] ; exact hx;
  have h_integral : MeasureTheory.integral MeasureTheory.volume (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) = (MeasureTheory.integral MeasureTheory.volume (f_L f)) ^ 2 := by
    apply integral_convolution_square; exact f_L_integrable f (MeasureTheory.integrable_of_integral_eq_one hf_int);
  have h_avg : MeasureTheory.eLpNorm (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≥ ENNReal.ofReal ((MeasureTheory.integral MeasureTheory.volume (f_L f)) ^ 2 / (1 / 2)) := by
    have h_avg : MeasureTheory.eLpNorm (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≥ ENNReal.ofReal ((MeasureTheory.integral MeasureTheory.volume (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)) / (1 / 2)) := by
      apply_rules [ averaging_principle ];
      any_goals exact Set.Ioo ( -1 / 2 ) 0;
      · apply_rules [ convolution_nonneg, f_L_nonneg ];
      · apply_rules [ MeasureTheory.Integrable.integrable_convolution, f_L_integrable ];
        · exact MeasureTheory.integrable_of_integral_eq_one hf_int;
        · exact MeasureTheory.integrable_of_integral_eq_one hf_int;
      · convert f_L_conv_supp f using 1;
      · norm_num;
      · norm_num;
    aesop;
  have h_final : (MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥ (MeasureTheory.integral MeasureTheory.volume (f_L f)) ^ 2 / (1 / 2) := by
    refine' le_trans _ ( ENNReal.toReal_mono _ <| h_avg.trans h_conv );
    · rw [ ENNReal.toReal_ofReal ( by positivity ) ];
    · assumption;
  convert h_final using 1 ; ring!

end -- noncomputable section
