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
-- Essential Supremum Bounds
-- Source: output (14).lean (UUID: 124a8efc)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- eLpNorm equals essSup for nonneg functions. -/
theorem eLpNorm_eq_essSup_ofReal {α : Type*} [MeasureTheory.MeasureSpace α]
    (f : α → ℝ) (hf : ∀ x, 0 ≤ f x) :
    MeasureTheory.eLpNorm f ⊤ MeasureTheory.volume =
    essSup (fun x => ENNReal.ofReal (f x)) MeasureTheory.volume := by
  simp +decide [ MeasureTheory.eLpNormEssSup ];
  simp +decide only [Real.enorm_eq_ofReal (hf _)]

/-- Helper: Lebesgue integral bounded by essSup times measure of support superset. -/
theorem lintegral_le_essSup_mul_measure_ennreal {α : Type*} [MeasureTheory.MeasureSpace α]
    (f : α → ENNReal) (S : Set α) (h_supp : Function.support f ⊆ S) :
    ∫⁻ x, f x ∂MeasureTheory.volume ≤
    (essSup f MeasureTheory.volume) * (MeasureTheory.volume S) := by
  have h_integral_le_essSup_mul_measure : ∫⁻ x, f x ∂MeasureTheory.MeasureSpace.volume ≤ essSup f MeasureTheory.MeasureSpace.volume * MeasureTheory.MeasureSpace.volume (Function.support f) := by
    have h_integral_le_essSup_mul_measure : ∀ᵐ x ∂MeasureTheory.MeasureSpace.volume, f x ≤ essSup f MeasureTheory.MeasureSpace.volume := by
      exact ENNReal.ae_le_essSup f
    generalize_proofs at *; (
    have h_integral_restrict : ∫⁻ x, f x ∂MeasureTheory.MeasureSpace.volume = ∫⁻ x in Function.support f, f x ∂MeasureTheory.MeasureSpace.volume := by
      exact (MeasureTheory.setLIntegral_eq_of_support_subset (fun x hx => hx)).symm
    generalize_proofs at *; (
    have h_integral_le_essSup_mul_measure : ∫⁻ x in Function.support f, f x ∂MeasureTheory.MeasureSpace.volume ≤ ∫⁻ x in Function.support f, essSup f MeasureTheory.MeasureSpace.volume ∂MeasureTheory.MeasureSpace.volume := by
      apply_rules [ MeasureTheory.lintegral_mono_ae ];
      exact MeasureTheory.ae_restrict_of_ae h_integral_le_essSup_mul_measure
    generalize_proofs at *; (
    simpa [ mul_comm ] using h_integral_restrict.le.trans h_integral_le_essSup_mul_measure)));
  exact h_integral_le_essSup_mul_measure.trans ( mul_le_mul_left' ( MeasureTheory.measure_mono h_supp ) _ )

/-- eLpNorm lower bound from integral / measure for nonneg functions. -/
theorem eLpNorm_ge_integral_div_measure_real {α : Type*} [MeasureTheory.MeasureSpace α]
    (f : α → ℝ) (hf : ∀ x, 0 ≤ f x) (S : Set α) (h_supp : Function.support f ⊆ S)
    (hS_fin : MeasureTheory.volume S ≠ ⊤) (hS_pos : MeasureTheory.volume S ≠ 0)
    (hf_int : MeasureTheory.Integrable f MeasureTheory.volume)
    (h_fin : MeasureTheory.eLpNorm f ⊤ MeasureTheory.volume ≠ ⊤) :
    (MeasureTheory.eLpNorm f ⊤ MeasureTheory.volume).toReal ≥
    (MeasureTheory.integral MeasureTheory.volume f) / (MeasureTheory.volume S).toReal := by
      refine' div_le_iff₀ ( ENNReal.toReal_pos _ _ ) |>.2 _;
      · exact hS_pos;
      · exact hS_fin;
      · have h_integral_le : ∫⁻ x, ENNReal.ofReal (f x) ∂MeasureTheory.volume ≤ (essSup (fun x => ENNReal.ofReal (f x)) MeasureTheory.volume) * (MeasureTheory.volume S) := by
          convert lintegral_le_essSup_mul_measure_ennreal _ _ _ using 1 ; aesop ( simp_config := { singlePass := true } ) ;
        convert ENNReal.toReal_mono _ h_integral_le using 1 <;> norm_num [ MeasureTheory.eLpNormEssSup ];
        · rw [ MeasureTheory.integral_eq_lintegral_of_nonneg_ae ];
          · exact Filter.Eventually.of_forall hf;
          · exact hf_int.1;
        · simp +decide [ Real.enorm_eq_ofReal ( hf _ ) ];
        · refine' ENNReal.mul_ne_top _ _ <;> simp_all +decide [ MeasureTheory.eLpNormEssSup ];
          simp_all +decide [ ENNReal.ofReal, Real.enorm_eq_ofReal ( hf _ ) ]

end -- noncomputable section
