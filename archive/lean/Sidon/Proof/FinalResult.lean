/-
Sidon Autocorrelation Project — Final Result

The main theorem: autoconvolution_ratio f ≥ 32/25 for all admissible f.
Uses the computational axiom (cascade_all_pruned) plus all preceding theory.
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.StepFunction
import Sidon.Proof.TestValueBounds
import Sidon.Proof.DiscretizationError
import Sidon.Proof.RefinementBridge

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
-- Final Result — Autoconvolution Constant Lower Bound
-- ═══════════════════════════════════════════════════════════════════════════════

-- *** COMPUTATIONAL AXIOM ***
-- The following axiom encodes the result of the fine-grid cascade.
-- Every other lemma and theorem in the Lean formalization is fully proved;
-- this axiom and cs_lemma3_per_window (C&S Lemma 3, in DiscretizationError.lean)
-- are the only unverified-in-Lean components.
--
-- The threshold uses the C&S Lemma 3 correction (2/m + 1/m²) with the fine
-- grid B_{n,m}: d=2n bins, S=4nm integers, heights a_i = c_i/m.
--
-- This uses CascadePruned: an inductive predicate that captures the multi-level
-- cascade structure. A composition is cascade-pruned if either:
--   (direct) some window has TV > threshold, OR
--   (refine) ALL valid children at the next resolution are cascade-pruned.
--
-- The old (INCORRECT) axiom claimed every d=4 composition was directly prunable.
-- This is false: [40,40,40,40] has max TV=1.25 < 1.3825. The new axiom correctly
-- says [40,40,40,40] is cascade-pruned via `refine` — all its d=8 children are
-- eventually pruned at higher levels.
--
-- Reproduction:
--   python -m cloninger-steinerberger.cpu.run_cascade \
--     --n_half 2 --m 20 --c_target 1.28 --use_flat_threshold --verify_relaxed
-- The cascade terminates with 0 survivors across all levels and all ±1 rounding
-- variants. This verifies the CascadePruned property with relaxed child constraint.
--
-- Fine grid: compositions sum to S = 4*n*m = 4*2*20 = 160.
-- Correction: 2/m + 1/m² = 2/20 + 1/400 = 0.1025.
/-- **Computational axiom**: The fine-grid cascade with parameters
    n_half=2, m=20, c_target=32/25 terminated with zero survivors.

    Every composition of S=160 into d=4 bins is CascadePruned:
    either directly prunable (TV > 32/25 + 2/20 + 1/400) or all its
    valid children (allowing ±1 floor rounding) are cascade-pruned.

    Reproduction:
      python -m cloninger-steinerberger.cpu.run_cascade \
        --n_half 2 --m 20 --c_target 1.28 --use_flat_threshold --verify_relaxed -/
axiom cascade_all_pruned :
  ∀ c : Fin (2 * 2) → ℕ, ∑ i, c i = 4 * 2 * 20 →
    CascadePruned 20 (32/25 : ℝ) (2 / 20 + 1 / 20 ^ 2) 2 c

/-- If a composition is cascade-pruned, then every continuous function
    whose canonical discretization matches it has R(f) ≥ c_target.

    Proof by induction on the CascadePruned derivation:
    - direct: f's discretization has high TV, so R(f) ≥ c_target
      by dynamic_threshold_sound_cs (already proved)
    - refine: f also discretizes at the finer grid to some child of c.
      That child is cascade-pruned (by hypothesis). Apply induction. -/
theorem cascade_pruned_implies_bound
    (n_half m : ℕ) (c_target : ℝ) (c : Fin (2 * n_half) → ℕ)
    (hn : n_half > 0) (hm : m > 0) (hct : 0 < c_target)
    (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (hdisc : canonical_discretization f n_half m = c)
    (hpruned : CascadePruned m c_target (2 / ↑m + 1 / ↑m ^ 2) n_half c) :
    autoconvolution_ratio f ≥ c_target := by
  /-
  Proof by induction on the CascadePruned derivation.

  direct case: f's discretization has high TV, so R(f) ≥ c_target by
  dynamic_threshold_sound_cs (already proved in DiscretizationError.lean).

  refine case: All valid children at 2*n_half are cascade-pruned.
  By refinement_preserves_discretization, canonical_discretization f (2*n_half) m
  is a valid child of c = canonical_discretization f n_half m.
  So canonical_discretization f (2*n_half) m is cascade-pruned.
  By the induction hypothesis (applied at 2*n_half with hdisc' := rfl),
  R(f) ≥ c_target.

  The induction terminates because CascadePruned is an inductive type:
  each `refine` constructor stores strictly smaller sub-proofs.
  -/
  induction hpruned with
  | direct h =>
    obtain ⟨ℓ, s_lo, hℓ, h_exc⟩ := h
    exact dynamic_threshold_sound_cs _ m c_target hn hm hct _ ℓ s_lo hℓ (by linarith)
      f hf_nonneg hf_supp hf_int h_conv_fin hdisc
  | refine h ih =>
    have h_rpd := refinement_preserves_discretization f _ m hn hm hf_nonneg hf_supp hf_int
    rw [hdisc] at h_rpd
    exact ih _ h_rpd (by omega) rfl

/-- Scale invariance of the autoconvolution ratio.
    R(a·f) = R(f) for a > 0. -/
theorem autoconvolution_ratio_scale_invariant (f : ℝ → ℝ) (a : ℝ) (ha : 0 < a) :
    autoconvolution_ratio (fun x => a * f x) = autoconvolution_ratio f := by
  unfold autoconvolution_ratio
  dsimp only []
  have h_conv : MeasureTheory.convolution (fun x => a * f x) (fun x => a * f x)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume =
      fun x => a ^ 2 * MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
    ext x; simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
    simp only [mul_comm (a) (f _), ← mul_assoc]
    rw [← MeasureTheory.integral_const_mul]
    congr 1; ext t; ring
  rw [h_conv]
  have h_norm : (MeasureTheory.eLpNorm (fun x => a ^ 2 * MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x) ⊤ MeasureTheory.volume).toReal =
      a ^ 2 * (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal := by
    have ha2 : 0 < a ^ 2 := by positivity
    have ha2_ne : a ^ 2 ≠ 0 := ne_of_gt ha2
    have : (fun x => a ^ 2 * MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ)
        MeasureTheory.volume x) = a ^ 2 • (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) := by
      ext x; simp [Pi.smul_apply, smul_eq_mul]
    rw [this, MeasureTheory.eLpNorm_const_smul, ENNReal.toReal_mul,
        Real.enorm_eq_ofReal (le_of_lt ha2), ENNReal.toReal_ofReal (le_of_lt ha2)]
  have h_int : MeasureTheory.integral MeasureTheory.volume (fun x => a * f x) =
      a * MeasureTheory.integral MeasureTheory.volume f := by
    exact MeasureTheory.integral_const_mul a f
  rw [h_norm, h_int]
  have ha2 : a ^ 2 ≠ 0 := pow_ne_zero 2 (ne_of_gt ha)
  have ha_ne : a ≠ 0 := ne_of_gt ha
  field_simp [ha_ne, ha2]

/-- **Main theorem**: Every nonneg function f supported on (-1/4, 1/4) with positive
    integral and finite ‖f*f‖_∞ satisfies ‖f*f‖_∞ / (∫f)² ≥ 32/25 = 1.28.

    The hypothesis h_conv_fin is necessary because autoconvolution_ratio uses
    ENNReal.toReal, which maps ⊤ to 0. For f ∈ L¹ \ L² (e.g., f(x) ~ |x|^{-3/4}),
    ‖f*f‖_∞ = ∞ and the mathematical ratio is ∞ ≥ 32/25, but the Lean-computed ratio
    would be 0. This hypothesis holds for all bounded, L², or step functions.

    Proof: Normalize f to g with ∫g = 1, discretize g at resolution n=2 with m=20,
    apply cascade_all_pruned to find a killing window (ℓ, s_lo) where TV exceeds the
    C&S threshold, then apply dynamic_threshold_sound_cs to conclude R(g) ≥ 32/25. -/
lemma eLpNorm_convolution_scale_ne_top (f : ℝ → ℝ) (a : ℝ)
    (h_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    MeasureTheory.eLpNorm (MeasureTheory.convolution (fun x => a * f x) (fun x => a * f x)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ := by
  have h_eq : MeasureTheory.convolution (fun x => a * f x) (fun x => a * f x)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume =
      a ^ 2 • MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume := by
    ext x; simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply',
      Pi.smul_apply, smul_eq_mul]
    simp only [mul_comm a (f _), ← mul_assoc]
    rw [← MeasureTheory.integral_const_mul]; congr 1; ext t; ring
  rw [h_eq, MeasureTheory.eLpNorm_const_smul]
  exact ENNReal.mul_ne_top ENNReal.coe_ne_top h_fin

theorem autoconvolution_ratio_ge_32_25 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ 32/25 := by
  set I := MeasureTheory.integral MeasureTheory.volume f with hI_def
  set g := fun x => (1/I) * f x with hg_def
  have hI_pos : 0 < I := hf_int_pos
  have h_ratio_eq : autoconvolution_ratio f = autoconvolution_ratio g := by
    rw [hg_def]
    exact (autoconvolution_ratio_scale_invariant f (1/I) (by positivity)).symm
  rw [h_ratio_eq]
  have hg_nonneg : ∀ x, 0 ≤ g x := by
    intro x; simp only [hg_def]; exact mul_nonneg (by positivity) (hf_nonneg x)
  have hg_supp : Function.support g ⊆ Set.Ioo (-1/4 : ℝ) (1/4) := by
    intro x hx; apply hf_supp; rw [Function.mem_support] at hx ⊢
    intro h; exact hx (by simp only [hg_def, h, mul_zero])
  have hg_int : MeasureTheory.integral MeasureTheory.volume g = 1 := by
    simp only [hg_def, MeasureTheory.integral_const_mul]
    rw [← hI_def]; exact div_mul_cancel₀ 1 (ne_of_gt hI_pos)
  set c := canonical_discretization g 2 20
  have h_mass_nz : ∑ j : Fin (2 * 2), bin_masses g 2 j ≠ 0 := by
    rw [sum_bin_masses_eq_one 2 (by norm_num) g hg_supp hg_int]; exact one_ne_zero
  have hc_sum : ∑ i, c i = 4 * 2 * 20 :=
    canonical_discretization_sum_eq_m g 2 20 (by norm_num) (by norm_num) h_mass_nz hg_nonneg
  have hpruned := cascade_all_pruned c hc_sum
  have h_conv_fin_g : MeasureTheory.eLpNorm (MeasureTheory.convolution g g
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ :=
    eLpNorm_convolution_scale_ne_top f (1/I) h_conv_fin
  exact cascade_pruned_implies_bound 2 20 (32/25 : ℝ) c (by norm_num) (by norm_num)
    (by norm_num : (0:ℝ) < 32/25) g hg_nonneg hg_supp hg_int h_conv_fin_g rfl hpruned

end -- noncomputable section
