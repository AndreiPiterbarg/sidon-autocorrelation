/-
Sidon Autocorrelation Project — W-Refined Discretization Error Bound

This file proves that the W-refined correction from C&S equation (1) is sound,
enabling tighter pruning thresholds in the cascade algorithm.

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL BACKGROUND (Cloninger & Steinerberger, arXiv:1403.7988)
═══════════════════════════════════════════════════════════════════════════════

C&S Lemma 2: The step function g with heights a_i = c_i/m satisfies
  |g(x) - f(x)| ≤ 1/m  a.e. on each bin (pointwise rounding error).

C&S equation (1): Using Lemma 2 pointwise in the convolution integral:
  (g*g)(x) = ∫ g(t)·g(x-t) dt
           = ∫ (f(t) + ε(t))·(f(x-t) + ε(x-t)) dt
           = (f*f)(x) + ∫ f(t)·ε(x-t) dt + ∫ ε(t)·f(x-t) dt + ∫ ε(t)·ε(x-t) dt
  where ε = g - f with |ε| ≤ 1/m.

  The cross terms satisfy:
    |∫ f(t)·ε(x-t) dt| ≤ (1/m)·∫_{supp(g)∩(x-supp(g))} f(t) dt
                        = (1/m)·W_f(x)
  where W_f(x) is the total mass of f in bins overlapping the integration
  region at point x.  Similarly for the other cross term.

  The ε² term: |∫ ε(t)·ε(x-t) dt| ≤ (1/m²)·meas(supp) ≤ 1/m²
  (since supp ⊆ [-1/4, 1/4] has measure 1/2, and we normalize).

  Therefore: |(g*g)(x) - (f*f)(x)| ≤ 2·W_f(x)/m + 1/m²

C&S Lemma 3 (FLAT bound): Since W_f(x) ≤ ∫f = 1:
  |(g*g)(x) - (f*f)(x)| ≤ 2/m + 1/m²
  This is the flat bound currently used in the Lean axiom.

THE W-REFINED BOUND: At convolution knot points x_k = -1/4 + k/(4n),
  the integration region aligns with bin boundaries, so W_f(x_k) = W_g(x_k)
  exactly.  In integer coordinates: W_g(x_k) = W_int/(4nm), so:
    correction = 2·W_int/(4nm·m) + 1/m² = W_int/(2n·m²) + 1/m²
               = (1 + W_int/(2n)) / m²

  Since W_int ≤ S = 4nm, we have W_int/(2n) ≤ 2m, recovering the flat bound.
  But for windows where only a few bins contribute, W_int ≪ S and the
  correction is much smaller — typically 3-5× tighter.

═══════════════════════════════════════════════════════════════════════════════
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.StepFunction
import Sidon.Proof.TestValueBounds
import Sidon.Proof.DiscretizationError
import Sidon.Proof.RefinementBridge
import Sidon.Proof.FinalResult
import Sidon.Proof.WRefinedDefs
import Sidon.Proof.TightDiscretizationBound
import Sidon.Proof.TightContinuousBridge

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
-- Part 1 + 2 (definitions): see `Sidon.Proof.WRefinedDefs` for
--   `W_int_for_window` and `w_refined_correction`.  Factored out to break the
--   import cycle:  WRefinedBound  →  TightContinuousBridge  →  WRefinedBound.
-- ═══════════════════════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: C&S Equation (1) — The W-Refined Axiom
--
-- This replaces cs_lemma3_per_window with a TIGHTER bound.
-- The flat bound (2/m + 1/m²) follows as a corollary (Part 4).
--
-- Justification: C&S arXiv:1403.7988, equation (1), peer-reviewed.
-- The derivation is given in the header comment.  Full formalization would
-- require ~300 lines of piecewise integration showing:
--   (a) |ε(t)| = |g(t) - f(t)| ≤ 1/m  a.e. (C&S Lemma 2)
--   (b) Cross-term: |∫ f(t)·ε(x-t) dt| ≤ (1/m)·W_f(x)
--   (c) At knot points: W_f(x_k) = W_g(x_k) (bin boundary alignment)
--   (d) W_g(x_k) = W_int/(4nm)
--   (e) Averaging over a window preserves the pointwise bound
-- Each step is individually straightforward but requires significant
-- Mathlib integration infrastructure.
-- ═══════════════════════════════════════════════════════════════════════════════

-- **`cs_eq1_w_refined` is now a THEOREM**, not an axiom.  See
-- `Sidon.Proof.TightContinuousBridge` (theorem `cs_eq1_w_refined`), where it
-- is derived from the new axiom `cs_eq1_tight` plus the universal D-v2
-- dominance `corr_tight_m2_le_corr_w_refined` plus the bijection
-- `W_int_overlap_eq_for_window`.  Project axiom count: 4 (was 5).
--
-- Downstream theorems in this file (`correction_term_bound_w_refined`,
-- `dynamic_threshold_sound_w_refined`, etc.) pick up `cs_eq1_w_refined` by
-- name from `TightContinuousBridge`, which is imported above.

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 4: W-Refined ≤ Flat (Backward Compatibility)
-- ═══════════════════════════════════════════════════════════════════════════════

-- (w_refined_correction_le_flat and cs_lemma3_from_w_refined removed in cleanup:
--  both depended on the now-removed W_int_le_total (sorry-only) and were
--  dead-code helpers. The flat axiom cs_lemma3_per_window and the W-refined
--  axiom cs_eq1_w_refined remain as independent stated axioms.)

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 5: W-Refined Correction Term Bound
-- ═══════════════════════════════════════════════════════════════════════════════

/-- W-refined correction bound: R(f) ≥ TV(c,ℓ,s) - w_correction.
    Analogous to correction_term_bound_cs but with tighter correction. -/
theorem correction_term_bound_w_refined (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    autoconvolution_ratio f ≥
      test_value n m (canonical_discretization f n m) ℓ s_lo -
        w_refined_correction n m (canonical_discretization f n m) ℓ s_lo := by
  have h_cont : autoconvolution_ratio f ≥ test_value_continuous n f ℓ s_lo :=
    continuous_test_value_le_ratio n hn f hf_nonneg hf_supp hf_int h_conv_fin ℓ s_lo hℓ
  have h_disc := cs_eq1_w_refined n m hn hm f hf_nonneg hf_supp hf_int ℓ s_lo hℓ
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 6: W-Refined Dynamic Threshold Soundness
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Dynamic threshold soundness with W-refined correction.**
    If TV > c_target + w_correction for SOME window, then R(f) ≥ c_target.

    This is the key theorem enabling tighter pruning in the cascade.
    The correction depends on the specific window (ℓ, s_lo) and composition c,
    so different windows have different thresholds.

    Matches CPU code: solvers.py lines 140-142
      corr_w = 1.0 + W_int / (2.0 * n_half)
      dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
      dyn_it = floor(dyn_x)

    Matches GPU code: cascade_kernel.cu inline_threshold()
      threshold = floor(A_ell + 2*ell*W_int) -/
theorem dynamic_threshold_sound_w_refined (n m : ℕ) (c_target : ℝ)
    (hn : n > 0) (hm : m > 0) (_hct : 0 < c_target)
    (c : Fin (2 * n) → ℕ)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (h_exceeds : test_value n m c ℓ s_lo > c_target +
      w_refined_correction n m c ℓ s_lo) :
    ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) →
      Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
      MeasureTheory.integral MeasureTheory.volume f = 1 →
      MeasureTheory.eLpNorm (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ →
      canonical_discretization f n m = c →
      autoconvolution_ratio f ≥ c_target := by
  intro f hf_nonneg hf_supp hf_int h_conv_fin hdisc
  have hbound := correction_term_bound_w_refined n m hn hm f hf_nonneg hf_supp
    hf_int h_conv_fin ℓ s_lo hℓ
  rw [hdisc] at hbound
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 7: CascadePrunedW — Cascade with Per-Window W-Refined Correction
--
-- The key difference from CascadePruned: the `direct` constructor uses
-- the W-refined correction (which depends on c, ℓ, s_lo) instead of a
-- single fixed correction for all windows.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- A composition is W-refined cascade-pruned if either:
    (direct) some window has TV > c_target + w_correction(window), OR
    (refine) ALL valid children at the next resolution are W-cascade-pruned.

    This is strictly more powerful than CascadePruned with flat correction:
    the per-window threshold is lower, so more compositions are directly pruned
    and fewer need to be refined to the next level. -/
inductive CascadePrunedW (m : ℕ) (c_target : ℝ) :
    (n_half : ℕ) → (Fin (2 * n_half) → ℕ) → Prop where
  | direct {n_half : ℕ} {c : Fin (2 * n_half) → ℕ}
      (h : ∃ ℓ s_lo, 2 ≤ ℓ ∧
        test_value n_half m c ℓ s_lo > c_target +
          w_refined_correction n_half m c ℓ s_lo) :
      CascadePrunedW m c_target n_half c
  | refine {n_half : ℕ} {c : Fin (2 * n_half) → ℕ}
      (h : ∀ child : Fin (2 * (2 * n_half)) → ℕ,
        is_valid_child n_half c child →
        CascadePrunedW m c_target (2 * n_half) child) :
      CascadePrunedW m c_target n_half c

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 8: CascadePrunedW Implies the Bound
-- ═══════════════════════════════════════════════════════════════════════════════

/-- If a composition is W-refined cascade-pruned, then every continuous function
    whose canonical discretization matches it has R(f) ≥ c_target.

    Proof by induction on the CascadePrunedW derivation:
    - direct: f's discretization has high TV in some window, so R(f) ≥ c_target
      by dynamic_threshold_sound_w_refined.
    - refine: f also discretizes at the finer grid to some child of c.
      That child is CascadePrunedW (by hypothesis). Apply induction. -/
theorem cascade_pruned_w_implies_bound
    (n_half m : ℕ) (c_target : ℝ) (c : Fin (2 * n_half) → ℕ)
    (hn : n_half > 0) (hm : m > 0) (hct : 0 < c_target)
    (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (hdisc : canonical_discretization f n_half m = c)
    (hpruned : CascadePrunedW m c_target n_half c) :
    autoconvolution_ratio f ≥ c_target := by
  induction hpruned with
  | direct h =>
    obtain ⟨ℓ, s_lo, hℓ, h_exc⟩ := h
    exact dynamic_threshold_sound_w_refined _ m c_target hn hm hct _ ℓ s_lo hℓ (by linarith)
      f hf_nonneg hf_supp hf_int h_conv_fin hdisc
  | refine h ih =>
    have h_rpd := refinement_preserves_discretization f _ m hn hm hf_nonneg hf_supp hf_int
    rw [hdisc] at h_rpd
    exact ih _ h_rpd (by omega) rfl

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 9: Flat-Pruned Implies W-Pruned (Backward Compatibility)
-- ═══════════════════════════════════════════════════════════════════════════════

-- (cascade_pruned_flat_implies_w removed in cleanup: dead-code helper marked
--  sorry, never invoked by any published theorem. The flat cascade and
--  W-refined cascade are computed independently in Python; each has its
--  own computational axiom (cascade_all_pruned, cascade_all_pruned_w).)

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 10: Computational Axiom with W-Refined Threshold
--
-- This is the NEW computational axiom, replacing cascade_all_pruned.
-- It uses the W-refined threshold, which is easier to satisfy computationally
-- (more compositions are directly prunable), enabling:
--   (a) Faster cascade convergence (fewer levels needed)
--   (b) Higher target bounds (c_target up to ~1.40 becomes feasible)
--
-- Reproduction:
--   python -m cloninger-steinerberger.cpu.run_cascade \
--     --n_half 2 --m 20 --c_target 1.28 --verify_relaxed
--   (NOTE: do NOT use --use_flat_threshold; W-refined is the default)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Computational axiom (W-refined)**: The W-refined cascade with parameters
    n_half=2, m=20, c_target=32/25 terminates with zero survivors.

    Every composition of S=160 into d=4 bins is CascadePrunedW:
    either directly prunable (TV > 32/25 + w_correction) or all its
    valid children (allowing ±1 floor rounding) are W-cascade-pruned.

    This is EASIER to verify computationally than cascade_all_pruned because
    the W-refined threshold is lower (more compositions pruned directly).

    Reproduction:
      python -m cloninger-steinerberger.cpu.run_cascade \
        --n_half 2 --m 20 --c_target 1.28 --verify_relaxed -/
axiom cascade_all_pruned_w :
  ∀ c : Fin (2 * 2) → ℕ, ∑ i, c i = 4 * 2 * 20 →
    CascadePrunedW 20 (32/25 : ℝ) 2 c

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 11: Main Theorem (W-Refined Version)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Main theorem (W-refined)**: R(f) ≥ 32/25 = 1.28 for all admissible f.

    Uses the W-refined computational axiom and W-refined threshold soundness.
    This is mathematically equivalent to autoconvolution_ratio_ge_32_25 but
    uses the tighter W-refined machinery throughout.

    The proof structure mirrors FinalResult.lean:
    1. Normalize f to g with ∫g = 1
    2. Discretize g at n=2, m=20
    3. Apply cascade_all_pruned_w
    4. Apply cascade_pruned_w_implies_bound -/
theorem autoconvolution_ratio_ge_32_25_w_refined (f : ℝ → ℝ)
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
  have hpruned := cascade_all_pruned_w c hc_sum
  have h_conv_fin_g : MeasureTheory.eLpNorm (MeasureTheory.convolution g g
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ :=
    eLpNorm_convolution_scale_ne_top f (1/I) h_conv_fin
  exact cascade_pruned_w_implies_bound 2 20 (32/25 : ℝ) c (by norm_num) (by norm_num)
    (by norm_num : (0:ℝ) < 32/25) g hg_nonneg hg_supp hg_int h_conv_fin_g rfl hpruned

end -- noncomputable section
