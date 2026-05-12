/-
Sidon Autocorrelation Project — Coarse Cascade Method

Formalizes the no-correction approach to proving C₁ₐ ≥ c_target.

PROVEN IN THIS FILE:
  • mass_test_value_le_ratio — Theorem 1: R(f) ≥ TV_W(bin_masses(f), ℓ, s)
    with NO correction term. The central result of the coarse cascade.
  • voronoi_coverage — Every point μ ∈ Δ_d is within L∞ distance < 1/S
    of some integer composition of S into d parts.
  • coarse_cascade_bound — C₁ₐ ≥ c_target, from Theorem 1 + simplex coverage.

AXIOMS (clearly marked, each with justification):
  • refinement_monotonicity — Unproven mathematical CONJECTURE:
    max_W TV_W(ν; 2d) ≥ max_W TV_W(μ; d) for any refinement ν of μ.
    Empirically verified (183K+ tests, zero violations) but NOT PROVEN.
    This axiom is needed to justify the cascade's subtree pruning.
    WITHOUT A PROOF OF THIS CONJECTURE, the overall result is CONDITIONAL.

  • simplex_tv_coverage — Computational result:
    ∀ μ ∈ Δ_{d_K}, ∃ window (ℓ, s) with TV_W(μ) ≥ c_target.
    Verified by: cascade (all compositions pruned, using refinement_monotonicity
    to skip subtrees) + box certification (continuous coverage of Voronoi cells).
    This is a COMPUTATIONAL AXIOM analogous to cascade_all_pruned in FinalResult.lean.
-/

import Sidon.Proof.TestValueBounds
import Sidon.Proof.FinalResult

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 16000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 40000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART 1: DEFINITIONS
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Mass-based test value: TV_W(μ, ℓ, s) = (2d/ℓ) · ∑_{k∈[s, s+ℓ-2]} ∑_{i+j=k} μ_i·μ_j.

    This equals the existing `test_value_continuous` when μ = bin_masses(f, n) and d = 2n.
    NO height quantization, step-function approximation, or correction term is involved.
    The bound depends only on bin masses of f, not on the shape of f within each bin. -/
noncomputable def mass_test_value (d : ℕ) (μ : Fin d → ℝ) (ℓ s : ℕ) : ℝ :=
  (2 * (d : ℝ) / (ℓ : ℝ)) *
    ∑ k ∈ Finset.Icc s (s + ℓ - 2), discrete_autoconvolution μ k

/-- A vector μ lies on the probability simplex Δ_d: all entries nonneg, sum = 1. -/
def on_simplex {d : ℕ} (μ : Fin d → ℝ) : Prop :=
  (∀ i, 0 ≤ μ i) ∧ (∑ i, μ i = 1)

/-- ν ∈ ℝ^{2d} is a mass-refinement of μ ∈ ℝ^d:
    adjacent pairs in ν sum to the parent entry in μ, and all entries are nonneg. -/
def is_mass_refinement {d : ℕ} (μ : Fin d → ℝ) (ν : Fin (2 * d) → ℝ) : Prop :=
  (∀ i : Fin d,
    ν ⟨2 * i.val, by omega⟩ + ν ⟨2 * i.val + 1, by omega⟩ = μ i) ∧
  (∀ j, 0 ≤ ν j)

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART 2: THEOREM 1 — Mass-Based Test Value Bound (FULLY PROVEN)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Scaling: discrete_autoconvolution(c · μ) = c² · discrete_autoconvolution(μ). -/
lemma discrete_autoconvolution_scale {d : ℕ} (μ : Fin d → ℝ) (c : ℝ) (k : ℕ) :
    discrete_autoconvolution (fun i => c * μ i) k =
    c ^ 2 * discrete_autoconvolution μ k := by
  unfold discrete_autoconvolution
  simp only [Finset.mul_sum]
  congr 1; ext i; congr 1; ext j
  split_ifs <;> ring

/-- Algebraic identity: test_value_continuous equals mass_test_value on bin masses.

    TV_continuous(n, f, ℓ, s)
      = (1/(4nℓ)) · ∑ DA(4n·μ, k)          [definition, with a_i = 4n·μ_i]
      = (1/(4nℓ)) · (4n)² · ∑ DA(μ, k)     [scaling lemma]
      = (4n/ℓ) · ∑ DA(μ, k)                 [simplify]
      = (2·(2n)/ℓ) · ∑ DA(μ, k)             [rewrite 4n = 2·2n]
      = mass_test_value(2n, μ, ℓ, s)         [definition] -/
theorem test_value_continuous_eq_mass_tv (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (ℓ s : ℕ) :
    test_value_continuous n f ℓ s =
    mass_test_value (2 * n) (bin_masses f n) ℓ s := by
  unfold test_value_continuous mass_test_value
  simp only []
  -- Pull out the (4n)² scaling factor from discrete_autoconvolution
  simp_rw [discrete_autoconvolution_scale (bin_masses f n) (4 * (n : ℝ)),
           ← Finset.mul_sum]
  -- Arithmetic: (1/(4nℓ)) · (4n)² = 2·(2n)/ℓ
  have hn_ne : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  push_cast
  by_cases hℓ : (ℓ : ℝ) = 0
  · simp [hℓ]
  · field_simp
    ring

/-- **Theorem 1** (No-Correction Test Value Bound).

    For any nonneg f with support ⊆ (-1/4, 1/4), ∫f = 1, ‖f*f‖_∞ < ∞:
      autoconvolution_ratio(f) ≥ mass_test_value(2n, bin_masses(f, n), ℓ, s)

    There is NO correction term — the bound uses bin masses directly.
    This eliminates the C&S correction (2/m + 1/m²) entirely.

    Proof: mass_test_value = test_value_continuous (algebraic identity above),
    and autoconvolution_ratio(f) ≥ test_value_continuous (proved in TestValueBounds.lean
    via Fubini's theorem + Minkowski sum containment + averaging bound). -/
theorem mass_test_value_le_ratio (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (ℓ s : ℕ) (hℓ : 2 ≤ ℓ) :
    autoconvolution_ratio f ≥ mass_test_value (2 * n) (bin_masses f n) ℓ s := by
  rw [← test_value_continuous_eq_mass_tv n hn f ℓ s]
  exact continuous_test_value_le_ratio n hn f hf_nonneg hf_supp hf_int h_conv_fin ℓ s hℓ

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART 3: VORONOI COVERAGE (FULLY PROVEN)
-- ═══════════════════════════════════════════════════════════════════════════════

/-! ### Cumulative partial sums and integer rounding

We prove that every point on the probability simplex Δ_d can be approximated
by an integer composition of S into d parts, with L∞ error < 1/S.

The construction uses cumulative rounding: define D(k) = ⌊S · ∑_{j<k} μ_j⌋
and set c_i = D(i+1) - D(i). This telescopes to ∑c_i = S and each
|c_i - S·μ_i| < 1 by the fractional-part argument. -/

/-- Partial sum of first k entries of μ. -/
private noncomputable def psum {d : ℕ} (μ : Fin d → ℝ) (k : ℕ) : ℝ :=
  ∑ j : Fin d, if j.val < k then μ j else 0

private lemma psum_zero {d : ℕ} (μ : Fin d → ℝ) : psum μ 0 = 0 := by
  unfold psum; apply Finset.sum_eq_zero; intro j _; simp [show ¬(j.val < 0) from by omega]

private lemma psum_full {d : ℕ} (μ : Fin d → ℝ) : psum μ d = ∑ i, μ i := by
  unfold psum; congr 1; ext j; simp [show j.val < d from j.isLt]

private lemma psum_mono {d : ℕ} (μ : Fin d → ℝ) (hμ : ∀ i, 0 ≤ μ i) :
    Monotone (psum μ) := by
  intro a b hab; unfold psum
  apply Finset.sum_le_sum; intro j _
  split_ifs with ha hb
  · exact le_refl _
  · exfalso; exact hb (Nat.lt_of_lt_of_le ha hab)
  · exact hμ j
  · exact le_refl _

/-- Splitting: psum(k+1) = psum(k) + μ_k for k < d. -/
private lemma psum_succ {d : ℕ} (μ : Fin d → ℝ) (k : ℕ) (hk : k < d) :
    psum μ (k + 1) = psum μ k + μ ⟨k, hk⟩ := by
  unfold psum
  -- Split: (j < k+1) ↔ (j < k) ∨ (j = k)
  trans (∑ j : Fin d, ((if j.val < k then μ j else 0) + (if j.val = k then μ j else 0)))
  · congr 1; ext j; split_ifs with h1 h2 h3 <;> simp_all <;> omega
  rw [Finset.sum_add_distrib]
  congr 1
  rw [Finset.sum_eq_single_of_mem ⟨k, hk⟩ (Finset.mem_univ _)]
  · simp
  · intro b _ hb; simp [show b.val ≠ k from fun h => hb (Fin.ext h)]

/-- Integer rounding of S · psum via floor, taking differences. -/
private noncomputable def round_int {d : ℕ} (S : ℕ) (μ : Fin d → ℝ) (i : Fin d) : ℤ :=
  ⌊(S : ℝ) * psum μ (i.val + 1)⌋ - ⌊(S : ℝ) * psum μ i.val⌋

/-- Each rounded value is nonneg (from monotonicity of psum). -/
private lemma round_int_nonneg {d : ℕ} (S : ℕ) (μ : Fin d → ℝ)
    (hμ : ∀ i, 0 ≤ μ i) (i : Fin d) : 0 ≤ round_int S μ i := by
  unfold round_int
  have h_mono := psum_mono μ hμ (Nat.le_succ i.val)
  have : ⌊(S : ℝ) * psum μ i.val⌋ ≤ ⌊(S : ℝ) * psum μ (i.val + 1)⌋ :=
    Int.floor_le_floor (mul_le_mul_of_nonneg_left h_mono (Nat.cast_nonneg S))
  omega

/-- Sum of rounded values equals S (telescoping argument). -/
private lemma round_int_sum {d : ℕ} (S : ℕ) (μ : Fin d → ℝ)
    (_hμ_nn : ∀ i, 0 ≤ μ i) (hμ_sum : ∑ i, μ i = 1) :
    ∑ i : Fin d, round_int S μ i = (S : ℤ) := by
  -- Telescope: ∑ (⌊S·ps(i+1)⌋ - ⌊S·ps(i)⌋) = ⌊S·ps(d)⌋ - ⌊S·ps(0)⌋
  set F : ℕ → ℤ := fun k => ⌊(S : ℝ) * psum μ k⌋ with hF_def
  have h_eq : ∀ i : Fin d, round_int S μ i = F (↑i + 1) - F ↑i := fun _ => rfl
  simp_rw [h_eq]
  rw [Fin.sum_univ_eq_sum_range (fun k => F (k + 1) - F k) d,
      Finset.sum_range_sub F]
  -- F(d) = ⌊S · 1⌋ = S and F(0) = ⌊S · 0⌋ = 0
  simp only [hF_def, psum_full, hμ_sum, mul_one, psum_zero, mul_zero,
             Int.floor_natCast, Int.floor_zero, sub_zero]

/-- Error bound: |round_int(i) - S·μ_i| < 1 (fractional-part argument).

    Key identity: round_int(i) - S·μ_i = fract(S·ps(i)) - fract(S·ps(i+1)).
    Since both fractional parts are in [0, 1), their difference has |·| < 1. -/
private lemma round_int_error {d : ℕ} (S : ℕ) (μ : Fin d → ℝ)
    (_hμ_nn : ∀ i, 0 ≤ μ i) (i : Fin d) :
    |(↑(round_int S μ i) : ℝ) - (S : ℝ) * μ i| < 1 := by
  unfold round_int
  -- S·μ_i = S·ps(i+1) - S·ps(i) from psum_succ
  have h_diff : (S : ℝ) * μ i = (S : ℝ) * psum μ (i.val + 1) - (S : ℝ) * psum μ i.val := by
    rw [psum_succ μ i.val i.isLt]; ring
  rw [h_diff]
  -- error = (⌊a⌋ - ⌊b⌋) - (a - b) = fract(b) - fract(a)
  set a := (S : ℝ) * psum μ (i.val + 1)
  set b := (S : ℝ) * psum μ i.val
  have h_rewrite : (↑⌊a⌋ : ℝ) - (↑⌊b⌋ : ℝ) - (a - b) = Int.fract b - Int.fract a := by
    simp only [Int.fract]; ring
  rw [show (↑(⌊a⌋ - ⌊b⌋) : ℝ) = (↑⌊a⌋ : ℝ) - (↑⌊b⌋ : ℝ) from Int.cast_sub ⌊a⌋ ⌊b⌋,
      h_rewrite]
  -- |fract(b) - fract(a)| < 1 since both ∈ [0, 1)
  have h1 := Int.fract_nonneg b
  have h2 := Int.fract_lt_one b
  have h3 := Int.fract_nonneg a
  have h4 := Int.fract_lt_one a
  rw [abs_lt]; constructor <;> linarith

/-- **Voronoi Coverage Theorem.**

    For any point μ on the probability simplex Δ_d and any grid resolution S ≥ 1,
    there exists an integer composition c of S into d nonneg parts such that
    each |μ_i - c_i/S| < 1/S.

    Construction: cumulative rounding c_i = ⌊S·∑_{j≤i} μ_j⌋ - ⌊S·∑_{j<i} μ_j⌋.
    The sum telescopes to S, and the error bound follows from the fact that
    fractional parts lie in [0, 1).

    NOTE: The correct radius is 1/S (not 1/(2S) as claimed in the proof document).
    This is because cumulative rounding cannot guarantee 1/(2S) in general. -/
theorem voronoi_coverage (d S : ℕ) (_hd : d > 0) (hS : S > 0)
    (μ : Fin d → ℝ) (hμ_nn : ∀ i, 0 ≤ μ i) (hμ_sum : ∑ i, μ i = 1) :
    ∃ c : Fin d → ℕ, (∑ i, c i = S) ∧ (∀ i, |μ i - (c i : ℝ) / (S : ℝ)| < 1 / (S : ℝ)) := by
  -- Construct c via cumulative rounding (ℤ → ℕ via toNat, safe since each value ≥ 0)
  refine ⟨fun i => (round_int S μ i).toNat, ?_, ?_⟩
  · -- Sum equals S
    have h_nn : ∀ i, 0 ≤ round_int S μ i := round_int_nonneg S μ hμ_nn
    have h_sum_int := round_int_sum S μ hμ_nn hμ_sum
    have h_cast : ∀ i : Fin d, ((round_int S μ i).toNat : ℤ) = round_int S μ i :=
      fun i => Int.toNat_of_nonneg (h_nn i)
    -- ∑ toNat = S: lift from ℤ equality
    have h_eq_int : (∑ i : Fin d, ((round_int S μ i).toNat : ℤ)) = (S : ℤ) := by
      simp_rw [h_cast]; exact h_sum_int
    exact_mod_cast h_eq_int
  · -- Error bound: |μ_i - c_i/S| < 1/S
    intro i
    have h_nn := round_int_nonneg S μ hμ_nn i
    have h_cast_i : ((round_int S μ i).toNat : ℝ) = (↑(round_int S μ i) : ℝ) := by
      exact_mod_cast Int.toNat_of_nonneg h_nn
    rw [h_cast_i]
    have hS_pos : (0 : ℝ) < (S : ℝ) := Nat.cast_pos.mpr hS
    have hS_ne : (S : ℝ) ≠ 0 := ne_of_gt hS_pos
    have h_err := round_int_error S μ hμ_nn i
    -- Transform: |μ - c/S| = |c - S·μ| / S
    have h_abs_eq : |μ i - ↑(round_int S μ i) / ↑S| =
        |↑(round_int S μ i) - ↑S * μ i| / ↑S := by
      rw [show μ i - ↑(round_int S μ i) / ↑S =
          (↑S * μ i - ↑(round_int S μ i)) / ↑S from by field_simp]
      rw [abs_div, abs_of_pos hS_pos, abs_sub_comm]
    rw [h_abs_eq]
    -- |c - S·μ| / S < 1 / S follows from |c - S·μ| < 1 via a/S < b/S ↔ a < b
    simp only [div_eq_mul_inv]
    exact mul_lt_mul_of_pos_right h_err (inv_pos.mpr hS_pos)

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART 4: AXIOMS
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **AXIOM A1: Refinement Monotonicity** (UNPROVEN MATHEMATICAL CONJECTURE).

    For any mass vector μ ∈ Δ_d and any refinement ν ∈ Δ_{2d}
    (with ν_{2i} + ν_{2i+1} = μ_i), if some window at dimension d has
    TV ≥ c_target, then some window at dimension 2d also has TV ≥ c_target.

    STATUS: **NOT PROVEN**. This is an open conjecture.
    Empirically verified: 183,377 tests across exhaustive (d ≤ 4),
    random (d ≤ 16), and adversarial cases — zero violations found.

    ROLE: Justifies the cascade's subtree pruning. Without this,
    the cascade cannot skip descendants of pruned parents.

    The entire coarse cascade proof is CONDITIONAL on this conjecture.
    See proof/coarse_cascade_method.md §4 for details. -/
axiom refinement_monotonicity {d : ℕ}
    (c_target : ℝ)
    (μ : Fin d → ℝ) (ν : Fin (2 * d) → ℝ)
    (hμ : on_simplex μ) (hν : on_simplex ν)
    (h_ref : is_mass_refinement μ ν)
    (ℓ s : ℕ) (hℓ : 2 ≤ ℓ) (hs : s + ℓ ≤ 2 * d)
    (h_tv : mass_test_value d μ ℓ s ≥ c_target) :
    ∃ (ℓ' s' : ℕ), 2 ≤ ℓ' ∧ s' + ℓ' ≤ 2 * (2 * d) ∧
      mass_test_value (2 * d) ν ℓ' s' ≥ c_target

/-- **AXIOM A2: Simplex TV Coverage** (COMPUTATIONAL RESULT).

    At the terminal cascade dimension d = 2·n_term, every point μ on the
    probability simplex Δ_d has some window (ℓ, s) with TV_W(μ) ≥ c_target.

    This is verified by the combination of:
    1. Cascade pruning: all integer compositions of S into d parts are pruned
       (using refinement_monotonicity to skip descendants of pruned parents).
    2. Box certification: for each grid cell, the minimum TV over the
       Voronoi box (radius < 1/S from voronoi_coverage) exceeds c_target.
    3. Voronoi coverage: every μ ∈ Δ_d lies in some grid cell.

    Analogous to cascade_all_pruned in FinalResult.lean but for the
    no-correction (mass-based) approach.

    To verify: run the coarse cascade prover:
      python coarse_cascade_prover.py --c_target <c> --S <S> -/
axiom simplex_tv_coverage (n_term : ℕ) (c_target : ℝ) :
    ∀ μ : Fin (2 * n_term) → ℝ,
      on_simplex μ →
      ∃ (ℓ s : ℕ), 2 ≤ ℓ ∧ mass_test_value (2 * n_term) μ ℓ s ≥ c_target

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART 5: MAIN THEOREM (FULLY PROVEN from Theorem 1 + Axiom A2)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Main Theorem** (Coarse Cascade Lower Bound on C₁ₐ).

    Assuming simplex_tv_coverage at some dimension d = 2·n_term:
    every nonneg function f supported on (-1/4, 1/4) with ∫f = 1 and ‖f*f‖_∞ < ∞
    satisfies autoconvolution_ratio(f) ≥ c_target.

    Proof:
    1. Compute bin masses μ = bin_masses(f, n_term). These lie on Δ_d.
    2. By simplex_tv_coverage: ∃ window (ℓ, s) with TV_W(μ, ℓ, s) ≥ c_target.
    3. By Theorem 1 (mass_test_value_le_ratio): R(f) ≥ TV_W(μ, ℓ, s).
    4. Combine: R(f) ≥ c_target.

    NOTE: This result is conditional on refinement_monotonicity (Axiom A1),
    which is used inside the verification of simplex_tv_coverage (Axiom A2). -/
theorem coarse_cascade_bound (n_term : ℕ) (hn : n_term > 0)
    (c_target : ℝ) (_hct : 0 < c_target)
    (h_coverage : ∀ μ : Fin (2 * n_term) → ℝ,
      on_simplex μ →
      ∃ (ℓ s : ℕ), 2 ≤ ℓ ∧ mass_test_value (2 * n_term) μ ℓ s ≥ c_target)
    (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ c_target := by
  -- Step 1: bin masses lie on the simplex
  set μ := bin_masses f n_term with hμ_def
  have hμ_simplex : on_simplex μ :=
    ⟨fun i => bin_masses_nonneg f hf_nonneg n_term i,
     sum_bin_masses_eq_one n_term hn f hf_supp hf_int⟩
  -- Step 2: by simplex_tv_coverage, there exists a killing window
  obtain ⟨ℓ, s, hℓ, h_tv⟩ := h_coverage μ hμ_simplex
  -- Step 3+4: Theorem 1 gives R(f) ≥ TV_W(μ) ≥ c_target
  calc autoconvolution_ratio f
      ≥ mass_test_value (2 * n_term) μ ℓ s :=
        mass_test_value_le_ratio n_term hn f hf_nonneg hf_supp hf_int h_conv_fin ℓ s hℓ
    _ ≥ c_target := h_tv

/-- Instantiation: C₁ₐ ≥ c_target for all admissible f (with ∫f > 0).

    Uses scale invariance R(a·f) = R(f) to normalize to ∫f = 1.
    This is the user-facing theorem. -/
theorem coarse_cascade_bound_general (n_term : ℕ) (hn : n_term > 0)
    (c_target : ℝ) (hct : 0 < c_target)
    (h_coverage : ∀ μ : Fin (2 * n_term) → ℝ,
      on_simplex μ →
      ∃ (ℓ s : ℕ), 2 ≤ ℓ ∧ mass_test_value (2 * n_term) μ ℓ s ≥ c_target)
    (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ c_target := by
  -- Normalize: g = (1/∫f) · f has ∫g = 1
  set I := MeasureTheory.integral MeasureTheory.volume f with hI_def
  set g := fun x => (1/I) * f x with hg_def
  have hI_pos : 0 < I := hf_int_pos
  -- R(f) = R(g) by scale invariance
  have h_ratio_eq : autoconvolution_ratio f = autoconvolution_ratio g :=
    (autoconvolution_ratio_scale_invariant f (1/I) (by positivity)).symm
  rw [h_ratio_eq]
  -- g satisfies all hypotheses
  have hg_nonneg : ∀ x, 0 ≤ g x :=
    fun x => mul_nonneg (by positivity) (hf_nonneg x)
  have hg_supp : Function.support g ⊆ Set.Ioo (-1/4 : ℝ) (1/4) := by
    intro x hx; apply hf_supp; rw [Function.mem_support] at hx ⊢
    intro h; exact hx (by simp only [hg_def, h, mul_zero])
  have hg_int : MeasureTheory.integral MeasureTheory.volume g = 1 := by
    simp only [hg_def, MeasureTheory.integral_const_mul, ← hI_def]
    exact div_mul_cancel₀ 1 (ne_of_gt hI_pos)
  have hg_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution g g
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ :=
    eLpNorm_convolution_scale_ne_top f (1/I) h_conv_fin
  exact coarse_cascade_bound n_term hn c_target hct h_coverage g
    hg_nonneg hg_supp hg_int hg_conv_fin

end -- noncomputable section
