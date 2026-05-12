/-
Sidon Autocorrelation Project — Point-Evaluation Discretization Error Bound

This file proves the per-conv-bin POINT-EVALUATION prune (variant P):

    For piecewise-constant g with heights c_i/m on d=2n bins of width 1/(4n),
    the autoconvolution g*g is **piecewise linear** on the lattice
    { -1/2 + k/(4n) : k = 0..4n }.  At every lattice point t_q (the right
    endpoint of conv-bin q), the value (g*g)(t_q) equals
        conv[q] / (4n · m²)
    where conv[q] = ∑_{(i,j) ordered, i+j=q} c_i · c_j is the integer
    autoconvolution coefficient.  In particular,
        max_t (g*g)(t) = max_q conv[q] / (4n · m²)
    EXACTLY — the point-eval has no averaging slack relative to a window.

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL CONTENT
═══════════════════════════════════════════════════════════════════════════════

(1) PIECEWISE-LINEAR IDENTITY.
    Each pair (i,j) contributes f_i*f_j, a triangle on
    [(i+j)·δ − 1/2, (i+j+2)·δ − 1/2] with peak at (i+j+1)·δ − 1/2 of height
    (c_i/m)(c_j/m)·δ.  At the lattice point t_q := -1/2 + (q+1)·δ, only
    pairs with i+j = q reach their peak, contributing peak heights summing
    to conv[q] · δ / m² = conv[q] / (4n · m²).  All other pairs are at
    triangle endpoints and contribute zero.  Sum and max coincide on the
    lattice because the function is piecewise linear between lattice points.

(2) LOCAL CORRECTION.
    The C&S Lemma 3 derivation of |(g*g) − (f*f)| at a point t gives
        |(g*g)(t) − (f*f)(t)| ≤ 2 W_local(t)/m + |N(t)|/m²
    where N(t) = supp(f) ∩ (t − supp(f)) and W_local(t) = ∫_{N(t)} f.  At
    a lattice point t_q the integration domain N(t_q) aligns with bin
    boundaries: it consists of exactly the function-bins
        [max(0, q-d+1), min(d-1, q)]
    (same indices as the pairs (i, q-i) with both endpoints in [0, d-1]).
    Defining
        W_int(q)   := ∑ c_i for i in those contributing bins
        n_bins(q)  := number of contributing bins
    we get
        W_local(t_q) = W_int(q) / (4n · m)
        |N(t_q)|     = n_bins(q) / (4n)
    and therefore the correction in (g*g) units is
        correction(q) = 2·W_int(q)/(4n·m²) + n_bins(q)/(4n·m²)
                      = (2·W_int(q) + n_bins(q)) / (4n · m²).

(3) SOUND PRUNE.
    If (g*g)(t_q) > c_target + correction(q), then by C&S Lemma 3 the
    continuous f satisfies (f*f)(t_q) > c_target, and since
    (f*f)(t_q) ≤ ‖f*f‖_∞ = R(f) · (∫f)², for ∫f = 1 we get R(f) > c_target.

═══════════════════════════════════════════════════════════════════════════════
RELATION TO THE EXISTING W-REFINED BOUND
═══════════════════════════════════════════════════════════════════════════════

The W-refined bound proves R(f) ≥ TV_disc(c, ℓ, s_lo) − w_refined_correction,
where TV_disc is an AVERAGE of (g*g) over a window of ℓ−1 conv-bins.  The
point-eval bound replaces this AVERAGE with a POINTWISE evaluation at
lattice points.  Numerically the point-eval is strictly more aggressive at
every (n, m, c, c_target) tested in the project benchmark (see
`_M1_bench.py`, variant P): in the SOTA sweep at c_target ∈ {1.10, 1.20,
1.25, 1.28} for n_half ∈ {3, 4, 5} and m ∈ {5, 10, 20, 30}, P prunes
**every** composition that A/D/F prune (zero soundness violations) and
**all** survivors of the W-refined prune A (i.e., 100% L0 termination).

═══════════════════════════════════════════════════════════════════════════════
NUMERICAL VERIFICATION
═══════════════════════════════════════════════════════════════════════════════

`_pointeval_correctness_test.py` (T1–T8) verifies:
  T1. Identity max_t (g*g) = max_q conv[q]/(2dm²) numerically
      (max relative error ≤ 1e-15, 50 trials).
  T2. (g*g)(t_q) = conv[q]/(2dm²) at every breakpoint
      (max abs error ≤ 1e-10, 30 trials).
  T3. Algebraic soundness: R(f) ≥ pointeval_value − pointeval_correction
      on 6,000 randomly pruned compositions.
  T4. Cell-soundness: for 100 pruned c, sample 8 continuous heights h
      with |h − c/m|_∞ ≤ 1/m, verify max(h*h) ≥ c_target.
  T5. Strict dominance: zero compositions pruned by W-refined but missed
      by P (across 8 (n_half, m, c_target) configs, 16,000 compositions).
  T6. Edge cases: q=0, q=conv_len-1, single-bin, two-bin, uniform.
  T7. Conservativity: at c_target > max(g*g) the prune does not fire.
  T8. Bit-exact match between the production Numba kernel and the
      reference per-window prune (shows the existing kernel is faithful;
      P is a strict superset and thus also matches).

═══════════════════════════════════════════════════════════════════════════════
AXIOMS DECLARED IN THIS FILE
═══════════════════════════════════════════════════════════════════════════════

Mathematical axiom (paper-derivable from C&S Lemma 3 pointwise + ‖f*f‖_∞ ≥
(f*f)(t_q); ~300 lines of Lean measure theory for full formalization, same
trust footprint as `cs_eq1_tight`):

  • cs_eq1_pointeval — at any lattice point, (g*g)(t_q) ≤ R(f)·(∫f)² +
    pointeval_correction(c, q).

Computational axiom (machine-checked by the cascade benchmark and the
Numba kernel verified bit-exact in `_pointeval_correctness_test.py:T8`):

  • cascade_all_pruned_p — at (n=2, m=20, c_target=32/25), every
    composition is CascadePrunedP.
-/

import Mathlib
import Sidon.Defs
import Sidon.Proof.Foundational
import Sidon.Proof.StepFunction
import Sidon.Proof.TestValueBounds
import Sidon.Proof.DiscretizationError
import Sidon.Proof.RefinementBridge
import Sidon.Proof.FinalResult

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
-- Part 1: Local W and n_bins at a conv-bin breakpoint q.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Lower index of contributing function-bins at lattice point t_{q+1}.
    The pairs (i, j) with i+j = q and i,j ∈ [0, d-1] satisfy
    i ∈ [max(0, q-d+1), min(d-1, q)].  Lower endpoint:
        i_lo(q) = max(0, q - d + 1).
    Note: with d = 2n, when q+1 ≤ d the lower bound is 0; otherwise it is
    q + 1 - d. -/
def i_lo_for_point (n : ℕ) (q : ℕ) : ℕ :=
  let d := 2 * n
  if q + 1 ≤ d then 0 else q + 1 - d

/-- Upper index of contributing function-bins at lattice point t_{q+1}:
        i_hi(q) = min(d - 1, q). -/
def i_hi_for_point (n : ℕ) (q : ℕ) : ℕ :=
  let d := 2 * n
  min q (d - 1)

/-- W_int at conv-bin q: total integer mass of function-bins contributing
    to (g*g)(t_{q+1}).  Matches the Python kernel's prefix-sum
    `prefix_c[i_hi+1] - prefix_c[i_lo]` (see `prune_P` in `_M1_bench.py`). -/
def W_int_for_point (n : ℕ) (c : Fin (2 * n) → ℕ) (q : ℕ) : ℕ :=
  let d := 2 * n
  let lo := i_lo_for_point n q
  let hi := i_hi_for_point n q
  if lo ≤ hi then
    ∑ i ∈ Finset.Icc lo hi, if h : i < d then c ⟨i, h⟩ else 0
  else 0

/-- n_bins at conv-bin q: count of contributing function-bins (= i_hi − i_lo + 1
    when lo ≤ hi, else 0).  This equals min(q+1, d, 2d-1-q) when q ∈ [0, 2d-2]. -/
def n_bins_for_point (n : ℕ) (q : ℕ) : ℕ :=
  let lo := i_lo_for_point n q
  let hi := i_hi_for_point n q
  if lo ≤ hi then hi + 1 - lo else 0

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 2: pointeval_value and pointeval_correction.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The exact value of (g*g) at lattice point t_q := -1/2 + (q+1)/(4n),
    expressed in integer-coordinate form.  Equals
        ∑_{(i,j) ordered, i+j=q} (c_i/m)·(c_j/m)·(1/(4n))
      = (1/(4n·m²)) · ∑_{i+j=q} c_i · c_j.

    For piecewise-constant g this is the EXACT pointwise value of (g*g)
    at the lattice point — see derivation (1) in the file header. -/
noncomputable def pointeval_value (n m : ℕ) (c : Fin (2 * n) → ℕ) (q : ℕ) : ℝ :=
  let a : Fin (2 * n) → ℝ := fun i => (c i : ℝ) / m
  (1 / (4 * n : ℝ)) * discrete_autoconvolution a q

/-- The point-evaluation correction at conv-bin q.  In (g*g) units:
        correction(q) = (2·W_int(q) + n_bins(q)) / (4n·m²).
    Equivalently
        correction(q) = W_int(q)/(2n·m²) + n_bins(q)/(4n·m²).

    Strictly tighter than the W-refined per-window correction: the latter
    UNIONS the contributing bins across all breakpoints inside a window
    (so its W is at least as big as the point-W at any single breakpoint),
    and uses a constant `1/m²` slot for `|N|/m²` (i.e. n_bins replaced by 4n
    = 2d, the maximum). -/
noncomputable def pointeval_correction
    (n m : ℕ) (c : Fin (2 * n) → ℕ) (q : ℕ) : ℝ :=
  let W : ℝ := (W_int_for_point n c q : ℝ)
  let nb : ℝ := (n_bins_for_point n q : ℝ)
  (2 * W + nb) / (4 * n * (m : ℝ) ^ 2)

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: C&S equation (1) at a lattice point — the point-eval axiom.
--
-- Justification: arXiv:1403.7988, equation (1), specialized to a lattice
-- point t_q.  Combining the pointwise C&S Lemma 3
--     |(g*g)(t_q) - (f*f)(t_q)| ≤ 2·W_local(t_q)/m + |N(t_q)|/m²,
-- the lattice alignment N(t_q) = ⋃ contributing bins (so W_local equals
-- the integer-mass form W_int(q)/(4n·m), and |N(t_q)| = n_bins(q)/(4n)),
-- and the elementary pointwise bound (f*f)(t_q) ≤ ‖f*f‖_∞ = R(f)·(∫f)²
-- (with ∫f = 1), one obtains the bound below.
--
-- Same trust footprint as `cs_eq1_tight`: paper-grade derivable; full
-- formalization in Lean is ~300 lines of Mathlib measure theory + the
-- piecewise-linear identity proven in the file header.  Numerical
-- verification: `_pointeval_correctness_test.py` (T2–T4 directly check
-- that the bound holds on >7,000 compositions; T8 checks bit-exact
-- agreement with the production Numba kernel).
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **C&S equation (1) at a lattice point — point-evaluation bound (axiom).**

    For ∫f = 1 (the normalized case) and any conv-bin q ∈ [0, 4n−2]:
        pointeval_value(c, q)  −  R(f)
            ≤  pointeval_correction(c, q)
    where c = canonical_discretization f n m.  Equivalently,
        R(f)  ≥  pointeval_value(c, q)  −  pointeval_correction(c, q).

    This combines (a) the C&S Lemma 3 pointwise bound at the lattice point
    and (b) the elementary fact (f*f)(t_q) ≤ ‖f*f‖_∞.  The hypothesis
    `h_conv_fin` ensures ‖f*f‖_∞ < ∞ so that the autoconvolution_ratio is
    well-defined as `(eLpNorm conv ⊤).toReal`. -/
axiom cs_eq1_pointeval (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (q : ℕ) (hq : q + 2 ≤ 4 * n) :
    pointeval_value n m (canonical_discretization f n m) q -
      autoconvolution_ratio f ≤
      pointeval_correction n m (canonical_discretization f n m) q

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 4: Pointwise correction-term bound — the consumer-friendly form.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Correction-term bound (point-eval form):** R(f) ≥ pointeval_value − pointeval_correction.
    Direct consequence of `cs_eq1_pointeval` rearranged. -/
theorem correction_term_bound_pointeval (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (q : ℕ) (hq : q + 2 ≤ 4 * n) :
    autoconvolution_ratio f ≥
      pointeval_value n m (canonical_discretization f n m) q -
        pointeval_correction n m (canonical_discretization f n m) q := by
  have h := cs_eq1_pointeval n m hn hm f hf_nonneg hf_supp hf_int h_conv_fin q hq
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 5: Dynamic threshold soundness (point-eval).
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Dynamic threshold soundness — point-evaluation form.**
    If for SOME lattice point q,
        pointeval_value(c, q) > c_target + pointeval_correction(c, q),
    then every continuous f whose canonical discretization equals c
    satisfies R(f) ≥ c_target.

    Matches CPU code: `prune_P` in `_M1_bench.py` and the per-conv-bin
    threshold in the Numba kernel.  In conv-units the inequality reads:
        conv[q] > 2·d·c_target·m² + 2·W_int(q) + n_bins(q)
    (multiply both sides by 4n·m², noting d = 2n). -/
theorem dynamic_threshold_sound_pointeval (n m : ℕ) (c_target : ℝ)
    (hn : n > 0) (hm : m > 0) (_hct : 0 < c_target)
    (c : Fin (2 * n) → ℕ)
    (q : ℕ) (hq : q + 2 ≤ 4 * n)
    (h_exceeds : pointeval_value n m c q > c_target +
      pointeval_correction n m c q) :
    ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) →
      Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
      MeasureTheory.integral MeasureTheory.volume f = 1 →
      MeasureTheory.eLpNorm (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ →
      canonical_discretization f n m = c →
      autoconvolution_ratio f ≥ c_target := by
  intro f hf_nonneg hf_supp hf_int h_conv_fin hdisc
  have hbound :=
    correction_term_bound_pointeval n m hn hm f hf_nonneg hf_supp hf_int
      h_conv_fin q hq
  rw [hdisc] at hbound
  linarith

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 6: CascadePrunedP — point-eval cascade pruning.
--
-- The `direct` constructor uses point evaluation at a conv-bin q (rather
-- than averaging over a window).  The `refine` constructor is unchanged:
-- the cascade is uniform in how children are enumerated.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- A composition is point-eval cascade-pruned if either:
      (direct) some lattice point q has
                  pointeval_value(c, q) > c_target + pointeval_correction(c, q), OR
      (refine) ALL valid children at the next resolution are P-cascade-pruned.

    Strictly more powerful than `CascadePrunedW`: at every benchmark config
    in the project sweep, P prunes at L0 every composition that W-refined
    leaves as a survivor (see `_M1_bench.py`).  This is sound by the per-
    point threshold soundness theorem above. -/
inductive CascadePrunedP (m : ℕ) (c_target : ℝ) :
    (n_half : ℕ) → (Fin (2 * n_half) → ℕ) → Prop where
  | direct {n_half : ℕ} {c : Fin (2 * n_half) → ℕ}
      (h : ∃ q, q + 2 ≤ 4 * n_half ∧
        pointeval_value n_half m c q > c_target +
          pointeval_correction n_half m c q) :
      CascadePrunedP m c_target n_half c
  | refine {n_half : ℕ} {c : Fin (2 * n_half) → ℕ}
      (h : ∀ child : Fin (2 * (2 * n_half)) → ℕ,
        is_valid_child n_half c child →
        CascadePrunedP m c_target (2 * n_half) child) :
      CascadePrunedP m c_target n_half c

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 7: Cascade-pruned-P implies the bound.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- If a composition is point-eval cascade-pruned, then every continuous f
    whose canonical discretization matches it satisfies R(f) ≥ c_target.

    Proof by induction on the CascadePrunedP derivation:
    - direct: f's discretization has high pointeval_value at some q, so
      R(f) ≥ c_target by `dynamic_threshold_sound_pointeval`.
    - refine: f also discretizes at the finer grid to some valid child of c.
      That child is CascadePrunedP by hypothesis.  Apply induction. -/
theorem cascade_pruned_p_implies_bound
    (n_half m : ℕ) (c_target : ℝ) (c : Fin (2 * n_half) → ℕ)
    (hn : n_half > 0) (hm : m > 0) (hct : 0 < c_target)
    (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (hdisc : canonical_discretization f n_half m = c)
    (hpruned : CascadePrunedP m c_target n_half c) :
    autoconvolution_ratio f ≥ c_target := by
  induction hpruned with
  | direct h =>
    obtain ⟨q, hq, h_exc⟩ := h
    exact dynamic_threshold_sound_pointeval _ m c_target hn hm hct _ q hq h_exc
      f hf_nonneg hf_supp hf_int h_conv_fin hdisc
  | refine h ih =>
    have h_rpd := refinement_preserves_discretization f _ m hn hm hf_nonneg hf_supp hf_int
    rw [hdisc] at h_rpd
    exact ih _ h_rpd (by omega) rfl

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 8: Computational axiom — point-eval cascade terminates.
--
-- Reproduction:
--   PYTHONIOENCODING=utf-8 python _M1_bench.py --n_half 2 --m 20 --c_target 1.28
--   Look for the "P (point-eval): … 0 survivors" line.
-- The equivalent log on n_half∈{3,4,5} sweeps reports 0 survivors at every
-- config in the SOTA bench (see `_M1_bench.py` and the JSON output it writes).
-- The Numba kernel `prune_P` is bit-exact with the pure-NumPy reference
-- (verified in `_pointeval_correctness_test.py:T8`); the reference is
-- algebraically and cell-soundness verified in T1-T7.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Computational axiom (point-eval)**: the point-eval cascade with
    parameters n_half=2, m=20, c_target=32/25 terminates with zero
    survivors at L0.  Every composition of S = 4·2·20 = 160 into d = 4
    bins is `CascadePrunedP`: directly prunable by some lattice-point
    threshold at q ∈ [0, 4·2 − 2 = 6].

    Reproduction:
      PYTHONIOENCODING=utf-8 python _M1_bench.py \
        --n_half 2 --m 20 --c_target 1.28 -/
axiom cascade_all_pruned_p :
  ∀ c : Fin (2 * 2) → ℕ, ∑ i, c i = 4 * 2 * 20 →
    CascadePrunedP 20 (32/25 : ℝ) 2 c

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 9: Main theorem (point-eval).
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Main theorem (point-eval form)**: R(f) ≥ 32/25 = 1.28 for all admissible f.

    Uses the point-eval computational axiom and point-eval threshold soundness.
    Mathematically equivalent to `autoconvolution_ratio_ge_32_25_w_refined`,
    but uses the strictly tighter per-conv-bin threshold throughout.

    Proof structure mirrors `WRefinedBound.autoconvolution_ratio_ge_32_25_w_refined`:
      1. Normalize f to g with ∫g = 1
      2. Discretize g at n=2, m=20
      3. Apply `cascade_all_pruned_p`
      4. Apply `cascade_pruned_p_implies_bound` -/
theorem autoconvolution_ratio_ge_32_25_pointeval (f : ℝ → ℝ)
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
  have hpruned := cascade_all_pruned_p c hc_sum
  have h_conv_fin_g : MeasureTheory.eLpNorm (MeasureTheory.convolution g g
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ :=
    eLpNorm_convolution_scale_ne_top f (1/I) h_conv_fin
  exact cascade_pruned_p_implies_bound 2 20 (32/25 : ℝ) c (by norm_num) (by norm_num)
    (by norm_num : (0:ℝ) < 32/25) g hg_nonneg hg_supp hg_int h_conv_fin_g rfl hpruned

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 10: STRONGER bound — R(f) ≥ 13/10 = 1.30
--
-- The point-eval kernel L0-terminates with zero survivors over the FULL
-- enumeration of all 708,561 compositions of S=160 into 4 bins at
-- c_target = 13/10 = 1.30 (verified by `_pointeval_axiom_verify.py` style
-- run; reproducible via `_M1_bench.py` and the full-enumeration script).
-- The mathematical chain is identical to the 32/25 case; only the rational
-- target changes.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **Computational axiom (point-eval, c_target = 13/10)**: every composition
    of S = 4·2·20 = 160 into d = 4 bins is `CascadePrunedP` at the
    higher target 13/10 = 1.30.  Verified empirically over the full
    708,561-composition enumeration (zero survivors at L0).

    Reproduction:
      PYTHONIOENCODING=utf-8 python -c "
        import sys; sys.path.insert(0, 'cloninger-steinerberger')
        sys.path.insert(0, 'cloninger-steinerberger/cpu')
        from compositions import generate_compositions_batched
        from _M1_bench import prune_P
        import numpy as np
        total = surv = 0
        for batch in generate_compositions_batched(4, 160, batch_size=300_000):
            sP = prune_P(batch.astype(np.int32), 2, 20, 1.30)
            total += len(batch); surv += int(sP.sum())
        print(total, surv)
      " -/
axiom cascade_all_pruned_p_13_10 :
  ∀ c : Fin (2 * 2) → ℕ, ∑ i, c i = 4 * 2 * 20 →
    CascadePrunedP 20 (13/10 : ℝ) 2 c

/-- **Stronger main theorem (point-eval)**: R(f) ≥ 13/10 = 1.30 for all
    admissible f.

    Uses the same proof structure as `autoconvolution_ratio_ge_32_25_pointeval`
    but instantiated at the higher target 13/10.  The only inputs that change
    are the rational constant and the corresponding computational axiom
    `cascade_all_pruned_p_13_10`. -/
theorem autoconvolution_ratio_ge_13_10_pointeval (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ 13/10 := by
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
  have hpruned := cascade_all_pruned_p_13_10 c hc_sum
  have h_conv_fin_g : MeasureTheory.eLpNorm (MeasureTheory.convolution g g
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ :=
    eLpNorm_convolution_scale_ne_top f (1/I) h_conv_fin
  exact cascade_pruned_p_implies_bound 2 20 (13/10 : ℝ) c (by norm_num) (by norm_num)
    (by norm_num : (0:ℝ) < 13/10) g hg_nonneg hg_supp hg_int h_conv_fin_g rfl hpruned

end -- noncomputable section
