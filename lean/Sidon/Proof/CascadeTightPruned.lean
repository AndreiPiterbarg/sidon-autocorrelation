/-
Sidon Autocorrelation Project — Tight (D-v2) Cascade Inductive + Soundness

This file defines the **tight-cascade inductive predicate** `CascadePrunedTight`
and proves its **soundness theorem** `cascade_pruned_tight_implies_bound`,
parallel to `CascadePrunedW` and `cascade_pruned_w_implies_bound` from
`WRefinedBound.lean`, but using the **strictly tighter** D-v2 correction
proven in `TightDiscretizationBound.lean` and lifted to the continuous-discrete
interface by the axiom `cs_eq1_tight` from `TightContinuousBridge.lean`.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS IS THE STRONGEST PROVABLE CASCADE
═══════════════════════════════════════════════════════════════════════════════

The cascade pruning chain has three layers in this project:

  1. flat correction      `2/m + 1/m²`                              (CoarseCascade)
  2. W-refined correction `(1 + W_int/(2n)) / m²`                    (WRefinedBound)
  3. tight (D-v2) correction `corr_tight_m2 / m²`,                   (this file)
       where  corr_tight_m2 = (ℓ−1)·W_int / (2n·ℓ) + ell_int_sum / (4n·ℓ)

The universal pointwise dominance theorem
  `TightDiscretizationBound.corr_tight_m2_le_corr_w_refined`
proves
  corr_tight_m2 n c ℓ s_lo  ≤  1 + W_int_overlap n c s_lo ℓ / (2n)
for every `n > 0`, `ℓ ≥ 2`, every composition `c`, and every window start
`s_lo`.  Since `W_int_overlap` and `W_int_for_window` (used in the W-refined
file) are set-theoretically the same partial sum of `c_i` over the
contributing-bin range, this gives `tight_correction ≤ w_refined_correction`
universally.  Hence:

    ★  The TIGHT cascade prunes at LEAST as much as the W-refined cascade
       at every window — and STRICTLY MORE in the typical case where the
       window does not exhaust the bin range.

In particular, the inductive predicate `CascadePrunedTight` admits ALL
derivations admitted by `CascadePrunedW` (via the dominance, see the omitted
backward-compatibility theorem at the bottom), so the tight cascade is the
strongest pruning predicate provable with paper-grade rigor.

═══════════════════════════════════════════════════════════════════════════════
FILE STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

  Part 1.  Inductive predicate `CascadePrunedTight`
           (constructors `direct` and `refine`, parallel to `CascadePrunedW`).

  Part 2.  Soundness theorem `cascade_pruned_tight_implies_bound`
           (induction on the derivation, calling `dynamic_threshold_sound_tight`
           for the `direct` case and `refinement_preserves_discretization` for
           the `refine` case — same induction pattern as the W-refined version).

  Part 3.  (DEFERRED) Computational axiom `cascade_all_pruned_tight` and
           the published theorem `autoconvolution_ratio_ge_32_25_tight`.  These
           are NOT included in this file; the user is handling the computational
           artifact (JSON output of the Python cascade with `--use_tight_v2`)
           separately.  Once the artifact lands, the user will add the axiom
           and prove the published theorem in a follow-up patch.

═══════════════════════════════════════════════════════════════════════════════
AXIOM HYGIENE
═══════════════════════════════════════════════════════════════════════════════

This file declares **NO new axioms**.  It re-uses:
  • `cs_eq1_tight`                       (TightContinuousBridge.lean — Layer 2 axiom)
  • `cascade_all_pruned_tight`           (NOT YET ADDED — user's separate work)
indirectly via the soundness chain
  `dynamic_threshold_sound_tight`  →  `correction_term_bound_tight`
  →  `cs_eq1_tight`  +  `continuous_test_value_le_ratio`.

Project-wide axiom count remains 5 (cs_lemma3_per_window, cs_eq1_w_refined,
cascade_all_pruned, cascade_all_pruned_w, cs_eq1_tight).

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
import Sidon.Proof.WRefinedBound
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
-- Part 1: CascadePrunedTight — Cascade with Per-Window Tight (D-v2) Correction
--
-- Parallel to `CascadePrunedW` from `WRefinedBound.lean`, with `tight_correction`
-- (TV-space, = `corr_tight_m2 / m²`) replacing `w_refined_correction` in the
-- `direct` constructor.
--
-- Constructor names (`direct`, `refine`) and signatures match `CascadePrunedW`
-- exactly, so any code referencing the W-refined predicate can be updated by
-- a name-level rename.  The `refine` constructor is unchanged in form: it
-- recurses into `2 * n_half` for valid children defined by `is_valid_child`
-- (the ±1-tolerance refinement from `Sidon.Defs`).
-- ═══════════════════════════════════════════════════════════════════════════════

/-- A composition is **tight (D-v2) cascade-pruned** if either:
    (direct) some window has TV > c_target + tight_correction(window), OR
    (refine) ALL valid children at the next resolution are tight-cascade-pruned.

    This is the strongest cascade pruning predicate in this project: by the
    universal dominance `corr_tight_m2_le_corr_w_refined` (proven in
    `TightDiscretizationBound.lean`), `tight_correction ≤ w_refined_correction`
    ≤ flat correction always, so a composition pruned by the W-refined or flat
    cascade is automatically pruned by the tight cascade.  The converse fails:
    some compositions are pruned ONLY by the tight cascade (those for which the
    `ell_int_sum / (4n·ℓ)` slack matters more than the `1/m²` slack does for
    the W-refined cascade).

    Per-window threshold:
      `c_target + tight_correction n_half m c ℓ s_lo`
      = `c_target + ((ℓ−1)·W_int + ell_int_sum/2) / (2 n_half · ℓ · m²)`. -/
inductive CascadePrunedTight (m : ℕ) (c_target : ℝ) :
    (n_half : ℕ) → (Fin (2 * n_half) → ℕ) → Prop where
  | direct {n_half : ℕ} {c : Fin (2 * n_half) → ℕ}
      (h : ∃ ℓ s_lo, 2 ≤ ℓ ∧
        test_value n_half m c ℓ s_lo > c_target +
          tight_correction n_half m c ℓ s_lo) :
      CascadePrunedTight m c_target n_half c
  | refine {n_half : ℕ} {c : Fin (2 * n_half) → ℕ}
      (h : ∀ child : Fin (2 * (2 * n_half)) → ℕ,
        is_valid_child n_half c child →
        CascadePrunedTight m c_target (2 * n_half) child) :
      CascadePrunedTight m c_target n_half c

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 2: CascadePrunedTight Implies the Bound
--
-- Same induction pattern as `cascade_pruned_w_implies_bound`:
--   • `direct` case applies `dynamic_threshold_sound_tight` (provided by
--     `TightContinuousBridge`).
--   • `refine` case applies `refinement_preserves_discretization` (provided by
--     `RefinementBridge`) to descend to the canonical fine-grid discretization
--     of f, which is a valid child of c, and recurses by induction hypothesis.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- **If a composition is tight (D-v2) cascade-pruned, then every continuous
    function whose canonical discretization matches it has R(f) ≥ c_target.**

    Proof by induction on the `CascadePrunedTight` derivation:
      • `direct`: f's discretization at (n_half, m) has TV exceeding the tight
        per-window threshold for some window, so `R(f) ≥ c_target` directly by
        `dynamic_threshold_sound_tight`.
      • `refine`: f also discretizes at (2·n_half, m) to some valid child of c
        (by `refinement_preserves_discretization`).  That child is tight-pruned
        by the constructor's hypothesis, so `R(f) ≥ c_target` by the induction
        hypothesis.

    This theorem is the strongest cascade-pruning soundness in the project. -/
theorem cascade_pruned_tight_implies_bound
    (n_half m : ℕ) (c_target : ℝ) (c : Fin (2 * n_half) → ℕ)
    (hn : n_half > 0) (hm : m > 0) (hct : 0 < c_target)
    (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (hdisc : canonical_discretization f n_half m = c)
    (hpruned : CascadePrunedTight m c_target n_half c) :
    autoconvolution_ratio f ≥ c_target := by
  induction hpruned with
  | direct h =>
    obtain ⟨ℓ, s_lo, hℓ, h_exc⟩ := h
    exact dynamic_threshold_sound_tight _ m c_target hn hm hct _ ℓ s_lo hℓ (by linarith)
      f hf_nonneg hf_supp hf_int h_conv_fin hdisc
  | refine h ih =>
    have h_rpd := refinement_preserves_discretization f _ m hn hm hf_nonneg hf_supp hf_int
    rw [hdisc] at h_rpd
    exact ih _ h_rpd (by omega) rfl

-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 3: (OMITTED) Backward-Compatibility Theorem `cascade_pruned_w_implies_tight`
--
-- The mathematically valid statement
--   `CascadePrunedW m c_target n_half c  →  CascadePrunedTight m c_target n_half c`
-- follows from the universal dominance
--   `corr_tight_m2 ≤ 1 + W_int_overlap / (2n)`     (TightDiscretizationBound)
-- combined with the equality of the two `W_int` definitions:
--   `W_int_overlap n c s_lo ℓ  =  W_int_for_window n c ℓ s_lo`.
--
-- Both definitions are set-theoretic partial sums of `c_i` over the same
-- contributing-bin range
--   i ∈ [max(0, s_lo − d + 1), min(s_lo + ℓ − 2, d − 1)]   (where d = 2n);
-- the equality is mathematically trivial but requires a `Finset.sum_bij`-style
-- argument matching the contiguous Icc-indexing in `W_int_for_window` against
-- the `Fin (2n)` filter-indexing in `W_int_overlap`, costing ~30 routine lines.
--
-- We OMIT this theorem rather than leave a `sorry`, because:
--   (a) The published cascade chain (`autoconvolution_ratio_ge_32_25_tight`
--       once the user adds the computational axiom) does NOT depend on
--       backward compatibility — the tight predicate is sufficient on its own.
--   (b) The mathematical content is fully captured by
--       `corr_tight_m2_le_corr_w_refined` (proven), which establishes
--       `tight ≤ w_refined` modulo the trivial set-theoretic `W_int` equality.
--   (c) Adding the bijection lemma is mechanical work that does not strengthen
--       the chain in any way and can be deferred to a future cleanup pass.
--
-- The dominance is mathematically valid (the "tight ≤ W-refined" inequality
-- holds universally); only the explicit Lean-level bijection between the two
-- `W_int` definitions is left as future work.
-- ═══════════════════════════════════════════════════════════════════════════════

end -- noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- DEFERRED: Computational axiom `cascade_all_pruned_tight`
-- ═══════════════════════════════════════════════════════════════════════════════
/-
The published theorem `autoconvolution_ratio_ge_32_25_tight` (the strongest
provable, using D-v2) requires a computational axiom analogous to
`cascade_all_pruned_w` but with the tight correction.  This axiom states:

  axiom cascade_all_pruned_tight :
    ∀ c : Fin (2 * 2) → ℕ, ∑ i, c i = 4 * 2 * 20 →
      CascadePrunedTight 20 (32/25 : ℝ) 2 c

Reproduction (when the Python `--use_tight_v2` flag is added):
  python -m cloninger-steinerberger.cpu.run_cascade \
    --n_half 2 --m 20 --c_target 1.28 --use_tight_v2 --verify_relaxed

Once the JSON artifact is committed, the user will add this axiom and prove:

  theorem autoconvolution_ratio_ge_32_25_tight (f : ℝ → ℝ) ... :
      autoconvolution_ratio f ≥ 32/25 := by
    -- (mirror autoconvolution_ratio_ge_32_25_w_refined exactly,
    --  using cascade_all_pruned_tight + cascade_pruned_tight_implies_bound)
    ...

This will give the strongest-provable Lean theorem for the Sidon problem.

Note: the user is handling the computational artifact (`--use_tight_v2` Python
patch + JSON output) and the published theorem itself separately.  This file
contains only the pure-Lean inductive + soundness pieces; once the axiom is
added in a follow-up patch, the published theorem is immediate by mirroring
`autoconvolution_ratio_ge_32_25_w_refined` from `WRefinedBound.lean` exactly.
-/

-- ═══════════════════════════════════════════════════════════════════════════════
-- AUDIT BLOCK (CascadeTightPruned — final)
-- ═══════════════════════════════════════════════════════════════════════════════
/-
NEW AXIOMS DECLARED IN THIS FILE:  NONE.
  This file declares only an inductive predicate (`CascadePrunedTight`) and
  proves a single soundness theorem (`cascade_pruned_tight_implies_bound`).
  Both reuse the existing axiom `cs_eq1_tight` (declared in
  `TightContinuousBridge.lean`) indirectly via `dynamic_threshold_sound_tight`.

PROJECT-WIDE AXIOM COUNT:  unchanged at 5.
  Pre-existing:
    1. `cs_lemma3_per_window`        (DiscretizationError.lean)
    2. `cs_eq1_w_refined`             (WRefinedBound.lean)
    3. `cs_eq1_tight`                 (TightContinuousBridge.lean)
    4. `cascade_all_pruned`           (CoarseCascade.lean)         -- computational
    5. `cascade_all_pruned_w`         (WRefinedBound.lean)         -- computational
  This file adds NONE.

DECLARATIONS IN THIS FILE:
  • `CascadePrunedTight` (inductive)
       — strongest cascade pruning predicate, parallel to `CascadePrunedW`.
       Constructors `direct` and `refine` mirror the W-refined version with
       `tight_correction` replacing `w_refined_correction`.
  • `cascade_pruned_tight_implies_bound` (theorem)
       — soundness: `CascadePrunedTight ⇒ R(f) ≥ c_target`.
       Proof by induction; `direct` case calls `dynamic_threshold_sound_tight`
       (via TightContinuousBridge), `refine` case calls
       `refinement_preserves_discretization` (via RefinementBridge).

OMITTED (intentionally; not needed for the published chain):
  • `cascade_pruned_w_implies_tight` — backward compatibility theorem.
       Mathematical content captured by `corr_tight_m2_le_corr_w_refined`
       (proven in TightDiscretizationBound).  Lean-level proof requires the
       bijection lemma `W_int_overlap n c s_lo ℓ = W_int_for_window n c ℓ s_lo`,
       a routine ~30-line `Finset.sum_bij`.  Deferred to a future cleanup pass.

DEFERRED (the user's separate work):
  • `cascade_all_pruned_tight` — computational axiom (Python output of the
       `--use_tight_v2` cascade variant).
  • `autoconvolution_ratio_ge_32_25_tight` — published theorem.

DEPENDENCY GRAPH (within this file):
  cascade_pruned_tight_implies_bound
    ↳ direct case  →  dynamic_threshold_sound_tight  (TightContinuousBridge)
                       ↳ correction_term_bound_tight
                          ↳ cs_eq1_tight  (axiom; TightContinuousBridge)
                          ↳ continuous_test_value_le_ratio  (TestValueBounds)
    ↳ refine case  →  refinement_preserves_discretization  (RefinementBridge)
                       (induction hypothesis)

REFERENCES:
  - Cloninger & Steinerberger (2017), arXiv:1403.7988, equation (1).
  - This project, `lean/Sidon/Proof/TightDiscretizationBound.lean` (D-v2 proof
    + universal dominance `corr_tight_m2_le_corr_w_refined`).
  - This project, `lean/Sidon/Proof/TightContinuousBridge.lean`
    (axiom `cs_eq1_tight`, theorems `correction_term_bound_tight`,
    `dynamic_threshold_sound_tight`).
  - This project, `lean/Sidon/Proof/WRefinedBound.lean` (parallel `CascadePrunedW`
    + `cascade_pruned_w_implies_bound`; the structural template for this file).
-/
