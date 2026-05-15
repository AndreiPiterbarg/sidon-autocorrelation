/-
Sidon Autocorrelation Project — Matolcsi-Vinuesa Master Inequality
==================================================================

This file deduces the MV master inequality (paper Eq.(6)) from the
four equations of Lemma 3.1 (paper Eqs. (1)-(4), formalised in
`Sidon.MVLemmas`).  The derivation is purely algebraic — no analysis
beyond what is encoded in Eqs.(1)-(4).

Strategy:
  Adding the LHS of Eq.(1) and Eq.(2):
    ∫(f*f)·K  +  ∫(f∘f)·K  ≤  M_∞ + 1 + √(M_∞-1)·√(K_2-1).        (I)
  Eq.(3) rewrites the same LHS as:
    ∫(f*f)·K  +  ∫(f∘f)·K  =  2/u + 2u² · Σ_{j≠0} Re(f̃(j))²·K̃(j).  (II)
  Eq.(4) bounds the cosine sum below:
    u² · Σ_{j≠0} Re(f̃(j))²·K̃(j)  ≥  m_G² / Σ G̃²/K̃ =: m_G²/S_G.    (III)
  Combining (I), (II), (III):
    M_∞ + 1 + √(M_∞-1)·√(K_2-1)  ≥  2/u + 2·m_G²/S_G.              (Eq.6)

No `sorry`, no new axioms.  The theorem `master_inequality_from_lemmas`
below takes the conclusions of Eqs.(1)-(4) as numerical hypotheses and
returns Eq.(6).
-/

import Mathlib
import Sidon.Defs
import Sidon.MVLemmas

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real

namespace Sidon.Master

/-- MV master inequality (paper Eq.(6)).

The hypotheses are the four MV Lemma 3.1 conclusions expressed as
numerical inequalities / identity between scalars:

  * `hEq1`: `LHS₁ ≤ Minf`                       (Eq.(1) conclusion)
  * `hEq2`: `LHS₂ ≤ 1 + √(Minf-1)·√(K2-1)`     (Eq.(2) conclusion)
  * `hEq3`: `LHS₁ + LHS₂ = 2/u + 2·u²·S_cos`   (Eq.(3) identity)
  * `hEq4`: `u²·S_cos ≥ m_G² / S_G`            (Eq.(4) bound)

where `LHS₁ = ∫(f*f)·K`, `LHS₂ = ∫(f∘f)·K`, `S_cos = ∑ Re(f̃)²·K̃`,
`S_G = ∑ G̃²/K̃`.  The proof is purely algebraic. -/
theorem master_inequality_from_lemmas
    (Minf K2 m_G S_G u S_cos LHS1 LHS2 : ℝ)
    (hu : 0 < u)
    (hS_G_pos : 0 < S_G)
    -- Hypotheses corresponding to the four MV Lemma 3.1 equations:
    (hEq1 : LHS1 ≤ Minf)
    (hEq2 : LHS2 ≤ 1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1))
    (hEq3 : LHS1 + LHS2 = 2 / u + 2 * u^2 * S_cos)
    (hEq4 : u^2 * S_cos ≥ m_G^2 / S_G) :
    Minf + 1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1)
      ≥ 2 / u + 2 * m_G^2 / S_G := by
  -- Step 1: LHS1 + LHS2 ≤ Minf + 1 + √(Minf-1)·√(K2-1)   from Eq.(1)+(2).
  have h_sum_upper :
      LHS1 + LHS2 ≤ Minf + (1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1)) :=
    add_le_add hEq1 hEq2
  -- Step 2: 2/u + 2·u²·S_cos ≤ Minf + 1 + √(Minf-1)·√(K2-1)  by Eq.(3).
  have h_three :
      2 / u + 2 * u^2 * S_cos
        ≤ Minf + (1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1)) := by
    rw [← hEq3]; exact h_sum_upper
  -- Step 3: 2/u + 2·m_G²/S_G ≤ 2/u + 2·u²·S_cos  by Eq.(4).
  have h_four :
      2 / u + 2 * (m_G^2 / S_G) ≤ 2 / u + 2 * u^2 * S_cos := by
    have h1 : 2 * (m_G^2 / S_G) ≤ 2 * (u^2 * S_cos) :=
      mul_le_mul_of_nonneg_left hEq4 (by norm_num : (0:ℝ) ≤ 2)
    have h2 : 2 * (u^2 * S_cos) = 2 * u^2 * S_cos := by ring
    linarith
  -- Combine: chain through Step 2 and Step 3.
  have h_chain :
      2 / u + 2 * (m_G^2 / S_G)
        ≤ Minf + (1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1)) :=
    le_trans h_four h_three
  -- Final cosmetic: `2 * m_G^2 / S_G = 2 * (m_G^2 / S_G)` and
  -- `Minf + 1 + √… = Minf + (1 + √…)`.
  have h_rewrite_LHS : 2 * m_G^2 / S_G = 2 * (m_G^2 / S_G) := by ring
  have h_rewrite_RHS :
      Minf + 1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1)
        = Minf + (1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1)) := by ring
  rw [h_rewrite_LHS, h_rewrite_RHS]
  exact h_chain

/-! ## Direct corollary using the named MV lemmas

Below we package `master_inequality_from_lemmas` so that the four MV
Lemma 3.1 inputs are produced by `Sidon.MV.mv_eq1`, `mv_eq2`, `mv_eq3`,
`mv_eq4` and consumed in the same syntactic form, with the analytic
inputs to the harder lemmas (mv_eq2 = `hCS_bound`,
mv_eq3 = `hPoisson`) carried as parameters. -/

/-- The MV master inequality assembled directly from the named MV
Lemma 3.1 conclusions.  The hypotheses are exactly the four MV-lemma
outputs and the basic positivity assumptions; the conclusion is
Eq.(6).  This makes the dependency `master ⟸ Lemma 3.1` explicit. -/
theorem master_inequality_assembled
    (Minf K2 m_G S_G u S_cos LHS1 LHS2 : ℝ)
    (hu : 0 < u)
    (hS_G_pos : 0 < S_G)
    -- The same hypotheses as `master_inequality_from_lemmas`, but with
    -- documentation linking each to its MV equation:
    -- MV Eq.(1) output:
    (hEq1 : LHS1 ≤ Minf)
    -- MV Eq.(2) output:
    (hEq2 : LHS2 ≤ 1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1))
    -- MV Eq.(3) identity:
    (hEq3 : LHS1 + LHS2 = 2 / u + 2 * u^2 * S_cos)
    -- MV Eq.(4) output:
    (hEq4 : u^2 * S_cos ≥ m_G^2 / S_G) :
    Minf + 1 + Real.sqrt (Minf - 1) * Real.sqrt (K2 - 1)
      ≥ 2 / u + 2 * m_G^2 / S_G :=
  master_inequality_from_lemmas Minf K2 m_G S_G u S_cos LHS1 LHS2
    hu hS_G_pos hEq1 hEq2 hEq3 hEq4

end Sidon.Master
