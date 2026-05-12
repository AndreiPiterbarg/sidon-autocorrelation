/-
Sidon Autocorrelation Project — Quick-Check Heuristic Soundness (Proof Stubs)

The quick-check optimization re-tries the previous child's killing window
on the next child. If the window sum still exceeds the threshold, the child
is pruned in O(ell) instead of the full O(d^2) scan.

This is TRIVIALLY SOUND: it's just testing a specific window. If that window
prunes, the child is legitimately pruned. If not, the full scan follows.

Source: proof/coarse_cascade_method.md Section 6.6.
-/

import Sidon.Proof.CoarseCascade

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-- **Quick-check soundness:** Testing a SPECIFIC window (ell, s) and finding
    that ws > threshold is sufficient to prune.

    This is sound because the pruning criterion is:
      exists window (ell, s) with TV_W(mu, ell, s) >= c_target

    The quick-check just happens to try a specific window first (the one
    that killed the previous child). If it works, no need to scan all windows.

    This is a special case of the window scan — no new mathematics needed. -/
theorem quick_check_sound {d : ℕ}
    (μ : Fin d → ℝ) (c_target : ℝ)
    (ell s : ℕ) (hell : 2 ≤ ell) (hs : s + ell ≤ 2 * d)
    (h_tv : mass_test_value d μ ell s ≥ c_target) :
    ∃ ell' s', 2 ≤ ell' ∧ mass_test_value d μ ell' s' ≥ c_target := by
  exact ⟨ell, s, hell, h_tv⟩

end -- noncomputable section
