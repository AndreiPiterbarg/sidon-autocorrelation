# archive/lean_exports/

One-shot helper scripts that emitted rational coefficients into the live
Lean development (which lives at the repo-root `lean/` tree, not here),
plus an axiom verifier. Not used by the current Lean build; retained for
reproducibility of the embedded literals.

## Files

- `_lean_emit_mv_coeffs.py`     — emit the 119 MV coefficients as exact
                                  rationals (`p/q : ℚ`); produced the body
                                  of `mv_coeffs` in `lean/Sidon/CohnElkies125.lean`.
- `_lean_mv_coeffs_emitted.txt` — the resulting Lean rational literals.
- `_lean_emit_v3_g_coeffs.py`   — re-runs the v3 sweep-best QP at
                                  $(\delta_2, \lambda_1) = (0.046, 0.85)$
                                  and emits the 119 $G$ coefficients as
                                  Lean ℚ expressions; certified $M \ge 1.2898$.
- `_lean_verify_axioms.py`      — Lean trusted-tool axiom verifier:
                                  re-checks every numerical axiom in
                                  `lean/Sidon/RationalCert.lean` via
                                  `flint.arb` at 256-bit precision.
- `_lean_verify_axioms_result.json` — last verification certificates output.
