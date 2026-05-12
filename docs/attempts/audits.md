# Audits

Three audit cohorts: (1) Python cascade prover `_coarse_bnb_v6.py` at d=16; (2) Lean 4 Sidon project axiom inventory and Cascade-125 pipeline; (3) Path A Hölder sub-attack agent findings.

## Group 1 — Coarse BnB v6 (d=16) audits

Seven specialised audits (A-G) of `_coarse_bnb_v6.py` plus an end-to-end synthesis, plus a Phase-Information SDP strand. Audit F triggered the v4 -> v5 unsoundness fix (MEMORY: project_v4_ljoint_unsound). v6 design synthesised but not run end-to-end at d=16.

| ID | Topic | Outcome |
|----|-------|---------|
| A | SDP/MOSEK tuning | ~3-5x SDP-layer speedup via L_single cache, adaptive tolerance, MOSEK hot-start, skip basis-id, process-pool parallel — ~24x end-to-end |
| B | Frank-Wolfe convergence in `tier_L_joint_lagrangian_v6` | FW closes **0 extra cells** — warm-start vertex is already global maximizer in 32/40 cells; "delta" was solver noise. R1 (early-exit on stationary subgrad) and R2 (skip FW when top-2 L_single gap > 0.05) reclaim ~38% of L_joint cost |
| C | Bound tightening | v6 single-window SDP already rank-1 tight; d=4 uniform's -0.044 IS true value. Only material intra-cell tightener is disjunctive joint Shor lift over top-K windows; palindromic symmetry cheap |
| D | Splitting/branching strategy | All 5 axis heuristics identical close/open; B45 lift closes 0/14 cells. **Split heuristic is not the bottleneck — tier strength is** |
| E | Vectorization | B1diag 19-33x, F-tier 68-85x, batched `_fW_value` 3.1x, `_aggregate` 2.2x — all numerically exact; L_single cache across tiers saves ~13 ms/cell at d=8 |
| F | Soundness | **PASS**: v6 byte-equal to v3/v5, MC-clean on 8 cells incl. degenerate, all guards intact |
| G | Cascade orchestration | Start `d0=4 S=80`; 97% closure at L1 (d=8) for canonical hard cell; `_v6_pod_run.py` driver spec (ProcessPool + checkpoint + adaptive c) |
| Synthesis | Path to C_{1a} >= 1.281 at d=16 | d=16 reality re-prioritises: drop B1diag and L_joint (close 0 at leaves), upgrade F-tier to einsum, restrict window subset to top-K=64, direct MOSEK Task API. v7 spec defined; not executed |
| Phase SDP | Bochner-Phase §5.3c relaxation at m=2..5 | M_cert: 1.270, 1.173, 1.128, 1.101 (numerical); rigorous floor (margin-based) matches within 5e-5. Below MV's 1.27481 at every order; route not on critical path |

**Group verdict:** v6 sound, ~2-5x engineering speedups identified, but the cascade route was superseded by the multi-scale arcsine result before v7 execution. See [cascade_estimator.md](cascade_estimator.md), [gpu_scs.md](gpu_scs.md).

## Group 2 — Lean audits

Three audits of the Lean 4 Sidon project (`lean/Sidon/...`): full axiom inventory, Cascade-125 audit, Cascade-128 pipeline design.

| ID | Topic | Outcome |
|----|-------|---------|
| Axiom audit | 41 `axiom` declarations across 14 files | Distribution A=18 (classical, missing mathlib lemma), B=14 (computational, trusted verifier), C=5 (conjectural / red flag), D=4 (reducible, 1-3 days tactic work each). 4 axioms are immediately discharge-able by `norm_num` or `sq_nonneg`; `mv_master_inequality` and `k26_numeric_constants` flagged as headline-axiom red flags requiring re-anchoring |
| Cascade-125 audit | `Sidon.Cascade125` (proof of C_{1a} >= 5/4 = 1.25) | `lake build Cascade125` succeeds, 0 sorries, 0 new axioms (only `simplex_tv_coverage` is reused, universal in `(n_term, c_target)`). Conditional on the (still open) `refinement_monotonicity` axiom |
| Cascade-128 pipeline design | Design doc for C_{1a} >= 1.279 via two-scale arcsine + MV master inequality | 5-layer dependency tree; 6 pure-math axioms (MV Lemmas 3.1) + 3 numeric axioms (Bessel J_0 cert, single-scale arb cert, multi-scale arb cert). Estimated 3 weeks (axiomatized) or 5-6 weeks (fully sorry-free) |

**Group verdict:** Lean infrastructure is in place. Headline result `C_{1a} >= 1.293` (Sidon.Cascade131, MEMORY: project_lean_cascade131) builds on this foundation with 5 new numerical axioms over Cascade-130 and no new structural axioms. See [../formalization.md](../formalization.md).

## Group 3 — Path A agent findings

Five parallel agent attacks (A-D plus F5) targeting variants of the Path A / Hölder strategy for closing Hyp_R.

| ID | Topic | Outcome |
|----|-------|---------|
| A | Moment couplings C1+C2+C3+C4 | Best LB 1.0000 across all configurations (config x N grid); Shor relaxation detaches moments from f*f structure — M[i,j] pushed to non-realizable configs. Higher N within this framework not expected to recover lost rigidity |
| B | §5.3(d) Smoothed-Autoconvolution Path A | **NEGATIVE** — blocked by 5 independent obstacles. Rescaling for MO 2.14 inflates `M_eff = (1 + 2*eps) M`, breaking validity range; `c_*` never reaches `log 16 / pi`; best LB curve gives only c_eps = 1.87 at eps = 0.3 |
| C | Hard-constrained adversarial Hyp_R(1.378) | **INCONCLUSIVE** — 2078 trials over 4 constraint-projection methods and 13 init strategies produced **zero** in-regime points; restricted class is essentially empty in piecewise-constant step-function space; no continuous bridge from BL (c=0.902, M=1.652) to feasible M<=1.378 |
| D | 2D Hausdorff 4-localizer | Global best LB 1.0000 (below trivial 1.0); probable LP infeasibility / numerical issue; 4-localizer contribution gap = 0 across D in {6, 8, 10, 12} |
| F5 (analysis) | de Branges / Paley-Wiener / Krein-Nudelman | Shor lift drops rank-1 structure of `m m^T`; SDP picks `H = e_N e_N^T` making `b_0 = 1, b_k = 0`, so `M = 1.0` to 1e-9 across N in {4, 6, 8, 10} |

**Group verdict:** All five strands NEGATIVE for unconditional improvement at the $\approx 1.2802$ level. Conditional restricted-Hölder >= 1.37842 remains the best Path A residual. See [path_a_holder.md](path_a_holder.md).

## References

- [../proof_outline.md](../proof_outline.md), [../formalization.md](../formalization.md)
- [multiscale_arcsine.md](multiscale_arcsine.md) — the route that delivered the headline 1.292
- [path_a_holder.md](path_a_holder.md), [path_b_kbk.md](path_b_kbk.md), [cascade_estimator.md](cascade_estimator.md), [lasserre.md](lasserre.md)
