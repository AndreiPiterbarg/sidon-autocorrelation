# Notes index

This repository carries a substantial layer of dated research notes
alongside its current documentation. This file is the map.

For the current state of the project, read these first:

| File | What it is |
| --- | --- |
| [`README.md`](README.md) | Project overview, results, build instructions. |
| [`AGENTS.md`](AGENTS.md) | Repository overview for coding agents (you may be one). |
| [`proof/cs-proof/lower_bound_proof.pdf`](proof/cs-proof/lower_bound_proof.pdf) | The cascade manuscript ($C_{1a} \ge 7/5$). |
| [`proof/lasserre-proof/lasserre_lower_bound.pdf`](proof/lasserre-proof/lasserre_lower_bound.pdf) | The Lasserre manuscript ($C_{1a} \ge 1.3$). |
| [`AWS_SETUP.md`](AWS_SETUP.md) | One-time AWS setup for the d=18 Lasserre SDP deploy. |
| [`proof/lower_bound_refs_audit.md`](proof/lower_bound_refs_audit.md) | Cascade bibliography audit (post-rewrite). |
| [`proof/cs-proof/figures/FIGURE_NOTES.md`](proof/cs-proof/figures/FIGURE_NOTES.md) | Figure-by-figure notes for the cascade manuscript. |

Everything below is *historical* — dated session output from earlier
phases of the project. The headline numbers in these files
(e.g. "aiming to push $C_{1a}$ past $1.2802$") have been superseded;
both proofs are now complete. The notes are preserved because they
record the reasoning trail and audits behind specific code changes.

## Top-level research records (historical)

| File | Date | Topic |
| --- | --- | --- |
| `BREAKTHROUGH_HANDOFF.md` | Apr 2026 | Handoff document for a pod-side run. |
| `CASCADE_UPDATE.md` | Apr 2026 | Five validated high-impact ideas for the CPU cascade. |
| `COARSE_CASCADE_PROVER_AUDIT_2.md` | Apr 2026 | Audit of the coarse cascade prover. |
| `COARSE_CASCADE_PROVER_FIXES.md` | Apr 2026 | Fix list following the audit. |
| `COARSE_CASCADE_PROVER_RESULTS.md` | Apr 2026 | Results of the post-fix run. |
| `COARSE_CASCADE_PROVER_SESSION_SUMMARY.md` | Apr 2026 | Session summary for the coarse-cascade work. |
| `DEFERRED_IMPROVEMENTS.md` | Apr 2026 | Improvements deferred from the current sprint. |
| `HIGH_IMPACT_AUDIT.md` | Apr 2026 | Multi-agent audit of tightenings and new directions. |
| `LP_SPEEDUP_AUDIT.md` | Apr 2026 | Audit aimed at speeding up the Pólya / Handelman LP. |
| `NEW.md` | Apr 2026 | Five ideas for cascade feasibility (per-parent certification, SDP relaxation, etc.). |
| `PROBLEM_STATE.md` | 2026-04-14 | Full problem-state snapshot at that date; superseded by the README. |
| `PROMPT_1_AUDIT.md` | Apr 2026 | Audit prompt for publication-grade correctness. |
| `PROMPT_2_FINAL_PUSH.md` | Apr 2026 | Final-push prompt for closing the last numerical gap. |
| `PROMPT_3_SDP_ESCALATION.md` | Apr 2026 | Escalation prompt for the d=30 BnB via Lasserre. |
| `SPEEDUPS_SUMMARY.md` | Apr 2026 | Summary of nine implemented speedups. |
| `STALL_DIAGNOSTIC.md` | Apr 2026 | Diagnostic of an Interval BnB stall. |
| `TRANSFER_SWEEP_PLAN.md` | Apr 2026 | Draft transfer-validation sweep plan. |
| `benefit.md` | Apr 2026 | Five high-impact CPU-cascade ideas (companion to `CASCADE_UPDATE.md`). |
| `new_ideas.md` | Apr 2026 | Re-audited list of ideas with survival counts. |
| `poss_ideas.md` | Apr 2026 | Possible-ideas brainstorm with feasibility notes. |
| `speedup.md` | Apr 2026 | Lasserre L2 d=128 speedup proposals. |
| `sweep_summary.md` | Apr 2026 | Pólya LP sweep mathematical-trend analysis. |

## Proof-supporting notes under `proof/`

| File | Topic |
| --- | --- |
| `proof/coarse_cascade_method.md` | Method description for the coarse cascade. |
| `proof/correction_soundness_gap.md` | Soundness gap analysis for the correction term. |
| `proof/discretization_error_proof.md` | Per-window discretization error proof. |
| `proof/formula_b_coarse_grid_proof.md` | Coarse-grid proof for formula B. |
| `proof/formula_b_soundness_analysis.md` | Soundness analysis for formula B. |
| `proof/fourier_xterm_assessment.md` | Fourier cross-term assessment. |
| `proof/lasserre_unconditional_writeup.md` | Unconditional Lasserre writeup (predecessor to the manuscript). |
| `proof/multifreq_mo217_proof.md` | Multifrequency MO217 proof draft. |
| `proof/part1_framework.md` | Part 1: mathematical framework and parameter derivations. |
| `proof/part2_composition_generation_and_canonical_symmetry.md` | Part 2: composition generation and reversal canonicalization. |
| `proof/part3_autoconvolution_test_values_and_window_scan.md` | Part 3: autoconvolution test values and window scan. |
| `proof/part4_fused_generate_prune_kernel.md` | Part 4: fused generate-and-prune kernel. |
| `proof/part5_fused_kernel_mathematical_soundness.md` | Part 5: fused-kernel mathematical soundness. |
| `proof/part6_cascade_orchestration_completeness_and_deduplication.md` | Part 6: cascade orchestration, completeness, deduplication. |
| `proof/path_a_unconditional_writeup.md` | Path A unconditional writeup. |
| `proof/theory_attempt5_rigorous.md` | Theory attempt 5 (rigorous form). |
| `proof/theory_breakthrough_attempts.md` | Earlier theory breakthrough attempts. |
| `proof/tightest_valid_pruning_bound.md` | Tightest valid pruning-bound derivation. |
| `proof/val_knot_findings.md` | Findings from the val-knot exploration. |

These are research-grade notes. They use math syntax inside markdown
and reference specific code paths in the cascade and Lasserre
pipelines. Treat them as the audit trail behind the published proofs,
not as standalone documentation.

## Subsystem READMEs

Several subdirectories have their own `README.md`:
`bochner_sos/README.md`, `delsarte_dual/README.md`,
`certified_lasserre/`, `gpu/`, etc. Those READMEs describe the
exploration in that subtree.
