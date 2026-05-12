# Attempts archive

A record of every research direction explored on the Sidon autocorrelation
constant $C_{1a}$. The line that produced the **Piterbarg--Bajaj--Vincent
Bound** $C_{1a} \ge 1.292$ is in
[`multiscale_arcsine.md`](multiscale_arcsine.md); the remaining entries
document routes that were tried and did not produce a publishable
improvement. Each file is self-contained and briefly explains the
construction, the outcome, and the reason for closure.

Code for each direction lives in the companion `archive/<direction>/`
folder at the repository root.

**Status legend.** WORKING — produced the headline rigorous bound;
PARTIAL — lemmas survive and feed into the working line, but the route by
itself did not yield a publishable bound; DEAD — route closed; META — audit
or cross-cutting documentation.

## Index

| File | Status | Direction |
|------|--------|-----------|
| [`multiscale_arcsine.md`](multiscale_arcsine.md) | WORKING | Three-scale arcsine kernel in the MV master inequality. $C_{1a} \ge 1292/1000$; two-scale precursor at $1651/1280$. |
| [`single_kernel_sweep.md`](single_kernel_sweep.md) | DEAD (capped) | 31-kernel single-family Bochner-admissible sweep + QP-reoptimised $G$. Ceiling $\approx 1.276$; only the multi-scale lift breaks past MV's $1.2748$. |
| [`cloninger_steinerberger.md`](cloninger_steinerberger.md) | NOT REPRODUCED | Notes on our limited reproduction effort for Cloninger--Steinerberger (2017). We were unable to replicate their $C_{1a} \ge 1.2802$ end-to-end within our compute budget; we worked instead from the rigorous analytic Matolcsi--Vinuesa prior $\ge 1.2748$. |
| [`lasserre.md`](lasserre.md) | DEAD | Lasserre/SOS hierarchy, Farkas-certified $\mathrm{val}(d)$ cascade, three-point SDP pilot, reverse-Young / $L^{3/2}$ attempt, $d{=}64/128$ plan. Route formally closed by user constraint. Companion paper in [`lasserre_writeup/`](lasserre_writeup/). |
| [`proof_framework.md`](proof_framework.md) | PARTIAL | Six-part formal skeleton for a CS17-style cascade route; supporting lemmas (correction-soundness gap, Lemma 3 discretisation, $m$-chain, windowed $M$-chain). Lemmas survived audit; the cascade did not. Legacy manuscript preserved in [`cs_writeup_legacy/`](cs_writeup_legacy/). |
| [`master_attacks.md`](master_attacks.md) | DEAD | 28-row catalogue of attempted attacks at the $C_{1a} \approx 1.2802$ level (analytic, spectral, SDP/LP, conditional, synthesis). All dead individually; survivors fed lemmas into the multi-scale line that gave the Piterbarg--Bajaj--Vincent Bound. |
| [`cascade_estimator.md`](cascade_estimator.md) | DEAD | Throughput/pruning estimator for the cascade prover; Route C white-transfer $L^{4/3}$ chain; Sonine derivation. Estimator validated, route trivialised to $C_{1a} \ge 1$. |
| [`coarse_lp_bnb.md`](coarse_lp_bnb.md) | DEAD | Coarse LP / branch-and-bound cascade; rigorous $K_2$ findings; sweep summary. Rigor-parity barrier. |
| [`interval_bnb.md`](interval_bnb.md) | DEAD | Interval branch-and-bound on the autoconvolution functional. Phase B T1/T2/T3 cuts went net-negative without a dual certificate. |
| [`gpu_scs.md`](gpu_scs.md) | DEAD | H100 + custom CUDA SCS solver for the Lasserre SDPs. Implementation validated; closed when the Lasserre route itself plateaued. |
| [`path_a_holder.md`](path_a_holder.md) | DEAD | Path A unconditional Hölder + MO Conjecture 2.9 direct attempt. Asymmetric Hyp_R fails; BL 2025 disproves on the full class. |
| [`path_b_kbk.md`](path_b_kbk.md) | DEAD | Path B Krein--Bochner--Krein. Closed. |
| [`audits.md`](audits.md) | META | Independent-probe audits ($A$ through $G$, $d{=}16$ synthesis, phase-SDP $m{=}4$), Lean pipeline / axiom / Cascade-125 audits, and per-research-agent findings. |

## Companion writeups (LaTeX)

| Path | Content |
|------|---------|
| [`cs_writeup_legacy/`](cs_writeup_legacy/) | Precursor manuscript for the CS-style cascade route (`lower_bound_proof.tex`/`.pdf`, refs, figures). Superseded by the publication paper at the repository root. |
| [`lasserre_writeup/`](lasserre_writeup/) | Self-contained Lasserre lower-bound writeup (`lasserre_lower_bound.tex`/`.pdf`, refs, figures). Archived for provenance; the route is closed. |

## See also

- [`../proof_outline.md`](../proof_outline.md) — mathematical summary of the headline result.
- [`../formalization.md`](../formalization.md) — the Lean 4 module and its axioms.
- [`../../CLAUDE.md`](../../CLAUDE.md) — repository-level project notes.
