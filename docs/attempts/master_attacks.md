# Master attacks — 28-line catalogue

One row per attempted attack at the $C_{1a} > 1.2802$ level, grouped by
family. Each attack was its own standalone writeup (the `_master_*.md`
files in the deprecated `master_attacks/` subfolder). **All 28 are dead**
at the $C_{1a} > 1.2802$ level *individually*; several feed surviving
lemmas or sub-questions into the multi-scale arcsine line that produces
the **Piterbarg--Bajaj--Vincent Bound** $C_{1a} \ge 1.292$ (see
[`multiscale_arcsine.md`](multiscale_arcsine.md)).

Status of project at synthesis time: working conservatively from the
rigorous analytic Matolcsi--Vinuesa bound $C_{1a} \ge 1.27481$ (2010)
because we were unable to reproduce the announced
Cloninger--Steinerberger (2017) value of $1.2802$ within our compute
budget (see [`cloninger_steinerberger.md`](cloninger_steinerberger.md)).
Target: strictly beat $1.27481$ by mathematical means (user constraint:
no further B&B / Lasserre / large SDP).

## Table

| Attack (stem) | Family | What was tried | Outcome |
|---|---|---|---|
| `beckner_sharp` | analytic | Beckner-sharpened Hausdorff-Young in MV's Hölder slack; isolate the two interpolation moves and ask if the sharp constant $A_p A_q A_{r'}$ tightens either | The sharp Beckner deficit is in the Cauchy-Schwarz tail (MV Lemma 3.3), not in any reachable step; gain $< 10^{-4}$. |
| `boyer_li_ai` | analytic | Adapt Boyer-Li 2025's AI-discovered $L^2$-flatness witness $c \ge 0.901564$ to $C_{1a}$ | Boyer-Li is a UB witness ($\|f*f\|_\infty$ lower-bounded by $L^2$ ratio cannot exceed 1 universally); no LB transfer. Salvage: their exact-rational step-function representation could replace MV's 119-cosine basis. |
| `cohn_goncalves` | analytic | Cohn-Gonçalves sign-uncertainty / $\pm$-eigenfunction split (the only untried Cohn-Elkies extension; see CLAUDE.md "next steps") | Mathematically clean; does not produce $C_{1a} > 1.2802$ in any tested form. **Still open as Cohn-Gonçalves-style $\pm$-eigenfunction split with complex Schur multiplier (`phase_sdp_design` companion).** |
| `fhat_squared` | analytic | $\widehat{f*f} = \hat f^2 \succeq 0$ as a Hamburger moment constraint on $[0,\|f\|_2^2]$ in an SDP | Univariate moment problem; gain estimated $10^{-3}$ at $d=4$, $\le 0.01$ at $d=8$. Plausibly $1.279-1.285$, below the Cloninger--Steinerberger value $1.2802$ and inside the MV $\approx 1.276$ analytic ceiling. |
| `b2_bridge` | analytic | Discrete $B_2[g]$ Sidon sets $\to$ continuous $C_{1a}$ via Schinzel-Schmidt / Cilleruelo-Ruzsa-Vinuesa $\lim_g \sigma_2(g) = 1/\sqrt\sigma$ | Bridge runs continuous $\to$ discrete only; no discrete-to-continuous transfer (Riblet-Schehr 2025, Carter-Hunter-O'Bryant 2023). Baseline stays at MV 1.2748. |
| `bochner_phase` | analytic | Phase-carrying (complex-valued) Bochner kernels lifting magnitude-only $|\hat f(\xi_j)|^2$ moments to phase moments $\cos 2\theta_j$ | Phase-only erased by the integrand $f*f + f\circ f$; equivalent to magnitude-only when hermitian, or trivially worse adversarially. Real-improvement requires joint kernel/window constraints. |
| `schoen` | spectral | Schoen / Croot-Sisask almost-periodicity, polynomial method, Bohr-set / Bogolyubov-Ruzsa | All techniques produce *integer-side* density bounds (cap-set, Roth) or work in $\mathbb F_p^n$; no continuous transfer. Closest in spirit: Eberhard-Manners-Mrazović 2023 Fourier-uniformity of extremal Sidon sets — does not bound $C_{1a}$. |
| `selberg_vaaler` | spectral | Selberg-Vaaler entire-function quadrature majorants / minorants for $\sup(f*f)$ via Paley-Wiener-constrained $\hat K$ | Subsumed by MV's 119-cosine $G$ optimum at 1.2748; marginal-gain variants (Vaaler-symmetrised band) tested — none crosses 1.2748. |
| `krein_milman` | spectral | Krein-Milman / Bauer max principle on the admissible set, atomic-extremizer characterisation | The minimiser of $\Phi(f) = \|f*f\|_\infty$ is *not* extreme in $\mathrm{ext}(\mathcal A) = \{\delta_x\}$ (the Diracs give $\Phi = \infty$). Convex but pointwise sup of quadratics; gives only structural info. |
| `spectral_concentration` | spectral | Bourgain-Katz / SCS spectral concentration; Logvinenko-Sereda / prolate eigenvalue inequalities on $\hat f \in PW_{\pi/2}$ | In continuous Paley-Wiener, no genuine Bohr-set obstruction; prolate eigenvalues give a constant gap but worse than MV. |
| `positive_def` | spectral | $C_{1a}^{\rm PD} := \inf$ over the doubly non-negative class $\{f\ge 0, \hat f \ge 0\}$; $f*f$ becomes positive-definite so $\sup = (f*f)(0) = \|f\|_2^2$ | $C_{1a}^{\rm PD} \approx 2.299$, far above UB 1.5029. Extremizers of $C_{1a}$ are **not** PD; the PD subclass costs ~53\% more. Useful **structural** result (rules out PD ansatz). |
| `lasserre_global` | sdp_lp | Global Farkas-certified $\mathrm{val}(d)$ at $d \ge 32$ + relaxation order $\ge 3$ | Memory scaling: order-3 dense at $d=64$ has $n_{\rm basis} = 47\,905 \to$ >20 TB PSD cone, infeasible. Sparse banded $(k,b)=(2,16)$ at $d=64$ feasible but tightness uncertain. Closed by user constraint. See [`lasserre.md`](lasserre.md). |
| `sos_clique` | sdp_lp | Clique-decomposed sparse Lasserre/SOS hierarchy via Waki-Kim-Kojima-Muramatsu correlative-sparsity | Correlative graph of $C_{1a}$ at level $d$ is $K_d$ (complete — windows link every pair), so naive WKKM decomposition is vacuous; banded $b=16$ chordal extension works but $\mathrm{val}^{(2,16)}(16) \le 1.25$. Same root cause as `lasserre_global`. |
| `ce_lp` | sdp_lp | Cohn-Elkies LP adapted to $C_{1a}$ via Bochner-positivity of $\hat K \ge 0$ and pairing $\int K(f*f) \ge \beta \int f^2$ | Reduces *exactly* to the MV dual already in `delsarte_dual/`; theoretical ceiling $\approx 1.276$, practical 1.27481 (MV's 119-term $G$). |
| `phase_info_sdp` | sdp_lp | Phase-information SDP (Path A §5.3c, Bochner-phase) — lift moments on phase angles $\theta_j$ of $\hat f(\xi_j)$, not just magnitudes | The only untried Path-A tool per MEMORY `project_6month_strategy_2026.md`; target 1.378. Worked-out m=2 example shows feasibility; closed by user SDP constraint before m=4 SDP could be run. |
| `phase_sdp_design` | sdp_lp | Master design: asymmetric Bochner-phase SDP with rank-1 / small-rank certificate discharged by hand | Honest baseline 1.2748; design + cost estimate only, never executed. Lane Backup-2 of `project_6month_strategy_2026.md`. |
| `hyp_r_unconditional` | conditional | Three candidate routes to make Hyp_R unconditional: $L^\infty$ truncation, Riesz rearrangement to symmetric class, BL-2025-style classification | Hyp_R must hold on the *entire sub-level set* $\{f : \|f*f\|_\infty \le 1.378\}$, not just at the minimiser (proof-by-contradiction structure); no route closes. |
| `restricted_holder_cubature` | conditional | Plug unconditional UB $M_{\max} = 1.5029$ (or witness threshold 1.652) into restricted-Hölder $1.37842$ | The LB 1.37842 does **not** depend on $M_{\max}$; depends on the unproved Hölder-slack $c < 1$. Plugging $M_{\max}$ unlocks nothing. See [`path_a_holder.md`](path_a_holder.md). |
| `mo217_m4` | conditional | Extend MO 2004 Prop 2.11 from $m=3$ to $m=4$ (multifreq lift) in joint MV+P framework | Same structural failure as m=3: P-side ceiling does not shrink with $m$ (in fact grows); MV-side remains binding at 1.2748. See MEMORY `project_mo217_p211_m3_dead.md`. |
| `lemma_a1_attack` | conditional | Bochner-phase identity $\sum_{n\ne 0}z_n^2 \cos(2\theta_n + 2\pi n t^*) = (M-1)/2$ to derive $\|f\|_2^2 \le 2M - 1$ | Lemma A1's proof has a sign error in step 3; correcting gives the opposite inequality $\|f\|_2^2 \ge M$. Even with the desired bound, insufficient to close Hyp_R at $M=1.378$. |
| `route_b_symmetric` | conditional | Theorem C: prove minimiser $f_\star$ is symmetric (modulo translation), then Hyp_R closes via symmetric-only steps S1, S2 | Schinzel-Schmidt symmetric-extremizer conjecture was **disproved** (BL 2025); MV's near-optimizer is asymmetric. Schwarz rearrangement runs wrong direction. |
| `compactness` | conditional | Direct method of CoV: existence of extremizer $f_\star$ for $\Phi(f) = \|f*f\|_\infty$ on $\mathcal A$ via weak-$*$ compactness of $\mathcal A \subset M_+([-1/4,1/4])$ | Existence theorem rigorous (Theorem 1, Bauer max principle). Gives structural info on the extremizer but no quantitative LB improvement on its own. |
| `publishable_outcome` | synthesis | "Brutally honest" decision of the best publishable rigorous LB from artifacts available 2026-05-11 | Verdict: $C_{1a} \ge 1.27428679 \approx 1.2742$ (MV via Cohn-Elkies dual cert in `_cohn_elkies_125.py`, Lean'd in `CohnElkies125.lean`). **Now superseded** by 3-scale arcsine 1.292. |
| `synth_round2` | synthesis | Round-2 master synthesis after 10 parallel agents (continuous-$\nu$ FW, Theorem 4 atomic-$\nu$, Route C, MO Conj 2.9, Hyp_R, Cohn-Gonçalves) | Headline lifts to $C_{1a} \ge 1293/1000 = 1.293$ (Cascade131 Lean module). Beyond-framework lanes all NEGATIVE this round; no path to 1.30 found. |
| `blocker_diagnosis` | synthesis | Identify the exact saturation point in MV's quadratic master equation; characterise why $C_{1a}$ stalled at 1.2748 since 2009 | MV's discriminant-tight $G_*(M)$ saturates at $M_*=1.2748$ for the arcsine $K$; structural cap is the magnitude-only ($z_j^2$) framework. Escape: phase moments or kernel mixes. Confirmed by external (Tao repo). |
| `literature_2024_26` | synthesis | Survey of 2022–2026 papers on $C_{1a}$ and adjacent autocorrelation / Sidon problems | No paper 2022-2026 improves on $C_{1a} \ge 1.2748$. All progress is UB-side (AlphaEvolve, ThetaEvolve, TTT-Discover, Together AI; LB stuck since MV 2009/2010). |
| `probabilistic` | synthesis | Random construction / concentration of measure / probabilistic-method LB | Random construction gives UBs, not LBs (wrong direction for $\inf$). Indirect route via anti-CE Bochner kernel sampling reduces to deterministic MV-style bound; sweep negative. |
| `tao_polymath` | synthesis | Mine Tao's optimization-constants repo, Polymath wikis, MathOverflow, Erdős-problems site for unpublished LB progress | No community-vetted unpublished LB above 1.28. Tao's repo lists Xie/Grok 1.2802 as **unpublished** AI-derived recipe with no audit trail. |

## Synthesis

**Most promising still-open route:** `cohn_goncalves` — Cohn-Gonçalves
sign-uncertainty is the only untried Cohn-Elkies extension per CLAUDE.md "next
steps" and MEMORY `project_cohn_elkies_status.md`. Its mathematical structure
(modular forms, $\pm$-eigenfunction split, Viazovska-style construction) does
not collapse to MV. Posterior $\sim 15\%$. Other lanes with non-trivial
posterior: `phase_info_sdp` / `phase_sdp_design` (closed by user SDP
constraint but mathematically alive at small rank), `beckner_sharp` /
`fhat_squared` (small-gain analytic, ceiling $\le 1.285$).

**Structural dead-ends** (cannot produce $C_{1a} > 1.2748$ even with
unlimited compute, by published obstruction proof):
`positive_def` (PD subclass costs $\ge 53\%$),
`b2_bridge` (one-way only),
`schoen` (no continuous transfer),
`probabilistic` (wrong inequality direction),
`route_b_symmetric` (BL 2025 disproved the symmetric-extremizer conjecture),
`krein_milman` (Diracs are extreme but give $\Phi = \infty$),
`spectral_concentration` (no Bohr obstruction in continuous Paley-Wiener),
`b2_bridge` and `tao_polymath` are *certified* dead (external audit / community).

**Feeds into multi-scale arcsine line** (lemmas reused obliquely): the
`compactness` existence theorem, `ce_lp` reduction to MV-dual, `bochner_phase`
identity at the autocorrelation peak, and `blocker_diagnosis`'s
characterisation of the MV saturation point are all cited in the multi-scale
writeup at the repo root.

**Verdict.** None of the 28 individual attacks crosses the $1.2802$ level on its own.
The 1.292 headline comes from a different mechanism: convex combination of
admissible kernels (multi-scale arcsine) lifts MV's $G$-side QP denominators
off the zeros of $J_0$, dropping $S_1$ from 87.4 to $\approx 30$ and boosting
the gain term — see [`multiscale_arcsine.md`](multiscale_arcsine.md). The
"surviving" lemmas across this catalogue serve as audit anchors for the
multi-scale paper, not as standalone improvements.

## Cross-references

[`multiscale_arcsine.md`](multiscale_arcsine.md) — the headline route;
[`lasserre.md`](lasserre.md) — SDP / SOS family (rows
`lasserre_global`, `sos_clique`, `ce_lp`, `phase_info_sdp`, `phase_sdp_design`);
[`path_a_holder.md`](path_a_holder.md) — Hyp_R / restricted-Hölder family
(rows `hyp_r_unconditional`, `restricted_holder_cubature`, `lemma_a1_attack`,
`route_b_symmetric`, `mo217_m4`);
[`cloninger_steinerberger.md`](cloninger_steinerberger.md) — the cascade route
the `compactness` and `blocker_diagnosis` writeups frame;
[`proof_framework.md`](proof_framework.md) — six-part CS-cascade skeleton;
[`single_kernel_sweep.md`](single_kernel_sweep.md) — the 14-kernel
Bochner-admissible sweep that confirmed no single non-arcsine $K$ beats MV;
[`audits.md`](audits.md) — independent verification of the multi-scale claim
and of the surviving lemmas catalogued here.
