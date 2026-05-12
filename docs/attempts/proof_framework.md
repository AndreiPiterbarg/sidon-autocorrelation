# Proof framework — Cloninger--Steinerberger-style cascade skeleton (legacy)

Older formal proof skeleton built for a Cloninger--Steinerberger-style
cascade route to $C_{1a} \ge 1.281$ (and then $\ge 1.30$ once
$\mathrm{val}(16)$ was empirically shown to exceed 1.30). After we
were unable to reproduce the published $1.2802$ figure within our
compute budget (see
[`cloninger_steinerberger.md`](cloninger_steinerberger.md)) and the
user closed B&B / Lasserre / large-SDP tracks, this framework's cascade
direction was **deprecated**. Several constituent lemmas survived audit
and are reused obliquely by the multi-scale arcsine paper at the repo
root.

**Status:** DEAD/PARTIAL. The cascade does not produce the publishable bound.
The active manuscript is the root-level `lower_bound_proof.tex` /
`lower_bound_proof.pdf` (multi-scale arcsine, $C_{1a} \ge 1.292$, Lean 4
formalised). The precursor CS-cascade manuscript has been archived at
[`cs_writeup_legacy/lower_bound_proof.tex`](cs_writeup_legacy/lower_bound_proof.tex).

## The six-part skeleton

A discrete cascade on integer compositions $c \in \mathbb Z_{\ge 0}^d$ with
$\sum c_i = S = 4nm$ (CS fine grid; height $a_i = c_i/m$). For each
$(\ell, s_{\rm lo})$ window, the test value $\mathrm{TV}(c; \ell, s_{\rm lo})$ is
a quadratic functional in $c$; bounding $\mathrm{TV}(c) - \Delta_{\rm tight} > c_{\rm target}$
rules out every step function in the cell $C(c) := \{a : |a-c|_\infty \le 1\}$.
The cascade refines surviving cells through $L_0 \to L_1 \to \dots$ until
exhausted.

| Part | Subject | Status |
|---|---|---|
| 1 | Framework, integer encoding, normalisation, $\mathrm{TV}$ formula, $4n/\ell$ correction factor | rigorous, verified |
| 2 | Composition generation + canonical $\mathbb Z/2$ symmetry; 249 checks pass | rigorous, verified |
| 3 | Autoconvolution, test values, window scan (fine-grid update); 71 checks pass | rigorous, verified |
| 4 | Fused generate+prune kernel vs MATLAB baseline; 98 checks pass | rigorous, verified |
| 5 | Mathematical soundness of every fused-kernel optimisation (9 items) | rigorous, verified |
| 6 | Cascade orchestration, completeness, deduplication, checkpoint integrity (9 items) | rigorous, verified |

The framework rigorously discharges everything *down to* the bin-mass
discretisation; what it cannot deliver is a closing $\mathrm{TV}$-side bound at
$d=16$ or higher without the (now-forbidden) numerical SDP / B&B lift.

## Key surviving lemmas

These were independently audited and remain sound; several are cited
obliquely by [`multiscale_arcsine.md`](multiscale_arcsine.md) and the
root-level paper.

- **Correction-soundness gap** (`correction_soundness_gap.md`). Resolved
  2026-04-07: the CS Lemma 3 correction $2/m + 1/m^2$ applies *directly* on
  the fine grid (compositions sum to $S = 4nm$, heights are multiples of
  $1/m$), not on the old coarse grid ($\sum c_i = m$, heights $c_i \cdot 4n/m$).
- **Lemma 3 (discretisation error)** (`discretization_error_proof.md`). Tightest
  per-window bound under cumulative-floor discretisation:
  $\mathrm{TV}_{n,m}(c;\ell,s_0) - \mathrm{TV}_n^{\rm cont}(f;\ell,s_0) \le (4n/\ell)\cdot(2W_\mu/m + |\mathcal B|/m^2)$.
  The MATLAB "Formula B" $\Delta_B = 1/m^2 + 2W/m$ is **not** a valid
  per-window bound — the $4n/\ell$ factor cannot be removed; verified
  counterexample at $d=4, m=10, c=(2,1,1,6)$, window $(\ell=2, s=6)$.
- **m-chain bound** (`m_chain_proof.md`). Per-conv-position LB built from
  cell-feasible $\delta$-perturbations; sound and empirically validated at
  $(n_{\rm half},m) = (1,20)$.
- **Windowed M-chain** (`windowed_M_chain.md`). Cannot strictly strengthen
  the cascade's existing dynamic-threshold check: the bridge applies the
  cell-allowance $\Delta_{\rm tight}$ to $\mathrm{TV}(c)$ directly, while a
  windowed M-chain bounds $\min_{a\in C(c)} \mathrm{TV}(a)$ — different
  quantities. 30/81 compositions pass at L0; 51/81 fail. Sound, but inert.
- **Bochner test-function bound** (`bochner_test_function_bound.md`),
  **bin-average functional**, **coarse cascade method**, **tightest valid
  pruning bound**, **multifreq MO 2.17 proof** — all standalone supporting
  lemmas, audited sound, none alone reaches the $1.2802$ regime.
- **Continuity / extraction**: `cs2017_continuity_extraction.md`,
  `fourier_continuity_bound.md`, `linfty_continuity_bound.md`,
  `tv_w_continuous_bridge.md`, `fourier_xterm_assessment.md` — fine grid to
  continuous bridge; rigorous, used to justify the fine-grid correction.
- **Formula-B coarse-grid proof** and **soundness analysis** — diagnose the
  unsound bound that motivated the fine-grid switch.
- **References audit** (`lower_bound_refs_audit.md`), **`val_knot_findings.md`**,
  **`FIGURE_NOTES.md`** — bookkeeping and figure provenance.

## Closed theory attempts

These were exploratory writeups that did not produce a publishable bound and
do not feed the multi-scale line:

- **`theory_attempt5_rigorous.md`** — Plancherel chain
  $C_{1a} \ge \inf_f \|\hat f\|_4^4$; uniform $f$ gives 4/3 ≈ 1.333 but
  $\inf_f \|\hat f\|_4^4 \le 1.276$ matches MV 2010; no new lower bound.
- **`theory_breakthrough_attempts.md`** — even-odd decomposition +
  Cauchy-Schwarz at $t=0$: $(f*f)(0) \ge 4 - \|f\|_2^2$; constrains only
  bounded-$L^2$ $f$, $\|f\|_2^2$ is unbounded over the admissible class.
- **`path_a_unconditional_writeup.md`** — 7 attacks on Hyp_R; none yields
  $c^* < 1$ for general asymmetric $f$. See [`path_a_holder.md`](path_a_holder.md);
  MEMORY `project_path_a_unconditional_status.md`.
- **`lasserre_unconditional_writeup.md`** — end-to-end formal target for
  Farkas $d=64$ cert; never executed (route closed by user constraint). See
  [`lasserre.md`](lasserre.md).
- **`lasserre_d64_status.md`** — audit confirming
  $(d,k,b)=(64,2,16)$ is not certified tight enough; an order-3 escalation
  or a $(16,3)$ dense cert is the canonical alternative, neither run.

## Verdict

Superseded by [`multiscale_arcsine.md`](multiscale_arcsine.md); the
[`cs_writeup_legacy/lower_bound_proof.tex`](cs_writeup_legacy/lower_bound_proof.tex)
is the precursor manuscript, replaced by the root-level
`lower_bound_proof.tex` (multi-scale arcsine, $C_{1a} \ge 1.292$).

The cascade skeleton produces no publishable bound under the conservative
analytic prior we adopted and the user constraint on B&B / Lasserre /
large SDP. Its surviving lemmas (discretisation error, m-chain,
continuity / fine-grid bridge, $4n/\ell$ correction factor) are sound and
reusable; the structural framework (parts 1-6) is also sound and
well-audited. The reason the framework is closed is **strategic**, not
technical: every closing move at $d \ge 16$ needs an SDP or large
enumeration the user has ruled out.

## Cross-references

[`lasserre.md`](lasserre.md) — SDP-side counterpart on the same `val(d)`
discretisation; [`cascade_estimator.md`](cascade_estimator.md) — orchestration
infrastructure for cell-cascade evaluation;
[`coarse_lp_bnb.md`](coarse_lp_bnb.md) — LP/BnB alternative to the SDP
cascade; [`interval_bnb.md`](interval_bnb.md) — interval-arithmetic B&B
on the same cell formulation; [`master_attacks.md`](master_attacks.md) —
table of analytic / spectral / SDP-LP / conditional attacks adjacent to
this framework; [`audits.md`](audits.md) — independent verification of
the surviving lemmas.
