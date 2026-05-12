# Deep-CG Transfer Validation Sweep

> *Historical session note. For current project state see README.md and NOTES_INDEX.md. Both lower-bound proofs are now complete; the framing below dates from earlier exploration.*


**Status:** Draft, pending execution after initial Phase-1/Phase-2 sweep completes.
**Target pod:** RunPod H100 (currently `103.207.149.60:19240`).
**Deliverable:** `data/transfer_sweep_report.json` + ranking-stability table.

---

## 1. Motivation

The initial sweep (`tests/sweep_hyperparams.py`, currently running) measures:

| Phase | What it tests     | CG depth              |
|-------|-------------------|-----------------------|
| P1    | ADMM internals    | Round 0 only (scalar) |
| P2    | Cut selection     | Round 0 + Round 1     |

But **production runs spend >90 % of wall time in CG rounds 3–15**. The
problem that the ADMM sees at Round 0 is not the problem it sees at Round 8 —
it accumulates hundreds of new window-PSD cones and its spectrum, conditioning,
and warm-start geometry shift substantially. A parameter that wins at Round 0
may lose at Round 5 and vice-versa.

We observed this first-hand in earlier d=32 runs:
```
R0: gc ~ 0%       <-- sweep measures ranking here
R1: gc 40-50 %    <-- Phase 2 of sweep measures ranking here
R3: gc 60 %       <-- production runtime starts dominating here
R5: gc 65 %       <-- ~35 % of total wall-time budget
R8: gc 70 %       <-- ~70 % of total wall-time budget
R12: gc > 78 %    <-- NEW lb crossing region
```
The initial sweep is blind to everything below the line at R3.

**Goal of this follow-up:** confirm that the top-k Phase-1/Phase-2 winners still
win at deeper CG rounds — and if any flip, flag it and either re-tune or record
a per-round schedule (e.g. "rho=0.1 early, 0.05 from R4").

---

## 2. Why parameters can flip at higher CG rounds

Mathematical reasons, in decreasing order of expected impact:

1. **ρ (ADMM penalty).**
   Optimal ρ scales with the spectral norm of the constraint matrix. Each new
   window cone adds ~153 rows with nontrivial ‖·‖. After 500 cones, ‖A‖ grows
   ~30 %, which suggests optimal ρ should *decrease* roughly proportionally.
   Flip risk: **HIGH**.

2. **Anderson memory (`aa_mem`).**
   AA Type-I works best when the residual history lives in a low-dimensional
   subspace. At Round 0 the active manifold is ~N_base; at Round 8 it is
   N_base + Σ m_w. Larger search subspace ⇒ larger memory may pay off. Flip
   risk: **MEDIUM**.

3. **AA interval / restart.**
   More constraints ⇒ longer correlations in iterate sequence ⇒ benefit of
   AA is larger but restart period may need to grow. Flip risk: **MEDIUM**.

4. **AA damping β.**
   β controls how aggressively we trust the AA step. Later rounds have more
   ill-conditioned residuals and benefit from stronger damping. Flip risk:
   **LOW–MEDIUM**.

5. **atom-frac (cut blend).**
   At Round 0 the M_1(y) atoms are poorly resolved (y is nearly uniform);
   later rounds have atoms that are clean rank-1 outer products of near-vertex
   μ̂ vectors. Atom ranking should get *better* with round number, so the
   sweet-spot atom-frac may *shift up*. Flip risk: **HIGH**.

6. **cuts-per-round.**
   Fewer cuts per round = cheaper per iteration but slower convergence per
   CG round; this tradeoff changes as the problem fills out. Flip risk:
   **MEDIUM**.

7. **k-vecs (eigvecs per window check).**
   Residual violations become smaller-magnitude but more numerous at deep
   rounds; k=3 may miss diverse directions. Flip risk: **LOW**.

---

## 3. Configurations to validate

After the initial sweep finishes, we select:

- **Top-3 Phase-1 winners** by `scalar_bound` with `scalar_iters ≤ baseline`.
  (Reject faster-but-worse configs.)
- **Top-2 Phase-2 winners** by `cg1_lb`.
- **Baseline** (current defaults).

That's ~6 configs to validate.

---

## 4. Dimension coverage (CG depth, not d)

The user clarified: the goal is *higher CG rounds at d=16 L3*, which is the
production target dimension. We stay at (d=16, order=3, bw=15) but push
deeper into the CG loop:

| Tier  | cg-rounds | bisect | scs-eps | est. wall per config |
|-------|-----------|--------|---------|----------------------|
| T-mid | **5**     | 8      | 1e-6    | ~12–15 min           |
| T-deep| **10**    | 10     | 1e-6    | ~28–35 min           |

The `T-deep` tier is where production runtime dominates. But at 6 configs × 30
min each = 3 h, it's too expensive to run for every config. So we do:

- **T-mid (all 6 configs):** budget 6 × 15 min = 1.5 h.
  Rationale: mid-depth is cheap enough that a full ranking crossover is
  detectable with statistical comfort.
- **T-deep (top 2 from T-mid + baseline):** budget 3 × 35 min = 1.75 h.
  Rationale: deep runs are expensive, so only validate the top candidates.

**Total wall-clock budget:** ~3.25 h, which is ~$45 on an 8×H100 spot.

---

## 5. Methodology

### 5.1 Script
`tests/sweep_transfer.py` (new file, patterned on `sweep_hyperparams.py`).

- Reads `data/sweep_report.json` (initial sweep output).
- Selects top-K Phase-1 winners by scalar_bound, top-K Phase-2 winners by cg1_lb.
- Generates config list for T-mid + T-deep.
- Subprocess-runs each with the appropriate CLI (same env/CLI as initial sweep).
- Parses per-round lb out of each log (regex for `Checkpoint.*cg(\d+).*lb=`).
- Writes incremental JSON so partial progress is never lost.

### 5.2 Metrics per config

For every CG round k = 0, 1, …, cg-rounds:

- `lb_k` (lower bound after round k)
- `gc_k` (gap closure % vs val(16) = 1.319)
- `Δ_k = lb_k − lb_{k-1}` (per-round improvement)
- `wall_k` (per-round wall time)
- `iters_k` (ADMM iters in the final bisection step of round k)

### 5.3 Summary outputs

1. **Per-round ranking table.** For each round, rank the 6 configs by `lb_k`.
   A config is "stable" if its rank varies by ≤ 1 across all rounds.
2. **Per-round ranking table.** Same, but for ADMM `iters_k` (convergence speed).
3. **Δ_k divergence plot.** For each pair (config, baseline) plot `lb_k - lb_k^baseline`
   vs k. If the curve is monotone in one direction, the config dominates or is
   dominated at all rounds; if it crosses zero, we have a round-dependent flip.
4. **Recommendation.** One of:
   - **Single winner**: "Use config X everywhere."
   - **Schedule**: "Use X for rounds 0–3, Y for rounds 4+."
   - **Inconclusive**: re-run with `n_seeds ≥ 3` to beat noise floor.

---

## 6. Success / failure criteria

### Success (accept parameter finding):
- Initial winner keeps top-3 rank at T-mid R5 **and** T-deep R10.
- Δ_k per-round improvement vs baseline is ≥ 0 for every k ≥ 2 (no
  late-round regression).
- Wall-time per round is within 1.3× of baseline.

### Failure (reject finding, fall back to baseline):
- Initial winner drops to rank > 3 at T-mid R5.
- Negative Δ_k vs baseline at any k ≥ 3.
- Any single round exceeds 3× baseline wall (pathological convergence).

### Soft warning (schedule candidate):
- Winner dominates rounds 0–3 but underperforms at rounds 4+, or vice-versa.
  Record as a round-dependent schedule to apply in production.

---

## 7. Soundness guardrails

Two classes of failure to watch for:

1. **LB > val(d):** if any config reports `lb_k > 1.319`, that is a soundness
   violation — halt the run and investigate (almost certainly a verdict
   misclassification or monotonicity break).
2. **Monotonicity break:** `lb_k < lb_{k-1}` is a bug. The CG loop should be
   monotone non-decreasing (each new cone is a *necessary* constraint, so
   removing infeasible points only helps). Flag any decrease > 1e-7.

Both are already checked inside `run_d16_l3.py`. We additionally log them in
the sweep report.

---

## 8. Execution plan

1. Wait for initial sweep (`data/sweep_report.json`) to finish.
2. Write `tests/sweep_transfer.py`:
   - Argument: `--initial-report data/sweep_report.json`.
   - Selects top-K from Phase-1 (by scalar_bound) and Phase-2 (by cg1_lb).
   - Generates config list for T-mid and T-deep.
3. Upload script to pod.
4. Launch in background: `nohup python -u tests/sweep_transfer.py ... &`.
5. Poll every 15 min (longer interval since runs are longer).
6. On completion: teardown pod; pull report locally; summarize.

---

## 9. Open design questions

- **Should we also vary `scs-eps` at higher rounds?** Current production uses
  graduated eps (1e-5 early, 1e-6 late). The initial sweep uses 1e-5 uniform.
  Possible that at R10+ we need 1e-7. *Defer to a Phase-3 sweep.*
- **Cross-seed variance.** ADMM is deterministic given RNG seed, but CG
  selection has tie-breaking stochasticity. Single-run rankings at small Δ_k
  may be noise. *Defer; measure noise floor first from the T-mid data.*
- **Should T-deep include d=32 sanity check?** The user explicitly said this
  is about *CG depth*, not *problem size*, so d=32 is out of scope for this
  plan. A separate dimension-transfer plan can come later if needed.

---

## 10. Appendix — CLI template

```
python tests/run_d16_l3.py \
  --d 16 --order 3 --bw 15 \
  --scs-iters 20000 \
  --scs-eps 1e-6 \
  --k-vecs 3 \
  --rho <CFG_RHO> \
  --atom-frac <CFG_ATOM_FRAC> \
  --cuts-per-round <CFG_CPR> \
  --cg-rounds <TIER_CG> \
  --bisect <TIER_BISECT> \
  --gpu \
  [env: SIDON_AA_MEM=<CFG_MEM> SIDON_AA_INTERVAL=<CFG_INT> \
        SIDON_AA_BETA=<CFG_BETA> SIDON_AA_RST=<CFG_RST>]
```
