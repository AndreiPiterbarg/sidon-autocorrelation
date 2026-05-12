# Prompt 3 — Engineer Lasserre-O2 SDP escalation to unblock the d=30 BnB

**Working dir:** `C:/Users/andre/OneDrive - PennO365/Desktop/compact_sidon`
**Goal:** Achieve a rigorous BnB certificate `val(d) >= 1.281` at `d = 30` (or higher), giving `C_{1a} >= 1.281`. This is the strongest unconditional lower bound that's been attempted in this codebase.

**THIS IS PUBLISHABLE RESEARCH** — every cert must carry a written proof of soundness, not just empirical pass.

---

## 1. Why SDP escalation is necessary (current state)

### 1.1 The cascade and where it stalls

The current parallel BnB at `interval_bnb/parallel.py` runs each box through this cascade (in order):

1. Cheap rigor (`natural`, `autoconv`, McCormick `SW`/`NE`, joint-face dual cert) — closes most "easy" boxes (high `f(center)`, concentrated mass)
2. Multi-anchor (`bound_anchor_multi_int_ge` at `mu*` and `sigma(mu*)`) — fast filter, ~10μs/call
3. CCTR (multi-α aggregates)
4. Epigraph LP (`bound_epigraph_int_ge_with_marginals`) — McCormick relaxation of `min_B max_W mu^T A_W mu`
5. Per-box centroid anchor (`bound_anchor_centroid_int_ge`) — last-resort, only at depth ≥ 80
6. Split

At `d=22, 24` the cascade is sufficient to close `target=1.281` modulo a tiny residual. At `d=30` the cascade stalls at the `hw ≈ 0.025` *transition zone* with cert rate **51%** (vs 90% at d=22) — random walk boundary, tree explodes.

### 1.2 The structural diagnosis (already done — don't redo)

Diagnostic files: `diagnose_d30.py`, `diagnose_cheap.py`, `diagnose_full.py`, `diagnose_sdp.py`, plus their `*_output.log`s.

Key findings:

- **LP cert threshold is `hw ≈ 0.013` at all `d`.** Once boxes are this small, the McCormick LP gap is below margin. (Verified empirically: `d=22, 24, 30` all show `100%` LP cert at `hw ≤ 0.01`, `~0%` at `hw ≥ 0.025`.)
- **The cheap tier collapses at `d=30`** (22% → 2% pass rate at `hw=0.025`) because at higher `d`, random Dirichlet `mu` are more uniform, fewer windows have `TV > target`, so `lb_fast` rarely passes.
- **At `d=30, hw=0.025`, total cert rate is 51%** — the BnB random walk has +0.02/box drift, tree grows exponentially before workers can dig deep enough.
- **Lasserre-2 SDP relaxation has `O(w^4)` gap** vs LP's `O(w^2)`. At `hw=0.025` the SDP gap is ~`0.025^4 * const ≈ 4e-7`, well below margin `0.051` for `target=1.281`. **SDP would close the d=30 transition zone in one shot.**

### 1.3 What's broken about the current SDP code

`interval_bnb/lasserre_cert.py` exists but:
- Uses `cvxpy` to model + `Clarabel` (or `MOSEK`) to solve.
- At `d=30, order=2`: CVXPY emits `UserWarning: Constraint #2 contains too many subexpressions. Consider vectorizing your CVXPY code to speed up compilation.`
- **Empirical: at `d=30` the CVXPY compilation uses 7.8 GB RAM and hangs ≥ 2.5 minutes BEFORE the solver runs.** Not viable for per-box BnB invocation.
- `interval_bnb/bound_sdp_escalation.py` was started by a prior agent (Agent 2 from `PROMPT_2_FINAL_PUSH.md`) but the agent was killed before completion. The file may be incomplete; treat as draft, do not trust.

The blocker is **CVXPY's symbolic compilation at the moment-matrix and localizing-constraint construction**, not the SDP solver itself. MOSEK is installed (verified: `cp.installed_solvers()` includes `'MOSEK'`), and at `d=22, order=2` MOSEK can solve in ~1-2s.

### 1.4 Quantitative target

After the SDP escalation tier is integrated:

- `d=30 hw=0.025` cert rate (any tier) ≥ **70%** (vs current 51%)
- Per-box SDP solve at `d=30, order=2`: **≤ 3s** (vs current 7.8GB hang)
- `d=30 t=1.281` BnB completes (`success=True`, `in_flight_final=0`) within **6h on 16 cores**
- All existing tests still pass (`pytest interval_bnb/test_*.py` — 50 tests as of last run)
- Soundness: each SDP cert backed by a written derivation (cushion ≥ 100× worst-case dual residual, OR a Farkas integer dual cert)

---

## 2. The audit + agent-deployment task

### Step 1 — Single-agent AUDIT (read-only, ~15 min)

Spawn ONE `Plan`/`Explore` agent (foreground) with this prompt:

> Read these files in order:
> 1. `interval_bnb/THEOREM.md` — the rigorous proof that `val(d) <= C_{1a}` and the H_d half-simplex symmetry reduction.
> 2. `interval_bnb/parallel.py` lines 1-300 (cascade structure, env-var thresholds, anchor and centroid integration points).
> 3. `interval_bnb/parallel.py` lines 500-650 (the actual cascade ordering: multi-anchor → CCTR → epi LP → centroid).
> 4. `interval_bnb/lasserre_cert.py` (the existing CVXPY-based SDP).
> 5. `interval_bnb/bound_sdp_escalation.py` (Agent 2's draft — assess completeness).
> 6. `lasserre/d64_farkas_cert.py` (existing rigorous Farkas-cert SDP machinery for the lasserre/ directory; this is a reference for what "rigorous SDP cert" looks like in this codebase).
> 7. `diagnose_sdp.py` and `diagnose_full.py` for the empirical context.
>
> Then answer concretely:
>
> A. Where exactly in `lasserre_cert.py` does CVXPY's compilation blow up at `d=30`? Identify the specific constraint construction loop. (Likely the moment-matrix or localizing-constraint block.)
> B. Could the existing CVXPY code be vectorized in-place (using `cp.Variable` shapes instead of element-wise loops)? If yes, estimate compilation memory at `d=30` after vectorization.
> C. Is `bound_sdp_escalation.py` salvageable, or should it be rewritten? What's its API and which parts work?
> D. Is the existing `lasserre/d64_farkas_cert.py` Farkas-cert pattern reusable for our per-box BnB use? It targets `d=64` so the moment matrix is even larger — how does it avoid CVXPY blow-up?
> E. The current cascade uses LP cushions of `100 * dual_feas_tol = 1e-7`. What's the equivalent rigorous cushion for an SDP solved via MOSEK? (Cite MOSEK's reported residuals or a Neumaier-Shcherbina-equivalent for SDP.)
> F. Propose three implementation paths in priority order: (1) vectorized CVXPY, (2) direct MOSEK Fusion API, (3) bypass-modeling-layer (build constraint matrices in numpy, call MOSEK Optimizer API). For each, estimate engineering effort and per-box solve time at `d=30`.
>
> Output a structured report with these answers and three implementation paths. **Do not write code.**

The audit is needed because the CVXPY bottleneck and the existing SDP code structure determine which agents to spawn next.

### Step 2 — DEPLOY AGENTS in parallel for the chosen implementation path

Based on Step 1's report, deploy 2-4 agents concurrently. Use disjoint files to avoid merge conflicts (use `isolation: "worktree"` if needed). Candidate agent specifications:

#### Agent A: Vectorized CVXPY rewrite of `lasserre_cert.py` (most likely path)

> **Goal**: Eliminate the "too many subexpressions" warning by replacing element-wise CVXPY construction with vectorized `cp.Variable` of correct shape. Soundness must be preserved.
>
> **Specific files**:
> - Modify: `interval_bnb/lasserre_cert.py` (the existing function `lasserre_box_lb_float`)
> - Add: `interval_bnb/test_lasserre_vectorized.py` (new tests)
>
> **Mathematical content**:
> Lasserre order-2 SDP for `min_{mu in B ∩ Δ_d} max_W mu^T A_W mu`:
>
> Variables:
> - `y_alpha` for `|alpha| ≤ 4`, multi-index `alpha ∈ N^d`. Use `cp.Variable(n_monos)`.
> - `u` (the upper bound on z, `cp.Variable()`).
>
> Constraints:
> 1. **Moment matrix M_2(y) ⪰ 0**: a `binom(d+2, 2)` × `binom(d+2, 2)` PSD constraint. Encode as `cp.bmat` over a *vectorized* index map. Use `cp.PSD` constraint with the symmetric matrix built via `cp.reshape` and indexing into `y`. Avoid Python loops over individual entries.
> 2. **Box localizing**: for each `i = 0..d-1`:
>    - `M_1((mu_i - lo_i) y) ⪰ 0`: a `binom(d+1, 1) × binom(d+1, 1)` PSD constraint.
>    - `M_1((hi_i - mu_i) y) ⪰ 0`: same shape.
>    Build these as block CVXPY expressions, vectorized.
> 3. **Simplex**: `y_0 = 1`, `sum_{|alpha|=1} y_alpha = 1`.
> 4. **Nonnegativity**: `y_alpha >= 0` for all alpha.
> 5. **Window constraints**: for each `W ∈ W_d`, `u >= scale_W * sum_{(i,j) ∈ pairs_all(W)} y_{e_i + e_j}`. Encode as a single matrix-vector multiplication: build `B (|W| × n_monos)` with `B[W, alpha_index(e_i + e_j)] = scale_W`, then `cp.constraint: B @ y <= u * np.ones(|W|)`.
> 6. **Objective**: `cp.Minimize(u)`.
>
> Rigor cushion (publication-required): solve with MOSEK at `mosek.dparam.intpnt_co_tol_pfeas=1e-9, intpnt_co_tol_dfeas=1e-9, intpnt_co_tol_rel_gap=1e-9`. After solve, certify only if `u_LB - cushion >= target` where `cushion = 100 * max(dfeas_residual, 1e-9)`. Document the cushion derivation in the docstring (Neumaier-Shcherbina-style argument: `|u* - u_true| <= ε * (||c||_∞ + ||b||_∞ + ||A||_∞ * ||y_max||_∞)`; bound each term in your derivation).
>
> **API**: keep the existing `lasserre_box_lb_float(lo, hi, windows, d, order=2, solver='MOSEK', verbose=False)` signature. Add a new function `lasserre_box_cert_int_ge(lo_int, hi_int, windows, d, target_num, target_den, *, order=2, solver='MOSEK')` that returns `True/False` with the rigor cushion, integer arithmetic at the cushion level. Empty domain returns `True` (vacuous, matches existing `bound_anchor_int_ge` and `bound_epigraph_int_ge` convention).
>
> **Tests** (`test_lasserre_vectorized.py`):
> - `test_vectorized_matches_legacy_d4` — both implementations give the same `u_LB` at `d=4` on a small box, within 1e-6.
> - `test_vectorized_compiles_at_d30` — compiles in `< 30s` and uses `< 1 GB` memory at `d=30, order=2`.
> - `test_vectorized_solves_at_d30_hw_0025` — solves a `d=30 hw=0.025` box around `sigma(mu_star_d30)` in `< 5s` and produces `u_LB` within 1e-6 of the legacy CVXPY value (if legacy can solve at all).
> - `test_vectorized_certifies_failing_lp_box_d30` — for the LP-failing boxes generated by `diagnose_full.py`, the new SDP cert returns `True`.
> - `test_vectorized_cushion_blocks_borderline` — construct a box where SDP `u_LB ≈ target + 5e-8`; confirm cushion blocks the cert (rigor preserved).
>
> **Regression**: run `python -m pytest interval_bnb/test_anchor.py interval_bnb/test_epigraph.py interval_bnb/test_epigraph_extra_cuts.py interval_bnb/test_hd_cut.py interval_bnb/test_lp_split.py interval_bnb/test_cctr.py -q`. All 50 tests must pass.
>
> **Smoke**: run `python interval_bnb/run_d10.py --target 1.215 --time_budget_s 120 2>&1 | tail -10`. Must `SUCCESS`.
>
> **Soundness obligation**: top-of-file docstring block titled "Lasserre-2 SDP rigor proof" with:
> 1. Why Lasserre-2 is a sound LB on `min_B max_W mu^T A_W mu` (cite Lasserre 2001 or Parrilo 2003).
> 2. Why the MOSEK dual residual cushion gives a rigorous certificate (numerical-tolerance argument).
> 3. Why the cushion size is sufficient for `d <= 64` (residual bound × Neumaier-Shcherbina factor).
>
> **Hard constraints**:
> - Use only MOSEK as the SDP solver (Clarabel is too slow). MOSEK is installed; verified.
> - Do NOT modify `parallel.py`, `bound_anchor.py`, `bound_epigraph.py`, `bound_cctr.py`. (Cascade integration is a separate agent's job.)
> - Keep all four existing public symbols of `lasserre_cert.py` intact (don't rename).
> - Comments terse. Soundness derivation goes in the docstring block, not inline.

#### Agent B: Direct MOSEK Fusion fallback (use only if Agent A fails)

> **Goal**: If vectorized CVXPY still hangs at d=30, bypass CVXPY entirely. Use MOSEK Fusion (`mosek.fusion`) Python API to build the SDP problem with native vectorized primitives.
>
> Same mathematical content, same rigor obligations as Agent A, but skip CVXPY. Build:
> - `M = M.Model()` (Fusion model)
> - `y = M.variable("y", n_monos, Domain.unbounded())`
> - `u = M.variable("u", 1, Domain.unbounded())`
> - For PSD constraints: `M.constraint(Expr.symmetric(...), Domain.inPSDCone(k))` with `k = binom(d+2, 2)` for moment matrix, `k = binom(d+1, 1)` for localizing.
> - Window constraint as a matrix-vector ineq: `M.constraint(Expr.sub(Expr.mul(u, ones), Expr.mul(B_sparse, y)), Domain.greaterThan(0.0))`.
>
> All other obligations (soundness derivation, cushion, tests, regression, smoke) are identical to Agent A.
>
> **API**: still `lasserre_box_cert_int_ge(...)` returning bool, lives in a new file `interval_bnb/lasserre_fusion.py`. Don't modify `lasserre_cert.py` (Agent A owns it).

#### Agent C: Cascade integration (depends on Agent A or B success)

> **Goal**: Integrate the new SDP cert into the per-box BnB cascade in `interval_bnb/parallel.py`.
>
> **When to invoke the SDP**: as the LAST tier before splitting, gated by depth and a fast pre-filter:
> - Depth threshold: `INTERVAL_BNB_SDP_DEPTH` env var, default 999 (disabled). At d=30 set to 30 (after epi LP has had chances at depths 24-30).
> - Pre-filter: only invoke SDP if epi LP value ≥ `target - 0.02` (i.e. epi LP is "close" but didn't certify). This avoids paying SDP cost on boxes the LP says are far from cert. Use the `lp_val` returned by `bound_epigraph_int_ge_with_marginals` (already captured as `epi_ineqlin`-companion in parallel.py).
> - Per-box SDP cost: ~3s. With 16 workers, sustainable at ~5 SDP calls/s aggregate. At d=30 expect ~10K boxes hit SDP tier in 6h, total SDP time ~30 min — affordable.
>
> Add diagnostics: `local_sdp_attempts`, `local_sdp_certs`, surface in the per-worker stats payload at exit. Print in the master "DONE" summary.
>
> Add the env var doc to `INTERVAL_BNB_SDP_DEPTH` near the other thresholds (around line 270 of parallel.py).
>
> Place the SDP cascade tier AFTER the centroid anchor (which currently runs at depth >= INTERVAL_BNB_CENTROID_DEPTH=80). The order should be: ... → epi LP → centroid anchor (cheap-ish, gated) → SDP escalation (expensive, gated, with epi-pre-filter) → split.
>
> **Tests**: add `interval_bnb/test_sdp_cascade.py` with at least:
> - `test_sdp_tier_disabled_by_default` — set `INTERVAL_BNB_SDP_DEPTH=999`, run d=10 t=1.215 BnB; SDP attempts == 0.
> - `test_sdp_tier_enabled_d10` — set `INTERVAL_BNB_SDP_DEPTH=20`, run d=10 t=1.215 60s; SDP attempts > 0; success=True; no regression.
> - `test_sdp_tier_pre_filter` — handcraft a box with `lp_val < target - 0.02`; confirm SDP is NOT invoked (pre-filter saves the cost).
>
> **Regression**: full suite + `run_d10.py --target 1.215 --time_budget_s 180` (slightly longer budget to amortize SDP startup cost).
>
> **Hard constraints**: only modifies `parallel.py` and adds the new test file. Do not touch `lasserre_cert.py` / `lasserre_fusion.py` / other tier files.

#### Agent D: rigorous SDP cushion analysis (parallel-deployable, low risk)

> **Goal**: Verify the SDP cushion is publication-rigorous. Equivalent to the work Agent 3 did for `bound_epigraph.py`'s LP cushion in `PROMPT_2_FINAL_PUSH.md`.
>
> **Tasks**:
> 1. Sample 50 random d=30 boxes at hw=0.025; solve their Lasserre-2 SDP via MOSEK with strict tolerances. Record reported `dual_feasibility_residual` and `complementarity_residual`.
> 2. Compute the worst observed residual; the cushion must be ≥ 100× this.
> 3. Derive the SDP equivalent of the Neumaier-Shcherbina LP bound. For an SDP `min c^T x s.t. A x = b, x ∈ K`, the dual feasibility residual `r_d` and primal residual `r_p` give `|f* - f_true| ≤ ε_p · ||y_max|| + ε_d · ||x_max||`. Bound `||x_max||` (= `||y||_∞` from Lasserre, which is `<= max_alpha ||mu^alpha||_∞ <= 1` since `mu` is on the simplex). So cushion = `1e-7` should suffice for `d <= 64`. Document this in the soundness derivation block of `lasserre_cert.py` (or `lasserre_fusion.py` — coordinate with Agent A/B).
> 4. Generate a small report `sdp_rigor_audit.md` with the residual distribution and the cushion safety factor.
>
> Deliverable: a short markdown file in the repo root + the cushion derivation injected into the appropriate SDP module's docstring (coordinate with whichever of A/B succeeded).

### Step 3 — INTEGRATE (after agents complete; this is the user's manual step)

After Agents A/B/C/D complete:
1. Verify all tests pass on the merged tree (50 + new tests).
2. Run a smoke `d=22 t=1.281` (known stall margin) with `INTERVAL_BNB_SDP_DEPTH=24` for 15 min; confirm SDP attempts > 0 and centroid+SDP combined push us beyond 99.99999%.
3. Run a smoke `d=30 hw=0.025` SDP cert rate test (using `diagnose_sdp.py` rewritten to use the new MOSEK-based cert): cert rate must be ≥ 70%.

### Step 4 — RUN d=30 t=1.281 BnB to completion

Once integration verified:
1. Update `run_d30_local.py` to set `INTERVAL_BNB_SDP_DEPTH=30` (turn on SDP escalation) and `time_budget_s=21600` (6h).
2. Launch `python -u run_d30_local.py > d30_t1281_log.txt 2>&1 &`.
3. Monitor for `success=True` event.

Goal: `success=True, in_flight_final=0, coverage_fraction=1.0` within 6h. The result is `C_{1a} >= 1.281`, the strongest published lower bound for the Sidon autocorrelation constant.

If d=30 still doesn't close (e.g. SDP cert rate is < 50% at hw=0.025), the fallback is `target=1.2805` at `d=22` or `d=24` — known to be feasible from the existing diagnostic data. Document the choice.

---

## 3. Specifications shared by all agents

For EVERY agent you deploy, brief them with these defaults:

- **Read `interval_bnb/THEOREM.md` first** for the problem statement and the proof that `val(d) <= C_{1a}`.
- **Re-derive the soundness of any cut/cert from scratch** in their report and in the code's docstring.
- **Write tests that EXPLICITLY exercise the cut on a surviving-class box** at d=30 hw≈0.025 (the binding case).
- **Verify all existing tests still pass** with: `python -m pytest interval_bnb/test_anchor.py interval_bnb/test_epigraph.py interval_bnb/test_epigraph_extra_cuts.py interval_bnb/test_hd_cut.py interval_bnb/test_lp_split.py interval_bnb/test_cctr.py -q`.
- **Run a fast `d=10 t=1.215` BnB** (`python interval_bnb/run_d10.py --target 1.215 --time_budget_s 120`) to confirm no regression — must report `SUCCESS`.
- **Report concrete diff stats**: lines added/changed, tests added, expected effect on the d=30 cert rate at hw=0.025.
- **Prefer integer arithmetic at denom `2^60`** for any rigor cert, matching the existing `bound_anchor_int_ge` / `bound_epigraph_int_ge` pattern. Where SDP residuals are float, use a documented cushion ≥ 100× worst observed residual.
- **Do NOT skip soundness for empirical pass.** If a fix passes tests but the soundness derivation has a gap, mark the gap explicitly and STOP — do not declare success.

---

## 4. Important context the agents need

### 4.1 The problem we're solving

For nonneg `f : R → R_{>=0}` supported on `[-1/4, 1/4]` with `int f = 1`:

`max_{|t| <= 1/2} (f * f)(t) >= C_{1a}`

The Cloninger-Steinerberger 2017 lower bound is `1.2802`. This codebase aims to push it higher via interval branch-and-bound on `val(d)`, where:

`val(d) := min_{mu in Δ_d} max_W mu^T M_W mu`

and `val(d) <= C_{1a}` is proved rigorously in `interval_bnb/THEOREM.md`. So a rigorous cert that `val(d) >= 1.281` immediately gives `C_{1a} >= 1.281`.

### 4.2 Why d=30 specifically

`val(d)` is monotone non-decreasing in `d`; we have:
- `val(22) ≈ 1.30933` (KKT, residual 1e-14)
- `val(24) ≈ 1.31369` (KKT, residual 1e-9)
- `val(30) ≈ 1.33203` (KKT, residual 1e-9)

The margin `val(d) - 1.281` grows from `0.028` at d=22 to `0.051` at d=30. Bigger margin should make BnB easier — *but* the cascade gets harder per box at higher d because the cheap tier collapses (verified in `diagnose_full.py`). SDP escalation is the principled fix.

### 4.3 Existing infrastructure

- KKT solver: `kkt_correct_mu_star.py`, `find_mu_star_d{22,24,30}.py`. mu_star files saved as `mu_star_d{22,24,30}.npz`.
- Cascade: `interval_bnb/parallel.py` (cascade dispatcher), `interval_bnb/bnb.py` (rigor replay), `interval_bnb/bound_*.py` (per-tier bounds).
- LP rigor: `interval_bnb/bound_epigraph.py` already has the publication-rigor cushion fix (Neumaier-Shcherbina-style, 1e-7 cushion). This is the model to follow for the SDP rigor cushion.
- Existing Lasserre code: `interval_bnb/lasserre_cert.py` (for per-box use, currently broken at d=30) and `lasserre/d64_farkas_cert.py` (for d=64 standalone certs, vectorized differently — study this).

### 4.4 Diagnostic logs to study

- `diagnose_d30_output.log` — cert rates by box width near sigma(mu*)
- `diagnose_full_output.log` — cert rates on random H_d boxes (the BnB-realistic distribution)
- `diagnose_cheap_output.log` — cheap-tier cert rates (showing the d=30 collapse)
- `d22_t1281_log.txt`, `d24_t1281_log.txt`, `d30_t1281_log.txt` — actual BnB runs at each d

Also check `kkt_d30.log` for the KKT solve's stats — useful for understanding mu_star_d30 structure.

---

## 5. Estimated timeline

| Phase | Agent | Hours |
|---|---|---|
| Step 1 audit | 1 agent (foreground) | 0.5 |
| Step 2 implementation | 2-4 agents in parallel | 4-6 |
| Step 3 integrate + smoke | manual + 1 agent | 1.5 |
| Step 4 d=30 BnB run | autonomous | 4-6 |
| **Total wall clock** | | **10-14 hours** |

Realistic for a one-day engineering effort.

---

## 6. Failure modes and fallbacks

If, after Step 3, the smoke test shows SDP cert rate at d=30 hw=0.025 is < 50%:

- **Possibility 1**: SDP solver is hitting its own gap (Lasserre-2 alone isn't enough for these boxes). Try Lasserre order 3 (much more expensive) or escalate target to 1.2805.
- **Possibility 2**: MOSEK solver tolerances aren't strict enough. Tighten and rerun.
- **Possibility 3**: The cushion derivation has a bug and we're missing certs. Audit Agent D's derivation.

Document the failure mode and fall back to `target=1.2805 at d=22` or `d=24` (known feasible from `d22_t1281_result.json` and the diagnostic data).
