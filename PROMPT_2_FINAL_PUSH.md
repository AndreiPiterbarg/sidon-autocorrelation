# Prompt 2 — Deploy agents to find the FINAL improvements that close the last 4.5e-13

**Why this is necessary.**

We are 99.99999849816% of the way to a rigorous proof of `C_{1a} >= 1.281`
via interval branch-and-bound at `d = 22`. That is **5 orders of magnitude
tighter** than the prior d=20 pod stall (which sat at 99.80189% indefinitely).
But "almost rigorous" is not rigorous — for the publication this MUST go to
**100% certified coverage** (uncovered volume = 0 modulo the dyadic-grid
floor). The remaining `4.5e-13` of state-space volume is held by ~1500
in-flight boxes at depth up to 114 that the current cascade cannot certify
within the time budget. Without closing these, we have no theorem.

**What WORKED — study these carefully before proposing anything new.**

A 16-core local d=22 BnB run with the following four fixes drove coverage
from the d=20 pod's hard 99.80189% stall to 99.99999849816% in 3 hours:

1. **Epigraph LP extra cuts** in `interval_bnb/bound_epigraph.py`:
   - column-sum RLT (`Sigma_i Y_{ij} = mu_j`) — mirror of pre-existing
     row-sum RLT
   - Y-symmetry (`Y_{ij} = Y_{ji}`)
   - diagonal SOS (`Sigma_i Y_{ii} >= 1/d`) — Cauchy-Schwarz on simplex
   - midpoint diagonal tangent (`Y_{ii} >= 2 m_i mu_i - m_i^2`,
     `m_i = (lo_i+hi_i)/2`) — strictly tighter than the existing SW/NE
     McCormick faces on `Y_{ii}`

2. **H_d half-simplex pre-filter** in `interval_bnb/symmetry.py`,
   `bnb.py`, `parallel.py`: drop any box with `lo_int[0] > hi_int[d-1]`
   (sound by sigma-symmetry of windows, proved in `THEOREM.md`). Roughly
   halves the search tree.

3. **mu\*-anchor cut** in `interval_bnb/bound_anchor.py`:
   ```
   f(mu) >= f(mu*) + g.(mu - mu*) + min(0, scale_W* * lambda_min(A_W*)) * D2(B)
   ```
   where `g` is the subgradient at the binding window `W*` and `D2(B)`
   is the maximum squared `(mu - mu*)` distance over the box. Note: a
   prior agent caught that `A_W` is NOT PSD for low-ell windows (eigenvalues
   to `-1` at `ell=2`), so the naive supporting-hyperplane bound is
   unsound — the curvature concession term restores soundness. **At the
   final d=22 run this fix activated 251763 attempts but produced 0
   certs** — i.e. it never fired successfully. Find out why.

4. **LP-binding-axis split heuristic** in `interval_bnb/parallel.py`:
   uses the just-solved epigraph LP's `ineqlin` marginals (SW/NE/NW/SE
   McCormick face duals) to pick the split axis whose McCormick face is
   most binding: `axis_score[i] = sum_j |marginals across faces touching
   pair (i,j)| * width[i]`.

**The exact end-state to study.**
- Final log: `d22_t1281_log.txt` (3.0h, ended at coverage_fraction
  `0.9999999849816406`)
- Final result: `d22_t1281_result.json` — `closed_volume = 3.0279159091e-05`,
  `total_volume = 3.0279159546e-05`, `in_flight_final = 1536`,
  `max_depth = 114`, `epi_attempts = 251763`, `epi_certs = 123046`
  (49% close rate at the LP tier), `anchor_attempts = 251763`,
  `anchor_certs = 0` (NEVER fired — diagnose this).
- Per-box max depth 114 with `D_SHIFT = 60` budget per axis: the survivor
  boxes are extremely deep. Some axes are likely at saturation, which
  blocks further splitting via the `splittable` mask.
- `mu_star_d22.npz` available; `f(mu*) = 1.30933556`, residual `3.4e-14`,
  29 active windows. Margin to target = `0.028` — the LP gap at the
  surviving boxes therefore exceeds `0.028`, which is large compared
  to typical McCormick gap at depth 114. Something is structurally
  wrong with the bound on those boxes.

**The audit + agent-deployment task.**

Step 1 — **AUDIT the failure mode of the surviving 1536 in-flight boxes.**

Look at `d22_t1281_log.txt` and `d22_t1281_result.json`. Read every
relevant module: `interval_bnb/parallel.py`, `bound_epigraph.py`,
`bound_anchor.py`, `bound_cctr.py`, `bound_eval.py`, `box.py`,
`symmetry.py`, `windows.py`. Then answer concretely:

A. Why does the anchor cut have `0 certs / 251763 attempts`? Is it
   a code bug, a math gap, or genuinely loose? Reproduce one
   surviving box (you can dump from a fresh short run) and trace
   `bound_anchor_int_ge` step by step.

B. Why does the epigraph LP close only 49% of boxes (`123046/251763`)?
   At depth 114 the box is essentially a point, so McCormick should
   be very tight. Is the LP actually capturing the new RLT/SOS/tangent
   cuts, or is one of them silently degenerate at deep depth (e.g.
   `Y_{ii} >= 2 m_i mu_i - m_i^2` becomes vacuous when `m_i ~ lo_i`)?

C. What is the structural form of the surviving boxes? Are they:
   - Clustered around `mu*` in one region (bad — anchor cut should
     be working there).
   - Spread along the `f = const` level set near `mu*` (worse — needs
     a new bound that respects the level set).
   - Trapped against the simplex boundary (different problem —
     boundary-anchored cuts).
   - Suffering from the dyadic saturation of the splittable-axis
     mask?

For each surviving box class, propose the SOUND, RIGOROUSLY VALID
bound or cut that would close it.

Step 2 — **DEPLOY AGENTS in parallel for each proposed final fix.**

For each fix you identify in Step 1, deploy one independent agent
with a self-contained prompt that:
- Explains the soundness obligation (mathematical proof of validity).
- Specifies the exact files to touch and the cascade tier where the
  new cut should fire.
- Specifies the test files to add and the existing test suites that
  must continue to pass.
- Specifies the integer-arithmetic / safety-cushion expectations
  (no float-only certs at publication tier; either exact integer
  arithmetic at denom `2^60`, or an explicit cushion larger than
  HiGHS's `1e-7` dual-feasibility tolerance).

Candidate fixes to consider (you must independently verify each is
both mathematically sound AND useful for the surviving box class
before deploying):

- **Lasserre-O2 SDP escalation** — at depth `>= 100` with LP-cert
  failure, solve a small (truncated) Lasserre-2 SDP locally; close
  the box if SDP value `>= target + cushion`. Use existing
  `interval_bnb/lasserre_cert.py` or `lasserre/d64_farkas_cert.py`
  as plumbing. Soundness via Farkas dual cert.

- **Multi-anchor (multi-mu\*) cut** — there may be multiple local
  minimizers of `f` near `mu*` (e.g. permutations under `sigma`,
  near-degenerate active sets); a single anchor at one of them
  leaves the others' neighborhood loose. Compute the top-K KKT
  points and apply each as an independent supporting hyperplane.

- **Active-set-aware cut at deep boxes** — at the surviving boxes,
  identify which subset of `mu*'s` 29 active windows is binding
  inside `B`, and replace the LP's window-aggregation with a tighter
  per-binding-window cut.

- **Boundary anchor cuts** — for boxes touching `{mu_i = lo_i}` or
  `{mu_i = hi_i}` faces, the McCormick lift is rank-deficient there;
  add the face-projected cut.

- **Tighter HiGHS tolerances + Neumaier-Shcherbina integer dual cert**
  in `bound_epigraph.py`: replace the `safety_only=True` cushion of
  `n_vars * 1e-14` with either (a) `1e-7` HiGHS-aware cushion, or
  (b) full integer dual cert. The current `1e-14` cushion is the
  audit's first BLOCKER — the LP could return a value within
  `1e-7` of the truth and the cert would be wrong.

- **Increased `D_SHIFT`** beyond 60 if dyadic-grid saturation is
  what's truly blocking the deep tail. Costs O(n) more bigint ops
  per split but removes the saturation floor.

For EACH agent you deploy, brief it to:
- Read `THEOREM.md` first for the problem statement.
- Re-derive the soundness of the cut from scratch in its report.
- Write tests that EXPLICITLY exercise the cut on a surviving-class box.
- Verify all 40 existing tests still pass:
  ```
  python -m pytest interval_bnb/test_epigraph.py interval_bnb/
    test_epigraph_extra_cuts.py interval_bnb/test_hd_cut.py interval_bnb/
    test_anchor.py interval_bnb/test_lp_split.py interval_bnb/test_cctr.py -q
  ```
- Run a fast d=10 t=1.215 BnB to confirm no regression.
- Report concrete diff stats: lines changed, tests added, expected
  effect on the surviving 4.5e-13 tail.

Step 3 — **After all agents complete**, re-run d=22 t=1.281 locally
on 16 cores with the combined fixes and a 6-hour budget. Goal:
**coverage_fraction = 1.0 exactly** (modulo dyadic floor) and
`success = True`. If any single fix is the bottleneck, identify it
and propose either a follow-up fix or a target relaxation as a
graceful fallback (`target = 1.2805` would still beat CS 2017's
1.2802).

**Working dir:** `C:/Users/andre/OneDrive - PennO365/Desktop/compact_sidon`.
**THIS IS PUBLISHABLE RESEARCH — every fix must carry a written
proof of soundness, not just empirical pass.**
