# `coarse_cascade_prover.py` — Full Session Summary

> Goal: rigorously prove $C_{1a} \ge c$ for the largest $c$ achievable, with **absolute correctness** (no claims without exhaustive certification).
>
> Standing CS17 bound: $C_{1a} \ge 1.2802$.
>
> Prior session best on this hardware: $C_{1a} \ge 1.15$ (but the original "proof" sampled 2000 random cells — **NOT a rigorous proof**).
>
> **Final achieved this session: $C_{1a} \ge 1.18$ rigorously proven** (d=8, S=76, 19 min, 1.46 B cells, exhaustive enumeration, 100% certified).

---

## §1 What was wrong with the original prover

The pre-session `coarse_cascade_prover.py` had **four bugs/weaknesses**:

| # | Issue | File:line | Severity |
|---|---|---|---|
| 1 | `run_box_certification` SAMPLED 2000 random cells (not exhaustive) — printed "PROOF" without certifying every cell | `:705-751` | **Soundness-fatal**: claim was mathematically false |
| 2 | `_box_certify_cell` used water-filling = ONE feasible point per window, not the cell minimum | `:631-702` | Tightness loss |
| 3 | "Subtree pruning" at `:417-456` was a fail-fast on partial state, not a true subtree LB | `:417-456` | Performance loss |
| 4 | When $c \cdot \ell \cdot S^2/(2d) \in \mathbb{Z}$, threshold lattice gives margin = 0 exactly (cells "cannot fix with S") | `:42-57` | Structural barrier |

---

## §2 Fixes implemented (with tests after each)

### §2.1 Phase 1 Lipschitz LB (Tier 1 cheap filter)

`_phase1_lipschitz_lb` (cheap, sound). For each window $W=(\ell, s)$:
$$\mathrm{LB}_W \;=\; TV_W(\mu^*) - L_1 \cdot U_1 - \mathrm{quad\_drop}$$

where
- $L_1 = (\max g - \min g)/2$ (centered grad ∞-norm — uses $\sum \delta = 0$)
- $U_1 \le 2h \cdot \lfloor d/2 \rfloor$ (||δ||_1 bound)
- $\mathrm{quad\_drop} \le \min(\mathrm{scale} \cdot U_1^2, \mathrm{scale} \cdot U_1 \cdot h \cdot (\ell-1), \mathrm{scale} \cdot U_2^2 \cdot (\ell-1))$

Plus L2 Cauchy-Schwarz alternative for the linear drop (added later):
$$|grad \cdot \delta| \le \|grad - \mathrm{mean}\|_2 \cdot \|\delta\|_2 \le \|grad - \mathrm{mean}\|_2 \cdot h\sqrt{d}$$
Take min(L1·U1, L2·U2) — both sound, take tightest.

**Empirical capture rate: 99.9-100%** at the binding c range (c=1.10-1.20).

**Soundness verified**: 1600-cell random test — 0 violations of "Tier 1 cert ⇒ vertex cert".

### §2.2 Adaptive 2-tier dispatch

`_box_certify_batch_adaptive` (parallel via `prange`):
1. **Tier 1**: `_phase1_lipschitz_lb` — captures 99.9%+ of cells
2. **Tier 3**: `_box_certify_cell_vertex` (exact vertex enum) — only on cells where Tier 1 fails

(Tier 2 / water-fill / McCormick LP not needed; jump from Tier 1 to Tier 3 sufficient.)

### §2.3 Soundness fix: exhaustive `run_box_certification`

Replaced random sampling with `generate_canonical_compositions_batched` iteration. Z2 dedup applied. `x_cap` filter removes trivially-won cells (any bin > $x_{\mathrm{cap}}$).

### §2.4 Subtree LB via `min_contrib`

Adapted from `cpu/run_cascade.py:2024-2110`. At each cursor descent point, compute lower bound on `conv` over all cursor completions; if any window's prefix sum already exceeds threshold, prune subtree.

**Test**: `_subtree_prune_keeps_all_survivors_d2` — bit-exact match against brute-force enumeration.

### §2.5 Parallel cascade kernels

Added `_cascade_level_count_parallel` and `_cascade_level_fill_parallel` using `numba.prange`. Replaces sequential `for p_idx in range(n_parents)` loops in `run_cascade_level`.

**Test**: `_parallel_cascade_matches_serial` — bit-exact match.

### §2.6 S-shift to dodge integer-threshold lattice

`s_shift_safe` minimizes lattice hits without raising for rational c. Empirically: S=50 → S=51 typically; for c=1.20 (=6/5) some lattice hits unavoidable.

### §2.7 fail-fast threading

`run_cascade(..., fail_fast=True)` propagates to `run_box_certification`; aborts at first failed cell. Useful since proof is invalid regardless of count.

### §2.8 CLI flags + progress reporting

- `--n_threads N` sets Numba thread count
- `--no_fail_fast` for full diagnostic
- Periodic progress: `[progress @ Ns] processed M cells, failed: 0, worst TV: V, Tier 1: K (Y%)`

---

## §3 Speedup measurements (laptop, 16 threads)

| Config | Pre-session | Post-fix | Speedup |
|---|---|---|---|
| c=1.10, d=6, S=50 | 33.6 s | 3.05 s | 11× |
| c=1.14, d=6, S=50 | 12.6 s | 3.01 s | 4.2× |
| c=1.15, d=6, S=75 | 192 s | 4.6 s | **41×** |
| c=1.16, d=6, S=75 | 295 s (FAILED) | 4.8 s (FAILED, same) | 61× |
| **c=1.16, d=6, S=100** | **infeasible** | **18.9 s — RIGOROUS PROOF** | n/a |

After L2 linear bound:
- Tier 3 dispatches at c=1.16 dropped 438 → **18** (24× reduction)

---

## §4 Test suite (16 tests, all passing)

| Test | Validates |
|---|---|
| `test_s_shift_finds_off_lattice_when_possible` | §2.6 |
| `test_s_shift_minimizes_for_unavoidable_lattice` | §2.6 |
| `test_count_lattice_offenders_at_known_point` | §2.6 |
| `test_vertex_drop_matches_brute_force_d4` | §2.2 vertex enum exact |
| `test_box_certify_cell_vertex_passes_known_winner` | §2.2 |
| `test_box_certify_cell_vertex_below_target` | §2.2 |
| `test_mccormick_lower_bounds_vertex` | McCormick is sound LB |
| `test_subtree_prune_keeps_all_survivors_d2` | §2.4 — sound |
| `test_subtree_prune_reduces_visited_leaves` | §2.4 — performance |
| `test_run_box_certification_passes_easy_config` | §2.3 — soundness end-to-end |
| `test_run_box_certification_fails_when_target_too_high` | §2.3 — correctly fails |
| `test_full_cascade_proves_c105` | Integration |
| `test_phase1_implies_vertex_cert_random` | §2.1 — 1600 random cells, 0 violations |
| `test_adaptive_batch_matches_serial` | §2.2 — adaptive vs serial |
| `test_phase1_high_cert_rate_at_easy_c` | §2.1 — 99% capture |
| `test_parallel_cascade_matches_serial` | §2.5 — parallel = serial |

---

## §5 Rigorous proofs achieved

All values **exhaustively certified**, every canonical cell verified:

| $c$ | $d$ | $S$ | Time | Cells certified | Tier 1 % | Tier 3 cells |
|---|---|---|---|---|---|---|
| 1.05 | 4 | 31 | 2.3 s | 1,360 | 100% | 0 |
| 1.10 | 6 | 51 | 3.1 s | 1.08 M | 100% | 0 |
| 1.14 | 6 | 51 | 12.6 s | 1.20 M | 100% | 0 |
| 1.15 | 6 | 76 | 4.6 s | 8.22 M | 100% | 92 |
| **1.16** | 6 | 101 | 18.9 s | 32.86 M | 100% | 18 (with L2 bound) |
| **1.17** | 8 | 51 | 68 s | 100.30 M | 100% | 268 |
| **1.18** | 8 | 76 | 1156 s (~19 min) | **1.46 B** | 100% | 145 |

All checks: `[BOX-CERT PASS] every cell certified — proof valid.`

---

## §6 Failed attempts (transparency — no false claims)

### 6.1 c=1.17 at d=6

Cell `[47, 17, 19, 25, 29, 64]/201` has cell-min TV that ASYMPTOTES to ~1.169 as S grows. Below 1.17 at any S.

| $S$ | worst TV at d=6, c=1.17 |
|---|---|
| 101 | 1.166944 (5 fail) |
| 151 | 1.169050 (1 fail) |
| 201 | 1.169303 (1 fail) |

**Conclusion**: $\mathrm{val}(6) \approx 1.171$, so d=6 cannot prove c=1.17. Required cascade to d=8.

### 6.2 c=1.19 at d=8 S=101

102 minutes, 9.4 billion cells, **2 cells failed** with worst TV = 1.189099. Need S ≥ 151 or cascade to d=16.

### 6.3 c=1.20 — pod run analysis

**Setup**: ubuntu pod (64 cores, 251 GB RAM), c=1.20 d_start=6 S=101 max_levels=2.

**Result on pod**:
- L0 (d=6): 25.9 M tested, 4,670 survivors, **1.86 s**
- L1 (d=12): 4,670 parents → **0 survivors** in 12 min (cascade converged)
- Box-cert at d=12: **3.89 BILLION cells processed in 32 minutes**, 0 failures, worst TV 1.200471

**Why we KILLED before completion**:

The total canonical cell count at d=12, S=101, x_cap=31:
$$N = \binom{112}{11} - 12\binom{80}{11} + 66\binom{48}{11} - 220\binom{16}{11} \approx 4.005 \times 10^{14}$$

Z2 dedup → **200 trillion canonical cells**.

We had processed 0.0017% in 32 minutes. ETA at 2M cells/s: **3.2 years**.

This is computationally intractable on any current hardware. Empirical evidence (3.89B cells, 0 fail) is extremely strong but **not a rigorous proof**.

### 6.4 Pod terminated mid-experiment

After killing the c=1.20 run and launching c=1.17 S=150 test, the SSH connection dropped and the pod became unreachable. The pod was a spot instance — provider preemption is the most likely cause.

---

## §7 The structural ceiling — why c ≥ 1.19 is hard

The cascade prover's box-certification at dimension $d$ has cell-min TV bounded above by $\mathrm{val}(d)$ (the simplex infimum):

| $d$ | $\mathrm{val}(d)$ | Max provable c (margin > Lipschitz drop) |
|---|---|---|
| 4 | 1.102 | ~1.10 |
| 6 | 1.171 | ~1.17 (we got 1.16) |
| 8 | 1.205 | ~1.18-1.19 (we got 1.18) |
| 10 | 1.241 | ~1.22-1.23 |
| 12 | 1.271 | ~1.25 (200T cells at S=101) |
| 14 | 1.284 | barely above 1.28 |
| 16 | 1.319 | comfortable margin to 1.28 |

For c=1.20 we need $d \ge 10$ ideally. Cell count at d=10, S=51: ~7B (~1-2h locally if Tier 1 captures 100%).

For c=1.28+ we need d ≥ 16. Cell count there is enormous.

---

## §8 Code state (production-ready improvements)

### Files modified
- `coarse_cascade_prover.py` — all fixes integrated
- `tests/test_coarse_cascade_prover_fixes.py` — 16 tests covering soundness + performance

### Documents written
- `COARSE_CASCADE_PROVER_FIXES.md` — original 4-fix plan
- `COARSE_CASCADE_PROVER_AUDIT_2.md` — 12-agent audit on what's blocking 1.28+
- `COARSE_CASCADE_PROVER_SESSION_SUMMARY.md` — this file

### Files NOT modified
- All other project files preserved

### Pod deployment
- `~/sidon/coarse_cascade_prover.py` (deployed but pod terminated)
- `~/sidon/cloninger-steinerberger/compositions.py` (deployed)
- `~/sidon/c120_run.log` (last log, lost when pod terminated)

---

## §9 Open paths to push beyond 1.18

In rough order of feasibility:

| # | Path | Estimated effort | Plausible new bound |
|---|---|---|---|
| 1 | c=1.19 d=8 S=151 locally | ~5-6 hours | 1.19 likely |
| 2 | c=1.20 d=10 S=51 locally | ~1-2 hours | 1.20 plausible |
| 3 | c=1.21 d=10 S=76 locally | ~3-5 hours | 1.21 plausible |
| 4 | New pod at c=1.20+ at d=10/12 small S | hours-days | 1.20-1.25 |
| 5 | Tighter Phase 1 LB (window-specific bounds) | 1-2 days code | +0.01-0.02 across the board |
| 6 | GPU CUDA box-cert kernel | 1 week code | unblocks d=12-16 |
| 7 | Multi-window combined bound | research | +0.005-0.02 |
| 8 | Adaptive cell decomposition | research | unclear |

Routes that are **dead** (per earlier audits):
- Smooth-g Theorem 1 (zero gain at d=8)
- Symmetric reduction (Jensen-on-Φ argument was wrong)
- Lasserre val(d=8) bypass (val(8)=1.205 ceiling)
- Fractional overlap in Theorem 1 (not a valid LB)

---

## §10 Bottom line

**Rigorous lower bound improvement this session: $C_{1a} \ge 1.15$ (with unsound sampling) → $C_{1a} \ge 1.18$ (exhaustively certified).**

The prover is now:
- **Sound** (no false claims; sampling replaced with exhaustive enumeration)
- **Tight** (Phase 1 Lipschitz LB captures 99.9%+; vertex enum exact for the rest)
- **Fast** (parallel cascade + Phase 1 + L2 bound: 41-61× faster than baseline at c=1.15-1.16)
- **Correct** (16 tests passing; bit-exact match against brute-force on subtree pruning, vertex enum, parallel cascade)

The CS17 standing bound (1.2802) requires d ≥ 14, which has cell counts that no current implementation can handle without either GPU acceleration or fundamentally different algorithms (Lasserre SDP at d ≥ 14, etc.). Within the cascade-prover paradigm on this hardware, **1.18-1.20 is the achievable rigorous bound range**.
