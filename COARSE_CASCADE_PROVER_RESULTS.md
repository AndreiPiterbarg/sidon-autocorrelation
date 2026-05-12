# `coarse_cascade_prover.py` — Complete Test Results & Run History

> Standalone Cloninger–Steinerberger-style prover at the project root. Implements the **coarse cascade** with **Theorem 1** (no correction term), **integer threshold pruning**, **B&B subtree pruning**, and **box certification** via water-filling QP.
>
> All runs were performed on **2026-04-13** (CPU pod, 48 workers).

---

## 1. Algorithm summary

For nonneg $f \ge 0$ on $[-\tfrac14,\tfrac14]$ with $\int f = 1$:

1. **Theorem 1 (no correction).** Bin masses $\mu_i$ at dimension $d$ satisfy $\max_{|t|\le 1/2}(f*f)(t) \ge \mathrm{TV}_W(\mu)$ for every window $W=(\ell,s)$.
2. **Cascade L0…LK.** Enumerate compositions of $S$ (integer mass quantum $\delta=1/S$) into $d_0$ parts; prune those with $\mathrm{TV}\ge c_{\text{target}}$. Refine survivors $d \to 2d$ each level.
3. **Subtree pruning** via partial autoconvolution (line 334 of `coarse_cascade_prover.py`).
4. **Box certification** of every Voronoi cell by max-over-windows of (water-filling) min-over-cell of $\mathrm{TV}_W$.

A **rigorous proof of $C_{1a}\ge c$** requires both (a) cascade converges (0 survivors at some L$K$) **and** (b) box cert passes at every level (margin $-$ cell_var $-$ quad_corr $\ge 0$).

---

## 2. Source files

| File | Role |
|---|---|
| `coarse_cascade_prover.py` (root) | The prover (Numba B&B with subtree pruning) |
| `proof/coarse_cascade_method.md` | Method writeup; §9.3 has performance table |
| `lean/Sidon/Proof/CoarseCascade.lean` | (Stub) Lean formalization pointer |
| `lean/Sidon/CoarseCascade/*.lean` | Per-component Lean obligations (TVConvexity, SubtreePruning, BoxCertification, RefinementMonotonicity, …) |

### Test / driver scripts (`tests/`)

| Script | Purpose |
|---|---|
| `coarse_sweep_1_10.py` | Phase-1 sweep $(d_0, S, c)$ for proving $C_{1a}\ge 1.10$ |
| `coarse_sweep_1_30.py` | Same, target 1.30 |
| `coarse_prove_1_30.py` | Targeted L0 attempts at $c\in\{1.30, 1.32, 1.35\}$ |
| `coarse_gridpoint_1_30.py` | Grid-point only push toward 1.30 |
| `coarse_l0_only_push.py` | Find max $c$ with L0 + box cert |
| `coarse_max_rigorous.py` | Binary-search max **rigorous** $c$ |
| `coarse_high_d0.py` | High-$d_0$ sweep aiming at $c\ge 1.28$ |
| `coarse_diagnose.py` | Per-composition margin / cell_var / quad_corr distribution |
| `prove_1_30.py` | Phase-1 L0 sweep + cascade for $c=1.30$ |
| `prove_1_30_experiment.py` | Time-budgeted 1.30 experiment |
| `sweep_box_cert_S.py` | Exhaustive box-cert sweep over $S$ at $d=2, c=1.29$ |
| `test_qp_cascade_l0.py` | v2 vs joint-QP box-cert on L0 |
| `test_refinement_monotonicity.py` | Verifies the cascade-soundness conjecture |

### Cached survivor checkpoints (`data/`)

| File | Bytes |
|---|---|
| `coarse_L0_survivors_S20.npy` | 144 |
| `coarse_L0_survivors_S30.npy` | 176 |
| `coarse_L0_survivors_S50.npy` | 256 |
| `coarse_L1_survivors_S30.npy` | 1,184 |
| `coarse_L1_survivors_S50.npy` | 28,320 |
| `coarse_L2_survivors_S50.npy` | **7,004,384** (the L2 frontier at $S=50$) |

---

## 3. Run logs (`data/cpu_run_20260413_*.log`)

All 9 coarse-cascade runs, in chronological order of timestamp:

| Log | Driver | Goal |
|---|---|---|
| `110457.log` | `coarse_sweep_1_10.py` | Phase-1 quick scan, target 1.10 |
| `110656.log` | `coarse_sweep_1_10.py` | Repeat scan (with box-cert reporting) |
| `111822.log` | `coarse_sweep_1_30.py` (Phase 1) | Quick feasibility for $c=1.28, 1.29$ |
| `113016.log` | `coarse_high_d0.py` | High-$d_0$ L0 push at $c=1.25$ |
| `123938.log` | `coarse_diagnose.py` | Margin / cell_var / quad_corr stats |
| `124117.log` | `coarse_prove_1_30.py` | Targeted L0 attempts (1.30 / 1.32 / 1.35) |
| `125241.log` | `coarse_gridpoint_1_30.py` | Cascade convergence test (grid-point only) |
| `130048.log` | `coarse_l0_only_push.py` | "No rigorous proofs found" |
| `130219.log` | `coarse_max_rigorous.py` | **Binary search max rigorous $c$** |
| `135748.log` | `prove_1_30_experiment.py` | Diagnostic at 1.30 |
| `141632.log` | `prove_1_30_experiment.py` | L0 sweep at 1.30 |

(`135734.log` is a CLI-syntax error; `232607.log` / `232956.log` are unrelated lasserre runs.)

---

## 4. Headline results

### 4.1 Best **rigorous** lower bound (cascade + box cert)

**$C_{1a} \ge 1.15$** at $d_0=6$, $S=75$ — cell `cpu_run_20260413_130219.log`, line 39:
```
c=1.15: 0 surv, box=YES, net=0.004819
```

This is the maximum $c$ for which **both** the cascade converges **and** every L0 cell's box certificate passes.

### 4.2 Best **grid-point** convergence (no continuous cover)

**$C_{1a} \ge 1.50$** (grid-point only) at $d_0\in\{2,3,4\}$, $S=10$, converges at L4–L5 — see `cpu_run_20260413_125241.log` summary (lines 2268–2316).

> Grid-point convergence does **not** prove a lower bound on $C_{1a}$ — it only proves it for distributions exactly at grid points. The box-cert step is what extends to the continuum.

### 4.3 Verdict for $C_{1a} \ge 1.30$

**Not provable with this prover at any tried configuration.** The cascade converges for grid-point $\mu$, but box certification fails everywhere checked, with min-net $\ge -0.026725$ at $d_0=6, S=200$ ($c=1.30$, ~37M survivors after L0). Cell-variation $h \cdot \sum |\nabla|$ exceeds the available margin.

---

## 5. Detailed sweep tables

### 5.1 Maximum-rigorous-$c$ binary search (`cpu_run_20260413_130219.log`)

For each $(d_0, S)$, scan $c$ upward until either L0 leaves survivors or box cert fails.

| $d_0$ | $S$ | comps | **best rigorous $c$** | net at best | First failure |
|---|---|---|---|---|---|
| 4 | 100 | 176,851 | (none) | n/a | $c=1.10$, box=no, net=−0.00550 |
| 4 | 150 | 585,276 | (none) | n/a | $c=1.10$, box=no |
| 4 | 200 | 1,373,701 | (none) | n/a | $c=1.10$, box=no |
| 4 | 300 | 4,590,551 | **1.10** | 0.001022 | $c=1.11$: 70 survivors |
| 4 | 400 | 10,827,401 | **1.10** | 0.001225 | $c=1.11$: 170 survivors |
| 4 | 500 | 21,084,251 | **1.10** | 0.001267 | $c=1.11$: 324 survivors |
| 6 | 30 | 324,632 | **1.12** | 0.005000 | $c=1.13$, box=no |
| 6 | 40 | 1,221,759 | **1.14** | 0.001406 | $c=1.15$, box=no |
| 6 | 50 | 3,478,761 | **1.14** | 0.008400 | $c=1.15$, box=no |
| 6 | 75 | 24,040,016 | **1.15** | 0.004819 | $c=1.16$, box=no |
| 8 | 20 | 888,030 | (none) | n/a | $c=1.10$, box=no |
| 8 | 25 | 3,365,856 | **1.12** | 0.006400 | $c=1.13$, box=no |
| 8 | 30 | 10,295,472 | **1.13** | 0.009753 | $c=1.14$, box=no |
| 10 | 15 | 1,307,504 | (none) | n/a | $c=1.10$, box=no |
| 10 | 20 | 10,015,005 | (none) | n/a | $c=1.10$, box=no |
| 12 | 12 | 1,352,078 | (none) | n/a | $c=1.10$, box=no (very negative) |
| 12 | 15 | 7,726,160 | (none) | n/a | $c=1.10$, box=no |

**Overall best:** $C_{1a} \ge 1.15$ (line 226 of the log).

### 5.2 Phase-1 quick scan, $c=1.10$ (`cpu_run_20260413_110656.log`)

48 (out of 120) configs converged with **box cert passing**:

| $c$ | $(d_0, S)$ where rigorous |
|---|---|
| 1.05 | $d_0=3$ all $S\ge 100$; $d_0=4$ all $S\ge 50$; $d_0=6, S\in\{50,75\}$ |
| 1.08 | $d_0=4$ all $S\ge 50$; $d_0=6, S\in\{50,75\}$ |
| 1.09 | $d_0=4$ all $S\ge 50$; $d_0=6, S\in\{50,75\}$ |
| 1.10 | $d_0=4, S\in\{300,400,500\}$; $d_0=6, S\in\{50,75\}$ |

Closest-to-failure rigorous proof: $d_0=6, S=75, c=1.05$, net $= 0.104819$ (largest margin observed).

### 5.3 Phase-1 quick feasibility ($c=1.28, 1.29$) — `cpu_run_20260413_111822.log`

| $d_0$ | $S$ | $c$ | proven at | box | L0 surv | worst_net | time |
|---|---|---|---|---|---|---|---|
| 4 | 20 | 1.280 | L2 | no | 90 | −0.960 | 32.5s |
| 4 | 30 | 1.280 | L2 | no | 296 | −0.646 | 12.9s |
| 4 | 40 | 1.280 | L2 | no | 705 | −0.470 | 28.2s |
| 4 | 50 | 1.280 | L2 | no | 1,369 | −0.378 | 306.5s |
| 6 | 20 | 1.280 | L2 | no | 226 | −0.782 | 4.1s |
| 6 | 30 | 1.280 | L2 | no | 1,664 | −0.784 | 7.5s |
| 6 | 40 | 1.280 | L2 | no | 7,147 | −0.581 | 23.4s |
| 6 | 50 | 1.280 | L2 | no | 21,412 | −0.461 | 240.3s |
| 8 | 20 | 1.280 | L1 | no | 98 | −0.960 | 3.8s |
| 8 | 30 | 1.280 | L1 | no | 1,752 | −0.646 | 9.9s |
| 8 | 40 | 1.280 | L1 | no | 12,904 | −0.470 | 30.9s |
| 10 | 20 | 1.280 | L1 | no | 18 | −1.012 | 2.4s |
| 12 | 20 | 1.280 | L1 | no | 1 | −0.782 | 56.0s |
| 4 | 20 | 1.290 | L2 | no | 108 | −1.010 | 32.0s |
| 4 | 30 | 1.290 | L2 | no | 362 | −0.656 | 14.3s |
| 4 | 40 | 1.290 | L2 | no | 800 | −0.470 | 55.5s |

(connection reset before Phase-2)

### 5.4 High-$d_0$ L0 push at $c=1.25$ (`cpu_run_20260413_113016.log`)

Selected rows:

| $d_0$ | $S$ | comps | L0 surv | min_net | time |
|---|---|---|---|---|---|
| 4 | 100 | 176,851 | 6,943 | −0.024000 | 0.1s |
| 6 | 100 | 96,560,646 | 205,322 | −0.049600 | 13.9s |
| 8 | 40 | 62,891,499 | 761 | −0.171667 | 28.8s |
| 10 | 18 | 4,686,825 | 0 (grid-point) | −0.347222 | 2.6s |
| 10 | 20 | 10,015,005 | 1 | −0.258333 | 10.3s |
| 12 | 18 | 34,597,290 | 0 (grid-point) | −0.324074 | 68.9s |
| 12 | 20 | 84,672,315 | 0 (grid-point) | −0.320000 | 159.5s |
| 14 | 15 | 37,442,160 | 0 (grid-point) | −0.482593 | 116.6s |
| 16 | 12 | 17,383,860 | 0 (grid-point) | −0.888889 | 95.1s |
| 18 | 12 | 51,895,935 | 0 (grid-point) | −1.072917 | 309.6s |
| 20 | 10 | 20,030,010 | 0 (grid-point) | −1.790000 | 75.3s |

**Box cert never passed**: cell_var grows like $O(d/S)$, killing the margin even at $d_0\ge 12$.

### 5.5 Targeted L0 attempts at $c=1.30, 1.32, 1.35$ (`cpu_run_20260413_124117.log`)

| $d_0$ | $S$ | $c$ | comps | L0 surv | min_net | time |
|---|---|---|---|---|---|---|
| 6 | 100 | 1.30 | 96,560,646 | 1,190,554 | −0.052900 | 38.0s |
| 6 | 150 | 1.30 | 698,526,906 | 8,886,209 | −0.036400 | 78.9s |
| 6 | 175 | 1.30 | 1,488,847,536 | 19,117,589 | −0.030939 | 85.4s |
| 6 | 200 | 1.30 | 2,872,408,791 | 37,150,499 | −0.026725 | 161.2s |
| 6 | 100 | 1.28 | 96,560,646 | 659,737 | −0.052000 | 5.7s |
| 6 | 125 | 1.28 | 286,243,776 | 1,999,645 | −0.041216 | 16.0s |
| 6 | 150 | 1.28 | 698,526,906 | 4,935,379 | −0.035600 | 39.8s |
| 8 | 30 | 1.30 | 10,295,472 | 6,987 | −0.242222 | 1.4s |
| 8 | 40 | 1.30 | 62,891,499 | 45,719 | −0.190000 | 6.5s |
| 8 | 50 | 1.30 | 264,385,836 | 218,883 | −0.144800 | 27.7s |
| 10 | 20 | 1.30 | 10,015,005 | 118 | −0.481250 | 2.3s |
| 10 | 25 | 1.30 | 52,451,256 | 1,139 | −0.376000 | 11.3s |
| 6 | 200 | 1.32 | 2,872,408,791 | 59,892,667 | −0.027825 | 165.1s |
| 6 | 200 | 1.35 | 2,872,408,791 | 105,140,704 | −0.028425 | 521.1s |
| 6 | 250 | 1.35 | (8,637,487,551) | (skipped — too large) | — | — |

**Conclusion:** at $c\ge 1.28$ box certification fails everywhere; survivor counts blow up with $S$ instead of shrinking, signaling that the box-cert margin is the binding constraint, not survivor count.

### 5.6 Phase-1 L0 sweep, $c=1.30$ (`cpu_run_20260413_141632.log`) — 46 configs

| $d_0$ | $S$ | comps | surv | min_net | box | time | mode |
|---|---|---|---|---|---|---|---|
| 6 | 8 | 1,287 | 3 | −0.6438 | no | 1.6s | |
| 6 | 50 | 3,478,761 | 39,087 | −0.1036 | no | 0.1s | |
| 6 | 100 | 96,560,646 | 1,190,554 | −0.0529 | no | 2.5s | |
| 8 | 50 | 264,385,836 | 218,883 | −0.1448 | no | 15.1s | |
| 10 | 30 | 211,915,132 | 5,069 | −0.3139 | no | 26.4s | |
| 12 | 12 | 1,352,078 | **0** | −0.8556 | no | 0.4s | grid-point |
| 12 | 15 | 7,726,160 | 2 | −0.7400 | no | 2.1s | |
| 12 | 20 | 84,672,315 | 29 | −0.5725 | no | 22.1s | |
| 14 | 8 | 203,490 | 0 | −1.5188 | no | 0.1s | grid-point |
| 14 | 15 | 37,442,160 | 0 | −0.8178 | no | 18.2s | grid-point |
| 16 | 8 | 490,314 | 0 | −1.8795 | no | 0.4s | grid-point |
| 16 | 12 | 17,383,860 | 0 | −1.0937 | no | 13.7s | grid-point |
| 16 | 15 | 155,117,520 | 0 | −0.9038 | no | 210.9s | grid-point |
| 20 | 8 | 2,220,075 | 0 | −2.9146 | no | 6.7s | grid-point |
| 20 | 10 | 20,030,010 | 0 | −1.9667 | no | 64.6s | grid-point |

(connection reset)

### 5.7 Cascade convergence (grid-point) — `cpu_run_20260413_125241.log`

Cascade-only test (no box-cert claim) showing the survivor flow $L0\to L1\to L2\to \dots$ for a representative case $d_0=2, S=15, c=1.28$:

| Level | $d$ | tested children | survivors | min_net | step time |
|---|---|---|---|---|---|
| L0 | 2 | 16 | 4 | −0.0193 | 3.16s |
| L1 | 2→4 | 200 | 35 | −0.187 | 7.1s |
| L2 | 4→8 | 12,412 | 17 | −0.604 | 0.0s |
| L3 | 8→16 | 35,976 | **0** | −1.351 | 0.0s |

→ **PROVEN (grid-point)**: $c\ge 1.28$ at L3, total 10.4s.

Larger example $d_0=4, S=10, c=1.50$:

| Level | $d$ | children | surv | quad_corr scale | step time |
|---|---|---|---|---|---|
| L0 | 4 | 286 | 49 | — | 0.05s |
| L1 | 4→8 | 4,265 | 652 | 0.320 | 3.6s |
| L2 | 8→16 | 190,882 | 1,557 | 1.280 | 5.5s |
| L3 | 16→32 | 1,105,764 | 11 | 5.120 | 5.8s |
| L4 | 32→64 | 11,264 | **0** | 20.480 | 0.0s |

→ **PROVEN (grid-point)**: $c\ge 1.50$ at L4, total 15.0s.

**Grid-point summary table** (lines 2268–2316 of the log):

| $c$ | $(d_0, S)$ proved at level |
|---|---|
| 1.28 | $d_0=2, S\in\{15,20,25,30\}$ → L3; $d_0=3, S\in\{10,15\}$ → L2, $S=20$ → L3; $d_0=4, S\in\{10,15,20\}$ → L2 |
| 1.30 | $d_0=2, S\in\{15..30\}$ → L3; $d_0=3, S=10$ → L2, $S\in\{15,20\}$ → L3; $d_0=4, S\in\{10,15,20\}$ → L2 |
| 1.32 | $d_0=2$ → L3/L4; $d_0=3, S=10$ → L2, $S=15$ → L3; $d_0=4$ → L2 |
| 1.35 | $d_0=2$ → L4; $d_0=3$ → L3; $d_0=4, S=10$ → L2, $S=15$ → L3 |
| 1.40 | $d_0=2$ → L4/L5; $d_0=3, S=10$ → L3; $d_0=4, S=10$ → L3 |
| 1.45 | $d_0=2, S=10$ → L4, $S=15$ → L5; $d_0=3, S=10$ → L4; $d_0=4, S=10$ → L3 |
| 1.50 | $d_0=2, S=10$ → L5; $d_0=3, S=10$ → L4; $d_0=4, S=10$ → L4 |

**BEST grid-point**: $C_{1a} \ge 1.50$ — but this is **NOT** a real bound on $C_{1a}$, only on grid-point distributions.

### 5.8 Diagnostic distributions (`cpu_run_20260413_123938.log`)

Per-composition margin / cell_var / quad_corr percentiles for a few configs:

**$d_0=4, S=20, c=1.25$** — 491/560 (87.7%) box-cert pass; worst case margin=0.010, cv=0.120, qc=0.010, $S$ needed $\approx 240$.

**$d_0=6, S=20, c=1.25$** — 19,287/20,523 (94.0%) pass; worst margin=0.010, cv=0.150, qc=0.0225; $S$ needed $\approx 300$ (giving ~21 G compositions, infeasible).

**$d_0=8, S=24, c=1.25$** — 1,069,343/1,089,966 (98.1%) pass; **worst case has zero/negative margin — cannot fix with larger $S$**. Same for $d_0=4, S=20, c=1.28$.

This is the structural barrier: at $c\ge 1.25$ some compositions have margin = 0 exactly, and increasing $S$ does not help — the bound is tight at those points.

---

## 6. Method-paper performance projection (`proof/coarse_cascade_method.md` §9.3)

Cascade plan for the canonical $c=1.30, S=50$ case (not actually executed end-to-end):

| Level | $d$ | tested | survivors | time |
|---|---|---|---|---|
| L0 | 2 | 16 | 16 | <0.01s |
| L1 | 4 | 1,859 | 1,762 | 0.03s |
| L2 | 8 | 219,429 | 218,883 | 3.5s |
| L3 | 16 | (subtree-pruned) | 0 (est.) | ~6.7h sequential / ~25 min on 16 cores |

Even if the cascade converged at L3, **box certification would still fail** for $c=1.30$ (see §5.5).

---

## 7. Bottom line

| Goal | Achieved? | Where |
|---|---|---|
| Rigorous $C_{1a}\ge 1.05$ | yes | many configs, e.g., $d_0=4, S=50, c=1.05$, net 0.0428 |
| Rigorous $C_{1a}\ge 1.10$ | yes | $d_0=4, S\ge 300$ or $d_0=6, S\in\{50,75\}$; max net 0.0484 |
| Rigorous $C_{1a}\ge 1.14$ | yes | $d_0=6, S=50$, net 0.0084 |
| Rigorous $C_{1a}\ge 1.15$ | **yes (best)** | $d_0=6, S=75$, net 0.004819 |
| Rigorous $C_{1a}\ge 1.16$ | no | box cert fails |
| Rigorous $C_{1a}\ge 1.28$ | no | box cert dominates everywhere |
| Rigorous $C_{1a}\ge 1.30$ | no | grid-point converges, box cert deeply negative |
| Grid-point only $C_{1a}\ge 1.50$ | yes | $d_0\in\{2,3,4\}, S=10$ (does not bound $C_{1a}$) |

**Final achieved rigorous bound from this prover: $C_{1a} \ge 1.15$.**

Falls well short of the project's standing bound $C_{1a}\ge 1.2802$ (Cloninger–Steinerberger 2017), which uses a different (correction-based) framework. The **structural barrier** is the cell-variation term in box cert: at $c\ge 1.25$ some compositions have zero margin, so no increase in $S$ rescues the bound.
