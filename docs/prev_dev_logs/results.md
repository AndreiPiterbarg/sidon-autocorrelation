# Results & Analysis — Sidon Autocorrelation Project

**Current bounds**: $C_{1a} \in [1.2802, 1.5029]$. Gap $\approx 0.223$.

---

## Upper Bound Methods (Best per Method)

| Method | Best Loss | Pieces | Notebook | Notes |
|--------|-----------|--------|----------|-------|
| **Cloud LSE hybrid (Modal)** | **1.5055** | 1500 | `sidon_cloud.py` | **New project best** — 9-round pipeline, warm_perturb |
| LSE hybrid (LSE+Polyak) | 1.5092 | 200 | `logsumexp_optimizer.ipynb` | Previous project best |
| LSE continuation (Nesterov) | 1.5112 | 150 | `logsumexp_optimizer.ipynb` | Wins 8/8 vs Polyak |
| LP iteration (MV10) | 1.5123 | 600 | — | Published method |
| SQP / Prox-linear | 1.5168 | 1000 | — | |
| Polyak subgradient (JIT) | 1.5163 | 150 | `primal_optimizer.ipynb` | 180 parallel runs |
| Peak redistrib. hybrid | ~1.508 | 1000 | `primal_optimizer.ipynb` | |
| L-BFGS-B (restarts) | ~1.584 | 10 | `primal_optimizer.ipynb` | Softmax reparam. |
| Basin hopping + L-BFGS | ~1.575 | 10 | `primal_optimizer.ipynb` | |
| Gaussian mixture | 1.545 | — | — | Grid-evaluated max |
| Adam (reduce on plateau) | 1.5754 | — | — | Limited tuning |
| TV surrogate | 1.6574 | — | — | Surrogate too loose |
| Adaptive grid | 1.537 | 500 | `adaptive_grid_optimizer.ipynb` | **Failed** — worse than uniform |
| Joint edge+height | ~1.51 | 200 | `joint_edge_height_optimizer.ipynb` | Non-uniform grid |

### LSE Head-to-Head vs Polyak (exact peak, uniform grid)

| P | LSE | Polyak | Delta |
|---|-----|--------|-------|
| 10 | 1.566445 | 1.569110 | -0.003 |
| 50 | 1.521646 | 1.524243 | -0.003 |
| 100 | 1.513841 | 1.520211 | -0.006 |
| 200 | 1.512395 | 1.519926 | -0.008 |

### LSE Hybrid Results (best overall)

| P | Hybrid | LSE only | Polyak only |
|---|--------|----------|-------------|
| 50 | 1.521775 | 1.521646 | 1.524243 |
| 100 | 1.515366 | 1.513841 | 1.520211 |
| 200 | **1.509246** | 1.512395 | 1.519926 |

### Cloud Run Results (Modal, 32-core, 9 rounds)

Full pipeline: strategy tournament at P=200, progressive upsampling through P=1500, cross-pollination. Best strategy: `warm_perturb` (perturb + re-optimize from best known solution).

| P | Best Peak | Round | Strategy |
|---|-----------|-------|----------|
| 200 | 1.509734 | r1 | cosine_shaped |
| 300 | 1.507367 | r2 | — |
| 500 | 1.506877 | r3 | — |
| 750 | 1.506413 | r4/r5 | — |
| 1000 | 1.505695 | r6/r7 | — |
| **1500** | **1.505549** | **r8** | **warm_perturb** |

Round 1 strategy ranking (P=200, top 6 advanced): cosine_shaped, random_sparse_k, dirichlet_uniform, boundary_heavy, triangle, dirichlet_concentrated.

Cross-pollination (r9) did not improve: P=750 gave 1.5076, P=1000 gave 1.5066, P=1500 gave 1.5076 — all worse than warm_perturb results at the same P.

**Key observation**: Monotone improvement with increasing P (1.5097 at P=200 → 1.5055 at P=1500), consistent with literature trend toward ~1.503 at very high P.

---

## Lower Bound Methods

| Method | Best Bound | Notebook | Notes |
|--------|-----------|----------|-------|
| Known (literature) | 1.2802 | — | Fourier kernel method |
| Shor+RLT SDP | ~1.01 | `sdp_certification.ipynb` | $2P/(2P-1) \to 1$, useless |
| Fourier SDP (cosine) | 2.0 | `sdp_certification.ipynb` | Trivial (even functions) |
| Lasserre level-2 (CLARABEL sweep) | **1.4925 at P=12 for $\eta_P$** | `lasserre_level2.ipynb`, `lasserre2_sweep.py` | Strong discretized relaxation bound; not a direct continuous $C_{1a}$ bound |
| PSD lower bound | ~1.0 | `lower_bound_sdp.ipynb` | Trivial |
| Copositivity-1 | ~1.0 | `lower_bound_sdp.ipynb` | Also trivial |

### 2026-02 Lasserre-2 Reproducible Sweep (new)

Command used:
`python prev_attempts/lasserre2_sweep.py --p-values 6,7,8,9,10,11,12 --primary-solver CLARABEL --crosscheck-solver SCS --crosscheck-max-p 8 --eta-tol 1e-3 --primal-restarts 20 --out prev_attempts/lasserre2_sweep_results.json`

Primary solver (`CLARABEL`) results:

| P | Lasserre-2 LB (primary) | Primal UB (comparison) | Gap |
|---|--------------------------|------------------------|-----|
| 6  | 1.5827 | 1.6025 | 0.0198 |
| 7  | 1.5760 | 1.6060 | 0.0300 |
| 8  | 1.5400 | 1.5964 | 0.0564 |
| 9  | 1.5328 | 1.5876 | 0.0547 |
| 10 | 1.5129 | 1.5970 | 0.0841 |
| 11 | 1.5037 | 1.5995 | 0.0958 |
| 12 | **1.4925** | 1.5883 | 0.0957 |

Cross-check (`SCS`) at low P was much weaker: 1.2842 (P=6), 1.2449 (P=7), 1.2127 (P=8), indicating strong solver/numerical sensitivity.

### What Is / Isn't Proved (exact relationships)

Define:

- $C_{1a} := \inf_{f \in \mathcal{F}} M(f)$, where $M(f)=\|f*f\|_\infty$ and $\mathcal{F}$ is the full continuous feasible set.
- $\eta_P := \inf_{f \in \mathcal{S}_P} M(f)$, where $\mathcal{S}_P \subset \mathcal{F}$ is the $P$-bin step-function subclass.
- $LB_P$ := level-2 Lasserre lower bound computed for the discretized $P$-bin problem.

What we proved numerically:

- $LB_P \le \eta_P$ (lower bound on the discretized minimax value).
- $C_{1a} \le \eta_P$ (because $\mathcal{S}_P \subset \mathcal{F}$).

What we did **not** prove:

- We did **not** prove $LB_P \le C_{1a}$.
- Therefore values like $LB_{12}=1.4925$ do **not** imply $C_{1a}\ge 1.4925$.

What is needed to certify continuous lower bounds from this pipeline:

- A discretization-transfer estimate $\eta_P - C_{1a} \le \varepsilon_P$.
- Then one gets a certified continuous bound: $C_{1a} \ge LB_P - \varepsilon_P$.

Current status: $\varepsilon_P$ is not yet controlled in this repo.

---

## Key Unknowns Status

### K1: Is $C_{1a}$ truly ~1.50?
**Open.** Five independent methods converge to ~1.50. Weight of evidence says yes.

### K2: Boundary singularity at $\pm 1/4$?
**Open.** Adaptive grid experiment (this project) was inconclusive — the approach failed for reasons unrelated to the singularity question. True non-uniform ansätze (free-knot splines, geometric grids) never tested.

### K3: Is the SDP relaxation tight?
**Partially resolved.** Shor+RLT is not tight (structural failure). Level-2 Lasserre with CLARABEL is much stronger for discretized $\eta_P$, but continuous-certification status remains open because discretization transfer is unproven.

### K4: Moment/SOS convexification?
**Open.** Most promising unexploited direction. Fourier-domain SDP with full basis (sine+cosine) + Fejér-Riesz never attempted.

### K5: Fourier kernel lower bound ceiling?
**Open.** Estimated ~1.276 by [MV10], essentially saturated by current 1.2802.

---

## Barrier Analysis

**All approaches hit the same fundamental wall**: non-convexity of $\|f*f\|_\infty$ (S1).

**Evidence**: Six independent paradigms (LP iteration, Polyak, LLM-evolved, RL, LogSumExp, cloud multi-strategy) converge to [1.500, 1.516]. Qualitatively different local minima have nearly identical values. Cloud run at P=1500 reached 1.5055, further narrowing the gap.

**Slack decomposition**:
- Upper bound slack: ~0.003–0.05 (dominated by non-uniform discretization unknowns)
- Lower bound slack: ~0.17–0.22 (dominant — ~98% of the gap)

**Actionable conclusion (for lower-bound certification)**: strong discretized SOS numbers are not enough; the bottleneck is turning $LB_P$ into certified bounds on continuous $C_{1a}$ via a rigorous transfer error $\varepsilon_P$.

---

## Recommended Attack Priorities

1. **Discretization-to-continuous transfer bound** — derive/compute $\varepsilon_P$ such that $\eta_P - C_{1a} \le \varepsilon_P$.
2. **Certified residual checking for SDP outputs** — convert `optimal_inaccurate` solutions into validated bounds using primal/dual residual and PSD-eigenvalue certificates.
3. **Combine steps 1+2 into continuous certificates** — report $C_{1a} \ge LB_P - \varepsilon_P$ with explicit error bars.
4. **Continuous dual-certificate search** — optimize Fourier/kernel certificates with rigorous quadrature/interval-error control (not grid-only heuristics).
5. **Only then scale hierarchy level** — level-3+ is useful only if paired with an explicit transfer/certification pipeline.

---

## Sources

- **[MV10]** Matolcsi & Vinuesa (2010): $C_{1a} \leq 1.5098$, step functions with $n=208$, LP iteration.
- **[AE25]** AlphaEvolve (Tao et al., 2025): $C_{1a} \leq 1.5032$, LLM-evolved cubic backtracking, $n=50$.
- **[TTT26]** TTT-Discover (Yuksekgonul et al., Jan 2026): $C_{1a} \leq 1.50286$, RL-at-test-time, 30,000 pieces.
