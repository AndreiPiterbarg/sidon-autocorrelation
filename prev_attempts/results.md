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
| Lasserre level-2 | ~1.28 at P=6 | `lasserre_level2.ipynb` | Tight at P=2-4, loose at P≥5 |
| PSD lower bound | ~1.0 | `lower_bound_sdp.ipynb` | Trivial |
| Copositivity-1 | ~1.0 | `lower_bound_sdp.ipynb` | Also trivial |

---

## Key Unknowns Status

### K1: Is $C_{1a}$ truly ~1.50?
**Open.** Five independent methods converge to ~1.50. Weight of evidence says yes.

### K2: Boundary singularity at $\pm 1/4$?
**Open.** Adaptive grid experiment (this project) was inconclusive — the approach failed for reasons unrelated to the singularity question. True non-uniform ansätze (free-knot splines, geometric grids) never tested.

### K3: Is the SDP relaxation tight?
**Resolved — NO.** Shor+RLT gives $2P/(2P-1) \to 1$. Full-rank $X^*$ at every P. Structural, not numerical. Lasserre-2 helps at P≤4 but not at practical sizes.

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

**Actionable conclusion**: The most promising attack for the upper bound is **structural** (SOS/moment convexification or singular ansätze), not better optimization on uniform grids.

---

## Recommended Attack Priorities

1. **SDP certification at moderate P** — bound local-vs-global gap
2. **Non-uniform grid experiments** — test for boundary singularity
3. **Fourier-domain SDP** (Fejér-Riesz + moment) — highest risk/reward
4. **Euler-Lagrange analysis** — necessary conditions for extremizer
5. **Warm-start polishing** from best known solutions

---

## Sources

- **[MV10]** Matolcsi & Vinuesa (2010): $C_{1a} \leq 1.5098$, step functions with $n=208$, LP iteration.
- **[AE25]** AlphaEvolve (Tao et al., 2025): $C_{1a} \leq 1.5032$, LLM-evolved cubic backtracking, $n=50$.
- **[TTT26]** TTT-Discover (Yuksekgonul et al., Jan 2026): $C_{1a} \leq 1.50286$, RL-at-test-time, 30,000 pieces.
