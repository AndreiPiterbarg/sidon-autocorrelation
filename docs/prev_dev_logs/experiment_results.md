# Optimization Experiment Results — Complete Log

All experiments for the Sidon autocorrelation upper bound optimization.
Goal: minimize `||f*f||_∞ / ||f||_1²` over step functions with P pieces on [-1/4, 1/4].

**Literature references**: MV10 ≤ 1.5098 (P=208), AE25 ≤ 1.5032 (P=50), TTT26 ≤ 1.50286 (P=30000).

---

## Project Best Results

| Method | Best Value | P | Source |
|--------|-----------|---|--------|
| **Cloud LSE hybrid (Modal, 9 rounds)** | **1.5055** | 1500 | `sidon_cloud.py` |
| Cloud fast (Modal, progressive) | 1.5067 | 1000 | `sidon_cloud_fast.py` |
| Curriculum learning (Modal) | 1.5065 | 1000 | `curriculum_cloud.py` |
| LSE hybrid (local, LSE+Polyak) | 1.5092 | 200 | `logsumexp_optimizer.ipynb` |

---

## Round 1: 14 Independent Methods (P=50/100/200, 60s budget)

Baseline: Hybrid LSE+Polyak (Nesterov continuation + adaptive Polyak subgradient).

| # | Method | P=50 | P=100 | P=200 | Verdict |
|---|--------|------|-------|-------|---------|
| 1 | **Baseline LSE+Polyak** | 1.5198 | 1.5151 | 1.5121 | Reference |
| 2 | Mirror Descent (KL proximal) | 1.5196 | 1.5175 | 1.5137 | Comparable at P=50, worse at P=100/200 |
| 3 | Coordinate Descent (2-swap) | 1.5423 | 1.5372 | 1.5359 | Worse — greedy local moves fail |
| 4 | Differential Evolution | 1.5366 | 1.5254 | 1.5279 | Worse — cheap evals, no deep optimization |
| 5 | Frank-Wolfe | 1.5257 | 1.5221 | 1.5231 | Worse — vertex steps too sparse |
| 6 | Simulated Annealing | 1.5253 | 1.5231 | 1.5222 | Worse — moves too local |
| 7 | CMA-ES (log-space) | FAIL | FAIL | FAIL | Overflow in exp() |
| 8 | Moreau Envelope Smoothing | 1.5205 | 1.5209 | 1.5234 | Worse — inner loop expensive |
| 9 | Game-Theoretic Min-Max | 1.5345 | 1.5244 | 1.5301 | Worse — GDA oscillates, 1 restart only |
| 10 | Randomized Smoothing | 1.5277 | 1.5283 | 1.5342 | Worse — too many evals per gradient |
| 11 | Symmetric Exploitation | 1.5211 | 1.5147 | 1.5125 | Comparable — slightly better at P=100 |
| 12 | Multi-Peak Subgradient | 1.5274 | 1.5241 | 1.5171 | Worse — diffuse gradient, 1 restart |
| 13 | Particle Swarm (PSO) | 1.5326 | 1.5291 | 1.5245 | Worse — cheap evals can't match deep optimization |
| 14 | Fourier Parameterization | 1.5332 | 1.5263 | 1.5221 | Worse — finite-diff gradients expensive |
| 15 | Genetic Algorithm | 1.5299 | 1.5469 | 1.5300 | Worse — crossover on simplex breaks structure |

---

## Round 2: 11 Hybrid Methods (P=50/100/200, 60s budget)

All build on the LSE+Polyak pipeline as a subroutine.

| # | Method | P=50 | P=100 | P=200 | Verdict |
|---|--------|------|-------|-------|---------|
| R2-1 | Elite Pool + Cross-Pollination | 1.5216 | 1.5147 | 1.5137 | Comparable — pool helps at P=100 |
| R2-2 | Diverse Init Tournament | 1.5195 | 1.5141 | 1.5133 | Comparable — good at P=100 |
| R2-3 | Accelerated Polyak (Nesterov) | 1.5229 | 1.5160 | 1.5155 | Worse — momentum + non-smooth = oscillation |
| R2-4 | LSE + Cyclic Polish | 1.5201 | 1.5153 | 1.5124 | Comparable — ≈ equivalent to Polyak polish |
| R2-5 | Short LSE + Long Polyak | 1.5206 | 1.5129 | 1.5167 | Mixed — better P=100, worse P=200 |
| R2-6 | Double Polyak Polish | 1.5185 | 1.5157 | 1.5151 | Better at P=50 — perturbation escape works |
| R2-7 | **Warm Cascade** | **1.5171** | **1.5129** | 1.5143 | **Better at P=50/100** — low-P explore + upsample |
| R2-8 | Softmax Temperature Gradient | 1.5279 | 1.5271 | 1.5339 | Worse — too slow, 1 restart |
| R2-9 | Alternating Peak Targeting | 1.5227 | 1.5178 | 1.5178 | Worse — too few restarts |
| R2-10 | **Heavy-Tailed Init** | 1.5193 | 1.5149 | **1.5117** | **Better** — Cauchy/power-law explores diverse basins |
| R2-11 | Aggressive Beta Schedule | 1.5218 | 1.5174 | 1.5197 | Worse — schedule doesn't warm up properly |

---

## Round 3: 8 Combination Methods (P=50/100/200, 60s budget)

Combining Round 2 winners.

| # | Method | P=50 | P=100 | P=200 | Verdict |
|---|--------|------|-------|-------|---------|
| R3-1 | Heavy-Tail + Warm Cascade | 1.5196 | 1.5149 | 1.5130 | Comparable — doesn't synergize |
| R3-2 | **Elite Breeding + Heavy-Tail** | 1.5192 | **1.5140** | **1.5115** | **Best at P=100/200** — pool + mutations |
| R3-3 | Interleaved LSE/Polyak | **1.5172** | 1.5171 | 1.5152 | Better at P=50 only |
| R3-4 | Peak Variance Penalty | 1.5227 | 1.5178 | 1.5177 | Worse — Python inner loop too slow |
| R3-5 | Multi-Seed Diverse Cycling | 1.5214 | 1.5147 | 1.5130 | Comparable |
| R3-6 | Rescaled Gradient (Adam-like) | 1.5229 | 1.5160 | 1.5155 | Worse — adaptive LR doesn't help |
| R3-7 | Quadratic Approximation | 1.6683 | 1.7556 | 1.9909 | Terrible — single peak min raises all others |
| R3-8 | **Warm from Best Known** | 1.5193 | 1.5142 | **1.5112** | **Better at P=200** — warm-start + perturbation |

---

## Final Comparison (P=50/100/200, 90s budget, 3 trials)

| Method | P=50 best | P=100 best | P=200 best | P=200 mean ± std |
|--------|-----------|------------|------------|------------------|
| **Baseline LSE+Polyak** | 1.5171 | 1.5141 | **1.5104** | 1.5114 ± 0.0007 |
| Heavy-Tail Init | 1.5172 | 1.5135 | 1.5121 | 1.5125 ± 0.0006 |
| **Elite Breeding** | 1.5192 | **1.5117** | 1.5115 | **1.5120 ± 0.0004** |
| Interleaved LSE/Polyak | 1.5172 | 1.5143 | 1.5147 | 1.5152 ± 0.0004 |
| Warm Cascade | 1.5187 | 1.5130 | 1.5120 | 1.5131 ± 0.0008 |

Key: Elite Breeding has the lowest variance at P=200 (most consistent).

---

## Session 2: Warm-Restart Methods (P=200, 120s budget)

### Screening (60s, single trial)

| # | Method | Value | Verdict |
|---|--------|-------|---------|
| S2-1 | Baseline LSE+Polyak | 1.5121 | Reference |
| S2-2 | **Iterated Warm Restart** | **1.5104** | Match — reaches 90s-best in 60s |
| S2-3 | Ensemble Averaging | ~1.513 | Worse — blending destroys basins |
| S2-4 | Multi-Resolution Cascade | ~1.5135 | Worse — budget too short |
| S2-5 | Solution Surgery | ~1.512 | Neutral |
| S2-6 | Perturbation Scale Sweep | ~1.5105 | Close |
| S2-7 | Very Long Single Run | ~1.5115 | Worse — depth < breadth |
| S2-8 | Spectral Initialization | ~1.512 | Neutral |
| S2-9 | Block Coordinate Descent | ~1.514 | Worse — too slow per restart |
| S2-10 | LP-Guided Refinement | ~1.514 | Worse — Frank-Wolfe converges slowly |
| S2-11 | Heavy-Tail Elite v2 | ~1.511 | Slightly better |
| S2-12 | **Mirrored Sampling** | **1.5104** | Match — antithetic pairs effective |
| S2-13 | **Adaptive Perturbation** | **1.5104** | Match — Thompson sampling works |

### 3-Trial Comparison (120s budget)

| Method | Best | Mean | Std |
|--------|------|------|-----|
| Baseline LSE+Polyak | 1.5104 | 1.5114 | 0.0007 |
| **Iterated Warm Restart** | **1.5104** | **1.5104** | **0.0000** |
| Perturbation Scale Sweep | 1.5104 | 1.5107 | 0.0003 |
| **Mirrored Sampling** | **1.5104** | **1.5104** | **0.0000** |
| **Adaptive Perturbation** | **1.5104** | **1.5104** | **0.0000** |

Three warm-restart methods achieve perfect consistency (zero variance).

---

## Push Below 1.5104 (P=200, 120s budget, 3 trials)

### Push Round 1

| Method | Best | Mean | Std | Verdict |
|--------|------|------|-----|---------|
| Super Polish (2M iters) | 1.51036 | 1.51036 | 0.0 | Stuck — confirms local min |
| **Extreme Sparse Init** | **1.51019** | 1.51203 | 0.0018 | **New P=200 record** |
| Breed + Warm Restart | 1.51036 | 1.51036 | 0.0 | Stuck |
| Multi-Warm Heavy Polish | 1.51268 | 1.51395 | 0.0010 | Worse |
| Basin Hunt + Upsample | 1.51070 | 1.51320 | 0.0020 | Close but high variance |

### Push Round 2

| Method | Best | Mean | Std | Verdict |
|--------|------|------|-----|---------|
| Ultra Sparse (1-5 bins) | 1.51236 | 1.51369 | 0.0010 | Worse — too sparse |
| Sparse + Warm Hybrid | 1.51019 | 1.51127 | 0.0008 | Match (1/3 trials) |
| **Warm from Record** | **1.51019** | **1.51019** | **0.0000** | **Confirms basin floor** |
| Sparse + Super Polish | 1.51106 | 1.51392 | 0.0022 | Worse |
| Mixed Sparse + Structured | ~1.514 | ~1.515 | ~0.002 | Worse |

### Push Round 3

| Method | Best | Mean | Std | Verdict |
|--------|------|------|-----|---------|
| Extreme Sparse Long (300s) | 1.51019 | 1.51146 | 0.0013 | Match (1/3 trials) |

**P=200 floor: 1.51019** — confirmed as a genuine local minimum.

---

## P=500 Experiments (300–600s budget)

### Screening (300s, single trial)

| Rank | Method | Value | vs Baseline |
|------|--------|-------|-------------|
| 1 | **Iterated Warm Restart** | **1.50802** | -0.00278 |
| 2 | Mirrored Sampling | 1.50855 | -0.00225 |
| 3 | Adaptive Perturbation | 1.50932 | -0.00148 |
| 4 | Heavy-Tail Elite v2 | 1.50985 | -0.00095 |
| 5 | Baseline LSE+Polyak | 1.51080 | reference |
| 6 | Basin Hunt + Upsample | 1.51371 | +0.00291 |
| 7 | Extreme Sparse Init | 1.51372 | +0.00292 |

### Extended (600s, partial 2-trial)

| Method | Best Seen |
|--------|-----------|
| Iterated Warm Restart | **1.50763** |
| Baseline LSE+Polyak | 1.50794 |
| Mirrored Sampling | 1.50855 |
| Adaptive Perturbation | 1.50913 |

---

## Cloud Run: Progressive Upsampling (Modal, 32-core)

9-round pipeline: strategy tournament at P=200, progressive upsampling to P=1500, cross-pollination.

| P | Best Value | Round | Strategy |
|---|-----------|-------|----------|
| 200 | 1.5097 | r1 | cosine_shaped |
| 300 | 1.5074 | r2 | — |
| 500 | 1.5069 | r3 | — |
| 750 | 1.5064 | r4/r5 | — |
| 1000 | 1.5057 | r6/r7 | — |
| **1500** | **1.5055** | **r8** | **warm_perturb** |

Cross-pollination (r9) did not improve results.

---

## Cloud Fast Run (Modal, progressive P=200→1000)

| P | Best Value | Method |
|---|-----------|--------|
| 200 | 1.5094 | — |
| 500 | 1.5074 | — |
| 750 | 1.5070 | — |
| 1000 | 1.5067 | adaptive_perturbation |

Did not beat the original cloud run (1.5055 at P=1500).

---

## Curriculum Learning (Modal, 32-core)

Bottom-up: 30,000 restarts per P value at P=30/40/50, diversity filter, cascade to P=1000.

| Phase | P | Best Value |
|-------|---|-----------|
| Explore | 30 | 1.5244 |
| Explore | 40 | 1.5204 |
| Explore | 50 | 1.5171 |
| Cascade | 100 | 1.5105 |
| Cascade | 200 | 1.5096 |
| Cascade | 500 | 1.5074 |
| Cascade | 1000 | 1.5065 |

**Did not beat the cloud run.** Diversity filtering at P=50 kept solutions in the same basin; upsampled warm-starts get trapped at high P.

---

## Speed Optimizations

Three optimizations applied to the LSE+Polyak core (P=200, 30s budget):

| Optimization | Speedup | Mechanism |
|--------------|---------|-----------|
| Lazy k_star in Polyak | 1.56x | Check neighbors instead of full autoconv every iter |
| Reduced LSE tracking | 1.28x | Check true objective every 15 iters instead of every iter |
| Shortened beta (warm only) | 1.68x | Skip early low-beta stages for warm-started solutions |

**Method-level impact:**

| Method | Speedup | Restarts (30s) |
|--------|---------|----------------|
| Iterated Warm Restart | **2.04x** | 8 → 15 |
| Mirrored Sampling | 1.84x | 9 → 16 |
| Adaptive Perturbation | 1.87x | 8 → 15 |
| Heavy-Tail Elite v2 | 1.30x | 8 → 10 |
| Extreme Sparse Init | 1.07x | 8 → 9 |

---

## Key Takeaways

### What Worked
1. **LSE continuation + Polyak subgradient** is the unbeatable core pipeline. All winning methods use it as a subroutine.
2. **Warm-starting from known solutions** is the single biggest advantage at high P.
3. **Heavy-tailed initialization** (Cauchy, power-law, sparse Dirichlet) explores basins that standard Dirichlet never reaches.
4. **Extreme sparsity** (3–14 active bins) finds new basins at moderate P, breaking the 1.5104 barrier at P=200.
5. **More pieces (higher P) monotonically improves** the bound: P=200→1.510, P=500→1.508, P=1000→1.506, P=1500→1.5055.

### What Didn't Work
1. **Population-based methods** (DE, GA, PSO, CMA-ES) without deep per-candidate optimization.
2. **Alternative smoothing** (Moreau, randomized smoothing, softmax temp) — LSE continuation is already optimal.
3. **Coordinate methods** (coordinate descent, alternating peaks) — landscape too coupled.
4. **Quadratic approximation** — minimizing one peak raises all others.
5. **Curriculum learning** — diversity filtering fails, upsampled solutions get trapped.

### The Fundamental Lesson
The bottleneck is **initialization**, not optimization. Given a good basin, LSE+Polyak converges reliably. The challenge is finding the right basin. The optimal solution is sparse/concentrated, not diffuse — heavy-tailed and sparse initializations matter.
