# Optimization Experiment Results

## Setup
- Compute budget: 60 seconds per method per P value (Round 1-3), 90 seconds (Final)
- P values tested: 50, 100, 200
- Baseline: Hybrid LSE+Polyak (Nesterov continuation + adaptive Polyak subgradient)
- Hardware: Local machine, single-threaded per restart
- All solutions verified via `autoconv_coeffs(x, P)` and stored as JSON

## Baseline Results (60s budget)
| P | Best Value | Effective Restarts | Time |
|---|-----------|-------------------|------|
| 50 | 1.519790 | 103 | 60.3s |
| 100 | 1.515085 | 49 | 60.2s |
| 200 | 1.512053 | 18 | 62.7s |

---

## Round 1: 14 Independent Methods (60s budget each)

### Experiment 1: Mirror Descent (KL Proximal)
**Idea:** Multiplicative weights update on the simplex — respects simplex geometry natively via KL divergence instead of Euclidean projection.
**Implementation:** Exponentiated gradient: `x_new[i] = x[i] * exp(-eta * g[i]) / Z`, with LSE continuation schedule + Polyak polish.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.519638 | -0.000152 | 41 | 60.6s |
| 100 | 1.517523 | +0.002438 | 22 | 60.9s |
| 200 | 1.513702 | +0.001649 | 8 | 60.3s |
**Verdict:** COMPARABLE at P=50, WORSE at P=100,200 -- fewer restarts per time unit (mirror update is slower)

### Experiment 2: Coordinate Descent (2-swap)
**Idea:** Cheapest possible iteration: pick two coordinates, move mass between them to reduce argmax peak.
**Implementation:** After short LSE warmup, greedy 2-coordinate swaps targeting the highest contributor to the current peak.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.542333 | +0.022543 | 5 | 60.0s |
| 100 | 1.537193 | +0.022108 | 4 | 60.0s |
| 200 | 1.535937 | +0.023884 | 2 | 60.0s |
**Verdict:** WORSE -- greedy local moves can't navigate the non-convex landscape

### Experiment 3: Differential Evolution
**Idea:** Population-based search with DE/rand/1 mutation and crossover on the simplex.
**Implementation:** 20-member population, F=0.8, CR=0.7, simplex projection after mutation.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.536584 | +0.016794 | 40909 gens | 60.0s |
| 100 | 1.525356 | +0.010271 | 26721 | 60.0s |
| 200 | 1.527945 | +0.015892 | 16778 | 60.0s |
**Verdict:** WORSE -- many cheap evaluations but no deep optimization per candidate

### Experiment 4: Frank-Wolfe (Conditional Gradient)
**Idea:** Linear minimization over simplex is trivial (pick argmin gradient). No projection needed.
**Implementation:** Step size gamma=2/(t+2), LSE continuation + Polyak polish.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.525747 | +0.005957 | 41 | 60.3s |
| 100 | 1.522061 | +0.006976 | 22 | 60.1s |
| 200 | 1.523135 | +0.011082 | 7 | 60.5s |
**Verdict:** WORSE -- Frank-Wolfe steps are too sparse (move to vertex), poor intermediate iterates

### Experiment 5: Simulated Annealing
**Idea:** Accept worse solutions probabilistically (Boltzmann schedule) to escape local minima.
**Implementation:** Simplex-preserving 2-coordinate swaps, exponential cooling T_init=0.05 to T_final=1e-5, + Polyak polish.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.525327 | +0.005537 | 11 | 60.3s |
| 100 | 1.523091 | +0.008006 | 11 | 60.7s |
| 200 | 1.522183 | +0.010130 | 8 | 61.2s |
**Verdict:** WORSE -- SA moves are too small/local, can't match gradient-based basin finding

### Experiment 6: CMA-ES (Log-space)
**Idea:** Covariance matrix adaptation in log-space, then exponentiate to simplex.
**Implementation:** Population of 15, rank-mu update, sigma adaptation.
**Results:** FAILED -- overflow in exp() produces NaN. Implementation buggy.
**Verdict:** FAILED -- needs numerical stabilization

### Experiment 7: Moreau Envelope Smoothing
**Idea:** Proximal smoothing: minimize min_y { max_k c_k(y) + ||x-y||^2/(2*mu) } with continuation on mu.
**Implementation:** Inner proximal gradient loop per mu level, then Polyak polish.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.520538 | +0.000748 | 99 | 60.6s |
| 100 | 1.520896 | +0.005811 | 54 | 60.2s |
| 200 | 1.523374 | +0.011321 | 25 | 61.7s |
**Verdict:** WORSE -- Moreau inner loop is expensive and doesn't converge well for this objective

### Experiment 8: Game-Theoretic Min-Max
**Idea:** Treat max_k c_k as a two-player game: x minimizes, adversary k maximizes. Simultaneous GDA with multiplicative weights for k.
**Implementation:** Primal gradient descent on weighted objective, dual multiplicative weights update for k.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.534509 | +0.014719 | 1 | 60.5s |
| 100 | 1.524437 | +0.009352 | 1 | 60.9s |
| 200 | 1.530112 | +0.018059 | 1 | 62.3s |
**Verdict:** WORSE -- GDA oscillates, only 1 restart possible in budget

### Experiment 9: Randomized Smoothing
**Idea:** Estimate gradient of E[max_k c_k(x+noise)] via antithetic sampling. Smoothed objective is differentiable.
**Implementation:** 10 antithetic samples per gradient estimate, sigma continuation, + Polyak polish.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.527719 | +0.007929 | 16 | 60.5s |
| 100 | 1.528318 | +0.013233 | 12 | 60.5s |
| 200 | 1.534166 | +0.022113 | 7 | 62.8s |
**Verdict:** WORSE -- too many evaluations per gradient estimate

### Experiment 10: Symmetric Exploitation
**Idea:** Restrict to symmetric functions f(x)=f(-x), halving effective dimension.
**Implementation:** Generate symmetric inits, run standard hybrid, re-symmetrize and polish.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.521140 | +0.001350 | 67 | 60.2s |
| 100 | 1.514697 | -0.000388 | 33 | 60.3s |
| 200 | 1.512513 | +0.000460 | 10 | 61.2s |
**Verdict:** COMPARABLE -- slightly better at P=100 but not consistent

### Experiment 11: Multi-Peak Subgradient
**Idea:** Instead of cutting just the argmax, average subgradients over ALL peaks within epsilon of max.
**Implementation:** Short LSE warmup, then multi-peak subgradient with adaptive epsilon.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.527397 | +0.007607 | 1 | 60.0s |
| 100 | 1.524091 | +0.009006 | 1 | 60.0s |
| 200 | 1.517073 | +0.005020 | 1 | 60.0s |
**Verdict:** WORSE -- the multi-peak gradient is diffuse, only 1 restart in budget

### Experiment 12: Particle Swarm Optimization
**Idea:** Swarm of 20 particles with personal/global best tracking, velocity update.
**Implementation:** Standard PSO with simplex projection, + Polyak polish of global best.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.532609 | +0.012819 | 132K iters | 60.0s |
| 100 | 1.529131 | +0.014046 | 126K | 60.0s |
| 200 | 1.524466 | +0.012413 | 98K | 60.0s |
**Verdict:** WORSE -- PSO with cheap evals can't match deep gradient optimization

### Experiment 13: Fourier Parameterization
**Idea:** Parameterize f as |sum_k a_k exp(2pi i k x)|^2 (automatically nonneg). Optimize Fourier coefficients.
**Implementation:** Finite-difference gradient on Fourier coefficients, beta continuation, + Polyak polish.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.533218 | +0.013428 | 2 | 60.5s |
| 100 | 1.526347 | +0.011262 | 1 | 60.8s |
| 200 | 1.522148 | +0.010095 | 1 | 62.4s |
**Verdict:** WORSE -- finite-difference gradients too expensive, Fourier parameterization overly constraining

### Experiment 14: Genetic Algorithm
**Idea:** Tournament selection + arithmetic crossover + Gaussian mutation on simplex.
**Implementation:** Population of 30, elitism (keep top 2), mutation rate 0.1, + Polyak polish of best.
**Results:**
| P | Best Value | vs Baseline | Effective Restarts | Time |
|---|-----------|------------|-------------------|------|
| 50 | 1.529924 | +0.010134 | 48K gens | 60.0s |
| 100 | 1.546947 | +0.031862 | 47K | 60.0s |
| 200 | 1.529961 | +0.017908 | 44K | 60.0s |
**Verdict:** WORSE -- crossover on simplex doesn't preserve good structure

---

## Round 2: Hybrid Methods Building on LSE+Polyak (60s budget)

### Experiment R2-1: Elite Pool + Cross-Pollination
**Idea:** Maintain pool of K best solutions. Breed new candidates by interpolating pool members + noise.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.521632 | +0.001842 | 88 |
| 100 | 1.514729 | -0.000356 | 38 |
| 200 | 1.513748 | +0.001695 | 15 |
**Verdict:** COMPARABLE -- pool breeding helps slightly at P=100

### Experiment R2-2: Diverse Init Tournament
**Idea:** Use ALL 11 initialization strategies from sidon_core, equal time each.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.519535 | -0.000255 | 89 |
| 100 | 1.514113 | -0.000972 | 42 |
| 200 | 1.513254 | +0.001201 | 15 |
**Verdict:** COMPARABLE -- diverse strategies help at P=50,100 but not P=200

### Experiment R2-3: Accelerated Polyak (Nesterov momentum)
**Idea:** Add Nesterov momentum to Polyak subgradient phase.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.522923 | +0.003133 | 10 |
| 100 | 1.515997 | +0.000912 | 6 |
| 200 | 1.515450 | +0.003397 | 4 |
**Verdict:** WORSE -- momentum on non-smooth subgradient causes oscillation, fewer restarts

### Experiment R2-4: LSE + Cyclic Polish
**Idea:** Replace Polyak polish with cyclic peak-cutting (round-robin over near-peak indices).
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.520116 | +0.000326 | 93 |
| 100 | 1.515320 | +0.000235 | 44 |
| 200 | 1.512433 | +0.000380 | 16 |
**Verdict:** COMPARABLE -- cyclic polish is ~equivalent to Polyak

### Experiment R2-5: Short LSE + Long Polyak
**Idea:** Fewer LSE stages (8 instead of 21), shorter per-stage (3K iters), longer Polyak (500K).
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.520594 | +0.000804 | 55 |
| 100 | 1.512913 | -0.002172 | 21 |
| 200 | 1.516670 | +0.004617 | 11 |
**Verdict:** MIXED -- better at P=100 (longer Polyak helps), worse at P=200 (LSE warmup matters)

### Experiment R2-6: Double Polyak Polish
**Idea:** After standard hybrid, perturb best solution and run a second Polyak pass.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.518490 | -0.001300 | 87 |
| 100 | 1.515706 | +0.000621 | 38 |
| 200 | 1.515055 | +0.003002 | 15 |
**Verdict:** BETTER at P=50 -- perturbation escape works at low dimension

### Experiment R2-7: Warm Cascade (low-P explore then upsample)
**Idea:** 40% of budget exploring at P_low, then upsample top-5 and polish at target P.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.517124 | **-0.002666** | 111 |
| 100 | 1.512876 | **-0.002209** | 53 |
| 200 | 1.514301 | +0.002248 | 37 |
**Verdict:** BETTER at P=50,100 -- low-P exploration finds good basins. Worse at P=200 (upsampling loses precision)

### Experiment R2-8: Softmax Temperature Gradient
**Idea:** Direct gradient descent with softmax-weighted combination of all peak gradients, temperature schedule.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.527873 | +0.008083 | 2 |
| 100 | 1.527111 | +0.012026 | 1 |
| 200 | 1.533866 | +0.021813 | 1 |
**Verdict:** WORSE -- too few restarts, method is slow

### Experiment R2-9: Alternating Peak Targeting
**Idea:** Track top-2 peaks and distribute gradient steps proportionally.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.522714 | +0.002924 | 6 |
| 100 | 1.517836 | +0.002751 | 4 |
| 200 | 1.517751 | +0.005698 | 2 |
**Verdict:** WORSE -- too few restarts (Python-level Polyak replacement is slow)

### Experiment R2-10: Heavy-Tailed Initialization
**Idea:** Use Cauchy, power-law, log-normal, very-sparse Dirichlet instead of standard Dirichlet.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.519322 | -0.000468 | 90 |
| 100 | 1.514898 | -0.000187 | 42 |
| 200 | **1.511674** | **-0.000379** | 15 |
**Verdict:** **BETTER** -- heavy-tailed inits explore more diverse basins, especially at P=200

### Experiment R2-11: Aggressive Beta Schedule
**Idea:** Much fewer, larger beta jumps (7 stages: 1,5,25,125,625,3K,10K), more restarts.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.521803 | +0.002013 | 92 |
| 100 | 1.517385 | +0.002300 | 47 |
| 200 | 1.519706 | +0.007653 | 19 |
**Verdict:** WORSE -- aggressive schedule doesn't warm up properly, more restarts don't compensate

---

## Round 3: Combining Winners (60s budget)

### Experiment R3-1: Heavy-Tail + Warm Cascade
**Idea:** Combine heavy-tailed init (R2 winner) with warm cascade (R2 winner at P=50,100).
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.519553 | -0.000237 | 119 |
| 100 | 1.514942 | -0.000143 | 54 |
| 200 | 1.513038 | +0.000985 | 27 |
**Verdict:** COMPARABLE -- combination doesn't synergize well

### Experiment R3-2: Elite Breeding + Heavy-Tail
**Idea:** Pool-based breeding with heavy-tailed mutations between pool members.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.519238 | -0.000552 | 96 |
| 100 | **1.514005** | **-0.001080** | 45 |
| 200 | **1.511509** | **-0.000544** | 16 |
**Verdict:** **BETTER** at P=100,200 -- pool breeding + heavy-tail mutations = good exploration

### Experiment R3-3: Interleaved LSE/Polyak
**Idea:** Alternate short LSE and short Polyak phases instead of sequential.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | **1.517150** | **-0.002640** | 102 |
| 100 | 1.517075 | +0.001990 | 52 |
| 200 | 1.515168 | +0.003115 | 18 |
**Verdict:** **BETTER at P=50** only -- interleaving helps at low P, hurts at high P

### Experiment R3-4: Peak Variance Penalty
**Idea:** Push down all peaks above the mean of top-5 peaks, not just the argmax.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.522701 | +0.002911 | 5 |
| 100 | 1.517836 | +0.002751 | 3 |
| 200 | 1.517745 | +0.005692 | 2 |
**Verdict:** WORSE -- Python-level inner loop is too slow, few restarts

### Experiment R3-5: Multi-Seed Diverse Cycling
**Idea:** Cycle through 4 init types (sparse Dirichlet, Cauchy, uniform Dirichlet, concentrated) with random seeds.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.521409 | +0.001619 | 96 |
| 100 | 1.514726 | -0.000359 | 45 |
| 200 | 1.513014 | +0.000961 | 16 |
**Verdict:** COMPARABLE -- cycling helps at P=100

### Experiment R3-6: Rescaled Gradient (Adam-like)
**Idea:** Per-coordinate adaptive learning rate in Polyak phase (second moment rescaling).
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.522923 | +0.003133 | 13 |
| 100 | 1.515997 | +0.000912 | 9 |
| 200 | 1.515450 | +0.003397 | 5 |
**Verdict:** WORSE -- adaptive LR doesn't help on non-smooth objective, fewer restarts

### Experiment R3-7: Quadratic Approximation
**Idea:** After LSE+Polyak, minimize the quadratic c_{k*}(x) for current argmax k* via projected gradient.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.668299 | +0.148509 | 137 |
| 100 | 1.755618 | +0.240533 | 61 |
| 200 | 1.990868 | +0.478815 | 22 |
**Verdict:** TERRIBLE -- minimizing a single peak's quadratic raises all other peaks

### Experiment R3-8: Warm Restart from Best Known
**Idea:** Seed from the best solution found in R2 (heavy_tail_init), perturb + re-optimize.
**Results:**
| P | Best Value | vs Baseline | Restarts |
|---|-----------|------------|---------|
| 50 | 1.519322 | -0.000468 | 96 |
| 100 | 1.514229 | -0.000856 | 47 |
| 200 | **1.511243** | **-0.000810** | 17 |
**Verdict:** **BETTER** at P=200 -- warm-starting from known good solutions + perturbation is powerful (but partly unfair since it uses prior knowledge)

---

## Final Comparison (90s budget, 3 trials per method)
*(Results pending -- running now)*

---

## Summary Table (60s budget, best across all rounds)

| Rank | Method | P=50 | P=100 | P=200 | Notes |
|------|--------|------|-------|-------|-------|
| 1 | **Warm Cascade** | **1.5171** | 1.5129 | 1.5143 | Best at P=50 (fresh exploration) |
| 2 | **Elite Breeding + Heavy-Tail** | 1.5192 | **1.5140** | 1.5115 | Best at P=100, strong at P=200 |
| 3 | **Heavy-Tailed Init** | 1.5193 | 1.5149 | 1.5117 | Simple and effective |
| 4 | **Warm from Best** | 1.5193 | 1.5142 | **1.5112** | Best at P=200 (uses prior) |
| 5 | Baseline (LSE+Polyak) | 1.5198 | 1.5151 | 1.5121 | The method to beat |
| 6 | Interleaved LSE/Polyak | 1.5172 | 1.5171 | 1.5152 | Great at P=50, bad at P=200 |
| 7 | Double Polyak | 1.5185 | 1.5157 | 1.5151 | Perturbation helps at P=50 |
| 8 | Mirror Descent | 1.5196 | 1.5175 | 1.5137 | Competitive at P=50 |
| 9 | Diverse Init Tournament | 1.5195 | 1.5141 | 1.5133 | Good at P=100 |
| 10 | LSE + Cyclic Polish | 1.5201 | 1.5153 | 1.5124 | ~Same as Polyak polish |
| 11 | Short LSE + Long Polyak | 1.5206 | 1.5129 | 1.5167 | Good at P=100, bad at P=200 |
| 12-26 | (remaining methods) | >1.52 | >1.52 | >1.52 | Significantly worse |

---

## Key Takeaways

### What Worked
1. **Heavy-tailed initialization** (Cauchy, power-law, very sparse Dirichlet) explores more diverse basins than standard Dirichlet. Improvement at P=200: -0.0004 vs baseline. This is the simplest improvement and requires zero extra computation.

2. **Elite pool breeding** amplifies the effect of diverse initialization: solutions that survive in the pool share "good genes" through interpolation, and heavy-tailed mutations prevent convergence to a single basin.

3. **Warm cascade** (explore at low P, upsample top solutions) is very effective at P=50 and P=100 where the low-P landscape is explored more thoroughly. It underperforms at P=200 because upsampling loses fine-grained structure.

4. **Interleaved LSE/Polyak** helps at low P by preventing the LSE phase from over-committing to a smooth approximation of the wrong peak.

### What Didn't Work
1. **Population-based methods** (DE, GA, PSO, CMA-ES) without deep per-candidate optimization. The key insight: one deep restart (LSE+Polyak, ~3s at P=200) is worth more than 10,000 cheap evaluations. Methods must use the LSE+Polyak pipeline as a building block.

2. **Alternative smoothing** (Moreau envelope, randomized smoothing, softmax temperature). LSE continuation is already a very effective smoothing strategy; alternatives are either more expensive or less effective.

3. **Coordinate methods** (coordinate descent, alternating peaks). The landscape is too coupled for axis-aligned moves.

4. **Aggressive beta schedules**. The standard 21-stage BETA_HEAVY schedule is well-tuned; shortcuts lose quality without compensating restarts.

5. **Quadratic approximation**. Minimizing a single peak raises all others. The coupling between peaks is the fundamental difficulty.

### Scaling Recommendations
Methods to scale up to higher P (500+) in cloud compute:
1. **Elite breeding + heavy-tail**: Run as the primary init strategy in the cloud pipeline (replace `dirichlet_uniform`)
2. **Warm cascade**: Use as the upsampling strategy (replace simple `warm_perturb`)
3. **Interleaved LSE/Polyak**: Only at P<100 in the early cascade stages
4. Combine all three: heavy-tail exploration at P=50-100, elite breeding at P=100-300, warm cascade to P=500+

### Fundamental Lesson
The bottleneck is **initialization**, not optimization. Given a good basin, LSE+Polyak converges reliably. The challenge is finding the right basin. Heavy-tailed distributions explore more extreme regions of the simplex where standard Dirichlet(1,...,1) never reaches. This matters because the optimal solution for C_{1a} appears to be sparse/concentrated, not diffuse.
