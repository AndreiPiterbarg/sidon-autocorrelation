# Full Problem State (2026-04-14)

> **Status (2026-05-12).** Historical snapshot. Both lower-bound
> proofs are now complete: $C_{1a} \ge 7/5 = 1.4$ via the cascade
> (`proof/cs-proof/`) and $C_{1a} \ge 1.3$ via Lasserre
> (`proof/lasserre-proof/`). For current state see `README.md` and
> the manuscript PDFs. The text below records the Lasserre solver
> bottlenecks as they stood on 2026-04-14.

## What We Are Trying to Do

We are trying to improve the lower bound on the **Sidon autocorrelation constant** C_{1a}. The current best bounds are:

    1.2802 <= C_{1a} <= 1.5029

The lower bound 1.2802 was proved by the cascade method. We want to push it higher using the **Lasserre SDP hierarchy**.

The mathematical chain is:

    lb (our SDP output) <= val(d) (discrete problem) <= C_{1a} (continuous constant)

If we can compute lb > 1.2802 at ANY discretization level d, we have a new world-record proof.

---

## The Discrete Problem

val(d) = min_{mu in Delta_d} max_W  mu^T M_W mu

where Delta_d is the standard d-simplex (masses mu_i >= 0, sum = 1) and M_W are window matrices encoding the autoconvolution test values. val(d) is monotonically increasing in d and converges to C_{1a}.

### Known val(d) Values (numerical upper bounds from multistart optimization)

| d   | val(d)     | gc needed to beat 1.2802 | possible? |
|-----|------------|--------------------------|-----------|
| 4   | 1.10233    | 273.8%                   | NO        |
| 6   | 1.17110    | 163.8%                   | NO        |
| 8   | 1.20464    | 136.9%                   | NO        |
| 10  | 1.24137    | 116.1%                   | NO        |
| 12  | 1.27072    | 103.5%                   | NO        |
| 14  | 1.28396    | 98.7%                    | YES (barely) |
| 16  | 1.31852    | 88.0%                    | YES       |
| 32  | 1.336      | 83.4%                    | YES       |
| 64  | 1.384      | 73.0%                    | YES       |
| 128 | 1.420      | 66.7%                    | YES       |
| 256 | 1.448      | 62.5%                    | YES       |

"gc needed" = what fraction of (val(d) - 1) our SDP lower bound must capture to exceed 1.2802.

**Key crossover**: d >= 14 is required. d=16 is the first comfortable target (88% gc needed).

---

## The Lasserre SDP Approach

The Lasserre order-k relaxation introduces pseudo-moment variables y_alpha = E[x^alpha] for all multi-indices |alpha| <= 2k, subject to:

- (L1) y_0 = 1 (normalization)
- (L2) y_alpha >= 0 (nonnegativity)
- (L3) M_k(y) >= 0 (moment matrix PSD)
- (L4) M_{k-1}(mu_i * y) >= 0 (localizing for mu_i >= 0)
- (L5) y_alpha = sum_i y_{alpha+e_i} (consistency from sum mu_i = 1)
- (L6) t * M_{k-1}(y) - sum M_W[i,j] * M_{k-1}(mu_i*mu_j*y) >= 0 (window PSD)
- (L7) M_{k-1}(y) - M_{k-1}(mu_i*y) >= 0 (upper localizing, optional)

The output lb satisfies lb <= val(d) (proven sound). Higher order k gives tighter bounds.

### Two Solver Implementations

**1. Full solver (lasserre_enhanced.py)**: Uses the complete moment set. Exact Lasserre bound.
- n_y = C(d+2k, 2k) moment variables
- Full moment PSD: n_basis x n_basis where n_basis = C(d+k, k)
- Works at d <= 8 for order=3, d <= 16 for order=2

**2. Sparse highd solver (lasserre_highd.py)**: Uses clique-restricted sparsity (Waki et al. 2006).
- Decomposes the d bins into overlapping cliques of size (bandwidth+1)
- Replaces full moment PSD with per-clique PSD cones (necessary condition)
- Uses reduced moment set S = {all degree <= 2k-1} union {clique degree-2k}
- Adds full M_1 PSD for cross-clique coupling
- Partial consistency (inequality not equality) for moments with children outside S
- Works at d=16-128

---

## Measured Results (All From Actual Runs)

### Full Solver (lasserre_enhanced, no sparsity)

| d | order | lb         | gap closure | time    | per_solve |
|---|-------|------------|-------------|---------|-----------|
| 4 | 2     | 1.07871    | 76.9%       | 0.8s    | 0.11s     |
| 4 | 3     | 1.10156    | 99.25%      | 7.6s    | 0.84s     |
| 6 | 2     | 1.12596    | 73.6%       | 5.5s    | 0.56s     |
| 6 | 3     | 1.17003    | 99.38%      | 211s    | 23.0s     |
| 8 | 2     | 1.15815    | 77.2%       | 81s     | 2.51s     |
| 8 | 3     | —          | —           | killed (>30min/round) | — |

**Key observation**: Order-3 achieves ~99% gap closure but is computationally infeasible at d >= 8 with full PSD.



### Enhanced Solver, Sparse Mode (lasserre_enhanced, psd_mode='sparse')

| d  | order | bw | per_solve | notes |
|----|-------|----|-----------|-------|
| 8  | 3     | 4  | 17.86s    | benchmark measured |
| 8  | 3     | 6  | ~90s (est)| benchmark was running when killed |

### McCormick/RLT Cuts and Simplex Valid Inequalities (tested at d=6,8)

| d | cuts added | delta lb | verdict |
|---|-----------|----------|---------|
| 6 | McCormick | -3.45e-6 | ZERO benefit (redundant with consistency) |
| 6 | Simplex   | -3.60e-5 | ZERO benefit (redundant with consistency) |
| 8 | McCormick | 0.00     | ZERO benefit |
| 8 | Simplex   | 0.00     | ZERO benefit |
| 8 | Both      | 0.00     | ZERO benefit |

These cuts are mathematically implied by order-2 full consistency + nonnegativity. They add nothing when consistency is already exact. They could only help in the sparse solver where consistency is partial.

### L2 Sweep Log (data/l2_sweep.log)

d=16 L2 CG: lb=1.1329, gap_closure=41.7%, time=2713.8s, RSS=10.2GB, active=40/496

---

## The Three Bottlenecks

### Bottleneck 1: Moment Variable Count (n_y)

MOSEK's interior-point method builds a Schur complement matrix of size n_y x n_y at each iteration. Cost per iteration: O(n_y^2 * total_PSD_dim).

| d   | order | n_y (full)  | n_y (highd reduced) |
|-----|-------|-------------|---------------------|
| 8   | 2     | 495         | 165                 |
| 8   | 3     | 3,003       | 1,287               |
| 16  | 2     | 4,845       | 969                 |
| 16  | 3     | 74,613      | 20,349              |
| 32  | 3     | 2,760,681   | 435,897             |
| 64  | 2     | 814,385     | 47,905              |
| 128 | 2     | 12,082,785  | 366,145             |

The highd reduced moment set cuts n_y significantly. The actual n_y depends on bandwidth (smaller bw = fewer clique monomials = smaller reduced set).

At d=128 order=2: Schur complement is 498K x 498K = ~34 GB. MOSEK can handle this in 256 GB RAM but each solve takes 30-60 minutes.

### Bottleneck 2: Binary Search Multiplier

The solver treats t as a PARAMETER and does binary search. Each CG round requires n_bisect (10-20) sequential SDP solves. With 3-15 CG rounds, total calls = 45-300.

The constraint L_W = t*M_{k-1}(y) - Q_W(y) >= 0 is LINEAR in (t, y) — it's a standard LMI. Making t a VARIABLE and minimizing directly would give the exact answer in 1 SDP solve per CG round.

The highd solver already does this for Round 0 (scalar-only optimization). But Rounds 1+ revert to bisection because the PSD window constraints involve t*y (bilinear). However, speedup.md proposes a Schur complement reformulation or a simpler hybrid: optimize for scalar rounds, bisect only for PSD rounds.

### Bottleneck 3: Bandwidth vs Gap Closure Tradeoff

This is the FUNDAMENTAL problem.

At d=16 order=3:

| bw | clique size | O3 basis | n_cliques | PSD cost ratio (vs bw=4) | est. gc |
|----|-------------|----------|-----------|--------------------------|---------|
| 4  | 5           | 56       | 12        | 1.0x                     | unmeasured |
| 6  | 7           | 120      | 10        | 8.2x                     | ~56%    |
| 8  | 9           | 220      | 8         | 40.4x                    | ~67%    |
| 10 | 11          | 364      | 6         | 137x                     | ~77%    |
| 12 | 13          | 560      | 4         | 333x                     | ~86%    |
| 14 | 15          | 816      | 2         | ~1000x                   | ~95%    |
| 15 | 16 (full)   | 969      | 1         | ~1500x                   | ~99%    |

To beat 1.2802 at d=16 we need 88% gc. The gc estimates above are extrapolated from the only solid data point: full (bw=d-1) at d=4,6 gives ~99% gc. No sparse O3 gc has been measured at d=16 yet. The actual gc at any bandwidth is unknown for d=16 O3.

---

## What Exists in the Codebase

### Solvers (4 implementations)
1. `lasserre/solvers.py` -> `solve_highd_sparse` (primary, d=64-128)
2. `lasserre/solvers.py` -> `solve_enhanced` (d=16-64, sparse/DSOS/BM modes)
3. `lasserre/solvers.py` -> `solve_cg` (full moment, d <= 32)
4. `lasserre/solvers.py` -> `solve_lasserre_fusion` (monolithic, d <= 16)

### Infrastructure
- `lasserre/core.py`: Monomial enumeration, Mersenne-prime hashing, window matrices
- `lasserre/precompute.py`: Index arrays, base constraints, violation checking
- `lasserre/cliques.py`: Banded clique decomposition, sparse PSD constraints

### Documented Optimizations (speedup.md, not yet implemented)
- Idea 0: Replace MOSEK with Clarabel/SCS/COSMO (avoids Schur complement entirely)
- Idea 1: Make t a variable, minimize directly (7x fewer SDP solves)
- Idea 2: MOSEK warm-start + incremental updates (3-5x per solve)
- Idea 3: Reduce bandwidth 16->10 (5-10x from smaller Schur)
- Idea 4: Batch violation checking with scalar pre-filter (5-10x per CG round)
- Idea 5: Moment set pruning via reachability (1.5-2x from n_y reduction)
- Combined estimate: 100-300 hours -> 30-90 minutes

### Cascade Prover (separate approach)
- `cloninger-steinerberger/cpu/run_cascade.py`: Branch-and-prune over discretized mass distributions
- 5 validated optimization ideas in `NEW.md` and `benefit.md`
- Infeasible at L3+ due to 10^15-10^35 children per parent

### Formal Proof
- `proof/lower_bound_proof.pdf`: LaTeX proof document (~400 pages)
- `lean/complete_proof.lean`: Lean 4 formalization (9K+ lines)

### External Resources
- RunPod API key in `.env` for GPU/CPU pod deployment
- Aristotle key for remote execution
- Requirements: numpy, numba, joblib, matplotlib, runpod, python-dotenv

---

## The Numbers That Define the Problem

**To beat 1.2802:**
- At d=16: need 88.0% gap closure. No sparse O3 data yet.
- At d=32: need 83.4%. No measured data at O3.
- At d=64: need 73.0%. No measured data at O3.
- At d=128: need 66.7%. No measured data at O3.

**What order-3 gives us (measured):**
- Full O3: ~99% gc (d=4,6 measured). Infeasible at d >= 8.
- Sparse O3: no gc measured yet at d >= 16. The gap between full and sparse is unknown.

**What bandwidth gives us:**
- No sparse O3 gc has been measured at d=16 at any bandwidth.
- bw=4,6,8,10,12,14 at d=16 O3: all unmeasured.
- Higher bw recovers gc but costs exponentially more compute.

**Timing (measured per-solve):**
- d=8 O2 full: 2.51s
- d=8 O3 sparse bw=4: 17.86s (benchmark measured via lasserre_enhanced sparse mode)
- d=6 O3 full: 23.0s
- d=128 O2 sparse bw=16: estimated 30-60 min/solve, ~300 solves needed = 100-300 hours
