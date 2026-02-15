  ---
  1. Gaps in Coverage

  FFT-based autoconvolution. The "lazy k_star" optimization (checking neighbors instead of full autoconv) suggests you're computing autoconvolution directly at O(P²). For P=1500+, switching to FFT-based computation (O(P   log P)) would give ~50-100x speedup at P=5000. This is likely the single largest unexploited speedup.

  LP iteration (MV10 method). MV10 got 1.5098 at P=208 using iterative LP — find the peak of f*f, solve an LP to reduce it, repeat. You tried Frank-Wolfe (R1-5) and LP-guided refinement (S2-10), both of which are weak  
  approximations to this. A faithful implementation with a modern interior-point LP solver (HiGHS, or Gurobi if available) at P=500+ is a fundamentally different optimization path that could find different basins.      

  Second-order methods on the LSE subproblem. During the LSE continuation phase, the objective is smooth. You're using gradient descent, but L-BFGS on the smoothed objective would converge in far fewer iterations,      
  leaving more budget for restarts. The Polyak phase is necessarily first-order, but the LSE phase doesn't have to be.

  Non-uniform grids / free-knot splines. All experiments use uniform bins. Making the bin edges free variables effectively gives you adaptive resolution. The CLAUDE.md flags this as a priority. A geometric grid
  concentrating near ±1/4 could match the effective resolution of P=5000 uniform bins with P=500 non-uniform.

  GPU-accelerated batch evaluation. The autoconvolution and its gradient are embarrassingly parallelizable. A JAX or PyTorch implementation could evaluate hundreds of candidates simultaneously on a single GPU, enabling 
  massive parallelism in the restart loop.

  Basin fingerprinting for diversity. You have strong evidence that the bottleneck is finding the right basin, but no method to characterize basins. Computing a fingerprint (e.g., the location and relative height of the   top 5 autoconvolution peaks) for each local minimum would let you explicitly enforce basin diversity across restarts, rather than hoping heavy-tailed initialization achieves it implicitly.

  ---
  2. Underexplored Winners

  Extreme Sparse Init found 1.51019 at P=200 (the record), but collapsed at P=500 (1.51372). This is the most interesting signal in the data — it found a genuinely different basin at moderate P but the
  upsampling/scaling failed. The question isn't whether sparse init works at P=500 directly; it's whether the P=200 basin it found can be upsampled more carefully (interpolation method, gentle refinement before
  aggressive Polyak).

  Iterated Warm Restart was the clear P=500 champion (1.50763 in 600s) and had 2.04x speedup from optimizations — the best of any method. Yet the cloud run at P=1500 used "warm_perturb," which may be a less tuned       
  version. Was iterated warm restart specifically run at P=1000-1500? If not, that's a direct gap.

  Elite Breeding had the lowest variance at P=200 (±0.0004) in the final comparison. Consistency matters enormously at high P where each restart is expensive. It was never tested at P=500+. Given that variance reduction   becomes more valuable as restart cost increases, this deserves a proper P=500 comparison.

  Double Polyak Polish (R2-6) was best at P=50 (1.5185) — perturbation + re-polish after convergence. This idea has a natural synergy with warm-starting at high P: converge → perturb → re-converge. It's essentially     
  iterated warm restart with a different perturbation strategy, and was never directly compared.

  ---
  3. Scaling Bottlenecks

  Your improvement curve is roughly:
  ┌──────┬────────┬─────────────────────┐
  │  P   │  Best  │     Δ per 2x P      │
  ├──────┼────────┼─────────────────────┤
  │ 200  │ 1.5102 │ —                   │
  ├──────┼────────┼─────────────────────┤
  │ 500  │ 1.5076 │ -0.0026             │
  ├──────┼────────┼─────────────────────┤
  │ 1000 │ 1.5057 │ -0.0019             │
  ├──────┼────────┼─────────────────────┤
  │ 1500 │ 1.5055 │ -0.0002 (only 1.5x) │
  └──────┴────────┴─────────────────────┘
  The P=1000→1500 gain is suspiciously small — either the returns are diminishing faster than log(P), or the P=1500 optimization didn't run long enough / with the right method. This is worth investigating: was the      
  P=1500 run given proportionally more budget?

  The most efficient path to P=2000-5000:

  1. FFT autoconvolution — converts the per-eval bottleneck from O(P²) to O(P log P). At P=3000, this is a ~200x theoretical speedup per eval.
  2. Progressive cascade from your existing P=1500 solution — you already have a good warm-start. Upsample to P=2000, run iterated warm restart (your best method) for 10-20 minutes, then cascade to P=3000.
  3. Parallelize restarts, not just cores — the Modal runs used 32 cores. With FFT+batch evaluation, you could run 100+ restarts per P level in the same wall-clock time.
  4. L-BFGS for the LSE phase — at P=3000, gradient descent on a 3000-dim smooth objective is wasteful. L-BFGS typically converges in 50-200 iterations where GD takes thousands.

  ---
  4. Concrete Next Steps (Ranked by Expected Impact)

  Experiment 1: FFT core + iterated warm restart, push to P=3000 (HIGH impact)

  Implement FFT-based autoconvolution (numpy.fft.rfft → pointwise square → irfft). Benchmark against direct computation at P=1000. Then run iterated warm restart (your best method) on a cascade: warm-start from P=1500  
  solution → P=2000 (15 min) → P=3000 (30 min). Expected result: ~1.503-1.504.

  Experiment 2: L-BFGS for the LSE phase (HIGH impact)

  Replace gradient descent in the LSE continuation phase with scipy.optimize.minimize(method='L-BFGS-B') using the analytic gradient. Keep Polyak subgradient for the polishing phase. Benchmark at P=500: how many more   
  restarts per budget? Expected: 3-5x more restarts → better basin exploration.

  Experiment 3: Sparse init → careful upsample → warm cascade (MEDIUM-HIGH impact)

  The P=200 sparse init basin (1.51019) was lost at P=500. Run 1000 extreme sparse init restarts at P=200 (should take <5 min). Take the top 20 diverse solutions (fingerprinted by peak locations). Upsample each to P=500   using linear interpolation (not nearest-neighbor). Run short LSE polish (not full restart) on each. Select best → cascade to P=1000. This tests whether sparse init basins can be preserved with gentler upsampling.    

  Experiment 4: Non-uniform grid with free knot positions (MEDIUM impact)

  Parameterize as (edges, heights) where edges are also optimized. Start with P=200 uniform, then let edges migrate. Use the LSE smoothed objective (which is smooth in edge positions too). This could find the boundary  
  behavior near ±1/4 that uniform grids miss. If the extremizer has singular behavior at the boundary, this will show a dramatic improvement.

  Experiment 5: LP iteration at P=500 (MEDIUM impact, different search space)

  Implement MV10's method faithfully: given current f, compute f*f, find the peak τ*, solve the LP "minimize (f*f)(τ*) subject to f≥0, ∫f=1" (this is a linear program in the bin heights since the peak location is       
  fixed). Update f, repeat. This explores a fundamentally different solution path from gradient methods and may find basins that LSE+Polyak misses entirely. Use an efficient LP solver (HiGHS via scipy.optimize.linprog).
  ---
  Bottom line: The biggest gains will come from (1) FFT acceleration to unlock P=3000+, and (2) L-BFGS on the smooth LSE phase to get more restarts per budget. These are infrastructure improvements, not new algorithms —   which aligns with your finding that the core method is already excellent and the bottleneck is search coverage (more restarts at higher P).