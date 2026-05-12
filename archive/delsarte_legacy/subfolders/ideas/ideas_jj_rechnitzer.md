# Transfer of JJ-2025 and Rechnitzer-2026 techniques to L^infty (C_{1a})

**Sources.** Jaech-Joseph, arXiv:2508.02803 (c >= 0.94136). Rechnitzer, arXiv:2602.07292 (128 digits of nu_2^2).

## 1. Jaech-Joseph upsampling algorithm

**Setup.** Piecewise-constant f on [-1/4, 1/4] with N heights h in R^N. Objective

    C(h) = ||f*f||_{L^2}^2 / ( ||f*f||_{L^1} * ||f*f||_{L^infty} )

maximised by gradient ascent (GPU, A100, under 10 min).

**Pipeline.**
1. Phase 1 exploration: 30k iterations, batch B=1024, lr=3e-2, Gaussian noise eta=1e-3, anneal gamma=0.65.
2. Phase 2 exploitation: 70k iter, lr=5e-3, no noise.
3. Elitist respawn every 20k: keep top 50%.
4. **4x upsample**: linear interpolation from N=559 heights to ~8944 heights (iterated doubling via `np.interp`).
5. Phase 4 high-res refine: 200k iterations, lr=3e-2.

Autoconvolution via `numpy.convolve` (FFT), Simpson's rule for L^2, Riemann for L^1, max entry for L^infty. **No rigorous certificate** - purely numerical.

**Transfer to L^infty (C_{1a})?** Partial.

- **Upsampling step itself is trivially transferable.** It is just "interpolate heights to a finer grid, then re-run local search." Our cascade/Lasserre pipelines both admit this: start from d=8 optimizer mu* in R^8, linearly interpolate to d=32 or d=64, then refine via the existing QP/SDP.
- **Objective mismatch matters.** C_{1a} minimises max_W mu^T M_W mu - a **sup-norm over W** that is non-smooth. JJ's gradient ascent on a smooth ratio does not directly port. One must replace max by a soft-max (log-sum-exp) for gradient computation, or use subgradient / bundle methods. We already partly do this in `lasserre/` via dualization to SDP.
- **The 4x upsample is essentially a warm-start heuristic.** Value: cheap way to seed large-d Lasserre/cascade runs from small-d optima. This is the main takeaway.
- **Not a rigour technique.** JJ do not certify c >= 0.94136 rigorously - they only evaluate the ratio numerically. Our `certified_lasserre/` Farkas pipeline is strictly stronger in the rigour dimension.

## 2. Rechnitzer 128-digit interval arithmetic

**Setup.** Parameterise f by the Matolcsi-Vinuesa / Cilleruelo ansatz

    f(x) = sum_{j=0}^{P-1} a_j * C(1/2, j) * (1 - 4x^2)^{j-1/2}

with P=101 coefficients. Fourier transform gives Bessel-series representation F(k); autoconvolution norm becomes a multinomial in (a_j) with Bessel-integral coefficients J(p,k) = J_p(pi k / 2) * p! * (4/(pi k))^p.

**Rigorous evaluation** at precision 384 decimal digits using **FLINT / Arb ball arithmetic** (arbitrary-precision floats with certified error balls).

1. **Upper bound** on nu_2^2: for a given a, evaluate C(a) rigorously via
   - exact finite sum for k <= N (with N = 8192),
   - asymptotic Kummer-transform tail sum for k > N (closed form),
   - certified truncation error.
2. **Lower bound**: Hoelder inequality + dual function G whose Fourier coefficients are matched to F^3 by asymptotic fitting. Coefficients determined systematically, then verified rigorously.
3. Optimization over a in R^101 done in standard floats; the rigorous wrapper only certifies the evaluation at the optimum.

**Runtime.** Seconds to tens of seconds per evaluation on a laptop. Projected scaling: 200 digits needs P ~= 160, 1000 digits needs P ~= 830, with log10(gap) ~ -7.5 - 1.2*P.

**Transfer to L^infty (C_{1a})?** Yes, but the harder part is the analytic dual.

- **Ball arithmetic (Arb/FLINT) transfers verbatim.** Our Lasserre dual certificate evaluation (Farkas multipliers, SOS weights) can be wrapped in `python-flint` or `arb` for rigorous evaluation, replacing mpmath. Gives us unlimited precision at ~negligible cost.
- **The ansatz trick is the clever part.** Rechnitzer's lower bound uses a Hoelder-dual function G with a Fourier ansatz matching F^3. Analog for C_{1a}: max_t (f*f)(t) can be attacked via a **measure dual** - lower-bound C_{1a} by exhibiting a specific kernel K and showing int K * (f*f) >= C for all admissible f. This is the classical Matolcsi-Vinuesa dual we already have in `delsarte_dual/mv_*.py`.
- **Missing ingredient for L^infty.** JJ/Rechnitzer both optimise a single functional (L^2 norm of f*f). C_{1a} has an inner sup_t that produces a **semi-infinite LP / SDP**. Rechnitzer-style closed-form Bessel evaluation does not handle this directly.
- **Most promising hybrid.** Use Rechnitzer's ansatz family `(1 - 4x^2)^{j-1/2}` as a primal restriction, then enforce sup_t via Chebyshev / trigonometric polynomial discretization of t, solve the Lasserre SDP, and certify with Arb ball arithmetic. This is close to what `tests/lasserre_mosek_cheby.py` already does - Arb wrapping would upgrade our numerical dual certificates to 128+ digits.

## 3. Numerical details side-by-side

| Item | Jaech-Joseph | Rechnitzer |
|---|---|---|
| Discretization | 2399 step heights (559 * 4 upsample) | 101-coefficient Bessel ansatz, N=8192 Fourier cutoff |
| Precision | float64 | 384 decimal digits (ball arithmetic) |
| Runtime | < 10 min A100 GPU | seconds-to-minutes laptop |
| Certificate | none (numerical) | rigorous (Arb balls + Hoelder dual) |
| Result | c >= 0.94136 | 128 digits of nu_2^2 |

## 4. Adaptations needed for C_{1a} (sup-norm)

1. **Handle inner sup_t.** L^infty norm of f*f is non-smooth. Options:
   - Discretize t on a Chebyshev grid W_1,...,W_m, giving max_i mu^T M_{W_i} mu (what cascade and Lasserre already do).
   - Log-sum-exp smoothing for gradient methods (JJ-style), then cool temperature.
   - Semi-infinite SDP dual (current Lasserre + sublevel approach).
2. **Replace float gradient ascent with subgradient / bundle.** JJ's pipeline assumes a smooth objective; add subgradient for max_i, or use SDP primal-dual as we already do.
3. **Wrap the Lasserre/Farkas dual in Arb.** The numerical Farkas multipliers from `certified_lasserre/` can be rounded-to-rational and then evaluated rigorously in Arb at 128+ digits. This upgrades val(d) lower bounds to fully certified digits.
4. **Warm-start at high d.** Apply JJ's linear upsample to the d=16 Lasserre optimiser to seed d=64 runs.
5. **Primal ansatz via Bessel basis.** Port Rechnitzer's `(1-4x^2)^{j-1/2}` family into a Lasserre moment problem; the ansatz dramatically shrinks the SDP and may tighten the bound.

## 5. Likely lift

- **JJ upsampling alone:** low lift. It is a standard warm-start, and our cascade/Lasserre pipelines are not primarily limited by local-optimum quality - they are limited by rigour/dimension. Saves wall-time, no bound improvement.
- **Rechnitzer Arb ball arithmetic:** moderate lift, high confidence. Direct path to rigorous 50-128 digit Lasserre certificates without dependence on Mosek/Clarabel tolerances. Estimated: val(4) >= 1.0963... extended to many digits; val(8), val(16) similarly upgraded. No change to the bound C_{1a} >= 1.2802 by itself, but removes the "is our dual numerically faithful?" question entirely.
- **Rechnitzer ansatz + sup_t discretization + Arb wrap (hybrid):** highest potential lift. Rechnitzer's constant for L^2 beat Matolcsi-Vinuesa by ~0.01. Porting the same ansatz to our sup-norm Lasserre formulation plausibly yields an improvement of order 1e-3 to 1e-2 in the L^infty lower bound - enough to push past 1.2802 if combined with sublevel/Z2 symmetry. **This is the path most likely to beat the Cloninger-Steinerberger lower bound.**
- **Effort estimate.** Arb wrap: 1-2 days (python-flint). Bessel-ansatz Lasserre rewrite: 1-2 weeks. Total expected: 2-3 weeks for the hybrid, with realistic chance of bound improvement.

## 100-word summary

Jaech-Joseph (2508.02803) maximise the L^2 autoconvolution ratio by gradient ascent on a 2399-step function, built via 4x linear-interpolation upsampling from a 559-interval optimizer; no rigorous certificate. Rechnitzer (2602.07292) gets 128 digits of nu_2^2 using a 101-term Bessel ansatz f ~ sum a_j (1-4x^2)^{j-1/2}, evaluating the autoconvolution norm rigorously via FLINT/Arb ball arithmetic at 384-digit precision, with Hoelder duality supplying the lower bound. For C_{1a} the upsampling transfers as a cheap warm-start; Arb wrapping directly upgrades our Farkas-certified Lasserre duals to unlimited precision; the highest-payoff move is porting Rechnitzer's Bessel ansatz into the sup_t Lasserre SDP with Arb certification - 2-3 weeks effort, realistic chance of beating 1.2802.
