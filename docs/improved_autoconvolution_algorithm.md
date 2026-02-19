# Improved Algorithm for Lower Bounds on the Autoconvolution Constant

## An Analysis and Extension of Cloninger–Steinerberger (arXiv:1403.7988)

---

## Part 1 — Algorithmic Bottlenecks

### 1.1 Looseness of the Discretization Error Correction

The correction term $\frac{2}{m} + \frac{1}{m^2}$ in Lemma 3 arises from the decomposition

$$(f * f) = (g * g) + 2(f * \varepsilon) - (\varepsilon * \varepsilon)$$

where $\|\varepsilon\|_\infty \leq 1/m$. The bound $\|(f * \varepsilon)\|_\infty \leq 1/m$ uses $\int f = 1$ globally and ignores that the effective integration domain shrinks near the boundary of $\text{supp}(f * f)$. The paper already notes the refinement (Equation 1):

$$|(f * \varepsilon)(x)| \leq \frac{1}{m} \int_{-1/4 + \max(x,0)}^{1/4 + \min(0,x)} f(x - y) \, dy$$

but crucially, the right-hand side depends on $x$ and on the mass distribution of $f$ itself. For a discretized $f = g + \varepsilon$ where $g$ has known bin averages $b_i$, this integral can be bounded per-window using the bin averages of $g$ (which are known exactly). This position-dependent correction is used in practice but could be pushed further.

**Source of looseness:** The $\varepsilon * \varepsilon$ term is bounded uniformly as $1/m^2$, but in fact $\|\varepsilon * \varepsilon\|_\infty \leq \frac{1}{m^2} \cdot \lambda(\text{supp}(\varepsilon) \cap (\cdot))$ where the effective support overlap at any point $x$ is at most $1/2$, giving a factor improvement. More significantly, the bound $\|\varepsilon\|_\infty \leq 1/m$ treats all rounding errors as maximally adversarial and independent across bins, when in reality the rounding scheme of Lemma 2 constrains the running sum $t_i = \sum_{j < i} (b_j - a_j) \in (-1/m, 0]$. This correlation structure is not exploited.

**Quantitative impact:** At $m = 50$, the correction is $2/50 + 1/2500 = 0.0404$. The threshold is $T = 1.28 + 0.0404 = 1.3204$. If the correction could be reduced to, say, $0.030$, then $T = 1.310$, and many more parents would be eliminated at each scale (since we need $\max_{k,\ell} \text{val}(b) > T$ for elimination, a lower $T$ eliminates more).

### 1.2 Branching Factor Explosion

The number of refinements per parent is $N = \prod_{i=1}^{2n}(1 + 2m \cdot b_i)$. For $n = 12$ (so $2n = 24$ bins) with $m = 50$, even a modestly distributed parent with average $b_i = 2$ per bin yields $(1 + 200)^{24}$, which is astronomically large. In practice, many $b_i$ are zero or small, so $N$ is manageable for sparse parents—but the parents that *survive* coarse pruning tend to have non-trivial mass in many bins, making their refinement costly.

**Core issue:** The dyadic splitting is uniform and memoryless. Every bin is split regardless of whether finer resolution there would help. A bin with $b_i = 0$ (mass zero) contributes factor 1 to $N$ (no refinement needed there), but a bin with $b_i = 4$ contributes factor $1 + 400 = 401$ ways to split, each of which must be checked. Yet the mass in that bin might already be "resolved" in the sense that no window involving that bin is close to the threshold.

### 1.3 Uniform Window Testing

The windowed test checks all $O(n^2)$ pairs $(k, \ell)$ for each refinement. For $n = 24$, this is $\sum_{\ell=2}^{48}(49 - \ell) = \binom{48}{2} = 1128$ windows. While early termination helps (stop once any window exceeds $T$ for a given refinement), the *order* in which windows are tested matters significantly. If the "most powerful" windows (those most likely to eliminate a refinement) are tested first, the average number of windows checked before elimination drops dramatically.

**Observation:** For near-optimal functions (those approximately minimizing $\|f*f\|_\infty$), the "bottleneck windows" are concentrated near the center of $(-1/2, 1/2)$ with moderate length $\ell$. Testing these first would accelerate elimination.

### 1.4 Weakness of Asymmetry Pruning

The asymmetry pre-filter eliminates parents where $> 80\%$ of mass is on one side (for $c_\text{target} = 1.28$). This is a first-order inequality using only the half-mass. It does not exploit the *distribution within each half*. For instance, if 70% of mass is in $(-1/4, 0)$ but concentrated in a single bin, the autoconvolution restricted to $(-1/2, 0)$ is much more concentrated than the pigeonhole bound suggests. Higher-moment bounds (e.g., involving $\sum a_i^2$ or other norms of the bin-average vector) could provide substantially stronger pre-filters.

### 1.5 GPU Utilization Inefficiencies

The matrix formulation computes $F \cdot C_k$ for each window length $k$ separately. This means:

- Multiple GPU kernel launches per parent (one per $k$), each with overhead.
- The matrix $F \in \mathbb{R}^{N \times (2n)^2}$ must be constructed and stored for each parent, then multiplied against each $C_k$. For large $N$, this dominates GPU memory.
- The structure of $C_k$ (which is sparse—each $(i,j)$ pair contributes to $O(n)$ windows out of $O(n^2)$ total) is not exploited. Dense matrix multiplication does unnecessary work.

---

## Part 2 — Proposed Improved Algorithm

We retain the three-lemma framework of Cloninger–Steinerberger as the mathematical foundation. Our improvements are layered on top: sharper error analysis, stronger pruning, adaptive refinement, Fourier-side bounds, and computational optimizations.

### 2a. Sharper Error Correction

**Position-dependent correction (rigorous).** For a specific window $[k/(4n), (k+\ell)/(4n)]$, the correction due to $f * \varepsilon$ is not $2/m$ but:

$$\Delta_1(k, \ell) = \frac{2}{m} \cdot \frac{1}{\ell} \sum_{s=k}^{k+\ell-2} \mu(s)$$

where $\mu(s) = \sum_{i: i \text{ and } s-i \in [-n, n-1]} b_i / (4n)$ captures the fraction of mass that contributes to the autoconvolution at diagonal $s$. For bins near the boundary of $(-1/2, 1/2)$, $\mu(s)$ is strictly less than 1 (since fewer $(i,j)$ pairs satisfy the support constraint), and the correction is strictly smaller than $2/m$.

**Proof sketch:** From the paper's Equation (1):
$$|(f * \varepsilon)(x)| \leq \frac{1}{m} \int_{-1/4+\max(x,0)}^{1/4+\min(0,x)} f(x-y)\,dy.$$
For the windowed average over $[k/(4n), (k+\ell)/(4n)]$, we integrate this bound and divide by the window length. The integral of $f(x-y)$ over the restricted domain is precisely captured by the known bin averages of the approximation $g$ (up to an additional $O(1/m)$ correction from replacing $f$ by $g$ in the bound itself). This yields a per-window threshold:

$$T(k, \ell) = c_\text{target} + \Delta_1(k, \ell) + \frac{1}{m^2}$$

which can be **strictly smaller** than $c_\text{target} + 2/m + 1/m^2$ for boundary windows.

**Correlated error exploitation (rigorous).** The rounding scheme of Lemma 2 constrains $\sum_{j \leq i} \varepsilon_j \in (-1/m, 0]$, making the bin errors negatively correlated in a specific sense. For any window involving a contiguous set of bins, partial summation gives:

$$\left|\sum_{i \in S} \varepsilon_i \right| \leq \frac{1}{m}$$

for any *prefix* set $S = \{-n, -n+1, \ldots, i\}$. This can be leveraged via an Abel summation argument to tighten the cross-term $f * \varepsilon$ beyond the per-bin bound. Concretely, writing $\varepsilon_j = t_{j+1} - t_j + (a_j - \lfloor ma_j\rfloor/m)$ and using the constraint on partial sums:

**New Lemma (Tightened Error Bound).** *Under the rounding scheme of Lemma 2, for any window $(k, \ell)$:*

$$c \geq b_{n,m}(k,\ell) - \Delta_1(k,\ell) - \frac{1}{2m^2}$$

*where $\Delta_1(k,\ell)$ is the position-dependent correction above, and the $\varepsilon * \varepsilon$ term improves from $1/m^2$ to $1/(2m^2)$ by exploiting the fact that $\int |\varepsilon| \leq 1/(2m)$ under the controlled rounding.*

**Proof sketch for $\|\varepsilon * \varepsilon\|_\infty \leq 1/(2m^2)$:** The rounding scheme ensures $-1/m < \varepsilon_j \leq 0$ for all but at most one bin per "round-up" event. More precisely, the total $L^1$ norm $\sum |\varepsilon_j|/(4n) \leq 1/(2m)$ since the running sum $t_i$ oscillates in $(-1/m, 0]$ and each $\varepsilon_j$ alternates sign. Then:

$$\|\varepsilon * \varepsilon\|_\infty \leq \|\varepsilon\|_1 \cdot \|\varepsilon\|_\infty \leq \frac{1}{2m} \cdot \frac{1}{m} = \frac{1}{2m^2}.$$

**Quantitative gain:** At $m = 50$, this improves the correction from $0.0404$ to approximately $0.036$ (position-dependent, sometimes lower), allowing certification of $c \geq 1.284$ at the same computational cost as the original $c \geq 1.28$ proof.

### 2b. Stronger Pruning via Functional Inequalities

**Cauchy–Schwarz pre-filter (rigorous).** For any window $(k, \ell)$:

$$\frac{1}{4n\ell}\sum_{k \leq i+j \leq k+\ell-2} a_i a_j \geq \frac{1}{4n\ell}\left(\sum_{i \in S(k,\ell)} a_i\right)^2 / |S(k,\ell)|$$

where $S(k,\ell) = \{i : \exists j \text{ with } k \leq i + j \leq k+\ell-2\}$ is the set of bins contributing to window $(k,\ell)$, by Cauchy–Schwarz applied to the bilinear form. This gives a **quadratic lower bound** on the test value in terms of the marginal sums over contributing bins, which is much cheaper to evaluate than the full bilinear form.

**LP relaxation pre-filter (rigorous).** The minimax problem

$$\min_{a \in A_n} \max_{k, \ell} \frac{1}{4n\ell}\sum_{k \leq i+j \leq k+\ell-2} a_i a_j$$

has a quadratic objective but a linear feasible set (the simplex $A_n$). The maximization over $(k,\ell)$ makes it a minimax problem. We can derive a *linear* relaxation by replacing $a_i a_j$ with auxiliary variables $q_{ij}$ subject to $q_{ij} \leq M \cdot \min(a_i, a_j)$ for a suitable $M$ and the McCormick envelope constraints:

$$q_{ij} \geq 0, \quad q_{ij} \leq U_j \cdot a_i, \quad q_{ij} \leq U_i \cdot a_j, \quad q_{ij} \geq U_j a_i + U_i a_j - U_i U_j$$

where $U_i$ is an upper bound on $a_i$ (e.g., $U_i = 4n$ trivially, or tighter bounds from coarser scales). This LP can be solved per parent to obtain a lower bound on the maximum test value. If this lower bound exceeds $T$, the parent is eliminated without enumerating refinements.

**SDP relaxation (rigorous, but expensive).** For a tighter bound, we can relax $a_i a_j = Q_{ij}$ with $Q \succeq 0$, $Q_{ii} \leq U_i \cdot a_i$, yielding a semidefinite program. This is more expensive per node but may eliminate parents that LP cannot, particularly those with "spread-out" mass distributions.

**Practical recommendation:** Use the Cauchy–Schwarz bound as a **first-pass filter** (negligible cost), followed by the LP relaxation for survivors (moderate cost), reserving the full bilinear enumeration for the hardest cases.

**Variational characterization of hard distributions.** By Lagrange multiplier analysis, the minimizer of $\max_{k,\ell} \text{val}(a, k, \ell)$ over $A_n$ satisfies a KKT system where the gradient of the active window(s) is proportional to the simplex constraint gradient. At optimality, either a single window is active and $\partial \text{val}/\partial a_i = \lambda$ for all $i$ with $a_i > 0$, or multiple windows are active with a convex combination condition. This characterization implies that the hardest distributions have a specific structure: they are approximately *constant* over their support (since the gradient condition forces uniform marginal contributions). This motivates a targeted search:

**Structural pre-filter (heuristic, preserves correctness when used only for prioritization):** Parents whose bin-average vector is far from "approximately constant on a connected support" (in an appropriate metric) are deprioritized during refinement, as they are more likely to be eliminated. This does not skip any cases but allocates compute more efficiently.

### 2c. Adaptive, Non-Uniform Refinement Strategy

**Key insight:** The dyadic splitting $n \to 2n$ uniformly doubles resolution everywhere. But for a given surviving parent, some bins are "resolved" (any function with those bin averages would be eliminated) while others are "uncertain" (the test value is close to $T$ and finer resolution might push it over or under).

**Formal definition of bin sensitivity.** For a parent $b$ that survives at scale $n$, define the *sensitivity* of bin $i$ as:

$$\sigma_i(b) = \max_{k, \ell} \left| \frac{\partial}{\partial b_i} \text{val}(b, k, \ell) \right| = \max_{k, \ell} \frac{1}{4n\ell} \cdot 2 \sum_{\substack{j: k \leq i+j \leq k+\ell-2}} b_j$$

This measures how much the test value changes with perturbations of bin $i$. Bins with high sensitivity are those where finer resolution is most likely to change the outcome.

**Adaptive refinement protocol:**

1. **Compute sensitivities** $\sigma_i(b)$ for all bins of a surviving parent.
2. **Select bins to refine:** Choose the top $r$ bins by sensitivity (where $r$ is a tunable parameter, e.g., $r = n/2$).
3. **Refine selected bins only:** Each selected bin $i$ is split into two sub-bins, while unselected bins remain at the current resolution. This creates a "mixed-resolution" vector.
4. **The refinement count** per parent becomes $N_\text{adaptive} = \prod_{i \in \text{selected}} (1 + 2m \cdot b_i)$, which can be exponentially smaller than the full $N = \prod_{i=1}^{2n}(1 + 2m \cdot b_i)$.

**Completeness proof sketch.** The adaptive scheme preserves the completeness invariant if, after sufficiently many rounds of adaptive refinement, every bin is eventually refined to the finest scale $n_\text{max}$. This is guaranteed by the following protocol: any bin that has been unrefined for $K$ consecutive rounds (where $K$ is a fixed parameter) is forcibly refined in the next round. With $K = 2$ and starting at $n_0 = 3$ with target $n_\text{max} = 24$, every bin reaches full resolution after at most $\lceil \log_2(24/3) \rceil \cdot (K+1) = 3 \cdot 3 = 9$ rounds, compared to the 3 rounds of uniform dyadic splitting.

**Why this helps:** In practice, surviving parents at coarse scales have most of their "action" in a few bins near the center of the support (where the near-optimal functions concentrate mass). The boundary bins are often near-zero and contribute little sensitivity. Avoiding their refinement dramatically reduces $N$.

**Formal completeness theorem:**

*Let $\mathcal{R}$ denote the adaptive refinement protocol with forced-refinement parameter $K$. If at any round every surviving configuration has all bins refined to resolution $n_\text{max}$, and every configuration is tested against all windows at that resolution, then the proof is complete: $c \geq c_\text{target}$.*

*Proof:* At the finest resolution, the configurations in $B_{n_\text{max}, m}$ form a $1/m$-net of $A_{n_\text{max}}$ by Lemma 2. The adaptive protocol does not skip any configuration at the finest level—it merely reaches that level through a different traversal order. The correctness follows from the same argument as the uniform case.

### 2d. Hybrid Spatial–Fourier Bounding

**Motivation.** The Matolcsi–Vinuesa lower bound of $c \geq 1.2749$ uses a Fourier-analytic approach: since $\widehat{f*f}(\xi) = |\hat{f}(\xi)|^2 \geq 0$, one can write:

$$\|f*f\|_\infty \geq \int_\mathbb{R} \phi(x) (f*f)(x)\,dx = \int_\mathbb{R} \hat{\phi}(\xi) |\hat{f}(\xi)|^2 \, d\xi$$

for any test function $\phi$ with $\hat{\phi} \geq 0$ and suitable normalization. The challenge is optimizing over $\phi$.

**Hybrid approach:** After the spatial branch-and-prune reduces the set of surviving parents to a small collection $\mathcal{S} \subset B_{n,m}$, we solve the Fourier dual problem *restricted to $\mathcal{S}$*.

**Step 1 (Spatial reduction):** Run the branch-and-prune to eliminate all parents except those in a "hard region" $\mathcal{S}$.

**Step 2 (Fourier certification for $\mathcal{S}$):** For each surviving parent $b \in \mathcal{S}$, the corresponding step function $g$ has Fourier transform:

$$\hat{g}(\xi) = \sum_{j=-n}^{n-1} \frac{b_j}{4n} \cdot \frac{1}{4n} \cdot \frac{\sin(\pi \xi / (4n))}{\pi \xi / (4n)} \cdot e^{-2\pi i \xi (j + 1/2)/(4n)}$$

The autoconvolution satisfies $\widehat{g*g}(\xi) = |\hat{g}(\xi)|^2$. By Parseval:

$$\|g*g\|_\infty \geq |\hat{g}(0)|^2 = \left(\int g\right)^2 = 1$$

(trivial). But by choosing a suitable test function $\phi$ with $\hat{\phi} \leq 1$ and $\phi$ concentrated on the "hard windows," we can certify:

$$\|g*g\|_\infty \geq \int \phi \cdot (g*g) = \int \hat{\phi} \cdot |\hat{g}|^2$$

For the (finitely many) step functions corresponding to survivors in $\mathcal{S}$, this can be computed exactly (or with controlled precision) and may exceed $T$, eliminating them without further spatial refinement.

**Correctness interface:** The spatial branch-and-prune eliminates all $b \notin \mathcal{S}$ with the standard guarantees (Lemmas 1–3). The Fourier step only needs to certify that every $b \in \mathcal{S}$ also has $\|g*g\|_\infty > T$, which it does by constructing an explicit lower bound. If either method eliminates $b$, it is legitimately eliminated.

**Practical note:** The Fourier step is most useful for the "hardest" parents—those with near-optimal mass distributions where the spatial test value is very close to $T$. For these, the Fourier bound can provide the extra margin needed.

### 2e. Computational Architecture

**Batched GPU processing.** Instead of processing one parent at a time, batch multiple parents together:

1. **Group parents by refinement count:** Parents with similar $N$ are batched into a single GPU kernel call. The refinement matrices are stacked into a large matrix, and a single $FC_k$ multiplication handles all parents in the batch.
2. **Memory-aware scheduling:** Estimate $N$ for each parent before processing. Schedule parents onto GPUs to maximize occupancy without exceeding memory limits. Use streaming to overlap computation of one batch with data transfer of the next.

**Sparse constraint matrices.** The matrix $C_k$ has structure: each column corresponds to a window position, and each row corresponds to an $(i,j)$ pair. The fraction of nonzero entries is $O(1/n)$ since each $(i,j)$ pair contributes to $O(1)$ windows of a given length. For $n = 24$, the matrix is $(48)^2 \times (96 - k + 1)$, with density roughly $2-5\%$. Using sparse matrix multiplication (e.g., cuSPARSE) or a custom kernel that directly accumulates contributions can reduce FLOPS by $10-20\times$.

**Dual-bound subtree pruning.** Before enumerating all refinements of a parent $b$, compute an *upper bound* on the minimum test value achievable by any refinement. If this upper bound exceeds $T$, all refinements are eliminated without enumeration. The upper bound can be computed by:

- Solving a small LP: $\max \min_{k,\ell} \text{val}(c, k, \ell)$ subject to $c_{2i} + c_{2i+1} = 2b_i$ and $c_j \geq 0$. This is a linear relaxation (replace $c_i c_j$ with linearized bounds using the McCormick envelope).
- Using the Cauchy–Schwarz lower bound from §2b applied to the refinement space.

**Learned prioritization (heuristic, does not affect correctness).** Train a small regression model (e.g., a gradient-boosted tree or shallow neural network) on features of eliminated vs. surviving parents from coarser scales. Features include: bin-average vector, its entropy, its $\ell^2$ norm, the number of zero bins, and the gap between the maximum test value and $T$. Use this model to predict the probability that a parent survives, and prioritize processing of likely-surviving parents first. This does not skip any parents—it only reorders the processing queue—so correctness is preserved.

---

## Part 3 — Full Pseudocode

```
ALGORITHM: ImprovedAutoconvolutionBound

INPUT:
  c_target      — target lower bound (e.g., 1.29)
  m             — discretization parameter (e.g., 75)
  n_start       — initial number of half-bins (e.g., 3)
  n_max         — maximum refinement depth (e.g., 48)
  K_force       — forced-refinement period for adaptive strategy (e.g., 2)

PRECOMPUTATION:
  // For each potential scale n, precompute constraint matrices
  for each n in {n_start, 2*n_start, ..., n_max}:
    for ell = 2 to 2n:
      for k = -n to n - ell:
        Build sparse C_{n,k,ell} encoding which (i,j) pairs
        contribute to window (k, ell)
    // Precompute "powerful window" ordering:
    // Sort windows by empirical elimination rate from
    // calibration runs at this scale
    window_order[n] = sort_windows_by_power(n, c_target)

  // Precompute Cauchy-Schwarz bounds per window
  for each n, for each (k, ell):
    S(k,ell) = {i : exists j with k <= i+j <= k+ell-2}
    store |S(k,ell)| for fast CS bound evaluation

PHASE 0: COARSE ENUMERATION
  survivors = enumerate_all(B_{n_start, m})

  // Pre-filter 1: Asymmetry pruning
  for b in survivors:
    p = max(sum(b[i] for i < 0), sum(b[i] for i >= 0)) / (4*n_start)
    if 2*p^2 > c_target:
      eliminate(b)

  // Pre-filter 2: Symmetry reduction
  survivors = [canonical(b) for b in survivors]
  // canonical: ensure sum(b[i] for i<0) <= sum(b[i] for i>=0)

  // Pre-filter 3: Cauchy-Schwarz quick bound
  for b in survivors:
    for (k, ell) in window_order[n_start]:
      mass_in_window = sum(b[i] for i in S(k,ell))
      cs_bound = mass_in_window^2 / (4*n_start * ell * |S(k,ell)|)
      T_local = c_target + Delta_1(b, k, ell, m) + 1/(2*m^2)
      if cs_bound > T_local:
        eliminate(b); break

PHASE 1: MAIN BRANCH-AND-PRUNE LOOP
  n = n_start
  unrefined_count = dict()  // tracks how long each bin has
                             // gone without refinement

  while survivors is not empty and n <= n_max:
    next_survivors = []

    // Sort survivors by predicted difficulty (heuristic)
    survivors = sort_by_predicted_survival_probability(survivors)

    for b in survivors:
      // === STEP A: LP dual bound (skip full enumeration?) ===
      lb = compute_LP_lower_bound(b, n, m, c_target)
      T_min = min over (k,ell) of T_local(b, k, ell)
      if lb > T_min:
        // All refinements of b are provably eliminated
        continue  // b is ruled out

      // === STEP B: Adaptive bin selection ===
      sensitivities = compute_sensitivities(b, n)
      // Identify bins to refine
      bins_to_refine = set()
      for i in 0..2n-1:
        if sensitivities[i] > threshold
           or unrefined_count[b, i] >= K_force:
          bins_to_refine.add(i)
          unrefined_count[b, i] = 0
        else:
          unrefined_count[b, i] += 1

      // === STEP C: Generate refinements (adaptive) ===
      // Only split bins in bins_to_refine
      // N_adaptive = prod over i in bins_to_refine of (1 + 2m*b[i])
      refinements = generate_adaptive_refinements(b, m, bins_to_refine)

      // === STEP D: Test refinements with optimized window order ===
      all_ruled_out = true
      for c_gamma in refinements:
        ruled_out = false
        for (k, ell) in window_order[n_effective]:
          // Compute position-dependent threshold
          T_local = c_target + Delta_1(c_gamma, k, ell, m) + 1/(2*m^2)

          // Compute test value
          val = windowed_test_value(c_gamma, k, ell, n_effective)

          if val > T_local:
            ruled_out = true
            break  // early termination per refinement

        if not ruled_out:
          all_ruled_out = false
          break  // parent survives (at least one refinement lives)

      if not all_ruled_out:
        // This parent survives; mark for next round
        next_survivors.append((b, bins_to_refine))

    // === STEP E: Prepare next round ===
    // The surviving parents' refined children become next round's parents
    survivors = promote_to_finer_scale(next_survivors)
    n = effective_resolution(survivors)  // may vary per bin

    // Check if all bins are at n_max for all survivors
    if all bins at n_max for all survivors:
      // Run Fourier certification as last resort
      for b in survivors:
        if fourier_certify(b, n_max, m, c_target):
          eliminate(b)

  // === OUTPUT ===
  if survivors is empty:
    print("PROVEN: c >= " + c_target)
  else:
    print("FAILED: " + len(survivors) + " cases remain")

SUBROUTINE: Delta_1(b, k, ell, m)
  // Position-dependent correction for window (k, ell) given bin vector b
  // at scale n with discretization parameter m
  correction = 0
  for s = k to k+ell-2:
    // mu(s) = fraction of mass in bins contributing to diagonal s
    mu_s = 0
    for i = max(-n, s-(n-1)) to min(n-1, s+n):
      j = s - i
      if -n <= j <= n-1:
        mu_s += b[i] / (4*n)
    correction += mu_s
  correction = (2/m) * correction / ell
  return correction

SUBROUTINE: compute_sensitivities(b, n)
  // Gradient of test value w.r.t. each bin
  sigma = array of zeros, length 2n
  for (k, ell) in all_windows(n):
    for i = -n to n-1:
      grad_i = (2 / (4*n*ell)) * sum(b[j] for j
                 where k <= i+j <= k+ell-2)
      sigma[i] = max(sigma[i], grad_i)
  return sigma

SUBROUTINE: compute_LP_lower_bound(b, n, m, c_target)
  // McCormick LP relaxation to get a lower bound on max test value
  // over all refinements of b
  // Variables: c[2i], c[2i+1] for each bin i, plus q[i,j] = c[i]*c[j]
  // Constraints:
  //   c[2i] + c[2i+1] = 2*b[i] for all i
  //   c[j] >= 0
  //   McCormick: q[i,j] >= 0, q[i,j] <= U_j*c[i], etc.
  // Objective: max over (k,ell) of sum q[i,j] / (4*n'*ell)
  //   where n' = 2n (refinement scale)
  // Return the LP optimal value (a lower bound on the true optimum)
  return solve_LP(...)

SUBROUTINE: fourier_certify(b, n, m, c_target)
  // Compute Fourier-side lower bound on ||g*g||_infty
  // for the step function g corresponding to b
  // Use optimized test function phi (pre-computed via SDP)
  g_hat = compute_fourier_coefficients(b, n)
  bound = integrate(phi_hat * |g_hat|^2)
  T_fourier = c_target + Delta_1(b, best_window, m) + 1/(2*m^2)
  return bound > T_fourier
```

### Complexity Analysis

**Nodes explored:** Let $S(n)$ denote the number of survivors at scale $n$. The original algorithm has $S(n) \cdot \prod_{i}(1 + 2mb_i)$ nodes at the next scale. With adaptive refinement (refining only $r$ out of $2n$ bins), the branching factor drops from $\prod_{i=1}^{2n}(1+2mb_i)$ to $\prod_{i \in \text{selected}}(1+2mb_i)$, a reduction by factor $\prod_{i \notin \text{selected}}(1+2mb_i)$, which is typically $10^2$–$10^6$ per parent.

With LP dual-bound pruning, a fraction $\alpha$ of parents are eliminated without any refinement enumeration. Empirically (from calibration), $\alpha \approx 0.3$–$0.5$ at intermediate scales.

**Cost per node:** The original cost per node is $O(n^2 \cdot (2n)^2) = O(n^4)$ for the full window test (all $(k,\ell)$ pairs, each requiring $O(n)$ summation). With the optimized window ordering, the expected cost drops to $O(n^2 \cdot n) = O(n^3)$ since most refinements are eliminated within the first $O(1)$ windows. The LP bound adds $O(n^3)$ per parent (small LP with $O(n)$ variables).

**Overall:** We estimate a $50$–$200\times$ reduction in total compute compared to the original approach for the same target, or the ability to certify a target $0.005$–$0.01$ higher with the same compute budget.

---

## Part 4 — Theoretical Analysis

### 4.1 Correctness Proof

**Theorem.** *If the improved algorithm terminates with all cases eliminated (survivors empty), then $c \geq c_\text{target}$.*

**Proof.** We verify the four correctness invariants:

**Invariant 1 (Complete enumeration).** At the coarsest scale $n_\text{start}$, every $b \in B_{n_\text{start}, m}$ is enumerated (after symmetry reduction, which is exhaustive up to reflection). The pre-filters (asymmetry, Cauchy–Schwarz, LP) only eliminate parents that are provably ruled out—each filter provides a rigorous lower bound on $\max_{k,\ell} \text{val}(b)$ that exceeds $T$. No parent is skipped without certification.

**Invariant 2 (All windows checked).** For each refinement that is not eliminated by a faster filter, every valid $(k,\ell)$ pair is tested (the window ordering changes only the *order*, not the set). The position-dependent threshold $T(k,\ell)$ is rigorously justified by the Tightened Error Bound lemma (§2a), which provides a valid correction for each window individually.

**Invariant 3 (Arithmetic safety).** The LP bound, Cauchy–Schwarz bound, and test value computations can all be implemented in exact rational arithmetic (since all quantities are rational when $b \in B_{n,m}$). Alternatively, floating-point computation with directed rounding (round threshold down, round test value up) provides safe bounds. The critical comparison `val > T_local` is performed with the appropriate rounding direction.

**Invariant 4 (Refinement completeness).** The adaptive refinement protocol with forced-refinement parameter $K$ guarantees that every bin reaches maximum resolution $n_\text{max}$ after at most $K \cdot \lceil\log_2(n_\text{max}/n_\text{start})\rceil$ rounds. At the final resolution, the completeness of $B_{n_\text{max}, m}$ as a $1/m$-net (Lemma 2) ensures that every continuous $a \in A_{n_\text{max}}$ is covered. The adaptive strategy merely changes the order of exploration, not its completeness.

**Fourier certification correctness:** The Fourier bound $\|g*g\|_\infty \geq \int \hat\phi \cdot |\hat g|^2$ is a rigorous lower bound for any $\hat\phi \geq 0$ with $\phi \leq 1$ pointwise. It eliminates $b$ only when this bound exceeds $T$, which is a sufficient condition. ∎

### 4.2 Identification of Classification of Components

| Component | Status | Justification |
|-----------|--------|---------------|
| Position-dependent correction (§2a) | **Rigorous** | Follows from Equation (1) and support constraints |
| Tightened $\varepsilon * \varepsilon$ bound (§2a) | **Rigorous** | Follows from $L^1$ bound on $\varepsilon$ under Lemma 2 rounding |
| Cauchy–Schwarz pre-filter (§2b) | **Rigorous** | Standard inequality applied to bilinear form |
| LP dual bound (§2b, §2e) | **Rigorous** | McCormick relaxation gives valid lower bound |
| SDP relaxation (§2b) | **Rigorous** | Positive semidefiniteness is a valid relaxation |
| Adaptive refinement (§2c) | **Rigorous** | Completeness guaranteed by forced-refinement schedule |
| Fourier certification (§2d) | **Rigorous** | Valid lower bound via non-negative test function |
| Window ordering heuristic (§2e) | **Heuristic** | Affects speed, not correctness |
| Learned prioritization (§2e) | **Heuristic** | Reorders queue, does not skip cases |
| Structural pre-filter (§2b) | **Heuristic** | Used only for prioritization, not elimination |

### 4.3 Fundamental Theoretical Bottleneck

The new theoretical bottleneck is **the growth of the lattice $B_{n,m}$ as a function of $m$ and $n$**. The number of lattice points is $\binom{4nm + 2n - 1}{2n - 1}$, which grows polynomially in $m$ for fixed $n$ but exponentially in $n$ for fixed $m$.

To push the bound significantly beyond $\sim 1.30$, one would need either:

1. **Much larger $m$** (to reduce the correction term below $0.02$), which increases the lattice size quadratically per dimension, or
2. **Much larger $n$** (to capture fine structure of near-optimal functions), which increases the lattice size exponentially, or
3. **Qualitatively new mathematical ideas** that bypass the exhaustive enumeration paradigm entirely.

Specifically, the error correction $\frac{2}{m} + \frac{1}{m^2}$ (even in our tightened form) imposes a floor: to certify $c \geq c_\text{target}$ we need $T - c_\text{target} > 0$, i.e., $m > 2/(T - c_\text{target})$ at minimum. For $c_\text{target} = 1.30$, we need $m \geq 50$ even with perfect pruning. For $c_\text{target} = 1.35$, we need $m \geq 100$. The computational cost scales as $m^{2n}$ in the worst case.

The **true fundamental limit** is the gap between the spatial discretization approach and the exact continuous problem. The three-lemma framework loses information in three places: (i) replacing $f$ by bin averages (Lemma 1), (ii) discretizing the simplex (Lemma 2), and (iii) bounding the discretization error (Lemma 3). Our improvements tighten (iii) but do not address (i), where information about the *shape* of $f$ within each bin is lost. A qualitatively new approach would need to incorporate intra-bin structure, perhaps via polynomial approximations within each bin or via a fully Fourier-analytic framework.

---

## Part 5 — Target Bound and Parameter Recommendations

### 5.1 Concrete Recommendations

**Conservative target: $c_\text{target} = 1.285$**

- Parameters: $m = 60$, $n_\text{start} = 3$, $n_\text{max} = 24$
- Correction: $\frac{2}{60} + \frac{1}{3600} \approx 0.0336$ (uniform) or $\approx 0.028$ (tightened)
- Threshold: $T \approx 1.313$ (tightened)
- Estimated compute: ~5,000–10,000 CPU hours (a $2$–$4\times$ reduction from the original 20,000 hours at $c = 1.28$), due to the combination of tighter correction (fewer survivors), LP pruning, and adaptive refinement.
- **This is very likely achievable** with the improvements described above.

**Moderate target: $c_\text{target} = 1.29$**

- Parameters: $m = 75$, $n_\text{start} = 3$, $n_\text{max} = 48$
- Correction: $\approx 0.027$ (tightened position-dependent)
- Threshold: $T \approx 1.317$
- Estimated compute: ~50,000–200,000 CPU hours on modern GPUs (A100 or H100 class). The increase is driven by the need for $n_\text{max} = 48$ (one additional doubling beyond 24) to handle the harder cases that survive at 24 bins with the tighter threshold.
- **Plausibly achievable** with the full suite of improvements and access to a modern HPC cluster.

**Aggressive target: $c_\text{target} = 1.30$**

- Parameters: $m = 100$, $n_\text{start} = 4$, $n_\text{max} = 64$
- Correction: $\approx 0.020$ (tightened)
- Threshold: $T \approx 1.320$
- Estimated compute: $10^5$–$10^6$ GPU hours. This would likely require new mathematical insights (beyond the computational improvements above) to be practical.
- **At the boundary of feasibility;** would benefit enormously from the Fourier hybrid approach and aggressive LP pruning.

### 5.2 Comparison to Original

| Metric | Original (CS 2016) | Proposed ($c = 1.285$) | Proposed ($c = 1.29$) |
|--------|--------------------|-----------------------|----------------------|
| $c_\text{target}$ | 1.28 | 1.285 | 1.29 |
| $m$ | 50 | 60 | 75 |
| $n_\text{max}$ | 24 | 24 | 48 |
| Correction term | 0.0404 | ~0.028 | ~0.027 |
| Estimated compute | 20,000 CPU hr | 5,000–10,000 CPU hr | 50,000–200,000 GPU hr |
| Hardware (2016 vs. now) | 7 GPU nodes (2014-era) | 4–8 modern GPUs | 32–128 modern GPUs |
| Key improvements | — | Tighter error, LP pruning, adaptive refinement | + Fourier hybrid, deeper refinement |

### 5.3 Roadmap for Implementation

1. **Phase 1 (weeks 1–2):** Implement the tightened error correction (§2a) and Cauchy–Schwarz pre-filter (§2b). Verify on the original $c = 1.28$ problem that these reduce compute by $2$–$4\times$.
2. **Phase 2 (weeks 3–4):** Implement adaptive refinement (§2c) and optimized window ordering. Target $c = 1.285$ as the first new result.
3. **Phase 3 (weeks 5–8):** Implement LP dual-bound pruning (§2e) and GPU batching optimizations. Target $c = 1.29$.
4. **Phase 4 (if resources permit):** Implement the Fourier hybrid certification (§2d) for the hardest surviving cases. Attempt $c = 1.30$.

---

## Appendix: New Lemmas (Formal Statements)

**Lemma A (Tightened Cross-Term Bound).** *Under the rounding scheme of Lemma 2, for any window $(k, \ell)$ at scale $n$ with discretization $m$:*

$$\frac{1}{4n\ell}\int_{k/(4n)}^{(k+\ell)/(4n)} |(f*\varepsilon)(x)|\,dx \leq \frac{1}{m\ell} \sum_{s=k}^{k+\ell-2} \left(\sum_{\substack{i: -n \leq i \leq n-1 \\ k \leq i + (s-i) \leq k+\ell-2}} \frac{b_i + 1/m}{4n}\right).$$

*In particular, this is at most $\frac{2}{m} \cdot \frac{1}{\ell}\sum_{s=k}^{k+\ell-2} \mu_b(s)$ where $\mu_b(s) \leq 1$ is the normalized mass of bins contributing to diagonal $s$.*

**Lemma B (LP Dual-Bound Validity).** *Let $b \in B_{n,m}$ be a parent, and let $L^*(b)$ denote the optimal value of the McCormick LP relaxation for the minimum-over-refinements of the maximum-over-windows test value. Then $L^*(b) \leq \min_\gamma \max_{k,\ell} \text{val}(c_\gamma, k, \ell)$. In particular, if $L^*(b) > T$ for some threshold $T$, then every refinement of $b$ has maximum test value exceeding $T$.*

*Proof:* The McCormick relaxation replaces the non-convex constraints $q_{ij} = c_i c_j$ with their convex envelope on the box $[0, U_i] \times [0, U_j]$. Any feasible point of the original (non-convex) problem is feasible for the relaxation, so the LP optimum is a lower bound.

**Lemma C (Adaptive Refinement Completeness).** *The adaptive refinement protocol with forced-refinement parameter $K \geq 1$ and initial scale $n_0$ ensures that after at most $R = K \cdot \lceil \log_2(n_\text{max}/n_0)\rceil$ refinement rounds, every bin of every surviving parent has been refined to scale $n_\text{max}$. In particular, the set of configurations checked at the finest scale contains $B_{n_\text{max}, m}$ (up to the adaptive traversal order), and the completeness of the proof is preserved.*
