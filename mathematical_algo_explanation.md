--
  Complete Implementation Guide: Proving Lower Bounds on the Autoconvolution Constant

  1. Problem Statement

  Goal: Prove that $c \geq c_{\text{target}}$, where:

  $$c = \inf_{\substack{f \geq 0 \ \text{supp}(f) \subseteq (-1/4,,1/4)}}
  \frac{|f*f|_{L^\infty}}{\left(\int f\right)^2}$$

  The approach works by exhaustive elimination: show that no valid mass distribution can keep all windowed   autoconvolution averages below the target. If every candidate is ruled out, the bound is proven.       

  ---
  2. Mathematical Foundation (Three Lemmas)

  Lemma 1 — Reduction to Bin Averages

  Input: Parameter $n$ (number of half-bins). Creates $d = 2n$ equal bins.

  Partition: Divide $(-1/4, 1/4)$ into $2n$ bins:
  $$I_j = \left(\frac{j}{4n},, \frac{j+1}{4n}\right), \quad j = -n, \ldots, n-1$$

  Bin averages: For any nonneg function $f$ supported on $(-1/4,1/4)$ with $\int f = 1$:
  $$a_j = 4n \int_{I_j} f(x),dx \geq 0, \qquad \sum_{j=-n}^{n-1} a_j = 4n$$

  Key result: $c \geq a_n$ where:
  $$a_n = \min_{a \in A_n}; \max_{\substack{2 \leq \ell \leq 2n \ -n \leq k \leq n - \ell}}
  \frac{1}{4n\ell} \sum_{k \leq i+j \leq k+\ell-2} a_i a_j$$

  Why it works: Two properties of convolution:
  1. Support containment: $\text{supp}(f_i * f_j) \subseteq I_i + I_j$ (Minkowski sum), so the
  cross-convolution of pieces from bins $i$ and $j$ lands in a predictable interval.
  2. Pigeonhole: $|h|_{L^\infty} \geq \frac{1}{|I|}\int_I |h|$ — if a function has large $L^1$ mass on an 
  interval, it must be large somewhere on that interval.

  Parameter ranges for the windowed test value:
  - Window length: $\ell \in {2, 3, \ldots, 2n}$ (integer, in bins)
  - Window start: $k \in {-n, -n+1, \ldots, n - \ell}$ (integer, index in $i+j$ space)
  - The double sum runs over all pairs $(i, j)$ with both $-n \leq i \leq n-1$, $-n \leq j \leq n-1$, and 
  $k \leq i+j \leq k + \ell - 2$

  Lemma 2 — Discretization of the Simplex

  Input: Parameters $n, m$.

  Discrete lattice:
  $$B_{n,m} = \left{b \in \left(\frac{1}{m}\mathbb{N}\right)^{2n} : \sum b_i = 4n\right}$$

  Every entry is a nonneg multiple of $1/m$, and they sum to $4n$.

  $B_{n,m}$ is a $1/m$-net of the continuous simplex $A_n$ in $\ell^\infty$-norm: for every continuous    
  vector $a \in A_n$, there exists a lattice point $b \in B_{n,m}$ with $|a - b|_\infty \leq 1/m$.        

  Rounding algorithm (to construct $b$ from $a$):
  1. If $a_i$ is already a multiple of $1/m$, set $b_i = a_i$
  2. Otherwise, maintain a running deficit $t_i = \sum_{j<i}(b_j - a_j)$
  3. Round down first, then alternate rounding up/down to keep $-1/m < t_{i+1} \leq 0$
  4. Set the last entry to enforce the exact sum: $b_{n-1} = 4n - \sum_{j=-n}^{n-2} b_j$

  Lemma 3 — Discretization Error Bound

  Result: $c \geq b_{n,m} - \frac{2}{m} - \frac{1}{m^2}$

  The correction $2/m + 1/m^2$ arises from bounding $|f*\varepsilon|\infty \leq 1/m$ and
  $|\varepsilon*\varepsilon|\infty \leq 1/m^2$ where $\varepsilon = f - g$ is the rounding error (bounded 
  by $1/m$).

  Refined correction (Equation 1 in paper — use this for better pruning):
  $$|(f * \varepsilon)(x)| \leq \frac{1}{m} \int_{-1/4 + \max(x,0)}^{1/4 + \min(0,x)} f(x-y),dy$$

  This exploits the support constraint to get a tighter bound that depends on position $x$, making it more   effective at excluding cases than the uniform $1/m$ bound.

  ---
  3. The Effective Threshold

  To prove $c \geq c_{\text{target}}$:

  $$T = c_{\text{target}} + \frac{2}{m} + \frac{1}{m^2}$$

  A vector $b$ is ruled out if there exists at least one window $(k, \ell)$ such that:

  $$\frac{1}{4n\ell} \sum_{k \leq i+j \leq k+\ell-2} b_i b_j > T$$

  If every $b \in B_{n,m}$ is ruled out, then $c \geq c_{\text{target}}$ is proven.

  ---
  4. The Multiscale Branch-and-Prune Algorithm

  This is the core algorithmic innovation. Rather than checking all of $B_{n_{\max}, m}$ directly
  (combinatorial explosion), use a coarse-to-fine strategy:

  Step 1 — Start coarse

  Begin at $n = n_0$ (e.g., $n_0 = 3$, giving $d = 6$ bins). Enumerate all $b \in B_{n_0, m}$.

  Step 2 — Test each parent

  For each parent $b$, compute the windowed autoconvolution test value across all windows $(k, \ell)$. If 
  the maximum exceeds $T$, this parent is ruled out.

  Step 3 — Refine survivors

  For each parent $b$ that survives (cannot be ruled out at the current scale), refine by dyadic
  splitting: each bin $I_j$ is split into two sub-bins, doubling $n$ to $2n$. This produces child vectors 
  in $B_{2n, m}$.

  Refinement details: Each parent component $b_i$ (which is a multiple of $1/m$) is split into two        
  sub-components $c_{2i}$ and $c_{2i+1}$ that sum to $b_i$, with both being multiples of $1/m$. The number   of ways to split $b_i$ is $(1 + m \cdot b_i)$.

  Total refinements per parent: $N = \prod_{i=1}^{2n}(1 + m \cdot b_i)$

  Step 4 — Test refinements

  A parent is ruled out if and only if every single one of its $N$ refinements is ruled out. Even one     
  surviving refinement means the parent survives.

  Step 5 — Repeat

  Continue refining survivors until all are ruled out, or until a computational budget is exhausted.      

  The paper used: $m = 50$, starting at $n = 3$, refining through $n = 6, 12, 24$ until all cases were    
  eliminated, proving $c \geq 1.28$.

  ---
  5. Computing the Windowed Autoconvolution Test Value

  For a vector $b$ of length $2n$, indexed $b_{-n}, \ldots, b_{n-1}$:

  function test_value(b, n):
      max_val = 0
      for ell = 2 to 2n:                          # window length
          for k = -n to (n - ell):                 # window start
              sum = 0
              for i = -n to n-1:
                  for j = -n to n-1:
                      if k <= i+j AND i+j <= k+ell-2:
                          sum += b[i] * b[j]
              val = sum / (4*n * ell)
              max_val = max(max_val, val)
      return max_val

  Optimization: The double sum over $(i,j)$ with constraint $k \leq i+j \leq k+\ell-2$ can be reorganized 
  by iterating over the sum $s = i+j$:

  for s = k to k+ell-2:
      for i = max(-n, s-(n-1)) to min(n-1, s+n):
          j = s - i
          sum += b[i] * b[j]

  ---
  6. Matrix Formulation (for GPU)

  The paper's key GPU insight: reformulate everything as matrix multiplication.

  Precompute constraint matrices $C_k$

  For each window length $k$ (corresponding to $\ell$ in the formulas), build a matrix $C_k \in
  \mathbb{R}^{(2n)^2 \times (4n - k + 1)}$ that encodes which $(i,j)$ pairs contribute to which window    
  position for that window length. These depend only on $n$ and are computed once.

  Build the refinement matrix

  For a parent $b$, construct all $N$ refinement vectors as rows of $c_b \in \mathbb{R}^{N \times 2n}$.   

  Compute the outer-product matrix

  $F \in \mathbb{R}^{N \times (2n)^2}$ where $F[\gamma, n(i-1) + j] = c_\gamma[i] \cdot c_\gamma[j]$.     

  This is the element-wise product of all pairs of components for each refinement.

  Matrix multiply to get test values

  $FC_k \in \mathbb{R}^{N \times (4n - k + 1)}$ gives the windowed autoconvolution sums for all
  refinements and all window positions at window length $k$.

  Check threshold

  Refinement $\gamma$ is ruled out if any entry in row $\gamma$ of any $FC_k$ exceeds $T$. The parent is  
  ruled out only if all rows (all $\gamma$) are ruled out.

  ---
  7. Pruning Optimizations

  7a. Asymmetry pruning (coarse pre-filter)

  If a parent has more than fraction $p$ of its mass on one half, then at least $p^2$ of the
  autoconvolution mass lands on the corresponding half of $(-1/2, 1/2)$, giving:
  $$|f*f|_\infty \geq \frac{p^2}{1/2} = 2p^2$$

  If $2p^2 > c_{\text{target}}$, this parent is immediately ruled out. For $c_{\text{target}} = 1.28$, any   parent with $> 80%$ of mass on one side is eliminated.

  By symmetry, you can also restrict to canonical orderings where $\sum_{i < 0} b_i \leq \sum_{i \geq 0}  
  b_i$, halving the search space.

  7b. Refined correction term (Equation 1)

  Use the position-dependent bound on $|(f * \varepsilon)(x)|$ instead of the uniform $1/m$ bound. This   
  tightens the effective threshold window-by-window, ruling out more cases.

  7c. Early termination per refinement

  When checking refinements, as soon as one window exceeds $T$ for a refinement $\gamma$, that refinement 
  is ruled out — no need to check remaining windows.

  7d. Early termination per parent

  If during refinement enumeration, a single refinement survives all windows, the parent survives — no    
  need to enumerate remaining refinements (for the run_single_level proof mode). For
  find_best_bound_direct mode, you want the minimum over all refinements.

  ---
  8. Data Structures and Sizing

  ┌──────────────────────────┬─────────────────────────────────────────┬───────────────────────┐
  │          Object          │                  Size                   │         Notes         │
  ├──────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ Parent grid $B_{n,m}$    │ $\binom{4nm + 2n - 1}{2n - 1}$ elements │ Exponential in $n$    │
  ├──────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ Refinements per parent   │ $N = \prod_{i=1}^{2n}(1 + m \cdot b_i)$ │ Can be huge           │
  ├──────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ Constraint matrix $C_k$  │ $(2n)^2 \times (4n - k + 1)$            │ One per window length │
  ├──────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ Outer-product matrix $F$ │ $N \times (2n)^2$                       │ Per parent, on GPU    │
  ├──────────────────────────┼─────────────────────────────────────────┼───────────────────────┤
  │ Result matrix $FC_k$     │ $N \times (4n - k + 1)$                 │ Per parent per $k$    │
  └──────────────────────────┴─────────────────────────────────────────┴───────────────────────┘

  ---
  9. Two Solver Modes

  Mode 1: find_best_bound_direct

  Computes $b_{n,m}$ exactly — the minimum over all parents of the maximum test value. Gives the best     
  bound achievable at parameters $(n, m)$ without multiscale refinement. Useful for exploration.

  Mode 2: run_single_level (the proof)

  For a fixed $c_{\text{target}}$: enumerate all parents, compute threshold $T$, and verify every parent  
  is ruled out. If yes, the proof is complete. If any parent survives, either the target is too aggressive   or refinement to larger $n$ is needed.

  ---
  10. Practical Parameter Choices

  ┌─────────────────────┬───────────────┬────────────────────────────────────────────────────────────────┐  │      Parameter      │  Paper value  │                      Effect of increasing                      │  ├─────────────────────┼───────────────┼────────────────────────────────────────────────────────────────┤  │ $m$                 │ 50            │ Reduces correction term $2/m + 1/m^2$, increases grid size     │  │ (discretization)    │               │ polynomially                                                   │  ├─────────────────────┼───────────────┼────────────────────────────────────────────────────────────────┤  │ $n$ (bins)          │ 3 → 6 → 12 →  │ Finer spatial resolution, exponential cost increase            │  │                     │ 24            │                                                                │  ├─────────────────────┼───────────────┼────────────────────────────────────────────────────────────────┤  │ $c_{\text{target}}$ │ 1.28          │ Higher targets are harder to prove                             │  └─────────────────────┴───────────────┴────────────────────────────────────────────────────────────────┘
  Trade-off: Larger $m$ gives a tighter bound (smaller correction) but more parents to check. Larger $n$  
  gives finer resolution but the number of parents and refinements explodes. The multiscale approach      
  mitigates this: most parents are eliminated at coarse $n$, only a few need refinement.

  ---
  11. End-to-End Algorithm Pseudocode

  function prove_lower_bound(c_target, m, n_start, n_max):
      T = c_target + 2/m + 1/m^2

      # Generate all parents at coarse level
      survivors = enumerate_all(B_{n_start, m})

      # Asymmetry pre-filter
      survivors = [b for b in survivors if not asymmetry_prune(b, c_target)]

      # Canonical form (exploit symmetry)
      survivors = [b for b in survivors if is_canonical(b)]

      n = n_start
      while survivors is not empty and n <= n_max:
          next_survivors = []
          for b in survivors:
              # Generate all N refinements of b
              refinements = generate_refinements(b, m)  # N = prod(1 + m*b_i)

              all_ruled_out = true
              for c_gamma in refinements:
                  ruled_out = false
                  for ell = 2 to 2n:
                      for k = -n to n-ell:
                          val = windowed_test_value(c_gamma, k, ell, n)
                          if val > T:    # (use refined correction here)
                              ruled_out = true
                              break
                      if ruled_out: break

                  if not ruled_out:
                      all_ruled_out = false
                      break   # parent survives, refine further

              if not all_ruled_out:
                  next_survivors.append(b)

          # Refine: each survivor's sub-bins get split (n -> 2n)
          survivors = refine_survivors(next_survivors)
          n = 2 * n

      if survivors is empty:
          print("PROVEN: c >= " + c_target)
      else:
          print("FAILED: " + len(survivors) + " survivors remain at n=" + n)

  ---
  12. Correctness Invariants

  An implementation must guarantee:

  1. Complete enumeration — every $b \in B_{n,m}$ must be checked (or shown to be covered by a checked    
  ancestor + its refinements)
  2. All windows checked — for each refinement, every valid $(k, \ell)$ must be tested before declaring it   a survivor
  3. Exact arithmetic or safe rounding — the comparison against $T$ must use the correct threshold        
  including the full correction term; floating-point errors must not cause false positives (wrongly ruling   out a valid case)
  4. Refinement completeness — every refinement of a surviving parent must itself be checked at the finer 
  level

  If all four are satisfied and every case is ruled out, the proof is mathematically valid.