# Adaptive Refinement for the CS14 Branch-and-Prune Algorithm

## Three Rigorous Approaches to Reduce Combinatorial Cost

---

## 1. The Problem

### 1.1 Background

The CS14 algorithm (Cloninger--Steinerberger, [arXiv:1403.7988](https://arxiv.org/abs/1403.7988)) proves $c \geq c_{\text{target}}$ by exhaustive elimination. At resolution $n$, partition $(-1/4, 1/4)$ into $d = 2n$ bins of width $1/(4n)$, with bin averages $b_i = 4n \int_{I_i} f \geq 0$, $\sum b_i = 4n$, discretized to the lattice $(1/m)\mathbb{N}_0$. The windowed test value is:

$$\text{val}_n(b, k, \ell) = \frac{1}{4n\ell} \sum_{k \leq i+j \leq k+\ell-2} b_i b_j$$

To prove $c \geq c_{\text{target}}$, show every $b \in B_{n,m}$ has $\max_{k,\ell} \text{val}_n(b,k,\ell) \geq T$ where $T = c_{\text{target}} + 2/m + 1/m^2$.

### 1.2 The refinement step

When a parent $b$ at resolution $n$ *survives* (no window exceeds $T$), it is refined to resolution $2n$. Each parent bin $i$ (value $b_i$) splits into two sub-bins at resolution $2n$:

$$c_{2i} + c_{2i+1} = 2b_i, \quad c_{2i}, c_{2i+1} \in \tfrac{1}{m}\mathbb{N}_0$$

(This follows from the averaging property: $b_i = (c_{2i} + c_{2i+1})/2$.)

The number of splits per bin is $1 + 2m b_i$, and the total number of children is $\prod_{i=0}^{d-1}(1 + 2m b_i)$. A parent is eliminated only if **every** child is eliminated at the finer resolution.

### 1.3 The adaptive refinement idea and its formalization gap

The key observation motivating adaptive refinement: the dyadic splitting $n \to 2n$ uniformly doubles resolution everywhere, but for a given surviving parent, some bins are "resolved" (the test value is far above $T$ in all windows involving that bin) while others are "uncertain" (near $T$, where finer resolution might change the outcome).

**Bin sensitivity.** For a parent $b$ surviving at scale $n$, the sensitivity of bin $i$ is:

$$\sigma_i(b) = \max_{k,\ell} \frac{\partial}{\partial b_i} \text{val}_n(b,k,\ell) = \max_{k,\ell} \frac{2}{4n\ell} \sum_{\substack{j:\, k \leq i+j \leq k+\ell-2}} b_j$$

Bins with high sensitivity are those where finer resolution is most likely to change the outcome.

The original adaptive proposal (in `improved_autoconvolution_algorithm.md`, Section 2b) suggests refining only the high-sensitivity bins, creating **mixed-resolution configurations**. This is problematic for three reasons:

1. **Lemma 1 requires a uniform partition.** The core inequality $\text{val}_n(b,k,\ell) \leq \|f*f\|_\infty$ relies on all bins having equal width $1/(4n)$, so that the Minkowski sum $I_i + I_j$ has width exactly $2/(4n)$ for all pairs. With mixed-width bins, the Minkowski sums have variable width and the window parametrization by integers $(k, \ell)$ breaks down.

2. **The completeness argument is incomplete.** The forced-refinement schedule (refine unrefined bins after $K$ rounds) guarantees that every bin *eventually* reaches full resolution. But between the initial and final rounds, the algorithm operates on mixed-resolution configurations whose test values are not directly comparable to the uniform-resolution test values that Lemma 1 bounds. A surviving mixed-resolution configuration does not correspond to any element of $B_{n_{\max}, m}$.

3. **Sensitivity is defined at the wrong scale.** The sensitivity $\sigma_i(b)$ measures the gradient of the coarse-scale test value, but the refinement produces fine-scale test values with different window structure. A bin with high coarse-scale sensitivity might have low fine-scale impact and vice versa.

The three approaches below recover the adaptive speedup without these issues.

---

## 2. Approach 1: Freezable Bins on the Uniform Grid (Recommended)

### 2.1 Key idea

Keep the dyadic refinement $n \to 2n$ with the standard uniform grid. **Lemma 1 applies unchanged.** The savings come from proving that certain bins' splits do not affect the pruning outcome, so their enumeration can be collapsed.

### 2.2 Setup and notation

Parent $b$ at resolution $n$ ($d = 2n$ bins). Refine to resolution $2n$ ($d' = 4n$ bins). For each parent bin $i$, the sub-bins are:

$$c_{2i} = b_i + \delta_i, \quad c_{2i+1} = b_i - \delta_i$$

where the deviation $\delta_i \in [-b_i,\, b_i] \cap \tfrac{1}{m}\mathbb{Z}$ takes $1 + 2mb_i$ values. The "flat split" is $\delta_i = 0$ (i.e., $c_{2i} = c_{2i+1} = b_i$).

### 2.3 Polynomial structure of the test value in $\delta_i$

Fix all bins except bin $i$: for $j \neq i$, fix $c_{2j}$ and $c_{2j+1}$ at some specific values. We show that the test value at resolution $2n$ is a **quadratic polynomial** in $\delta_i$.

Write the windowed test value as:

$$\text{val}_{2n}(c, k', \ell') = \frac{1}{8n\ell'} \sum_{\substack{p + q \in W}} c_p c_q$$

where $W = W(k', \ell') = \{k', k'+1, \ldots, k'+\ell'-2\}$ is the set of active anti-diagonals.

Let $u = 2i$, $v = 2i+1$ be the two sub-bin indices. Partition the sum by involvement of $u, v$:

$$\sum_{p+q \in W} c_p c_q = S_0 + S_{\mathrm{cross}} + S_{\mathrm{self}}$$

where:

- $S_0 = \sum_{\substack{p,q \notin \{u,v\}\\p+q \in W}} c_p c_q$ (constant in $\delta_i$)

- $S_{\mathrm{cross}} = 2\, c_u \cdot A_u + 2\, c_v \cdot A_v$, where $A_u = \sum_{\substack{p \notin \{u,v\}\\u+p \in W}} c_p$ and $A_v = \sum_{\substack{p \notin \{u,v\}\\v+p \in W}} c_p$

- $S_{\mathrm{self}} = c_u^2 \cdot \alpha + 2\, c_u c_v \cdot \beta + c_v^2 \cdot \gamma$, where $\alpha = \mathbb{1}[2u \in W]$, $\beta = \mathbb{1}[u+v \in W]$, $\gamma = \mathbb{1}[2v \in W]$

Substituting $c_u = b_i + \delta_i$, $c_v = b_i - \delta_i$:

$$S_{\mathrm{cross}} = 2b_i(A_u + A_v) + 2\delta_i(A_u - A_v)$$

$$S_{\mathrm{self}} = (\alpha + 2\beta + \gamma)\,b_i^2 + 2(\alpha - \gamma)\,b_i\,\delta_i + (\alpha - 2\beta + \gamma)\,\delta_i^2$$

So $\sum c_p c_q = P_0 + P_1\,\delta_i + P_2\,\delta_i^2$ where:

$$P_0 = S_0 + 2b_i(A_u + A_v) + (\alpha + 2\beta + \gamma)\,b_i^2$$

$$P_1 = 2(A_u - A_v) + 2(\alpha - \gamma)\,b_i$$

$$P_2 = \alpha - 2\beta + \gamma$$

The test value is $\text{val} = (P_0 + P_1\,\delta_i + P_2\,\delta_i^2) / (8n\ell')$.

**The coefficient $P_2$.** Since $W$ is a set of consecutive integers:

| Anti-diagonals $4i, 4i+1, 4i+2$ in $W$ | $(\alpha, \beta, \gamma)$ | $P_2$ |
|---|---|---|
| None | $(0,0,0)$ | $0$ |
| $\{4i\}$ only | $(1,0,0)$ | $+1$ |
| $\{4i+1\}$ only | $(0,1,0)$ | $-2$ |
| $\{4i+2\}$ only | $(0,0,1)$ | $+1$ |
| $\{4i, 4i+1\}$ | $(1,1,0)$ | $-1$ |
| $\{4i+1, 4i+2\}$ | $(0,1,1)$ | $-1$ |
| All three | $(1,1,1)$ | $0$ |

Note that $\{4i, 4i+2\}$ without $4i+1$ is impossible since $W$ is a consecutive range. So $P_2 \in \{-2, -1, 0, 1\}$.

**The coefficient $P_1$.** The term $A_u - A_v$ depends on the values of **all other bins**, not just the flat-split values. Since $v = u + 1$, the index sets $\{p : u + p \in W\}$ and $\{p : v + p \in W\}$ differ by at most two boundary elements:

$$A_u - A_v = c_{k'+\ell'-2-u}\,\mathbb{1}[\text{in range}] - c_{k'-v}\,\mathbb{1}[\text{in range}]$$

where "in range" means the index is in $\{0, \ldots, 4n-1\} \setminus \{u, v\}$. So $|A_u - A_v|$ is bounded by the sum of at most two sub-bin values.

### 2.4 Freezability: single-bin version

**Definition (absolute gap).** For a child configuration $c$ at resolution $2n$:

$$\text{gap}(c, k', \ell') = \text{val}_{2n}(c, k', \ell') - T$$

**Definition (perturbation bound).** For a specific assignment of all other bins, the maximum absolute change in test value from varying $\delta_i$ is:

$$\Delta_i(k', \ell') = \max_{\delta_i \in [-b_i, b_i]} \frac{|P_1\,\delta_i + P_2\,\delta_i^2|}{8n\ell'} \leq \frac{|P_1|\,b_i + |P_2|\,b_i^2}{8n\ell'}$$

**Proposition (Single-Bin Freezability).** *Fix all bins except bin $i$ at specific values. If for every window $(k', \ell')$ at resolution $2n$:*

$$\Delta_i(k', \ell') < |\text{gap}(c|_{\delta_i=0},\; k', \ell')|$$

*then the pruning outcome (whether the configuration is eliminated) is the same for all $\delta_i \in [-b_i, b_i]$.*

**Proof.** The pruning outcome depends on whether $\max_{k',\ell'} \text{val}_{2n}(c, k', \ell') \geq T$, i.e., whether any window exceeds the threshold. For a specific window $(k', \ell')$:

$$\text{val}(c|_{\delta_i}) = \text{val}(c|_{\delta_i=0}) + \frac{P_1\,\delta_i + P_2\,\delta_i^2}{8n\ell'}$$

The deviation from the $\delta_i = 0$ value is at most $\Delta_i(k', \ell')$. By hypothesis, this is strictly less than $|\text{gap}(c|_{\delta_i=0}, k', \ell')|$. Therefore:

- If $\text{gap}(c|_{\delta_i=0}, k', \ell') > 0$ (window exceeds $T$ at $\delta_i = 0$): the perturbed value satisfies $\text{val}(c|_{\delta_i}) > T - \Delta_i + \text{gap} > T$. So the window still exceeds $T$. $\checkmark$
- If $\text{gap}(c|_{\delta_i=0}, k', \ell') < 0$ (window below $T$ at $\delta_i = 0$): the perturbed value satisfies $\text{val}(c|_{\delta_i}) < T + \Delta_i - |\text{gap}| < T$. So the window stays below $T$. $\checkmark$

Since this holds for **every** window, the set of windows exceeding $T$ is the same for all $\delta_i$. In particular, the set is nonempty (configuration eliminated) for all $\delta_i$ iff it is nonempty at $\delta_i = 0$. $\blacksquare$

### 2.5 Freezability: multi-bin version

To reduce the enumeration product from $\prod_{i=0}^{d-1}(1+2mb_i)$ to $\prod_{i \notin F}(1+2mb_i)$ for a set $F$ of "frozen" bins, we need: for every assignment of non-frozen bins $\{\delta_j\}_{j \notin F}$, each frozen bin $i \in F$ satisfies the single-bin freezability criterion.

The complication is that $P_1$ (the linear coefficient) depends on the other bins' values through $A_u$ and $A_v$. When non-frozen bins take non-flat values, $P_1$ changes from its flat-split value.

**Universal perturbation bound.** For bin $i$, define the worst-case perturbation over all possible assignments of other bins:

$$\bar{\Delta}_i(k', \ell') = \max_{\substack{\text{all other-bin} \\ \text{assignments}}} \;\; \max_{\delta_i \in [-b_i,\, b_i]} \frac{|P_1\,\delta_i + P_2\,\delta_i^2|}{8n\ell'}$$

Since $|A_u - A_v| \leq c_{\max,1} + c_{\max,2}$ where $c_{\max,1}$ and $c_{\max,2}$ are the two boundary sub-bin values, and each sub-bin value is at most $2B_{\max}$ where $B_{\max} = \max_j b_j$, we get:

$$|P_1| \leq 2|A_u - A_v| + 2|\alpha - \gamma|\,b_i \leq 8B_{\max} + 2b_i$$

Combined with $|P_2| \leq 2$:

$$\bar{\Delta}_i(k', \ell') \leq \frac{(8B_{\max} + 2b_i)\,b_i + 2b_i^2}{8n\ell'} = \frac{b_i(8B_{\max} + 4b_i)}{8n\ell'} = \frac{b_i(2B_{\max} + b_i)}{2n\ell'}$$

**Difficulty with a static multi-bin criterion.** One might hope to state a condition purely in terms of the universal bounds $\bar{\Delta}_i$ and the gap at the "all frozen bins flat" baseline that guarantees freezability for all non-frozen-bin assignments simultaneously. However, such a condition must also account for the fact that non-frozen bins' variations shift the baseline gap itself. The universal bound $\bar{\Delta}_i$ controls how much the frozen bins' perturbations affect the test value, but it does not control how the gap moves as non-frozen bins change. A condition like $\sum_{i \in F} \bar{\Delta}_i < |\text{gap}(c_{\text{all flat}}, k', \ell')|$ is therefore insufficient: the gap at a specific non-frozen-bin assignment could be much smaller than the gap at the all-flat baseline, potentially smaller than $\sum \bar{\Delta}_i$.

One could in principle strengthen the condition by also bounding how much the non-frozen bins can shift the gap (using the same quadratic analysis applied to non-frozen bins). But this leads to a circular dependency: the bound on the gap shift depends on which bins are frozen, which depends on the gap bound. Resolving this requires iterative tightening or a global analysis that is unlikely to be practical.

**Per-child freezability (recommended).** The clean solution is to evaluate the freezability criterion at runtime, using the actual current values of all other bins. This is fully rigorous because it applies the Single-Bin Freezability Proposition (Section 2.4) with all other bins at their specific (known) values — exactly the setting in which the proposition was proved. The implementation:

1. In the nested enumeration loop, when about to iterate over bin $i$'s splits:
2. Compute the test value at $\delta_i = 0$ (one evaluation), with all other bins at their current values.
3. Compute $\Delta_i$ using the actual current $P_1$ and $P_2$ (not the universal bound).
4. If $\Delta_i < \min_{k',\ell'} |\text{gap}|$, skip the $\delta_i$ loop (use $\delta_i = 0$).
5. Otherwise, enumerate all $1 + 2mb_i$ values of $\delta_i$.

This gives per-child savings without the looseness of the universal bound.

### 2.6 Completeness

**Lemma 1 applies unchanged** because we are on the standard uniform grid at resolution $2n$. The freezability optimization does not skip any configuration whose pruning outcome differs from the flat-split representative. A parent is declared eliminated only when every child — after collapsing provably equivalent frozen-bin splits — is verified to have $\max \text{val} \geq T$. The correctness argument is identical to the original CS14 proof.

### 2.7 Expected savings

For **boundary bins** with $b_i \approx 0$: the perturbation $\bar{\Delta}_i \approx 0$, so freezability is easy to achieve. However, the branching factor $1 + 2mb_i \approx 1$, so the savings are negligible.

For **interior bins** with $b_i \sim 2B_{\max}$: the universal bound gives $\bar{\Delta}_i \sim B_{\max}^2 / (n\ell')$, which may exceed the gap for small $n$ or $\ell'$. The per-child bound $\Delta_i$ is much tighter and more likely to achieve freezability.

The main savings come from **moderate bins** where $b_i$ is nonzero (branching factor $> 1$) but the gap is large relative to the perturbation. This occurs for bins in the interior of the support of near-optimal functions, where the autoconvolution is well above the threshold in nearby windows.

---

## 3. Approach 2: Generalized Lemma 1 for Non-Uniform Partitions

### 3.1 Key idea

If one genuinely wants mixed-resolution grids, the core inequality (Lemma 1) must be reproved for general (non-uniform) partitions. This is possible but adds implementation complexity.

### 3.2 Generalized inequality

**Setup.** Let $\mathcal{P} = \{I_1, \ldots, I_N\}$ be a partition of $(-1/4, 1/4)$ into intervals (not necessarily equal width). Let $w_k = |I_k|$ and $a_k = \frac{1}{w_k}\int_{I_k} f(x)\,dx$ (the average on bin $k$). Note $\sum_k w_k a_k = \int f = 1$ (assuming $\int f = 1$ WLOG).

**Generalized Lemma 1.** *For any interval $J \subseteq (-1/2, 1/2)$, define $S_J = \{(k, l) : I_k + I_l \subseteq J\}$ (Minkowski sum containment). Then:*

$$\frac{1}{|J|} \sum_{(k,l) \in S_J} w_k w_l \cdot a_k a_l \leq \|f * f\|_{L^\infty}$$

**Proof.** For each bin $I_k$, define $f_k = f \cdot \mathbb{1}_{I_k}$. Then $\int f_k = w_k a_k$, $\text{supp}(f_k * f_l) \subseteq I_k + I_l$ (Minkowski sum), and $\int (f_k * f_l) = w_k a_k \cdot w_l a_l$. For $(k, l) \in S_J$, we have $I_k + I_l \subseteq J$, so $\text{supp}(f_k * f_l) \subseteq J$. Therefore:

$$\sum_{(k,l) \in S_J} w_k a_k \cdot w_l a_l = \sum_{(k,l) \in S_J} \int (f_k * f_l) \leq \int_J (f * f)(x)\,dx \leq |J| \cdot \|f * f\|_{L^\infty}$$

The first inequality uses that $f * f \geq \sum_{(k,l) \in S_J} f_k * f_l$ pointwise on $J$ (since $(k,l) \in S_J$ gives only a subset of all pairs, and all cross-terms $f_k * f_l \geq 0$). The second is the definition of $L^\infty$ norm. $\blacksquare$

**Recovery of the uniform case.** For equal-width bins $w_k = 1/(4n)$, $a_k = 4n \int_{I_k} f$, and $J = (k/(4n),\, (k+\ell)/(4n))$ of width $\ell/(4n)$, the containment condition $I_i + I_j \subseteq J$ reduces to $k \leq i + j \leq k + \ell - 2$ (since $I_i + I_j = ((i+j)/(4n),\, (i+j+2)/(4n))$). The bound becomes:

$$\frac{1}{\ell/(4n)} \sum_{k \leq i+j \leq k+\ell-2} \frac{a_i a_j}{(4n)^2} = \frac{1}{4n\ell} \sum_{k \leq i+j \leq k+\ell-2} a_i a_j$$

matching the standard formula.

### 3.3 Containment checks for non-uniform bins

For the uniform case, the window parameters are two integers $(k, \ell)$, and the containment condition $k \leq i+j \leq k+\ell-2$ is trivial to check. For non-uniform bins:

- $I_k = [L_k, R_k)$ (left and right endpoints).
- $I_k + I_l = [L_k + L_l,\; R_k + R_l)$.
- The test window $J = [a, b)$ must satisfy $a \leq L_k + L_l$ and $R_k + R_l \leq b$.

The set of valid windows is no longer parametrized by two integers. One must consider all intervals $J \subseteq (-1/2, 1/2)$, though in practice one restricts to windows whose endpoints align with the partition's sub-grid:

$$\mathcal{J} = \{[L_k + L_l,\; L_{k'} + L_{l'}] : k, l, k', l' \text{ valid}\}$$

The number of such windows is $O(N^2)$ in the number of bins $N$, compared to $O(n^2)$ for the uniform case. For a mixed-resolution grid where most bins have width $1/(4n)$ and a few have width $1/(8n)$, the overhead is modest.

### 3.4 Discretization

The $1/m$-net argument (Lemma 2) generalizes directly to non-uniform partitions. The key property is that the lattice $(1/m)\mathbb{N}_0$ approximates the continuous simplex coordinate-wise with error $1/m$, regardless of the partition structure. The error correction $2/m + 1/m^2$ (Lemma 3) depends only on the lattice spacing, not the bin widths.

### 3.5 Practical assessment

This approach works in principle and gives complete flexibility in choosing which bins to refine. The costs are:

1. **Bookkeeping complexity.** Tracking variable-width bins, their Minkowski sums, and the valid window set requires more complex data structures than the uniform case.
2. **More windows to check.** The number of valid windows grows with the number of distinct bin widths.
3. **Implementation effort.** The existing CUDA kernels are heavily optimized for the uniform grid; adapting them to non-uniform partitions would require substantial rewriting.

---

## 4. Approach 3: Branch-and-Bound with Sensitivity Ordering

### 4.1 Key idea

Work at the finest resolution $n_{\max}$ from the start. Rather than enumerating the entire lattice $B_{n_{\max},m}$ (which is impossibly large), build a **search tree** where bins are fixed one at a time, pruning branches early when the test value is guaranteed to exceed $T$ regardless of the remaining free bins.

### 4.2 Search tree structure

**Nodes.** Each node at depth $k$ has bins $i_1, \ldots, i_k$ fixed to specific values, with the remaining $d - k$ bins free (subject to the sum constraint $\sum b_i = 4n_{\max}$ and $b_i \geq 0$).

**Branching.** At each node, choose an unfixed bin $i_{k+1}$ and branch on its possible values $b_{i_{k+1}} \in \{0, 1/m, 2/m, \ldots\}$ (subject to the remaining mass being non-negative).

**Leaves.** At depth $d$, all bins are fixed and we compute exact test values.

**Pruning.** At each internal node, compute a lower bound on $\max_{k,\ell} \text{val}(b, k, \ell)$ over all completions of the free bins. If this lower bound exceeds $T$, prune the entire subtree.

### 4.3 Relaxed lower bounds for pruning

**Fixed-variable bound.** For any window $(k', \ell')$, the contribution from pairs of already-fixed bins gives a valid lower bound:

$$\text{val}(b, k', \ell') \geq \frac{1}{4n_{\max}\ell'} \sum_{\substack{i,j \text{ both fixed}\\k' \leq i+j \leq k'+\ell'-2}} b_i b_j$$

This holds because all $b_i b_j \geq 0$, so restricting to fixed bins drops only non-negative terms. Taking the maximum over windows gives a lower bound on $\max_{k',\ell'} \text{val}(b, k', \ell')$.

**Mass concentration heuristic (informal).** If the total fixed mass $M_{\text{fixed}} = \sum_{i \text{ fixed}} b_i$ is large and concentrated in a small index range, the fixed-variable bound is likely to be large. As rough intuition: if all fixed bins fall in a contiguous range of $r$ indices, one might expect the autoconvolution to concentrate roughly $M_{\text{fixed}}^2$ worth of cross-terms in nearby anti-diagonals. However, this does not yield a clean universal inequality — when fixed bins are spread across many anti-diagonals, no single window captures all $b_i b_j$ cross-terms, and the sum within any one window can be strictly less than $M_{\text{fixed}}^2$. The rigorous bound is always the fixed-variable bound above, evaluated exactly for each window.

**LP relaxation.** For a tighter bound, formulate a linear program: minimize $\max_{k',\ell'} \text{val}$ over all completions of the free bins, using McCormick envelope constraints to linearize the products $b_i b_j$. The LP optimum is a valid lower bound.

### 4.4 Sensitivity-guided branching order

**At each node, choose the unfixed bin with the largest impact on the test values.** The sensitivity $\sigma_i$ (defined in Section 1.3) provides a measure of impact. Branching on the highest-sensitivity bin first leads to:

1. Faster pruning: the fixed-variable bound grows quickly when high-impact bins are fixed first.
2. Smaller trees: branches corresponding to "easy" configurations (where the test value is far above $T$) are pruned early.

This is a standard strategy in integer programming (variable selection by impact).

### 4.5 Correctness

**Theorem.** *If the branch-and-bound tree is fully explored (every leaf either pruned or verified to have $\max \text{val} \geq T$), then $c \geq c_{\text{target}}$.*

**Proof.** Every $b \in B_{n_{\max}, m}$ corresponds to exactly one leaf of the tree. At each pruned node, the lower bound $L > T$ guarantees that every $b$ in the subtree has $\max \text{val} \geq L > T$. This is because the lower bound was obtained by dropping non-negative terms from the exact bilinear sum, so the exact value is at least as large. At non-pruned leaves, the exact test value exceeds $T$ by direct computation. Therefore, every $b \in B_{n_{\max}, m}$ has $\max \text{val} \geq T$. By the standard CS14 argument (Lemmas 1--3), $c \geq c_{\text{target}}$. $\blacksquare$

**Remark on the branching-order claim.** The user's original text states that pruning validity "follows from convexity." This is imprecise: the maximum of a bilinear form is not convex in general. The correct justification is the **monotonicity of restriction**: restricting a sum of non-negative terms to a subset of pairs gives a lower bound. No convexity is needed.

### 4.6 Relationship to CS14

The original CS14 dyadic scheme is a specific traversal of this tree. At resolution $n$, a "parent" corresponds to fixing bins in groups (each group of $d/(2n)$ consecutive fine-resolution bins is constrained by the coarse-resolution value). The sensitivity-ordered tree is a different traversal that can reach pruning decisions faster for parts of the search space where only a few bins determine the outcome.

### 4.7 Practical considerations

1. **Memory.** The tree can be explored depth-first, requiring only $O(d)$ memory for the current path.
2. **Parallelism.** Subtrees rooted at depth $k$ can be processed independently on different GPU threads, similar to the current $c_0$-based parallelization.
3. **Compatibility.** The existing pruning heuristics (asymmetry, ell=2 max-element bound, factored rectangle) all apply as additional pruning criteria at each node.
4. **Initialization.** Starting from a coarse-scale solution (e.g., survivors at resolution $n = 3$) can warm-start the tree by fixing the first few bins to the coarse values.

---

## 5. Recommendation

**Start with Approach 1.** It requires zero changes to the theoretical framework. The per-child freezability check (Section 2.5, Remark) is the most practical variant: when iterating over bin $i$'s splits in the enumeration loop, compute the gap at $\delta_i = 0$ and the perturbation bound $\Delta_i$ using the actual current values of all other bins. If $\Delta_i < \min_{k',\ell'} |\text{gap}|$, skip the loop and use $\delta_i = 0$. The implementation is a local optimization of the existing code (adding a conditional skip inside the existing nested loop), with a one-page correctness proof that reduces directly to the original CS14 argument.

**If Approach 1 gives insufficient savings:** move to Approach 3 (branch-and-bound). This is the most flexible framework and the natural home for this kind of combinatorial search. The existing CS14 algorithm is a special case, so the transition is incremental: start with the current dyadic traversal and gradually shift to sensitivity-ordered branching as the implementation matures.

**Approach 2 (non-uniform grids) is not recommended** unless there is a compelling mathematical reason to use mixed-resolution configurations. The bookkeeping overhead and implementation complexity outweigh the benefits, especially since Approach 1 achieves adaptive savings without changing the grid.

---

## Appendix: Formal Statements

**Lemma (Quadratic Structure).** *For parent $b$ at resolution $n$, child $c$ at resolution $2n$ with $c_{2i} = b_i + \delta_i$ and $c_{2i+1} = b_i - \delta_i$, and all other sub-bins fixed, the test value $\text{val}_{2n}(c, k', \ell')$ is a polynomial of degree 2 in $\delta_i$:*

$$\text{val}_{2n}(c, k', \ell') = \frac{P_0 + P_1 \delta_i + P_2 \delta_i^2}{8n\ell'}$$

*with $P_0, P_1$ depending on the full configuration and window, and $P_2 = \alpha - 2\beta + \gamma \in \{-2, -1, 0, 1\}$ depending only on the window and bin position. The maximum absolute deviation from $\delta_i = 0$ satisfies:*

$$|\text{val}|_{\delta_i} - \text{val}|_{\delta_i=0}| \leq \frac{|P_1|\,b_i + |P_2|\,b_i^2}{8n\ell'}$$

**Proposition (Freezability).** *Fix all bins except bin $i$. If for every window $(k', \ell')$:*

$$\frac{|P_1|\,b_i + |P_2|\,b_i^2}{8n\ell'} < \left|\text{val}_{2n}(c|_{\delta_i=0}, k', \ell') - T\right|$$

*then the set of windows exceeding $T$ is the same for all $\delta_i \in [-b_i, b_i]$, and in particular the pruning outcome is independent of $\delta_i$.*

**Theorem (Branch-and-Bound Completeness).** *If the branch-and-bound algorithm (Approach 3) terminates with all leaves pruned or verified, then every $b \in B_{n_{\max}, m}$ has $\max_{k,\ell} \text{val}(b, k, \ell) \geq T$, and therefore $c \geq c_{\text{target}}$.*
