    # Repairing the Pre-filter: Valid Lower Bounds for Pruning

    ## Context

    The branch-and-prune algorithm (Cloninger–Steinerberger, arXiv:1403.7988) proves lower bounds on the autoconvolution constant $c$ by exhaustively checking that every discrete mass distribution $b \in B_{n,m}$ has at least one windowed autoconvolution average exceeding the threshold $T = c_{\text{target}} + \frac{2}{m} + \frac{1}{m^2}$. The full test value for a window $(k, \ell)$ is the bilinear form:

    $$\text{val}(b,k,\ell) = \frac{1}{4n\ell}\sum_{\substack{i,j \in \{-n,\ldots,n-1\} \\ k \leq i+j \leq k+\ell-2}} b_i b_j$$

    where we write $b = (b_{-n}, \ldots, b_{n-1})$ for the components of the discrete mass distribution (corresponding to the bin averages $a_i$ in the continuous setting; in the discretized algorithm we work directly with $b \in B_{n,m}$, so $b_i \geq 0$ and $\sum b_i = 4n$). The sum runs over all **ordered** pairs $(i,j)$ independently (so $(i,j)$ and $(j,i)$ are counted separately when $i \neq j$). Throughout this document we use $b_i$ for the discrete components being tested.

    Computing this exactly for every configuration and every window is expensive. A **pre-filter** provides a cheap lower bound on $\text{val}(b,k,\ell)$: if the lower bound already exceeds $T$, the configuration is eliminated without evaluating the full bilinear form. Since the lower bound is always $\leq$ the true value, a valid pre-filter never falsely eliminates a configuration (soundness is preserved).

    This is Part 2a of the improved algorithm described in `improved_autoconvolution_algorithm.md`. The goal is to reduce the number of configurations that require full (expensive) bilinear evaluation, without sacrificing the rigor of the proof.

    ---

    ## The Bug in the Original Pre-filter

    The original proposal in Part 2a applied Cauchy–Schwarz to the bilinear sum as follows. Define $S(k,\ell) = \{i \in \{-n,\ldots,n-1\} : \exists\, j \in \{-n,\ldots,n-1\} \text{ with } k \leq i+j \leq k+\ell-2\}$ (the set of all bin indices that contribute to at least one pair in window $(k,\ell)$), and claim:

    $$\frac{1}{4n\ell}\sum_{\substack{i,j:\, k \leq i+j \leq k+\ell-2}} b_i b_j \;\geq\; \frac{1}{4n\ell} \cdot \frac{\left(\sum_{i \in S} b_i\right)^2}{|S|}$$

    **This bound is invalid.** The left-hand side sums $b_i b_j$ over a restricted set of ordered pairs — those $(i,j)$ satisfying $k \leq i+j \leq k+\ell-2$. This is *not* the full Cartesian product $S \times S$, so the sum does not equal $\left(\sum_{i \in S} b_i\right)^2$. The standard Cauchy–Schwarz inequality $\sum x_i^2 \geq (\sum x_i)^2/N$ applies to sums of squares, not to bilinear forms over restricted pair sets.

    **Counterexample.** Take $n=2$, with bins indexed $\{-2,-1,0,1\}$, and bin values $b = (3, 3, 1, 1)$ (i.e., $b_{-2}=3$, $b_{-1}=3$, $b_0=1$, $b_1=1$, so $\sum b_i = 8 = 4n$). Consider window $k=0, \ell=2$, so the constraint is $0 \leq i+j \leq 0$, i.e., $i+j = 0$.

    The contributing ordered pairs are:

    | $(i,j)$ | $b_i b_j$ |
    |---|---|
    | $(-1, 1)$ | $3 \cdot 1 = 3$ |
    | $(0, 0)$ | $1 \cdot 1 = 1$ |
    | $(1, -1)$ | $1 \cdot 3 = 3$ |

    (The pair $(-2, 2)$ does not contribute because index $2 \notin \{-2,-1,0,1\}$.)

    **Actual test value:**
    $$\text{val} = \frac{3+1+3}{4 \cdot 2 \cdot 2} = \frac{7}{16} = 0.4375$$

    **The set $S$** consists of all $i$-indices that participate in at least one contributing pair: $i=-1$ (paired with $j=1$), $i=0$ (paired with $j=0$), and $i=1$ (paired with $j=-1$). So $S = \{-1, 0, 1\}$, $|S| = 3$, and $\sum_{i \in S} b_i = 3 + 1 + 1 = 5$.

    **The invalid bound:**
    $$\frac{(\sum_{i \in S} b_i)^2}{4n\ell \cdot |S|} = \frac{25}{16 \cdot 3} = \frac{25}{48} \approx 0.5208$$

    Since $0.5208 > 0.4375$, the claimed lower bound **exceeds** the actual test value. If used as an elimination filter, this would falsely eliminate a configuration whose true test value is below the threshold, invalidating the proof.

    ### Why the bound fails conceptually

    The full product $\left(\sum_{i \in S} b_i\right)^2 = \sum_{i,j \in S} b_i b_j$ sums over all $|S|^2 = 9$ ordered pairs in $S \times S$. But only 3 of these 9 pairs satisfy $i + j = 0$. The constraint surface $\{(i,j) : i+j \in [k, k+\ell-2]\}$ is a narrow anti-diagonal band in the $i$-$j$ plane, and most of $S \times S$ lies outside it. Mass can be concentrated on indices whose pairwise sums fall outside the window, making the restricted bilinear form much smaller than $(\sum b_i)^2$.

    Below are four valid replacements. Each provides a rigorous lower bound on $\text{val}(b,k,\ell)$. All proofs rely on the same principle: since all $b_i \geq 0$, summing over any subset of contributing pairs gives a valid lower bound on the full sum.

    ---

    ## Fix 1: Factored Rectangle Bound (Recommended)

    **Key idea.** Choose a Cartesian product $A \times B$ of index intervals such that every pair $(i,j) \in A \times B$ satisfies the window constraint. Then the restricted bilinear form is at least the product of two independent interval sums.

    **Note on generality.** The proof below works for *arbitrary* subsets $A, B$, not just contiguous intervals. We restrict to intervals for computational efficiency: this enables $O(n^2)$ optimization via prefix sums. The tightness of this bound is therefore "best among interval-based bounds," not best among all possible subset pairs. In practice, if mass is concentrated in two non-adjacent bins, the optimal non-contiguous $A$ could give a tighter bound than any interval. For the parameter regimes we target, the interval restriction is a reasonable tradeoff.

    **Statement.** For any intervals $A, B \subseteq \{-n, \ldots, n-1\}$ satisfying $\min(A) + \min(B) \geq k$ and $\max(A) + \max(B) \leq k + \ell - 2$:

    $$\text{val}(b,k,\ell) \geq \frac{1}{4n\ell}\left(\sum_{i \in A} b_i\right)\left(\sum_{j \in B} b_j\right)$$

    **Proof.** The sumset constraint on the endpoints guarantees that for every $(i,j) \in A \times B$:
    $$\min(A) + \min(B) \leq i + j \leq \max(A) + \max(B) \implies k \leq i + j \leq k + \ell - 2.$$

    Therefore $A \times B$ is a subset of the set of all contributing pairs. Since every term $b_i b_j \geq 0$:

    $$\sum_{\substack{i,j:\, k \leq i+j \leq k+\ell-2}} b_i b_j \;\geq\; \sum_{(i,j) \in A \times B} b_i b_j \;=\; \left(\sum_{i \in A} b_i\right)\left(\sum_{j \in B} b_j\right) \qquad \blacksquare$$

    **Computational cost.** With prefix sums $P[r] = \sum_{i=-n}^{r} b_i$ precomputed in $O(n)$, each interval sum is $O(1)$. The optimization over interval endpoints $A = [a_1, a_2]$, $B = [b_1, b_2]$ subject to $a_1 + b_1 \geq k$ and $a_2 + b_2 \leq k + \ell - 2$ is a four-variable discrete optimization. For fixed $A$, the optimal $B$ is determined in $O(1)$: take $B$ as wide as the constraints allow, since all $b_j \geq 0$ means widening $B$ can only increase $\sum_{j \in B} b_j$. This reduces the search to $O(n^2)$ choices of $(a_1, a_2)$, each evaluated in $O(1)$. **Total: $O(n^2)$ per window with prefix sums.**

    **Verification on the counterexample.** $n=2$, $b=(3,3,1,1)$, window $k=0,\ell=2$ (constraint $i+j=0$):

    - Best feasible choice: $A = \{-1\}$, $B = \{1\}$.
    - Constraint check: $\min(A)+\min(B) = -1+1 = 0 \geq k = 0$ ✓ and $\max(A)+\max(B) = -1+1 = 0 \leq k+\ell-2 = 0$ ✓.
    - Bound: $\frac{1}{16} \cdot 3 \cdot 1 = \frac{3}{16} = 0.1875$.
    - Actual value: $\frac{7}{16} = 0.4375$.
    - Bound $\leq$ actual ✓

    (The invalid bound gave $\approx 0.5208 > 0.4375$. This valid bound correctly does not exceed the true value.)

    ---

    ## Fix 2: Squared Interval Bound (Special Case of Fix 1)

    When $A = B$, the sumset condition $A + A \subseteq [k, k+\ell-2]$ can be satisfied by choosing the interval centered around the midpoint of the window:

    $$A = B = \left\{i \in \{-n,\ldots,n-1\} : \left\lceil k/2 \right\rceil \leq i \leq \left\lfloor (k+\ell-2)/2 \right\rfloor\right\}$$

    This gives:

    $$\text{val}(b,k,\ell) \geq \frac{1}{4n\ell}\left(\sum_{i \in A} b_i\right)^2$$

    **Why it's valid.** This is Fix 1 with $A = B$. The sumset condition is automatic:
    - $\min(A) + \min(A) = 2\lceil k/2 \rceil \geq k$ ✓
    - $\max(A) + \max(A) = 2\lfloor(k+\ell-2)/2\rfloor \leq k+\ell-2$ ✓

    **Edge case: $A = \emptyset$.** When $\ell = 2$ and $k$ is odd, $\lceil k/2 \rceil > \lfloor k/2 \rfloor$, so $A = \emptyset$ and the bound evaluates to $0^2/(4n\ell) = 0$ (trivially valid). This occurs because $i+j = k$ with $k$ odd has no diagonal solutions ($2i$ is always even). Fix 1 still provides a nontrivial bound in these cases by choosing $A \neq B$ (e.g., $A = \{(k-1)/2\}$, $B = \{(k+1)/2\}$).

    **Computational cost.** $O(1)$ per window with prefix sums (interval endpoints are determined by $k$ and $\ell$). Simpler to implement but weaker than the optimized two-interval version.

    ---

    ## Fix 3: Diagonal + Cauchy–Schwarz (Closest to the Original Formula)

    This is the closest valid analogue to what was originally attempted. It applies Cauchy–Schwarz correctly by restricting to the **diagonal** of the bilinear form, where $j = i$.

    **Statement.** Let $D = \{i \in \{-n,\ldots,n-1\} : k \leq 2i \leq k+\ell-2\}$ (indices whose diagonal pair $(i,i)$ contributes). Then:

    $$\text{val}(b,k,\ell) \geq \frac{(\sum_{i \in D} b_i)^2}{4n\ell \cdot |D|}$$

    **Proof.** Two steps:

    1. **Drop off-diagonal pairs** (valid since all $b_i b_j \geq 0$):
    $$\sum_{\substack{i,j:\,k \leq i+j \leq k+\ell-2}} b_i b_j \;\geq\; \sum_{i \in D} b_i^2$$

    2. **Apply Cauchy–Schwarz** to the sum of squares (now correctly a sum of squares, not a general bilinear form):
    $$\sum_{i \in D} b_i^2 \;\geq\; \frac{(\sum_{i \in D} b_i)^2}{|D|} \qquad \blacksquare$$

    This has the form $(\text{mass})^2 / (4n\ell \cdot |S|)$ resembling the original proposal, but with $S$ restricted to the diagonal set $D$ rather than all of $S$. It is **no tighter** than Fix 2: both use the same index set (since $D = A$ from Fix 2), but Fix 2 computes $(\sum b_i)^2 / (4n\ell)$ while Fix 3 divides by an additional factor of $|D|$. (When $|D| = 1$ — which occurs for every $\ell = 2$ window and for $\ell = 3$ with $k$ even — the two bounds coincide.)

    **Edge case: $D = \emptyset$.** When $\ell = 2$ and $k$ is odd, $D = \emptyset$ and the formula becomes $0/0$. Implementations must handle this by convention: define the bound as $0$ when $D = \emptyset$ (trivially valid since $\text{val} \geq 0$). Fix 2 avoids this issue because $(\sum_{i \in \emptyset} b_i)^2 = 0$ with no division by $|D|$.

    **Computational cost.** $O(1)$ per window with prefix sums.

    ---

    ## Fix 4: Partial Convolution Evaluation

    Compute the exact convolution $(b * b)[m] = \sum_{\substack{i+j=m \\ i,j \in \{-n,\ldots,n-1\}}} b_i b_j$ for a few strategically chosen values of $m$ in $[k, k+\ell-2]$, then:

    $$\text{val}(b,k,\ell) \geq \frac{1}{4n\ell} \sum_{m \in M} (b*b)[m]$$

    for any subset $M \subseteq \{k, k+1, \ldots, k+\ell-2\}$. This is exact on the selected anti-diagonals and a valid lower bound overall (dropping non-negative terms from the remaining anti-diagonals).

    **When to use.** Good when the convolution is expected to peak at a few predictable values of $m$ (e.g., near $m = 0$ for symmetric configurations). Each convolution value $(b*b)[m]$ costs $O(n)$ to compute, so evaluating $|M|$ values costs $O(|M| \cdot n)$.

    ---

    ## Comparison

    | Property | Fix 1 (Rectangle) | Fix 2 (Squared) | Fix 3 (Diagonal + C-S) | Fix 4 (Partial Conv.) |
    |---|---|---|---|---|
    | Tightness | Best among interval bounds (optimized $A,B$) | Good | No tighter than Fix 2 (factor of $\|D\|$ lost) | Can be tight |
    | Cost per window | $O(n^2)$ with prefix sums | $O(1)$ with prefix sums | $O(1)$ with prefix sums | $O(\|M\| \cdot n)$ |
    | Implementation | Moderate | Simple | Simplest | Simple |
    | $D = \emptyset$ handling | N/A (uses $A,B$ independently) | Clean ($0^2 = 0$) | Requires $0/0 = 0$ convention | N/A |
    | False eliminations? | **Never** | **Never** | **Never** | **Never** |

    **Relationship between the fixes.** Fix 2 is a special case of Fix 1 (with $A = B$ centered at $k/2$). Fix 3 is no tighter than Fix 2: both use the same index set (since $D = A$ from Fix 2), but Fix 3 divides by an additional factor of $|D|$; equality holds when $|D| = 1$. Fix 2 dominates Fix 3 at the same $O(1)$ cost, making Fix 3 redundant in practice (see Practical Recommendation below). Fix 1 with optimized $A \neq B$ can capture mass distributions where the two "halves" contributing to the convolution are concentrated in different regions. Note that Fix 1 optimizes over contiguous intervals for efficiency; the underlying proof is valid for arbitrary subsets, so non-contiguous choices could be tighter in principle.

    ---

    ## Practical Recommendation

    **Use a multi-pass cascade with increasing precision and cost:**

    1. **Pass 1** (cheapest — $O(1)$ per window): Apply Fix 2 (Squared Interval Bound) to all windows of all surviving configurations. Eliminate any configuration where this bound exceeds $T$ for some window. Fix 2 dominates Fix 3 at the same cost (same index set, no division by $|D|$), so there is no reason to use Fix 3 as Pass 1.
    2. **Pass 2** (moderate — $O(n^2)$ per window): Apply Fix 1 with optimized interval pairs to surviving configurations.
    3. **Pass 3** (exact — full bilinear evaluation): Only for configurations surviving both filters.

    **Soundness.** Every bound in the cascade is a valid lower bound: Fix 2 $\leq$ Fix 1 $\leq$ true value. A configuration is eliminated only when its test value provably exceeds $T$ for some window. No valid candidate is ever lost, so the completeness invariant of the branch-and-prune algorithm is preserved.

**Formal statement.** Let $\text{LB}(b,k,\ell)$ denote any of the four lower bounds above. Then:

$$\text{LB}(b,k,\ell) \leq \text{val}(b,k,\ell) \quad \text{for all } b, k, \ell.$$

If $\text{LB}(b,k,\ell) \geq T$ for some window $(k,\ell)$, then $\text{val}(b,k,\ell) \geq T$, and configuration $b$ is correctly eliminated. The proof's correctness argument (Lemma 1 and Lemma 2 of arXiv:1403.7988) is unaffected: the pre-filter only removes configurations that would pass the full test anyway.

The LP relaxation and SDP relaxation described in `improved_autoconvolution_algorithm.md` (Part 2a) remain valid as additional pruning layers on top of this cascade, applied to parents that survive the pre-filter before enumerating their refinements.
