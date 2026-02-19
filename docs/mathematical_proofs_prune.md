● Mathematical Proofs of Pruning Validity

  Background: The Quantity Being Bounded

  The algorithm works with compositions — integer vectors $(c_0, \ldots, c_{d-1})$ with $c_i \geq 0$ and $\sum c_i = S = 4nm$. The bin averages are $a_i = c_i/m$, so $\sum a_i = 4n$.

  The autoconvolution is:
  $$\text{conv}[s] = \sum_{i+j=s} a_i a_j, \qquad s = 0, 1, \ldots, 2(d-1)$$

  The test value is:
  $$\text{tv}(a) = \max_{\substack{2 \leq \ell \leq d \ k}} ; \frac{1}{4n\ell} \sum_{s=k}^{k+\ell-2} \text{conv}[s]$$

  where the window at position $k$ with length $\ell$ covers $\ell - 1$ consecutive conv values.

  A configuration is pruned (ruled out) if $\text{tv}(a) \geq T$, where $T = c_{\text{target}} + 2/m + 1/m^2$ is the effective threshold. Any lower bound on $\text{tv}(a)$ that exceeds $T$ is sufficient to prune.

  The key property we exploit everywhere is:

  Non-negativity: All $a_i \geq 0$, therefore all $\text{conv}[s] \geq 0$, and any subset of terms in $\text{conv}[s]$ is a lower bound on $\text{conv}[s]$.

  ---
  Bound 1: Contiguous Block Sum

  Theorem. For any contiguous block of $p$ bins ${q, q{+}1, \ldots, q{+}p{-}1}$ with block sum $B = \sum_{i=q}^{q+p-1} c_i$:
  $$\text{tv}(a) ;\geq; \frac{B^2}{4n \cdot 2p \cdot m^2}$$

  Proof. Consider the sum of all autoconvolution values from index $2q$ to $2(q+p-1)$:

  $$\sum_{s=2q}^{2(q+p-1)} \text{conv}[s] ;=; \sum_{s=2q}^{2(q+p-1)} ;\sum_{i+j=s} a_i a_j$$

  We split the right side into two groups:

  - Intra-block pairs: Both $i \in [q, q{+}p{-}1]$ and $j \in [q, q{+}p{-}1]$. Their sum-index satisfies $2q \leq i+j \leq 2(q{+}p{-}1)$, so they are captured by this range.
  - Cross-block pairs: At least one of $i, j$ is outside $[q, q{+}p{-}1]$.

  Since every $a_i \geq 0$, the cross-block pairs contribute $\geq 0$. Therefore:

  $$\sum_{s=2q}^{2(q+p-1)} \text{conv}[s] ;\geq; \sum_{\substack{q \leq i \leq q+p-1 \ q \leq j \leq q+p-1}} a_i a_j ;=; \left(\sum_{i=q}^{q+p-1} a_i\right)^{!2} = \frac{B^2}{m^2}$$

  The equality $\sum_{\text{intra}} a_i a_j = (\sum a_i)^2$ holds because summing $a_i a_j$ over all pairs $(i,j)$ in the block (including $i=j$) and over all conv indices that capture them yields every ordered pair exactly   once.

  Window identification: The range $[2q, , 2(q{+}p{-}1)]$ contains $2p - 1$ consecutive conv values. In the test-value formula, a window of length $\ell$ covers $\ell - 1$ consecutive conv values. So this corresponds to    
  $\ell = 2p$, at window start $k = 2q$:

  $$\text{tv}(a) ;\geq; \frac{1}{4n \cdot 2p} \sum_{s=2q}^{2(q+p-1)} \text{conv}[s] ;\geq; \frac{B^2}{4n \cdot 2p \cdot m^2} \qquad \blacksquare$$

  Why existing checks are special cases:
  - $p = 1$ (single bin): $B = c_i$, $\ell = 2$, bound $= c_i^2/(4n \cdot 2 \cdot m^2)$ — this is the existing max-element check.
  - $p = n = d/2$ (half the bins): $B = \text{left_sum}$, $\ell = d$, bound $= \text{left_sum}^2/(4n \cdot d \cdot m^2)$ — this is the existing half-sum check.

  What's new (the intermediate gaps):
  - For D=6: $p=2$ ($\ell=4$) checks pairs ${c_0,c_1}$, ${c_1,c_2}$, ${c_2,c_3}$, ${c_3,c_4}$, ${c_4,c_5}$. Of these, ${c_0,c_1}$ and ${c_4,c_5}$ are subsets of the existing $\ell=6$ half-sum, but the interior pairs        
  ${c_1,c_2}$, ${c_2,c_3}$, ${c_3,c_4}$ are entirely new.
  - For D=6: $p=3$ ($\ell=6$) checks triples. The edge triples ${c_0,c_1,c_2}$ and ${c_3,c_4,c_5}$ are the existing half-sum checks, but interior triples ${c_1,c_2,c_3}$ and ${c_2,c_3,c_4}$ are new.
  - For D=4: $p=2$ ($\ell=4$) interior pair ${c_1,c_2}$ is new.

  Why these catch configs the existing checks miss: A configuration can have balanced left/right halves (surviving the half-sum check at $\ell=d$) and no single dominant element (surviving max-element at $\ell=2$), yet have   a large concentration in a middle block like ${c_2,c_3}$. The intermediate block-sum checks catch these.

  ---
  Bound 2: Two-Max Enhanced $\ell=2$

  Theorem. Let $c_{p_1}$ and $c_{p_2}$ be the two largest values in ${c_0, \ldots, c_{d-1}}$, with $p_1 \neq p_2$. Then:
  $$\text{tv}(a) ;\geq; \frac{2 \cdot c_{p_1} \cdot c_{p_2}}{4n \cdot 2 \cdot m^2}$$

  Proof. Consider the single conv entry at index $s = p_1 + p_2$:

  $$\text{conv}[p_1 + p_2] ;=; \sum_{i+j ,=, p_1+p_2} a_i a_j$$

  Since $p_1 \neq p_2$, the ordered pairs $(p_1, p_2)$ and $(p_2, p_1)$ are distinct and both satisfy $i + j = p_1 + p_2$. Their contribution is:

  $$a_{p_1} a_{p_2} + a_{p_2} a_{p_1} ;=; 2 , a_{p_1} a_{p_2} ;=; \frac{2 , c_{p_1} c_{p_2}}{m^2}$$

  All other terms in the sum have $a_i a_j \geq 0$, so:

  $$\text{conv}[p_1 + p_2] ;\geq; \frac{2 , c_{p_1} c_{p_2}}{m^2}$$

  A single conv entry is an $\ell = 2$ window (covering $\ell - 1 = 1$ conv value):

  $$\text{tv}(a) ;\geq; \frac{\text{conv}[p_1 + p_2]}{4n \cdot 2} ;\geq; \frac{2 , c_{p_1} c_{p_2}}{4n \cdot 2 \cdot m^2} \qquad \blacksquare$$

  Combined with existing max-element bound:
  $$\text{tv}(a) ;\geq; \frac{\max\bigl(c_{p_1}^2,; 2,c_{p_1} c_{p_2}\bigr)}{4n \cdot 2 \cdot m^2}$$

  The two-max term dominates when $2,c_{p_1} c_{p_2} > c_{p_1}^2$, i.e., when $c_{p_2} > c_{p_1}/2$. This happens for configurations where mass is spread among the top two elements rather than concentrated in one.

  Why this is effective: The existing max-element bound scales as $(\max c_i)^2$, which is weak when no single bin dominates. But if the top two elements each hold a significant share, their cross-term $2,c_{p_1} c_{p_2}$  
  can be much larger than $c_{p_1}^2$. In the extreme, if $c_{p_1} = c_{p_2} = x$, the two-max bound gives $2x^2$ vs. the existing $x^2$ — a factor of 2 improvement.

  ---
  Bound 3: Cross-Term $c_0 \cdot c_{d-1}$

  Theorem. For any pair $(i, j)$ with $i \neq j$:
  $$\text{tv}(a) ;\geq; \frac{2,c_i , c_j}{4n \cdot 2 \cdot m^2}$$

  Applied specifically as $2,c_0,c_5$ for D=6.

  Proof. Identical to Bound 2 with the specific pair $(i, j) = (0, d{-}1)$:

  $$\text{conv}[0 + (d{-}1)] = \text{conv}[d{-}1] = \sum_{i'+j'=d-1} a_{i'} a_{j'} ;\geq; 2,a_0,a_{d-1} = \frac{2,c_0,c_{d-1}}{m^2}$$

  The $\ell=2$ window at $k = d-1$ gives:

  $$\text{tv}(a) ;\geq; \frac{2,c_0,c_{d-1}}{4n \cdot 2 \cdot m^2} \qquad \blacksquare$$

  Why this specific pair matters and isn't redundant with Bound 2: In the CUDA kernel, Bound 2 (two-max) already catches any pair where both elements are the top two. The value of the $c_0 \cdot c_5$ check is that it fires 
  in the inner loop where $c_5$ is freshly computed and $c_0$ is known — it's a single multiply and compare, extremely cheap, and targets a specific class of configs:

  Canonical form enforces $c_0 \leq c_5$. Configs that survive the asymmetry test have roughly balanced left/right sums. Among these, "edge-heavy" configs have large $c_0$ and $c_5$ but moderate interior elements. Such     
  configs can survive:
  - Max-element: if $c_5$ isn't the single dominant element (moderate $c_5$)
  - Two-max: if the top-two product isn't large enough (e.g., when $c_5$ is max but the second-largest is some interior element, not $c_0$)
  - Half-sum: left and right halves are balanced

  But $c_0 \cdot c_5$ can still be substantial. This check catches these edge-heavy balanced configs.

  ---
  Bound 4: Sum-of-Squares Pigeonhole

  Theorem.
  $$\text{tv}(a) ;\geq; \frac{\sum_{i=0}^{d-1} c_i^2}{d \cdot 4n \cdot 2 \cdot m^2}$$

  Proof. The diagonal entries of the autoconvolution are the values $\text{conv}[2i]$ for $i = 0, \ldots, d{-}1$. Each satisfies:

  $$\text{conv}[2i] = \sum_{j+k=2i} a_j a_k ;\geq; a_i \cdot a_i = a_i^2 = \frac{c_i^2}{m^2}$$

  The inequality holds because the $j = k = i$ term contributes $a_i^2$, and all other terms (pairs with $j \neq k$ but $j + k = 2i$) are $\geq 0$.

  Summing over all $d$ diagonal entries:

  $$\sum_{i=0}^{d-1} \text{conv}[2i] ;\geq; \frac{1}{m^2}\sum_{i=0}^{d-1} c_i^2$$

  By the pigeonhole principle, at least one diagonal entry achieves at least the average:

  $$\max_{0 \leq i \leq d-1} \text{conv}[2i] ;\geq; \frac{1}{d} \sum_{i=0}^{d-1} \text{conv}[2i] ;\geq; \frac{\sum c_i^2}{d \cdot m^2}$$

  Each diagonal entry $\text{conv}[2i]$ is a single conv value — an $\ell = 2$ window:

  $$\text{tv}(a) ;\geq; \frac{\max_i \text{conv}[2i]}{4n \cdot 2} ;\geq; \frac{\sum c_i^2}{d \cdot 4n \cdot 2 \cdot m^2} \qquad \blacksquare$$

  Why this is weak (not implemented in CUDA): The $1/d$ factor from pigeonhole loses a factor of $d = 2n$ (6 for D=6). By Cauchy-Schwarz, $\sum c_i^2 \geq S^2/d$, so this bound is at most $S^2/(d^2 \cdot 4n \cdot 2 \cdot   
  m^2) = (4nm)^2/(d^2 \cdot 8nm^2) = 2n/d^2$, which is small. It only helps when mass is very uniformly distributed, but such configs are already caught by the block-sum bounds. The benchmarks confirmed 0 additional prunes,   so this was implemented only in Python for completeness.

  ---
  Integer Threshold Correctness (CUDA Safety)

  In fused_prove_target, the host precomputes:

  $$\texttt{int_thresh}[\ell-2] ;=; \bigl\lfloor T \cdot m^2 \cdot 4n \cdot \ell \cdot (1 - 4\varepsilon) \bigr\rfloor$$

  where $\varepsilon = $ DBL_EPSILON $\approx 2.2 \times 10^{-16}$. The $(1 - 4\varepsilon)$ factor guards against FP64 rounding producing a value slightly above the true mathematical product — it makes the threshold       
  conservative (lower), meaning fewer configs pruned, never more.

  The exact mathematical pruning condition for a window of length $\ell$ with integer sum $W = \sum_{\text{window}} c_i c_j$ is:

  $$\frac{W}{4n \cdot \ell \cdot m^2} > T \quad\Longleftrightarrow\quad W > T \cdot 4n \cdot \ell \cdot m^2$$

  Since $\texttt{int_thresh}[\ell-2] \leq T \cdot 4n \cdot \ell \cdot m^2$, the check $W > \texttt{int_thresh}[\ell-2]$ is a sufficient condition — it may miss a few borderline configs (conservative), but never falsely     
  prunes.

  For each new bound, the connection to integer thresholds:

  ┌──────────────────────┬───────────────────────────────┬───────────────┬──────────────────────────────────┐
  │        Bound         │ Lower bound on window sum $W$ │ Window $\ell$ │          Integer check           │
  ├──────────────────────┼───────────────────────────────┼───────────────┼──────────────────────────────────┤
  │ Block-sum ($p$ bins) │ $(\sum_{\text{block}} c_i)^2$ │ $2p$          │ bs*bs > local_thresh[2p-2]       │
  ├──────────────────────┼───────────────────────────────┼───────────────┼──────────────────────────────────┤
  │ Two-max              │ $2 \cdot \max_1 \cdot \max_2$ │ 2             │ 2*max_c*max2_c > local_thresh[0] │
  ├──────────────────────┼───────────────────────────────┼───────────────┼──────────────────────────────────┤
  │ Cross-term $c_0 c_5$ │ $2 \cdot c_0 \cdot c_5$       │ 2             │ 2*c0*c5 > local_thresh[0]        │
  └──────────────────────┴───────────────────────────────┴───────────────┴──────────────────────────────────┘

  All comparisons are exact integer arithmetic — no floating-point error. Combined with the conservative threshold, this makes the integer checks strictly safe: they never prune a configuration whose true test value is     
  below $T$.

  ---
  Chain of Validity

  Putting it all together, the logical chain for each pruned configuration is:

  1. A mathematical theorem establishes: some specific subset of terms in some specific window sum $\geq$ our bound value
  2. Since all $c_i \geq 0$, the full window sum $\geq$ the subset sum $\geq$ our bound value
  3. Our bound value exceeds the (conservative) integer threshold
  4. Therefore the full window sum exceeds the threshold
  5. Therefore the test value exceeds $T$
  6. Therefore this configuration is ruled out

  No step introduces approximation error. The FP32 checks in fused_find_min use the same conservative threshold pattern as the existing code (acceptable for the heuristic minimum-finding pass). The mathematically rigorous  
  fused_prove_target kernel uses exact integer comparisons throughout.