# Boyer-Li 2025 (arXiv:2506.16750) — Ideas for Sidon $C_{1a}$

Boyer-Li improve the **$L^2$ autoconvolution constant** (NOT the $L^\infty$ Sidon constant $C_{1a}$):
$$c := \sup_{f \ge 0,\, f \in L^1 \cap L^2} \frac{\|f*f\|_{L^2}^2}{\|f*f\|_{L^\infty}\,\|f*f\|_{L^1}} \ge 0.901564.$$
Previous records: Matolcsi-Vinuesa (2010) 0.88922 (20 steps); DeepMind AlphaEvolve (May 2025) 0.8962 (50 steps). Boyer-Li use **575 equally spaced steps** on $[0, N]$ with $N = 575$, with integer heights $v_n$ (multiplied by $2^{24}$, rounded) listed in Appendix A of the paper. The exact ratio is 960860894104088840289339769 / 1065770134514315452423544944.

## 1. Algorithm

Discretized objective: given $\mathbf{v} = (v_0,\dots,v_{N-1}) \in \mathbb{R}_{\ge 0}^N$, define
$$L_j := \sum_{n=\max(0,j-N)}^{\min(j-1,N-1)} v_n\, v_{j-n-1}, \qquad j = 0,\dots,2N-1$$
and maximize
$$Q_N(\mathbf{v}) := \frac{\tfrac{1}{3}\sum_{j=0}^{2N-1}(L_j^2 + L_j L_{j+1} + L_{j+1}^2)}{(\max_j L_j)\cdot (\tfrac{1}{2}\sum_j (L_j + L_{j+1}))}.$$
This is **exact** because $f*f$ is piecewise linear on $[j, j+1]$ when $f = \sum v_n \mathbf{1}_{[n,n+1)}$.

Two-phase search:
1. **Simulated annealing (coarse)** with $N=23$. Perturbation scale $S^{(0)} = 25$, decay $0.999$, ~1000 iterations; reflect-to-positive via $|\cdot|$. Restart ~400 times with fine scale $S^{(0)} = 0.05$, decay $0.99998$.
2. **Gradient ascent (refinement)** — upscale $\mathbb{R}^{23} \to \mathbb{R}^{115}$ by 5x replication, then apply
$$\mathbf{V}^{(s)} \leftarrow \mathbf{V}^{(s)} + \frac{0.01}{(i+1)^{1/4}(s+1)^{1/3}}(1 - Q_N(\mathbf{V}^{(s)})) \cdot \frac{\nabla Q_N}{\|\nabla Q_N\|}$$
zero out negatives after each step. $10^6$ independent restarts, ~1000 gradient steps each. Finally upscale $115 \to 575$ and repeat phases. HPC resources at NC State.

## 2. Main inequality

**Theorem 1:** There exists a nonnegative step function $f = \sum_{n=0}^{574} v_n \mathbf{1}_{[n,n+1)}$ with
$$\frac{\|f*f\|_{L^2}^2}{\|f*f\|_{L^\infty}\,\|f*f\|_{L^1}} \ge 0.901564.$$
Problem is translation/dilation invariant (Prop. 2), so the support $[0, 575]$ is immaterial — can rescale to $[-1/4, 1/4]$. Upper bound $c \le 1$ is Hölder; O'Bryant conjectures $c = \log(16)/\pi \approx 0.88254$, but already beaten, so **$c > \log(16)/\pi$ is proved**.

## 3. Applicability to $L^\infty$ problem ($C_{1a}$)

**Direct transfer: NO.** Boyer-Li optimize $\|f*f\|_{L^2}^2 / (\|f*f\|_{L^\infty} \|f*f\|_{L^1})$ — a Hölder-ratio close-to-indicator witness. The Sidon constant is a pure $L^\infty$ lower bound: $\|f*f\|_\infty \ge C_{1a}$ when $\|f\|_1 = 1$, $\text{supp}(f) \subseteq [-1/4,1/4]$. Their result implies nothing directly about $C_{1a}$ because the ratio being close to 1 is consistent with $f*f$ flat and near-indicator — exactly the regime where $\|f*f\|_\infty$ is **small**, not large.

**Indirect relevance:** their optimizer $F_3$ has $F_3 * F_3$ graphically very similar to Matolcsi-Vinuesa's $F_1*F_1$ (inner product $\approx 0.9963$ after normalization), which is the **leading lower-bound construction for $C_{1a}$** (value 1.2748). So the step-function shape Boyer-Li lock onto is in the same family as the $C_{1a}$ extremizers. A two-phase SA + gradient search at $N = 575$ for the *dual* problem $\max_f \|f*f\|_\infty$ under $\|f\|_1 = 1$ is the obvious transposition.

## 4. Computational details

- Discretization: $N = 23 \to 115 \to 575$ equally spaced intervals (upscaling by $5\times$).
- Integer heights after rounding $v \cdot 2^{24}$ — allows **exact rational verification** (ratio given as exact quotient of 27-digit integers). Referee's suggestion, gave a small boost over floats.
- Phase-1 SA: $\sim 1000$ iterations $\times \sim 400$ restarts.
- Phase-2 gradient: $10^6$ restarts $\times \sim 1000$ steps.
- North Carolina State HPC.
- Code: `github.com/zkli-math/autoconvolutionHolder` (`Inital_HPC_code.py`, `Refine_HPC_code.py`, `Verify.nb`, `coeffBL.txt`, `f95.py`).
- $Q_N$ is polynomial of degree 4 in $\mathbf{v}$, with a max/denominator — projected gradient ascent after removing the max via KKT on the argmax index is cheap.

## 5. Transferable ideas for $C_{1a}$

1. **Exact rational arithmetic via $2^{24}$ scaling**: our cascade/Lasserre primal witnesses are in floating point. Scale step heights to integers, compute ratio exactly as a fraction of large integers. This gives a **machine-checkable certificate** analogous to the 27-digit rational here — directly usable for Lean rigorous bounds. Especially valuable for the Farkas-certified Lasserre pipeline.
2. **Exact piecewise-linear $f*f$ on integer grid**: for a nonnegative step function $f = \sum v_n \mathbf{1}_{[n,n+1)}$, $f*f$ is exactly piecewise linear with slopes $\pm 1$ and nodes at integers, so $\|f*f\|_\infty = \max_j L_j$ exactly. This means the $L^\infty$ Sidon problem also reduces to a **finite polynomial optimization** at any step count — no discretization error in the objective, only in the ansatz class. Combined with (1), this is a **rigorous** lower bound: every step-function witness gives an exact rational lower bound on $C_{1a} = 4 \cdot \max L_j / (\sum v)^2$.
3. **Upscale-by-replication warm start** ($N \to 5N$): escape local optima of larger-$N$ landscape by initializing from $N/5$ optimum. We currently solve cascade at fixed $d$; Boyer-Li show the $N=23 \to 115 \to 575$ ladder is practical. For Sidon, $N = 20$ (MV) $\to 100 \to 500$ with fine-grid rescale could push beyond 1.2748.
4. **Simulated annealing + projected gradient two-phase**: our `pod_parallel_search.py` uses basin-hopping; SA with the decay schedule above ($0.999$ coarse, $0.99998$ fine) + projected gradient with $1/(i+1)^{1/4}$ step is cheap and proven to beat AlphaEvolve/LLM-based search on this type of problem. Worth direct replication for $C_{1a}$ maximization — objective is the same "ratio of polynomials in $v$".
5. **Correlation diagnostic** (Section 3): inner product $\langle F_i * F_i, F_j * F_j \rangle$ between candidate autoconvolutions measures how different two witnesses are. Useful tool to confirm our Lasserre/cascade optimizers are in the Matolcsi-Vinuesa basin vs. a genuinely new extremizer.
6. **Comparison with MV + AlphaEvolve coefficients**: `coeffMV` (20 steps), `coeffAE` (50 steps), `coeffBL` (575 steps) in `Verify.nb` — can test our $C_{1a}$ tooling by computing $\|f*f\|_\infty$ on all three and confirming our solvers find the expected values as sanity check.
7. **AlphaEvolve beaten by plain SA/GD**: strong signal that for smooth step-function problems in this dimension, classical continuous optimization outperforms LLM evolutionary search. Supports keeping our cascade/Lasserre approach over reaching for agentic discovery.

## 100-word summary

Boyer-Li improve the $L^2$ autoconvolution constant $c \ge 0.901564$ (from AlphaEvolve's 0.8962) via a 575-step nonnegative step function found by simulated annealing followed by projected gradient ascent with upscaling $N=23 \to 115 \to 575$. Integer heights (scaled by $2^{24}$) give an exact rational certificate. Not directly applicable to the Sidon $L^\infty$ constant $C_{1a}$: their ratio targets closeness-to-indicator, not the supremum $\|f*f\|_\infty$. Transferable: exact-integer step heights for rigorous witnesses, piecewise-linear $f*f$ formula giving finite-dimensional polynomial optimization without discretization error, SA+gradient two-phase search with upscale-by-replication warm starts, and autoconvolution inner-product correlation as basin diagnostic.
