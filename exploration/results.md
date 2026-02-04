# Results Summary

Best known bounds: $C_{1a} \in [1.2802, 1.5029]$.

## Upper Bound Methods (Primal)

Methods that produce feasible solutions $f \geq 0$, $\|f\|_1 = 1$ and report $\|f*f\|_\infty$.

### LP Iteration (Matolcsi & Vinuesa 2010)

- **Best loss: 1.5123** (600 pieces, random init)
- This is the classical method from [MV10]. Iteratively solves LPs to improve the step function.
- Closest to the published best of 1.5029 (which used $n=208$ with more careful tuning).

### SQP / Prox-Linear

- **Best loss: 1.5168** (1000 pieces)
- Increasing from 600 to 1000 pieces gave modest improvement.

### L-BFGS with Weak Wolfe Line Search

- **Best loss: 1.5380**
- 100k iterations, history size 1000, default weak Wolfe parameters.
- Exited line search loop after 2000 failed iterations.
- Warm starting from previous solutions did not help much.

### Gaussian Mixture

- **Best loss: 1.545**
- $f$ parameterized as mixture of Gaussians; $f*f$ is then also a mixture of Gaussians.
- Max evaluated on a grid (no closed-form for $\|f*f\|_\infty$).
- Optimized means, variances, and mixture weights.

### Polyak Subgradient on Simplex

- **Best loss: 1.5200** (P=50, exact=1.520036)
- Source: `primal_optimizer.ipynb`, Strategy B
- Polyak step with target values 1.50/1.49/1.48, 50k iterations, 10 restarts per target.
- Consistently the best strategy in the notebook across all $P$ values tested.
- Full sweep results (exact peak autoconvolution):

| P   | Loss     |
|-----|----------|
| 10  | 1.569309 |
| 20  | 1.542988 |
| 30  | 1.529321 |
| 50  | 1.520036 |
| 75  | 1.525705 |
| 100 | 1.524759 |
| 150 | 1.522328 |
| 200 | 1.520114 |

Best at P=50 (1.520036), with P=200 close behind (1.520114). Non-monotonic in P due to local minima.

### Adam with Reduce-on-Plateau Scheduler

- **Best loss: 1.5754**
- Source: manual runs (limited tuning budget)
- "Reduce on plateau" type scheduler gave the best performance among Adam variants.

### L-BFGS-B with Random Restarts

- **Best loss: ~1.584** (P=10, 150 restarts)
- Source: `primal_optimizer.ipynb`, Strategy A
- Softmax reparametrization of the simplex constraint.
- Generally worse than Polyak subgradient; 1.58--1.63 range across P values.

### Basin Hopping + L-BFGS-B

- **Best loss: ~1.575** (P=10)
- Source: `primal_optimizer.ipynb`, Strategy C
- 200 basin hopping iterations with L-BFGS-B local minimizer.
- 1.57--1.60 range; better than plain L-BFGS-B but worse than Polyak.

### Total Variation Surrogate

- **Best loss: 1.6574**
- Source: manual runs
- Bounded $\|f*f\|_\infty$ by the total variation of $f*f$ (plus endpoint value).
- In practice, minimized the $\ell_1$ norm of the discrete derivative to encourage a partly flat $f*f$.
- Surrogate is too loose; the max remains high.

---

## Lower Bound Methods (Dual / Certification)

Methods attempting to certify $C_{1a} \geq \text{something}$ from below.

### Shor + RLT SDP Relaxation (Spatial Domain)

- **Best bound: ~1.01** (P=50)
- Source: `SDP_certification.ipynb`
- The SDP bound exactly matches the theoretical floor $2P/(2P-1) \to 1$ as $P$ grows.
- Rank of $X^*$ equals $P$ (full rank) at every $P$ tested --- maximally loose.
- The relaxation spreads mass uniformly across anti-diagonals, which no rank-1 solution can achieve.
- The returned $h$ from the SDP gives loss 2.2889 --- not useful as a primal solution either.
- **Conclusion: Shor+RLT is structurally too weak.** The gap is not numerical; no solver tuning or additional RLT cuts can help. Answers K3 negatively.

### Fourier-Domain SDP (Cosine Basis + Fejer-Riesz)

- **Best bound: 2.0** (trivial, all K tested: 2--30)
- Source: `fourier_sdp.ipynb`
- Cosine basis restricts to even functions, for which $\|f*f\|_\infty = \|f\|_2^2 \geq 2$ always.
- The true $C_{1a} \approx 1.50$ requires non-even $f$. This formulation can never see that.
- Secondary issue: aperiodic autoconvolution of compactly supported $f$ is not a trigonometric polynomial, so Fejer-Riesz characterization is only approximate.
- **Conclusion: Even-function restriction makes this formulation uninformative.**

### Spatial-Domain SDP (Shor + RLT, Fourier Notebook Version)

- **Best bound: ~1.01** (P=50)
- Source: `fourier_sdp.ipynb` (Section 3)
- Same $2P/(2P-1) \to 1$ behavior as the dedicated SDP certification notebook.
- Confirms the Shor relaxation fails because it linearizes each quadratic form independently.

### Lasserre Level-2 (Moment/SOS Hierarchy)

- Source: `lasserre_level2.ipynb`
- Lifts to fourth-order moments via $M_2$ matrix; localizing matrices enforce $\eta \geq p_k(x)$ pointwise.
- Uses binary search on $\eta$ (bilinear structure prevents direct optimization).
- Tested for P = 2--6. Moment matrix size grows as $\binom{P+2}{2}$, limiting scalability.
- Expected to improve over Shor but results at small P are not yet conclusive.

---

## Summary Table (Upper Bounds, Best per Method)

| Method                          | Best Loss | Pieces | Notes                        |
|---------------------------------|-----------|--------|------------------------------|
| LP iteration (MV10)             | 1.5123    | 600    | Closest to published 1.5029 |
| SQP / Prox-linear               | 1.5168    | 1000   |                              |
| Polyak subgradient               | 1.5200    | 50     | Best in notebook sweep       |
| L-BFGS (weak Wolfe)             | 1.5380    | --     | 100k iters, history=1000     |
| Gaussian mixture                 | 1.545     | --     | Grid-evaluated max           |
| Adam (reduce on plateau)        | 1.5754    | --     | Limited tuning               |
| Basin hopping + L-BFGS-B        | ~1.575    | 10     |                              |
| L-BFGS-B (random restarts)      | ~1.584    | 10     | 150 restarts                 |
| Total variation surrogate        | 1.6574    | --     | Surrogate too loose          |

## Summary Table (Lower Bounds)

| Method                    | Best Bound | Notes                                    |
|---------------------------|------------|------------------------------------------|
| Known (literature)        | 1.2802     | Fourier kernel method                    |
| Shor + RLT SDP            | ~1.01      | $2P/(2P-1) \to 1$, structurally useless  |
| Fourier SDP (cosine)      | 2.0        | Trivial (even functions only)            |
| Lasserre level-2          | TBD        | Tested P=2--6, scalability limited       |
