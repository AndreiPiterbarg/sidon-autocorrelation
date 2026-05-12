# Probabilistic / Random-Construction Approaches to $C_{1a}$

**Date:** 2026-04-20
**Budget:** 8 fetches, 15 min
**Question:** Can a random $f$ (e.g., i.i.d. heights on partitioned support) give a rigorous lower bound on $C_{1a} = \sup_t (f*f)(t)$?

---

## 100-word summary

No known random-$f$ construction yields a rigorous lower bound on $C_{1a}$ competitive with 1.2802. The probabilistic method in this area (Erdős–Rényi, Cilleruelo–Ruzsa–Vinuesa) produces *upper* bounds on $C_{1a}$ via random $B_h[g]$-set constructions, not lower bounds — because random $f$ has *smaller* sup-autoconvolution by concentration, which is the wrong direction. All serious lower-bound work (Matolcsi–Vinuesa, Cloninger–Steinerberger, Boyer–Li, DeepMind 2508.02803) uses **stochastic optimization** (SA, Adam+noise, gradient ascent) to *discover* deterministic step functions, then verifies rigorously. Random construction + concentration is unlikely to break 1.2802; it is a wild-card at best.

---

## 1. Why random $f$ pushes the *wrong* way

The lower-bound problem is
$$C_{1a} = \inf_{f \geq 0,\ \mathrm{supp}(f)\subset[-1/4,1/4],\ \int f = 1}\ \sup_{|t|\leq 1/2}\ (f*f)(t).$$

If we take $f$ random — e.g., i.i.d. heights $X_1,\dots,X_n$ on $n$ bins of width $1/(2n)$ — then:

- **Mean autoconvolution** at any $t$ is $\mathbb{E}[(f*f)(t)] = $ something close to $\int f \cdot \int f = 1$ (normalized per bin-width by $\approx 2n$).
- **Variance**: $(f*f)(t)$ is a sum of $\Theta(n)$ independent products $X_i X_{j}$ with $i+j$ matching $t$. By concentration (Chernoff/Bernstein), $(f*f)(t)$ concentrates around its mean $\simeq 2$ with std $O(n^{-1/2})$.
- Max over $O(n)$ grid points of $t$: $\sup_t (f*f)(t) \leq \mathbb{E}[\text{mean}] + O(\sqrt{\log n}/\sqrt{n})$.

**Concentration makes random $f$ *flat*, hence **small** $\sup_t$.** A random $f$ witnesses a *small* value of $\max(f*f)$, which would contradict $C_{1a} \geq 1.28$ only if the random $f$ were in the feasible set and its realized max were $<1.28$ — but this gives an **upper** bound on the infimum (a value $\geq C_{1a}$), not a lower bound.

**Conclusion:** Random i.i.d. heights cannot certify $C_{1a} > 1.2802$. They can only certify $C_{1a} \leq \text{something}$.

---

## 2. Literature findings (8 fetches)

### (A) Matolcsi–Vinuesa 2010 ([arXiv:0907.1379](https://arxiv.org/abs/0907.1379))
- Explicit step-function construction, **deterministic**.
- Numerical optimization over step heights to push the upper bound on $C_{1a}$ downward (their target is the *upper* bound $1.5098 \to 1.5029$).
- No random construction, no concentration.

### (B) Cloninger–Steinerberger 2017 ([arXiv:1403.7988](https://arxiv.org/abs/1403.7988))
- Discretization + numerical analysis → $C_{1a} \geq 1.28$.
- Extremizer existence proof uses compactness, not probability.
- No random-$f$ argument.

### (C) DeepMind / Further Improvements 2025 ([arXiv:2508.02803](https://arxiv.org/html/2508.02803))
- **Stochastic optimization** (Adam with Gaussian gradient noise, elitist respawn, JAX on A100) — but to search for a deterministic step function for the **upper bound** on $C_{1a}$ (equivalently lower bound on the reciprocal autoconvolution constant $c$).
- No concentration-based rigorous bound. Noise is exploration heuristic, not proof.

### (D) Boyer–Li 2025 ([arXiv:2506.16750](https://arxiv.org/abs/2506.16750))
- Simulated annealing + gradient descent to find a 575-interval step function.
- Again: stochastic search, deterministic output, rigorous verification afterwards.

### (E) Cilleruelo–Ruzsa–Vinuesa (generalized Sidon / $B_h[g]$)
- **Probabilistic construction of $B_h[g]$ sets** (integer Sidon-type sets).
- Upper bound on density $D(x)$ of discrete analog — random subsets of $[N]$ kept with probability $p$, deletion lemma.
- Translates to an **upper** bound on the continuous autoconvolution sup (wrong direction for our $C_{1a}$ lower bound).

### (F) Erdős–Rényi probabilistic Sidon constructions
- Exist since 1960s. Give $(1-o(1))\sqrt{n}$ density Sidon sets with positive probability.
- Relate to a *different* normalization: max pair-sum multiplicity, not sup of autoconvolution density.
- Transfer to continuous $C_{1a}$ lower bound: **not known**, and the signs work against it (concentration suppresses peaks).

### (G) Lovett-style sparse Fourier decomposition
- Log-rank / XOR function bounds. Not applicable — our $f$ is nonneg on continuum, not boolean.
- No known adaptation to autoconvolution extremal problems.

### (H) Barnard–Steinerberger $L^2$ autoconvolution
- Optimal $L^2$ inequality has explicit extremizer (tent function) — entirely deterministic and exact.
- No probabilistic content.

---

## 3. Could a randomized algorithm yield a rigorous LB via a different mechanism?

### Idea 3.1 — Lovász local lemma / alteration on a grid dual
Use LLL to construct a dual witness (e.g., a Lagrange multiplier / measure $\nu$ on $t$) with probability $>0$, then derandomize. Issue: our problem is a QP/SDP min-max, and LLL is combinatorial. No natural bad-event structure on heights $\mu_i$ that LLL would beat deterministic SDP at.

### Idea 3.2 — Random restriction + union bound on adversarial $t$
Fix a random $f_\omega$, bound $\max_t (f_\omega * f_\omega)(t)$ in distribution. **Anti-concentration** (Paley–Zygmund) would lower-bound $\Pr[\max > \tau]$. But we need $\forall f \colon \max > C_{1a}$, i.e., **min over $f$**. Anti-concentration on a fixed $f$ gives nothing about the worst $f$.

### Idea 3.3 — Randomized B&B / cascade with certified pruning
`cloninger-steinerberger/cpu/` already does deterministic exhaustive B&B. Randomizing the branch order does not change the rigor; could improve speed but not the bound.

### Idea 3.4 — Neural / RL search + rigorous verify (wild card)
Parallel to 2508.02803: use NN to propose candidate dual certificates (measures $\nu_t$, Lagrange multipliers) for the Lasserre SDP, then verify via Farkas (see `project_farkas_certified_lasserre`). This **is** a promising direction but it's "random search for deterministic certificate," not probabilistic-method-in-the-proof.

---

## 4. Verdict

| Approach | Rigor | LB direction | Viable for $>1.2802$? |
|---|---|---|---|
| Random i.i.d. heights on grid | Yes (concentration) | **Wrong way** (upper bound on $\inf_f$) | No |
| LLL / alteration | Unclear | Unclear | Unlikely |
| Random dual search + verify | Yes (verify step) | Correct | **Yes — parallel to existing Lasserre push** |
| Probabilistic $B_h[g]$ transfer | Yes discretely | Continuous analog gives UB | No |
| Anti-concentration on fixed $f$ | Yes for one $f$ | Doesn't cover worst-$f$ | No |

**Recommendation:** Do not pursue random $f$ + concentration for rigorous LB; it is structurally the wrong tool. If probabilistic flavor is desired, use randomized *search* for dual certificates of the existing Lasserre / Farkas pipeline — which is already the plan under `project_farkas_certified_lasserre`.

---

## Sources

- [Matolcsi–Vinuesa, arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [Cloninger–Steinerberger, arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- [Further Improvements (DeepMind), arXiv:2508.02803](https://arxiv.org/html/2508.02803)
- [Boyer–Li, arXiv:2506.16750](https://arxiv.org/abs/2506.16750)
- [Dense sumsets of Sidon sequences, arXiv:2103.10349](https://arxiv.org/pdf/2103.10349)
- [Strong $B_h$ sets and random sets, arXiv:1911.13275](https://arxiv.org/pdf/1911.13275)
- [Continuous Ramsey Theory and Sidon Sets, math/0210041](https://arxiv.org/pdf/math/0210041)
- [Alon–Spencer, The Probabilistic Method (3rd ed.)](https://math.bme.hu/~gabor/oktatas/SztoM/AlonSpencer.ProbMethod3ed.pdf)
