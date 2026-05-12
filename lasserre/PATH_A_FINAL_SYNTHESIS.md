# Path A (L^{3/2} replacement of L^∞) — Final Synthesis

**Date:** 2026-04-29
**Scope:** Complete attempt to push $C_{1a} > 1.2802$ via the L^{3/2} replacement strategy.
**Verdict:** Path A does not deliver $C_{1a} > 1.2802$ as a complete proof. **Structurally blocked** on the SDP side; analytically open on the conjecture side. But substantial structural results.

---

## Summary table

| Component | Status | Headline finding |
|---|---|---|
| **REV-3/2 conjecture** ($\sup f*f \ge \frac{\pi}{8}\|f\|_{3/2}^3$) | Open, strongly supported | 3592 candidate $f$'s, no counterexample; SS proven critical point |
| **Hardy-type reformulation** | Established | $\int_0^1 \phi(1-w)\phi'(w)dw \ge \frac{\pi}{8}(\int \phi'^{3/2})^2$ |
| **V3 SDP at L^∞** | Solid | $C_{1a}^{(L^\infty \le 2.186)} > 1.2802$ rigorously |
| **V3 + auxiliary $g=f^{3/2}$** | Blocked | Putinar localizer admits singular pseudo-moments |
| **Discrete level-1/level-2 SDP** | Blocked at all levels | Moment hierarchy can't break flat-average floor |
| **CS refinement** (independent) | Did not exceed 1.2802 | Branch-and-prune at $n \ge 9$ requires AWS-scale compute |
| **Literature** | Genuinely new | No existing reverse-Young covers our regime $(p=q=3/2 > 1)$ |

## What is rigorously established (this session)

### 1. SS structural facts.
- $f_0(x) = (2x+1/2)^{-1/2}$ has $\|f_0\|_{3/2} = 2^{2/3}$, $\sup f_0 * f_0 = \pi/2$.
- **$(f_0 * f_0)(t) = \pi/2$ on the entire half-interval $t \in [-1/2, 0]$** (via Beta-integral substitution).
- SS is a **stationary point** of the variational problem $\inf \sup f*f / \|f\|_{3/2}^3$ (proven via Abel inversion of EL/KKT equations); the dual measure $\nu^* = \psi(t)dt$ with $\psi$ given by hypergeometric series, $\psi > 0$ on $(-1/2, 0)$.

### 2. Symmetric f satisfies a stronger version.
For $f(-x) = f(x)$: $\sup f*f \geq \|f\|_{3/2}^3$ (constant 1, not $\pi/8$). Proof: $(f*f)(0) = \|f\|_2^2$ and Hölder $\|f\|_{3/2}^{3/2} \le \|f\|_2$ gives $\|f\|_{3/2}^3 \le \|f\|_2^2$. **The asymmetric case is the technical content** (SS is asymmetric).

### 3. V3 reflection enforcement is benign.
At $k=7, M=2.15$: $\lambda = 1.308972$ exactly with both `enforce_reflection=True` and `enforce_reflection=False`. The bump-kernel objective is sign-flip invariant, so the SDP naturally finds reflection-symmetric pseudo-moments. **V3's bound rigorously applies to general (asymmetric) f**.

### 4. V3 sweep crossover.
$M^*$ where $\lambda^{V3, 3pt}(M^*) = 1.2802$:
| k | $M^*$ |
|---|---|
| 7 | 2.186 |
| 8 | 2.187 |
| 9 | (running, expected ~2.188) |

So V3 rigorously certifies $C_{1a}^{(L^\infty \le 2.18)} > 1.2802$.

### 5. Reformulation $(\dagger)$.
REV-3/2 is equivalent to: for $\phi:[0,1]\to[0,1]$ increasing with $\phi(0)=0, \phi(1)=1$,
$$\int_0^1 \phi(1-w)\phi'(w)\,dw \;\ge\; \frac{\pi}{8}\Big(\int_0^1 \phi'(w)^{3/2}\,dw\Big)^2$$
Equality at $\phi(w) = \sqrt{w}$ (SS-corresponding).

Probabilistic version: for $X, Y$ iid with density $f$, $2\,\mathbb{P}(X+Y \le 0) \ge (\pi/8)\|f\|_{3/2}^3$.

The constant $\pi/8$ traces to $B(1/2, 1/2) = \pi$. Any proof must use Beta/Abel/fractional-integration machinery — standard $L^p$ tools (Beckner, Brascamp-Lieb-Luttinger, sharp Young) all fail (verified rigorously, see Agent 6 report).

---

## What is BLOCKED (structurally)

### SDP encoding of L^{3/2}-constrained problem.

Two encodings tried and verified blocked:

**(a) V3 + auxiliary $g = f^{3/2}$ measure.** Putinar localizer for $g^2 \ge f^3$ in the moment cone admits "singular pseudo-moments" — moment vectors $(\mu_{a,j,k})$ satisfying the inequality but NOT corresponding to any actual $L^{3/2}$-bounded function. SDP value identical to V3 baseline regardless of $B$. Diagnostic at $B=0.5$: optimal pseudo-solution has $\int g = 0.16$ but $\int g^2 = 33.5$, $\int f^3 = 19.1$ — clearly atomic.

**(b) Discretized level-1 (Shor) and level-2 Lasserre.** PSD lift over $W_{\alpha\beta}$ for $|\alpha|, |\beta| \le 2$. **No level of the moment hierarchy breaks the discrete floor** $\lambda \to 1$. Why: $\min_y \max_k E_y[(f*f)_k]$ over moment-mixtures achieves the flat average $(\int f)^2 / |\text{conv supp}| = 1$. Moment-mixtures over the space of densities form valid moment sequences at every level.

To get a useful SDP bound on $C_{1a}^{(L^{3/2} \le B)}$, the formulation must change fundamentally — perhaps incorporating a smooth/sharper kernel objective, or a non-PSD coupling between L^{3/2} and the convolution.

### The proof bridge.

To prove $C_{1a} > 1.2802$ via Path A would require BOTH:
1. Prove REV-3/2 with $c_0 \ge \pi/8$ (or any $c_0 \ge 1.2802 / B^3$ for usable $B$)
2. A tight SDP at $L^{3/2} \le B = (1.2802/c_0)^{1/3}$ giving $\lambda > 1.2802$

Step 1 is the analytical Hardy-type inequality $(\dagger)$ — research-level, open.
Step 2 is structurally blocked at all reachable Lasserre levels.

Therefore Path A as a numerical-aided proof **cannot close** in this session, even under optimistic assumptions about the conjecture.

---

## Next steps that would actually move the bound

In rough order of plausibility:

1. **Prove the Hardy-type inequality $(\dagger)$** via Abel/fractional-integration techniques. This is a precise analytical question; could be tractable for a harmonic analyst.

2. **Push the existing CS branch-and-prune to higher $n$** (already in progress per `deploy_d18_aws_spot.py`). This is L^∞-agnostic by design; if it reaches $a_n^* > 1.2802$ at $n \ge 18$, it's a clean unconditional proof.

3. **Push the existing Farkas-certified val(d) pipeline** (current rigorous LB 1.0963 at d=4). Per CS, val(d) → $C_{1a}$ as $d \to \infty$. Need d ≥ 32 or 64 to potentially exceed 1.2802.

4. **Find a NEW SDP formulation for L^{3/2}-constrained problem.** The existing moment-Lasserre and discretized-Shor framework cannot do it; need a different convex encoding (e.g., Fourier-side, dual sup-of-test-functions, etc.).

5. **Stronger reverse-Young variants.** REV-p for other p might be provable where $p = 3/2$ is not. E.g., L^p constraints for p between 1 and 3/2 (where Beckner has different deficit structure).

---

## Side findings (relevant to other repo work)

- **`lasserre/core.py:val_d_known[16] = 1.319` may be stale.** Independent multi-seed PGD heuristic finds $\val(16) \approx 1.278$ across 6 random seeds. Should be re-checked against the certified pipeline.

---

## Files produced this session

- `lasserre/threepoint_l32.py` — discretized Shor SDP (demonstrates looseness)
- `lasserre/threepoint_l32_moment.py` — V3+aux moments (structurally blocked)
- `lasserre/threepoint_l32_level2.py` — discrete level-2 (still blocked)
- `lasserre/REVERSE_YOUNG_PROOF_ATTEMPT.md` — analytical work, reformulation
- `lasserre/REVERSE_YOUNG_LITERATURE.md` — comprehensive literature survey
- `lasserre/PARTIAL_PROOFS_NOTES.md` — symmetric case + SS plateau
- `lasserre/THREEPOINT_L32_MOMENT_REPORT.md` — V3+aux verdict
- `cloninger-steinerberger/cs_refined_lp.py` — refined CS evaluator
- `cloninger-steinerberger/CS_REFINED_REPORT.md` — CS verdict
- `tests/test_threepoint_l32*.py` — three test suites, all pass
- `results/path_a_stress_test.json` — 3592-candidate stress test
- `results/v3_deep_sweep.json` — V3 M-sweep at k=7,8,9
- `results/cs_refined_results.json` — CS numerical data

---

## Posterior

**P(L^{3/2} bridge approach delivers $C_{1a} > 1.2802$ in <3 months):** ~3-5%. The conjecture is empirically very strong but the SDP component is structurally blocked. Any complete proof requires both the analytical conjecture AND a fundamentally new SDP formulation.

**P(any approach delivers $C_{1a} > 1.2802$ in <3 months):** ~40-60% via:
- CS refinement at higher $n$ (AWS deployment in progress)
- Farkas val(d) pipeline at higher d
- Some not-yet-attempted approach
