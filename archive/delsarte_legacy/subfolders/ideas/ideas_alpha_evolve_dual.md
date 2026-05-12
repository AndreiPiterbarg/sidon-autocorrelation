# Redirecting AlphaEvolve / FunSearch / TTT-Discover to the DUAL Side of $C_{1a}$

**Date:** 2026-04-20
**Status:** Scoping / idea memo. No code yet.
**Upstream papers surveyed:**

- AlphaEvolve (Georgiev, Gomez-Serrano, Tao, Wagner, Novikov et al.), arXiv:2511.02864
- TTT-Discover, arXiv:2601.16175
- ThetaEvolve, arXiv:2511.23473
- FunSearch (Romera-Paredes et al., *Nature* 2023)
- Min-overlap "Further Improvements to the Lower Bound …", arXiv:2508.02803 (White / step-function LP)
- AlphaEvolve Problem Repository (DeepMind GitHub)

---

## 0. Why this memo exists

All three recent evolutionary-LLM systems that have touched our problem or a neighbour
(AlphaEvolve on $C_6$, ThetaEvolve on the first autocorrelation inequality $C_1$,
TTT-Discover on Erdős' minimum overlap) have attacked the **upper bound**. They evolve
a **primal construction** $f \in \mathcal{F}$ and score it by the objective
$\sup_{|t|\le 1/2}(f*f)(t)$.

Concretely, ThetaEvolve's scoring function for $C_1$ is literally

```
score = 2 * n * max_b / (sum_a ** 2)         # lower = better, UB on C_1
```

over step-function heights `a[0..n-1]` with `b = autoconv(a)` (v1 of the paper, §4.2).
This is a **primal minimiser**: smaller score $\Rightarrow$ tighter upper bound.

Nobody has pointed the machinery at the **dual / lower-bound** side. That is the gap
this memo argues we can exploit. The current lower bound $C_{1a}\ge 1.2802$ comes from
Matolcsi–Vinuesa style LP duals over low-moment / autoconvolution test functions, and
from the Cloninger–Steinerberger (C-S) combinatorial cascade. Both are dual-style
witnesses; both are amenable to evolutionary program search in exactly the same sense
that primal step-function heights are.

---

## 1. Specific modification: what to evolve, and how to score

### 1.1 What the dual looks like for $C_{1a}$

The cleanest dual witness is a pair $(K, G)$ of test functions (Matolcsi–Vinuesa-style)
such that for every admissible $f$,

$$\int (f*f) \cdot K \;\le\; \|f\|_1^2 \cdot G\bigl(\text{moment data of } f\bigr),$$

which, after the $\int f = 1$ normalisation, rearranges into a lower bound of the form

$$\sup_{|t|\le 1/2}(f*f)(t) \;\ge\; \frac{\int(f*f)K}{\sup_t K(t)} \;\ge\; c^\star(K,G).$$

Our own `delsarte_dual/family_f{1..4}_*.py` already implements four parametric
families ($F_1$ Selberg, $F_2$ Gauss-poly, $F_3$ Vaaler, $F_4$ MV-arcsine). Each
family exposes an `evaluate(params) -> float` that returns the certified lower
bound from that particular $(K,G)$. **That is already a FunSearch-compatible
oracle.** It is deterministic, fast (< 1 s), and monotone in solution quality.

### 1.2 What to feed the evolutionary loop

Three layers, in increasing order of ambition:

1. **Parametric search (FunSearch-lite).** Evolve the parameter vector
   $\theta$ inside a *fixed* family (say $F_4$). Score = `family_f4_mv_arcsine.evaluate(θ)`.
   Equivalent to what BasinHopping already does in `mv_delta_u_scan.py`; the
   value of LLM evolution here is re-parameterisation hints, not raw optimisation.

2. **Program search (FunSearch-classic / AlphaEvolve).** Evolve *the Python
   function that returns $K$ and $G$* given a small symbolic toolbox
   (polynomials, arcsine, Selberg block, dilations, convolutions). This is the
   direct analogue of FunSearch's `priority()` in cap-set, or ThetaEvolve's
   `solve()` returning step-function heights. **Key change vs ThetaEvolve:** the
   output is not the primal $f$, but the kernel $K$; the evaluator solves the
   **inner LP** (`inner_moment_lp.py`) to certify the induced lower bound.

3. **SDP-aware program search.** Evolve code that emits a Lasserre dual slice:
   a sum-of-squares (SOS) multiplier polynomial $\sigma(x,t)$ for the
   Putinar certificate attached to our `lasserre/` pipeline. Score = the
   rational lower bound obtained after the Farkas round-off
   (see `project_farkas_certified_lasserre.md`: val(4) $> 1.0963$ already
   certified this way). This is the only route that directly sharpens the
   rigorous lower bound without introducing a second proof layer.

### 1.3 Score function (Delsarte-style dual, concrete)

```python
def score_dual_kernel(K_code: str, G_code: str) -> float:
    """
    Lower-bound score for a candidate dual pair (K, G).
    Higher is better. Infeasible certificates return -inf.
    Wall-clock budget: ~30 s / candidate.
    """
    K = compile_symbolic(K_code)           # must be <= 1 everywhere, supp ⊂ [-1/2, 1/2]
    G = compile_symbolic(G_code)           # concave majorant of moment data
    if not verify_positivity(K, G):
        return float('-inf')
    lb = inner_moment_lp.solve(K, G)       # rigorous LP / SDP lower bound on C_1a
    return lb                              # e.g. 1.2815 > 1.2802 is a win
```

`verify_positivity` should be an interval-arithmetic Chebyshev expansion (we
already have `lasserre_mosek_cheby.py`) — same rigor tier as the Farkas pipeline
so any evolved winner is directly publishable.

---

## 2. Implementation sketch

### 2.1 Pick the skeleton

ThetaEvolve is the best fit because it is:

- open source (paper explicitly provides repo),
- already configured for autocorrelation scoring (trivial to swap the scorer),
- runs on a single 8B model, so budget is tractable.

AlphaEvolve's repo ships problems but not the agent. FunSearch's repo
likewise lacks the LLM + sandbox layers. TTT-Discover requires Tinker API
(Thinking Machines) which is paid-but-affordable ($\sim$few hundred USD/problem).

### 2.2 Recommended stack

| layer | choice | rationale |
|---|---|---|
| evo loop | fork ThetaEvolve | already has island model, lazy penalties, RL hooks |
| LLM | DeepSeek-R1-0528-Qwen3-8B (default) + Claude-4.7/Opus for "explore" prompts | small model for bulk; frontier model for re-seeding stagnant islands |
| RL | ThetaEvolve's test-time SFT/GRPO hooks, 8 × A100-80G | proven in the paper |
| scorer | new `delsarte_dual/score_dual.py` wrapping `inner_moment_lp.py` + `verify.py` | rigorous LB only |
| sandbox | reuse ThetaEvolve's subprocess/timeout harness | 30 s cap / candidate |
| certification | auto-emit Lean stub via existing `certified_lasserre/` pipeline when a new record is found | keeps bounds publishable |

Estimated integration effort: **1–2 days** for layer-1 (parametric), **3–5
days** for layer-2 (program evolution of $K$), **1–2 weeks** for layer-3
(SOS-multiplier evolution integrated with Lasserre / Farkas).

### 2.3 Prompt design (layer 2)

System prompt seeds the LLM with:

- problem statement (CLAUDE.md first paragraph),
- Matolcsi–Vinuesa template `K(x) = Σ c_i · arcsine_block_i(x)`,
- worked example: current $F_4$ implementation achieving 1.2802,
- four failure modes to avoid (non-symmetric $K$, unbounded $G$, moment mismatch,
  spurious positivity gap).

Mutation prompt: "Here are the top-3 programs and their lower bounds. Propose
one structural modification that might push the bound past 1.2802."

---

## 3. Required compute

- **Layer 1 (parameter evolution).** Trivial. 1 GPU-day on a 3090 is enough;
  our existing `pod_parallel_search.py` already does equivalent work.
- **Layer 2 (program evolution).** Match ThetaEvolve's published setup:
  8 × A100-80G for ~72 h, B=32, 16 responses per prompt, 1150 s timeout.
  Estimated cloud cost: ~\$1.5–3k on Lambda / RunPod at current A100 rates.
- **Layer 3 (SOS / Lasserre-dual evolution).** Scorer itself (Mosek /
  Clarabel on d=8–16 with $\sigma$ multipliers) is ~30–120 s. Same GPU cluster
  but with an extra 16-core CPU node for the SDP calls. Budget: ~\$4–8k for a
  two-week run.

Total, end-to-end, to obtain a publishable bound lift: **\$5–10k and 2–3
weeks of wall-clock.** That is *inside* the typical AlphaEvolve/ThetaEvolve
campaign budget.

---

## 4. Expected bound lift — realistic vs aspirational

Three calibration datapoints:

1. ThetaEvolve on $C_1$ upper bound: 1.5032 $\to$ 1.50286 ($\Delta \approx 4\times 10^{-4}$).
2. AlphaEvolve on min-overlap upper bound: previous best $\to$ 0.380924; TTT-Discover to 0.380876 ($\Delta \approx 5\times 10^{-5}$).
3. Our own Farkas-certified Lasserre already gives val(4) $> 1.0963$; the
   analytic gap from 1.2802 to the conjectured true value $\approx$ 1.4–1.5
   is *much* larger than the gaps LLM search has chewed through on upper bounds.

Calibrated expectation:

| layer | expected lift | probability |
|---|---|---|
| 1 — parametric re-tune within $F_4$ / $F_{1..3}$ | +$10^{-4}$ to $+10^{-3}$ | 80% |
| 2 — new parametric family $K$ | **+$10^{-3}$ to $+10^{-2}$** (1.2802 → 1.28–1.29) | 40–50% |
| 3 — SOS multiplier evolution on Lasserre d=16 | **+$10^{-2}$ to $+10^{-1}$** (1.28 → 1.30–1.38) | 15–25% |

The asymmetry matters: upper-bound campaigns saturate because the true value
is known to be $\ge 1.2802$, leaving only $\sim$0.22 of headroom. The
**lower-bound side has ~0.22 of headroom too, but untouched by LLM search**,
so even a layer-2 campaign is materially more promising than any further
layer-2 attempt on the upper bound.

---

## 5. 100-word summary

AlphaEvolve, ThetaEvolve, and TTT-Discover have all pushed *upper* bounds
for autocorrelation constants by evolving primal step functions against a
scalar objective. The dual side — Matolcsi–Vinuesa $(K,G)$ test pairs,
Delsarte-type kernels, and Lasserre SOS multipliers — has an identically
shaped search space and an identically cheap scorer (our
`inner_moment_lp.py` plus interval verification), yet has never been
targeted. Forking ThetaEvolve, swapping the scorer to the rigorous LP/SDP
lower bound, and evolving either $K$-programs (layer 2) or SOS multipliers
(layer 3) is a direct \$5–10k / 2–3 week campaign with a realistic shot at
pushing $C_{1a}$ past 1.2802 for the first time since 2010.

---

## 6. Appendix: why this is unusually well-suited to LLM search

- Dual certificates are **small programs** (arcsine + polynomial + dilation
  composites), exactly what FunSearch/AE excel at.
- The scorer is a **cheap, deterministic convex solve**, not a Monte-Carlo
  estimate.
- Rigor is **automatically preserved** by interval arithmetic + Farkas round-off;
  no need for a separate AlphaProof stage.
- The search space is **untouched** by the recent wave of campaigns, so
  diminishing-returns doesn't apply.

Recommended next step: stand up layer 1 against $F_4$ this week as a
sanity-check that the scorer wiring is rigorous, then escalate to layer 2 with
ThetaEvolve's fork if the smoke test passes.
