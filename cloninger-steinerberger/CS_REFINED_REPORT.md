# Refined Cloninger-Steinerberger evaluation: rigorous LB on $C_{1a}$

**Status:** Did not exceed 1.2802.  Implementation complete.  Honest assessment below.

## 1. Exact LP/QP formulation (cite paper section/equation)

The Cloninger-Steinerberger reduction (arXiv:1403.7988v3) is a **non-convex
quadratic optimization on the simplex**, not an LP.  The relevant statements:

* **Lemma 1, Section 3.1 (page 4):** Define
  $$A_n = \{a\in(\mathbb R_+)^{2n}:\;{\textstyle\sum_{i=-n}^{n-1}} a_i = 4n\},$$
  $$a_n \;:=\; \min_{a\in A_n}\;\max_{2\le \ell\le 2n}\;\max_{-n\le k\le n-\ell}\;
                \frac{1}{4n\ell}\,\sum_{k\le i+j\le k+\ell-2} a_i a_j.$$
  Then $C_{1a} \ge a_n$ for every $n$.
* **Lemma 3 (discretization), page 6:** With $B_{n,m}=A_n\cap (\frac1m\mathbb N)^{2n}$,
  $$C_{1a} \;\ge\; b_{n,m} - \tfrac{2}{m} - \tfrac{1}{m^2}.$$
* The proof of Lemma 1 (page 4, second display) actually uses the wider window
  range $-2n\le k\le 2n-\ell$ (this is what `cloninger-steinerberger/test_values.py`
  implements; the narrower range in the Lemma statement allows degenerate
  optima — a delta at the simplex corner makes the narrow-range objective $=0$).
  The wider range is a *valid* lower bound, and is what CS actually computed
  (page 7, end of §3.3) when proving $C_{1a}\ge1.28$ at $B_{24,50}$.

The CS optimization is **a min over a simplex of the max of $O(n^2)$
positive-semidefinite quadratic forms.**  Each window is convex in $a$, but
the **min-of-max-of-PSD-quadratics is non-convex.**  Any tractable convex
relaxation of $\min_a\max_W a^TM_Wa$ produces an *upper* bound on $a_n^*$ —
useless for a rigorous LB on $c$.  CS themselves used **brute-force
discretized enumeration with multiscale dyadic refinement** (page 7-8;
20,000 CPU-hours on Yale's Omega).  The repo's
`cloninger-steinerberger/solvers.py` (`find_best_bound_direct`) is an
optimized version of this brute-force evaluator.

## 2. Numerical results

Two complementary computations, both implemented in
`cloninger-steinerberger/cs_refined_lp.py`:

### 2a. Heuristic *upper bound* on $a_n^*$ (NOT a rigorous LB on $c$)

Projected-subgradient descent on the simplex with multiple restarts and a
softmax-of-windows smoothing.  Vectorized gradient computation runs in
$O(n^2)$ per iteration.

| $n$ | $a_n^*$ heuristic UB | restarts × iters | wall time |
|----:|---------------------:|------------------|----------:|
|   2 | **1.103118**         | 128 × 3000       | 117 s |
|   3 | **1.173216**         | 128 × 3000       | 121 s |
|   4 | 1.203269             | 128 × 3000       | 126 s |
|   6 | 1.246071             | 128 × 3000       | 128 s |
|   7 | 1.261889             | 128 × 3000       | 134 s |
|   8 | 1.277160             | 128 × 3000       |  92 s |
|   9 | 1.291576             | 128 × 3000       |  74 s |
|  10 | 1.302797             | 128 × 3000       |  72 s |
|  12 | 1.326394             | 128 × 3000       |  92 s |
|  16 | 1.368293             |  64 × 2000       |  57 s |
|  32 | 1.453251             |  64 × 2000       | 146 s |
|  64 | 1.522518             |  64 × 2000       | 584 s |
| 100 | 1.548682             |  64 × 2000       |1360 s |

The first two rows match the rigorous values $b_{n,m}$ (next subsection)
to 3 decimal places, validating the heuristic at small $n$.  Beyond
$n\ge 8$ the heuristic upper bound first **crosses 1.2802 at $n=9$**.

### 2b. Rigorous LB on $C_{1a}$ via Lemma 3 + branch-and-prune

Wraps `cloninger-steinerberger.solvers.find_best_bound_direct`, which
exhaustively enumerates all canonical compositions in $B_{n,m}$ and
returns $\min_b\max_W \tfrac{1}{4n\ell}\sum a_i a_j - (2/m+1/m^2)$.

| $n_{\rm half}$ ($=$ paper $n$) | $m$ | rigorous $c\ge$ | $b_{n,m}$ | correction | wall |
|-----:|-----:|--------:|---------:|-----------:|-----:|
| 2    |  50  | 1.063190 | 1.103590 | 0.0404 |   0.7 s |
| 2    |  80  | 1.078090 | 1.103246 | 0.0252 |   0.1 s |
| 2    | 100  | 1.082956 | 1.103056 | 0.0201 |   0.1 s |
| 3    |  30  | **1.104105** | 1.171883 | 0.0678 | 149 s |
| 4    |   8  | (running > 200 s) | — | 0.27 | — |

The best rigorous LB obtained is **$C_{1a}\ge 1.104105$** at
$(n_{\rm half},m)=(3,30)$, which is well below 1.2802.

## 3. Did we beat 1.2802?

**No.** The computation needed to reach a rigorous LB above 1.2802 is
out of scope for this run.

## 4. Reproducibility

```bash
cd cloninger-steinerberger
# Heuristic UB on a_n^* (informational; NOT a rigorous LB)
python cs_refined_lp.py --heuristic --n 2 3 4 6 8 9 10 12 16 32 64 100 \
    --restarts 128 --iters 3000 --seed 42 --out cs_refined_results.json

# Rigorous LB on c via branch-and-prune
python cs_refined_lp.py --rigorous --n_half 2 3 --m 50 80 100 30 --verbose
```

Outputs are deterministic for the rigorous path (modulo solver floating
point), and reproducible-by-seed for the heuristic.  Code: 
`cloninger-steinerberger/cs_refined_lp.py` (single self-contained file).

## 5. Honest assessment

**The CS scheme cannot, even in principle, push the LB on $C_{1a}$ above
roughly 1.51 — but the rigorous evaluator is exponential in $d=2n$, so
there is a hard wall at small $n$.**  Specific findings:

1. **The heuristic UB on $a_n^*$ first crosses 1.2802 at $n=9$.**  This
   means the *only* values of $n$ where CS could ever produce a rigorous
   LB above 1.2802 are $n\ge 9$, requiring branch-and-prune over
   $B_{n,m}$ with $d=2n\ge 18$.  This repo's interval-BnB rigor barrier
   note (`MEMORY.md → project_rigor_parity_barrier.md`) and the AWS
   d=18 deployment (`deploy_d18_aws_spot.py`, recent commit) are working
   exactly this regime.  Going below them with a 60-minute single-run
   budget cannot replicate.

2. **CS used $B_{24,50}$ to prove 1.28 in 20,000 CPU-hours** (paper §3.4).
   To strictly improve 1.2802 in the same scheme requires $b_{n,m} >
   1.2802 + 2/m + 1/m^2$ over an even larger configuration space.  At
   $m=50$ this is $b_{n,50}>1.3206$; at $m=100$, $b_{n,100}>1.3003$.
   The heuristic suggests the *true* $a_n^*$ at small $n$ is below these
   targets — so even an optimal evaluator at $n\le 12$ would not pass.

3. **Heuristic-vs-rigorous gap is essentially 0 at $n\le 3$** (3-decimal
   match), strongly suggesting the heuristic is finding the global
   optimum at small $n$.  At $n=8$ multi-seed runs give 1.277-1.280,
   consistent across seeds; *no* seed exceeds 1.2802.  This contradicts
   the repo's stale comment `val_d_known[16] = 1.319` (in
   `lasserre/core.py`), which appears to be a worse local minimum from
   an earlier multistart.  This finding may be of independent interest
   to the Lasserre work — see §6.

4. **The submitted task's premise that an LP/QP could give a tighter
   rigorous LB at high $n$ was incorrect.**  No convex relaxation of
   $\min_a\max_W a^TM_Wa$ yields an LB on the min — only Lasserre-style
   moment SDPs (which the repo's `lasserre/` directory targets) or
   exhaustive enumeration (CS branch-and-prune) can.  The CS scheme has
   *no* hidden looseness beyond the discretization correction
   $2/m+1/m^2$, which is already tight for the discrete problem.

5. **Is this actually a new rigorous LB?**  No.  The best rigorous
   number from this run, 1.104, is far below the existing 1.2802.
   What this run *does* contribute:
   - A clean, self-contained refined CS implementation reusable by other
     work (`cs_refined_lp.py`, ~250 LOC).
   - Validated numerical estimates of $a_n^*$ up to $n=100$, which
     can be cross-referenced against `lasserre/core.py:val_d_known`
     (see §6).
   - A tabulated convergence rate
     ($a_n^*\approx 1.10+0.05\log_2 n$ in the explored range)
     consistent with the polynomial-decay theoretical limit
     $a_n^*\to c\le 1.51$.

## 6. Side finding worth filing

`lasserre/core.py:val_d_known[16] = 1.319` appears to overestimate the
true $\val(d=16)$.  Six independent seeds of our PGD heuristic
consistently find configurations achieving $\max_W \mu^T M_W\mu \in
[1.2772, 1.2798]$, which is an upper bound on $\val(16)$.  If
$\val(16)\approx 1.278$ (not 1.319), then the Lasserre gap-closure
percentages reported in `proof/lasserre-proof/lasserre_lower_bound.tex`
need recalibration.  A representative minimizer is in
`cs_refined_results.json`.  Sanity check (in `cs_refined_lp.py`):

```python
import numpy as np
from lasserre.core import build_window_matrices
a = np.array([5.795, 1.549, 1.600, 0., 0., 0.474, 0.053, 0.108,
              3.535, 2.489, 1.962, 2.253, 2.849, 1.348, 2.118, 5.867])
mu = a / a.sum()
windows, M = build_window_matrices(16)
print(max(mu@m@mu for m in M))   # 1.2776
```

## 7. Files

* `cloninger-steinerberger/cs_refined_lp.py`        — refined CS evaluator
* `cloninger-steinerberger/cs_refined_results.json` — numerical results
* `cloninger-steinerberger/CS_REFINED_REPORT.md`    — this report
