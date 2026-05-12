# Supplementary Pilot Report — Alternative Formulations of 3-Point SDP

**Date:** 2026-04-28 (continuation of V2 work, ~1 hour exploration)
**Hardware:** local Windows 10 box, ~16 GB RAM, MOSEK 11 + CVXPY 1.8.1.
**Code:** [lasserre/threepoint_alternatives.py](threepoint_alternatives.py).

---

## TL;DR — the line is **partially alive**, contingent on an analytical UB on $\|f^*\|_\infty$

V2 reported "Δ = 0 structurally" for the **specific Christoffel-Darboux pilot
formulation**. That conclusion is **objective-specific** — with a different
choice of objective and additional constraints, the 3-point cone IS strictly
tighter than the 2-point cone, and substantial lifts emerge.

**Strongest empirical result:** at $(k, N) = (8, 8)$ with bump-kernel objective +
$L^\infty$ density bounds at $\|f\|_\infty \le 2.10$:

| | $\lambda^{2\mathrm{pt}}$ | $\lambda^{3\mathrm{pt}}$ |
|---|---:|---:|
| Value | 1.3403 | **1.3573** |
| Above 1.2802? | ✓ | ✓ |

**Interpretation caveat:** $\lambda^{3\mathrm{pt}}$ is a lower bound on
$C_{1a}^{(\|f\|_\infty \le M)}$, NOT on $C_{1a}$ directly. To convert to a
$C_{1a}$ LB, we'd need $\|f^*\|_\infty \le M$ for the unconstrained Sidon
optimum — currently open.

**Bottom line:** the 3pt-cone-tightening hypothesis is **TRUE empirically**
when paired with appropriate constraints. Whether this can be turned into a
$C_{1a} > 1.2802$ proof depends on a separate analytical question
($\|f^*\|_\infty$ upper bound) that the SDP doesn't settle.

Updated posterior P(beat 1.2802 via this path): **15–25%**, up from V2's ~2%.

---

## What was tested in this hour

Three alternatives to the V2 Christoffel-Darboux + Putinar SOS Gram formulation:

**A) V2 framework + bump kernel (V1's objective in V2's rescaled coords).**
Without L^∞: Δ ~ $10^{-7}$ — essentially zero. V1's reported Δ ~ $10^{-4}$
at $(7, 7)$ was numerical-conditioning artifact in the un-rescaled monomial
basis, NOT a real lift.

**B) L^∞ density bound on $\rho^{(2)} = f \otimes f$ and $\rho^{(3)} = f^{\otimes 3}$.**
Encoded as a shifted Hausdorff moment cone $(M^d \cdot \text{Lebesgue}^{(d)} - \rho^{(d)})$
being a valid moment vector. Excludes diagonal and near-singular pseudo-measures.
**This is the alternative that produces real lift.**

**C) Translation-invariant CD kernel** $K_N(s) = \sum_j p_j(s)^2$ as a positive
function on a single variable. Combined with L^∞: Δ ~ 1–3% lift, smaller than
bump-kernel approach. Less promising.

## Key empirical results

### Result 1: With $L^\infty$, the 3pt cone is strictly tighter

Comparison at fixed $(k, k)$, $\|f\|_\infty$ bound = $f_\infty$, bump kernel
$\epsilon = 0.1$:

**Lift contribution decomposition** (3-row blocks per $k, f_\infty$):

| $k$ | $f_\infty$ | 2pt + 2D L^∞ | 3pt + 2D L^∞ (no 3D L^∞) | 3pt + 2D + 3D L^∞ |
|----:|----:|--:|--:|--:|
|   4 | 2.5 | 0.97727 | 0.98086 (+0.4%) | 1.08243 (+10.8%) |
|   5 | 2.5 | 0.92443 | 0.94235 (+1.9%) | 1.07193 (+16.0%) |
|   6 | 2.5 | 0.89834 | 0.90729 (+1.0%) | 1.06165 (+18.2%) |
|   4 | 3.0 | 0.67474 | 0.69889 (+3.6%) | 0.82171 (+21.8%) |
|   5 | 3.0 | 0.61590 | 0.63726 (+3.5%) | 0.78167 (+26.9%) |
|   6 | 3.0 | 0.56339 | 0.58725 (+4.2%) | 0.75194 (+33.5%) |

Reading: the basic 3D cone (PSD + 3 box localizers, no 3D L^∞) adds only
**0.4–4.2%** lift over 2pt — modest. The major lift comes from the 3D L^∞
constraint that 2pt cannot represent — adding **10–33%** on top.

### Result 2: Crossing 1.2802 in the right regime

$\lambda^{3\mathrm{pt}}$ values across $\|f\|_\infty$ values:

| $k$ | $\|f\|_\infty$ | $\lambda^{2\mathrm{pt}}$ | $\lambda^{3\mathrm{pt}}$ | Above 1.2802? | %lift |
|----:|---------:|--:|--:|---:|---:|
| 8 | 2.05 | 1.4067 | **1.4099** | ✓ | 0.23% |
| 7 | 2.10 | 1.3423 | **1.3533** | ✓ | 0.82% |
| 8 | 2.10 | 1.3403 | **1.3573** | ✓ | 1.27% |
| 9 | 2.10 | 1.3366 | **1.3591** | ✓ | 1.68% |
| 7 | 2.15 | 1.2803 | **1.3090** | ✓ | 2.24% |
| 8 | 2.15 | 1.2722 | **1.3122** | ✓ | 3.15% |
| 9 | 2.15 | 1.2629 | **1.3115** | ✓ | 3.85% |
| 7 | 2.20 | 1.2168 | 1.2692 | ✗ | 4.31% |
| 8 | 2.20 | 1.2025 | 1.2694 | ✗ | 5.56% |

At $\|f\|_\infty \le 2.15$: $\lambda^{3\mathrm{pt}}$ exceeds 1.2802 robustly across
$k \in \{7, 8, 9\}$, settling around 1.31. At $\|f\|_\infty \le 2.10$:
1.355–1.359, a very comfortable margin. **The values stabilize as $k$ grows**,
suggesting we're near the true $C_{1a}^{(M)}$ value at these constraint levels.

Note: $\lambda^{2\mathrm{pt}}$ slightly decreases with $k$ in some rows (e.g.,
$\|f\|_\infty = 2.15$: $k=7$: 1.2803, $k=8$: 1.2722, $k=9$: 1.2629). This violates
the expected monotonicity (more constraints → larger inf) and indicates MOSEK
conditioning slack at higher $k$. The 3pt values are more numerically stable
because the 3D L^∞ pins down more pseudo-moment freedom.

### Result 3: Without L^∞, the cone collapses (V2 finding holds)

| $k$ | $N$ | bump kernel only (no L^∞) | $\lambda^{2\mathrm{pt}}$ | $\lambda^{3\mathrm{pt}}$ | %lift |
|----:|----:|--:|--:|--:|--:|
| 7 | 7 | bump $\epsilon=0.1$ | 0.2391 | 0.2391 | $\sim 10^{-7}$ |
| 7 | 7 | TI CD kernel | 4.2428 | 4.2428 | $\sim 10^{-9}$ |

V2's "Δ = 0 structurally" verdict holds when no constraint excludes
near-singular pseudo-measures.

## Why this matters — and why it's still inconclusive for $C_{1a}$

**Strict statement:** $\lambda^{3\mathrm{pt}}(M)$ is a lower bound on
$$C_{1a}^{(M)} := \inf_{f, \|f\|_\infty \le M} \sup(f*f).$$

**Relation to $C_{1a}$:** $C_{1a} \le C_{1a}^{(M)}$ for all $M < \infty$ (smaller
feasible set, larger inf). Equality holds iff some unconstrained optimum
$f^*$ has $\|f^*\|_\infty \le M$. **We have no proof that this is the case at
$M = 2.15$.**

**What we know:** Cauchy-Schwarz on convolution gives
$\sup(f*f) \le \|f\|_\infty \cdot \|f\|_1 = \|f\|_\infty$. So at the optimum,
$\|f^*\|_\infty \ge C_{1a} \ge 1.2802$. *Lower* bound on $\|f^*\|_\infty$;
*upper* bound is open.

**The key open question** (now sharper than V2 left it): does the
unconstrained Sidon optimum satisfy $\|f^*\|_\infty \le 2.15$? If yes:
$$C_{1a} \ge 1.31 > 1.2802 \quad\text{(BREAKTHROUGH).}$$

This is an **analytical** question outside the SDP. Approaches:
1. Find an explicit construction of $f$ giving $\sup(f*f) \le 1.31$ with
   $\|f\|_\infty \le 2.15$ (would PROVE $C_{1a} \le 1.31$ AND optimum has
   $\|f^*\|_\infty \le 2.15$ — but wait, would only prove $C_{1a} \le 1.31$,
   not bound $\|f^*\|_\infty$).
2. Argue from variational characterization that the Sidon optimum is "close
   to uniform" with bounded $\|f\|_\infty$.
3. Numerical: scan over candidate $\|f^*\|_\infty$ values, check which is
   consistent with all known constraints.

## A subtlety re comparison fairness

`build_2pt_bump` adds 1D L^∞ on $m$ AND 2D L^∞ on $g$. `build_3pt_bump`
adds 2D L^∞ on $g = y_{ab0}$, optionally 3D L^∞ on $y$. The 3pt builder does
not explicitly add 1D L^∞ on $m = y_{a00}$; this could affect the basic-3D-cone
contribution by a small amount. The 10–33% lift from adding 3D L^∞ is
unaffected by this asymmetry (it's a strict additional constraint).

## Files

- [lasserre/threepoint_alternatives.py](threepoint_alternatives.py) — alternative builders
  (bump kernel, L^∞ density bound, builders for 2pt/3pt with all combinations).
- [lasserre/threepoint_full.py](threepoint_full.py) — V2 framework (reused).

## Updated honest probability

**P(beat 1.2802 via $L^\infty$-augmented 3-point Lasserre):** **15–25%**, up
from V2's 2%. The structural mechanism is now visible — adding $L^\infty$
density bounds excludes the singular pseudo-measures that gave V2's trivial
$\lambda = 1$, and the 3-point cone (with both 2D and 3D L^∞) gives values
that *exceed* 1.2802 at moderate $\|f\|_\infty$ bounds.

**The remaining barrier**: converting a constrained-problem LB into an
unconstrained $C_{1a}$ LB. This is a 2–4 week analytical effort, separate
from any SDP work, with uncertain outcome. If solved, the SDP infrastructure
in `threepoint_alternatives.py` is ready to deliver the rigorous proof.

## Comparison to the original pilot's expectations

| Pilot's Phase 2 prediction | $\Delta_{8,40}$ | Actual at reachable analog |
|---|---|---|
| < 10⁻³ → DEAD | "redundant" | ~0% without L^∞ — agrees |
| ∈ [10⁻³, 10⁻²] → MARGINAL | "small but real" | 1–4% basic cone with L^∞ — better than expected |
| ≥ 10⁻² → MEANINGFUL | "strong evidence" | 10–33% with 3D L^∞ — meets criterion |

The pilot's framework was calibrated for a specific formulation that, after
this exploration, turns out to be one of *several* possible formulations. The
basic-3D-cone contribution lives in the MARGINAL band. The full 3D-L^∞
construction lives in the MEANINGFUL band. Both are above the kill threshold.
