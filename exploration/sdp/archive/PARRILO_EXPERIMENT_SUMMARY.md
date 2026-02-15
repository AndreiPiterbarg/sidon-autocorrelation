# Parrilo Copositivity Experiments — Summary

## Problem
V(P) = min_{x ∈ Δ_P} max_{k=0,...,2P-2} 2P · x^T A_k x, where A_k are convolution matrices.
Known: C_1a = lim V(P) ∈ [1.2802, 1.5029].

## Round 1: Exploring the landscape (Exp 1–10)

### Exp 1: Primal copositivity per-k — WRONG FORMULATION
Requiring η·J − 2P·A_k to be copositive per-k bounds min_x min_k q_k(x), not the minimax.
**Result:** Dead end (formulation error).

### Exp 2: Moment-3 strengthening — BEATS SHOR ✓
Added degree-3 moments y3[i,j,l] with the constraint:
  `2P · Σ_{i+j=k} y3[i,j,l] ≤ η · x[l]`  (multiply convolution bound by x_l ≥ 0)

| P  | Shor   | Moment-3 | Lasserre-2 |
|----|--------|----------|------------|
| 5  | 1.1111 | 1.3425   | 1.6327     |
| 8  | 1.0667 | 1.2556   | 1.5483     |
| 10 | 1.0526 | 1.2321   | 1.5248     |

**Key finding:** Beats Shor at every P. Fast (3s at P=10).

### Exp 3: M3 + convolution localizing matrices — SIGNIFICANT IMPROVEMENT ✓
Added degree-4 moments + PSD localizing matrices L_k for each convolution constraint.

| P  | Shor   | M3+Loc  | Lasserre-2 |
|----|--------|---------|------------|
| 5  | 1.1111 | 1.4729  | 1.6327     |
| 8  | 1.0667 | 1.3774  | 1.5483     |

**Key finding:** Roughly doubles improvement over Shor compared to M3 alone.

### Exp 4: M3 + product localizing — FURTHER IMPROVEMENT ✓
Added x_i·x_j ≥ 0 product localizing matrices (PSD constraint on y4[i,j,a,b]).

| P  | Shor   | Conv+Prod | Conv only | Lasserre-2 |
|----|--------|-----------|-----------|------------|
| 5  | 1.1111 | 1.5376    | 1.4729    | 1.6327     |
| 7  | 1.0769 | 1.4741    | 1.4069    | 1.5817     |

### Exp 5: Spectral / higher-moment bounds — INTERESTING STRUCTURE
- Primal optima have 3–5 active diagonals
- Higher-p bounds (max q_k ≥ (Σ q_k^p)^{1/(p-1)}) beat Shor
- At p=10, P=5: bound = 1.627 (close to V(5)=1.634!)
- But requires minimizing degree-2p polynomial — expensive to certify

### Exp 6: Richardson extrapolation with bootstrap CIs ✓
- LOO-CV consistently selects the simple 1/P model
- **UB (primal, P≥7): C_inf = 1.516 ± 0.003** (95% CI: [1.511, 1.522])
- LB (Lasserre-2, P≥7): C_inf = 1.412 ± 0.004 (weaker because L2 is a relaxation)
- BMA estimate: UB → 1.510, LB → 1.399

### Exp 7: SOS epigraph / DNN analysis — NEGATIVE RESULT
- DNN inner approximation of copositivity is trivially weak (requires η ≥ 2P)
- Off-diagonal entries of η·J − 2P·A_k are negative for reasonable η
- Positivstellensatz is the dual of Lasserre — no computational advantage
**Result:** Dead end (DNN too weak, Psatz ≡ Lasserre).

### Exp 8: SDP-certified S2 bounds — MIXED
- Shor-level SDP for min Σq_k² gives exactly uniform S2 = 1/(2P−1) → Shor bound
- Quartic relaxation beats Shor at small P but degrades (bound < 1 as P→∞)
**Result:** Inherently limited — Cauchy-Schwarz gap grows with #diagonals.

### Exp 9: Mini-Lasserre with reduced basis — VERY PROMISING ✓
Diagonal basis {1, x_i, x_i²} captures 96–99% of Lasserre-2 quality!

| P  | Shor   | Diag   | BW=1   | BW=2   | Full   | Lasserre-2 |
|----|--------|--------|--------|--------|--------|------------|
| 5  | 1.1111 | 1.6230 | 1.6327 | 1.6327 | 1.6327 | 1.6327     |
| 8  | 1.0667 | 1.4847 | 1.4976 | 1.5232 | 1.5398 | 1.5483     |

### Exp 10: M3 scaling + ablation — KEY INSIGHT ✓
**x² cuts contribute NOTHING.** The entire improvement comes from convolution multipliers.

| P  | Full M3 | No x² cuts | No conv mult | Neither | Shor   |
|----|---------|------------|--------------|---------|--------|
| 5  | 1.3425  | 1.3425     | 1.1112       | 1.1112  | 1.1111 |
| 10 | 1.2321  | 1.2321     | 1.0527       | 1.0527  | 1.0526 |


## Round 2: Combining the best ideas (Exp 11–17)

### Exp 11: Iterated convolution multipliers — CLARIFYING ✓
Scalar (entrywise) constraints from localizing matrices vs full PSD:

| P  | Shor   | M3 base | ScalDiag | ScalAll | PSD Loc | Lasserre-2 |
|----|--------|---------|----------|---------|---------|------------|
| 5  | 1.1111 | 1.3425  | 1.3761   | 1.4588  | 1.4729  | 1.6327     |
| 8  | 1.0667 | 1.2556  | 1.2652   | 1.3715  | 1.3774  | 1.5483     |

**Key finding:** ScalAll (entrywise nonneg) captures ~96% of PSD localizing benefit.
The PSD structure adds only ~4% more.

### Exp 14: Cross-diagonal product cuts — REDUNDANT
(x^T A_{k1} x)(x^T A_{k2} x) ≤ (η/2P)² adds NOTHING beyond ScalAll.
**Result:** These cuts are already implied by the entrywise localizing constraints.

### Exp 15: Sparse localizing — SURPRISING RESULT
Need nearly ALL diagonals for good bounds. Even 5 evenly-spaced localizing matrices add nothing at P≥5!

| P=7 Config | Bound  | #Loc |
|------------|--------|------|
| none       | 1.3501 | 0    |
| center     | 1.3501 | 1    |
| sparse5    | 1.3501 | 5    |
| half       | 1.4069 | 7    |
| all        | 1.5306 | 13   |

**Key finding:** Information is distributed across ALL diagonals. Can't sparsify cheaply.

### Exp 16: Hybrid mini-Lasserre + conv multipliers — DOMINATED
Conv multiplier constraints are REDUNDANT when localizing matrices are present.
Diag+Loc (diagonal basis + all conv localizing) remains the best compact approach.

| P  | Shor   | Diag+CM | BW1+CM | Diag+Loc | Diag+All | Lasserre-2 |
|----|--------|---------|--------|----------|----------|------------|
| 5  | 1.1111 | 1.4003  | 1.4003 | 1.6230   | 1.6230   | 1.6327     |
| 8  | 1.0667 | 1.3239  | 1.3239 | 1.4891   | 1.4901   | 1.5483     |
| 9  | 1.0588 | 1.3197  | 1.3197 | 1.4788   | 1.4804   | 1.5456     |

### Exp 17: Scale Diag+Loc to P=10 + extrapolate
Diag+Loc extrapolation (1/P model, P≥5): **C_inf ≈ 1.298**

| Sequence    | C_inf (1/P, P≥5) |
|-------------|-------------------|
| Diag+Loc LB | 1.298 ± 0.017     |
| Lasserre-2  | 1.422 ± 0.005     |
| Primal UB   | 1.511 ± 0.003     |


## Hierarchy of Relaxations (tightest → weakest at P=8)

| Method                        | P=5    | P=8    | Cost          |
|-------------------------------|--------|--------|---------------|
| Primal UB (true V(P))        | 1.6338 | 1.5802 | —             |
| Full Lasserre Level-2         | 1.6327 | 1.5483 | O(P^8) SDP    |
| Mini-Lasserre (diag) + Loc   | 1.6230 | 1.4891 | O(P^4) SDP    |
| M3 + PSD localizing          | 1.4729 | 1.3774 | O(P^4) SDP    |
| M3 + scalar-all (entrywise)  | 1.4588 | 1.3715 | O(P^3) LP+SDP |
| M3 + conv multiplier only    | 1.3425 | 1.2556 | O(P^2) LP+SDP |
| Shor bound                   | 1.1111 | 1.0667 | O(P) SDP      |

## Key Theoretical Insights

1. **Convolution multiplier is the fundamental new cut.** Multiplying g_k(x) ≥ 0 by x_l ≥ 0 gives the simplest constraint that breaks the Shor barrier. Everything else builds on this.

2. **PSD adds little beyond entrywise nonneg.** The localizing matrix being PSD vs just entrywise nonneg matters only ~4%. The dominant information is in the individual entries.

3. **All diagonals matter.** Sparse selection of localizing matrices fails badly. The bound information is distributed uniformly across convolution diagonals.

4. **DNN copositivity is trivially weak** for this problem because the matrices η·J − 2P·A_k have necessarily negative off-diagonal entries.

5. **The minimax gap is real** — copositivity on the dual is structurally broken (confirmed by the original notebook), but copositivity IDEAS applied to the PRIMAL side (via localizing matrices) do work.

6. **Best extrapolated estimate of C_1a ≈ 1.50–1.52** (from primal UB sequence, which is the most reliable).
