# Round 2 Experiment Plan

## What we know
- Conv-multiplier constraints (multiply g_k(x) >= 0 by x_l >= 0) are the key
- Degree-4 localizing matrices help significantly
- Product localizing (y4[i,j,a,b] PSD) helps more
- x² cuts are useless
- Mini-Lasserre with diagonal basis gets 96-99% of Lasserre-2

## New experiments (5-7 ideas)

### Exp 11: Iterated convolution multipliers
The key constraint is 2P*sum y3[i,j,l] <= eta*x[l]. What if we multiply AGAIN?
2P*sum y4[i,j,l,m] <= eta*y3_or_Y[l,m] — this needs degree-4 moments but
uses them differently than the localizing approach. Chain: multiply g_k >= 0
by x_l*x_m >= 0 to get degree-4 constraints without needing PSD matrices.
This is LP-strength (no new SDP constraints), just linear inequalities on y4.
Could be a cheap way to extract value from degree-4 moments.

### Exp 12: Convolution-aware localizing matrices
Instead of generic localizing matrices for each k, use STRUCTURED localizing
that respects the A_k sparsity. For diagonal k, only indices i,j with i+j=k
matter. Build small localizing matrices per-diagonal-pair instead of full PxP.
This could dramatically reduce SDP size while keeping most of the information.

### Exp 13: Degree-2 SOS multipliers for convolution constraints
Instead of multiplying g_k(x) >= 0 by degree-0 (x_l) or degree-1 (via loc matrix),
use degree-2 SOS multipliers: sigma(x) * g_k(x) >= 0 where sigma is SOS.
This gives degree-6 constraints but with a specific structure. Use a LOW-RANK
sigma (e.g., single quadratic (a^T x)^2) to keep it tractable.

### Exp 14: Cross-diagonal valid inequalities
For any two diagonals k1, k2: x^T A_{k1} x + x^T A_{k2} x <= 2*eta/(2P).
But also: (x^T A_{k1} x)(x^T A_{k2} x) <= (eta/(2P))^2.
The product gives degree-4 constraints: sum y4[i1,j1,i2,j2] <= eta^2/(4P^2)
for i1+j1=k1, i2+j2=k2. These are LINEAR in y4 (eta is parameter).

### Exp 15: Aggregate moment approach
Instead of per-k localizing matrices (expensive: 2P-1 PSD constraints),
use a WEIGHTED combination: sum_k w_k * L_k must be PSD.
Choose w_k to maximize the bound. This gives ONE PSD constraint instead
of 2P-1. The optimal w_k can be found by alternating optimization.

### Exp 16: Hybrid moment-3 + mini-Lasserre
Combine the best of both worlds: use the mini-Lasserre degree-2 moment
matrix (diagonal basis) for the PSD structure, PLUS the moment-3
convolution multiplier constraints. This adds the conv-multiplier cuts
to the mini-Lasserre framework.

### Exp 17: Push best approach to P=20-30 for extrapolation
Take whatever gives the best bounds and scale to larger P. Then do
Richardson extrapolation on the new bound sequence. If we get bounds
between Shor and Lasserre-2, the extrapolated limit gives a new
estimate of C_1a from a different vantage point.
