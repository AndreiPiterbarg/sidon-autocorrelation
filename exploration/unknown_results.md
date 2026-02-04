# Key Unknowns — Status and Results

## K1: Is C_{1a} truly ~1.50 or could it be much lower?

**Status: Open.**

Current bounds: C_{1a} in [1.2802, 1.5029]. Five independent optimization methods (LP iteration, Polyak, L-BFGS, NTD, AlphaEvolve) all converge to primal values ~1.50, suggesting the upper bound is near-tight. The ~0.22 gap is dominated by weakness on the lower bound side. No evidence yet that C_{1a} could be significantly below 1.50, but no proof it cannot be.

## K2: Does the extremizer have a boundary singularity at +/-1/4?

**Status: Open.**

All published work uses uniform grids, which cannot resolve boundary behavior. Non-uniform (geometrically refined) grids near +/-1/4 have never been tested. If the true extremizer blows up or vanishes at the boundary, uniform discretizations would converge slowly.

## K3: Is the SDP relaxation tight at small P?

**Status: Resolved — NO.**

**Experiment:** Shor + RLT SDP relaxation (`exploration/SDP_certification.ipynb`), tested at P = 2..30 and P = 50.

**Formulation:** Lift x x^T to PSD variable X >= x x^T on the simplex {x >= 0, 1^T x = 1}, with RLT cuts (X >= 0 elementwise, X 1 = x). Minimize eta subject to eta >= 2P Tr(A_k X) for each anti-diagonal k.

**Result:** The SDP bound equals 2P/(2P-1) at every P tested, to machine precision. This is the theoretical floor: the 2P-1 anti-diagonal sums of X are constrained to total 1 (from 1^T X 1 = 1), so the SDP spreads mass uniformly at 1/(2P-1) per anti-diagonal, achieving eta = 2P/(2P-1). This converges to 1.0 as P grows, far below the true value ~1.50.

**Why it fails:** The relaxation is full-rank (Rank(X*) = P at every P). The lifted matrix exploits the gap between X >= x x^T and X = x x^T to flatten all anti-diagonal sums, which no rank-1 (i.e., physically realizable) solution can do. The gap is structural, not numerical — no solver tuning or additional RLT cuts can improve it.

**Concrete numbers:**
- P=2: SDP=1.333, Primal=1.778, gap=0.444 (25%)
- P=10: SDP=1.053, Primal=1.578, gap=0.526 (33%)
- P=50: SDP=1.010, Primal=1.668, gap=0.658 (39%)

**Implication:** Certifying global optimality of the discretized C_{1a} problem requires either (a) Lasserre hierarchy at level >= 2, (b) a Fourier-domain SDP using Fejer-Riesz / trigonometric moment structure, or (c) an entirely different dual approach.

## K4: Can the L^inf problem be convexified via moment/SOS formulation?

**Status: Open.**

The Fourier-domain reformulation is the most promising unexploited idea. The autoconvolution has a natural representation via trigonometric polynomials, and Fejer-Riesz factorization could yield a tighter SDP than the spatial-domain Shor relaxation that failed for K3. This has been suggested independently by [MV10], [AE25], and [KB] but never attempted.

## K5: What is the exact ceiling of the Fourier kernel lower bound method?

**Status: Open.**

The current lower bound C_{1a} >= 1.2802 comes from Fourier-analytic methods. The gap between this and the upper bound ~1.50 is ~0.22. No work has characterized whether this method can be pushed further or has a hard ceiling below 1.50.
