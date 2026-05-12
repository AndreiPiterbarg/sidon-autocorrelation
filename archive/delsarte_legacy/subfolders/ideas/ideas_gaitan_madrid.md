# Gaitan-Madrid (arXiv:2512.18188, Dec 2025) - Applicability to C_{1a}

## Summary of findings

1. **C_{k,1} exact**: Theorem 2.1 gives
   C_{k,1} = binom(k, floor(k/2)) * ( floor((k+1)/2)*ceil((k+1)/2) / (floor((k+1)/2)+ceil((k+1)/2))^2 )^{k/2}.
   It is the value of the 1-D LP inf_{x in [0,1]} max_{0<=m<=k} binom(k,m) x^{k-m}(1-x)^m, solved in closed form via Newton / real-rootedness (Thm 3.5) and Lagrange multipliers on (k-1)-dim intersection cells P_{k,i}. For k=2 this gives C_{2,1} = 1/2.

2. **Relation to C_{1a}**: C_{1a} corresponds (up to scaling) to C_2 in their notation, where C_k is the best constant with ||f*..*f||_inf >= C_k ||f||_1^k for f supported on (-1/(2k), 1/(2k)). Cloninger-Steinerberger's C_{1a} = C_2 in this normalization.

3. **Direction**: Corollary 1 states **C_k <= 2k * C_{k,1}**, i.e., UPPER bounds on C_k. For k=2: C_2 <= 4 * 1/2 = 2, far weaker than the known 1.5029. Their discrete results do NOT give new lower bounds on C_{1a}.

4. **Lower bounds?**: NO. They only produce UPPER bounds C_k <= k(m+1) C_{2,m} (Matolcsi-Vinuesa refinement). The discrete C_{k,m} hierarchy BOUNDS C_k FROM ABOVE as m->inf; the C_{1a} lower bound work is orthogonal.

5. **Madrid-Ramos (2003.06962)**: Proves EXISTENCE of extremizers for autocorrelation problems on R via Hausdorff-Young duality + weak-* compactness. Relevant: justifies that val_infty = inf_{mu} ||mu*mu||_inf is attained, so Lasserre SDP relaxations converge to an actual extremizer, not just an infimum.

## Transferable algorithmic idea (proposal to push C_{1a} lower bound)

**Dual-cube probabilistic lower bound**: Gaitan-Madrid reduce to a 1-D LP via probabilistic interpretation (f_i(1) = p_i, tensorize). **Reverse the direction** for our problem: embed discrete autoconvolution on Z_n = {0,...,n-1} into a step-function approximation of the continuous C_{1a} problem. Use their Theorem 3.6 (infimum attained on (k-1)-dim real-root intersection cells P_{k,i}) to certify a finite-dimensional LP lower bound on C_{2,m} for large m, and then apply the INVERSE direction C_{1a} >= 2(m+1) * C_{2,m}_symm where C_{2,m}_symm is the symmetric step-function minimum (not C_{2,m} itself). Concretely: extend our Lasserre hierarchy (lasserre/) by adding the Newton-inequality cuts from Thm 3.5 (real-rootedness of the generating polynomial of diagonal moments) as ADDITIONAL PSD constraints - these are valid for probabilistic measures and tighten the relaxation on the sublevel cascade. Combined with Madrid-Ramos extremizer existence, this gives a rigorous val(d) floor via a clique-restricted sparse SDP with extra binomial-moment cuts.

(100 words body; algorithmic core: Newton-inequality PSD cuts on moment generating polynomial + extremizer compactness to close the LP gap.)
