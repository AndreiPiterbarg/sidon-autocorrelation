# Cloninger-Steinerberger (2017): reproduction notes

**Status:** in this project we worked under the conservative prior of
the rigorous analytic Matolcsi-Vinuesa bound $C_{1a} \ge 1.27481$ while
exploring our independent multi-scale arcsine line. The notes below
record what we did and did not manage to reproduce from the public
artefacts associated with Cloninger and Steinerberger (2017,
[arXiv:1403.7988](https://arxiv.org/abs/1403.7988)); they are not a
critique of that work and do not constitute a refutation of its bound
$C_{1a} \ge 1.2802$.

## The claim

Cloninger and Steinerberger
[arXiv:1403.7988](https://arxiv.org/abs/1403.7988) introduced a
discrete cascade for $C_{1a}$ based on a min-of-max-of-PSD-quadratics
optimisation on a simplex of integer compositions. Their Lemma 1
states
$$ C_{1a} \;\ge\; a_n \;:=\; \min_{a \in A_n}\;\max_{2 \le \ell \le 2n,\; -n \le k \le n-\ell}\; \frac{1}{4n\ell}\!\!\sum_{k \le i+j \le k+\ell-2}\!\! a_i a_j, $$
with $A_n = \{a \in (\mathbb R_+)^{2n} : \sum a_i = 4n\}$. Lemma 3
gives a discretisation correction $b_{n,m} - 2/m - 1/m^2 \le C_{1a}$
for compositions in $B_{n,m} = A_n \cap (m^{-1} \mathbb N)^{2n}$. The
paper reports $C_{1a} \ge 1.2802$ via $B_{24,50}$ on Yale Omega
(20{,}000 CPU-hours).

## What we tried, and what we observed

Our reproduction efforts were limited and did not match the published
$1.2802$ figure end-to-end; we record the observations here so future
readers can take them at face value, but we emphasise that we were
unable to fully replicate the original setup.

1. *Direct re-implementation.* A clean Python re-implementation
   (`cs_refined_lp.py`) of the branch-and-prune over $B_{n,m}$ within
   our available resources reproduced the rigorous lower bound only to
   $1.104105$ at $(n_{\rm half}, m) = (3, 30)$; a heuristic
   projected-subgradient variant first crossed $1.2802$ at $n = 9$,
   beyond the rigorous reach of the reduced configuration we ran.
2. *Reading of the public MATLAB artefact.* In a parse of one
   discretisation step in the public MATLAB routine, the conversion
   between bin-mass and bin-height appeared to us inconsistent with
   the mass-conservation constraint $\sum a_i = 4n$. We could not
   rule out that this reflects a difference in convention rather than
   an error, and we did not attempt to contact the authors. We
   therefore treat this only as an unverified reproducibility note, not
   as a substantive claim about the published result.

The CS-2017 bound is plausible on the basis of the published method;
our reproduction at much smaller $(n, m)$ was simply too coarse to
reach $1.2802$, and the gap is consistent with the reported
20{,}000-CPU-hour budget at $B_{24,50}$ being well outside our
operating envelope. Accordingly we did not pursue this line further
and instead refined the analytic Matolcsi-Vinuesa framework, which
yielded the Piterbarg-Bajaj-Vincent Bound
$C_{1a} \ge 1292/1000 = 1.292$ via the multi-scale arcsine construction
in [`multiscale_arcsine.md`](multiscale_arcsine.md).

## References

- A. Cloninger, S. Steinerberger, *On suprema of autoconvolutions with an application to Sidon sets*, Proc. AMS 145(8) (2017), [arXiv:1403.7988](https://arxiv.org/abs/1403.7988).
- M. Matolcsi, C. Vinuesa, *Improved bounds on the supremum of autoconvolutions*, J. Math. Anal. Appl. **372** (2010), 439-447, [arXiv:0907.1379](https://arxiv.org/abs/0907.1379).
- T. Tao et al., *Optimization Constants Repo*, AlphaEvolve, [github.com/teorth/optimizationproblems](https://github.com/teorth/optimizationproblems).
- See [`multiscale_arcsine.md`](multiscale_arcsine.md), [`lasserre.md`](lasserre.md), [`cascade_estimator.md`](cascade_estimator.md).
