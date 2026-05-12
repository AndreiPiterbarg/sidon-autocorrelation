# Rearrangement inequalities and C_{1a}: a structural obstruction

**Date:** 2026-04-20
**Verdict (short):** Symmetric-decreasing rearrangement gives the **wrong direction** for a lower bound on C_{1a}. The naive reduction "WLOG f is symmetric decreasing" is UNSOUND here. This is not a new gap; it is confirmed by the Matolcsi--Vinuesa (2010) disproof of the Schinzel--Schmidt conjecture, and it is the structural reason Cloninger--Steinerberger (2017) avoid rearrangement entirely.

## 1. The problem and why rearrangement is tempting

We want
  C_{1a} = inf_{f >= 0, supp f subset [-1/4,1/4], int f = 1}  sup_{|t| <= 1/2} (f*f)(t).
The inner functional sup_t (f*f)(t) looks like a "concentration" quantity that ought to increase under symmetrization of f. If that were true, the infimum would be attained on the symmetric-decreasing cone, collapsing the variable to a 1-parameter-family-like object and unlocking Fourier/convex tools.

## 2. What Riesz--Sobolev actually gives

Riesz rearrangement inequality (Lieb--Loss, Thm 3.7; Burchard notes):
  int f(x) g(y) h(x-y) dx dy  <=  int f^*(x) g^*(y) h^*(x-y) dx dy.
Setting g = h = f and looking at it pointwise in the shift s,
  (f*f)(s) = int f(x) f(s-x) dx.
Applied with the translate h_s(u) = f(s-u), Riesz gives only the **0-shift** estimate
  (f^* * f^*)(0)  >=  sup_s (f*f)(s).                          (*)
That is: the symmetric-decreasing autoconvolution at the **origin** dominates the sup of the original autoconvolution.

Direction of the sup-functional is therefore:
  sup_t (f^* * f^*)(t)  =  (f^* * f^*)(0)  >=  sup_t (f*f)(t).  (**)
The first equality holds because f^* * f^* is symmetric and unimodal (convolution of two symmetric decreasing functions is symmetric decreasing; Burchard, Prop 2.2), so its sup is at 0.

## 3. Why this kills the lower-bound strategy

From (**), symmetrization can only **increase** the inner functional sup_t (f*f)(t). For an infimum over f, this means:
  inf over sym-decreasing f  of  sup_t (f^* * f^*)(t)   >=   inf over all f  of  sup_t (f*f)(t)  =  C_{1a}.
So restricting to symmetric decreasing f gives an **upper bound** on C_{1a}, not a lower bound. Matolcsi--Vinuesa (arXiv:0907.1379) exploited exactly this asymmetry: they constructed explicit non-symmetric f achieving sup(f*f) <= 1.5098 · (int f)^2 / |supp f|, below what any symmetric-decreasing candidate can reach, thereby **disproving the Schinzel--Schmidt conjecture** that the extremizer is symmetric decreasing. The current UPPER bound 1.5029 ultimately traces to such asymmetric constructions.

## 4. Brascamp--Lieb--Luttinger, Baernstein, Lieb--Young

- **BLL** generalizes Riesz to m-linear forms int prod f_i(sum a_{ij} x_j); the direction is always "integral increases under symmetric-decreasing rearrangement." Same obstruction as above: we need the integral to **decrease** for a lower bound on an inf-sup.
- **Baernstein's *-function** / star-symmetrization controls subharmonic-type functionals; the relevant autoconvolution is not of the right form (no maximum principle structure on (f*f)(t) as a function of f in the required direction).
- **Lieb's sharp Young** ||f*g||_r <= C_{pqr} ||f||_p ||g||_q bounds sup (f*f) = ||f*f||_infty from above by ||f||_p ||f||_q with 1/p+1/q = 1. For f >= 0, supp f subset [-1/4,1/4], int f = 1:
    ||f*f||_infty <= ||f||_2^2  (Cauchy--Schwarz/Young with p=q=2,r=infty),
  and ||f||_2^2 can be as small as 2 (uniform f), giving only ||f*f||_infty <= 2. This is a trivial **upper** bound; no lower bound emerges because Young goes the wrong way for inf over f.

## 5. One half-usable corollary

The 0-shift inequality (*) is still a valid **sub-problem upper bound**:
  C_{1a} = inf_f sup_t (f*f)(t)  <=  inf_{f sym-dec} (f*f)(0)  =  inf_{f sym-dec} ||f||_2^2.
By Cauchy--Schwarz with supp f subset [-1/4,1/4] and int f = 1, ||f||_2^2 >= 2, saturated by uniform f. So the best this approach ever gives on the upper side is 2 -- worse than the existing 1.5029. It yields **nothing** on the lower side.

## 6. Direct attack on f^* * f^*

For symmetric decreasing f^* on [-1/4,1/4] with int f^* = 1:
  (f^* * f^*)(0) = ||f^*||_2^2 >= 2, with equality iff f^* is uniform.
But f^* uniform gives (f^**f^*)(t) = 2(1/2 - |t|)_+ on [-1/2,1/2], whose sup is 2 · 1/2 · 2 = ... the triangular tent with peak 2 at 0. Every symmetric-decreasing candidate has autoconvolution sup = ||f^*||_2^2, and this is minimized by the uniform, giving **2**. So the symmetric-decreasing-only problem has answer 2, far from C_{1a} in [1.2802, 1.5029]. This quantitatively confirms the gap: non-symmetric f is essential.

## 7. Recommendation

Do not pursue Riesz / BLL / Baernstein / Lieb--Young as primary tools for a lower bound on C_{1a}. They run the wrong direction for inf-sup of autoconvolution. Two salvageable uses:
  (a) **Sanity upper bound** via (*): re-derives ||f||_2^2 >= C_{1a}, giving 2 -- trivially dominated by 1.5029.
  (b) **Asymmetry penalty**: use (**) in reverse to **certify** how far a candidate f is from symmetric, e.g. sup_t (f*f)(t) - (f*f)(|barycenter|) controlled by a symmetrization defect. This might feed the dual Lasserre moment relaxation as an auxiliary constraint, but does not itself produce a lower bound.

Primary lower-bound effort should remain on the Farkas-certified Lasserre hierarchy (val(4) > 1.0963 certified; working rigorous pipeline per MEMORY).

---

**100-word summary.** Riesz--Sobolev rearrangement and its generalizations (BLL, Baernstein, Lieb--Young) all move integrals/sups of autoconvolutions in the direction that **increases** under symmetrization. For C_{1a} = inf_f sup_t (f*f)(t), symmetric-decreasing rearrangement therefore yields only an UPPER bound, not a lower bound -- it gives 2, worse than the existing 1.5029. Matolcsi--Vinuesa (2010) disproved the Schinzel--Schmidt conjecture that the extremizer is symmetric decreasing, confirming the extremizer is genuinely asymmetric. This structural obstruction is why Cloninger--Steinerberger (2017) and the Lasserre pipeline avoid rearrangement entirely. Rearrangement inequalities cannot push C_{1a} above 1.2802.

## References
- Riesz rearrangement: https://en.wikipedia.org/wiki/Riesz_rearrangement_inequality
- Burchard short course: https://www.math.utoronto.ca/almut/rearrange.pdf
- Matolcsi--Vinuesa (disproof of Schinzel--Schmidt): https://arxiv.org/abs/0907.1379
- Cloninger--Steinerberger 1.28 lower bound: https://arxiv.org/abs/1403.7988
- Barnard--Steinerberger autocorrelation: https://arxiv.org/abs/2001.02326
- Optimal autocorrelation inequalities: https://arxiv.org/abs/2003.06962
- Classical autocorrelation/autoconvolution inequalities: https://arxiv.org/abs/2106.13873
- Rearrangement methods survey: https://arxiv.org/pdf/2207.05153
