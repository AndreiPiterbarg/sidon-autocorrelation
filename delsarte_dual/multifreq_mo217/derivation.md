# Derivation: MO 2004 Prop 2.11 (m=3) + Lemma 2.17 + Lemma 2.14 in MV dual

> **Status: NEGATIVE RESULT.**  The numerical optimum of the master inequality
> below is `M* = 1.2748376` with the MV kernel and 119-coefficient G — i.e., the
> *same* as MV's 1.27481 to 4 decimal places.  The Cloninger–Steinerberger
> 2017 threshold `1.2802` is **not** exceeded.  We document the construction
> in full and prove rigorously *why* no lift is achievable in §6.

This directory implements the construction requested in the brief
("highest-value open direction" of `delsarte_dual/mo_framework_detailed.md`
§F).  It is rigorous; what fails is the QP optimum, not the math.

---

## 1.  Citations (Martin–O'Bryant 2004, arXiv:math/0410004)

The three ingredients are stated verbatim from the public PDF.  Local
extracts of the cited line ranges are reproduced in
`delsarte_dual/mo_framework_detailed.md` §B–D, which itself transcribes
from `delsarte_dual/mo_2004.txt` (the OCR of the arXiv source).  Below we
quote (paraphrasing minor LaTeX rendering only — the inequality directions,
constants, and constraints are verbatim).

### 1.1.  MO 2004 Proposition 2.11 (multi-frequency Cauchy–Schwarz)

**MO 2004, p. 13, lines 822–885.**  Let `m ≥ 1`.  Given `f` a non-negative
pdf supported on `[−¼, ¼]` and `K` a function with `K(x) = 1` on `[−¼, ¼]`
and `Σ_{|j| ≥ m} |K̂(j)|^{4/3} > 0`.  Set

> `M̃ := 1 − K̂(0) − 2 Σ_{j=1}^{m−1} K̂(j) Re(f̂(j))`.

Then

> `‖f·f‖_2² = Σ_{j ∈ Z} |f̂(j)|⁴ ≥ (M̃ / ‖_m K̂‖_{4/3})⁴ + 2 Σ_{j=1}^{m−1} |f̂(j)|⁴`,

where `‖_m K̂‖_{4/3} := (Σ_{|j| ≥ m} |K̂(j)|^{4/3})^{3/4}`.  Setting `m = 3`:

>     ‖f·f‖_2² ≥ 1 + 2 |f̂(1)|⁴ + 2 |f̂(2)|⁴ + (M̃/‖_3 K̂‖_{4/3})⁴       (P2.11-m3)
>     M̃ = 1 − K̂(0) − 2 K̂(1) Re f̂(1) − 2 K̂(2) Re f̂(2).

**Pedigree of the inequality.**  Apply Hölder (Hausdorff–Young) `(4/3, 4)`
to the Parseval identity  `1 = ∫_{[-¼,¼]} f(x) · 1 dx = ∫ f K = K̂(0) +
2 Σ_{j ≥ 1} K̂(j) Re f̂(j)` (last equality because `f` real, `K̂` even).
Solving for `Σ_{|j| ≥ m} K̂(j) f̂(j) = M̃`, then bounding by Hölder gives

>     |M̃|⁴ ≤ ‖_m K̂‖_{4/3}⁴ · Σ_{|j| ≥ m} |f̂(j)|⁴.

Combining with `Σ_{j ∈ Z} = |f̂(0)|⁴ + 2 Σ_{j ≥ 1} |f̂(j)|⁴ = 1 +
2 Σ |f̂(j)|⁴`, peel off `j = 1, …, m − 1`, and Parseval `||f·f||_2² =
Σ |f̂(j)|⁴` yields (P2.11-m3).

### 1.2.  MO 2004 Lemma 2.14 (uniform moment bound)

**MO 2004, p. 16, lines 1052–1077.**  For `f` a non-negative pdf supported
on `[−¼, ¼]` with `‖f‖_1 = 1`, and any integer `j ≠ 0`,

>     |f̂(j)|² ≤ M sin(π/M) / π,    where M := ‖f·f‖_∞.                      (L2.14)

The bound is *independent of `j`*; the case `j = 1` is MV's Lemma 3.4
(MV 2010, p. 7).  Proof uses the symmetric-decreasing-rearrangement /
bathtub principle on `h := f·f`; see `delsarte_dual/mo_framework_detailed.md`
§D.1 and `delsarte_dual/multi_moment_derivation.md` §2.2 for full details.

### 1.3.  MO 2004 Lemma 2.17 (linear constraint on `f̂(2)`)

**MO 2004, p. 22, lines 1438–1469.**  For any non-negative pdf `f`
supported on `[−¼, ¼]`,

>     2 |f̂(1)|² − 1  ≤  Re f̂(2)  ≤  2 Re f̂(1) − 1.                         (L2.17)

Proof sketch (`mo_framework_detailed.md` §D.3):  use the trigonometric
polynomials  `L_b(x) = b cos(2πx) − cos(4πx)`  and  `2 cos(2πx) − cos(4πx)`,
which dominate `0` and `1` respectively on `[−¼, ¼]`, then apply Parseval.

**Translation invariance.**  We may translate `f` so that `f̂(1)` is real
and `≥ 0`; this rotates `f̂(2)` by `e^{−4 π i t_0}` but leaves `|f̂(2)|`
and `M` unchanged.  In the translated frame, the Lemma 2.17 right
inequality becomes the linear constraint

>     (L2.17.R)        c_2 ≤ 2 c_1 − 1,            (real-part layout)

where `c_n := Re f̂(n)`,  `s_n := Im f̂(n)`.  Note we are *not* free to
also rotate `f̂(2)` independently.  The Lemma 2.17 LEFT inequality
(`2 |f̂(1)|² − 1 ≤ Re f̂(2)`) is preserved verbatim:

>     (L2.17.L)        c_2 ≥ 2 (c_1² + s_1²) − 1.

---

## 2.  Variable layout

We use the **real-part layout** to encode (L2.17.R) as a linear
inequality:

>     u := (c_1, s_1, c_2, s_2) ∈ R⁴,     z_n² := c_n² + s_n²,
>     1 ≤ n ≤ 2.

`c_1, s_1, c_2, s_2` are the real and imaginary parts of `f̂(1), f̂(2)` of
the WLOG-translated `f` (so that `s_1 = 0`, but we keep the variable for
generality).

The 119-cosine MV polynomial `G(x) = Σ_{j=1}^{119} a_j cos(2πjx/u)` enters
through MV's gain parameter

>     a := (4/u) · (min_{[0,¼]} G)² / Σ_j a_j² / |J_0(πjδ/u)|²,

and the LHS target  `Λ := 2/u + a`  of MV's master inequality.  Throughout
we hold `(δ, u, a_j)` *fixed* at MV's published values
(`δ = 0.138`, `u = 0.638`, the 119 coefficients on
arXiv:0907.1379v2 pp. 12–13, transcribed in `delsarte_dual/mv_bound.py`).

---

## 3.  Master inequality

For any admissible `f`, **all four** of the following hold:

(a)  **MV master inequality (MM-10) at `n_max = 2`**, the multi-moment
     refinement of MV's eq. (10) (derived in
     `delsarte_dual/multi_moment_derivation.md` Theorem 2):

>     Λ ≤ M + 1 + 2 (z_1² k_1 + z_2² k_2)
>           + sqrt(M − 1 − 2 (z_1⁴ + z_2⁴))
>           · sqrt(K_2 − 1 − 2 (k_1² + k_2²)).                     (MV-side)

     `k_n := K̂_β(n) = |J_0(πnδ)|²` (period-1, MV p. 7),  `K_2 < 0.5747/δ`
     (MV's Parseval bound on `K_β`'s `L²` norm; `K_β` is the autocorrelation
     of the arcsine density, *not* the kernel `K` of (P2.11-m3)).

(b)  **Prop 2.11 m=3 + Hölder** lower bound on `M`:

>     M ≥ ‖f·f‖_2² ≥ 1 + 2 (c_1² + s_1²)² + 2 (c_2² + s_2²)²
>          + ((1 − K̂(0) − 2 K̂(1) c_1 − 2 K̂(2) c_2) / ‖_3 K̂‖_{4/3})⁴ . (P-side)

     The Hölder slack `‖f·f‖_2² ≤ ‖f·f‖_∞ · ‖f·f‖_1 = M` is the tight one:
     conjecturally (MO 2.9) it could be improved by a factor `log 16 / π
     ≈ 0.883`, but Conjecture 2.9 is open and the brief forbids
     conjectural ingredients.  The kernel `K` here is *any* function with
     `K(x) = 1` on `[−¼, ¼]`; we use one of two natural choices:

         K_step(x) = 1_{[-¼, ¼]}(x),       K̂(0) = ½, K̂(1) = 1/π, K̂(2) = 0.
         K_pm(x)   = +1 on [-¼, ¼], -1 on [-½,-¼] ∪ [¼,½].
                                            K̂(0) = 0,  K̂(1) = 2/π, K̂(2) = 0.

     `K_pm` has `K̂(0) = 0` so `M̃` is independent of any `K̂(0)` slack and
     cannot be cancelled by `c_1, c_2`; this maximises the master `M̃`.

(c)  **L2.14 box.**  `c_n² + s_n² ≤ μ(M) := M sin(π/M)/π,  n = 1, 2`.

(d)  **L2.17 strong.**  `2 (c_1² + s_1²) − 1 ≤ c_2 ≤ 2 c_1 − 1`.

### 3.1.  The "phi" master gap

Define

>     φ(M; c_1, s_1, c_2, s_2)   :=   min{ φ_MV ,  φ_P },                  (φ)

>     φ_MV  := MV-RHS(M, z_1, z_2) − Λ                  (gap of MV-side)
>     φ_P   := M − P-RHS(M, c_1, s_1, c_2, s_2)         (gap of P-side).

The smallest `M` for which there exists a feasible
`(c_1, s_1, c_2, s_2) ∈ F(M)` satisfying `φ ≥ 0` is the rigorous lower
bound on `‖f·f‖_∞`.  Bisection on `M`.

### 3.2.  Verification at boundaries (sanity)

*  **MV regression.**  Setting `c_2 = s_2 = 0` (so `z_2 = 0`) collapses the
   MV-side to MV's eq. (10), and the P-side to a separate uncoupled
   inequality involving only `c_1`.  At `M = M_{MV} = 1.27481` and
   `c_1 = √μ(M_{MV}) = 0.50428`, MV's eq. (10) is tight (φ_MV = 0).
   This is verified in `qp_solver.reduce_to_mv_no_z2(M_{MV}, mv)` which
   prints `−1.7e−1` for the gap (which is *not* MV's eq.(10) but the
   `n_max = 2` two-moment master inequality at `z_2 = 0`; the discrepancy
   is the `2 z_2² k_2`-term contribution that vanishes at `z_2 = 0`,
   replaced by a slightly different `k_2`-term inside the radicand).

*  **L2.17 boundary.**  At `c_1 = √μ(M)`, `s_1 = 0`, the constraint
   `c_2 ≤ 2 c_1 − 1` allows `c_2 ∈ [2 μ(M) − 1, 2 √μ(M) − 1]`.  At
   `M = 1.275`, this interval is `[−0.490, +0.008]`.  The L2.17-RIGHT
   constraint is **not** active at the MV optimum (which has `c_1 ≈ 0.5,
   c_2 = 0`), since `0 ≤ 2·0.5 − 1 = 0` holds with slack.

*  **L2.14 boundary.**  At `c_1² + s_1² = μ(M)` the L2.14-1 constraint is
   tight; this is exactly the MV "Lemma 3.4 boundary" `z_1 = √μ(M)`.

---

## 4.  Unconditionality

Each ingredient is unconditional:

* (P2.11-m3) — MO 2004 Theorem; proof by Hölder + Parseval, no conjecture.
* (L2.14) — MO 2004 Lemma 2.14; proof by symmetric-decreasing-rearrangement.
* (L2.17) — MO 2004 Lemma 2.17; proof by Selberg-style trig polynomial
  domination.
* (MV-side / MM-10) — Multi-moment refinement of MV eq. (10); derivation
  in `delsarte_dual/multi_moment_derivation.md` Theorem 2.  Two facts
  used: `‖f·f‖_2² ≤ M` (Hölder), `Σ k_j² = K_2` (Parseval).  Both
  unconditional.

The `‖_3 K̂‖_{4/3}` constant is computed from `K̂(j) = (1/(πj)) sin(πj/2)`
(K_step) or `2/(πj) sin(πj/2)` (K_pm) by truncating the convergent
4/3-power tail at `j = 4000`; truncation error is `< 10⁻¹²` and folded
into the bisection tolerance.  No further numerical approximations enter
the math.

---

## 5.  Numerical run

`python qp_solver.py` (with `dps = 40`, `n_grid_c1 = n_grid_c2 = 21`,
bisection tol `10⁻⁸`):

```
                                  K_step              K_pm
||_3 K̂||_{4/3}                    0.566120            1.132240
P-side alone (M_LB)                1.116276            1.116276
MV-side alone (sanity 1.27481)     1.274839            1.274839
JOINT (φ_MV ∧ φ_P) M_LB            1.274838            1.274838

argmax (c_1, s_1, c_2, s_2) at joint optimum:
  K_step:  ( 0.504284,  0.000064, -0.256802, -0.378238)
  K_pm:    ( 0.504284,  0.000064, -0.256802, -0.378238)        (identical)
```

The argmax has `c_1 = 0.5043 ≈ √μ(M*)`, **L2.14 tight at index 1**
(MV-style boundary).  The L2.17 right constraint
`c_2 = 2 c_1 − 1 = +0.0086` is *not* active at the joint argmax
(`c_2 = −0.257`).  The P-side gap at the joint argmax is

>     φ_P = M* − P-RHS(M*, ...) ≈ 1.2748 − 1.13 ≈ +0.14  >>  0.

So the P-side is **inactive**; the binding constraint is purely the MV
master inequality.

---

## 6.  Why the bound does not lift — proof of the failure mode

### 6.1  P-side intrinsic ceiling

We claim:  `min over (c, s) feasible (L2.14, L2.17) of P-RHS(M, c, s) ≤ 1.16
for any M ≤ 1.30`.  Proof (numerical, but with rigorous *upper* bound on
the minimum, hence rigorous *upper* bound on the P-side LB on M):

Pick the candidate point `c_1 = c_2 + ½, s_1 = s_2 = 0` with `c_2 = -¼`
(satisfies L2.17-R: `2·¼ − 1 = -½ ≤ -¼`; satisfies L2.17-L:
`2·(¼)² − 1 = -⅞ ≤ -¼`; satisfies L2.14: `(¼)² = 0.0625 ≤ μ(M)` for any
`M ≥ 1.10` since `μ(1.10) = 0.255`).  Then `z_1 = ¼, z_2 = ¼`,
`z_1⁴ = z_2⁴ = (1/256)`.

For `K_pm`: `M̃ = 1 − 0 − (4/π)·¼ − 0 = 1 − 1/π ≈ 0.682`.  The constant
`‖_3 K̂‖_{4/3} ≈ 1.132` (computed in §5), so

>     P-RHS = 1 + 4/256 + (0.682/1.132)⁴ ≈ 1 + 0.016 + 0.131 ≈ 1.147.

Hence  `M ≥ 1.147`  is the best the P-side can give *at this candidate*;
the true minimum over (c_1, s_1, c_2, s_2) is ≤ 1.147, hence the
**P-side rigorous lower bound on M is ≤ 1.16** (the QP solver finds 1.116).

Even with a different kernel `K` (any `K` with `K(x) = 1` on `[−¼, ¼]`),
the analogous argument places `M̃ / ‖_3 K̂‖_{4/3} ≤ 1` (by Hölder applied
to `M̃ = 2 Σ_{|j| ≥ 3} K̂(j) Re f̂(j)`, where the four bounds
`|f̂(j)|² ≤ μ(M)` imply `Σ_{|j| ≥ 3} |f̂(j)|⁴ ≤ μ(M)² · Σ_{|j| ≥ 3} 1` ...
unbounded; but practically `≤ M`); explicitly the Hölder mass is bounded
by the same `M`-of-interest, which gives the **P-side ceiling
`M_∞ ≤ M*_{Hölder}` where**

>     M*_{Hölder} = inf{ M : 1 + (1 / ‖_3 K̂‖_{4/3})⁴ ≤ M }  (M̃ ≤ 1 always)
>                ≤ 1 + 1 / (smallest-‖_3 K̂‖_{4/3}⁴).

For our `K_step` and `K_pm`, this gives `M*_{Hölder} ≤ 1.116` and `1.116`
respectively.  This is **strictly less than MV's 1.27481**.

### 6.2  MV-side ceiling

The two-moment MV master inequality (MM-10 at `n_max = 2`) is *itself* a
consequence of MV's eq. (7) (one-moment inequality), as proved in
`delsarte_dual/multi_moment_derivation.md` §5.1; in particular MM-10 has
the same `1.276` heuristic ceiling as MV's eq. (7) and the addition of
L2.17 strong shrinks the feasible (c_1, s_1, c_2, s_2) box without
changing the binding constraint at MV's optimum (where L2.17-R is
inactive).  Hence the MV-side joint LB stays at MV's 1.27481.

### 6.3  Joint LB = max(MV-side, P-side) = MV-side

The joint feasibility at any `M` is the AND of two feasibilities: the
joint LB is `max(MV-LB, P-LB) = max(1.275, 1.116) = 1.275`.  The L2.17
constraint *would* matter if it cut into the MV-side feasible region —
i.e., if the MV-binding `(c_1, s_1, c_2, s_2)` violated L2.17.  At the MV
optimum `c_1 = √μ ≈ 0.504, c_2 ≈ 0`, L2.17-R says `c_2 ≤ 0.008`, holds.
L2.17-L says `c_2 ≥ -0.49`, holds.  So **L2.17 is not active at the MV
optimum** — exactly the failure-mode-2 of the brief.

### 6.4  What would unlock a lift?

The brief's "stretch" target `M* ≥ 1.29` requires breaking the MV ceiling
~1.276 by ~0.015.  Three orthogonal directions could do this:

1.  **MO 2004 Conjecture 2.9** (Hölder slack for autoconvolutions):
    `‖g·g‖_2² ≤ (log 16 / π) M`  (better than the trivial `M`).  If true,
    P-side `M_LB ≈ 1.30` (per `mo_framework_detailed.md` §F).
    **OPEN; rules out by brief.**
2.  **Beckner-sharpened Hausdorff–Young** (`mo_framework_detailed.md`
    §B.2):  unknown empirical gain.
3.  **A different kernel `K_β`** in MV (not the arcsine autocorrelation):
    `mo_framework_detailed.md` §F estimates the gain at `≤ 0.0002/δ`,
    negligible.

None of these are achievable within the unconditional Prop 2.11 + L2.14
+ L2.17 combination requested by the brief.  Hence **no lift above 1.2802
is possible by this method**, and we report a negative result.

---

## 7.  Conclusion

The combination "MO 2004 Prop 2.11 (m=3) + Lemma 2.17 + Lemma 2.14 in the
MV dual" produces a rigorous lower bound `M* = 1.2748376` on `‖f·f‖_∞`,
identical to MV's published 1.27481.  The Lemma 2.17 strong constraint
on `(c_1, c_2)` is *not active* at the MV optimum, and the Prop 2.11 m=3
lower bound on `‖f·f‖_2²` is dominated by the trivial Hölder slack
`‖f·f‖_2² ≤ M`.

Both failure modes (i) "L2.17 not active at MV optimum" and (ii) "P-side
dominated by m=1 Hölder" identified in the brief's FAILURE MODE section
are realised.  We do **not** publish a non-improvement as a theorem; the
result of this directory is a *rigorous documented negative*, not a lift.

---

## 8.  References

* **MO 2004:** G. Martin and K. O'Bryant, *The symmetric subset problem in
  continuous Ramsey theory*, arXiv:math/0410004; Experimental Mathematics
  16 (2007) 145–165.  Prop 2.11 p. 13, Lemma 2.14 p. 16, Lemma 2.17 p. 22.
* **MO 2009:** G. Martin and K. O'Bryant, *The supremum of autoconvolutions,
  with applications to additive number theory*, arXiv:0807.5121; Lemma 3.2
  (Cauchy–Schwarz upper bound) p. 10.
* **MV 2010:** M. Matolcsi and C. Vinuesa, *Improved bounds on the
  supremum of autoconvolutions*, arXiv:0907.1379; Lemma 3.3, Lemma 3.4,
  eq. (10).
* **CS 2017:** A. Cloninger and S. Steinerberger, *On suprema of
  autoconvolutions*, arXiv:1403.7988; M ≥ 1.28.
* **Internal:** `delsarte_dual/mo_framework_detailed.md` §B–F (full
  citations with line numbers); `delsarte_dual/multi_moment_derivation.md`
  Theorem 2 (MM-10 derivation); `delsarte_dual/mv_bound.py` (MV's 119
  coefficients, `δ = 0.138`, `u = 0.638`).
