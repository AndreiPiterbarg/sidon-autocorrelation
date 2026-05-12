# Hölder-generalised tail bound for the MV/MM-10 master inequality

**Scope.** We generalise the single Cauchy–Schwarz step in the MV dual
master inequality (Matolcsi–Vinuesa, arXiv:0907.1379, Lemma 3.3 +
eq. (10)) and in its multi-moment extension MM-10
(`delsarte_dual/multi_moment_derivation.md`, Theorem 2, eq. (MM-10)) to
an arbitrary Hölder conjugate pair `(p, q)` with
`1/p + 1/q = 1, p >= 2`.  The goal is to quantify empirically whether
this lifts the lower bound `M_cert` that MM-10 achieves on the
autocorrelation constant `C_{1a}`.

## §1 Setup and notation

Throughout let `f : R -> R_{>=0}` be supported on `[-1/4, 1/4]` with
`int f = 1`, and write

  h := f * f,   h >= 0,   supp h subset [-1/2, 1/2],   int h = 1,
  M := ||h||_inf = ||f * f||_inf,
  z_n := |hat f(n)|    (period-1 Fourier),
  so hat h(n) = hat f(n)^2   =>   |hat h(n)| = z_n^2.

Fix MV's kernel

  K(x) = (1/delta) (beta * beta)(x/delta),
  beta(x) = (2/pi) (1 - 4x^2)^{-1/2} * 1_{(-1/2, 1/2)}(x),
  k_n := hat K(n) = J_0(pi n delta)^2 >= 0,
  K_2 := sum_{j in Z} k_j^2   (MV's regularised surrogate <= 0.5747/delta).

For the Hölder family we additionally need, for each `q in (1, 2]`,

  K_q := sum_{j in Z} k_j^q    (kernel-side q-norm of the coefficient sequence).

`K_q` reduces to `K_2` at `q = 2`; for general `q` it is computed from a
validated finite sum plus a Poisson-tail bound (see §5).

## §2 The Hölder tail step

As in MV Lemma 3.3 and MM-10, Parseval and the triangle/Cauchy–Schwarz
inequalities combined with `int K = 1` give the master chain

  2/u + a <= ||f * f||_inf * int K + int (f*f) * K
         =  M + 1 + 2 sum_{n in N} z_n^2 k_n
            + 2 sum_{|j| > n_max} Re hat h(j) k_j,        (*)

where `N = {1, ..., n_max}` is the set of *kept* moments.  (The factor-of-2
arises from `hat h(n) = overline{hat h(-n)}`.)

The MV/MM-10 derivation bounds the tail `T := 2 sum_{|j| > n_max} Re
hat h(j) k_j` via Cauchy–Schwarz:

  |T| <= 2 * sqrt(sum_{|j|>n_max} |hat h(j)|^2)
           * sqrt(sum_{|j|>n_max} k_j^2).               (C-S)

**Our replacement.**  We use instead Hölder's inequality with conjugate
exponents `(p, q)`, `1/p + 1/q = 1`, `p >= 2` (so `q in (1, 2]`):

  |T| <= 2 * (sum_{|j|>n_max} |hat h(j)|^p)^{1/p}
           * (sum_{|j|>n_max} k_j^q)^{1/q}.             (H-pq)

At `(p, q) = (2, 2)` this is exactly (C-S).

## §3 Bounding the two factors

### §3.1 The h-side: tight bound via Lemma 1 + Parseval

**Naïve Hausdorff–Young bound (DEPRECATED as main path — see below).**
For `p >= 2`, `p' = p/(p-1) in [1, 2]`.  Hausdorff–Young gives
`(sum_j |hat h(j)|^p)^{1/p} <= ||h||_{p'}`, and
`||h||_{p'}^{p'} <= M^{p'-1}`, so
`(sum |hat h|^p)^{1/p} <= M^{1/p}`, i.e.,

  `sum_j |hat h(j)|^p <= M`.                                   (B_h_naive)

Equivalently,
`sum_{|j|>n_max} |hat h|^p <= M - 1 - 2 sum_{n in N} z_n^{2p}`.

**Problem with (B_h_naive).**  It is independent of `p` on its RHS `M`,
while for `p > 2`, Lemma 1 (`|hat h(n)| = z_n^2 <= mu(M)`) gives a
*strict* pointwise decrease: `z_n^{2p} = z_n^{2(p-2)} * z_n^4 <=
mu(M)^{p-2} * z_n^4`.  Exploiting this yields a tighter tail.

**Tight h-tail bound (main path).**  For `p >= 2` and every admissible
`(M, z)`,

  `sum_{|j|>n_max} |hat h(j)|^p
       =  2 sum_{n > n_max} z_n^{2p}
       <=  mu(M)^{p-2} * 2 sum_{n > n_max} z_n^4
       <=  mu(M)^{p-2} * (2 sum_{n>=1} z_n^4 - 2 sum_{n in N} z_n^4)
       =   mu(M)^{p-2} * (||h||_2^2 - 1 - 2 sum_{n in N} z_n^4)
       <=  mu(M)^{p-2} * (M - 1 - 2 sum_{n in N} z_n^4)`.      (h-tail)

The last step uses `||h||_2^2 = int h^2 <= M * int h = M`.

**At p = 2: (h-tail) equals MM-10 exactly.**  `mu(M)^0 = 1` and the
radicand is the MM-10 radicand.  (h-tail) is therefore a conservative
generalisation that STRICTLY tightens for `p > 2` and collapses to
MM-10 at `p = 2`.

**Implementation.**  `phi_holder.phi_holder` uses (h-tail) as the
canonical h-side bound.  At `p = q = 2` the code is bit-identical to
`phi_mm` (see test `test_reduces_to_mv_at_p_equals_2`).

### §3.2 The kernel side

Write `K_q := sum_{j in Z} k_j^q`.  Then trivially

  sum_{|j|>n_max} k_j^q = K_q - 1 - 2 sum_{n in N} k_n^q.      (k-tail)

(Valid as long as `K_q` is finite, which holds for `q > 1` because
`k_j ~ 2/(pi^2 j delta)` asymptotically and `sum j^{-q}` converges.)

### §3.3 Hölder RHS

Combining (*) with (H-pq), (h-tail), and (k-tail) yields our master
inequality:

> **Theorem H (Hölder-generalised master inequality).**
> For `p >= 2`, `q = p/(p-1)`, `n_max >= 1`:
>
> ```
> 2/u + a  <=  M + 1 + 2 sum_{n=1..n_max} z_n^2 k_n
>              + [ mu(M)^{p-2} * (M - 1 - 2 sum_{n=1..n_max} z_n^4) ]^{1/p}
>                * ( K_q - 1 - 2 sum_{n=1..n_max} k_n^q )^{1/q}       (HM-10)
> ```
>
> subject to the constraints (Lemma 1 of `multi_moment_derivation.md`)
>
>   `0 <= z_n^2 <= mu(M) := M sin(pi / M) / pi,   n = 1, ..., n_max.`

**Corollary.**  At `(p, q) = (2, 2)` and `K_q = K_2`, (HM-10) reduces
exactly to (MM-10), and hence at `n_max = 1` to MV's eq. (10).

### §3.4 What would Conjecture 2.9 change?

MO 2004 Conjecture 2.9 (`mo_2004.txt` line 783) strengthens the trivial
Parseval bound `sum |hat h|^2 <= M * 1` to

  sum_j |hat h(j)|^2 = ||h||_2^2  <=  (log 16 / pi) * M * 1     (conditional)

(constant `log 16 / pi ≈ 0.883` **tighter** than 1).  Applied to (C-S)
this multiplies the tail bound by `sqrt(log 16 / pi) ≈ 0.940`, i.e. a
`≈ 6.3%` shrinkage of the tail — not the 2.266x often quoted.  The
"2.266" factor is `2 * pi / log 16`, which is the ratio in the *reverse*
Parseval/interpolation estimate

  `C(pi / log 16) <= ||f*f||_2^2 / (||f*f||_inf ||f*f||_1) <= 1`.

The Hölder(p, q) bound we implement does **not** depend on Conjecture
2.9 being true; it is unconditional and coincides with MV at `p = q = 2`.
Any empirical gain we measure is entirely due to the **substitution of
Hölder for Cauchy–Schwarz given the unconditional norm bounds (B_h) and
the exact `K_q`**.

### §3.5 Why we do not expect a break of 1.276

The MV ceiling `~1.276` is pinned by MV's eq. (8) (`multi_moment_derivation.md`
§5.1): the `inf ||f_s * phi||_2^2` reformulation.  Any RHS substitution
inside the MV framework — Hölder included — is bounded above by this
`L^2` ceiling.  Specifically, Hölder (HM-10) is a **consequence** of
the rearranged Parseval–plus–master-chain that yields (C-S) in the p=2
limit; replacing C-S with Hölder can only change the *finite-precision*
tightening of the tail, not the underlying `L^2` infimum.  Thus the
empirical gain on top of MM-10 at `n_max = 1` is expected to be
**small and p-dependent**, and the lower bound `M_cert` from (HM-10) is
expected to stay strictly below the `1.276` ceiling **and** therefore
strictly below the Cloninger–Steinerberger target `1.28`.

The numerical sweep in `sweep_p.py` is our empirical test of this
prediction at `N = 1, 2` across `p in {2, 9/4, 5/2, 11/4, 3, 7/2, 4, 5, 6, 10}`.

### §3.6 Proof status

The derivation of (HM-10) from (*), (HY), and (h-tail) / (k-tail) is
**PROVED unconditionally** modulo the known MV framework premises
(Lemma 3.3 and MV's surrogate bound on `K_2`).  The only **CONJECTURAL**
ingredient, should one choose to apply it, is MO 2004 Conjecture 2.9 —
which we do NOT use in the code path.  The implementation of (HM-10) is
rigorous arb interval arithmetic at `>=256` bits.

## §4 Symbolic cross-check

This section records a sympy verification that the Hölder-generalised
RHS reduces, at the Cauchy-Schwarz point `p = q = 2`, to the MM-10 /
MV eq. (10) RHS.

The script is [`symbolic_crosscheck.py`](./symbolic_crosscheck.py).  It
defines the two symbolic expressions (`N_max = 2`)

```
RHS_CS     = M + 1 + 2 sum z_n^2 k_n
             + sqrt(M - 1 - 2 sum z_n^4) * sqrt(K2 - 1 - 2 sum k_n^2)

RHS_Holder = M + 1 + 2 sum z_n^2 k_n
             + ( M^{1/(p_hy - 1)} - 1 - 2 sum z_n^{2 p_hy'} )^{1/p_hy'}
             * ( K_q_total - 1 - 2 sum k_n^{q_kernel} )^{1/q_kernel}
```

where `p_hy'` is the conjugate of the Hausdorff–Young exponent
`p_hy` and `K_q_total = sum_j k_j^{q_kernel}`.  (The `p_hy` / `p_hy'`
bookkeeping in `symbolic_crosscheck.py` corresponds to `(p', p) = (p_hy,
p_hy')` in the main-text `(p, q) = (p_hy', q_kernel)`; the substitution
is the same at the CS point `p_hy = 2`.)

Substituting `p_hy = 2, q_kernel = 2, K_q_total = K2` collapses the first
factor to `sqrt(M - 1 - 2 sum z_n^4)` and the second to
`sqrt(K2 - 1 - 2 sum k_n^2)`.  `sympy.simplify(RHS_Holder_sub - RHS_CS)`
evaluates to exactly `0`, confirming that MM-10 is the `(p, q) = (2, 2)`
case of HM-10 and that the exponent bookkeeping (`1/(p-1)` on `M`,
`2p` on `z_n`, `q` on `k_n`) is internally consistent.

### Script output (last lines)

```
    RHS_Holder |_{p_hy=2, q_kernel=2, K_q_total=K2} =
                             ________________________    _____________________
           2          2     /          2       2        /         4       4
M + 2*k1*z1  + 2*k2*z2  + \/  K2 - 2*k1  - 2*k2  - 1 *\/  M - 2*z1  - 2*z2  - 1  + 1

[4] simplify(RHS_Holder_sub - RHS_CS) =
0

[5] Reduction verified symbolically: True

[6] Numeric spot-check at a small test point:
    RHS_CS        = 2.7747222563968721545
    RHS_Holder_sub= 2.7747222563968721545
    |difference|  = 0

PASS: RHS_Holder at (p_hy, q_kernel) = (2, 2) reduces to RHS_CS (MM-10).
```

A numeric spot-check at
`(M, K2, z1, z2, k1, k2) = (1.2, 4.2, 0.2, 0.1, 0.85, 0.5)` confirms
bit-for-bit agreement of the two RHS values (difference exactly `0`).

## §5 Rigorous computation of `K_q` at `q != 2`

For `q in (1, 2]`, `q != 2`, we compute a rigorous UPPER bound on
`K_q = sum_{j in Z} k_j^q` by combining a direct arb sum over `|j| <=
J_tail` with an analytic Krasikov tail bound.

**NB.**  `sum_j k_j` diverges (because `K = (1/delta)(beta*beta)(./delta)`
is *not* in `L^1` of the line — its samples are NOT Poisson-summable in the
usual way; `K(0) = +infinity`).  So the naïve Poisson identity
`sum k_j = 2/(pi delta)` is **false**, and a tail bound of the form
`sum_{|j|>J} k_j^q <= sum_{|j|>J} k_j` is useless (divergent on the
RHS).  We instead use a direct envelope on `|J_0(x)|`.

**Krasikov (2001) bound** (sharp for `x >= sqrt(5/2)`):

  `J_0(x)^2  <=  (2 / pi) / sqrt(x^2 + 3/2)`    for `x >= sqrt(5/2) ~ 1.58`.

Dropping `+3/2` in the denominator gives the looser but clean

  `J_0(x)^2  <=  (2 / pi) / x`    for `x >= sqrt(5/2)`,

hence for `j` such that `pi j delta >= sqrt(5/2)`, i.e.
`j >= sqrt(5/2)/(pi delta) ~ 3.65`,

  `k_j^q  <=  ( 2 / (pi^2 j delta) )^q  =  (2/(pi pd))^q / j^q`,   pd := pi delta.

**Tail via integral test** (valid because `q > 1`):

  `sum_{j > J_tail} j^{-q}  <=  J_tail^{1-q} / (q - 1)`.

**Final bound.**  With `J_tail >= 8` (comfortably above the Krasikov
threshold) and paired `+j / -j`,

```
K_q  <=  1  +  2 * sum_{j=1..J_tail} k_j^q
         +  2 * (2/(pi pd))^q * J_tail^{1-q} / (q - 1).                    (K_q-bound)
```

The first two summands are computed rigorously via arb `bessel_j` /
`**` at `prec_bits >= 256`; the third summand is the Krasikov-tail
padding.

**Tightness at `q -> 2`.**  At `q = 2` the direct sum `1 + 2 sum_{j=1..J_tail} k_j^2`
approaches `K_2 = 0.5747/delta` as `J_tail -> infinity`; the tail
padding `2 * (2/(pi^2 j delta))^2 * J_tail^{-1} / 1 =
8/(pi^4 delta^2) * J_tail^{-1}`, quite small for `J_tail ~ 1024`.
**However** at `q = 2` exactly we SHORT-CIRCUIT to `K_q := K_2 =
0.5747/delta` from `PhiMMParams`, guaranteeing bit-identical agreement
with MM-10 at `p = q = 2`.

**Implementation.**  `phi_holder.PhiHolderParams` stores a precompiled
`K_q_upper : arb` computed by (K_q-bound) with `J_tail = 1024` (default).
`certify_holder.py` re-computes the same bound independently in its
verifier.

## §6 Empirical sweep protocol

We sweep `p in {2, 9/4, 5/2, 11/4, 3, 7/2, 4, 5, 6, 10}` and their
conjugates `q = p/(p-1) in (1, 2]`.  At each `p` we compile
`PhiHolderParams`, then bisect `M` in `[1.27, 1.276]` at `N = 1` with
`tol_q = 10^{-4}`, cell budget `5 * 10^5`, arb `256` bits.

Acceptance criteria:
  (i) at `p = 2`, `M_cert` matches the MM-10 Phase-2 N=1 value within
      `10^{-6}`;
  (ii) for the best `p*`, rerun at `N = 2` with the full filter panel
       (`F1, F2, F4_MO217, F7, F8, F_bathtub`) and verify via
       `certify_holder.py`.

See `sweep_p.py` and `run_all.py` for the driver.

## References

  * Matolcsi, Vinuesa, arXiv:0907.1379 (Lemma 3.3, eq. (10)).
  * Multi-moment derivation:
    `delsarte_dual/multi_moment_derivation.md` (Theorem 2, eq. MM-10,
    Lemma 1).
  * Martin, O'Bryant, Conj 2.9: `delsarte_dual/mo_2004.txt` line 783;
    Hausdorff–Young context at line 474, (p*, q*) = (4/3, 4) numeric
    heuristic at line 755.
  * This folder's literature notes: `literature_notes.md`.
