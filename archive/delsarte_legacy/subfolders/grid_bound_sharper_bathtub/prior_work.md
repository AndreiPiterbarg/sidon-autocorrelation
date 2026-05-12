# Prior work: sharper bathtub attempts in the repo

**Scope of scan.** `delsarte_dual/`, `lasserre/`. Keyword sweep: "bathtub",
"Chebyshev-Markov", "sharper", "Toeplitz", "Akhiezer", "Krein", "F8", "mu_sharper",
"Lemma 3.4", "autoconvolution".

## 1. Existing `F_bathtub` and `mu(M)`

- **`delsarte_dual/grid_bound/filters.py` lines 67-94**
  Rigorous arb-interval implementation of MV Lemma 3.4:
  `a_n^2 + b_n^2 <= mu(M) = M sin(pi/M)/pi`. Accepts an `arb` upper bound on mu.
- **`delsarte_dual/grid_bound/phi_mm.py` line 172**
  `mu_of_M(M) = M * sin(pi/M) / pi` in arb form.
- **`delsarte_dual/mv_bound.py` lines 55-60**
  mpmath MV-bound reproduction; `MV_BOUND_FINAL = 1.27481`.
- **`delsarte_dual/multi_moment_derivation.md` Lemma 1**
  Proof that `mu(M)` is `n`-INDEPENDENT (extremizer is the `n`-peak comb for `n >= 2`).
  Fixes the naive `M sin(pi n / M)/(pi n)` which is unsound.

## 2. Existing `F8` (Toeplitz PSD on `hat h`)

- **`delsarte_dual/grid_bound/filters.py` lines 300-341**
  F8 constructs `T_h = [hat h(i-j)]_{i,j=0..N}` with `hat h(n) = hat f(n)^2` and
  rejects cells where the leading principal minor is rigorously `< 0`.
- **Relation to sharper bathtub**: F8 *already* enforces the autoconvolution
  structural Toeplitz-PSD constraint as an admissibility cut per-cell in the
  N-D cell search. It does **not** derive a scalar sharper `mu'(M, n)`; it cuts
  jointly on the vector `(hat f(1), ..., hat f(N))`. The scalar bathtub and the
  joint PSD are **complementary**, not redundant.

## 3. MV's own acknowledged open problem #1

Quoted from `delsarte_dual/mv_construction_detailed.md` lines 493-495:

> **1. Sharper Lemma 3.4.** The current Lemma 3.4 only uses `h >= 0`, `int h = 1`,
> `h <= M`. Exploiting that `h = f*f` (and not just any nonneg function) could give
> a tighter bound on `|hat f(1)|`.

No attempt in the repo has produced a proven quantitative sharper `mu' < mu`.

## 4. Adjacent efforts (multi-moment refinement)

- **`delsarte_dual/mv_multimoment.py`** — extends MV eq. (10) to pull out
  `z_1, z_2, ..., z_N` jointly; still bounds each `|hat h(n)|` via `mu(M)`. This is
  NOT a sharpening of mu — it tightens the dual master inequality by adding cheap
  moments on the LHS.
- **`delsarte_dual/mv_lemma217.py`** — adds Martin-O'Bryant Lemma 2.17:
  `Re hat f(2) <= 2 Re hat f(1) - 1`. Orthogonal to bathtub (linear cut, not Fourier
  magnitude bound).
- **`delsarte_dual/grid_bound_holder/`**, **`delsarte_dual/grid_bound/`** —
  sibling Phase-2 efforts. None ship a sharper mu.
- **`IMPROVEMENT_LIST.md`, `MASTER_PLAN.md`** — identify MO-2.17 and "union of
  forbidden sets" as next priorities; sharper Lemma 3.4 is not scheduled.

## 5. Related code snippets

**F8's Toeplitz construction (existing):**
```python
def _T_h_acb(ab, N):
    def hat_h(k):
        if k == 0: return acb(1)
        if k > 0:
            fc = acb(ab[2*(k-1)], ab[2*(k-1)+1])
            return fc * fc
        k = -k
        fc = acb(ab[2*(k-1)], -ab[2*(k-1)+1])
        return fc * fc
    return [[hat_h(i-j) for j in range(N+1)] for i in range(N+1)]
```

**F_bathtub consumer (existing):**
```python
def F_bathtub(ab, N, mu_arb):
    for n in range(1, N+1):
        z_sq = _arb_sqr(ab[2*(n-1)]) + _arb_sqr(ab[2*(n-1)+1])
        if _arb_nonneg(mu_arb - z_sq) == REJECT:
            return REJECT
    return ...
```

## 6. Summary

The repo currently has:
- A sound `mu_MV(M)` at filter level (`F_bathtub`).
- A sound Toeplitz-PSD cut on `hat h` (`F8`).

The repo does **not** have:
- Any scalar sharper `mu'(M, n) < mu_MV(M)`.
- Any SDP-based rigorous dual certificate for Fourier coefficients of autoconvolutions.

Open: this directory is the first attempt in the repo at a scalar sharper bound.
