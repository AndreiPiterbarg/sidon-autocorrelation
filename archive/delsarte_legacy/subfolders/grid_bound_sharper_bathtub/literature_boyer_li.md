# Boyer-Li (2025) and related literature on sharper bathtub bounds

**Source audit of:** arXiv:2506.16750 (Boyer-Li 2025), arXiv:0907.1379 (Matolcsi-Vinuesa 2010),
arXiv:2510.10172 (Kolountzakis-Lev-Matolcsi 2025), arXiv:math/0302193 (Kolountzakis-Revesz
2003), and project notes `ideas_boyer_li.md`, `ideas_turan_krein.md`, `theory.md`.

## 1. Matolcsi-Vinuesa Lemma 3.4 (baseline bound)

**Statement.** If `h >= 0` on `R`, `supp(h) in [-1/2, 1/2]`, `int h = 1`, `h <= M`, then for
every integer `n` with `|n| >= 1`:

    |hat h(n)|  <=  (M / pi) sin(pi / M)  =:  mu(M).

**Extremizer (MV).** `h* = M * 1_{A*}` with `A*` a super-level set of `cos(2 pi n x)` on
`[-1/2, 1/2]` of Lebesgue measure `1/M`.

**MV acknowledge the gap** (their Remark after Theorem 3.2, MV p. 7 lines 356-380):
> "Lemma 3.4 does not exploit the fact that h is an autoconvolution. It is possible
> that a much better upper bound on |hat h(1)| can be given in terms of M if we
> exploit that h = f * f."

This is open problem #1, also recorded at
`delsarte_dual/mv_construction_detailed.md` lines 493-495.

## 2. Boyer-Li 2025 (arXiv:2506.16750)

**Searched for:** explicit "sharper Lemma 3.4", tighter `|hat h(n)|` bound exploiting
`h = f*f`, Toeplitz-PSD arguments, Fejer-Riesz / Akhiezer factorization tricks, Krein-Nudelman
truncated moment arguments.

**Findings:** Boyer-Li **do not state a sharper bathtub bound.** Their improvements on
the Sidon constants target the `L^2` ratio `c` (they achieve `c >= 0.901564`); they do not
produce any formula `mu'(M, n) < mu(M)` for autoconvolutions.

Their relevant techniques:

1. **Integer-scaled step heights** (Section 4.1). Scale `f`'s piecewise-constant values by
   `2^24` and compute `f*f` in exact rational arithmetic. Gives exact rigorous witnesses
   but is a technique, not a bound.
2. **Piecewise-linearity of `f*f`.** For `f` a step function on unit-spaced grid, `f*f` is
   exactly piecewise linear with slopes in `{..., -1, 0, 1, ...}` on the integer grid.
   `||f*f||_inf` is attained at a vertex, so exact. Useful for witnesses, not dual bounds.
3. **Two-phase search.** Simulated annealing -> gradient ascent on upscaled grids. Search
   heuristic; no bathtub sharpening.

**Verdict on Boyer-Li:** zero direct bearing on the sharper-Lemma-3.4 question.

## 3. Kolountzakis-Revesz (2003, arXiv:math/0302193)

A sharper pointwise bound on positive-definite functions supported on a compact set:

    h(x) <= h(0) * (1 - |x|/a)_+   (h is PD and supported on [-a, a])

If this applied to `h = f*f` (supported on `[-1/2, 1/2]`, i.e. `a = 1/2`), Fourier-transforming
gives `hat h(n) <= h(0) * triangle_hat(n)` where `triangle_hat(n) = sin^2(pi n / 2) / (pi n / 2)^2 / 2`.

**Caveat:** Kolountzakis-Revesz assume `h` is positive-definite AND supported on `[-a, a]`
and have the bound `h(x) <= h(0) * (1 - |x|/a)`. For `h = f*f` this is NOT automatic —
autoconvolutions are positive-definite only after a sign convention. The triangle majorant
applies to `h(x) / h(0)` which requires `h(0)` in the numerator and one has `hat h(n) / h(0)`
on the RHS. Since `h(0)` for autoconvolutions is typically `>= 2 >> 1 = int h`, the
bound becomes vacuous after dividing by `h(0) >> 1`.

**Verdict:** KR-2003 does not directly give a sharper `mu'(M, n)`.

## 4. Kolountzakis-Lev-Matolcsi (2025, arXiv:2510.10172)

Strong duality for the Turan problem on locally compact abelian groups. Not a direct
bathtub sharpening; conceptually similar SDP-duality framework.

## 5. Summary table

| Technique | Source | Sharper `mu'(M, n)`? | Applies to `h=f*f`? |
|-----------|--------|----------------------|---------------------|
| MV Lemma 3.4 | MV 2010 | baseline | yes but loose |
| Boyer-Li exact arithmetic | BL 2025 | no (witness technique) | yes |
| Kolountzakis-Revesz triangle majorant | KR 2003 | vacuous at `h(0) >> 1` | yes |
| Fejer-Riesz / Akhiezer factorization | classical | **potentially yes** | yes |
| Toeplitz-PSD SDP with L_inf LMI (Krein-Nudelman) | Dumitrescu 2007 | **potentially yes** | yes |

## 6. Bottom line

**No published paper** contains an explicit formula `mu'(M, n) < mu(M)` for autoconvolutions
in the regime `M in [1.2, 1.3]` that we need. MV themselves flagged this as open. The
two promising paths are:

1. **SDP dual** with the positive-definite sequence `{hat h(n)}` constrained to nonneg
   entries and bounded trigonometric realization `<= M` (Path 1 in `proof_paths.md`).
2. **Fejer-Riesz / Akhiezer factorization** `h = |p|^2` with `p` entire of exponential
   type `pi` (Path 2 in `proof_paths.md`).

Both require nontrivial rigorous development (dual certificate + rational arithmetic
verification). Neither is a 2-hour job.
