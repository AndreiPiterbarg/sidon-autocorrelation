# delsarte_dual

The **Piterbarg-Bajaj-Vincent Bound**: a rigorous lower bound on the
Sidon autocorrelation constant

```
C_{1a}  :=  inf { ||f * f||_inf / (int f)^2
                  :  f >= 0,  supp f subset (-1/4, 1/4),  int f > 0 }

        >=  1292 / 1000  =  1.292
```

via a multi-scale arcsine kernel applied to the Matolcsi-Vinuesa (2010)
master inequality, with all transcendentals computed in `flint.arb`
interval arithmetic at 256-bit precision and all algebraic inputs as
exact `flint.fmpq`.

The bound improves on the previously announced value `1.2802` of
Cloninger and Steinerberger (2017, arXiv:1403.7988) and on the rigorous
analytic bound `1.27481` of Matolcsi and Vinuesa (2010, arXiv:0907.1379).
The accompanying writeup *A New Lower Bound for the Supremum of
Autoconvolutions* is at the repository root in
[`lower_bound_proof.pdf`](../lower_bound_proof.pdf); the Lean 4
formalisation of the same statement is at
[`lean/Sidon/MultiScale.lean`](../lean/Sidon/MultiScale.lean).

## Pipeline

1. **Kernel.**  Build the three-scale convex combination

   ```
   K = sum_{i=1}^{3} lambda_i K_arc(delta_i; .)
   ```

   with `(delta_1, delta_2, delta_3) = (138, 55, 25) / 1000` and
   `(lambda_1, lambda_2, lambda_3) = (85, 10, 5) / 100`.  See
   [`grid_bound_alt_kernel/kernels.py`](grid_bound_alt_kernel/kernels.py):
   class `MultiScaleArcsineKernel`.

2. **Cosine multiplier `G`.**  Solve the semi-infinite quadratic
   programme

   ```
   min  sum_{j=1}^{200} a_j^2 / hat K(j/u)
   s.t. sum_{j=1}^{200} a_j cos(2 pi j x / u) >= 1   for x in [0, 1/4]
   ```

   discretised to a fine grid and rounded to `fmpq` with denominator
   `1e8`.  See
   [`grid_bound_alt_kernel/optimize_G.py`](grid_bound_alt_kernel/optimize_G.py).

3. **Rigorous `min G`.**  Certify a lower bound on
   `min_{x in [0, 1/4]} G(x)` via Taylor branch-and-bound in arb
   intervals.  See
   [`grid_bound/G_min.py`](grid_bound/G_min.py).

4. **Master inequality `Phi`.**  Evaluate

   ```
   Phi(M, y)  =  M + 1 + 2 y k_1 + sqrt((M - 1 - 2 y^2)(K_2 - 1 - 2 k_1^2))
                  - (2/u + a)
   ```

   as an arb enclosure.  See
   [`grid_bound/phi.py`](grid_bound/phi.py).

5. **Cell search.**  Adaptive priority-queue cell bisection on
   `y in [0, mu(M)]`; verdict `CERTIFIED_FORBIDDEN` iff every terminal
   cell has `Phi.upper() < 0`.  See
   [`grid_bound/cell_search.py`](grid_bound/cell_search.py).

6. **`M` bisection.**  Bisect to find the largest `M` with a certifiable
   ``Phi < 0`` witness; the result is the rigorous lower bound `M_cert`.
   See
   [`grid_bound_alt_kernel/bisect_alt_kernel.py`](grid_bound_alt_kernel/bisect_alt_kernel.py).

## Layout

```
delsarte_dual/
|-- README.md
|-- __init__.py
|-- grid_bound/                          MV master inequality machinery
|   |-- __init__.py
|   |-- bessel.py                        arb wrappers for J_0(pi j delta / u)
|   |-- bisect.py                        single-scale MV reproduction driver
|   |-- cell_search.py                   priority-queue cell B&B certifier
|   |-- certify.py                       standalone independent verifier
|   |-- coeffs.py                        119-coefficient MV baseline data
|   |-- G_min.py                         Taylor B&B for min G
|   `-- phi.py                           PhiParams + phi_N1 + mu_of_M
`-- grid_bound_alt_kernel/               multi-scale kernel + production driver
    |-- __init__.py
    |-- bisect_alt_kernel.py             production pipeline -> certificate
    |-- certificates/
    |   `-- reference_anchors.json       canonical anchor values
    |-- kernels.py                       Kernel base + Arcsine + MultiScale
    `-- optimize_G.py                    QP solver for G
```

The accompanying paper ``lower_bound_proof.{pdf,tex}`` and the Lean 4
formalisation ``lean/Sidon/MultiScale.lean`` live at the repository
root.

## Running the production bound

```bash
python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel
```

Produces a JSON certificate at
`delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`
containing the rational `M_cert`, all input parameters, the QP-rounded
`G` coefficients, the five interval-arithmetic anchors
`(k_1, K_2, S_1, min_G, a)`, the complete bisection history, and the
terminal cell list of the certifying cell-search.

## Independent verification

`grid_bound/certify.py` re-checks every quantitative claim from a
certificate using only `python-flint` primitives, with no imports from
the rest of the package:

```bash
python -m delsarte_dual.grid_bound.certify <certificate.json>
```

The verifier:

1. Recomputes the SHA-256 body hash and confirms it matches the
   certificate.
2. Recomputes `k_1`, `K_2`, `S_1`, `min G`, `a` in arb at the declared
   precision.  For the multi-scale certificate this uses the
   cross-Bessel integrals plus asymptotic tail bound; the diagonal
   `K_2`-surrogate flag is honoured.
3. Confirms the certificate's terminal cells cover
   `[0, mu(M_cert).upper()]` contiguously.
4. Confirms every terminal cell has `Phi.upper() < 0`.

Exit code `0` on success; `1` on any failure.

## Numerical anchors (3-scale, N = 200, 256-bit precision)

| Anchor       | Value                                       |
|--------------|---------------------------------------------|
| `k_1`        | `>= 0.92124658`                             |
| `K_2`        | `in [4.788823, 4.788906]`                   |
| `S_1`        | `<= 29.840907`                              |
| `min G`      | `>= 0.99997987`                             |
| `a`          | `>= 0.21009214`                             |
| `M_cert` (production)   | `= 66167/51200 ≈ 1.29232422` (tighter bound from the live driver)|
| `M_cert` (slack-anchor) | `>= 1.29215650` (recorded in `reference_anchors.json`)|
| Headline rational       | `1292/1000` (used in the paper and Lean)|

These are the values quoted by the writeup and by the slack rationals
in `lean/Sidon/MultiScale.lean`; the sole user axiom in that module
(`MV_master_inequality_for_extremiser`) substitutes these rational
slacks for the analytic `K_2` and `a`.  The values are reproduced
exactly by the production driver above, and recomputed independently
by `certify.py`.

## Tests

```bash
pytest tests/grid_bound_alt_kernel/
```

The test suite at the repo-root `tests/` directory exercises kernel
admissibility, Bochner positivity, the QP solver convergence, and the
single-scale arcsine baseline against the published Matolcsi-Vinuesa
value of `1.27481`.

## Dependencies

  * `python-flint`  (arb / acb / fmpq backend; required)
  * `numpy`
  * `cvxpy`         (QP solver; with MOSEK preferred, CLARABEL/SCS/ECOS fallbacks)

## References

  * Matolcsi, M., Vinuesa, C.  *Improved bounds on the supremum of
    autoconvolutions.*  J. Math. Anal. Appl. **372** (2010), 439-447,
    arXiv:0907.1379.
  * Cloninger, A., Steinerberger, S.  *On suprema of autoconvolutions
    with an application to Sidon sets.*  Proc. Amer. Math. Soc.
    **145** (2017), 3191-3200, arXiv:1403.7988.
  * Martin, G., O'Bryant, K.  *The supremum of autoconvolutions, with
    applications to additive number theory.*  Illinois J. Math.
    **53** (2009), 219-235, arXiv:0807.5121.
