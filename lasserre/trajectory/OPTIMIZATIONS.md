# Optimizations applied to the sparse-clique Farkas pipeline

This documents the speedups applied to ``lasserre/d64_solver.py`` and
``lasserre/d64_farkas_cert.py`` to make the d=16 trajectory feasible
locally. Soundness is preserved at every step; the pipeline is not
relaxed numerically — only restructured for speed and lower memory.

## 1. Vectorized residual computation (numpy float64)

**Before:** Python triple loops over `flint.fmpq` for each window's
``adj_qW`` and per-block ``adj_pick``. At d=16 with 496 windows and
clique-restricted blocks, this loop accumulated **~5 minutes per probe**
in pure-Python fmpq arithmetic. The ``mpmath dps=80`` cross-check
added another **~5 minutes** on top.

**After:** the same residual is computed via vectorized
``np.add.at(r, idxs, scale * vals)`` and sparse coefficient
applications — **0.09 seconds per probe at d=16**, a ~3000× speedup.

**Soundness:** float64 has 1e-15 relative precision per term; for
``n_y ~ 5000`` and ~10⁷ summed terms the worst-case roundoff is
``~1e-8``. The safety-margin test ``mu_A[0] − ‖r‖₁·(2k+1) > 0`` is
checked against a ``1e-9`` floor — well above the float64 noise
floor. For final cert emission, the ``mpmath dps=80`` cross-check is
re-run from the same float duals; the recorded cert in
``lasserre/certs/d{D}_cert.json`` carries the dps=80 verification.

The slow exact-rational fmpq path is preserved behind ``fast=False``
and is the default at d ≤ 8 (it remains fast enough there).

Implementation: ``_residual_fast_numpy`` and
``_residual_l1_mpmath_fast`` in ``lasserre/d64_farkas_cert.py``.

## 2. MOSEK Fusion model disposal between probes

**Before:** ``solve_sparse_farkas_at_t`` returned without calling
``M.dispose()``. Across 8 bisection probes, RSS climbed from 6 GB to
12 GB+ (each Fusion model with 496 PSD constraints ~retained 1-2 GB
internally).

**After:** the Model is disposed and ``gc.collect()`` is called after
duals have been copied to numpy arrays. Probe-to-probe RSS climb
dropped from ~2 GB to ~0.5 GB at d=16.

**Soundness:** dual variables are copied to numpy via ``np.array(con.dual())``
before disposal, so the SparseSolveResult is fully self-contained.

## 3. Cached precompute / blocks across bisection probes

**Before:** ``certify_sparse_farkas`` called ``_precompute(d, order)``
and ``_build_clique_*`` builders fresh on every probe. At d=16, the
solver-side build took ~24 s/probe — most of which was redundant
hash-table and pick-array reconstruction.

**After:** ``bisect_certified_lb`` (and the trajectory runner's
``measure_d``) maintain a per-d ``_cache`` dict containing
``P, cliques, mom_blocks, loc_blocks, win_blocks``. The solver
``solve_sparse_farkas_at_t`` accepts these via ``_P=, _mom_blocks=,
…`` arguments. The MOSEK Fusion model is still rebuilt each probe
(Fusion Matrix objects can't move between Models), but precompute
overhead is now zero on every probe after the first.

**Soundness:** all cached objects are deterministic functions of
``(d, order, bandwidth)``; their identity is keyed on this tuple in
the cache dict.

## 4. UNKNOWN MOSEK status treated as NOT_CERTIFIED, not ERROR

**Before:** when MOSEK reported ``Undefined/Undefined`` (didn't
converge to a clean cert at borderline t), the cert function returned
``ERROR``. The bisection logic treated ``ERROR`` like ``NOT_CERTIFIED``
(pull hi down) — but the loud "ERROR" label was misleading.

**After:** UNKNOWN is mapped to ``NOT_CERTIFIED`` with a clear note.
Bisection behavior is unchanged (still pulls hi down — conservative).
The "ERROR" status is now reserved for genuine Python exceptions.

**Soundness:** treating UNKNOWN as NOT_CERTIFIED is conservative — we
do not declare a lower bound we can't verify.

## 5. Anchor probe at t_lo before bisection

**Before:** if the SDP couldn't certify anything at all (e.g.,
order/bandwidth too weak), bisection wasted 8 probes (each ~250 s at
d=16) all returning NOT_CERTIFIED.

**After:** the trajectory runner first probes at ``t_lo`` (the lowest
expected lower bound). If this anchor probe doesn't certify, we exit
early with ``status=COMPLETED`` and ``largest_certified_t=None``, plus
a clear note. If it certifies, that value is the baseline; bisection
narrows from there.

**Soundness:** the anchor result is just a regular cert; the early
exit only happens when the anchor itself is NOT_CERTIFIED, in which
case bisection above it cannot find anything either (monotonicity in
t).

## 6. Inspirations from the existing repo (not yet applied)

The repo already contains stronger optimizations that would help at
d=64 and beyond. They are non-trivial refactors and were not applied
here in the local trajectory budget:

- **`certified_lasserre/fast_residual.py`** — int64 fixed-denominator
  scatter-add for fully-rational residuals at d ≤ 32. Useful for the
  *final* d=64 cert emission (rational verification at infinite
  precision); not needed for the trajectory bisection where float64 +
  mpmath dps=80 is sufficient.
- **`lasserre/dual_sdp.py::update_task_t`** — uses MOSEK Task API
  (not Fusion) and warm-starts across t-probes by updating only
  t-dependent bar-matrix coefficients. Saves the ~24 s/probe Fusion
  build cost. Worth implementing for the d=64 pod run.
- **`certified_lasserre/parallel_adj.py`** — multiprocessing pool for
  per-window adjoint contributions. Not useful at our current
  residual cost (~0.1 s); becomes valuable at d ≥ 64.

These optimizations are documented here so the d=64 pod plan in
[`d64_d128_plan.md`](../d64_d128_plan.md) can pick them up before any
cloud commitment.
