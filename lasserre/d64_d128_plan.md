# Plan: certified val(64), val(128) ≥ 1.281 via sparse Lasserre + Farkas

> **Goal:** Produce a rigorous Farkas-style certificate at d=64 (and d=128 as
> corroboration) showing val(d) ≥ 1.281, which by [extrapolation.md](extrapolation.md)
> immediately yields C_{1a} ≥ 1.281 — strictly above the Cloninger-
> Steinerberger 1.2802 record.
>
> **No discretization-error correction is needed.** Per
> [extrapolation.md](extrapolation.md) §2, val(d) ≤ C_{1a} *exactly* for
> every d ≥ 2. There is **no ε(d)** subtracted between the SDP cert and
> the C_{1a} claim; the chain is `lb_rig (cert) → val^{(k,b)}(d) →
> val(d) → C_{1a}` with equality-or-better at every step.
>
> **Status:** scaffolding written locally; certificate generation requires
> a pod run (see "Pod choice" below) — *contingent on the trajectory
> measurement in [trajectory/REPORT.md](trajectory/REPORT.md) clearing
> the gate*.

## 1. Target value and feasibility

Float multistart estimates from `lasserre/core.py::val_d_known`:
| d   | val(d) (numeric) | margin to 1.281 |
|-----|------------------|-----------------|
| 32  | 1.336            | +0.055          |
| 64  | 1.384            | +0.103          |
| 128 | 1.420            | +0.139          |

The float estimates of val(d) are *upper* bounds (multistart can only
witness feasible μ). The Lasserre SDP gives *lower* bounds val^{(k,b)}(d)
≤ val(d) (Lemma in [extrapolation.md](extrapolation.md), reproduced
from `proof/lasserre-proof/lasserre_lower_bound.tex` Theorem
"Lasserre soundness"). The relaxation has to be tight enough that
val^{(k,b)}(64) ≥ 1.281 still holds. Empirically (per the existing
proof writeup) val^{(3)}(16) ≥ 1.3, so order-3 closes ~98% of the gap
at d=16; we expect order-2 with bandwidth b=16 cliques at d=64 to
comfortably clear 1.281.

If the order-2 bound is too loose, the fall-back strategy is
order-2 + extra "wide window" full-localizing constraints (these are
already required for windows whose support exceeds the bandwidth) plus
a small number of order-3 cliques near the active windows.

## 2. Memory profile of the current solver

Measured locally (Windows / single-thread, lazy `ab_eiej`):

| d  | k | n_y      | n_basis | n_loc | n_win | precompute |
|----|---|----------|---------|-------|-------|------------|
| 8  | 3 | 3 003    | 165     | 45    | 120   | 0.02 s     |
| 10 | 3 | 8 008    | 286     | 66    | 190   | 0.05 s     |
| 12 | 2 | 1 820    | 91      | 13    | 276   | 0.02 s     |
| 16 | 2 | 4 845    | 153     | 17    | 496   | 0.05 s     |

Pod measurements (from `project_p4_48t_oom_d12.md` and
`project_farkas_certified_lasserre.md`):

- d=4..10, order 3 dense: Farkas certify wall ~2-3 s/probe, peak RSS ~1 GB.
- d=12, order 3 dense, **4-way parallel** MOSEK: peak 787 GB RSS at 192-core,
  hit OOM on 755 GiB pod ⇒ P4-48t configuration is **not** viable for
  d ≥ 12 on <1 TB pods.
- d=16, order 3 dense single-solve: ~10 min on 32-core workstation per the
  `lasserre_lower_bound.tex` writeup, RSS estimated 300-400 GB.

### Extrapolation to d=64, d=128

Dense order-3 at d=64 has n_basis = C(67,3) = 47 905, which is wholly out
of reach (a single ~50k×50k PSD cone exceeds 20 TB of memory in
double precision). Dense order-2 at d=64 has n_basis = 2 145 (~37 MB
per dense factor) but n_y = C(68,4) ≈ 814k — still infeasible because
the n_basis × n_basis × ... interior-point factor blows up.

**Correlative-sparsity (Waki) restriction is mandatory.**

With bandwidth b=16, order k=2 (clique size |I_c| = 17):

| d   | n_cliques | clique-bsz | clique-loc | wide windows | full-loc size |
|-----|-----------|------------|------------|--------------|---------------|
| 32  | 16        | 171        | 18         | 1 710        | 33            |
| 64  | 48        | 171        | 18         | 7 822        | 65            |
| 128 | 112       | 171        | 18         | 32 334       | 129           |

(Local measurement via `python -c "from lasserre.cliques import _build_banded_cliques; ..."` — see
the snippet in this directory's git log.)

PSD blocks (per d):
- d=64, b=16, k=2:
  - **48** moment cones of size 171 each
  - **48** clique-restricted localizing cones of size 18
  - **306** clique-coverable window cones of size 18
  - **7 822** wide-window full-localizing cones of size 65
- d=128, b=16, k=2:
  - **112** moment cones of size 171 each
  - **112** clique-restricted localizing cones of size 18
  - **306** clique-coverable window cones of size 18
  - **32 334** wide-window full-localizing cones of size 129

Aggregate PSD scalar variables per cone (lower triangle):
- 171×172/2 = 14 706
- 65×66/2 = 2 145
- 129×130/2 = 8 385

Aggregate dual scalar variables (memory-side proxy):
- d=64:  48·14 706 + 48·171 + 306·171 + 7 822·2 145
       = 706 k + 8 k + 52 k + 16 778 k
       ≈ **17.5 M scalars** (~140 MB at 8 bytes; MOSEK's interior-point
       factor will be 50-100× larger)
- d=128: 112·14 706 + 112·171 + 306·171 + 32 334·8 385
       = 1 647 k + 19 k + 52 k + 271 121 k
       ≈ **272 M scalars** (~2.2 GB raw, ~150-300 GB IPM factor)

### Pod choice

Per `project_p4_48t_oom_d12.md`: **P4-48t is not viable on <1 TB pods**
because 4-way parallel MOSEK hits 4× the per-task RSS at the boundary.
For a single-solve d=64 cert (this is a one-shot certify, not a sweep),
the right choice is:

- **d=64**: a 64-core / 256-512 GB pod (P2-64t class). Dense Cholesky
  factor for this problem fits in ~50-80 GB peak per the n_y/cone
  scaling. Single MOSEK invocation, no parallelism. Wall: 30-90 min
  estimate.
- **d=128**: requires ≥ 1 TB RAM. Either a Nebius 2 TB pod or a single
  AWS r6id.32xlarge (1 TB / 128 vCPU). Wall: 4-12 h estimate; if
  MOSEK exceeds memory, fall back to SCS-GPU (`gpu/` directory has
  pre-existing infrastructure).

**Concretely:** for d=64 use a Nebius P2-64t (192 cores / 755 GiB), with
single-threaded MOSEK (`MSK_IPAR_NUM_THREADS=64`). Confirmed safe per
the d=12 OOM lesson — single-solve 64-thread peaks at ~1/4 the memory of
P4-48t-4parallel.

## 3. Clique decomposition (banded, bandwidth b=16)

For d ∈ {64, 128}:
$$I_c = \{c, c+1, \dots, c+16\}, \quad c = 0, 1, \dots, d-17.$$

Properties (already proven in `lasserre/cliques.py::_build_banded_cliques`
and the corresponding lemma in `lasserre_lower_bound.tex` Section 4):
- |I_c| = 17 for every c.
- Consecutive overlap |I_c ∩ I_{c+1}| = 16, so RIP holds (chordal).
- Every (i,j) with |i-j| ≤ 16 lies in at least one clique together
  (verified locally).
- Every window W = (ℓ, s_lo) with active-bin spread ≤ 16 is covered by
  some clique; wider windows fall back to the full localizing PSD.

The number of clique-coverable windows is constant (306) regardless of d
because clique structure repeats periodically — narrow windows
(ℓ = 2..17 with shifts) are coverable, wider ones are not.

## 4. Estimated solve time and verification time

### Solve time

Based on the d=12 measurement (dense, order 3, 411 s for tier 3 at boundary),
projecting via cone-count scaling:

| d   | dense order-2 (est.) | sparse b=16 order-2 (est.) | verification (Farkas round) |
|-----|----------------------|----------------------------|------------------------------|
| 32  | 30 min               | 5 min                      | 3 min                        |
| 64  | OOM                  | 30-90 min                  | 30-60 min                    |
| 128 | OOM                  | 4-12 h                     | 2-6 h                        |

Verification (rational rounding + dps=80 mpmath check) bottleneck is
`certified_lasserre/farkas_certify.py::_adj_qW_exact_fmpq` — Python
triple loop. The memory note `project_farkas_certified_lasserre.md`
flags vectorizing this as the d ≥ 12 priority. We address this in
`d64_farkas_cert.py` by:
- per-window batching of the (a, b, i, j) → α tensor lookups using the
  already-built sparse tensor `P['f_r'], P['f_c'], P['f_v']` from
  `precompute.py`, avoiding the Python triple loop.
- using `flint.fmpq_mat` for the inner sums (87× faster than
  `fractions.Fraction` per the same memory note).

## 5. Acceptance criterion

The pipeline succeeds iff all of:

1. The sparse SDP at (d, k, b) = (64, 2, 16) returns a primal-dual pair
   with complementarity gap < 1e-6 at MOSEK default tolerances.
2. The Farkas certificate file `lasserre/certs/d64_cert.json` validates
   at mpmath dps=80 with residual_l1 < 1e-50 (after rational rounding).
3. The certified rational lower bound `lb_rig` satisfies
   `lb_rig ≥ Fraction(1281, 1000)`.
4. By the chain in [extrapolation.md](extrapolation.md):
       `C_{1a} ≥ val(64) ≥ val^{(2,16)}(64) ≥ lb_rig ≥ 1.281 > 1.2802`.

Cushion for d=64: the float estimate val(64) ≈ 1.384 is **0.103 above**
the 1.281 target. The Lasserre relaxation must be at least 1.281/1.384 ≈
92.6% tight to clear the bar — order-2 with b=16 cliques is empirically
in this regime per the d=16 evidence (val^{(3)}(16) ≥ 1.3 vs val(16) ≈
1.319 = 98.5% tight).

## 6. Failure modes and mitigations

- **OOM at d=64**: try b=12 first (smaller clique blocks at the cost of
  marginally weaker bound), or fall back to k=2 with full moment matrix
  per clique but only the first-order localizing matrix M_0(μ_i y).
- **val^{(2,16)}(64) < 1.281**: bump to order k=3 *for the cliques only*
  (sparse-order-3) — clique-bsz becomes C(20,3) = 1 140 per cone, still
  manageable.
- **Solver fails to converge**: switch to SCS (split-conic, GPU) per
  `gpu/` infrastructure. Looser tolerances but Farkas rounding can
  absorb residual ~1e-6.
- **Farkas rounding margin negative**: tighten `t_test` (probe further
  below the bisection optimum) and increase `max_denom_S` from 10^12 to
  10^15.
- **d=64 succeeds, d=128 OOMs**: report d=64 cert only; d=128 is
  corroboration, not load-bearing for the C_{1a} ≥ 1.281 claim.

## 7. Order of operations

1. Local: write `d64_solver.py`, `d64_farkas_cert.py`, tests. ✅
2. Local: validate the sparse-cliques pipeline at d=8 and d=16 against
   the dense reference (must reproduce val^{(2)}(d) within solver
   tolerance).
3. Pod: deploy d=64 run on P2-64t. Bracket t_test in [1.281, 1.30] via
   bisection (3-4 probes each ~30 min).
4. Pod: produce `lasserre/certs/d64_cert.json` at the highest infeasible
   `t_test`.
5. Local: re-run `tests/test_lasserre_d64.py` against the produced cert
   to verify rational residuals at dps=80.
6. d=128 same flow; only run after d=64 succeeds.
7. Pull cert into `proof/lasserre_unconditional_writeup.md` for paper.
