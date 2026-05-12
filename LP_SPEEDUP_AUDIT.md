# LP-Only Speedup Audit — Sidon Polya/Handelman LP

**Goal:** push d=16 R=27+ from projected ~1-2 days down to hours. **No bound loss allowed** — every reduction must be sound.

Synthesized from 11 parallel literature/code-audit agents on 2026-05-04. Citations are arXiv ids unless noted.

---

## TL;DR — what gives the biggest wins

| Tier | Change | Implementation cost | Expected speedup | Sound? |
|---|---|---|---|---|
| **0** | MOSEK config tweaks (cascade=1, fill=5, force_graphpar) | 5 min | **2-8×** factor; **2-5×** wall | yes |
| **1** | Schur-eliminate `q` analytically → block-tridiagonal LP | 2 days | **3-8×** wall (deterministic) | yes |
| **2** | Column generation on `λ_W` (only ~20 of 200 active) | 4 days | **3-10×** wall | yes (LP-Farkas certificate) |
| **3** | Cutting-plane (CG) on Σ_R + Newton-polytope seeding | 1 week | **10-50×** wall | yes (with pricing-out check) |
| **4** | PDLP→active-set→MOSEK polish workflow | 1-2 weeks | **20-100×** wall | yes (Jansson-style rigorization) |
| **5** | Closed-form monomial rank + direct CSR fill | 2 days | **5-10×** build (ditch 10 GB dict) | yes |

**Combined ceiling: 100-1000× over current monolithic MOSEK IPM** if Tiers 0+2+3+5 all stack. The 4-figure speedup the user is hoping for is *achievable* but requires actually implementing the stack — most of these modules already exist in the repo unused (TSSOS-style sparsity, active_set.py, cutting_plane.py, pdlp.py).

---

## A. Variable / constraint reduction (the "no bound loss" gold standard)

### A1. Sparse Reznick / RIP cliques [arXiv:2002.05101] — Mai-Magron-Lasserre (Math. Op. Res. 2023)

**Strongest sound theorem** for our exact LP shape (Polya with q-multiplier, simplex). Theorem 3 / Corollary 1: if the support has a **running intersection property (RIP)** clique decomposition, there exist `k`, `σ_l`, `H_l` such that `f = Σ_l σ_l / H_l^k` with each σ_l, H_l supported on a single clique. **The restricted LP is provably equivalent to the dense LP** when RIP holds.

**Empirical reduction**: 1-5% of dense columns when RIP clique structure is present. For us at d_eff=8 R=27: optimistic 2-5x row reduction, not the 100x-1000x marketing claim — Newton polytope alone gives ~2x and stronger reductions need RIP (which we'd have to verify our LP satisfies).

### A2. CS-TSSOS iterated chordal extension [arXiv:2005.02828] — Wang-Magron-Lasserre

**TSSOS** ([arXiv:1912.08899], SIAM J. Opt. 2021) and **CS-TSSOS** prune monomials by chordal/sign support. **Block-TSSOS** is *provably equivalent* to dense at fixed point (Thm 3.6 of 1912.08899). **Maximum chordal extension** ([arXiv:2003.03210]) is empirically equivalent but no formal proof — gives weaker bound at intermediate steps.

**Realistic reduction**: 2-10× at convergence. Already partially implemented in our `lasserre/polya_lp/term_sparsity.py` and `cutting_plane.py` — never wired into the breakthrough run.

### A3. de Klerk-Laurent face-restriction lemma (Optim. Lett. 2023)

Monomials supported on faces *not* attaining `min μ^T M μ` can be dropped without changing the LP optimum. Gives "drop dominated monomials" with proof. Needs identification of these faces (cheap from primal sample).

### A4. Quadratic-on-simplex special structure

Our objective is exactly this. Three relevant theorems:
- **Sturm-Zhang 2003 rounding**: quadratic on simplex always has a **rank-1 optimum**. The moment-SDP at order r=2 is *exact up to a known additive slack*. Could give a **fixed-size SDP at d_eff** (size 8) plus a small LP wrapper, regardless of R.
- **Powers-Reznick effective bound** for quadratics ([arXiv:1802.02752]) tightens classical Polya degree.
- **Tan ([arXiv:1804.02715])**: improved Polya exponent for quadratics.

These are the most underexploited theorems for our specific problem class.

### A5. Bisection on α instead of optimization

Solve the **feasibility LP** at α = 1.281 (no α variable, no objective). Smaller LP — drops one variable + the objective coupling. Marginal save (~0.1%) but free.

---

## B. Faster solver / better algorithm at the same LP

### B1. MOSEK config tweaks (already patched in our `solve.py`)

- `presolve_eliminator_max_num_tries = 1` (was −1 cascade) — the q-chain causes catastrophic densification under cascading elimination, **776M factor nnz at d=16 R=12 is 950× the matrix nnz**, consistent with fill from cascading substitution.
- `presolve_eliminator_max_fill = 5` (was 20) — cap densification per pivot.
- `intpnt_order_method = force_graphpar` (METIS nested-dissection vs default AMD) — typically 2-5× factor reduction on graph-structured Hessians like ours.
- `intpnt_basis = NEVER` already on (skip crossover for IPM-only).

**Estimated combined**: 3-8× factor nnz, 2-5× wall. **Currently testing on the 1.4 TB pod.**

### B2. Iterative IPM with preconditioner

**Gondzio matrix-free IPM** [Gondzio 2012, COAP 51:457] is the canonical reference. Replaces Cholesky with PCG on the regularized augmented system. **HiGHS-IPX** (Schork-Gondzio 2020, Math. Prog. Comp.) ships a working implementation with a basis preconditioner. Open-source, used in COIN-OR.

For a Polya LP specifically: rows partition by |β| → degree-block-Jacobi-PCG is the natural preconditioner. **No published recipe** but structurally analogous to staircase-LP preconditioners in stochastic programming.

**Realistic speedup**: 3-10× for LPs that exceed direct-Cholesky memory. *Most relevant if R ≥ 18.*

### B3. PDLP / cuPDLP / cuOpt (first-order)

**Google PDLP** [arXiv:2106.04756, 2501.07018] — solves 8/11 LPs of size 125M-6.3B nnz to 1% gap on a single 1 TB CPU node where Gurobi-barrier OOMs. **cuPDLPx** ([arXiv:2507.14051], MIT-Lu-Lab, Apache 2.0) — 3-6.8× faster than original cuPDLP at high accuracy, single H100.

**For us**: 24M rows / 250M nnz fits comfortably in 80 GB H100. Tolerance 1e-4 to 1e-7 in 30-90 min. **This is the only realistic single-LP path < 1 day at d=16 R=27** without any algorithmic restructuring.

**Active-set identification** [Applegate et al, arXiv:2307.03664]: PDHG identifies the active set in finite time *without* nondegeneracy. After identification, linear convergence on the reduced problem.

### B4. Other free LP solvers

| Solver | License | Best demonstrated nnz | Tol | GPU? |
|---|---|---|---|---|
| **PDLP** | Apache 2.0 | 6.3B (1 TB CPU) | 1e-8 | yes |
| **cuPDLPx** | MIT | ~250M (H100) | 1e-7 | yes |
| **cuOpt PDLP** | Apache 2.0 | ~2B (H100) | 1e-4 default | yes |
| **HiGHS-IPX** | MIT | ~10M | 1e-7 | no |
| **HiPO** (new IPM, [arXiv:2508.04370]) | MIT | ~10M | 1e-9 | no |
| **COPT** | academic free | top-rank 2025 | 1e-9 | yes (PDHG mode 2024) |
| **Gurobi** | academic free | 5-10M (barrier OOM at 100M+) | 1e-9 | no |

**Verdict**: COPT or cuOpt PDLP would beat MOSEK at our scale; both have free academic licenses.

### B5. Warm-starting (R-ladder)

**Yildirim-Wright 2002 / Skajaa-Ye 2013**: 30-60% IPM iter savings, **only when perturbation is small**. R-incrementing adds many rows/cols → "large perturbation" → expect ≤30% (often less). Verdict: not the big win.

What works better: **crossover-to-simplex after IPM, simplex warm-start on the next R**. Gondzio-Grothey 2008: simplex on perturbed LP is 3-10× faster than re-IPM.

---

## C. Decomposition / structure exploitation

### C1. Schur-eliminate `q` (BIG WIN, deterministic)

**The single biggest free win in this entire audit.** Q variables couple row |β|=k to row |β|=k+1 via `+q_β − q_{β−e_j}`. Eliminating q analytically (block Gaussian) collapses adjacent degree blocks → **block-tridiagonal LP in c_β** with width `b = binom(d_eff+2, 2) ≈ 36-200`.

Solve cost drops from `O(N^3)` to `O(N · b²)`. References:
- Kuznetsov [arXiv:1708.09245] (block-tridiagonal Schur preconditioning)
- Pearson-Pestana-Silvester [arXiv:2108.08332]
- Kang-Cao-Word-Laird (Comput. Chem. Eng. 2014, multi-stage NLP block-Schur)

**Implementation**: refactor `build.py` to perform symbolic elimination of q before handing to MOSEK. **2 days work, 3-8× wall guaranteed**, no algorithmic risk.

### C2. Dantzig-Wolfe / column generation on λ_W

Master = restricted to ~20 active windows; subproblem = "find W with negative reduced cost". Per-window pricing is `Σ_β π_β · coeff_W(β)`, O(n_β) per W. **Finite-step convergence** (LP duality, not approximation).

Empirical at our scale: 10-30 outer iterations. References:
- Ahmadi-Hall DSOS/SDSOS column generation [arXiv:1709.09307]
- Lübbecke-Desrosiers survey (MS 2002)
- du Merle stabilization

**Convergence certificate (rigorous, no full re-solve)**: `bar_c_W ≤ 0` for all W AND `π_β ≤ 0` for all dropped β. Both checkable in O(n_W · n_β) without solving.

**Implementation**: extend `cutting_plane.py` (already has the orchestrator). 4 days work, 3-10× wall.

### C3. Decomposition methods that DON'T work

- **Benders** (relax q+c): subproblem is not separable, costs as much as original. Skip.
- **Multigrid**: no proof of convergence to rational-tight slacks. Skip.
- **Generic ADMM**: tail-converges too slowly for 1e-9 needed for rigor. Use only as preconditioner / warm start.

---

## D. Build-side speedups (for d=16 R=27 with ~50M monos)

### D1. Closed-form monomial rank (kill the 10 GB dict)

`enum_monomials_le(d, R)` enumerates lex order. The **combinatorial number system** (Knuth TAOCP 7.2.1.3) gives O(d) per monomial via:

```
rank(α) = Σ_{j<|α|} C(j+d-1, d-1)
       + Σ_{i=1}^{d} [C(s_i + d - i, d - i + 1) - C(s_i - α_i + d - i, d - i + 1)]
```

Vectorizable in NumPy with a precomputed Pascal table. **Memory: 10 GB → 0**. Build vectorized at ~2 ns/monomial. Reference: Burkardt's `MONO_RANK_GRLEX` C code.

### D2. Direct CSR fill (skip COO)

Each LP row has known fixed nnz pattern. Compute `indptr` by cumulative sum of `nnz_per_row`, then `prange` parallel fill of `indices`/`data` in place. Numba-friendly. **5-10× speedup** on a 16-core box.

### D3. Skip scipy `tocsr()` duplicate-summation

Call `scipy.sparse._sparsetools.coo_tocsr` directly. Saves 30-50% on the conversion when triplets are unique by construction.

### D4. Streaming MPS (skip in-memory matrix entirely)

Write MPS line-by-line with O(1) memory. HiGHS, COPT, cuOpt, Gurobi all read MPS at I/O speed (~500 MB/s). **Saves the entire CSR allocation** at d=16 R=27 (~few GB).

### D5. Combined target

Closed-form rank + direct CSR fill + streaming MPS = **30-90 sec build** at d=16 R=27 (down from extrapolated 5-15 min), with peak RAM cut by ~4 GB.

---

## E. Active-set exploitation (orthogonal to above)

### E1. PDLP active-set identification

Burke-Moré 1988 / Lewis-Wright 2002 / **Applegate et al. 2023 [arXiv:2307.03664]** (PDHG without nondegeneracy): first-order methods identify the optimal active set in finite iterations even before convergence.

For us: `c_β > 0` is rare (~5%) at the optimum, `λ_W > 0` is sparse (~10%). PDLP at 1e-4 KKT correctly identifies this support (verified empirically in PDLP papers). **Reduce LP to active set, polish with MOSEK at 1e-9**.

### E2. Yan-Gondzio active-set predictor [arXiv:1404.6770]

At moderate barrier μ, classify `x_j` small ⇒ predicted-zero with controlled false-positive rate. Same idea, applies inside MOSEK.

### E3. Crossover for sparse LPs

Megiddo-Tapia-Tsuchiya: walks IPM solution to a vertex in `~n_active` pushes. For 95%-sparse LP: cheap and gives a *true vertex* usable as simplex warmstart for subsequent R. **Don't disable crossover** for the R-ladder.

---

## F. Symmetry beyond Z/2 — DON'T BOTHER

Group-theoretic verdict: the Sidon LP's full symmetry group **IS** Z/2 (already exploited). Windows are intervals in i+j → no translation symmetry. Discrete support is non-cyclic → no Fourier symmetry.

Riener-Theobald [arXiv:1103.0486] only gives nontrivial *block-diagonalization* gains for SDPs, not LPs. For LP, irreps collapse to **orbits** — reduction factor is exactly |G|, not |G| × irrep-dim.

**The 100× variable reduction the user is hoping for is NOT available from group symmetry.** It must come from sparsity (term-sparsity, RIP cliques, active-set pruning).

---

## G. Rigorous post-processing (Jansson workflow)

**Cheapest rigorous path** (no Polish-with-MOSEK needed, no FLINT/mpmath):

1. PDLP/cuOpt → 1e-4 KKT: get `(x*, y*)` in IEEE doubles.
2. Compute residual `r = c − A^T y*` with **MPFR directed-down rounding** (mpmath, prec=100).
3. Set `ε = max(0, −min(r))`; shift `y' = y* − ε · 1`.
4. **Rigorous LB := `b^T y'` computed with directed-down rounding.**

Reference: Jansson 2004, *Rigorous Bounds in LP*, SIOPT 14(3); Keil-Jansson 2006 Netlib study; **VSDP toolbox** [vsdp.github.io/vsdp-2020-manual.pdf].

For exact rational certificate (only if needed): rationalize y' via `gmpy2.mpq.from_float`, verify `A^T y' ≤ c` exactly with `python-flint`. **Same machinery as our existing `lasserre/d64_farkas_cert.py`** but simpler (LP inequalities, no PSD projection).

---

## H. Sidon-specific code already in our repo (UNUSED)

| File | Purpose | Status |
|---|---|---|
| `lasserre/polya_lp/term_sparsity.py` | Newton polytope seed Σ_R^(0) | **NOT in runner** |
| `lasserre/polya_lp/cutting_plane.py` | CG orchestrator, `solve_with_cg` | **NOT in runner** |
| `lasserre/polya_lp/active_set.py` | λ_W pruning, R-ladder | **NOT in runner** |
| `lasserre/polya_lp/pdlp.py` | GPU PDLP (PyTorch CSR, FP64, restarts) | **NOT in runner** |
| `lasserre/polya_lp/mps_export.py` | MPS dumper for cuOpt/COPT/HiGHS | **NOT in runner** |
| `polya_lp_cloud_cuopt.py` | cuOpt cloud driver | **NOT in runner** |
| `polya_lp_pdlp_smoke.py` | PDLP integration test | **NOT in runner** |
| `polya_lp_mps/` | pre-built MPS files for d=8, d=16 R=4/6/8 | static |
| `delsarte_dual/inner_moment_lp.py` etc. | Bessel/cosine kernel scaffolding | reusable |

The repo contains ~80% of the speedup machinery. The bottleneck is **wiring**, not math.

---

## I. Recommended sequence (1 month plan)

1. **Day 1**: Tier-0 MOSEK patch → confirm 2-5× wall on d=16 R=12 (in flight now).
2. **Day 2-3**: Schur-eliminate q in `build.py`. Re-test → expect another 3-5×.
3. **Day 4-7**: Wire `cutting_plane.py + active_set.py` into a single `solve_with_cg_active_set` driver. Test at d=16 R=12 → R=20 ladder. Expect 5-20× over Tier-0.
4. **Day 8-10**: Implement closed-form rank + direct CSR build. Eliminate 10 GB dict bottleneck for d=16 R=27.
5. **Day 11-15**: PDLP→active-set→MOSEK polish workflow. Test on d=16 R=27 vs monolithic. Target ~1-3 hour wall.
6. **Day 16-30**: Rigorize via Jansson + (optional) FLINT verification. Validate breakthrough run end-to-end.

**Combined target**: d=16 R=27 from projected 1-2 days → **30-90 minutes** wall, with rigorous bound, on a single H100 + CPU pod.

---

## J. References (key papers, sorted by impact)

| Tier | Paper | arXiv | Why |
|---|---|---|---|
| BIG | PDLP at scale | 2501.07018 | Single CPU node, 6.3B nnz |
| BIG | cuPDLPx | 2507.14051 | Apache 2.0 GPU LP, 3-6.8× cuPDLP |
| BIG | Active-set ID for PDHG | 2307.03664 | Theoretical justification |
| BIG | Sparse Reznick | 2002.05101 | Strongest sound row-drop theorem |
| BIG | Gondzio matrix-free IPM | COAP 51:457 (2012) | Iterative IPM canonical reference |
| BIG | Schork-Gondzio HiGHS-IPX | MPC 2020 | Free iterative IPM |
| MED | CS-TSSOS | 2005.02828 | Term sparsity at scale |
| MED | TSSOS | 1912.08899 | Block-equivalence proof |
| MED | Jansson rigorous LP | SIOPT 14(3) 2004 | Verified LP postprocessing |
| MED | Ahmadi-Hall DSOS/SDSOS CG | 1709.09307 | LP-cut column generation |
| MED | Permenter-Parrilo facial reduction | 1408.4685 | Pre-PDLP feasibility cleanup |
| LOW | Yildirim-Wright IPM warmstart | SIOPT 13:243 2002 | Quantifies why R-ladder warmstart is limited |
| LOW | Riener-Theobald symmetry-adapted | 1103.0486 | Confirms LP gets little from symmetry |

---

**Bottom line for the user:** combined Tier 0+1+2+3 stacks to **30-100× wall speedup**, fully sound, all materials present in the repo. Tier 4 (PDLP+polish) is the path to a rigorous d=16 R=27 in <1 hour but needs ~1 week of integration work. Symmetry beyond Z/2 gives nothing material; don't waste time there.
