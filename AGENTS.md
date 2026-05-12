# Repository overview for coding agents

> **State of the project.** Both lower-bound proofs are written up:
> - cascade (`proof/cs-proof/`): $C_{1a} \ge 7/5 = 1.4$ via multiscale
>   branch-and-prune.
> - Lasserre (`proof/lasserre-proof/`): $C_{1a} \ge 1.3$ via the order-3
>   SDP hierarchy with correlative sparsity (joint with A. Piterbarg).
>
> The active workstreams are: maintaining the two LaTeX manuscripts and
> their Lean formalizations, regenerating the cascade certificate at
> $c_{\mathrm{target}} = 7/5$ inside the Lean kernel, and continuing
> SDP optimization for the Lasserre track at $d \in \{32, 64, 128\}$.

## Problem statement

For any nonnegative $f : \mathbb{R} \to \mathbb{R}_{\geq 0}$ supported
on $[-1/4, 1/4]$ with $\int f = 1$,

$$\max_{|t| \le 1/2} (f * f)(t) \;\ge\; C_{1a}.$$

The constant is unchanged across both proofs; the techniques differ.

## Cascade track (`cloninger-steinerberger/`, `proof/cs-proof/`)

The cascade extends Cloninger–Steinerberger by:

1. partitioning $[-1/4, 1/4)$ into $d = 2n$ half-open bins and tracking
   bin masses;
2. lower-bounding $\|f*f\|_\infty$ by windowed quadratic sums of the
   bin masses (Lemma 2.3 of the manuscript);
3. discretizing the mass simplex via a cumulative-floor map with a
   sharp per-window error correction (Lemma 3.2);
4. branching and pruning dyadically from $d = 4$ through $d = 128$
   with reversal canonicalization, asymmetry pruning, energy-cap
   pruning, and the dynamic per-window threshold.

The terminal cascade run reaches zero survivors at $d = 128$, $m = 20$
for $c_{\mathrm{target}} = 7/5$. The analytic reduction is formalized
in Lean under `lean/Sidon/Proof/`; the current verified instantiation
is the $32/25$ baseline (`autoconvolution_ratio_ge_32_25`). Updating
to $7/5$ only requires replacing the computational axiom
`cascade_all_pruned` with one matching the $7/5$ cascade output.

### Cascade entry points

```bash
# Multi-level cascade
python -m cloninger-steinerberger.cpu.run_cascade \
    --n_half 2 --m 20 --c_target 1.4 --max_levels 6

# Coarse variants used during the audit
python cloninger-steinerberger/cpu/run_cascade_coarse_v3.py
```

## Lasserre track (`lasserre/`, `proof/lasserre-proof/`)

The Lasserre track discretizes to the polynomial program

$$\mathrm{val}(d) = \min_{\mu \in \Delta_d}\;\max_W \mu^\top M_W \mu,$$

shows $\mathrm{val}(d) \le C_{1a}$ for every $d$, and certifies
$\mathrm{val}(16) \ge 1.3$ via the order-3 Lasserre relaxation. At
higher resolutions the moment cone is replaced by its
correlative-sparsity restriction (Waki, Kim, Kojima, Muramatsu, 2006),
which keeps soundness while making the SDP tractable at
$d \in \{32, 64, 128\}$.

### Critical constraints on the high-$d$ Lasserre solver

- **Clique-restricted window PSD only.** Window PSD constraints may be
  added only for windows whose active bins fit inside a single clique;
  the partial-$Q$ PSD constraint is unsound for windows that straddle
  cliques (the deficit matrix is entrywise nonnegative but not PSD).
- **Inequality consistency.** Partial consistency must use the
  inequality $y_\alpha \ge \sum_{i \in S'} y_{\alpha + e_i}$, not
  equality. Forcing equality with missing children would zero out
  unmapped moments, producing $\mathrm{lb} > \mathrm{val}(d)$.
- **Global low-degree moments.** All moments of degree $\le 2k - 1$
  must be tracked globally (not just within cliques) for full
  consistency on the critical degree chain.

### Lasserre entry points

```bash
# High-d sparse solver (correlative sparsity, d = 64–128)
python tests/lasserre_highd.py --d 128 --bw 16

# Full moment solver (d ≤ 32)
python tests/lasserre_scalable.py --d 16 --order 2 --mode cg

# Enhanced sparse solver (d ≤ 64)
python tests/lasserre_enhanced.py --d 32 --order 2 --psd sparse --bw 8

# Library usage
python -c "from lasserre.solvers import solve_highd_sparse; solve_highd_sparse(d=8, bandwidth=6)"
```

## Repository layout

```
compact_sidon/
├── proof/
│   ├── cs-proof/                  # Cascade manuscript
│   ├── lasserre-proof/            # Lasserre manuscript
│   ├── lower_bound_refs.bib       # Cascade bibliography
│   └── *.md                       # Proof-supporting notes
├── lean/
│   ├── Sidon/                     # Cascade Lean development
│   └── lasserre/                  # Lasserre Lean development
├── cloninger-steinerberger/       # Cascade implementation
├── lasserre/                      # Lasserre SDP package
│   ├── core.py
│   ├── precompute.py
│   ├── cliques.py
│   └── solvers.py
├── tests/                         # Sweep scripts, benchmarks, solver runs
└── data/                          # Cascade snapshots, SDP logs, checkpoints
```

## When editing the manuscripts

- Both `.tex` files share the cascade bibliography file
  `proof/lower_bound_refs.bib` (cascade) and `lasserre_refs.bib`
  (Lasserre, lives next to its `.tex`). Symlink the cascade bib into
  `proof/cs-proof/` before running `latexmk` if you have not already.
- Don't introduce Claude/Anthropic attribution in commit messages;
  commit iteratively, one concern per commit.
- The Lean theorem names listed in each manuscript's correspondence
  table must match the names actually defined in `lean/`; verify by
  grepping `lean/Sidon/Proof/*.lean` (cascade) or
  `lean/lasserre/*.lean` (Lasserre) before claiming a name.

## References

- [Cloninger & Steinerberger (2017), arXiv:1403.7988](https://arxiv.org/abs/1403.7988) — prior $C_{1a} \ge 1.2802$ baseline.
- [Matolcsi & Vinuesa (2010), arXiv:0907.1379](https://arxiv.org/abs/0907.1379) — earlier upper bound $1.50992$.
- [Boyer & Li (2025), arXiv:2506.16750](https://arxiv.org/abs/2506.16750) — current best upper bound $1.50988$.
- [Waki, Kim, Kojima, Muramatsu (2006)](https://doi.org/10.1007/s10957-006-9030-5) — correlative sparsity for SDP.
- [Tao's optimization-constants catalog](https://teorth.github.io/optimizationproblems/constants/1a.html) — live bracket tracker.
