# A New Lower Bound on the Sidon Autocorrelation Constant $C_{1a}$

> **Result:** $C_{1a} \ge 7/5 = 1.4$, up from the previous best $1.2802$
> (Cloninger & Steinerberger, 2017).
>
> **Companion result:** $C_{1a} \ge 1.3$ via a Lasserre SDP hierarchy.
>
> **Current bracket:** $1.4 \le C_{1a} \le 1.50988$.

## Problem

For any nonnegative function $f : \mathbb{R} \to \mathbb{R}_{\geq 0}$
supported on $[-1/4, 1/4]$ with $\int f = 1$:

$$\max_{|t| \le 1/2} (f * f)(t) \;\geq\; C_{1a}.$$

The upper bound comes from constructing explicit functions with small
autoconvolution peaks. The lower bound requires proving that *no*
function can do better — a fundamentally harder task.

## Two complementary proofs

This repository contains two independent lower-bound proofs, each with
its own LaTeX manuscript and Lean 4 formalization.

### `proof/cs-proof/` — Multiscale cascade ($C_{1a} \ge 7/5$)

Extends the Cloninger–Steinerberger branch-and-prune framework with

1. a sharper per-window discretization correction,
2. dyadic refinement from $d = 4$ through $d = 128$,
3. fused Gray-code generation and energy-cap pruning,
4. exact integer arithmetic in the threshold comparison.

The cascade reaches zero survivors at $d = 128, m = 20$ after roughly
70 hours on a 128-core machine. The analytic reduction is formalized
in Lean 4 parametrically in the target $c_{\mathrm{target}}$; the
$7/5$ instantiation depends on the cascade certificate at the
matching target.

Manuscript: [`proof/cs-proof/lower_bound_proof.pdf`](proof/cs-proof/lower_bound_proof.pdf).

### `proof/lasserre-proof/` — Lasserre SDP hierarchy ($C_{1a} \ge 1.3$, joint with A.\ Piterbarg)

Discretizes the continuous problem to the polynomial program
$\mathrm{val}(d) = \min_{\mu \in \Delta_d} \max_W \mu^\top M_W \mu$
and certifies $\mathrm{val}(16) \ge 1.3$ by the order-3 Lasserre
relaxation, with correlative-sparsity restrictions for $d \in \{32, 64, 128\}$.
The single computational dependency is a finite SDP whose dual
solution is recorded as the certificate.

Manuscript: [`proof/lasserre-proof/lasserre_lower_bound.pdf`](proof/lasserre-proof/lasserre_lower_bound.pdf).

## Repository layout

```
compact_sidon/
├── proof/
│   ├── cs-proof/                # Cascade manuscript + figures
│   ├── lasserre-proof/          # Lasserre manuscript + figures
│   ├── lower_bound_refs.bib     # Shared bibliography for cs-proof
│   └── *.md                     # Proof-supporting notes
├── lean/                        # Lean 4 formalization (Sidon + lasserre)
├── cloninger-steinerberger/     # Cascade implementation
│   └── cpu/run_cascade*.py      # Multi-level cascade runners
├── lasserre/                    # Lasserre SDP package
│   ├── core.py                  # Monomials, windows, base SDP
│   ├── precompute.py            # Base constraints, window PSD
│   ├── cliques.py               # Banded cliques, correlative sparsity
│   └── solvers.py               # solve_highd_sparse, solve_cg, solve_enhanced
├── tests/                       # Sweep scripts and benchmarks
└── data/                        # Cascade snapshots and SDP run logs
```

## Building the manuscripts

The bibliography lives one directory up from each `.tex`, so compile
from inside the proof directory with `latexmk` (or symlink the bib
file):

```bash
cd proof/cs-proof
ln -sf ../lower_bound_refs.bib .
latexmk -pdf lower_bound_proof.tex

cd ../lasserre-proof
latexmk -pdf lasserre_lower_bound.tex
```

## Running the cascade

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

python -m cloninger-steinerberger.cpu.run_cascade \
    --n_half 2 --m 20 --c_target 1.4 --max_levels 6
```

The cascade produces a JSON survivor log per level in `data/`; the
terminal level should have zero survivors.

## Running the Lasserre solvers

```bash
# High-d sparse Lasserre (correlative sparsity)
python tests/lasserre_highd.py --d 128 --bw 16
python tests/lasserre_highd.py --d 16  --bw 15  # tight, kernel-checkable

# Full moment Lasserre (d ≤ 32)
python tests/lasserre_scalable.py --d 16 --order 2 --mode cg

# Sparse enhanced solver (d ≤ 64)
python tests/lasserre_enhanced.py --d 32 --order 2 --psd sparse --bw 8
```

## Related work

- [Cloninger & Steinerberger (2017), arXiv:1403.7988](https://arxiv.org/abs/1403.7988) — the prior $C_{1a} \ge 1.2802$ baseline that both proofs extend.
- [Matolcsi & Vinuesa (2010), arXiv:0907.1379](https://arxiv.org/abs/0907.1379) — best published upper bound prior to Boyer & Li.
- [Boyer & Li (2025), arXiv:2506.16750](https://arxiv.org/abs/2506.16750) — current best upper bound $C_{1a} \le 1.50988$.
- [Martin & O'Bryant (2007, 2009)](https://personal.math.ubc.ca/~gerg/papers/downloads/SAAANT.pdf) — earlier lower-bound progression.
- [Waki, Kim, Kojima & Muramatsu (2006)](https://doi.org/10.1007/s10957-006-9030-5) — correlative-sparsity SDP used in the Lasserre proof.
- [Tao's optimization-constants catalog](https://teorth.github.io/optimizationproblems/constants/1a.html) — informal live tracker for the bracket.
