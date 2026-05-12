# Figure notes for `lower_bound_proof.tex`

Source: `generate_figures.py` in this directory. Run from the repo
root with `python proof/cs-proof/figures/generate_figures.py`; the
script regenerates both the PDF and PNG variants of each figure. The
PDFs are what the manuscript includes; the PNGs are convenient
previews for review tools.

## Figures

### `01_partition_bins.pdf` — Bin partition and induced averages

Shows a nonnegative profile supported on $[-\tfrac14, \tfrac14]$
together with the half-open partition into equal bins. The upper
panel illustrates the support decomposition; the lower panel records
the induced averages $a_i = 4n\,\mu_i$ and the simplex identity
$\sum_i a_i = 4n$ after normalizing $\int f = 1$.

Manuscript placement: end of §2.2 (*Bin Masses and Bin Averages*).

### `02_simplex_lattice.pdf` — Lattice approximation of the simplex

A three-coordinate slice of the normalized simplex. The continuous
point $\mu$ is replaced by a nearby lattice point $b \in B_{n,m}$,
chosen in the formal proof by the cumulative-floor map described in
the text.

Manuscript placement: §3.1, immediately before the canonical
discretization definition.

### `03_reversal_canonical.pdf` — Reversal symmetry and canonicalization

Reversal symmetry of the test value and the lexicographic
canonicalization rule $c \mapsto \min(c, \mathrm{rev}(c))$.
Palindromic compositions are fixed by the involution; non-palindromic
pairs contribute one canonical representative each to the search.

Manuscript placement: §4.3 (*Reversal Symmetry*), after the corollary.

### `04_cascade_flow.pdf` — Multiscale cascade overview

Stylized overview of the multiscale cascade. The annotations record
the exact survivor counts from the terminal-certificate run: $345$ at
$d = 4$, then $48{,}443$, $7{,}499{,}382$, $147{,}279{,}894$,
$76{,}829$, and finally $0$ at $d = 128$.

Manuscript placement: start of §6 (*The Multiscale Cascade*).

## Regenerating

```bash
python proof/cs-proof/figures/generate_figures.py
```

The script produces both `.pdf` (LaTeX-included) and `.png` (preview)
variants for each of the four figures. Captions live in the
manuscript, not here — modifying a caption requires editing the
`\caption{...}` macro in `lower_bound_proof.tex`.
