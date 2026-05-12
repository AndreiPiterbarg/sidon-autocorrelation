# `lower_bound_proof_papa.tex` — overview of changes

Source: `lower_bound_proof.tex`. Output: `lower_bound_proof_papa.tex` +
`lower_bound_proof_papa.pdf` (9 pages, `pdflatex`-clean, no overfull
boxes). Driven by `papa_tex_instructions.md`.

## Preamble / look and feel

- Dropped `titling` and `titlesec`; removed every `\titleformat`,
  `\titlespacing`, `\droptitle`, `\pretitle`/`\posttitle`, etc. Sections
  now use the stock `article`-class formatting.
- Switched body + math font from Latin Modern to Times Roman via
  `mathptmx` (with Helvetica via `helvet` for sans), matching a
  standard CS-journal look.
- Added `authblk` for a clean author/affiliation block; all three
  authors share a single UPenn affiliation.
- Replaced the custom `paperplain`/`paperdef` theorem styles with the
  stock `amsthm` `plain` and `definition` styles.
- Added `\sloppy` and `\setlength{\emergencystretch}{3em}` to absorb
  the few long inline formulas that were previously near the margin.

## Title and authorship

- New title: **"An Improved Lower Bound on the Autoconvolution
  Constant"** — concise, descriptive, no surname-bound branding.
- Author line uses `\quad`-separated names (`Authsep`/`Authand`/
  `Authands` overridden), giving the typical journal look:
  *Andrei Piterbarg   Jai Bajaj   Derrick Vincent*.
- Affiliation line: *University of Pennsylvania, Philadelphia, PA, USA*.
- PDF metadata (`pdftitle`, `pdfsubject`) updated accordingly.

## "Piterbarg–Bajaj–Vincent" usage

- Full name appears **exactly once**, in the abstract, where it
  introduces the abbreviation: *"the Piterbarg–Bajaj–Vincent (PBV)
  Bound"*.
- All other occurrences (introduction prose, Table 1 row, Theorem 1.1
  header) now read *PBV Bound*.

## New plain-language overview (§1.3 *Outline of the Approach*)

Replaces the old terse "Strategy and Main Result" subsection. Six
short paragraphs aimed at a technical reader without prior exposure
to the Matolcsi–Vinuesa machinery:

- *The duality framework* — what the auxiliary pair `(K, G)` is and
  why exhibiting one is enough.
- *Why a single kernel runs out of room* — what the previous arcsine
  construction leaves on the table.
- *What changes here* — three-scale mixture, why admissibility
  survives, why the cosine multiplier no longer pays a single-frequency
  penalty.
- *From real numbers to a checkable certificate* — interval arithmetic
  + outward rounding to rationals, giving an integer-arithmetic final
  comparison.
- *Mechanization* — Lean 4 reduction to a single user axiom.
- *Comparison with prior work* — gain ≈ 1/S₁; three-scale mixture
  drops S₁ from ≈ 87.4 to ≤ 29.841.

The Theorem 1.1 statement is preserved unchanged after the outline.

## Inline-fraction polish

Every inline `\tfrac`/`\frac` of the form `a/b` rewritten to the
text-style `a/b`:

- `\tfrac14`, `\tfrac12`, `\tfrac\delta2`, `\tfrac{638}{1000}` → `1/4`,
  `1/2`, `\delta/2`, `638/1000` throughout.
- The `\frac{a_j^2}{\widetilde K_{\rm ms}(j)}` inside the inline
  Lemma 4.3 statement → `a_j^2/\widetilde K_{\rm ms}(j)`.
- Display-math `\frac`s left untouched (they read better stacked).

## Tables

- Table 1 (`tab:published-bounds`): column spec `lcl` → `l l l`, so the
  *Value* column is left-aligned, as requested.
- Caption "short forms" that merely duplicated the long form
  (`\caption[…]{…}`) removed in all three tables for cleanliness.
- No other structural changes to Tables 2 and 3.

## Overfull boxes

- Pre-edit: original source had no overfull boxes in the current
  recompile, but the new font (Times) and tighter margins changed line
  breaks. `\sloppy` + `\emergencystretch` were enough to absorb the
  shifts.
- Post-edit: `pdflatex` reports zero overfull `\hbox`es. One cosmetic
  underfull (`badness 1502`) inside the bibliography, which is
  standard for `thebibliography` and not worth chasing.

## References

All nine bibliography entries spot-checked against the canonical
sources (AMS, IEEE Xplore, Springer, Project Euclid, ScienceDirect,
ACM DL, arXiv, Cambridge). Volumes, issues, pages, years, and arXiv
identifiers match the cited works in every case; no corrections
needed.

## Build

```
pdflatex lower_bound_proof_papa.tex   # twice for cross-refs
```

The build is reproducible with the same MiKTeX/`pdftex` toolchain used
for `lower_bound_proof.tex`; no new package dependencies beyond
`mathptmx`, `helvet`, and `authblk`, all of which ship in standard
MiKTeX / TeX Live distributions.
