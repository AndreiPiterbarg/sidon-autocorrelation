# Cascade bibliography audit

Status as of 2026-05-12, after the post-rewrite QA pass.

## Verified entries

All entries in `proof/lower_bound_refs.bib` resolve to canonical
sources via Crossref or arXiv, and the cite sites in
`proof/cs-proof/lower_bound_proof.tex` are supported by the cited
work:

| Bib key | Reference | DOI / arXiv |
| --- | --- | --- |
| `CS17` | Cloninger & Steinerberger, *Suprema of autoconvolutions*, Proc. AMS 145 (2017) | 10.1090/proc/13690 |
| `MV10` | Matolcsi & Vinuesa, *Improved bounds on the supremum of autoconvolutions*, JMAA 372 (2010) | 10.1016/j.jmaa.2010.07.030 |
| `ET41` | Erdős & Turán, *On a problem of Sidon*, J. London Math. Soc. (1941) | 10.1112/jlms/s1-16.4.212 |
| `MO2007` | Martin & O'Bryant, *Symmetric subset problem in continuous Ramsey theory*, Exp. Math. 16(2) (2007) | 10.1080/10586458.2007.10128993 |
| `MO2006` | Martin & O'Bryant, *Constructions of generalized Sidon sets*, JCTA 113 (2006) | 10.1016/j.jcta.2005.04.011 |
| `MO2009` | Martin & O'Bryant, *Supremum of autoconvolutions*, Illinois J. Math. 53 (2009) | 10.1215/ijm/1264170847 |
| `CRV10` | Cilleruelo, Ruzsa & Vinuesa, *Generalized Sidon sets*, Adv. Math. 225 (2010) | 10.1016/j.aim.2010.05.010 |
| `CillerueloVinuesa2008` | Cilleruelo & Vinuesa, *B₂[g] sets and a conjecture of Schinzel and Schmidt*, CPC 17 (2008) | 10.1017/S0963548308009450 |
| `SS2002` | Schinzel & Schmidt, *L¹ vs L^∞ norms of squares of polynomials*, Acta Arith. 104 (2002) | 10.4064/aa104-3-4 |
| `Bomze2014` | Bomze, Gollowitzer & Yildirim, *Rounding on the standard simplex*, JOGO 59 (2014) | 10.1007/s10898-013-0126-2 |
| `White22` | White, *A new bound for Erdős' minimum overlap problem*, Acta Arith. 208 (2023) | 10.4064/aa220728-7-6 |
| `BL26` | Boyer & Li, *Improved example for an autoconvolution inequality*, Exp. Math. (online Feb 2026) | 10.1080/10586458.2025.2607423 |
| `Lean4` | de Moura & Ullrich, CADE 28 (2021) | 10.1007/978-3-030-79876-5_37 |
| `Mathlib` | The mathlib Community, CPP 2020 | 10.1145/3372885.3373824 |
| `Tao` | Tao, optimization-constants catalog | live URL |
| `Hales2017` | Hales et al., *A formal proof of the Kepler conjecture*, Forum Math. Pi 5 (2017) | 10.1017/fmp.2017.1 |
| `PlattTrudgian2021` | Platt & Trudgian, *Riemann hypothesis up to 3·10¹²*, BLMS 53 (2021) | 10.1112/blms.12460 |

## Key renames applied during QA

- `MO2004` → `MO2007`: the original key reflected the arXiv submission
  year, but the publication is Experimental Mathematics 16(2), 2007.
- `BL25` → `BL26`: matches the issued date (Feb 2026) and the
  `amsalpha` label.

## Build-fix applied

`White22` title needed `{Erd\H{o}s}` instead of `Erd\H{o}s` — without
the brace, `amsalpha` lowercased the title and produced `erd\h{o}s`
in the bbl, which is an undefined control sequence and crashed
pdflatex.

## Citation-precision adjustments in the manuscript

- The Tao catalog is described as "informal" so readers do not mistake
  it for a peer-reviewed source.
- The White cite is contextualized as a related extremal problem
  (Erdős' minimum overlap) rather than overstated as "related
  lower-bound techniques".
- `CillerueloVinuesa2008` is now cited alongside the other
  generalized-Sidon references in §1.1 (it was previously unused).
- The known-bounds table includes both Martin–O'Bryant entries and
  the Boyer–Li improved upper bound, mirroring the companion Lasserre
  paper's table.

## Open items

- `BL26` has no print pages yet (online-first); the bibtex
  `missing pages` warning is harmless and will resolve once the print
  issue appears.
- The Tao catalog access date (`2026-05-12`) should be refreshed
  before any final submission.
