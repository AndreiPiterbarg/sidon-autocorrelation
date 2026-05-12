# Literature notes: Hölder refinement of the MV Cauchy-Schwarz step

*(Agent A deliverable; target <=200 words plus quoted evidence.)*

## Source of the "2.266" constant

**`mo_2004.txt` line 783, Conjecture 2.9** (Martin–O'Bryant 2004):

> "Conjecture 2.9. If f is a pdf supported on [-1/4, 1/4], then
> ||f*f||_2^2 <= (pi / log 16) * ||f*f||_inf * ||f*f||_1 ,
> with equality only if either f(x) or f(-x) equals 2/(4x+1) on |x|<=1/4."

The factor `pi/log 16 ≈ 1.1330`; the **2.266 slack** quoted in the task is
`2 * pi/log 16 ≈ 2.266`, i.e. the square of the sqrt-ratio that appears
inside the Cauchy-Schwarz tail step `sqrt(sum|hat h|^2)*sqrt(sum k_j^2)`
when one uses the conjectured tight `sum|hat h|^2 <= (log 16/pi) * M`
instead of the trivial Parseval `sum |hat h|^2 <= M`.

## Status and (p,q) choice

**`mo_2004.txt` lines 779–780**: *"we have not been able to realize any
success with this idea, although we believe Conjecture 2.9 below."*
Hence Conjecture 2.9 is **OPEN**; MO offer heuristics but no proof.

**`mo_2004.txt` line 474** (Hausdorff–Young): *"||fhat||_q <= ||f||_p
whenever p and q are conjugate exponents with 1 <= p <= 2 <= q <= inf."*

**`mo_2004.txt` lines 753–758** (Section 2.7 item 3): *"The application
of Parseval's identity can be replaced with the Hausdorff–Young
inequality … Numerically, (p,q) = (4/3, 4) appear to be optimal."*
So MO do pick a specific pair `(p*, q*) = (4/3, 4)` (Hausdorff–Young
exponent `p=4/3`, Fourier exponent `q=4`); they do NOT optimise a sup.

**`mo_2004.txt` lines 760–769**: Beckner's sharpening
`C(q) = (2/q)^{q/2-1}(1-2/q)^{q/2-1}` could allow `q>4`, never
numerically exploited.

## Cauchy-Schwarz line replaced

In `multi_moment_derivation.md` §3 **line 149** the step
`|sum_{|j|>N} hat h(j) k_j| <= (sum|hat h|^2)^{1/2} (sum k_j^2)^{1/2}`
is the single C-S line a Hölder(p,q) step (with `1/p + 1/q = 1`)
replaces. Upstream, MV's Lemma 3.3 in `mv_construction_detailed.md`
line 96 is the corresponding bound on `int (f*f) K`.

## Negative findings

- No `Lemma 3.3` in `mo_2004.txt` (that label is MV's, not MO's).
- "Lukacs" does not appear in `mo_2004.txt`.
- `Hölder` (accented) does not appear; only "Holder" (OCR-stripped).
