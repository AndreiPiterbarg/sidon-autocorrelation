# Prior Work Pointers: Hölder, L^p, Paley-Wiener, Hausdorff-Young for ||h||_p Bounds

**Deliverable: file:line pointers + one-line annotations for Hölder-inequality and Paley-Wiener work.**

## Direct Hits: Ideas & Frameworks

1. **creative_audit.md:21** – Hölder-inequality proposal replacing Cauchy–Schwarz for moment bounds.
2. **creative_audit.md:80–92** – Explicit Hölder/p-norm variant plan with exponent p; cost ~2–3 days, expected lift 0.001–0.01.
3. **creative_audit.md:101** – Hausdorff PSD + ||f*f||_∞ ≤ M constraint coupling.
4. **mo_framework_detailed.md:113–115** – Hölder step (#4) identified as slack: ||f·f||_2² ≤ ||f·f||_∞ · ||f·f||_1.
5. **mo_framework_detailed.md:120–137** – MO 2004 Conjecture 2.9 (Hölder-slack tightening, π/log16 ≈ 2.266 factor).
6. **mo_framework_detailed.md:143–149** – Hausdorff–Young + Beckner sharpening (MO 2004 §2.7, untested q > 4).
7. **ideas_fourier_ineqs.md:9–12** – Paley-Wiener structure in f̂ (exponential type π/2) + Logvinenko-Sereda bounds.
8. **ideas_fourier_ineqs.md:79–90** – Paley-Wiener entire fn tail bounds; ||ĝ||_L²(R) via prolate spheroidal wave functions.
9. **ideas_lesser_known.md:50–79** – Hausdorff-Young + Hardy-Littlewood route via dual-extremizer measures; Barnard-Steinerberger.
10. **ideas_boyer_li.md:25, 29** – Hölder ratio ||f*f||_L²² / (||f*f||_L∞ · ||f*f||_L¹) ≈ 0.90 via step-function upsampling.

## Framework & Fourier Analysis

11. **ideas_fourier_ineqs.md** (full file) – Siegel Turán duality + Paley-Wiener tailoring; references Reznikov (Paneyah-Logvinenko-Sereda).
12. **family_f3_vaaler.py:56** – Paley-Wiener for dual design (exponential type π/2 matching).
13. **forbidden_region.py:16, 33, 39, 63, 79** – Hausdorff moment PSD, Hausdorff-Young duality, Toeplitz/Hausdorff constraints.

## Existing Grid-Bound Module

**certified_lasserre/ & chebyshev_dual/:** No p-norm or Hölder-specific techniques found; focus on SDP/Lasserre duality, not L^p interpolation.

**grid_bound/ directory:** No prior Hölder or fractional-exponent work; coeffs.py stores exact MV coefficients (fmpq rationals), not p-norm bounds.

---

**Summary:** Hölder slack (Conj. 2.9) is the single biggest untapped gain (~1.367 from 1.276); Hausdorff-Young/Paley-Wiener appear in creative proposals but are untested numerically in the MV framework. Transferable: Logvinenko-Sereda tail bounds, Barnard-Steinerberger dual framework.
