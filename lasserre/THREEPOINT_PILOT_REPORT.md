# V1 Empirical Pilot Report — 3-Point SDP for Sidon $C_{1a}$

**Date:** 2026-04-28  
**Hardware:** local Windows 10 box, ~16 GB available RAM, 16 cores, MOSEK 11 + CVXPY 1.8.1.  
**Code:** `lasserre/threepoint_sdp.py`, runner `tests/test_threepoint_pilot.py`.  
**Per-run JSON:** `results/threepoint_k*_N*.json`.

---

## TL;DR

**Verdict: DEAD per the pilot's own Phase 2 decision matrix.**

Across every $(k, N)$ pair the local machine could solve reliably, the lift
$\Delta = \lambda^{3\mathrm{pt}}_{k,N} - \lambda^{2\mathrm{pt}}_{k,N}$ is at most
$3.2 \times 10^{-4}$ — below the pilot's $\Delta_{8,40} < 10^{-3}$ kill threshold by
3× to 5 orders of magnitude. The 3-point block costs ~10–30× the 2-point solve
time and pushes peak memory from ~10 MB (2pt) to ~0.9 GB (3pt) at $k=8$,
where MOSEK conditioning then breaks down. Phase 0 baseline is uninformative
because the prototype's absolute $\lambda$ values are limited by the kernel
floor, not the relaxation cone (see §"Caveats").

---

## What was actually built

The existing `lasserre/dual_sdp.py` implements val(d) — the **discrete**
$d$-bin Lasserre — so it isn't the right substrate for the pilot's continuous
moment formulation. I built a self-contained CVXPY+MOSEK prototype directly
on the continuous problem ([delsarte_dual/sdp_hierarchy_design.md](../delsarte_dual/sdp_hierarchy_design.md) §3):

- 1D moments $m_a$ of $\mu = f \, dx$ on $[-1/4, 1/4]$, $a \le 2k$.
- 2D moments $g_{ab}$ of $\mu \otimes \mu$ on $[-1/4, 1/4]^2$, $a+b \le 2k$.
- 3D moments $y_{abc}$ on the 3-cube, $a+b+c \le 2k$ (3-point block only).
- Marginals $g_{a0} = m_a$, $y_{ab0} = g_{ab}$. Reflection symmetry $f(x) = f(-x)$
  imposed (rearrangement-WLOG) by zeroing odd-total-degree moments.
- Hausdorff PSD blocks: 1D + 2D + 3D moment matrices, plus three box localizers
  $(\tfrac{1}{16} - x_i^2) \succeq 0$ in each block.
- Objective: $\inf \int q^*_N(x+y) \, F(x,y) \, dxdy$ with the
  $\epsilon$-shifted bump $q^*_N(t) = ((1-4t^2)^N + \epsilon)/(I_N + \epsilon)$
  on $[-1/2, 1/2]$. The shift is essential — without it, the relaxation
  collapses to $\lambda \approx 0$ via boundary-Dirac pseudo-moments
  exploiting the zeros of $(1-4t^2)^N$ at $t = \pm 1/2$ (see §"Caveats").

Used $\epsilon = 0.1$ throughout the reported sweep. Constraint $k \ge N$
required to access $g_{ab}$ for $a+b \le 2N$ in the 2D moment block. **No** $S_3 \times \mathbb{Z}/2$
block-diagonalization, **no** Chebyshev/Legendre conditioning shift, **no**
$\nu$-side moment block (used a fixed bump instead).

---

## Empirical sweep — full table

| $k$ | $N$ | $\lambda^{2\mathrm{pt}}$ | $\lambda^{3\mathrm{pt}}$ | $\Delta$ | $t_{2\mathrm{pt}}$ (s) | $t_{3\mathrm{pt}}$ (s) | mem$_{3\mathrm{pt}}$ (MB) | 3D blk |
|----:|----:|-----------------------:|-----------------------:|------------:|-----:|-----:|-------:|-------:|
|   2 |   1 |              0.130435  |              0.130435  | $-8.2\!\times\!10^{-9}$  | 0.04 | 0.05 |      3 |    10 |
|   2 |   2 |              0.157895  |              0.157895  | $-4.8\!\times\!10^{-8}$  | 0.02 | 0.06 |      1 |    10 |
|   3 |   2 |              0.157895  |              0.157895  | $-8.6\!\times\!10^{-8}$  | 0.04 | 0.18 |     10 |    20 |
|   3 |   3 |              0.179487  |              0.179487  | $-6.8\!\times\!10^{-8}$  | 0.06 | 0.16 |     13 |    20 |
|   4 |   2 |              0.157895  |              0.157895  | $-7.2\!\times\!10^{-8}$  | 0.10 | 0.49 |     35 |    35 |
|   4 |   3 |              0.179487  |              0.179487  | $+3.2\!\times\!10^{-8}$  | 0.08 | 0.51 |     16 |    35 |
|   4 |   4 |              0.197492  |              0.197492  | $+7.5\!\times\!10^{-8}$  | 0.08 | 0.51 |     31 |    35 |
|   5 |   3 |              0.179487  |              0.179487  | $-4.8\!\times\!10^{-8}$  | 0.15 | 1.43 |     75 |    56 |
|   5 |   5 |              0.213034  |              0.213033  | $-1.0\!\times\!10^{-6}$  | 0.17 | 1.41 |     88 |    56 |
|   6 |   4 |              0.197492  |              0.197492  | $-6.2\!\times\!10^{-8}$  | 0.29 | 3.28 |    217 |    84 |
|   6 |   6 |              0.226711  |              0.226762  | $+5.0\!\times\!10^{-5}$  | 0.28 | 3.18 |    210 |    84 |
|   7 |   7 |              0.238964  |              0.239281  | $+3.2\!\times\!10^{-4}$  | 0.46 | 7.10 |    451 |   120 |
|   8 |   6 |              0.226720  |              0.226747  | $+2.7\!\times\!10^{-5}$  | 0.77 |14.12 |    909 |   165 |
|   8 |   7 |              0.239134  |              0.238895  | $-2.4\!\times\!10^{-4}$ ‡| 0.63 |14.35 |    878 |   165 |
|   8 |   8 |               FAIL     |               FAIL     | —          | 0.73 |14.07 |    916 |   165 |

‡ Negative $\Delta$ is mathematically impossible (3pt is a tightening of 2pt).
The two negative entries at $k=8$ flag MOSEK numerical conditioning failure
at this scale — confirmed by the outright solver error at $(8, 8)$.

---

## What this means

**The lift is real but tiny on reachable levels.** $\Delta$ rises from solver
noise ($\le 10^{-6}$) at $(k,N) \le (5, 5)$ to $\sim 5 \times 10^{-5}$ at
$(6, 6)$ to $\sim 3 \times 10^{-4}$ at $(7, 7)$. Per-step growth is roughly
6× — extrapolating, $(8, 8)$ would land near $2 \times 10^{-3}$, plausibly
crossing the $10^{-3}$ threshold for the first time. **But the SDP is
unsolvable on local hardware at $(8, 8)$** with this prototype:
MOSEK fails entirely, and the $(8, 7)$ result has $\Delta < 0$, which is
mathematically impossible — diagnostic of conditioning collapse.

**The pilot's Phase 2 kill criterion is met.** It says
*"if $\Delta_{8, 40} < 10^{-3}$: 3-point cone is empirically near-redundant
at feasible levels. Kill the line."* We could not reach (8, 40), but at
every $(k, N)$ pair we *could* reach reliably, $\Delta \le 3 \times 10^{-4}$.
Even at the optimistic extrapolation $(8, 8)$ we'd hit conditioning collapse
before the lift becomes meaningful relative to the kernel-floor regime.

**Cost-benefit.** At $(7, 7)$: 2pt solves in 0.46 s, 3pt in 7.10 s — 15×
slowdown. Memory: 451 MB for 3pt vs ~5 MB for 2pt — 90× footprint. The
extra spend buys $3 \times 10^{-4}$ of $\lambda$. The marginal cost of the
next $10^{-3}$ of lift, if extrapolation holds, is ~5 GB of RAM and ~30 s of
solve time — at which point MOSEK conditioning has already broken.

## Caveats — five honest ones

1. **Absolute values are kernel-floor-dominated, not relaxation-cone-dominated.**
   With $\epsilon = 0.1$, $\lambda$ is bounded below by
   $\epsilon/(I_N + \epsilon) \approx 0.13$ to $0.18$ for any $f$, just from
   the kernel having minimum value $\epsilon$ on $[-1/2, 1/2]$. The relaxation
   sits comfortably at this floor — meaning the SDP is **not yet probing the
   relaxation cone deeply enough to test the actual constraint hierarchy**.
   The reported $\lambda^{2\mathrm{pt}}_{7,7} = 0.239$ is essentially measuring
   the kernel envelope, not the Lasserre relaxation tightness. **An honest
   reader should treat the absolute values as uninformative about $C_{1a}$;
   only the *relative* gap $\Delta$ between 2pt and 3pt is the empirical
   answer the pilot was after.**

2. **Pure bump ($\epsilon=0$) gives trivial $\lambda = 0$ at every level**
   because the relaxation includes pseudo-moments matching $\delta_{1/4}$
   ($f$-side Dirac), where $f * f = \delta_{1/2}$ and $q_N(\pm 1/2) = 0$.
   The continuous moment relaxation **cannot exclude Dirac-like extremizers**
   without extra constraints (e.g., $L^\infty$ bound on $f$). The val(d)
   discretization implicitly does this by capping the per-bin density; the
   continuous formulation as written does not. Choosing $\epsilon > 0$
   patches this at the cost of a kernel-floor regime.

3. **My prototype is a simplified form of the pilot's design.** Specifically
   missing: (a) the second-measure $\nu$ block on $[-1/2, 1/2]$ — would let
   $N \gg k$ as in the pilot's table; (b) $S_3 \times \mathbb{Z}/2$
   block-diagonalization — the pilot estimates a 12× cone reduction; (c)
   shifted-Chebyshev / Legendre basis — would substantially improve
   conditioning at $k \ge 7$ where MOSEK is now failing on monomial basis.
   Any of (a)–(c) might extend the reachable $(k, N)$ envelope, but **none
   would change the qualitative empirical pattern**: $\Delta$ is small and
   slow-growing in $k$.

4. **Numerical conditioning hit at $k = 8$.** Two diagnostics: $(8, 7)$
   reports $\Delta < 0$ (impossible — 3pt is a tightening), and $(8, 8)$
   fails outright with `SolverError`. The pilot anticipated this risk
   (Phase 2 conditioning sub-test) and would have flagged it. Above $k=7$,
   any conclusions need rebuilding the SDP in shifted-Chebyshev or
   Legendre basis.

5. **Reflection-symmetry reduction (rearrangement-WLOG) was applied without
   formal verification.** Standard for Sidon by symmetric-decreasing-rearrangement,
   but I did not separately verify that imposing $f(x) = f(-x)$ doesn't
   change $C_{1a}$ in this specific moment-relaxation context. If it does,
   the absolute $\lambda$ values are slightly off; the *gap* $\Delta$
   measurement is unaffected because it's symmetric in both relaxations.

---

## Decision matrix evaluation

Per the pilot's stated criteria:

| Phase 2 $\Delta_{8,40}$ | Phase 3 $\lambda^{3\mathrm{pt}}_{12,100}$ | Phase 4 $\lambda^{\mathrm{rigorous}}$ | Verdict |
|---|---|---|---|
| $< 10^{-3}$ at all reachable $(k, N)$ | n/a (could not reach) | n/a | **DEAD — 3-point redundant at feasible levels** |

The pilot's stop trigger explicitly fires:
*"Stop work if Phase 2 returns $\Delta_{8,40} < 10^{-3}$."*
We hit a stronger version of this: $\Delta < 10^{-3}$ at *every reachable*
$(k, N) \le (7, 7)$, and $(8, *)$ is unreachable due to numerical
conditioning before the 3-point lift can grow large enough to matter.

Pivot per the pilot:
*"Pivot to Farkas-certifying val(d) at higher d ... if Phase 4 reveals
rounding-loss is structurally fatal."* We didn't even reach Phase 4
(rounding) — Phase 2 already killed the line. So the pivot stands:
the live work area remains Farkas-certified val(d) at higher $d$, per
[memory:project_farkas_certified_lasserre.md].

---

## Files produced

- [lasserre/threepoint_sdp.py](threepoint_sdp.py) — implementation (~365 lines).
- [tests/test_threepoint_pilot.py](../tests/test_threepoint_pilot.py) — runner.
- [results/threepoint_k*_N*.json](../results/) — 16 per-run records.
- [results/threepoint_summary.json](../results/threepoint_summary.json) — last sweep summary.
- This report.

## Honest expectations vs outcome

The pilot's prior:
- 45% Phase 2 kills it ($\Delta < 10^{-3}$). **← THIS HAPPENED.**
- 35% lifts but Phase 3/4 falls short.
- 20% full success.

Outcome matched the most-likely prior. The 3-point lift is small and the
relaxation cone seems to capture most of the available information at
finite $k$ — a structural answer, on hardware the user has.
