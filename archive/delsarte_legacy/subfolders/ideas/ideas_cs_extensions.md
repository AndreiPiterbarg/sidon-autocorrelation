# Extensions to Cloninger–Steinerberger 2017 for C_{1a}

**Date**: 2026-04-20
**Problem**: $C_{1a} = \inf\{\|f*f\|_\infty : f\ge 0,\ \mathrm{supp}\,f\subset[-1/4,1/4],\ \int f=1\}$, currently $1.2802 \le C_{1a} \le 1.5029$.
**Scope**: What additional juice is there in the Cloninger–Steinerberger branch-and-prune (B&P) cascade over discrete step functions?

---

## 0. State of the published literature (CS-style lower bounds)

Systematic web + local-survey check (20 Apr 2026):

- **No peer-reviewed paper since CS 2017 (arXiv:1403.7988) has improved 1.28 on C_{1a}.** The only claim > 1.28 is the unpublished `1.2802` attributed to Xie on Tao's optimizationproblems page — a "Grok chat" with no method disclosed.
- AI-era work (AlphaEvolve arXiv:2511.02864, ThetaEvolve arXiv:2511.23473, TTT-Discover arXiv:2601.16175) has only reduced the **upper** bound (1.5099 → 1.5029). All of it is primal (explicit step-function f). None has been turned on the dual / lower bound side.
- Closely related but different problem: the L²-style autoconvolution ratio inequality (Boyer–Li arXiv:2506.16750, Jaech–Joseph arXiv:2508.02803). Achieves c ≥ 0.941, via **step-function upsampling + gradient ascent**. Method ports to our cascade as a warm-start refinement, but the constant is for a different functional.
- Rechnitzer arXiv:2602.07292 (Feb 2026): rigorous 128-digit enclosure of the L² autoconvolution constant using **high-precision interval arithmetic**. Shows the template for turning numerical SDP/LP output into a certificate. Directly transferable to rigor of our Lasserre bounds, not to CS.
- Madrid–Ramos arXiv:2003.06962, de Dios Pont–Madrid arXiv:2106.13873, Barnard–Steinerberger-extensions arXiv:2001.02326 — all on neighbouring Barnard–Steinerberger-type autocorrelation integrals. None improves 1.28. Transferable idea: Fourier/Hausdorff–Young duality + extremizer existence.
- Gaitan–Madrid arXiv:2512.18188 (Dec 2025): exact constants for k-fold convolution on the discrete hypercube. Provides discrete pointwise lower-bound lemmas usable as seeds/cuts for Lasserre at small d, but explicitly does **not** move 1.28.

In short: the space of published CS-cascade improvements is empty; everything is either primal-upper-bound work or a different functional. The room for original contribution via cascade improvements is real.

---

## 1. Concrete cascade improvements (ranked by bound-lift potential)

All of the following are cascade-internal changes over CS 2017's L∞-norm, step-function discretisation on [-1/4, 1/4].

### 1.1 Theorem-1 pruning (drop the correction term) — **3 % threshold reduction per level, 83 % total work cut**

CS use Lemma 3 (step-function + correction) for pruning; Theorem 1 (exact window-sum bound, no correction) is **strictly tighter**. Current project has this proven and verified (see `CASCADE_UPDATE.md` Idea 1). At m=20, n=16, ell=20, W_int=500: correction δ = 21 280, Th1 threshold = 716 800, so +3 % relative tightness per level ⇒ (0.7)⁵ ≈ 0.17 compound factor over five cascade levels. **Does not by itself raise the proved constant** — it enables reaching larger d within the same compute budget.

### 1.2 Cursor-range tightening via AC-3 interval arithmetic — **10¹¹–10²¹ search-space reduction at d = 16**

For every parent bin i and candidate cursor v, compute a guaranteed worst-case window-sum (self-contribution exact; cross-contribution by interval arithmetic with {lo_j, hi_j}). If v is dead under some window W, strike it. Iterate AC-3 style. Uses the proven identity v² + 2v(2P–v) + (2P–v)² = (2P)². Sound, no approximation. Detail: `CASCADE_UPDATE.md` Idea 2. Enables L3 (d=16) enumeration that is currently infeasible.

### 1.3 Full branch-and-bound at every level (not only L0) — **10⁵–10¹⁵ subtree-pruning factor at d = 16**

Current code B&Bs at L0 only; L1+ uses Gray-code enumeration with sub-tree pruning only at J_MIN = 7. Extending `_l0_bnb_inner` to all levels gives per-node partial-window-sum checks. Since all c_i ≥ 0, partial_conv ≤ full_conv — so any partial excess at depth k prunes the full 1/R^(d-k) subtree. Exact (no approximation).

### 1.4 Whole-parent pre-pruning via 2^d box-corner evaluation — **10–50 % parents skipped at L2+**

Before entering enumeration, evaluate the quadratic window sum at all 2^d_parent cursor-box corners. Empirically min attained at a corner for 64/66 tested windows (d_parent ≤ 16). The one case where the min is interior (ell=2 boundary windows) is covered by adding a clamped-gradient critical-point check. O(2^d · d²) per parent; feasible to d_parent ≤ 16, not beyond. Sound (sum-of-minimums + corner check is a valid lower bound).

### 1.5 Canonical (palindromic) enumeration — **exact 2× speedup**

conv(c) ≡ conv(rev(c)) ⇒ enumerate only lex-canonical children. Eliminates post-hoc dedup. Free 2×, compounds with everything.

### 1.6 Step-function **upsampling warm start** (2025 literature transfer) — enables d = 32, 64

Port Boyer–Li / Jaech–Joseph "coarse-to-fine" idea: start a tight optimum at d = 16, upsample to d = 32 by 2× refinement, seed the cascade L-level corresponding to d = 32 with the neighbourhood of the upsampled optimum. Shrinks the effective enumeration envelope exponentially at higher d. Provably sound iff the upsampled neighbourhood fully contains the level-wise survivor ball.

### 1.7 Local QP exact box certification (already in project's `qp_bound.py`) — replaces box triangle inequality

The box QP (`cloninger-steinerberger/cpu/qp_bound.py`) already computes the exact worst-case TV decrement over a cell Cell = {δ : |δ_i| ≤ h, Σδ_i = 0} via vertex enumeration (d · 2^(d−1) vertices). Replaces the loose `cell_var + quad_corr` bound with the exact quadratic maximum on the box. Feasible d ≤ 16. This gives **rigorous** box certification, removing the one place the cascade currently uses a looser triangle-inequality bound.

### 1.8 SDP-assisted node bounds (hybrid CS + Lasserre) — **speculative, high ceiling**

At each internal B&B node, solve a tiny Lasserre order-1 (or order-2) SDP restricted to the still-free cursor variables; use the SDP lower bound as the prune threshold instead of the sum-of-minimums. This is the standard SDP-branch-and-bound technique from Waki–Kim–Kojima–Muramatsu 2006 (and the 2023 AAAI paper on SDP-relaxation B&B for neural-net verification, ojs.aaai.org/26745). Cost: a few seconds per node × typically 10⁵–10⁸ nodes. Worth it only with warm-started solver reuse. Not yet implemented here; large engineering effort but unlocks deeper d than pure interval arithmetic because SDP dominates IA on non-trivial quadratic forms.

### 1.9 Gradient-descent primal beating (AI transfer) for tighter **target c** — **operational, not rigorous**

Use AlphaEvolve-style or gradient refinement to maintain the best-known primal candidate; the cascade must only certify c below this primal upper bound. As the primal upper bound drops (1.5053 → 1.503 → …), the cascade can target a higher c_target and get speedup indirectly. Does not change the lower-bound algorithm.

---

## 2. Algorithmic complexity

Cascade enumerates at level L integer compositions c ∈ ℤ^{d_L} with Σ c_i = 4nm fixed, d_L = 2^L · d_0.

| item | per-parent cost | work factor | feasibility ceiling |
|------|----------------|-------------|---------------------|
| baseline Gray-code CS 2017 | 160^d | 10^35 at d=16 | d ≤ 8 |
| + 1.1 Th1 threshold | same | × 0.17 | d ≤ 10 |
| + 1.2 AC-3 range tightening | O(d² · R) amortised | (R/5)^d, best (R/20)^d | d ≤ 16 |
| + 1.3 full B&B | O(visited · d²) | × 10⁻⁵–10⁻¹⁵ | d ≤ 20 |
| + 1.4 whole-parent corner | O(2^d · d²) per parent | × 0.5–0.9 | d_parent ≤ 16 |
| + 1.5 canonical | ÷ 2 exact | × 0.5 | any d |
| + 1.6 upsampling warm start | O(neighbourhood) | exponential at higher d | d ≤ 32 realistic, 64 optimistic |
| + 1.7 QP box certification | O(d · 2^(d−1)) per cell | constant, replaces triangle | d ≤ 16 |
| + 1.8 SDP node bounds | O(SDP(m,k)) ≈ seconds | prune factor 10–100× extra | d ≤ 20–32 |

Memory: dominated by L-level survivor sets; range-tightening keeps survivor growth at most polynomial per level under generic Sidon-like c_targets.

Compute barrier: d = 16 is the first $d$ where val(d) ≈ 1.3185 > 1.2802 comfortably (see `PROBLEM_STATE.md`). Ideas 1.1–1.5 make d = 16 reachable on a workstation; d = 32 requires 1.6 + 1.8 and a pod. Beyond d = 32 the combinatorial growth is fatal regardless.

---

## 3. Current project state (local file reading)

From `CLAUDE.md`, `PROBLEM_STATE.md`, `CASCADE_UPDATE.md`, `MEMORY.md`, `cloninger-steinerberger/cpu/*`:

- **Primary focus has shifted to the Lasserre SDP hierarchy** (`lasserre/`, `certified_lasserre/`), not the CS cascade. Rationale: cascade is infeasible at L3+ (d ≥ 16) due to 10³⁵ children-per-parent.
- **Farkas-certified Lasserre pipeline is working** (2026-04-19 memory note): val(4) > 1.0963 rigorously certified. Still ≪ 1.28. Path-to-1.28 requires d ≥ 14 (where val(d) numerically exceeds 1.284).
- **Cascade improvements 1.1–1.5 above are all proven and documented in `CASCADE_UPDATE.md`** but several are **not implemented** in the running code. `qp_bound.py` (item 1.7) is implemented. `_l0_bnb_inner` exists; full-level B&B (1.3) does not.
- **Fine-grid soundness fix was applied 2026-04-07**: S = m → S = 4nm (memory note `project_fine_grid_fix.md`). Prior pruning was unsound; this is now corrected.
- **Flat threshold + Lean fine-grid alignment (2026-04-07)**: `--use_flat_threshold` flag and matching S = 4nm in Lean proof artefacts.
- **Interval B&B rigor-parity is a net-negative under the current cuts** (Phase B T1/T2/T3). Memory note: unlocking needs either a dual certificate or an exact rational LP.
- The Lasserre d = 16 O3 gap-closure is unmeasured at every bandwidth. O3 sparse at d = 8 already runs in ~18 s/solve; the missing data point is the full O3/sparse-bw comparison at d ≥ 14.
- `delsarte_dual/` is a separate MV/Cauchy–Schwarz dual line (see `IMPROVEMENT_LIST.md`): B1 + B2 combo is projected to reach 1.28–1.30 analytically, independent of the cascade.

**Net**: Cascade is in maintenance mode; the active fronts are (a) Lasserre SDP with Farkas certification, and (b) MV/arcsine dual optimisation.

---

## 4. Realistic lower bound reachable by an enhanced cascade

Assume every improvement 1.1–1.7 is implemented, with serious compute (one workstation or modest pod). Grounding numbers from `PROBLEM_STATE.md`:

| d   | val(d) numerical | implied ceiling on proved LB (cascade-only) |
|-----|------------------|---------------------------------------------|
| 8   | 1.2046           | < 1.21 (below 1.28, nothing useful) |
| 14  | 1.2840           | ≈ 1.28–1.284 (marginal, needs exact certification) |
| 16  | 1.3185           | **≈ 1.29–1.31**, contingent on rigour stack |
| 32  | 1.336            | ≈ 1.30–1.33, only with 1.6 upsampling + 1.8 SDP nodes |
| 64  | 1.384            | **out of reach** for pure cascade |

**Concrete realistic target**: with items 1.1 + 1.2 + 1.3 + 1.5 + 1.7 implemented and certified rigorously (QP box-certification replacing triangle inequality, fine-grid soundness already fixed), the cascade can plausibly certify **C_{1a} ≥ 1.29 at d = 16** in 1–2 weeks of wall-clock compute. Reaching 1.30+ requires the SDP-assisted node bounds (1.8) which is a larger engineering effort (≥ 1 month). Beyond 1.33 is not accessible without going to d ≥ 32 which is unlikely to close — Lasserre with high-bandwidth sparsity is the correct tool there.

**Pessimistic path** (only 1.1+1.5+1.7 without AC-3 range-tightening): ceiling is d = 10–12, LB ≈ 1.27, **still below** 1.28. Cascade without 1.2 does not clear the bar.

---

## 5. 100-word assessment

CS-cascade has room for a modest but real improvement beyond 1.28. No rigorous paper since 2017 has beaten it, so even 1.29 would be a first. The levers are: (i) Theorem-1 pruning and full-level B&B for tractability at d = 16, (ii) AC-3 cursor-range tightening to collapse 10³⁵ → ~10¹⁴ children, (iii) exact box-QP certification to eliminate triangle-inequality slack, (iv) step-function upsampling as warm-start transfer from 2025 literature. Realistic ceiling: 1.29–1.31 at d = 16 in 1–2 weeks; 1.33 only with SDP-assisted branching. For anything above 1.33, switch to Lasserre, not cascade.

---

## Sources

- [Cloninger–Steinerberger 2017 (arXiv:1403.7988)](https://arxiv.org/abs/1403.7988)
- [Matolcsi–Vinuesa 2010 (arXiv:0907.1379)](https://arxiv.org/abs/0907.1379)
- [Madrid–Ramos, CPAA 2020 (arXiv:2003.06962)](https://arxiv.org/abs/2003.06962)
- [Barnard–Steinerberger extension (arXiv:2001.02326)](https://arxiv.org/abs/2001.02326)
- [de Dios Pont–Madrid (arXiv:2106.13873)](https://arxiv.org/abs/2106.13873)
- [White, optimal L² autoconvolution (arXiv:2210.16437)](https://arxiv.org/abs/2210.16437)
- [Boyer–Li 2025 (arXiv:2506.16750)](https://arxiv.org/abs/2506.16750)
- [Jaech–Joseph 2025 (arXiv:2508.02803)](https://arxiv.org/abs/2508.02803)
- [AlphaEvolve / Georgiev–Gomez-Serrano–Tao–Wagner 2025 (arXiv:2511.02864)](https://arxiv.org/abs/2511.02864)
- [ThetaEvolve / Wang et al. 2025 (arXiv:2511.23473)](https://arxiv.org/abs/2511.23473)
- [Gaitan–Madrid 2025 (arXiv:2512.18188)](https://arxiv.org/abs/2512.18188)
- [TTT-Discover / Yuksekgonul et al. 2026 (arXiv:2601.16175)](https://arxiv.org/abs/2601.16175)
- [Rechnitzer 2026 — 128-digit rigorous enclosure (arXiv:2602.07292)](https://arxiv.org/abs/2602.07292)
- [Waki–Kim–Kojima–Muramatsu, correlative sparsity 2006](https://doi.org/10.1007/s10957-006-9030-5)
- [SDP-relaxation B&B for verification, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26745)
- [Tao, Mathematical exploration and discovery at scale (blog, 2025-11-05)](https://terrytao.wordpress.com/2025/11/05/mathematical-exploration-and-discovery-at-scale/)
- [Tao optimizationproblems page — C_{1a}](https://teorth.github.io/optimizationproblems/constants/1a.html)
