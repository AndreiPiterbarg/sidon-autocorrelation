# Lasserre / SDP / SOS attempts

Consolidated record of the Lasserre / semidefinite / sum-of-squares family of
attacks on the Sidon autocorrelation constant
$C_{1a} := \inf\{\|f*f\|_\infty : f \ge 0,\ \mathrm{supp}\,f \subseteq [-1/4,1/4],\ \int f = 1\}$.
All routes below are **dead** with respect to the headline bound
$C_{1a} \ge 1.292$ (3-scale arcsine, see [`multiscale_arcsine.md`](multiscale_arcsine.md)).
A self-contained LaTeX writeup for the global Farkas-Lasserre track has been
archived at [`lasserre_writeup/`](lasserre_writeup/lasserre_lower_bound.pdf).

## 1. Farkas-certified Lasserre cascade (`val(d)` track)

Discrete relaxation: partition $[-1/4,1/4]$ into $d = 2n$ equal bins, let
$\mu \in \Delta_d$ be the bin-mass vector, and define the windowed test value
$\mu^\top M_W \mu$ with $(M_W)_{ij} = (2d/\ell)\mathbf 1[s_{\rm lo} \le i+j \le s_{\rm lo}+\ell-2]$.
Then $\mathrm{val}(d) := \min_{\mu} \max_W \mu^\top M_W \mu$ is a rigorous lower
bound on $C_{1a}$ for every $d \ge 2$ (no discretisation correction; see
`extrapolation.md`). The Farkas pipeline solves the order-$k$ Lasserre
relaxation $\mathrm{val}^{(k,b)}(d) \le \mathrm{val}(d)$, rounds MOSEK duals to
$\mathbb Q$, and verifies a rational Farkas witness at `mpmath dps=80`.
Rigorous result obtained: $\mathrm{val}(4) > 1.0963$ (see MEMORY
`project_farkas_certified_lasserre.md`); validated locally at $d \in \{6,8,16\}$.

Float multistart estimates of $\mathrm{val}(d)$:

| $d$ | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|
| $\mathrm{val}(d)$ | 1.102 | 1.171 | 1.205 | 1.241 | 1.271 | 1.284 | **1.319** |

So $\mathrm{val}(d)$ crosses 1.30 between $d=14$ and $d=16$ *numerically*. A
rigorous certificate at $d \ge 16$ was never produced: at $(k,b)=(2,16)$ the
relaxation is too loose ($\mathrm{val}^{(2,16)}(16) \le 1.25$), and dense
order-$3$ at $d \ge 16$ blows memory (P4-48t OOMs at $d=12$ on a 755 GiB pod,
see MEMORY `project_p4_48t_oom_d12.md`). **Tier-1 cascade speedups**
(direct MOSEK 20.7$\times$, persistent pool 5.14$\times$, fused F+FN 1.86$\times$;
see MEMORY `project_cascade_tier1_speedups.md`) made $d=10$ practical but did
not unlock higher $d$.

## 2. Three-point SDP pilot

Continuous-moment Lasserre with a 3D moment block (Christoffel-Darboux /
$\epsilon$-shifted bump objective, $S_3 \times \mathbb Z/2$ orbit reduction).
Two phases were run.

- **V1 prototype**, bump kernel: lift $\Delta := \lambda^{3{\rm pt}} - \lambda^{2{\rm pt}}$
  rose from $10^{-6}$ at $(k,N)=(5,5)$ to $3.2 \times 10^{-4}$ at $(7,7)$; MOSEK
  conditioning collapsed at $(8,8)$. **Below the Phase 2 kill threshold
  $\Delta_{8,40} < 10^{-3}$.**
- **V2 full design** with Christoffel-Darboux objective: $\lambda^{2{\rm pt}} = \lambda^{3{\rm pt}} = 1$
  *exactly* at every $(k,N)$ — the kernel-mean trivial value
  $\int_{-1/2}^{1/2}(f*f)\,dt = 1$. The 2-point pseudo-moments 3-extend, so
  no 3D constraint is violated. **$\Delta = 0$ by construction**, structurally
  stronger than V1's "small empirically".

The only positive 3-point empirical lift requires an explicit $\|f\|_\infty \le M$
constraint (V3 alternatives report, $\lambda^{3{\rm pt}} = 1.357$ at $(k,N)=(8,8)$,
$M=2.10$). This is **conditional**; bridging it to unconditional $C_{1a}$
would require an a-priori $\|f^*\|_\infty$ upper bound, which Track 1 ruled out
(see §4). See MEMORY `project_threepoint_sdp_dead.md`.

## 3. Reverse-Young / $L^{3/2}$ attempt (Path A bridge)

Identified the conjecture
$$\sup_{|t|\le 1/2}(f*f)(t) \;\ge\; \frac{\pi}{8}\,\|f\|_{3/2}^3 \qquad (\text{REV-3/2})$$
with equality at the Schinzel-Schmidt extremizer $f_0(x) = (2x+1/2)^{-1/2}$ on
$[-1/4,1/4]$. Verified numerically across 3592 candidates with no
counterexample; $(f_0 * f_0)(t) = \pi/2$ proven on the half-interval
$[-1/2, 0]$ via Beta-integral substitution. The constant $\pi/8$ traces to
$B(1/2, 1/2)$ and cannot be obtained from sharp Young (Beckner constant
$A_{3/2}^2 A_3 = 0.9532$ gives the wrong direction) or from classical Brascamp-Lieb
reverse Young (which requires $p,q < 1$, not $3/2$). The SDP side is
**structurally blocked**: V3+auxiliary $g = f^{3/2}$ moments admit singular
pseudo-moments (atomic $g$ with $\int g = 0.16$, $\int g^2 = 33.5$); discrete
Shor / level-2 Lasserre cannot break the flat-average floor $\lambda \to 1$.
Posterior <5%. See MEMORY `project_path_a_l32_attempt.md`.

## 4. Track 1 — $\|f^*\|_\infty$ a-priori bound (settled negative)

V3's L-infinity-augmented SDP gives $\lambda > 1.2802$ for $\|f\|_\infty \le 2.18$
at $k=7$, but this only bounds $C_{1a}^{(M)}$, not $C_{1a}$. Three independent
lines of evidence settle the bridge in the negative:

1. The Cauchy-Schwarz floor $\|f^*\|_\infty \ge \|f\|_2^2 \ge 2$ is matched by
   every shrink-and-mollify construction (best achievable
   $\|f'\|_\infty \approx 4\sqrt 2$ at sup-blowup factor 2).
2. The $\mathrm{val}(d)$ optima have $2d\cdot\max_i \mu_i$ growing
   $2.88, 3.83, 4.22, 4.68, 5.21$ for $d \in \{4,6,8,10,12\}$ — diverges, no
   saturation.
3. MV's near-optimizer $\sim 1.39/(0.002-2x)^{1/3}$ and White's
   $\sim 1/\sqrt{1/2-|x|}$ are integrably singular at the endpoint; no paper
   proves any a-priori UB on $\|f^*\|_\infty$.

Conclusion: $C_{1a}^{(M)} > C_{1a}$ strictly for finite $M$, with gap not
closing at $M = 2.15$.

## 5. `val(d)` numerical sweep + extrapolation

Numerical $\mathrm{val}(d) \le C_{1a}$ holds for every $d \ge 2$ with
**no $\varepsilon(d)$ correction** (proof in `extrapolation.md`: discretise
any feasible $f$ to its bin-mass vector; the windowed test-value inequality
$\|f*f\|_\infty \ge \mu^\top M_W \mu$ holds for every window, so
$\|f*f\|_\infty \ge \max_W \mu^\top M_W \mu \ge \mathrm{val}(d)$). The empirical
crossing of 1.30 at $d=16$ motivated the $d=64$ / $d=128$ scale-up plan
(`d64_d128_plan.md`): banded clique decomposition with bandwidth $b=16$,
order-2 Lasserre, 48 moment cones at $d=64$. Estimated wall: 30–90 min on
P2-64t (256–512 GB). Never executed before the user constraint closed the
route.

## 6. Track 1 $f$-inf trace

Exhaustive 8-angle search for any constraint formulation that converts the
3-point Lasserre tightening into a rigorous $C_{1a} > 1.2802$ proof at
reachable $k$. **All failed.** Identified: Cauchy-Schwarz lower bound
$\|f^*\|_\infty \ge 2$ (rigorous); no truncation/mollification reaches
$\|f'\|_\infty \le 2.15$ without blowing sup; numerical $\|f^{(d)}\|_\infty \to \infty$
with $d$. Only remaining angle inside this track: $\ge 4$-point Lasserre with
$L^p$ ($p \in (1,3)$) + heavy infra (Chebyshev basis, full block-diag, MOSEK
Task API). Estimated 4-6 weeks; 5-10% probability.

## 7. $d=64$ / $d=128$ plan

Sparse-clique Lasserre with bandwidth $b=16$, order $k=2$:

| $d$ | $n_{\rm cliques}$ | clique block | wide windows | est. wall |
|---|---|---|---|---|
| 32 | 16 | 171 | 1 710 | 5 min |
| 64 | 48 | 171 | 7 822 | 30-90 min |
| 128 | 112 | 171 | 32 334 | 4-12 h |

Acceptance criterion: rational Farkas witness at `mpmath dps=80` with
`lb_rig >= 1281/1000`. Cushion at $d=64$: float estimate
$\mathrm{val}(64) \approx 1.384$ vs target 1.281 ($+0.103$). Relaxation
tightness required $\approx 92.6\%$; empirically $\mathrm{val}^{(3)}(16) = 98.5\%$
of $\mathrm{val}(16)$.

## Verdict

The Lasserre / SDP / SOS route is **formally closed** by user constraint
(CLAUDE.md: "no further B&B / Lasserre / large SDP"). The infrastructure
works: Tier-1 speedups
brought $d=10$ Farkas certification under wall budget, the
$\mathrm{val}(d) \le C_{1a}$ extrapolation needs no $\varepsilon(d)$
correction, and the LaTeX writeup
[`lasserre_writeup/lasserre_lower_bound.pdf`](lasserre_writeup/lasserre_lower_bound.pdf)
is publication-quality through Theorem 1 (main result conditional on
the unproduced $d \ge 16$ certificate). The bound stalled because (i) memory
scaling at $d \ge 12$ exceeds <1 TB pods at order $\ge 3$, (ii) the 3-point
lift is $\Delta = 0$ structurally on the natural Christoffel-Darboux objective,
and (iii) the $L^{3/2}$ bridge requires both an unproved Hardy-type
inequality and a non-existent SDP encoding. Surviving lemmas (the
$\mathrm{val}(d) \le C_{1a}$ extrapolation, the $\pi/8$ conjecture proof
roadmap, the cubature-based correction lemmas) are referenced obliquely from
[`multiscale_arcsine.md`](multiscale_arcsine.md), but no publishable
$C_{1a}$ improvement was obtained from this family of attacks.

The archived companion paper at
[`lasserre_writeup/`](lasserre_writeup/lasserre_lower_bound.tex) is
**superseded** by the root-level `lower_bound_proof.tex` (multi-scale
arcsine, $C_{1a} \ge 1.292$). It is preserved for provenance and for the
Farkas certification infrastructure, which remains the only working route
to rigorous lower bounds *inside* the Lasserre family.

## Cross-references

Related dead-route writeups:
[`path_a_holder.md`](path_a_holder.md) (Hölder family, Hyp_R unconditional),
[`path_b_kbk.md`](path_b_kbk.md) (KBK / dual-positive direction),
[`master_attacks.md`](master_attacks.md) (28-attack table, includes the
SDP/LP entries above as separate rows),
[`coarse_lp_bnb.md`](coarse_lp_bnb.md) (LP/BnB on the same `val(d)` discretisation),
[`proof_framework.md`](proof_framework.md) (CS17 cascade skeleton; this is
the discrete-side counterpart to `val(d)`).
