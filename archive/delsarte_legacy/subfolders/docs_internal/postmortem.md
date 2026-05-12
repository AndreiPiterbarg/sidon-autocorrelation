# Postmortem — Delsarte-type Dual Attempt on $C_{1a}$

## Outcome

**Did not beat 1.2802.** Best certified lower bound from this run:

| Family | Params | `lb_low` | Width |
|---|---|---|---|
| F1 (Fejér-modulated) | $T{=}1,\ \omega{=}2,\ a_1{=}0.5$ | $0.4645$ | $7.9\text{e-}3$ |
| F2 (Gaussian × poly) | $\alpha{=}4,\ P{=}1$ | $0.6685$ | $2.6\text{e-}2$ |
| F3 (Vaaler) | — | not implemented | — |

The infrastructure works end-to-end; the bounds it produces are rigorous but far below the Matolcsi–Vinuesa record $1.2802$.

## Anatomy of the gap

The bound has the form
$$
L(g)\ =\ \frac{1}{M_g}\int_{-1}^{1}\widehat g(\xi)\,w(\xi)\,d\xi,\qquad M_g=\max_{[-1/2,1/2]} g,
$$
where $w(\xi)=\cos^2(\pi\xi/2)$ on $[-1,1]$ is the **sharp, rigorous, elementary** Paley–Wiener lower bound on $|\widehat f(\xi)|^2$ (derivation in `theory.md` Section 2).

### Two sources of slack

1. **The weight $w$.** An "idealised" dual replacing $w(\xi)$ by a Dirac $\delta_0$ gives the simple ratio $\widehat g(0)/M_g$. For the plain Gaussian with $\alpha{=}4$, this is $\sqrt{\pi/4}\approx 0.886$ vs. our rigorous $0.668$. A ratio-of-numerators $0.886/0.668 \approx 1.32$ — so the cost of rigor here is about 32%.

2. **The test function $g$.** Even the **idealised** ratio $\widehat g(0)/M_g$ for simple $g$ is only $\sim 0.9$, not $1.28$. MV's actual record uses a far more carefully tuned construction.

Budget-wise, to beat $1.2802$ we would need a $g$ achieving an **idealised** ratio of at least $1.2802 / (\text{rigor factor}) \approx 1.7$. That is **above even MV's idealised value**, which is exactly the reason their record stands.

## What was tried

### F1 — Fejér-modulated Selberg
$g(t) = \bigl(\tfrac{\sin(\pi T t)}{\pi t}\bigr)^2 (1+\sum a_k\cos(2\pi k\omega t))$, closed-form $\widehat g$ is a sum of shifted triangles. PD certificate: evaluate $\widehat g$ at all breakpoints. The MV-style reference $T{=}1,\omega{=}2,a_1{=}0.5$ gives $\widehat g(0)=1$, $M_g=1.5$ (the cosine modulation creates a sharp peak at $t=0$ because $\cos(4\pi\cdot 0)=1$), so the **idealised** ratio is $1/1.5 = 0.667$; rigorously this drops to $0.464$. Longer optimisation runs (not completed in-session due to the F1 interval B&B being expensive — 13k splits for one evaluation) would explore the $(T,\omega,a_1,a_2)$ space. Float-64 pre-screens suggest the sup over F1 is close to $0.7$.

### F2 — Gaussian × even polynomial, $\deg P\le 2$
Plain Gaussian $\alpha{=}4$ already gives $M_g=1$ trivially ($g(t)\le 1$). Closed-form PD test on $Q(\xi^2)$ for $\deg\le 2$. Rigorous bound $0.668$, best in this run. Small-N polynomial modulation searches did not improve this meaningfully.

### F3 — Vaaler
Stub only. Full implementation requires Taylor-model handling of $\pi\xi\cot(\pi\xi)$ at removable singularities — ~1 week. Not obviously more competitive than F1/F2.

## What was fixed mid-session

- **Weight upgrade.** Initial implementation used the crude
  $w(\xi)=(1-\pi|\xi|/2)^2_+$ which zeroes at $|\xi|=2/\pi\approx 0.637$ and decays quadratically; numerator halved. Replaced by the sharp
  $w(\xi)=\cos^2(\pi\xi/2)\cdot\mathbb{1}_{|\xi|\le 1}$ — same elementary proof, supported on $[-1,1]$, cosine decay. Bound roughly doubled.
- **F1 reference parameters.** Original heuristic `a_1=-0.5` violates PD (the triangle at shift $\pm 2$ becomes $-0.25$). Switched to `a_1=+0.5`.

## Gates

| Gate | Status |
|---|---|
| **G1** Reproduce 1.2802 | **NOT PASSED** (best $0.668$, gap analyzed above) |
| **G2** Unit tests | **PASSED** 10/10 |
| **G3** Each family prints a ball | **PASSED** |
| **G4** max-$g$ rel width $\le 10^{-10}$ | **PARTIAL**: rel-tol $10^{-4}$ achieved in $\le 15k$ splits, the algorithm scales to $10^{-10}$ but wasn't run at that precision in this session. |
| **G5** PD certificate | **PASSED** (closed-form for F1 + F2, documented in `theory.md` Section 4) |

`CLAUDE.md` bound line **NOT** updated (G1 fails).

## What it would take to actually beat 1.2802

**Necessary**: a new analytic construction. The software stack here is the mechanical part; the research part is choosing the right $g$. Specific next steps in order of return-on-effort:

1. **Richer F1 families** ($K=5$ cosines): PD is a polytope constraint on $(a_1,\ldots,a_5)$; solve as LP. Expected float-64 improvement: $0.7 \to 0.9$ idealised.

2. **Replace $w$ by a higher-moment Paley–Wiener bound**: $|\widehat f(\xi)|^2 \ge \cos^2(\pi\xi/2)$ uses only $\int f = 1$, $f\ge 0$, supp $\subset[-1/4,1/4]$. Incorporating moments $\int x f\,dx$ or $\int x^2 f\,dx$ (which are constrained for feasible $f$) tightens $w$, especially near $|\xi|=1$. A Chebyshev-type moment inequality could push the rigor factor from $\sim 0.75$ closer to $1$.

3. **Full F3 (Vaaler)**: not obviously superior, but the only family that explicitly encodes the asymmetry of the support $[-\tfrac14,\tfrac14]$ via shifted extremal majorants.

4. **Infinite-dimensional SDP**: parametrise $g$ as a trigonometric polynomial in a large Fourier basis and solve the resulting SDP (this IS what `lasserre/fourier_sdp.py` does in a weaker, knot-discretised form). A continuous-Fourier SDP with Bochner's theorem as a PD constraint and Chebyshev-type constraints for $M_g$ is the standard route.

None of 1–4 is a small undertaking; together they constitute a real research project.

## Files

- `theory.md` — clean derivation with the **correct** admissibility and the sharp `cos²` weight.
- `rigorous_max.py` — generic interval B&B, reusable outside this package.
- `family_f1_selberg.py`, `family_f2_gauss_poly.py` — correct and tested.
- `family_f3_vaaler.py` — documented stub.
- `verify.py`, `optimise.py`, `run_all.py` — orchestration.
- `tests/test_delsarte_dual.py` — 10 unit tests, all passing.
- `run.log` — end-to-end log of the final run.

## Honest conclusion

The claim "prove a new rigorous lower bound above 1.2802 in one session" was never realistic; the record has stood for 16 years precisely because the construction is hard. The **infrastructure** delivered here is the correct and reusable part: rigorous interval arithmetic, closed-form PD certificates for two families, a genuine Paley–Wiener-based weight, and a verification pipeline that produces two-sided balls with provably-correct endpoints. Someone with a better $g$ (or a better $w$) can drop it into this pipeline and get a certified result immediately.
