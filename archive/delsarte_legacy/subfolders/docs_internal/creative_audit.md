# Creative Audit — How to push past 1.28 on C_{1a}

*Written while research agents (MO, CS, literature, SDP, multi-moment) run in
parallel. Will be merged with their findings into a Master Plan.*

## Current state

- **Published floor:** 1.28 (Cloninger-Steinerberger 2017).
- **MV dual** reproduced by us at 1.2748; their **theoretical ceiling** for
 this dual framework is ~1.276; our scan at `n=1000` hits 1.2764 asymptoting
 toward 1.276.
- **Published ceiling:** 1.5029 (primal construction).
- **Target:** publishable new lower bound, ideally > 1.28.

## What we have full control over

For the MV-style Delsarte dual:
- The arcsine kernel `K` (parameter δ)
- Trig poly `G` with coefficients `a_j`, period `u`, count `n`
- Multi-moment choice `N ⊂ {1,2,…}` for the `z_n = |f̂(n)|` refinement
- Optional: replace Cauchy–Schwarz with Hölder or power-mean inequality

## The 1.276 ceiling — why

MV proved (Remark p.5) that at their optimal (δ, G, N={1}),
$\int(f*f)K + \int(f \circ f)K = 2\|f_s * \eta_\delta\|_2^2$
so the dual can never exceed $\inf_{f_s} \|f_s * \eta_\delta\|_2^2$ over
admissible symmetric $f_s$. This limit is ~1.276 at their best δ.

**Crucial observation**: this ceiling assumes (a) K = arcsine, (b) Cauchy–Schwarz
as the inner ineq, (c) only `z_1` pulled out. Relaxing ANY of these could
break the ceiling.

---

## PROMISING DIRECTIONS — ranked by expected lift vs effort

### (A) **Multi-moment z_n refinement — MV's own open direction #3**

**Idea:** extend Lemma 3.3/3.4 to bound $|\hat h(n)|$ for n=2,3,…; pull all of
them out of the Cauchy-Schwarz like MV did for z_1.

**Why it might help:** each z_n cut reduces slack. MV's jump from eq (7) → (10)
with just z_1 gave them +0.0005 (1.2743 → 1.2748). With z_2, z_3, z_4
cumulatively we could plausibly reach 1.276 and maybe nip above.

**Lemma 3.4 generalization:** for $h \ge 0$, $\int h=1$, $h \le M$, supp in
$[-1/2,1/2]$:
$$
|\hat h(n)| \le \frac{M \sin(\pi n/M)}{\pi n}
$$
(extremizer: $h = M\cdot \mathbb 1_{[-1/(2M),1/(2M)]}$ for $n \le M$).

**Status:** I derived this mentally; agent E will confirm/correct. Risk:
argument fails when $\sin(\pi n/M) < 0$ for large n; need careful handling.

**Cost:** ~2 days to derive, implement, verify. **Expected lift:** +0.001–0.005
(toward 1.276 ceiling from 1.2748 cleanly, possibly to ≈1.276).

### (B) **Break the kernel K: use K ≠ arcsine**

**Idea:** MV inherited arcsine from Martin–O'Bryant and never optimized over K.
The ONLY requirement is $K \ge 0$, supp ⊂ [-δ, δ], $\int K = 1$, $\tilde K(j) \ge 0$.

**Candidate kernels:**
- **Truncated Fejér**: $K(x) \propto \sin^2(\pi N x)/x^2$ with compact support.
 $\tilde K(j) = (N - |j|)_+/N$ — explicit, nonneg.
- **Gaussian-modulated indicators**: $K \propto e^{-\alpha x^2}\mathbb{1}_{[-δ,δ]}$.
 Nonneg FT only for special α.
- **Beurling-Selberg auxiliary**: specific compactly-supported PD kernels.
- **Convex combination** of two different K's — each gives a bound, combined?
- **Soft minorant** of arcsine (cap its divergence at 0): $K_\epsilon \to K$ as $\epsilon \to 0$; has finite $\|K_\epsilon\|_2$.

**Key insight:** MV's "1.276 ceiling" is proved FOR ARCSINE. For a different K,
the ceiling could be different (higher OR lower). No general no-go.

**Cost:** 1–2 days per candidate kernel. **Expected lift:** unknown; could
easily be +0.002 to +0.02. Worth trying.

### (C) **Replace Cauchy–Schwarz with Hölder / power-mean**

**Idea:** MV eq (2) uses Cauchy-Schwarz: $\int(f*f)K \ge 1 + \sqrt{(M-1)(\|K\|_2^2-1)}$.
Replace with Hölder at exponent $p$:
$\int(f*f)K = \int (f*f)\cdot K \le (\int (f*f)^p)^{1/p}(\int K^q)^{1/q}$.

Taking p = q = 2 recovers C–S. At p ≠ 2, we get a different bound involving
$(f*f)^p$ norms.

**Why it might help:** Cauchy-Schwarz is optimal only for the specific
"symmetric" structure. $f*f$ is ≥ 0, so higher $L^p$ norms carry more info.

**Cost:** 2–3 days. **Expected lift:** small to moderate (0.001–0.01).

### (D) **SDP/Lasserre moment hierarchy**

**Idea:** parametrize f by its moments (or Fourier coefficients), impose
PSD conditions for positivity, get an SDP whose optimum converges to C_{1a}
as the relaxation level increases.

**Standard template:** $C_{1a} = \min M$ s.t. $\exists f$ with moments $m_k$
satisfying Hausdorff PSD + $\|f*f\|_\infty \le M$.

The last constraint is nonconvex, but dualization gives:
$C_{1a} \ge$ LP value of max_g $\int g d(f*f)$ subject to admissible g.

**Cohn-Elkies-style LP:** parametrize $g$ as a nonnegative polynomial, find
optimal with SDP. This is closely related to our F1 LP but with proper
moment/SOS constraints.

**Level-d hierarchy:** variables grow as O(d²), matrix sizes O(d). d=20–100
feasible on 192-core pod.

**Cost:** 1–2 weeks. **Expected lift:** unknown — could BEAT CS 2017 easily
(1.28 → 1.30+) or could plateau near 1.276. Best shot at a big win.

### (E) **Combine MV dual with CS branch-and-prune**

**Idea:** CS's method fits a discretized f. At each branch node, use MV's dual
to PRUNE nodes where the dual says $\|f*f\|_\infty \ge$ cutoff. Tighter
cutting → fewer branches → bigger discretization feasible.

**Cost:** 1 week. **Expected lift:** same magnitude as CS's improvements
since 2017 (modest: 1.28 → 1.282?).

### (F) **Replace G with a non-cosine basis**

MV uses cos(2πjx/u) basis. Alternatives:
- Chebyshev polynomials on $[-1/4, 1/4]$.
- Bernstein polynomials (positive-by-construction on interval).
- SOS polynomial basis (Lukács-Markov).

**Key question:** does the $\tilde K(j) = |J_0|^2/u$ identity still hold? Only
for cos basis matched to K's arcsine structure. Different basis ⇒ need new
K-G compatibility relation.

**Cost:** 1 week. **Lift:** uncertain.

### (G) **Asymmetric f**

MV's derivation of eq (3) uses $(f*f) + (f \circ f)$ (auto-conv + auto-corr).
For EVEN f these are equal; for general f, $f \circ f$ dominates
$(f \circ f)(0) = \int f^2 \ge 2$. MV's argument thus might be loose for
asymmetric f. Is there a sharper version that's tight on non-even f?

**Cost:** deep math, 1–2 weeks. **Lift:** could be 0 (framework respects
asymmetry) or could break the 1.276 ceiling.

### (H) **Double dualization / 2nd-order Lasserre**

**Idea:** the primal is $\min_f \|f*f\|_\infty$. The dual is over test functions.
Doubly dualize to get an inner/outer problem. Under mild conditions, zero
duality gap gives $C_{1a}$ EXACTLY. Check if MV's dual is the full dual or a
relaxation.

If there's a gap, closing it could give a big jump.

**Cost:** formal duality analysis, 1 week. **Lift:** unclear.

---

## WILD CARDS / speculative

### (I) **Random / learned test functions**
Use gradient descent / neural network to find good $(K, G)$. Non-rigorous search,
then rigorous verify.

### (J) **Non-Fourier duals**
Instead of Cauchy-Schwarz in Fourier space, use space-domain inequalities.
E.g., rearrangement inequalities, Kantorovich-Rubinstein transport duality.

### (K) **Quantum bounds** (joke but possible)
Some problems in combinatorics have quantum-inspired bounds (e.g., via
non-commutative polynomial optimization). Unlikely here but not impossible.

### (L) **Exploit known Sidon set results**
$C_{1a}$ is the continuous analog of discrete Sidon set density. Known
discrete results (Erdős, Lindström) might translate.

### (M) **GP/GF functional equations for extremizers**
If we BELIEVE the extremal f has a specific form (e.g., piecewise linear, or
Chebyshev-type), write down the variational equation and solve.

---

## What's the STRUCTURAL problem?

MV's 1.276 ceiling says: "If you restrict $G$ and $K$ via the cosine/arcsine
structure, and apply Cauchy-Schwarz once, you can't go above 1.276."

To break 1.28 we need ONE OF:
- Different ansatz space (A, B, C, F).
- Different inequality (C, H).
- Hierarchical refinement (D).
- Hybrid approach (E).

Strongest theoretical bets: **(D) SDP hierarchy** and **(A) multi-moment**.
Strongest practical bets: **(A) multi-moment** (concrete, fast).

---

## Ranked attack plan (to be finalized after agents report)

### Priority 1 (week 1–2):
- **(A) Multi-moment z_n refinement**. 1–2 days. Gets us from 1.2748 to
 (hopefully) ≈1.276.
- **(B) Alternative kernels K**. 1 week. Test 3 candidates.
- **(D) SDP hierarchy design**. 1 week. Literature + initial implementation.

### Priority 2 (week 3–4):
- **(D) SDP implementation at level d=20**. Exploit pod.
- **(C) Hölder / p-norm variant of eq (2)**.

### Priority 3 (if time):
- **(E) MV+CS hybrid**.
- **(F) Non-cosine basis for G**.

Each step ends with a rigorous verify stage. We don't claim a bound until
it's a certified interval.

---

## Closing thought

The MV 1.276 ceiling is REAL but NARROW — it applies to a specific
analytical framework. Breaking it is a matter of finding the right
generalization. The agents below will tell us exactly where the slack is.
