# `coarse_cascade_prover.py` — Concrete Fixes to Make It Correct + Tight

> Scoped strictly to `C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\coarse_cascade_prover.py`.
>
> The prover currently produces $C_{1a} \ge 1.15$ rigorously and stops there.
> **Worse, the rigorous claim is itself UNSOUND** — `run_box_certification` samples random cells. A proof requires every cell.
>
> Below: what's wrong, and the exact code changes that fix it, ranked by impact.

---

## §0 The 4 problems, ranked

| # | Problem | Severity | Lines |
|---|---|---|---|
| 1 | **Box certification is not a proof — it samples** | **Soundness-fatal** | `coarse_cascade_prover.py:705-751` |
| 2 | **Box-cert inner bound is loose** (water-filling, not exact min) | Tightness | `coarse_cascade_prover.py:631-702` |
| 3 | **"Subtree pruning" is a heuristic on the partial state, not a real subtree LB** | Performance + tightness | `coarse_cascade_prover.py:417-456` |
| 4 | **Threshold integer-lattice gives margin=0 quantization** when $c \cdot \ell \cdot S^2 / (2d) \in \mathbb{Z}$ | Structural barrier | `coarse_cascade_prover.py:42-57` |

Fix all four and the prover should clear $c=1.25$, plausibly $c=1.30$, at $d=12, S\in[20, 21]$ — with rigorous soundness restored.

---

## §1 SOUNDNESS FIX (must do before claiming any bound)

### 1.1 The problem at lines 705-751

```python
# coarse_cascade_prover.py:705-751
def run_box_certification(survivors_final_level, d_final, S, c_target,
                          n_sample=1000, verbose=True):
    ...
    rng = np.random.RandomState(42)
    ...
    for _ in range(n_sample):
        # Random composition of S into d_final parts
        mu_int = np.zeros(d_final, dtype=np.int32)
        remaining = S
        for i in range(d_final - 1):
            mu_int[i] = rng.randint(...)  # <-- random cells
        ...
```

This certifies a **random sample**, not every cell. The driver at line 832 passes `n_sample=2000` then prints "PROOF: C_{1a} >= {c_target}". **The printed claim is false** — no proof has been constructed.

### 1.2 The fix — exhaustive iteration over canonical compositions

Replace `run_box_certification` with an exhaustive enumerator. The cascade has already established that every canonical composition is pruned at the grid points (Theorem 1 + integer threshold); box-cert must extend this to every Voronoi cell of every grid composition.

```python
# Replace coarse_cascade_prover.py:705-751
def run_box_certification(d_final, S, c_target, verbose=True, max_workers=None):
    """Exhaustively box-certify every canonical composition's Voronoi cell.
    
    Returns (all_certified, n_failed, worst_min_tv, failure_compositions).
    """
    delta = 1.0 / S
    x_cap = compute_xcap(c_target, S, d_final)
    n_cert = 0
    n_fail = 0
    worst_tv = 1e30
    failures = []
    
    # Enumerate canonical compositions (deduped by reflection if Z2-canonical).
    # Reuse generate_canonical_compositions_batched from cloninger-steinerberger.
    from compositions import generate_canonical_compositions_batched
    for batch in generate_canonical_compositions_batched(d_final, S, x_cap):
        for mu_int in batch:
            mu_center = mu_int.astype(np.float64) / S
            certified, min_tv = _box_certify_cell(mu_center, d_final, delta, c_target)
            if certified:
                n_cert += 1
            else:
                n_fail += 1
                failures.append(mu_int.copy())
            if min_tv < worst_tv:
                worst_tv = min_tv
    
    all_certified = (n_fail == 0)
    if verbose:
        total = n_cert + n_fail
        print(f"    Cells: {n_cert}/{total} certified, {n_fail} failed")
        print(f"    Worst QP min TV: {worst_tv:.6f} (need >= {c_target})")
        if not all_certified:
            print(f"    [PROOF INVALID] {n_fail} cells failed box-cert")
            for f in failures[:5]:
                print(f"      failure: {f.tolist()}")
    return all_certified, n_fail, worst_tv, failures
```

Update `run_cascade` lines 829-834: rename printed banner from "PROOF" to "GRID-POINT PROOF" until `all_certified == True`. Then upgrade to "RIGOROUS PROOF".

**Why this is fast enough**: at $d=8, S=50$, $x_{\mathrm{cap}}=20$, total canonical compositions ≈ 1.4 M. Box-cert at ~1 ms/cell ≈ 25 minutes single-threaded. Parallelize via `numba.prange` on `_box_certify_cell` invocation. At $d=12, S=20$ (where the L0 cascade leaves 29 survivors per cpu_run_20260413_141632.log:48), this is **seconds**.

---

## §2 TIGHTNESS FIX — replace water-filling with vertex enumeration

### 2.1 The problem at lines 631-702

Current `_box_certify_cell` does:
1. For each window $(\ell, s)$, identify contributing bins.
2. Set $\mu_i = \mathrm{lo}_i$, then pour excess mass into non-contributing bins first, then contributing bins.
3. Compute TV at this single feasible point.
4. Return that as `min_tv`.

This is a **first-order heuristic**: it's just one feasible point inside the cell, not the true minimum of $TV_W$ over the cell.

The exact minimum of a (possibly indefinite) quadratic over a polytope sits at a vertex (by the vertex theorem of quadratic-on-polytope). For a cell `{|δᵢ|≤h, 1ᵀδ=0}` of half-side $h=1/(2S)$, vertex enum is $O(d \cdot 2^{d-1})$ — exact, no slack.

### 2.2 The fix — wire in the existing vertex enumerator

Module `cloninger-steinerberger/cpu/qp_bound.py:50` already has `qp_bound_vertex(mu*, h, A_W, c_target)`. Use it.

```python
# coarse_cascade_prover.py — add at top
import sys, os
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(_REPO, 'cloninger-steinerberger'))
from qp_bound import qp_bound_vertex, build_window_matrix

@njit(cache=True)
def _box_certify_cell_exact(mu_center, d, delta, c_target):
    """EXACT min-over-cell of max_W TV_W via vertex enumeration.
    
    Sound for d <= 16; for d > 16 dispatch to McCormick LP.
    """
    h = delta / 2.0
    conv_len = 2 * d - 1
    best_min_tv = 0.0
    
    for ell in range(2, 2 * d + 1):
        for s in range(conv_len - ell + 2):
            A_W = build_window_matrix(d, ell, s)  # sparse 0/1, see qp_bound.py:37
            # vertex enum: min over cell of (2d/ell) * mu^T A_W mu
            min_tv = qp_bound_vertex(mu_center, h, A_W, ell, d)
            if min_tv > best_min_tv:
                best_min_tv = min_tv
            if best_min_tv >= c_target:
                return True, best_min_tv
    return best_min_tv >= c_target, best_min_tv
```

Replace the `_box_certify_cell` call at line 738 (and the new exhaustive driver in §1.2) with `_box_certify_cell_exact`.

### 2.3 Expected gain (per Agent 2's analysis)

The reported `min_net = -0.5725` at d=12, S=20, c=1.30 is from the **un-corrected L0 prune**, not from the box-cert. After L0 leaves 29 surviving cells (cpu_run_20260413_141632.log:48), box-cert only needs to certify those 29 cells. At d=12, vertex enum is `12 · 2^11 = 24,576` vertices/cell × ~300 windows × ~144 pair-ops ≈ $10^9$ ops total — **single-digit seconds**.

Plausibly certifies **c=1.25 directly at d=12, S=20**; possibly c=1.30 if those 29 cells contain near-extremal arcsine-like distributions whose true TV is well above 1.30.

### 2.4 d > 16 fallback

Vertex enum is `O(d · 2^{d-1})`. At d=16: 524k vertices, OK. At d=32: $2^{31}$, infeasible. The deep cascade levels (L4 at d=32, L5 at d=64) need a McCormick LP fallback.

`tests/test_joint_qp_box_cert.py:324-482` already has a McCormick LP. Move it to `cpu/qp_bound.py` as `mccormick_lp_bound` and wire as fallback when d > 16:

```python
@njit(cache=True)  # may need to drop @njit if scipy.linprog can't be JITed
def _box_certify_cell_robust(mu_center, d, delta, c_target):
    if d <= 16:
        return _box_certify_cell_exact(mu_center, d, delta, c_target)
    else:
        # scipy LP — slower but sound
        return _box_certify_cell_mccormick(mu_center, d, delta, c_target)
```

---

## §3 PERFORMANCE FIX — port a real subtree LB

### 3.1 The problem at lines 417-456

`_cascade_child_bnb` "subtree pruning" tests `ws > thr[ell]` on the **prefix only** (positions assigned so far). That's a fail-fast: it kills configurations that have already crossed the threshold on the prefix. It does **not** kill subtrees on the basis of an upper bound on what completions can reach. So it does no actual subtree pruning — every subtree gets visited eventually.

### 3.2 The fix — port `min_contrib` from `cpu/run_cascade.py:2024-2110`

The mature B&B in `cpu/run_cascade.py` lines 1991-2110 (and 2640-2740) already implements a sound subtree LB:

```
For each unassigned cursor position p, with bounds [lo[p], hi[p]]:
  add the GUARANTEED MINIMUM contribution to conv[k]:
    self-terms     ml*ml, mh*mh             (line ~2036-2037)
    mutual         2*c1*(2P - c1) at endpts (line ~2043-2045)  
    cross-fixed    2 * c[fixed] * c[unfix]  (line ~2047-2053)
    cross-unfixed  2 * lo*lo or 2 * hi*hi   (line ~2055-2068)

partial_conv + min_contrib  is then a sound LB on `conv` over the full subtree.
ws_partial + min_contrib_window > thr[ell]  =>  PRUNE the subtree.
```

This is a verbatim port — ~80-100 LOC. Add it inside `_cascade_child_bnb` between the partial-prune block (line 417) and the quick-check (line 470 area).

### 3.3 Expected gain

Per Agent 9: 5-50× pruning at internal levels, geometric to **~600× at L4**. Converts d_0=8, S=50, full L4 cascade from a $10^9$-node tree to ~$1.6 \times 10^6$ — single-CPU tractable in hours.

This **does not** affect the box-cert margin (which is the binding constraint at c≥1.25). It only fixes enumeration cost. But combined with §2 it lets us run **larger d_0 and deeper cascades**, where the empirical val(d) is higher and box-cert margin grows.

---

## §4 STRUCTURAL FIX — S-shift to dodge margin=0 quantization

### 4.1 The problem at lines 42-57

```python
# coarse_cascade_prover.py:42-57
def compute_thresholds(c_target, S, d):
    thr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr[ell] = np.int64(c_target * ell * S * S / (2.0 * d) - eps)
```

When `c_target * ell * S² / (2d)` is **exactly** an integer, the smallest pruned `ws` (= `thr[ell]+1`) gives `TV = c_target` exactly. Margin = 0.

Per `cpu_run_20260413_123938.log:120,156`: at d=4, S=20, c=1.30, $\ell=2d=8$: $1.30 \cdot 8 \cdot 400 / 8 = 520 \in \mathbb{Z}$, so margin=0. At d=8, S=24, c=1.25: also hits the lattice. These are the cells that **cannot be certified at any cell-size** under cell-var > 0, because TV is exactly $c_{\mathrm{target}}$ with non-zero gradient.

This is **arithmetic, not structural**. Cured trivially by S±1.

### 4.2 The fix — auto-shift S

```python
# Add helper, call from run_cascade()
def s_shift_safe(c_target, S, d_max):
    """Return smallest S' >= S avoiding margin=0 lattice on every (ell, d <= d_max)."""
    while True:
        bad = False
        for d in range(2, d_max + 1):
            for ell in range(2, 2*d + 1):
                v = c_target * ell * S * S / (2.0 * d)
                if abs(v - round(v)) < 1e-9:
                    bad = True
                    break
            if bad:
                break
        if not bad:
            return S
        S += 1

# In run_cascade(c_target, S, d_start, max_levels):
d_max = d_start * (2 ** max_levels)
S = s_shift_safe(c_target, S, d_max)
```

For c=1.30, d_start=2, max_levels=4 (d_max=32): S=50 has margin=0 collisions; S=51 may resolve them. Cheap to check; trivial to fix.

### 4.3 Expected gain

Eliminates the entire family of cells flagged "margin=0 — cannot fix with S" in `cpu_run_20260413_123938.log`. No more permanent failures from divisibility.

---

## §5 Combined effect — projected rigorous bound after fixes

| Config | Pre-fix | After §1 (sound) | After §2 (vertex) | After §1+2+4 (S-shift) | After §1-3 (subtree) |
|---|---|---|---|---|---|
| d=4, S=300, c=1.10 | rigorous (net=0.0010) | rigorous | rigorous | rigorous | unchanged |
| d=6, S=75, c=1.15 | rigorous (net=0.0048) | rigorous | rigorous | rigorous | unchanged |
| d=6, S=50, c=1.15 | NO box (net=−0.0016) | NO | **YES** (vertex tight) | YES | YES |
| d=8, S=24, c=1.25 | NO (margin=0) | NO | **YES** (vertex tight on 19 surv) | YES | unchanged |
| d=12, S=20, c=1.30 | NO (29 surv, no box) | NO | **YES, plausibly** | YES | enables larger d |
| d=8, S=50, c=1.30, L4 | infeasible (~10⁹ nodes) | n/a | n/a | n/a | **tractable** |

**Plausible bound after all four fixes: $C_{1a} \ge 1.25$ rigorously, possibly $\ge 1.30$**.

This is *still below* CS17's 1.2802 in the 1.25 case but *would beat it* in the 1.30 case. Either way, the prover would be **correct** (currently it is not).

---

## §6 Order to do them in

1. **§1 first.** No rigorous claim can stand without it. The current "PROOF" output is mathematically false. ~0.5 day.
2. **§4.** Trivial 20-LOC change; immediately removes the margin=0 family of failures. ~0.5 day.
3. **§2.** The headline tightening — wire `qp_bound_vertex` into the new exhaustive driver. ~1 day.
4. **§3.** Largest engineering effort but only matters once §1+§2 unlock the deeper cascade. ~3-5 days.

**Total: ~1 week to a sound, tight prover.**

---

## §7 Files to write/edit (concrete list)

| File | Change |
|---|---|
| `coarse_cascade_prover.py:42-57` | Add `s_shift_safe`; call from `run_cascade` |
| `coarse_cascade_prover.py:417-456` | Insert `min_contrib` subtree LB (port from `cpu/run_cascade.py:2024-2110`) |
| `coarse_cascade_prover.py:631-702` | Replace `_box_certify_cell` water-filling with `_box_certify_cell_exact` (vertex-enum) |
| `coarse_cascade_prover.py:705-751` | **Rewrite** `run_box_certification` to enumerate all canonical compositions, no sampling |
| `coarse_cascade_prover.py:758-851` | Update `run_cascade` to call the new exhaustive driver; gate the "RIGOROUS PROOF" banner on `all_certified == True` |
| `cloninger-steinerberger/cpu/qp_bound.py` | Move `mccormick_lp_bound` from `tests/test_joint_qp_box_cert.py:324-482`; expose `qp_bound_vertex(mu*, h, A_W, ell, d)` |
| `tests/test_coarse_cascade_prover_soundness.py` (new) | Test that `run_box_certification` certifies every canonical composition for known-good (d=4, S=50, c=1.05) |
| `tests/test_coarse_cascade_prover_vertex.py` (new) | Test that `_box_certify_cell_exact` agrees with brute-force vertex enum on small d=4, d=6 cases |

---

## §8 Soundness checklist after fixes

For the prover to certify "$C_{1a} \ge c$" rigorously, ALL of the following must hold:

- [ ] L0 cascade leaves 0 survivors at the final level $d_K$.
- [ ] **Every** canonical composition at $d_K$ has `_box_certify_cell_exact` return `True` (no sampling).
- [ ] The vertex enum / McCormick LP returns a **rigorous** lower bound on the cell's max-over-window-min-over-cell of $TV_W$.
- [ ] Refinement monotonicity (Theorem 1 child ≥ parent) is invoked correctly — `tests/test_refinement_monotonicity.py` already verifies empirically; for the proof to be unconditional, it must be either proven analytically or replaced with a per-cell box-cert that doesn't rely on the conjecture.

The current prover satisfies (1), violates (2), and treats (3) heuristically. After the four fixes, it satisfies (1)-(3); (4) remains as the only conjectural step.
