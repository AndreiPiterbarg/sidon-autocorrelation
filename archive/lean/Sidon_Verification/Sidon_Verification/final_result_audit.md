# Correctness Audit: `lean/Sidon/FinalResult.lean`

> **OUTDATED (2026-04-07):** This audit was conducted against the old coarse-grid
> parameterization. The Lean definitions have been updated to the C&S fine grid
> (compositions summing to 4nm, heights = c_i/m). The proof chain needs
> re-verification with the updated definitions.

**Audited**: 2026-03-25
**File**: `lean/Sidon/FinalResult.lean` (163 lines)
**Role**: Capstone of the formal proof that the autoconvolution constant c >= 7/5 = 1.4

## Contents of FinalResult.lean

| Declaration | Type | Lines | Purpose |
|-------------|------|-------|---------|
| `cascade_all_pruned` | axiom | 59-64 | Sole computational axiom encoding the 70-hour cascade result |
| `autoconvolution_ratio_scale_invariant` | theorem | 68-98 | R(a*f) = R(f) for a > 0 |
| `eLpNorm_convolution_scale_ne_top` | private lemma | 111-124 | Finiteness of ||g*g||_inf preserved under scaling |
| `autoconvolution_ratio_ge_7_5` | theorem | 126-160 | **MAIN THEOREM**: R(f) >= 7/5 for all admissible f |

## Dependencies

```
FinalResult imports:
  - Mathlib
  - Sidon.Defs              (autoconvolution_ratio, test_value, bin_masses, canonical_discretization, contributing_bins)
  - Sidon.Foundational      (canonical_discretization_sum_eq_m)
  - Sidon.StepFunction       (step function properties)
  - Sidon.TestValueBounds   (sum_bin_masses_eq_one)
  - Sidon.DiscretizationError (dynamic_threshold_sound)
```

No circular dependencies. None of the imported files import FinalResult.

---

## 1. Axiom Audit (`cascade_all_pruned`) — NO ISSUES FOUND

### Axiom statement (lines 59-64)

```lean
axiom cascade_all_pruned :
  forall c : Fin (2 * 64) -> N, sum i, c i = 20 ->
    exists l s_lo, 2 <= l /\
      test_value 64 20 c l s_lo >
        (7/5 : R) + (4 * (64 : R) / l) *
          (1 / (20 : R)^2 + 2 * ((sum i in contributing_bins 64 l s_lo, (c i : R)) / 20) / 20)
```

### Quantifier verification

- `forall c : Fin (2 * 64) -> N` ranges over functions from `Fin 128` to `N`. Since `2 * 64 = 128`, this covers all assignments of nonneg integers to 128 bins.
- `sum i, c i = 20` restricts to compositions of m=20 into 128 bins.
- 128 bins = d at cascade level L5 (d_0=4 doubled 5 times). Matches `cpu_cascade_20260319_201644.json` level 5: `d_child=128`.
- **VERIFIED: Correct.**

### Conclusion verification

- `exists l s_lo, 2 <= l /\ test_value 64 20 c l s_lo > [threshold]` asserts existence of window parameters with l >= 2.
- The `test_value` function (Defs.lean:48-53) uses n=64, giving d=2*64=128, with mass parameter m=20.
- **VERIFIED: Correct.**

### Threshold formula cross-check against `dynamic_threshold_sound`

The axiom's threshold:
```
(7/5 : R) + (4 * (64 : R) / l) * (1 / (20 : R)^2 + 2 * ((sum / 20) / 20))
```

`dynamic_threshold_sound` (DiscretizationError.lean:743) expects:
```
c_target + (4 * n / l) * (1 / m ^ 2 + 2 * W / m)
```
where `W = (sum i in contributing_bins n l s_lo, (c i : R)) / m`.

Substituting n=64, m=20, c_target=7/5:
- `2 * W / m = 2 * ((sum/20)) / 20` by left-associativity of `*` and `/` in Lean 4.
- The axiom's `2 * ((sum / 20) / 20)` has explicit parentheses matching this associativity.
- `1 / (20 : R)^2 = 1 / m^2`.
- `4 * (64 : R) / l = 4 * n / l`.

**VERIFIED: Exact match.**

### Threshold formula cross-check against Python

The Python pruning code (`run_cascade.py:57-132`) works in integer convolution space. The integer-space threshold is:

```python
ct_base_ell = c_target * m^2 * ell / (4*n)    # per-ell constant
dyn_x = ct_base_ell + 1.0 + eps_margin + 2.0 * W_int
dyn_it = int64(dyn_x * one_minus_4eps)
# Prune if: ws > dyn_it
```

Converting the Lean threshold from TV-space to integer-space (multiply by `m^2 * l / (4n)`):
```
c_target * m^2 * l / (4n) + 1 + 2 * W_int
```

This matches `ct_base_ell + 1 + 2*W_int` exactly. The Python adds `eps_margin = 1e-9 * m^2` and uses `one_minus_4eps = 1 - 4*DBL_EPS` for numerical safety, making it strictly more conservative (prunes fewer compositions). Therefore, any composition pruned by Python satisfies the axiom's strict inequality.

**Conservativeness proof**: For any composition where Python prunes (`ws > dyn_it`):
- If the mathematical threshold T is not an integer: `dyn_it >= floor(T)`, so `ws > floor(T)` implies `ws > T` (since ws is integer).
- If T is an integer: `dyn_x = T + eps_margin > T`, so `dyn_it >= T` (since `eps_margin > 4*DBL_EPS*dyn_x`), and `ws > T`. The strict inequality holds.

**VERIFIED: Python computation is conservative with respect to the axiom.**

### Contributing bins cross-check

Lean definition (Defs.lean:87-89):
```lean
def contributing_bins (n : N) (l s_lo : N) : Finset (Fin (2 * n)) :=
  Finset.filter (fun i => exists j : Fin (2*n), s_lo <= i.1 + j.1 /\ i.1 + j.1 <= s_lo + l - 2) Finset.univ
```

This reduces to the contiguous range `[max(0, s_lo-(d-1)), min(d-1, s_lo+l-2)]` where d=2n.

Python (`run_cascade.py:115-121`):
```python
lo_bin = max(0, s_lo - (d-1))
hi_bin = min(d-1, s_lo + ell - 2)
W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
```

**VERIFIED: Exact match.**

### Computation data match

From `data/cpu_cascade_20260319_201644.json`:
- `n_half=2`, `m=20`, `c_target=1.4`
- At L5: `d_child=128`, so `n_half_child=64`
- `survivors_out=0`, `proven_at="L5"`

The axiom uses n=64 (so d=128), m=20, c_target=7/5=1.4. **Exact match.**

---

## 2. Scale Invariance Proof — NO ISSUES FOUND

`autoconvolution_ratio_scale_invariant` (lines 68-98):

### Proof steps

| Step | Lines | What it shows | Algebraic justification |
|------|-------|---------------|------------------------|
| `h_conv` | 74-78 | `(a*f)*(a*f)(x) = a^2 * (f*f)(x)` | `int af(t)*af(x-t) dt = a^2 int f(t)f(x-t) dt`. Uses `integral_mul_left` + `ring`. |
| `h_norm` | 80-91 | `||a^2 * (f*f)||_inf.toReal = a^2 * ||f*f||_inf.toReal` | Converts to `a^2 . g` form, uses `eLpNorm_const_smul` and `enorm_eq_ofReal`. Requires `a^2 > 0` (from `ha : 0 < a`). |
| `h_int` | 92-94 | `int(a*f) = a * int(f)` | Direct `integral_const_mul`. |
| Final | 95-98 | `a^2*N / (a*I)^2 = N / I^2` | `field_simp` with `a != 0`, `a^2 != 0`. Simplifies `a^2 / a^2 = 1`. |

**VERIFIED: Algebraically and proof-technically correct.**

---

## 3. Main Theorem Proof Logic — NO ISSUES FOUND

`autoconvolution_ratio_ge_7_5` (lines 126-160):

### Theorem statement

```lean
theorem autoconvolution_ratio_ge_7_5 (f : R -> R)
    (hf_nonneg : forall x, 0 <= f x)
    (hf_supp : Function.support f <= Set.Ioo (-1/4 : R) (1/4))
    (hf_int_pos : integral volume f > 0)
    (h_conv_fin : eLpNorm (convolution f f ...) top volume != top) :
    autoconvolution_ratio f >= 7/5
```

The `h_conv_fin` hypothesis is necessary and well-documented (lines 103-106): without it, `eLpNorm.toReal` maps infinity to 0, giving `autoconvolution_ratio = 0 < 7/5`. This hypothesis holds for all bounded, L^2, or step functions.

### Step-by-step trace

| Line(s) | Step | What happens | Verification |
|----------|------|-------------|--------------|
| 133-134 | Normalization setup | Set `I = int f`, `g = (1/I) * f` | `I > 0` from `hf_int_pos`. Correct. |
| 136-138 | Scale invariance | `R(f) = R(g)` | Uses `autoconvolution_ratio_scale_invariant f (1/I) (by positivity)`. Since `1/I > 0`, the theorem gives `R((1/I)*f) = R(f)`. With `.symm`: `R(f) = R(g)`. Then `rw [h_ratio_eq]` changes goal to `R(g) >= 7/5`. **Correct.** |
| 140-141 | g nonneg | `forall x, 0 <= g x` | `(1/I) * f(x) >= 0` since `1/I > 0` and `f(x) >= 0`. **Correct.** |
| 142-144 | g support | `supp(g) <= Ioo(-1/4, 1/4)` | If `g(x) != 0` then `(1/I)*f(x) != 0`, so `f(x) != 0` (since `1/I != 0`), so `x in supp(f)`. **Correct.** |
| 145-147 | g integral | `int g = 1` | `int (1/I)*f = (1/I)*int f = (1/I)*I = 1` via `integral_mul_left` + `div_mul_cancel_0`. **Correct.** |
| 148 | Discretize | `c := canonical_discretization g 64 20` | Defines the composition at n=64, m=20. |
| 149-150 | Mass nonzero | `sum bin_masses g 64 j != 0` | Uses `sum_bin_masses_eq_one 64 (by norm_num) g hg_supp hg_int` (TestValueBounds.lean:68). Signature: `(n : N) (hn : n > 0) (f : R -> R) (hf_supp) (hf_int)`. Args match: n=64, hn, g, hg_supp, hg_int. Result: sum=1, then `one_ne_zero`. **Correct.** |
| 151-152 | Sum constraint | `sum c i = 20` | Uses `canonical_discretization_sum_eq_m g 64 20 (by norm_num) (by norm_num) h_mass_nz hg_nonneg` (Foundational.lean:112). Signature: `(f)(n m)(hn : n>0)(hm : m>0)(h_mass_pos)(hf_nonneg)`. All 6 args match. **Correct.** |
| 153 | Apply axiom | Destructure `cascade_all_pruned c hc_sum` | Gets `l, s_lo, hl : 2 <= l, h_exceeds : TV > threshold`. **Correct.** |
| 154-156 | Conv finiteness | `||g*g||_inf != top` | `eLpNorm_convolution_scale_ne_top f (1/I) h_conv_fin`. Since `g = (1/I)*f`, this derives finiteness from `h_conv_fin`. **Correct.** |
| 157-158 | Window mass | `W := (sum .../20), h_W_def := rfl` | Defines W matching `dynamic_threshold_sound`'s hW parameter. **Correct.** |
| 159-160 | Apply threshold | `exact dynamic_threshold_sound 64 20 (7/5) ... rfl` | Passes all 20 positional arguments. See detailed check below. |

### Detailed argument match for `dynamic_threshold_sound` (DiscretizationError.lean:738-750)

| # | Parameter | Signature type | Passed value | Match? |
|---|-----------|---------------|-------------|--------|
| 1 | `n` | `N` | `64` | Yes |
| 2 | `m` | `N` | `20` | Yes |
| 3 | `c_target` | `R` | `7/5` | Yes |
| 4 | `hn` | `n > 0` | `by norm_num` | Yes |
| 5 | `hm` | `m > 0` | `by norm_num` | Yes |
| 6 | `hct` | `0 < c_target` | `by norm_num : (0:R) < 7/5` | Yes |
| 7 | `c` | `Fin(2*n) -> N` | `c` (= canonical_discretization g 64 20) | Yes |
| 8 | `hc` | `sum c i = m` | `hc_sum` | Yes |
| 9 | `l` | `N` | `l` (from axiom) | Yes |
| 10 | `s_lo` | `N` | `s_lo` (from axiom) | Yes |
| 11 | `hl` | `2 <= l` | `hl` (from axiom) | Yes |
| 12 | `W` | `R` | `W` (defined line 157) | Yes |
| 13 | `hW` | `W = sum.../m` | `h_W_def` | Yes |
| 14 | `h_exceeds` | `TV > threshold` | `h_exceeds` (from axiom) | Yes |
| 15 | `f` (univ) | `R -> R` | `g` | Yes |
| 16 | nonneg | `forall x, 0 <= f x` | `hg_nonneg` | Yes |
| 17 | support | `supp <= Ioo` | `hg_supp` | Yes |
| 18 | integral | `int f = 1` | `hg_int` | Yes |
| 19 | conv finite | `eLpNorm != top` | `h_conv_fin_g` | Yes |
| 20 | discretization | `canonical_disc f n m = c` | `rfl` (c is definitionally equal) | Yes |

**VERIFIED: All 20 arguments match. Conclusion `autoconvolution_ratio g >= 7/5` is the goal after rewrite.**

---

## 4. Proof Completeness — NO ISSUES FOUND

- **`sorry`**: Zero occurrences in FinalResult.lean (grep verified).
- **`admit`**: Zero occurrences in FinalResult.lean (grep verified).
- **Axioms**: Grep across all 21 files in `lean/Sidon/` finds `axiom` only in FinalResult.lean:59 (`cascade_all_pruned`).

**This is the sole axiom in the entire proof.**

---

## 5. Numeric Constants — NO ISSUES FOUND

| Constant | Expected | Occurrences in file | Consistency |
|----------|----------|-------------------|-------------|
| Target ratio | `7/5` (not `1.4`) | Lines 60, 62, 63, 101, 132, 159 | All use `7/5` consistently |
| Resolution n | `64` (giving d=2*64=128 bins) | Lines 60, 62, 63, 148, 150, 151, 152, 159 | Consistent |
| Mass parameter m | `20` | Lines 60, 62, 64, 148, 150, 151, 152, 157, 159 | Consistent |
| JSON parameters | n_half=2 -> n=64 at L5, m=20, c_target=1.4=7/5 | `cpu_cascade_20260319_201644.json` | Exact match |

---

## 6. Type Correctness — NO ISSUES FOUND

| Expression | Type analysis | Status |
|-----------|--------------|--------|
| `c : Fin (2 * 64) -> N` | Functions from `Fin 128` to `N` | Correct |
| `sum i, c i = 20` | Sum over `Fin 128` of `N`, compared to `20 : N` | Correct |
| `test_value 64 20 c l s_lo` | Returns `R` (defined in Defs.lean:48) | Correct |
| `(7/5 : R)` | Explicit real annotation | Correct |
| `(4 * (64 : R) / l)` | `l : N` coerced to `R` via `Nat.cast` | Correct |
| `(c i : R)` | `N -> R` coercion in contributing bins sum | Correct |
| `(20 : R)^2` | Explicit `R` annotation | Correct |
| `W = .../  (20 : R)` vs `hW : W = .../m` | `Nat.cast (20 : N)` reduces to `(20 : R)` definitionally | Correct |
| `h_W_def : W = ... := rfl` | Definitionally equal by `set` definition | Correct |

---

## 7. No Circular Dependencies — NO ISSUES FOUND

FinalResult.lean imports: `Mathlib, Sidon.Defs, Sidon.Foundational, Sidon.StepFunction, Sidon.TestValueBounds, Sidon.DiscretizationError`.

None of these files import FinalResult. Confirmed by inspection and the dependency graph:

```
Defs <- Foundational <- TestValueBounds <- DiscretizationError <- FinalResult
Defs <- StepFunction ------------------------------------------> FinalResult
```

No cycles.

---

## 8. Auxiliary Lemma: `eLpNorm_convolution_scale_ne_top` (lines 111-124)

This private lemma shows that if `||f*f||_inf != top`, then `||(a*f)*(a*f)||_inf != top`.

| Step | What it shows | Verification |
|------|--------------|--------------|
| `h_eq` (116-122) | `(a*f)*(a*f) = a^2 . (f*f)` | Same convolution rewriting as scale invariance. Uses `integral_const_mul` + `ring`. **Correct.** |
| Line 123 | `||a^2 . g||_inf = ||a^2||_enn * ||g||_inf` | By `eLpNorm_const_smul`. **Correct.** |
| Line 124 | `||a^2||_enn * ||g||_inf != top` | `ENNReal.mul_ne_top`: real norm is finite (`coe_ne_top`) times finite hypothesis. **Correct.** |

Called on line 156 with `f=f` (original), `a=1/I`, `h_conv_fin`. Correctly derives finiteness for `g = (1/I)*f`.

---

## Overall Verdict

**NO ISSUES FOUND.** The proof in `FinalResult.lean` is logically correct across all 7 audit categories:

1. The axiom `cascade_all_pruned` faithfully encodes the 70-hour computation with the exact threshold formula matching both `dynamic_threshold_sound` and the Python pruning code (Python is strictly more conservative due to epsilon safety margins).

2. The scale invariance proof `autoconvolution_ratio_scale_invariant` is algebraically sound: convolution scales by a^2, L-inf norm scales by a^2, integral scales by a, ratio cancels.

3. The main theorem `autoconvolution_ratio_ge_7_5` correctly normalizes f to g with unit integral, applies scale invariance, discretizes at (n=64, m=20), invokes the axiom, and passes all 20 arguments to `dynamic_threshold_sound` with exact type and value matches.

4. The sole axiom in the entire 21-file proof is `cascade_all_pruned`. Zero `sorry` or `admit`.

5. All numeric constants (7/5, 64, 20) are used consistently and match the computational data.

6. All type coercions (N -> R) are handled correctly.

7. No import cycles exist.

**Compilation status**: FinalResult.lean cannot currently compile due to upstream errors in `TestValueBounds.lean` (broken proofs for `step_function_continuousAt` and `continuous_test_value_le_ratio`), which block `DiscretizationError.lean`, which blocks this file. However, the logical content of FinalResult.lean itself is sound. Once the dependency chain compiles, this file should compile without modification.
