"""ADVERSARIAL AUDIT — AGENT 9
Edge-case / adversarial composition tests for `prune_P` and the Lean
`pointeval_*` definitions.

Approach:
 1. Build a pure-Python reference of the SAME formula that the Numba
    `prune_P` kernel uses (so we can drive it without JIT noise).
 2. Build a pure-Python reference of the LEAN definitions
    (i_lo_for_point, i_hi_for_point, W_int_for_point, n_bins_for_point,
     pointeval_value, pointeval_correction).
 3. For each adversarial composition, evaluate (a) the analytical (g*g),
    (b) the Numba kernel decision, (c) the Python kernel decision,
    (d) the Lean formulas (interpreted in Python with INTEGER subtraction
    where Lean uses Nat, so we faithfully simulate the Nat truncation).
 4. Cross-check.

All edge cases live in `EDGE_CASES` below; we run each through every
test and print a structured table.
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np

# Avoid Numba-precompile noise for these tiny tests, but still call the
# REAL kernel (so we test the JIT'd code, not just our reference).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from _M1_bench import prune_P  # the Numba-compiled kernel under audit


# =====================================================================
# (1) Pure-Python REFERENCE of prune_P (to catch JIT/Numba quirks too)
# =====================================================================
def prune_P_ref(c, n_half, m, c_target, eps_margin_factor=1e-9, verbose=False):
    """Pure-Python reference matching the Numba kernel formula exactly."""
    d = 2 * n_half
    conv_len = 2 * d - 1
    c = np.asarray(c, dtype=np.int64)
    assert c.shape == (d,)

    # Autoconv (matches kernel's ordered pair construction):
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        ci = int(c[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = int(c[j])
                if cj != 0:
                    conv[i + j] += 2 * ci * cj

    # Prefix sum:
    prefix_c = np.zeros(d + 1, dtype=np.int64)
    for i in range(d):
        prefix_c[i + 1] = prefix_c[i] + int(c[i])

    m_d = float(m)
    cs_base_m2 = c_target * m_d * m_d
    eps_margin = eps_margin_factor * m_d * m_d
    base = 2.0 * float(d) * cs_base_m2

    pruned_at_q = None
    for q in range(conv_len):
        i_lo = (q - (d - 1)) if q >= d - 1 else 0
        i_hi = q if q < d else d - 1
        W_int = int(prefix_c[i_hi + 1] - prefix_c[i_lo])
        n_bins = int(i_hi - i_lo + 1)
        thr = (base + 2.0 * float(W_int) + float(n_bins) + eps_margin)
        thr_int = int(thr)
        if verbose:
            print(f"      q={q:3d}: i_lo={i_lo}, i_hi={i_hi}, W_int={W_int}, "
                  f"n_bins={n_bins}, conv[q]={conv[q]}, thr_int={thr_int}, "
                  f"fires={int(conv[q]) > thr_int}")
        if int(conv[q]) > thr_int:
            pruned_at_q = q
            return True, pruned_at_q, dict(conv=conv, q=q, i_lo=i_lo, i_hi=i_hi,
                                           W_int=W_int, n_bins=n_bins,
                                           thr_int=thr_int, conv_q=int(conv[q]))
    return False, None, dict(conv=conv)


# =====================================================================
# (2) LEAN-FAITHFUL reference (Nat subtraction etc.)
# =====================================================================
def lean_i_lo(n: int, q: int) -> int:
    """Lean: if q + 1 <= d then 0 else q + 1 - d, with d = 2*n.
    All Nat. q+1-d only evaluates when q+1 > d, so well-defined."""
    d = 2 * n
    if q + 1 <= d:
        return 0
    return q + 1 - d


def lean_i_hi(n: int, q: int) -> int:
    """Lean: min q (d - 1).  Nat subtraction: when n=0, d=0, d-1=0."""
    d = 2 * n
    # In Lean (Nat), if d = 0 then d-1 truncates to 0.
    d_minus_1 = max(d - 1, 0)
    return min(q, d_minus_1)


def lean_W_int(n: int, c, q: int) -> int:
    d = 2 * n
    lo = lean_i_lo(n, q)
    hi = lean_i_hi(n, q)
    if lo <= hi:
        s = 0
        for i in range(lo, hi + 1):  # Finset.Icc is inclusive
            if i < d:
                s += int(c[i])
        return s
    return 0


def lean_n_bins(n: int, q: int) -> int:
    lo = lean_i_lo(n, q)
    hi = lean_i_hi(n, q)
    if lo <= hi:
        # Lean: hi + 1 - lo with Nat subtraction (always >= 0 since lo<=hi).
        return hi + 1 - lo
    return 0


def lean_pointeval_value(n: int, m: int, c, q: int) -> float:
    """In Lean: (1/(4n)) * discrete_autoconvolution a q with a_i = c_i/m.
    discrete_autoconvolution: sum over ALL ordered pairs (i,j), Fin d × Fin d.
    """
    d = 2 * n
    s = 0.0
    for i in range(d):
        for j in range(d):
            if i + j == q:
                s += float(c[i]) * float(c[j]) / (m * m)
    return (1.0 / (4.0 * n)) * s


def lean_pointeval_correction(n: int, m: int, c, q: int) -> float:
    W = lean_W_int(n, c, q)
    nb = lean_n_bins(n, q)
    return (2.0 * W + nb) / (4.0 * n * (m ** 2))


# =====================================================================
# (3) Analytical (g*g) for piecewise-constant g (truth value)
# =====================================================================
def autoconv_value_pwconst(c, n_half, m, t):
    """Compute (g*g)(t) where g is piecewise constant on d bins of
    width 1/(4n), heights c_i/m, supported on [-1/2, 1/2]?  Actually
    Sidon convention: support [-1/4, 1/4], bins width 1/(4n) on
    that interval with d = 2n bins.  So total support width is 2n*1/(4n)
    = 1/2.  Range of (g*g): [-1/2, 1/2] = [-d*w, d*w] with w=1/(4n).
    """
    d = 2 * n_half
    w = 1.0 / (4.0 * n_half)
    # g lives on [-1/4, 1/4], bin i is [i*w - 1/4, (i+1)*w - 1/4].
    # Height of bin i is c_i / m.
    # (g*g)(t) = ∫ g(s) g(t-s) ds.
    # Use the ordered-pair triangle: each pair (i,j) contributes a
    # triangle on the t-axis with peak at -1/2 + (i+j+1)*w of value
    # h_i * h_j * w; supported on [-1/2 + (i+j)*w, -1/2 + (i+j+2)*w].
    val = 0.0
    for i in range(d):
        hi = float(c[i]) / m
        if hi == 0:
            continue
        for j in range(d):
            hj = float(c[j]) / m
            if hj == 0:
                continue
            base = i + j  # peak at -1/2 + (base + 1)*w
            t_lo = -0.5 + base * w
            t_hi = -0.5 + (base + 2) * w
            t_pk = -0.5 + (base + 1) * w
            if t_lo <= t <= t_pk:
                val += hi * hj * (t - t_lo)
            elif t_pk < t <= t_hi:
                val += hi * hj * (t_hi - t)
            # else 0
    return val


def lattice_t(n_half, q):
    """t_q = -1/2 + (q+1) * w, w = 1/(4n)."""
    return -0.5 + (q + 1) / (4.0 * n_half)


def max_gg_at_lattice(c, n_half, m):
    """Max (g*g) over lattice points q = 0..conv_len-1."""
    d = 2 * n_half
    conv_len = 2 * d - 1
    best = -1e18
    arg = -1
    for q in range(conv_len):
        t = lattice_t(n_half, q)
        v = autoconv_value_pwconst(c, n_half, m, t)
        if v > best:
            best = v
            arg = q
    return best, arg


def conv_array(c, n_half):
    """Numba kernel's `conv[q]` exactly."""
    d = 2 * n_half
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        ci = int(c[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = int(c[j])
                if cj != 0:
                    conv[i + j] += 2 * ci * cj
    return conv


# =====================================================================
# (4) The edge cases
# =====================================================================
def call_numba(c, n_half, m, c_target):
    """Call the JIT'd kernel on a single composition.
    Returns True if pruned (kernel returns survived; we negate)."""
    d = 2 * n_half
    arr = np.zeros((1, d), dtype=np.int32)
    arr[0, :] = np.asarray(c, dtype=np.int32)
    survived = bool(prune_P(arr, n_half, m, c_target)[0])
    return not survived


def banner(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def case(label, c, n_half, m, c_target, S_expected=None):
    banner(label)
    print(f"  c = {list(c)}, n_half={n_half}, m={m}, d={2*n_half}, "
          f"c_target={c_target}, S={sum(c)} (expected={S_expected})")

    d = 2 * n_half
    conv_len = 2 * d - 1
    conv = conv_array(c, n_half)
    print(f"  conv (kernel ordered-pair) = {conv.tolist()}")

    # Lean's discrete_autoconvolution iterates ALL (i,j), so its conv
    # values are the same as the kernel's (they are equal since (i,j)
    # and (j,i) symmetric for autocorrelation).
    #
    # Verify identity at every lattice point:
    print("\n  Lattice point comparison: (g*g)(t_q) vs conv[q]/(4n*m^2)")
    print("    q | i_lo | i_hi | W_int | n_bins | conv[q] | (g*g)(t_q)*(4n*m^2)")
    max_err = 0.0
    for q in range(conv_len):
        i_lo_ker = (q - (d - 1)) if q >= d - 1 else 0
        i_hi_ker = q if q < d else d - 1
        i_lo_lean = lean_i_lo(n_half, q)
        i_hi_lean = lean_i_hi(n_half, q)
        W_lean = lean_W_int(n_half, c, q)
        n_lean = lean_n_bins(n_half, q)
        # Kernel's: prefix_c[i_hi+1] - prefix_c[i_lo]
        c_arr = np.asarray(c, dtype=np.int64)
        W_ker = int(np.sum(c_arr[i_lo_ker:i_hi_ker + 1]))
        n_ker = i_hi_ker - i_lo_ker + 1
        # gg value
        t = lattice_t(n_half, q)
        gg = autoconv_value_pwconst(c, n_half, m, t)
        gg_scaled = gg * 4.0 * n_half * m * m
        kern_val = int(conv[q])
        err = abs(gg_scaled - kern_val)
        max_err = max(max_err, err)
        flag = "  "
        if i_lo_ker != i_lo_lean or i_hi_ker != i_hi_lean:
            flag = "**"
        if W_ker != W_lean or n_ker != n_lean:
            flag = "**"
        print(f"   {flag}{q:2d} |  {i_lo_ker:2d}/{i_lo_lean:<2d} | "
              f"{i_hi_ker:2d}/{i_hi_lean:<2d} | "
              f"{W_ker:>3d}/{W_lean:<3d} | {n_ker:>2d}/{n_lean:<2d} | "
              f"{kern_val:>8d} | {gg_scaled:>10.2f}   (err={err:.3e})")

    print(f"  max |conv[q] - (g*g)(t_q)*(4n*m^2)|  =  {max_err:.6e}")

    # Decisions:
    pyref_pruned, pyref_q, pyref_info = prune_P_ref(c, n_half, m, c_target)
    numba_pruned = call_numba(c, n_half, m, c_target)
    print(f"\n  prune_P (Numba)   pruned: {numba_pruned}")
    print(f"  prune_P (Pyref)   pruned: {pyref_pruned} "
          f"(at q={pyref_q})")
    if numba_pruned != pyref_pruned:
        print("  *** MISMATCH: Numba kernel disagrees with Pure-Python ref! ***")

    # Lean prune (using Lean formulas):
    lean_pruned = False
    lean_q = None
    for q in range(conv_len):
        # Lean has implicit hypothesis q + 2 <= 4*n  ↔  q <= 4n - 2 = conv_len - 1.
        # So all q in [0, conv_len-1] are valid.
        v = lean_pointeval_value(n_half, m, c, q)
        cor = lean_pointeval_correction(n_half, m, c, q)
        if v > c_target + cor:  # strict, matches dynamic_threshold_sound_pointeval
            lean_pruned = True
            lean_q = q
            break
    print(f"  Lean (real-valued, strict >) pruned: {lean_pruned} "
          f"(at q={lean_q})")

    # Truth: max (g*g)
    gg_max, gg_argmax = max_gg_at_lattice(c, n_half, m)
    print(f"  Truth max (g*g) at lattice = {gg_max:.6f} (at q={gg_argmax}, "
          f"t={lattice_t(n_half, gg_argmax):+.6f})")
    if pyref_pruned and gg_max < c_target:
        # Sound prune requires that EVERY h in the cell satisfies max(h*h)>=ct.
        # The piecewise-constant g IS one such h (with eps=0).  So if we
        # pruned but max(g*g) < c_target, the prune is UNSOUND.
        # (More precisely: prune means "for all h in cell, max(h*h) >= ct",
        #  equivalently "this c contributes only solutions >= ct".  If at
        #  the c-itself (eps=0) the max is < ct, the prune is suspect:
        #  we are excluding c, but c with eps=0 produces a g with R(g)<ct.)
        # The C&S correction allows for the cell, so "prune fires" + "max < ct
        # at the centre" is in fact OK iff (g*g)(t_q*) > ct + correction.
        # But it does signal we should double-check the correction term.
        delta = c_target + (2*pyref_info['W_int'] + pyref_info['n_bins'])/(4*n_half*m*m) \
                - (pyref_info['conv_q']/(4*n_half*m*m))
        print(f"  (Note: prune fires while max gg<ct.  ok iff conv_q/(4n m^2) > ct + corr; "
              f"slack = {-delta:+.6e}  meaning: {'OK (slack>0)' if -delta>0 else 'CHECK'})")

    return dict(label=label, c=list(c), n_half=n_half, m=m, c_target=c_target,
                conv=conv.tolist(), pyref_pruned=pyref_pruned,
                pyref_q=pyref_q, numba_pruned=numba_pruned,
                lean_pruned=lean_pruned, lean_q=lean_q,
                gg_max=gg_max, gg_argmax=gg_argmax,
                max_lattice_err=max_err)


def main():
    out = []

    # ---- 1. ALL MASS IN BIN 0 -------------------------------------------
    # c = (S, 0, 0, ..., 0).  conv: only conv[0] = S^2.
    # We'll use n_half=2 (d=4) so S=4nm=8m mass at bin 0.
    n_half, m, ct = 2, 20, 1.20
    d = 2 * n_half
    S = 4 * n_half * m  # 160
    c = [S, 0, 0, 0]    # all mass at bin 0
    out.append(case("CASE 1: All mass in bin 0", c, n_half, m, ct, S_expected=S))

    # ---- 2. ALL MASS IN BIN d-1 -----------------------------------------
    c = [0, 0, 0, S]
    out.append(case("CASE 2: All mass in bin d-1 (mirror of 1)",
                    c, n_half, m, ct, S_expected=S))

    # ---- 3. TWO-BIN SPARSE: (S/2, 0, 0, S/2)  ---------------------------
    n_half, m, ct = 3, 10, 1.20  # d=6
    d = 2 * n_half
    S = 4 * n_half * m  # 120
    c = [S // 2, 0, 0, 0, 0, S // 2]
    out.append(case("CASE 3: Two-bin sparse (S/2 at bin 0 and bin d-1)",
                    c, n_half, m, ct, S_expected=S))

    # ---- 4. UNIFORM: (m, m, ..., m), S = d*m=4nm  -----------------------
    # 4nm = d*m means m_each = 4nm/d = 2m.  Wait, S=4nm and d=2n, so each
    # bin = 4nm/(2n)=2m.  So uniform is c_i = 2m.
    n_half, m, ct = 4, 5, 1.20  # d=8, 2m=10
    d = 2 * n_half
    S = 4 * n_half * m  # 80
    c = [2 * m] * d  # uniform 10,10,10,10,10,10,10,10
    out.append(case("CASE 4: Uniform c_i = 2m (q* should be d-1)",
                    c, n_half, m, ct, S_expected=S))

    # ---- 5. Off-by-one near d-1: just inspect the i_lo, i_hi map ---------
    banner("CASE 5: i_lo/i_hi mapping at q = d-2, d-1, d (off-by-one check)")
    for n_half in [1, 2, 3, 4]:
        d = 2 * n_half
        for q in [d - 2, d - 1, d, d + 1]:
            if q < 0 or q >= 2 * d - 1:
                continue
            i_lo_ker = (q - (d - 1)) if q >= d - 1 else 0
            i_hi_ker = q if q < d else d - 1
            i_lo_lean = lean_i_lo(n_half, q)
            i_hi_lean = lean_i_hi(n_half, q)
            print(f"  n={n_half}, d={d}, q={q}: kernel i_lo={i_lo_ker}, i_hi={i_hi_ker}; "
                  f"Lean i_lo={i_lo_lean}, i_hi={i_hi_lean}; "
                  f"match={i_lo_ker == i_lo_lean and i_hi_ker == i_hi_lean}")

    # ---- 6. EMPTY RANGE? Can i_lo > i_hi ever happen for q in [0, conv_len-1]?
    banner("CASE 6: Empty range (i_lo > i_hi)?")
    for n_half in [1, 2, 3, 4, 5]:
        d = 2 * n_half
        conv_len = 2 * d - 1
        for q in range(conv_len):
            i_lo_ker = (q - (d - 1)) if q >= d - 1 else 0
            i_hi_ker = q if q < d else d - 1
            i_lo_lean = lean_i_lo(n_half, q)
            i_hi_lean = lean_i_hi(n_half, q)
            if i_lo_ker > i_hi_ker:
                print(f"  *** EMPTY in kernel: n={n_half}, d={d}, q={q}: i_lo={i_lo_ker}, i_hi={i_hi_ker}")
            if i_lo_lean > i_hi_lean:
                print(f"  *** EMPTY in Lean: n={n_half}, d={d}, q={q}: i_lo={i_lo_lean}, i_hi={i_hi_lean}")
    print("  (no output above means: no empty range for any valid q.)")

    # ---- 7. ALL-ZERO COMPOSITION ----------------------------------------
    n_half, m, ct = 2, 20, 1.20
    d = 2 * n_half
    c = [0, 0, 0, 0]
    out.append(case("CASE 7: All-zero composition (S=0, not in cascade)",
                    c, n_half, m, ct, S_expected=0))

    # ---- 8. Single nonzero in middle: c = (0, 0, S, 0)  for d=4 -----------
    # Or (0, 0, S, 0, 0, 0) for d=6.  Let's use d=4 with bin index 2.
    n_half, m, ct = 2, 20, 1.20
    d = 2 * n_half
    S = 4 * n_half * m  # 160
    c = [0, 0, S, 0]
    out.append(case("CASE 8: Single nonzero bin 2 (out of 4)",
                    c, n_half, m, ct, S_expected=S))

    # ---- 9. BOUNDARY OF HEIGHT-CELL (c_i = m exactly) ---------------------
    # c_i = m means height = 1, but the kernel works on integers, so this is
    # already lattice-aligned.  More interesting: c_i = m / 2 = 10 (m=20).
    # Test the kernel (no float comparison anywhere in the kernel; everything
    # is int arithmetic on conv[q] vs thr_int).
    n_half, m, ct = 2, 20, 1.20
    d = 2 * n_half
    # Build a c whose conv[q] is RIGHT AT the threshold: see CASE 10.
    # Here we just show that boundary c_i are handled fine.
    c = [m, m * 4, m * 2, m]  # sum = 8m = 160 = S
    out.append(case("CASE 9: Mixed heights, c_i multiples of m boundaries",
                    c, n_half, m, ct, S_expected=S))

    # ---- 10. NEAR-THRESHOLD: conv[q*] exactly equals thr_int -----
    banner("CASE 10: Adversarial near-threshold (test strict > at integer threshold)")
    # Construct c so that conv[q*] equals thr_int exactly; the kernel uses
    # `if conv[q] > thr_int` (strict).  Lean uses `pointeval_value > ct + corr`
    # which is also strict, so should agree.
    # We'll use c_target chosen so the threshold is integer-exact, and tweak
    # c so conv[q*] hits it.
    n_half, m = 2, 20
    d = 2 * n_half  # 4
    S = 4 * n_half * m  # 160
    # Pick c = (60, 20, 20, 60). conv[3] = 2*(60*60 + 20*20) = 7200+800=8000.
    c = [60, 20, 20, 60]
    print(f"  c={c}, S={sum(c)}")
    cv = conv_array(c, n_half)
    print(f"  conv = {cv.tolist()}")
    # At q=3 (center), conv[3] = ?  With i+j=3: pairs (0,3),(3,0),(1,2),(2,1)
    # so kernel ordered-pair: 2*(60*60 + 20*20) = 2*(3600+400)=8000.
    print(f"  conv[3] = {cv[3]}")
    # Now find c_target s.t. base + 2*W_int + n_bins == 8000 - 1, then set ct to that.
    # i_lo(q=3) = 0 (since q=3 = d-1, q+1=d so q+1<=d), i_hi(q=3) = 3.
    # W_int(q=3) = sum c = S = 160.  n_bins = 4.
    # thr = 2*d*ct*m^2 + 2*W + n_bins + eps = 2*4*ct*400 + 320 + 4 + eps
    #     = 3200*ct + 324 + eps_margin.
    # Want thr_int = 7999 to exactly match conv[3]-1.
    # 3200*ct = 7999 - 324 = 7675   →  ct = 7675/3200 = 2.39843...
    # Then conv[3]=8000 > thr_int=7999 → fires.
    # But cs_constant > 1.5 doesn't make sense for the project.  Let's try a
    # case where the threshold is exactly 8000 (not strict): conv[q]>thr_int
    # is FALSE.  ct = (8000-324)/3200 = 7676/3200 = 2.39875.  Still huge.
    # Better: smaller m, bigger relative.
    n_half, m = 1, 5  # d=2
    d = 2  # 2 bins: c=(c0, c1), S=4nm=20.
    # conv: conv[0] = c0^2, conv[1] = 2*c0*c1, conv[2] = c1^2.
    # i_lo(q=0)=0, i_hi=0 (q<d=2 so i_hi=q=0 - wait, q=0<2 so i_hi=q=0).
    #   W_int(0)=c0, n_bins(0)=1. thr = 2*2*ct*25 + 2*c0 + 1 = 100*ct + 2*c0 + 1.
    # Pick c=(20,0):
    #   conv[0]=400.  thr = 100*ct + 40 + 1 = 100*ct + 41.
    #   conv[0]>thr iff 400 > 100*ct+41 iff ct < 3.59.
    # That's > 1.28 always.  Use ct=3.50.  Then thr = 350+41 = 391.  conv=400>391 fires.
    # ct=3.59 gives thr=359+41=400 → 400 > 400 is FALSE (no prune).
    # ct=3.5901 gives thr=359.01+41=400.01 → conv>thr_int=400 → 400>400=FALSE.
    # ct=3.58 gives thr=358+41=399 → 400>399 fires.
    # Right at: ct = 359/100 = 3.59.  thr_int = int(400 + eps_margin*25) = 400. Strict > fails.
    c = [20, 0]
    n_half, m, ct = 1, 5, 3.59
    print(f"\n  Sub-case 10a: c={c}, n_half={n_half}, m={m}, ct={ct}")
    out.append(case("CASE 10a: conv[q] == thr_int (strict > should NOT fire)",
                    c, n_half, m, ct, S_expected=20))
    # Now bump ct slightly down so we DO fire:
    c = [20, 0]
    ct = 3.589
    out.append(case("CASE 10b: conv[q] > thr_int (strict > SHOULD fire)",
                    c, n_half, m, ct, S_expected=20))

    # ---- 11. MAXIMUM-S: deepest cascade level ----------------------------
    banner("CASE 11: Maximum-S overflow check")
    # Deep cascade: n_half=8 (d=16), m=20, S=4*8*20=640.
    # conv[q] up to 2*S^2 ~ 2*640^2 = 819200; well within int64.
    # But if c_i = S (all in one bin), conv[2i] = S^2 = 409600.
    # Let's check the most extreme: max possible conv[q] for n_half=12, m=64
    # (deepest cascade).  S = 4*12*64 = 3072.  S^2 = 9.4M.  int64 ample.
    n_half, m = 12, 64
    d = 2 * n_half
    S = 4 * n_half * m  # 3072
    c_max = [S] + [0] * (d - 1)
    cv_max = conv_array(c_max, n_half)
    print(f"  Max-mass-bin0: c[0]=S={S}, conv[0]=S^2={cv_max[0]}")
    print(f"  int64 max = {2**63-1}, S^2 = {S*S}, ratio = {(S*S)/(2**63-1):.2e}")
    # Also check uniform deep:
    c_unif = [2 * m] * d  # each c_i = 2m = 128
    cv_unif = conv_array(c_unif, n_half)
    print(f"  Uniform-deep: c_i=2m={2*m}, conv[d-1] = {cv_unif[d-1]} (expected = d*(2m)^2 = {d*(2*m)**2})")
    print(f"  All conv values fit in int64? {cv_unif.max() < 2**62}")
    out.append(dict(label="CASE 11: max-S overflow check",
                    n_half=n_half, m=m, S=S, max_conv=int(cv_max[0]),
                    int64_safe=(int(cv_max[0]) < 2**62)))

    # Persist:
    with open(os.path.join(_HERE, "_agent9_edge_audit.json"), 'w') as fp:
        json.dump([{k: (v if not isinstance(v, np.ndarray) else v.tolist())
                    for k, v in d.items()} for d in out], fp,
                  indent=2, default=str)
    banner("DONE")


if __name__ == '__main__':
    main()
