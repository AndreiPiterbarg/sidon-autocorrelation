"""
Definitive verification of threshold formula soundness.

The C&S algorithm approximates continuous functions by piecewise-constant
functions on the fine grid B_{n,m}: heights h_i = c_i/m (multiples of 1/m),
summing to 4n. Lemma 3: ||(g*g)(x) - (f*f)(x)|| <= 2/m + 1/m^2.

This script verifies:
1. Whether the knot-point test value equals the integral test value
   (i.e., whether the Python code computes the right physical quantity)
2. Whether the flat Lemma 3 correction (2/m + 1/m^2) bounds the
   per-window discretization error
3. Whether the W-refined correction is also valid
4. Whether the MATLAB formula (Formula B) is valid
"""
import numpy as np
from math import comb
import sys


# =====================================================================
# Exact autoconvolution for piecewise-constant functions
# =====================================================================

def exact_autoconv_at_knots(heights, bin_width):
    """Compute (f*f)(x) at all knot points x = k*bin_width for k=0..2d.

    For piecewise-constant f with d bins of width bin_width:
    (f*f)(k*bw) = bw * sum_{i+j=k, valid} h_i * h_j
    where i,j in {0,...,d-1} and the integral contribution from pair (i,j)
    at knot k is bw if both bins overlap with position k*bw.

    Actually, (f*f)(x) = integral f(t) f(x-t) dt.
    For piecewise constant f with bins [i*bw, (i+1)*bw), height h_i:
    The integral involves all pairs of bins (i,j).
    At position x, the overlap of bin i (in t) and bin j (in x-t) is:
    t in [i*bw, (i+1)*bw) AND x-t in [j*bw, (j+1)*bw)
    => t in [x-(j+1)*bw, x-j*bw)
    => overlap = max(0, min((i+1)*bw, x-j*bw) - max(i*bw, x-(j+1)*bw))

    At knot x = (i+j+1)*bw: overlap = bw (full overlap)
    At knot x = (i+j)*bw: overlap = 0 (just touching)
    At knot x = (i+j+2)*bw: overlap = 0 (just touching)

    So (f*f)((i+j+1)*bw) gets a contribution of bw * h_i * h_j.
    """
    d = len(heights)
    n_knots = 2 * d + 1  # knots at 0, bw, 2*bw, ..., 2d*bw
    ff = np.zeros(n_knots)

    for k in range(n_knots):
        x = k * bin_width
        for i in range(d):
            for j in range(d):
                t_lo = max(i * bin_width, x - (j + 1) * bin_width)
                t_hi = min((i + 1) * bin_width, x - j * bin_width)
                if t_hi > t_lo + 1e-15:
                    ff[k] += heights[i] * heights[j] * (t_hi - t_lo)

    return ff


def exact_ff_max(heights, bin_width):
    """Max of (f*f) for piecewise-constant f.
    Since (f*f) is piecewise linear, max occurs at a knot point."""
    ff = exact_autoconv_at_knots(heights, bin_width)
    return np.max(ff)


# =====================================================================
# Test values: knot-point based (Python code) vs integral based
# =====================================================================

def knot_point_conv(heights):
    """Discrete autoconvolution: conv[k] = sum_{i+j=k} h_i * h_j."""
    d = len(heights)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len)
    for i in range(d):
        for j in range(d):
            conv[i + j] += heights[i] * heights[j]
    return conv


def tv_knot_point(heights, n_half, ell, s_lo):
    """Test value at knot points: TV = sum(conv[s_lo:s_lo+ell-1]) / (4n*ell).

    This is what the Python code computes (test_values.py).
    """
    conv = knot_point_conv(heights)
    n_cv = ell - 1
    ws = sum(conv[s_lo:s_lo + n_cv])
    return ws / (4.0 * n_half * ell)


def tv_integral(heights, bin_width, ell, s_lo, d):
    """Test value via exact integral of (f*f) over window.

    TV = (1/(ell*bw)) * integral_{s_lo*bw}^{(s_lo+ell)*bw} (f*f)(x) dx

    Since (f*f) is piecewise linear, use trapezoid rule (exact).
    """
    ff_knots = exact_autoconv_at_knots(heights, bin_width)
    # Window covers knot indices s_lo to s_lo + ell
    if s_lo + ell > 2 * d:
        return None

    integral = 0.0
    for k in range(s_lo, s_lo + ell):
        integral += (ff_knots[k] + ff_knots[k + 1]) / 2 * bin_width

    return integral / (ell * bin_width)


# =====================================================================
# Cumulative-floor discretization (on heights, for fine grid)
# =====================================================================

def cumul_floor_fine(h, m, n_half):
    """Discretize height vector h to fine-grid integer composition c.

    h: (d,) array of heights, sum = 4n
    m: grid resolution (heights are rounded to multiples of 1/m)
    Returns: (d,) array of integers c, sum = 4nm, where h_i^disc = c_i/m
    """
    d = len(h)
    H = np.zeros(d + 1)  # cumulative heights
    for i in range(d):
        H[i + 1] = H[i] + h[i]
    D = np.floor(m * H).astype(int)
    c = np.zeros(d, dtype=int)
    for i in range(d - 1):
        c[i] = D[i + 1] - D[i]
    S = int(4 * n_half * m)
    c[d - 1] = S - D[d - 1]
    return c


def sample_preimages_fine(c_int, m, n_half, n_samples=5000, rng=None):
    """Sample height vectors h (sum = 4n) that discretize to c under cumul_floor.

    c_int: integer composition on fine grid, sum = 4nm
    Returns: list of height vectors h with sum = 4n
    """
    if rng is None:
        rng = np.random.default_rng(42)
    d = len(c_int)
    S = int(4 * n_half * m)
    total_h = 4.0 * n_half

    # D(k) = sum(c[0:k])
    D = np.zeros(d + 1, dtype=int)
    for k in range(d):
        D[k + 1] = D[k] + c_int[k]

    # For cumulative-floor, H(k) must satisfy floor(m * H(k)) = D(k)
    # So H(k) in [D(k)/m, (D(k)+1)/m) for k=1..d-1
    # H(0) = 0, H(d) = 4n

    results = []
    for _ in range(n_samples):
        H = np.zeros(d + 1)
        H[0] = 0.0
        H[d] = total_h

        valid = True
        for k in range(1, d):
            lo = D[k] / m
            hi = (D[k] + 1) / m
            # Need H(k) > H(k-1) and H(k) < H(d) eventually
            lo = max(lo, H[k - 1] + 1e-14)
            hi = min(hi, total_h - 1e-14)

            if lo >= hi - 1e-14:
                valid = False
                break
            H[k] = rng.uniform(lo, hi - 1e-14)

        if not valid:
            continue

        h = np.diff(H)
        if np.any(h < -1e-12):
            continue

        h = np.maximum(h, 0)
        # Renormalize to sum exactly 4n
        h = h * (total_h / h.sum())

        # Verify
        c_check = cumul_floor_fine(h, m, n_half)
        if np.array_equal(c_check, c_int):
            results.append(h)

    return results


def sample_preimages_coarse(c_int, m, n_half, n_samples=5000, rng=None):
    """Sample mass vectors mu (sum = 1) that discretize to c under cumul_floor.

    For the coarse grid: c_i are mass quanta, sum = m.
    Heights h_i = c_i * 4n/m.
    The pre-image is mu_i (masses, sum = 1). Heights = mu_i / bin_width = mu_i * 2d = mu_i * 4n.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    d = len(c_int)

    # D(k) = sum(c[0:k]) for the mass quanta
    D = np.zeros(d + 1, dtype=int)
    for k in range(d):
        D[k + 1] = D[k] + c_int[k]

    # M(k) = cumulative mass, M(k) in [D(k)/m, (D(k)+1)/m)
    results = []
    for _ in range(n_samples):
        M = np.zeros(d + 1)
        M[0] = 0.0
        M[d] = 1.0

        valid = True
        for k in range(1, d):
            lo = D[k] / m
            hi = (D[k] + 1) / m
            lo = max(lo, M[k - 1] + 1e-14)
            hi = min(hi, 1.0 - 1e-14)

            if lo >= hi - 1e-14:
                valid = False
                break
            M[k] = rng.uniform(lo, hi - 1e-14)

        if not valid:
            continue

        mu = np.diff(M)
        if np.any(mu < -1e-12):
            continue
        mu = np.maximum(mu, 0)
        mu = mu / mu.sum()

        # Heights from masses
        bin_width = 1.0 / (4 * n_half)
        h = mu / bin_width  # h_i = mu_i / Delta

        # Verify: discretize h back
        c_check = cumul_floor_fine(h, m, n_half)
        # For coarse grid, c_int has sum = m. c_check has sum = 4nm.
        # The coarse composition maps to fine composition c_fine_i = 4n * c_coarse_i
        c_fine_expected = c_int * int(4 * n_half)
        if np.array_equal(c_check, c_fine_expected):
            results.append(h)

    return results


# =====================================================================
# Test 1: Knot-point TV vs Integral TV — are they the same?
# =====================================================================

def test_knot_vs_integral():
    """Check if the Python code's knot-point TV matches the integral TV."""
    print("=" * 70)
    print("TEST 1: Knot-point TV vs Integral TV")
    print("  Are these the same for the discrete function itself?")
    print("=" * 70)

    n_half = 2
    d = 4
    m = 10
    bin_width = 1.0 / (2 * d)  # = 1/(4n)

    # Fine grid composition: c sums to 4nm = 80
    c = np.array([20, 10, 10, 40])
    h = c / float(m)  # heights: [2, 1, 1, 4], sum = 8 = 4n [ok]
    print(f"\nc = {c}, heights h = c/m = {h}")
    print(f"Sum(h) = {sum(h)}, 4n = {4*n_half}")

    # Compute knot-point autoconvolution
    conv = knot_point_conv(h)
    print(f"\nconv[k] = sum_{{i+j=k}} h_i*h_j:")
    for k, v in enumerate(conv):
        print(f"  conv[{k}] = {v:.4f}")

    # Compute exact (f*f) at knots
    ff = exact_autoconv_at_knots(h, bin_width)
    print(f"\n(f*f)(k*bw) at knot points (bw={bin_width}):")
    for k, v in enumerate(ff):
        print(f"  k={k}: (f*f)({k*bin_width:.4f}) = {v:.6f}")

    # The relationship: (f*f) at knot (k+1)*bw should involve conv[k]
    # For piecewise const f, at interior knot x=(k+1)*bw:
    # (f*f)(x) = bw * conv[k] = conv[k]/(4n)  [since bw = 1/(4n)]
    print(f"\nRelationship: (f*f)((k+1)*bw) vs bw*conv[k] = conv[k]/(4n):")
    for k in range(2*d - 1):
        expected = bin_width * conv[k]
        actual = ff[k + 1]
        print(f"  k={k}: conv[k]/(4n) = {expected:.6f}, "
              f"(f*f)({(k+1)*bin_width:.4f}) = {actual:.6f}, "
              f"diff = {actual - expected:.2e}")

    # Compare test values
    print(f"\n{'ell':>4} {'s_lo':>4} {'TV_knot':>10} {'TV_integ':>10} {'ratio':>8}")
    print("-" * 50)
    max_ratio = 0
    for ell in range(2, 2*d + 1):
        n_cv = ell - 1
        conv_len = 2*d - 1
        for s_lo in range(conv_len - n_cv + 1):
            tv_k = tv_knot_point(h, n_half, ell, s_lo)
            tv_i = tv_integral(h, bin_width, ell, s_lo, d)
            if tv_i is not None and tv_i > 1e-10:
                ratio = tv_k / tv_i
                if abs(ratio - 1) > 1e-10:
                    print(f"{ell:4d} {s_lo:4d} {tv_k:10.6f} {tv_i:10.6f} {ratio:8.4f}")
                    max_ratio = max(max_ratio, abs(ratio - 1))

    if max_ratio < 1e-10:
        print("ALL MATCH — knot-point TV equals integral TV!")
    else:
        print(f"\nMax ratio deviation: {max_ratio:.6f}")
        print("KNOT-POINT TV != INTEGRAL TV — they compute DIFFERENT quantities!")


# =====================================================================
# Test 2: Lemma 3 pointwise bound verification
# =====================================================================

def test_lemma3_pointwise():
    """Verify Lemma 3: (g*g)(x) <= (f*f)(x) + 2/m + 1/m^2 for all x.

    g = fine-grid approximation, f = continuous pre-image.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Lemma 3 pointwise bound")
    print("  Does (g*g)(x) - (f*f)(x) <= 2/m + 1/m^2 at all knots?")
    print("=" * 70)

    n_half = 2
    d = 4
    m = 10
    S = int(4 * n_half * m)  # 80
    bin_width = 1.0 / (2 * d)
    correction = 2.0 / m + 1.0 / (m * m)

    print(f"\nd={d}, n_half={n_half}, m={m}, bin_width={bin_width}")
    print(f"Lemma 3 correction = 2/m + 1/m^2 = {correction:.6f}")

    # Test several compositions
    test_comps = [
        np.array([20, 10, 10, 40]),
        np.array([40, 20, 10, 10]),
        np.array([5, 5, 5, 65]),
        np.array([60, 10, 5, 5]),
        np.array([20, 20, 20, 20]),
        np.array([1, 1, 1, 77]),     # extreme concentration
        np.array([76, 2, 1, 1]),     # extreme left
    ]

    max_violation = 0
    for c in test_comps:
        if sum(c) != S:
            continue
        h_disc = c / float(m)  # discrete heights (multiples of 1/m)

        preimages = sample_preimages_fine(c, m, n_half, n_samples=2000)
        if len(preimages) == 0:
            print(f"\nc = {c}: no pre-images found")
            continue

        # Compute (g*g) at knots
        ff_disc = exact_autoconv_at_knots(h_disc, bin_width)

        worst_excess = -float('inf')
        for h_cont in preimages:
            ff_cont = exact_autoconv_at_knots(h_cont, bin_width)
            # Check: (g*g)(x_k) - (f*f)(x_k) <= 2/m + 1/m^2
            excess = np.max(ff_disc - ff_cont) - correction
            if excess > worst_excess:
                worst_excess = excess

        if worst_excess > max_violation:
            max_violation = worst_excess

        status = "VIOLATION" if worst_excess > 1e-10 else "OK"
        print(f"c = {c}: max excess over correction = {worst_excess:.8f}  [{status}]"
              f" ({len(preimages)} pre-images)")

    if max_violation > 1e-10:
        print(f"\n*** LEMMA 3 VIOLATED! Max excess = {max_violation:.8f} ***")
    else:
        print(f"\nLemma 3 holds for all tested compositions.")


# =====================================================================
# Test 3: Per-window TV error bound (flat Lemma 3 on integral TV)
# =====================================================================

def test_per_window_integral_tv():
    """If TV is the integral test value (what Lemma 3 bounds), does
    TV_disc - TV_cont <= 2/m + 1/m^2 hold per-window?

    Since Lemma 3 is pointwise: (g*g)(x) <= (f*f)(x) + corr for ALL x,
    averaging over any window should preserve the bound.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Per-window bound on INTEGRAL test value")
    print("  Does Lemma 3 imply TV_disc(integral) - TV_cont <= 2/m + 1/m^2?")
    print("=" * 70)

    n_half = 2
    d = 4
    m = 10
    S = int(4 * n_half * m)
    bin_width = 1.0 / (2 * d)
    correction = 2.0 / m + 1.0 / (m * m)

    test_comps = [
        np.array([20, 10, 10, 40]),
        np.array([5, 5, 5, 65]),
        np.array([1, 1, 1, 77]),
        np.array([40, 20, 10, 10]),
    ]

    max_violation = -float('inf')
    for c in test_comps:
        if sum(c) != S:
            continue
        h_disc = c / float(m)
        preimages = sample_preimages_fine(c, m, n_half, n_samples=2000)
        if not preimages:
            continue

        worst = -float('inf')
        for h_cont in preimages:
            for ell in range(2, 2*d + 1):
                for s_lo in range(2*d - ell + 1):
                    tv_d = tv_integral(h_disc, bin_width, ell, s_lo, d)
                    tv_c = tv_integral(h_cont, bin_width, ell, s_lo, d)
                    if tv_d is not None and tv_c is not None:
                        err = tv_d - tv_c
                        if err > worst:
                            worst = err

        status = "VIOLATION" if worst > correction + 1e-10 else "OK"
        print(f"c = {c}: max TV_integ error = {worst:.8f}, "
              f"correction = {correction:.6f}  [{status}]")
        if worst > max_violation:
            max_violation = worst

    if max_violation > correction + 1e-10:
        print(f"\n*** INTEGRAL TV exceeds Lemma 3 bound! ***")
    else:
        print(f"\nIntegral TV bound holds (as expected from Lemma 3).")


# =====================================================================
# Test 4: Per-window error on KNOT-POINT TV (what Python code computes)
# =====================================================================

def test_per_window_knot_tv():
    """Does TV_disc(knot) - TV_cont(knot) <= 2/m + 1/m^2?

    The Python code uses knot-point TV. If knot-point TV != integral TV
    for the continuous function, the Lemma 3 bound might not apply.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Per-window bound on KNOT-POINT test value")
    print("  Does TV_disc(knot) - TV_cont(knot) <= 2/m + 1/m^2?")
    print("=" * 70)

    n_half = 2
    d = 4
    m = 10
    S = int(4 * n_half * m)
    bin_width = 1.0 / (2 * d)
    correction = 2.0 / m + 1.0 / (m * m)

    test_comps = [
        np.array([20, 10, 10, 40]),
        np.array([5, 5, 5, 65]),
        np.array([1, 1, 1, 77]),
        np.array([40, 20, 10, 10]),
        np.array([60, 10, 5, 5]),
    ]

    max_violation = -float('inf')
    for c in test_comps:
        if sum(c) != S:
            continue
        h_disc = c / float(m)

        preimages = sample_preimages_fine(c, m, n_half, n_samples=2000)
        if not preimages:
            print(f"c = {c}: no pre-images")
            continue

        worst = -float('inf')
        worst_info = None
        conv_len = 2*d - 1

        for h_cont in preimages:
            conv_disc = knot_point_conv(h_disc)
            conv_cont = knot_point_conv(h_cont)

            for ell in range(2, 2*d + 1):
                n_cv = ell - 1
                for s_lo in range(conv_len - n_cv + 1):
                    ws_disc = sum(conv_disc[s_lo:s_lo + n_cv])
                    ws_cont = sum(conv_cont[s_lo:s_lo + n_cv])
                    tv_d = ws_disc / (4.0 * n_half * ell)
                    tv_c = ws_cont / (4.0 * n_half * ell)
                    err = tv_d - tv_c
                    if err > worst:
                        worst = err
                        worst_info = (ell, s_lo, h_cont)

        status = "VIOLATION" if worst > correction + 1e-10 else "OK"
        print(f"c = {c}: max knot-TV error = {worst:.8f}, "
              f"correction = {correction:.6f}  [{status}]")
        if worst_info:
            ell, s_lo, _ = worst_info
            print(f"  Worst at (ell={ell}, s_lo={s_lo})")
        if worst > max_violation:
            max_violation = worst

    if max_violation > correction + 1e-10:
        print(f"\n*** KNOT TV EXCEEDS LEMMA 3 BOUND! ***")
        print("  The Python code computes knot-point TV, which may not be")
        print("  bounded by the flat Lemma 3 correction.")
    else:
        print(f"\nKnot-point TV also bounded by Lemma 3 correction.")


# =====================================================================
# Test 5: Does pruning a composition guarantee ||f*f||_inf >= c_target?
# =====================================================================

def test_pruning_soundness_fine():
    """For fine-grid compositions: if TV_disc > c_target + correction,
    does ||f*f||_inf >= c_target for ALL f mapping to the composition?

    This is the ULTIMATE soundness check.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Pruning soundness (fine grid)")
    print("  If pruned, is ||f*f||_inf >= c_target for all pre-images?")
    print("=" * 70)

    n_half = 2
    d = 4
    m = 5  # small m for many pre-images
    S = int(4 * n_half * m)  # = 40
    bin_width = 1.0 / (2 * d)
    c_target = 1.3
    flat_corr = 2.0 / m + 1.0 / (m * m)

    print(f"d={d}, n_half={n_half}, m={m}, S={S}")
    print(f"c_target={c_target}, flat correction={flat_corr:.4f}")
    print(f"Total compositions: {comb(S + d - 1, d - 1)}")

    n_checked = 0
    n_pruned_flat = 0
    n_unsound = 0

    # Generate all compositions of d bins summing to S
    def gen_comps(d, S):
        if d == 1:
            yield [S]
            return
        for c0 in range(S + 1):
            for rest in gen_comps(d - 1, S - c0):
                yield [c0] + rest

    for c_list in gen_comps(d, S):
        c = np.array(c_list)
        n_checked += 1

        # Compute max knot-point TV
        h = c / float(m)
        conv = knot_point_conv(h)
        conv_len = 2*d - 1

        best_tv = 0
        for ell in range(2, 2*d + 1):
            n_cv = ell - 1
            for s_lo in range(conv_len - n_cv + 1):
                ws = sum(conv[s_lo:s_lo + n_cv])
                tv = ws / (4.0 * n_half * ell)
                best_tv = max(best_tv, tv)

        if best_tv <= c_target + flat_corr:
            continue

        n_pruned_flat += 1

        # Check soundness: all pre-images must have ||f*f||_inf >= c_target
        preimages = sample_preimages_fine(c, m, n_half, n_samples=1000)
        if not preimages:
            continue

        for h_cont in preimages:
            ff_inf = exact_ff_max(h_cont, bin_width)
            if ff_inf < c_target - 1e-8:
                n_unsound += 1
                print(f"  *** UNSOUND: c={c}, ||f*f||_inf={ff_inf:.6f} < {c_target}")
                break

        if n_checked % 5000 == 0:
            print(f"  ... checked {n_checked} compositions, {n_pruned_flat} pruned")

    print(f"\nResults:")
    print(f"  Total compositions checked: {n_checked}")
    print(f"  Pruned by flat threshold: {n_pruned_flat}")
    print(f"  UNSOUND prunings: {n_unsound}")
    if n_unsound == 0:
        print("  ALL prunings are SOUND!")


# =====================================================================
# Test 6: Coarse grid — exhaustive soundness check
# =====================================================================

def test_coarse_grid_exhaustive():
    """Exhaustive check of coarse-grid compositions with MATLAB threshold."""
    print("\n" + "=" * 70)
    print("TEST 6: Coarse grid (MATLAB-style) exhaustive soundness")
    print("=" * 70)

    n_half = 2
    d = 4
    m = 10
    S_coarse = m
    bin_width = 1.0 / (2 * d)
    c_target = 1.3

    flat_corr = 2.0 / m + 1.0 / (m * m)

    print(f"d={d}, n_half={n_half}, m={m}")
    print(f"Coarse grid: S={S_coarse}, total compositions: {comb(S_coarse+d-1,d-1)}")
    print(f"c_target={c_target}")
    print(f"flat correction (2/m+1/m^2) = {flat_corr:.4f}")

    # On the coarse grid: c_i are mass quanta (sum = m)
    # Heights: h_i = c_i * 4n/m
    # The correction should use height resolution = 4n/m, giving:
    #   2*(4n/m) + (4n/m)^2 = 8n/m + 16n^2/m^2
    coarse_corr = 8*n_half/m + 16*n_half**2/(m*m)
    print(f"Correct coarse correction (8n/m + 16n^2/m^2) = {coarse_corr:.4f}")
    print(f"MATLAB correction (1/m^2 + 2W/m) ~= {1/(m*m) + 2*0.5/m:.4f} (W=0.5)")

    def gen_comps(d, S):
        if d == 1:
            yield [S]
            return
        for c0 in range(S + 1):
            for rest in gen_comps(d - 1, S - c0):
                yield [c0] + rest

    n_total = 0
    n_pruned_matlab = 0
    n_pruned_flat = 0
    n_pruned_coarse = 0
    unsound_matlab = []
    unsound_flat = []

    for c_list in gen_comps(d, S_coarse):
        c_coarse = np.array(c_list)
        n_total += 1

        # Heights on coarse grid
        h = c_coarse * (4.0 * n_half / m)
        conv = knot_point_conv(h)
        conv_len = 2*d - 1

        # Check all windows with different corrections
        pruned_matlab = False
        pruned_flat = False
        pruned_coarse = False

        prefix_c = np.zeros(d + 1)
        for i in range(d):
            prefix_c[i+1] = prefix_c[i] + c_coarse[i]

        for ell in range(2, 2*d + 1):
            n_cv = ell - 1
            for s_lo in range(conv_len - n_cv + 1):
                ws = sum(conv[s_lo:s_lo + n_cv])
                tv = ws / (4.0 * n_half * ell)

                # MATLAB formula: c_target + 1/m^2 + 2*W_mass/m
                lo_bin = max(0, s_lo - (d-1))
                hi_bin = min(d-1, s_lo + ell - 2)
                W_int = int(prefix_c[hi_bin+1] - prefix_c[lo_bin])
                W_mass = W_int / float(m)
                matlab_corr = 1/(m*m) + 2*W_mass/m

                if tv > c_target + matlab_corr:
                    pruned_matlab = True
                if tv > c_target + flat_corr:
                    pruned_flat = True
                if tv > c_target + coarse_corr:
                    pruned_coarse = True

        if pruned_matlab:
            n_pruned_matlab += 1
        if pruned_flat:
            n_pruned_flat += 1
        if pruned_coarse:
            n_pruned_coarse += 1

        # For pruned compositions, check soundness via pre-images
        if pruned_matlab:
            preimages = sample_preimages_coarse(c_coarse, m, n_half, n_samples=2000)
            if preimages:
                for h_cont in preimages:
                    ff_inf = exact_ff_max(h_cont, bin_width)
                    if ff_inf < c_target - 1e-8:
                        unsound_matlab.append((c_coarse.copy(), ff_inf))
                        break

        if pruned_flat and not pruned_matlab:
            preimages = sample_preimages_coarse(c_coarse, m, n_half, n_samples=2000)
            if preimages:
                for h_cont in preimages:
                    ff_inf = exact_ff_max(h_cont, bin_width)
                    if ff_inf < c_target - 1e-8:
                        unsound_flat.append((c_coarse.copy(), ff_inf))
                        break

    print(f"\nResults:")
    print(f"  Total compositions: {n_total}")
    print(f"  Pruned by MATLAB formula: {n_pruned_matlab}")
    print(f"  Pruned by flat (2/m+1/m^2): {n_pruned_flat}")
    print(f"  Pruned by coarse corr (8n/m+16n^2/m^2): {n_pruned_coarse}")
    print(f"  Unsound MATLAB: {len(unsound_matlab)}")
    print(f"  Unsound flat: {len(unsound_flat)}")

    if unsound_matlab:
        print("\n  *** UNSOUND MATLAB PRUNINGS: ***")
        for c, ff_inf in unsound_matlab[:10]:
            print(f"    c={c}, ||f*f||_inf={ff_inf:.6f} < c_target={c_target}")
    else:
        print("  MATLAB pruning: ALL SOUND")


# =====================================================================
# Test 7: What does MATLAB actually compute as test value?
# =====================================================================

def test_matlab_tv_reconstruction():
    """Reconstruct what the MATLAB code computes and compare with Python."""
    print("\n" + "=" * 70)
    print("TEST 7: MATLAB test value reconstruction")
    print("=" * 70)

    d = 4
    n_half = 2
    m = 10
    bin_width = 1.0 / (2 * d)

    # Coarse-grid composition: mass quanta summing to m
    c_coarse = np.array([2, 1, 1, 6])
    v = c_coarse / float(m)  # mass values (what MATLAB stores)
    h = c_coarse * (4.0 * n_half / m)  # heights

    print(f"c_coarse = {c_coarse}")
    print(f"Mass values v = c/m = {v}")
    print(f"Heights h = c*4n/m = {h}")

    # MATLAB computes: TV = (2*d/ell) * sum v_a * v_b for contributing pairs
    # Python computes: TV = sum conv[k] / (4n*ell) where conv uses heights h

    # Let's verify these are the same.
    conv_h = knot_point_conv(h)
    conv_v = knot_point_conv(v)
    print(f"\nconv (heights): {conv_h}")
    print(f"conv (masses):  {conv_v}")
    print(f"ratio conv_h/conv_v: {conv_h / (conv_v + 1e-20)}")
    print(f"Expected ratio: (4n)^2 = {(4*n_half)**2}")

    # For ell=2:
    for ell in [2, 3, 4, 6]:
        for s_lo in [0, 3, 6]:
            n_cv = ell - 1
            if s_lo + n_cv > 2*d - 1:
                continue
            # Python: TV = sum(conv_h[s_lo:s_lo+n_cv]) / (4n*ell)
            ws_h = sum(conv_h[s_lo:s_lo + n_cv])
            tv_py = ws_h / (4.0 * n_half * ell)

            # MATLAB: TV = (2d/ell) * sum(conv_v[s_lo:s_lo+n_cv])
            # But MATLAB uses a different convolution structure!
            # MATLAB's sumIndicesStore has pairs contributing to windows
            # where BOTH conv bins of the pair are within the window.
            # This is different from Python's knot-point conv.

            # For a simple comparison:
            ws_v = sum(conv_v[s_lo:s_lo + n_cv])
            tv_matlab_simple = (2 * d / ell) * ws_v

            # Check if they're proportional
            if tv_py > 1e-10:
                ratio = tv_matlab_simple / tv_py
            else:
                ratio = float('nan')

            print(f"  ell={ell}, s={s_lo}: TV_py={tv_py:.6f}, "
                  f"MATLAB_simple={tv_matlab_simple:.6f}, ratio={ratio:.4f}")

    # The test values should match if the MATLAB normalization is consistent
    # TV_py = sum conv_h / (4n*ell) = sum (4n)^2 * conv_v / (4n*ell) = 4n * sum conv_v / ell
    # TV_matlab = (2d/ell) * sum conv_v = (4n/ell) * sum conv_v
    # These are the same! TV_py = 4n * sum conv_v / ell... wait no.
    # TV_py = sum conv_h / (4n*ell) = (4n)^2 * sum conv_v / (4n*ell) = 4n * sum conv_v / ell
    # TV_matlab = (2d/ell) * sum conv_v = (4n/ell) * sum conv_v
    # Ratio: TV_py / TV_matlab = (4n * sum conv_v / ell) / (4n * sum conv_v / ell) = 1
    # So they ARE the same! The factor (4n)^2 in conv_h is canceled by the 4n in the denominator.
    print("\n  -> TV_py and MATLAB_simple should match (both = 4n * sum_conv_mass / ell)")


# =====================================================================
# Test 8: The key relationship — what does Lemma 3 actually bound?
# =====================================================================

def test_lemma3_interpretation():
    """Verify: Lemma 3 bounds (g*g)(x) pointwise. Since the Python code's
    knot-point TV is computed from conv[k] = sum h_i*h_j, and
    (g*g)((k+1)*bw) = bw * conv[k], the TV is:

    TV = (1/ell) * sum_{k in window} (g*g)((k+1)*bw) / bw

    Wait, that doesn't simplify nicely. Let me verify the exact relationship.
    """
    print("\n" + "=" * 70)
    print("TEST 8: What the Python TV actually computes physically")
    print("=" * 70)

    d = 4
    n_half = 2
    m = 10
    bin_width = 1.0 / (2*d)  # = 1/(4n) = 0.125

    c = np.array([20, 10, 10, 40])
    h = c / float(m)  # [2, 1, 1, 4]

    conv = knot_point_conv(h)
    ff = exact_autoconv_at_knots(h, bin_width)

    print(f"h = {h}, bin_width = {bin_width}")
    print(f"\nKnot index | position | (f*f) at knot | bw*conv[k-1]")
    print("-" * 60)
    for k in range(2*d + 1):
        pos = k * bin_width
        ff_val = ff[k]
        if k > 0 and k <= 2*d - 1:
            bw_conv = bin_width * conv[k-1]
            print(f"  k={k:2d} | x={pos:.4f} | (f*f)={ff_val:.6f} | bw*conv[{k-1}]={bw_conv:.6f}")
        else:
            print(f"  k={k:2d} | x={pos:.4f} | (f*f)={ff_val:.6f} | (boundary)")

    # The Python test value:
    # TV = sum_{k=s_lo}^{s_lo+ell-2} conv[k] / (4n*ell)
    # = sum (f*f)(k*bw)/bw for knots k=s_lo+1..s_lo+ell-1... no

    # Actually:
    # conv[k] = sum_{i+j=k} h_i*h_j
    # (f*f)((k+1)*bw) = bw * sum_{i+j=k} h_i*h_j  [for interior knots]
    # Wait, I need to check this more carefully.

    # Let's just verify: (f*f)(x) at x = (k+1)*bw vs bw*conv[k]
    print(f"\nVerification: (f*f)((k+1)*bw) = bw * conv[k]?")
    for k in range(2*d-1):
        actual = ff[k+1]
        expected = bin_width * conv[k]
        match = "[ok]" if abs(actual - expected) < 1e-12 else f"[FAIL] (diff={actual-expected:.2e})"
        print(f"  k={k}: (f*f)({(k+1)*bin_width:.4f}) = {actual:.6f}, "
              f"bw*conv[{k}] = {expected:.6f}  {match}")

    # If (f*f)((k+1)*bw) = bw*conv[k], then:
    # TV = sum conv[k] / (4n*ell)
    # = sum (f*f)((k+1)*bw) / (bw * 4n * ell)
    # = sum (f*f)((k+1)*bw) * (4n / ell) * (1/bw) / (4n)^2
    # Hmm, bw = 1/(4n), so 1/bw = 4n
    # TV = sum (f*f)(x_k) * 4n / (4n*ell) = sum (f*f)(x_k) / ell
    # = average of (f*f) at ell-1 interior knot points

    print(f"\n=> TV = average of (f*f) at {2*d-1-1} interior knot points")
    print(f"   (NOT the integral-based average)")
    print(f"\n   BUT: if Lemma 3 holds pointwise at these knots, then:")
    print(f"   TV_disc <= TV_cont + correction still holds")
    print(f"   provided we define TV_cont = average of (f*f) at the SAME knots")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    test_knot_vs_integral()
    test_lemma3_interpretation()
    test_lemma3_pointwise()
    test_per_window_integral_tv()
    test_per_window_knot_tv()
    test_coarse_grid_exhaustive()
    # test_pruning_soundness_fine()  # slow, enable if needed
