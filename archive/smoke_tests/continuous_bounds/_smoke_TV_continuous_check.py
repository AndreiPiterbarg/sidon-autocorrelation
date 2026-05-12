"""TV_W continuous-f bridge check.

Tests whether the cascade's TV_W formula (computed from integer bin counts c)
is equal to (1/|W|) * integral_W (f*f)(t) dt for:
  (1) STEP functions f_a with heights a_i/m on bin_i.
  (2) CONTINUOUS functions f with bin AVERAGES a_i/m on bin_i (but arbitrary
      intra-bin shape).

If (1) holds: cascade TV_W = window average of (f*f) for step functions.
If (2) holds for all continuous f with given bin avgs: cascade TV_W is determined
solely by bin averages.

Setup:
  bins: bin_i = [-1/4 + i*w, -1/4 + (i+1)*w], w = 1/(2d).
  d = 2n.
  f >= 0 on [-1/4, 1/4], integral f = 1.
  bin AVG of f = (1/w) * integral_{bin_i} f = a_i/m.
  So integral_{bin_i} f = a_i * w / m.
  Sum of a_i = 4nm (since sum of bin avgs * w = m * 1).

Window W:
  conv positions: k = 0, ..., 2d-2 (2d-1 positions).
  window covers conv positions [s_lo, s_lo + ell - 2] (ell-1 positions).
  In t-coordinates: W = [tau_{s_lo}, tau_{s_lo + ell}] where tau_k = -1/2 + k*w.
  |W| = ell * w.

Cascade TV_W formula (for integer composition c):
  TV_W(c) = (sum_{k in [s_lo, s_lo+ell-2]} conv[k]) / (4n * ell * m^2)
where conv[k] = sum_{i+j=k} c_i * c_j.

For step f_a (heights a_i/m, so f_a^2 average over bin_i is (a_i/m)^2):
  conv[k] for the cascade is sum_{i+j=k} a_i * a_j (as integers, for bin counts).

Continuous (f*f)(t) involves convolution of indicators (tents of width 2w
centered at midpoints). Each tent at conv index k has support
[tau_k, tau_{k+2}] (width 2w). Window W = [tau_{s_lo}, tau_{s_lo+ell}].

For step f_a, integral_W (f_a*f_a) dt:
  k in [s_lo, s_lo+ell-2]: tent FULLY IN W, contributes w^2 * conv[k] / m^2.
  k = s_lo - 1: tent partial, half mass = (w^2/2) * conv[s_lo-1] / m^2.
  k = s_lo + ell - 1: similarly half mass at right edge.

So:
  integral_W (f_a*f_a) dt = (w^2/m^2) * [sum_{k in W_full} conv[k]
                                          + (conv[s_lo-1] + conv[s_lo+ell-1])/2]

  (1/|W|) * integral_W (f_a*f_a) dt = (w/(ell*m^2)) * [...]
                                    = cascade_TV_W
                                      + (conv[s_lo-1] + conv[s_lo+ell-1]) / (8n*ell*m^2)

Therefore: cascade TV_W != window average of (f*f) for step functions, off by
edge terms.
"""
import os
import sys
import json
import numpy as np


def cascade_tv_w(c, n_half, m, ell, s_lo):
    """Cascade's TV_W formula: (sum_{k in [s_lo, s_lo+ell-2]} conv[k]) / (4n*ell*m^2)."""
    d = len(c)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += float(c[i]) * float(c[j])
    # Window is [s_lo, s_lo + ell - 2] inclusive
    ws = float(np.sum(conv[s_lo:s_lo + ell - 1]))
    return ws / (4.0 * n_half * ell * m * m)


def step_function_window_integral(a, n_half, m, ell, s_lo):
    """For step f_a (heights a_i/m on bin_i, w = 1/(4n)):
    Compute integral_W (f_a * f_a)(t) dt EXACTLY using closed-form tent integrals.

    Tent at conv index k (for f_a*f_a) is the convolution of indicators of
    bins (i, j) with i+j=k.  For unit-height (1_bin_i * 1_bin_j), the tent
    has total mass w^2 (height w, width 2w).

    For the heights a_i/m: contribution to (f*f) at conv index k from pair
    (i,j) with i+j=k is (a_i a_j / m^2) * tent_w.

    Window W = [tau_{s_lo}, tau_{s_lo+ell}], length ell*w.
    Tent at index k has support [tau_k, tau_{k+2}].

    Cases:
      tau_{k+2} <= tau_{s_lo}: tent entirely LEFT of W, contributes 0.
      tau_k >= tau_{s_lo+ell}: tent entirely RIGHT of W, contributes 0.
      tau_{s_lo} <= tau_k AND tau_{k+2} <= tau_{s_lo+ell}: tent fully inside,
        contributes w^2.  (i.e., k in [s_lo, s_lo+ell-2].)
      k = s_lo - 1 (tent at [tau_{s_lo-1}, tau_{s_lo+1}]): right half in W,
        contributes w^2/2.
      k = s_lo + ell - 1 (tent at [tau_{s_lo+ell-1}, tau_{s_lo+ell+1}]):
        left half in W, contributes w^2/2.
    """
    d = len(a)
    w = 1.0 / (4.0 * n_half)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += float(a[i]) * float(a[j])

    integral = 0.0
    # Fully inside k in [s_lo, s_lo+ell-2]: tent_mass = w^2
    for k in range(s_lo, s_lo + ell - 1):
        if 0 <= k < conv_len:
            integral += conv[k] * w * w
    # Left edge k = s_lo - 1: tent_mass = w^2 / 2
    k_left = s_lo - 1
    if 0 <= k_left < conv_len:
        integral += conv[k_left] * w * w / 2.0
    # Right edge k = s_lo + ell - 1: tent_mass = w^2 / 2
    k_right = s_lo + ell - 1
    if 0 <= k_right < conv_len:
        integral += conv[k_right] * w * w / 2.0

    integral /= (m * m)  # Convert to f-coordinate (height = a_i/m)
    return integral


def numerical_window_integral(f_func, n_half, m, ell, s_lo, n_grid=4000):
    """Numerically compute integral_W (f*f)(t) dt for any function f.

    Uses fine quadrature for both the convolution f*f and the window integral.
    f_func: function taking x in [-1/4, 1/4] returning f(x) >= 0.
    """
    w = 1.0 / (4.0 * n_half)
    d = 2 * n_half
    # tau_k = -1/2 + k*w
    t_lo = -0.5 + s_lo * w
    t_hi = -0.5 + (s_lo + ell) * w

    # Compute (f*f)(t) numerically on a fine grid covering t_lo to t_hi
    # (f*f)(t) = integral f(s) f(t-s) ds, support t in [-1/2, 1/2]
    # Use a fine grid for s in [-1/4, 1/4]
    n_s = n_grid
    s_grid = np.linspace(-0.25, 0.25, n_s)
    ds = s_grid[1] - s_grid[0]
    f_s = f_func(s_grid)

    n_t = n_grid
    t_grid = np.linspace(t_lo, t_hi, n_t)
    dt = (t_hi - t_lo) / (n_t - 1)

    # For each t in window, compute (f*f)(t) = integral f(s) f(t-s) ds
    ff = np.zeros(n_t)
    for it in range(n_t):
        t = t_grid[it]
        # f(t - s) for s in s_grid
        ts = t - s_grid
        # Evaluate f at ts (zero outside [-1/4, 1/4])
        f_ts = np.where((ts >= -0.25) & (ts <= 0.25), f_func(ts), 0.0)
        ff[it] = np.sum(f_s * f_ts) * ds

    # Trapezoidal integration over t in W
    integral = np.trapz(ff, t_grid)
    return integral


def step_f(a, m, n_half):
    """Build step function f_a: returns callable f_a(x).

    Heights a_i/m on bin_i = [-1/4 + i*w, -1/4 + (i+1)*w], w = 1/(2d) = 1/(4n)."""
    w = 1.0 / (4.0 * n_half)
    d = len(a)
    a_arr = np.asarray(a, dtype=np.float64)

    def f(x):
        x = np.atleast_1d(x)
        out = np.zeros_like(x, dtype=np.float64)
        for i in range(d):
            lo = -0.25 + i * w
            hi = -0.25 + (i + 1) * w
            mask = (x >= lo) & (x < hi)
            out[mask] = a_arr[i] / m
        # Edge case: x = 0.25 maps to last bin
        out[x == 0.25] = a_arr[d - 1] / m
        return out

    return f


def continuous_f_with_bin_averages(a, n_half, m, shape='triangle'):
    """Build a continuous f with given bin averages a_i/m.

    Within each bin_i, place mass a_i*w/m as a non-negative function with
    integral matching, but with a non-trivial intra-bin distribution.

    shape options:
      'triangle': hat function within each bin (peak at bin center).
      'left_concentrated': mass concentrated at left half of each bin.
      'right_concentrated': mass concentrated at right half of each bin.
      'gaussian_per_bin': Gaussian centered at bin center with sigma=w/4.
    """
    w = 1.0 / (4.0 * n_half)
    d = len(a)
    a_arr = np.asarray(a, dtype=np.float64)

    def f(x):
        x = np.atleast_1d(x)
        out = np.zeros_like(x, dtype=np.float64)
        for i in range(d):
            lo = -0.25 + i * w
            hi = -0.25 + (i + 1) * w
            mid = (lo + hi) / 2.0
            mask = (x >= lo) & (x <= hi)
            xb = x[mask]
            mass_target = a_arr[i] * w / m  # integral over bin_i
            if mass_target <= 0:
                continue
            if shape == 'triangle':
                # Hat: f(x) = peak * (1 - |x - mid| / (w/2))
                # integral = peak * w/2, so peak = 2*mass_target/w = 2*a_i/m.
                peak = 2.0 * a_arr[i] / m
                vals = peak * np.maximum(0.0, 1.0 - np.abs(xb - mid) / (w / 2.0))
                out[mask] = vals
            elif shape == 'left_concentrated':
                # Mass on [lo, lo + w/2], height = 2*a_i/m.
                vals = np.where(xb < lo + w / 2.0, 2.0 * a_arr[i] / m, 0.0)
                out[mask] = vals
            elif shape == 'right_concentrated':
                # Mass on [lo + w/2, hi], height = 2*a_i/m.
                vals = np.where(xb >= lo + w / 2.0, 2.0 * a_arr[i] / m, 0.0)
                out[mask] = vals
            elif shape == 'gaussian_per_bin':
                # Truncated Gaussian centered at mid; renormalize to mass_target
                sigma = w / 4.0
                g = np.exp(-((xb - mid) ** 2) / (2 * sigma * sigma))
                # Compute Gaussian normalization on this bin
                xb_fine = np.linspace(lo, hi, 200)
                g_fine = np.exp(-((xb_fine - mid) ** 2) / (2 * sigma * sigma))
                z = np.trapz(g_fine, xb_fine)
                vals = mass_target * g / z
                out[mask] = vals
            else:
                raise ValueError(f"Unknown shape: {shape}")
        return out

    return f


def main():
    print("=" * 78)
    print("TV_W continuous-f bridge check")
    print("=" * 78)

    test_cases = [
        # (n_half, m, c, ell, s_lo)
        (2, 10, [3, 5, 7, 5], 4, 1),    # d=4
        (2, 10, [4, 4, 6, 6], 3, 2),    # d=4
        (3, 8, [2, 3, 4, 5, 4, 6], 5, 3),  # d=6
        (2, 5, [3, 2, 1, 4], 2, 3),     # d=4 small ell
    ]

    results = []
    for case_idx, (n_half, m, c, ell, s_lo) in enumerate(test_cases):
        d = 2 * n_half
        S = sum(c)
        # For correct normalization: integral f = 1 means S = 4*n*m
        # Adjust to make S match if needed; just rescale c
        target_S = 4 * n_half * m
        if S != target_S:
            print(f"Case {case_idx}: rescaling c sum {S} -> {target_S}")
            scale = target_S / S
            a = [ci * scale for ci in c]
        else:
            a = list(c)

        a_int = [int(round(ai)) for ai in a]  # integer for cascade formula

        cascade_val = cascade_tv_w(a, n_half, m, ell, s_lo)
        step_int = step_function_window_integral(a, n_half, m, ell, s_lo)
        w = 1.0 / (4.0 * n_half)
        win_len = ell * w
        step_avg = step_int / win_len  # window average for step f_a

        f_a_func = step_f(a, m, n_half)
        # Verify normalization
        x_grid = np.linspace(-0.25, 0.25, 4000)
        norm_step = np.trapz(f_a_func(x_grid), x_grid)

        num_step = numerical_window_integral(f_a_func, n_half, m, ell, s_lo,
                                              n_grid=2000)
        num_step_avg = num_step / win_len

        # Continuous variants with same bin averages
        cont_results = {}
        for shape in ['triangle', 'left_concentrated', 'right_concentrated',
                      'gaussian_per_bin']:
            f_cont = continuous_f_with_bin_averages(a, n_half, m, shape)
            norm_cont = np.trapz(f_cont(x_grid), x_grid)
            int_cont = numerical_window_integral(f_cont, n_half, m, ell, s_lo,
                                                  n_grid=2000)
            avg_cont = int_cont / win_len
            cont_results[shape] = {
                'norm': float(norm_cont),
                'window_avg': float(avg_cont),
                'diff_from_cascade': float(avg_cont - cascade_val),
            }

        result = {
            'case_idx': case_idx,
            'n_half': n_half, 'm': m, 'ell': ell, 's_lo': s_lo,
            'a': list(a), 'd': d,
            'norm_check_step': float(norm_step),
            'cascade_TV_W': float(cascade_val),
            'step_window_integral_closed_form': float(step_int),
            'step_window_avg_closed_form': float(step_avg),
            'step_window_integral_numerical': float(num_step),
            'step_window_avg_numerical': float(num_step_avg),
            'closed_form_minus_cascade': float(step_avg - cascade_val),
            'numerical_step_minus_cascade': float(num_step_avg - cascade_val),
            'continuous_variants': cont_results,
        }
        results.append(result)

        print(f"\n=== Case {case_idx}: n={n_half}, m={m}, d={d}, ell={ell}, s_lo={s_lo} ===")
        print(f"  a = {a}")
        print(f"  norm(f_step)    = {norm_step:.6f} (should = 1)")
        print(f"  cascade TV_W           = {cascade_val:.10f}")
        print(f"  step (closed-form) avg = {step_avg:.10f}")
        print(f"  step (numerical) avg   = {num_step_avg:.10f}")
        print(f"  closed-form - cascade  = {step_avg - cascade_val:+.6e}")
        print(f"  numerical   - cascade  = {num_step_avg - cascade_val:+.6e}")
        for shape, r in cont_results.items():
            print(f"  {shape:25s} avg = {r['window_avg']:.10f}  "
                  f"diff = {r['diff_from_cascade']:+.6e}  "
                  f"norm = {r['norm']:.4f}")

    out_path = os.path.join(os.path.dirname(__file__),
                             '_smoke_TV_continuous_check.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults written to {out_path}")

    # Final verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    max_diff_step = max(abs(r['closed_form_minus_cascade']) for r in results)
    print(f"Max |step_window_avg - cascade_TV_W| = {max_diff_step:.6e}")
    if max_diff_step < 1e-10:
        print("=> For STEP f, cascade TV_W = window average of (f*f). [MATCH]")
    else:
        print("=> For STEP f, cascade TV_W != window average of (f*f).")
        print("   Cascade TV_W is OFFSET from window average by edge terms")
        print("   = (conv[s_lo-1] + conv[s_lo+ell-1]) / (8n*ell*m^2).")
        # Verify that step window avg = cascade + edge terms (in closed form).
        all_match_edge = True
        for r in results:
            cascade = r['cascade_TV_W']
            step = r['step_window_avg_closed_form']
            # We need to compute the edge-term prediction
            d = r['d']
            n_half = r['n_half']
            m = r['m']
            ell = r['ell']
            s_lo = r['s_lo']
            a = r['a']
            conv_len = 2 * d - 1
            conv = np.zeros(conv_len)
            for i in range(d):
                for j in range(d):
                    conv[i + j] += a[i] * a[j]
            edge_lr = 0.0
            if 0 <= s_lo - 1 < conv_len:
                edge_lr += conv[s_lo - 1]
            if 0 <= s_lo + ell - 1 < conv_len:
                edge_lr += conv[s_lo + ell - 1]
            predicted_offset = edge_lr / (8.0 * n_half * ell * m * m)
            actual_offset = step - cascade
            if abs(predicted_offset - actual_offset) > 1e-10:
                all_match_edge = False
                print(f"   Case {r['case_idx']}: predicted {predicted_offset:.6e},"
                      f" actual {actual_offset:.6e}, diff {predicted_offset - actual_offset:+.2e}")
        if all_match_edge:
            print("   Edge-term formula VERIFIED:"
                  " step_avg = cascade + (conv[s_lo-1]+conv[s_lo+ell-1])/(8n*ell*m^2). [MATCH]")

    max_diff_cont = 0.0
    for r in results:
        for shape, c in r['continuous_variants'].items():
            max_diff_cont = max(max_diff_cont, abs(c['diff_from_cascade']))
    print(f"\nMax |continuous_window_avg - cascade_TV_W| = {max_diff_cont:.6e}")
    if max_diff_cont < 1e-10:
        print("=> For CONTINUOUS f with same bin avgs, cascade TV_W = window avg.")
    else:
        print("=> For CONTINUOUS f with same bin avgs, cascade TV_W != window avg.")
        print("   Window avg DEPENDS on intra-bin distribution of f at boundaries.")
        # Verify cascade TV_W <= continuous window avg for all variants tested
        all_geq = True
        for r in results:
            for shape, c in r['continuous_variants'].items():
                if c['window_avg'] + 1e-3 < r['cascade_TV_W']:  # 1e-3 numeric tol
                    all_geq = False
                    print(f"   VIOLATION case {r['case_idx']} {shape}: "
                          f"avg={c['window_avg']:.6f}, cascade={r['cascade_TV_W']:.6f}")
        if all_geq:
            print("   LOWER BOUND verified: cascade_TV_W <= window_avg for all"
                  " continuous f tested.")


if __name__ == '__main__':
    main()
