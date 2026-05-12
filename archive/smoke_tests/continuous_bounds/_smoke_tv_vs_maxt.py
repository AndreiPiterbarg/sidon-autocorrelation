"""Smoke test: is the cascade's `max_W TV_W(a)` a tight lower bound on
`max_t (f_a * f_a)(t)`?

KEY: f is the step function f(x) = c_i / S where S = 4*n_half*m (so int f = 1).
That is, height = c_i/S, NOT c_i/m.  TV_W(a) = ws_W / (4n*ell*m^2)
applies WHEN c is INTEGER and S=4nm; equivalently, with mu_i = c_i/S
=> TV_W = (1/ell) * sum_{k in window} sum_{i+j=k} mu_i mu_j  -- a LOWER BOUND
on max_t (f*f) by Theorem 1 (Cloninger-Steinerberger 2017).

We verify:
 (a) cascade max_W TV_W <= max_t (f*f)(t)  -- soundness
 (b) at ell=2, max_k conv[k]/(8n*m^2) =? max_t (f*f) (i.e. is ell=2 already tight?)
 (c) what's the residual gap, if any?

USAGE:
    python _smoke_tv_vs_maxt.py
"""
import os, sys, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))


def cascade_max_TV_W(c, n_half, m):
    """Compute max_W TV_W(a) using the cascade's formula.
    TV_W(a) = ws_W / (4n*ell*m^2) where ws_W = sum of conv[s_lo..s_hi].
    Returns (max_TV, best_ell, best_s_lo).
    """
    c = np.asarray(c, dtype=np.int64)
    d = len(c)
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

    four_n = 4.0 * n_half
    m2 = m * m
    max_TV = -np.inf
    best_ell = -1
    best_s = -1
    max_ell = 2 * d
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            ws = int(conv[s_lo:s_lo + n_cv].sum())
            TV = ws / (four_n * ell * m2)
            if TV > max_TV:
                max_TV = TV
                best_ell = ell
                best_s = s_lo
    return max_TV, best_ell, best_s, conv


def piecewise_linear_ff(c, n_half, m, n_pts=20001):
    """Compute (f*f)(t) for the L^1-NORMALIZED step function.

    f(x) = mu_i / h on bin i, where h = 1/(2d) and mu_i = c_i / S, S = 4*n_half*m.
    Equivalently f(x) = c_i / (S*h) = (2d*c_i)/S on bin i.
    Then int f = sum mu_i = 1 (correct L^1 normalization).

    (f * f)(t) is piecewise linear in t with knots at integer multiples of h
    (shifted by -1/2).  At t_k = k*h - 1/2 + h/2 = (k+0.5)*h - 1/2 ... actually,
    for bin width h, autoconv is piecewise-linear with knots at t = k*h - 1/2
    for k = 0,...,2d, AND has VALUES at t_k = k*h - 1/2 + h that equal:
    (1/h) * sum_{i+j=k-1} mu_i mu_j  ... let me just compute numerically.
    """
    c = np.asarray(c, dtype=np.float64)
    d = len(c)
    h = 1.0 / (2 * d)
    S = 4.0 * n_half * m
    mu = c / S  # bin masses, sum = 1
    f_height = mu / h  # = (2d * c) / S, height of f on bin i

    t_grid = np.linspace(-0.5 + 1e-12, 0.5 - 1e-12, n_pts)
    ff = np.zeros_like(t_grid)
    # Sample x on a finer grid for the integral
    x_grid = np.linspace(-0.25 + 1e-12, 0.25 - 1e-12, 2 * n_pts)
    dx = x_grid[1] - x_grid[0]

    def f_eval(x):
        idx = np.clip(np.floor((x + 0.25) / h).astype(int), 0, d - 1)
        in_support = (x >= -0.25) & (x <= 0.25)
        return np.where(in_support, f_height[idx], 0.0)

    f_x = f_eval(x_grid)
    for i, t in enumerate(t_grid):
        f_t_minus_x = f_eval(t - x_grid)
        ff[i] = np.sum(f_x * f_t_minus_x) * dx

    max_idx = int(np.argmax(ff))
    max_t = t_grid[max_idx]
    max_val = ff[max_idx]

    # Knot values: at t_k = (k+1)*h - 1/2 for k=0..2d-2,
    # (f*f)(t_k) = h * (f_height * f_height conv)[k] = h * sum_{i+j=k} f_h[i] f_h[j].
    # Substituting f_h = (2d) * c / S = c / (h * S):
    #   knot value = h * (1/(h*S))^2 * conv_int[k] = conv_int[k] / (h * S^2)
    # = conv_int[k] * 2d / S^2.
    discrete_conv = np.convolve(f_height, f_height)  # length 2d-1
    knot_t = np.array([(k + 1) * h - 0.5 for k in range(2 * d - 1)])
    knot_vals = h * discrete_conv

    return t_grid, ff, max_val, max_t, knot_t, knot_vals


def study_cell(c, n_half, m, label=""):
    """Compare max_W TV_W vs max_t (f*f)(t)."""
    print(f"\n=== Cell {label}: c = {tuple(c)}, n_half = {n_half}, m = {m} ===")
    d = len(c)
    h = 1.0 / (2 * d)
    S = 4 * n_half * m
    print(f"  d = {d}, h = 1/(2d) = {h}, S = 4nm = {S}")
    print(f"  sum c_i = {sum(c)} (should equal S = {S})")

    # Cascade quantity
    max_TV, best_ell, best_s, conv = cascade_max_TV_W(c, n_half, m)
    print(f"  conv (integer) = {conv.tolist()}")
    print(f"  cascade max_W TV_W = {max_TV:.10f}  (ell={best_ell}, s_lo={best_s})")

    # ell=2 specifically: TV at each conv position
    ell2_TVs = conv / (4.0 * n_half * 2 * m * m)
    max_ell2 = float(np.max(ell2_TVs))
    print(f"  ell=2 max TV (= max_k conv[k]/(8n*m^2)) = {max_ell2:.10f}")

    # EXACT (f*f) at the conv-position knots:
    #   At t_k = (k+1)*h - 1/2 + h  (centerline of bin-overlap), the value is
    #   (2d/S^2) * conv[k]
    # since f(x) = mu_i/h = c_i/(h*S), and the autoconv of two width-h pulses on common
    # grid evaluated at t_k = the center of the (i+j+1)-th overlap segment is
    # h * f_height[i] * f_height[j] summed over i+j=k.
    knot_vals_exact = (2 * d / (S * S)) * conv  # (f*f) values at conv-position knots
    max_knot_exact = float(np.max(knot_vals_exact))
    # SANITY: max_knot_exact = (2d/S^2) * max_k conv[k]
    # For fine grid (S=4nm, d=2n): 2d/S^2 = 4n/(16n^2 m^2) = 1/(4nm^2)
    # So max_knot = max_k conv[k] / (4nm^2) = 2 * max_k(conv[k]/(8nm^2)) = 2 * max_ell2_TV.
    # In other words, RAW (f*f) values at the conv-position knots are 2x the cascade's
    # ell=2 TV values.  The cascade's TV_W is the Theorem 1 LOWER BOUND on max_t (f*f),
    # which equals (2d/ell) * (1/S^2) * ws -- the prefactor 2d/(ell*S^2) is the ABM
    # average, NOT a pointwise value of (f*f).
    print(f"  EXACT max_t (f*f)(t) at conv-knots = {max_knot_exact:.10f}")
    print(f"     (cascade TV is a Theorem 1 LOWER bound; equality requires conv to be CONSTANT in W)")

    # Full piecewise-linear (f*f) computed numerically
    t_grid, ff, max_val, max_t, knot_t, knot_vals = piecewise_linear_ff(c, n_half, m, n_pts=2001)
    print(f"  numerical max_t (f*f)(t)            = {max_val:.10f}  (at t = {max_t:.6f})")
    print(f"  knot values: max conv-position knot = {max_knot_exact:.10f}")

    # KEY GAPS:
    # Gap A: max_t(f*f) - cascade max_W TV_W (cascade is a Theorem 1 LOWER BOUND on max_t(f*f)).
    # Gap C: cascade max_W TV_W - ell=2 max TV (does averaging over wider windows help?).
    # Gap K: max_t(f*f) - 2 * max_ell2_TV  -- gap to "(f*f) at conv-position knot"
    #         (a different bound on max_t(f*f); this should be 0 in absence of integration error).
    gap_A = max_knot_exact - max_TV
    gap_C = max_TV - max_ell2
    # tight_at_ell2: is the cascade's bound dominated by ell=2 (= max_t(f*f) over conv knots / 2)?
    cascade_pct_of_max_t = max_TV / max_knot_exact if max_knot_exact > 0 else 0.0
    cascade_pct_of_2x_ell2 = max_TV / (2.0 * max_ell2) if max_ell2 > 0 else 0.0

    print(f"  GAP A: max_t(f*f) - cascade max_W TV_W   = {gap_A:+.10f}")
    print(f"     (Theorem 1: A >= 0 always.  cascade/max_t(f*f) = {cascade_pct_of_max_t*100:.1f}%)")
    print(f"  GAP C: cascade max_W TV_W - ell=2 max TV = {gap_C:+.10f}")
    print(f"     (>0 => ell>=3 STRICTLY better;  cascade/(2*ell2) = {cascade_pct_of_2x_ell2*100:.1f}%)")

    soundness_ok = (max_TV <= max_knot_exact + 1e-9)
    if not soundness_ok:
        print(f"  *** SOUNDNESS VIOLATED: cascade={max_TV} > max_t(f*f)={max_knot_exact}")

    return {
        'c': list(int(x) for x in c),
        'n_half': int(n_half),
        'm': int(m),
        'd': int(d),
        'h': float(h),
        'S_4nm': int(S),
        'cascade_max_TV_W': float(max_TV),
        'cascade_best_ell': int(best_ell),
        'cascade_best_s_lo': int(best_s),
        'ell2_max_TV': float(max_ell2),
        'max_t_ff_exact': float(max_knot_exact),
        'max_t_ff_numerical': float(max_val),
        'numerical_argmax_t': float(max_t),
        'cascade_pct_of_max_t': float(cascade_pct_of_max_t),
        'cascade_pct_of_2x_ell2': float(cascade_pct_of_2x_ell2),
        'knot_t': [float(x) for x in knot_t],
        'knot_vals_exact': [float(x) for x in knot_vals_exact],
        'ell2_TVs': [float(x) for x in ell2_TVs],
        'conv': [int(x) for x in conv],
        'gap_A_max_t_minus_cascade_TV': float(gap_A),
        'gap_C_cascade_TV_minus_ell2_TV': float(gap_C),
        'soundness_ok': bool(soundness_ok),
    }


def study_d2_n1_m20():
    """d=2, n_half=1, m=20.  4n*m = 80, so c_0 + c_1 = 80.
    Palindromic constraint: c_0 = c_1 = 40 (the only palindromic point).
    But the cell mentioned is c=(40,40)."""
    results = []
    # c=(40,40)
    results.append(study_cell([40, 40], 1, 20, "d=2 c=(40,40)"))
    # also check non-palindromic
    results.append(study_cell([30, 50], 1, 20, "d=2 c=(30,50)"))
    results.append(study_cell([20, 60], 1, 20, "d=2 c=(20,60)"))
    results.append(study_cell([0, 80], 1, 20, "d=2 c=(0,80) extreme"))
    return results


def study_d4():
    """d=4, n_half=2, m=10. 4n*m=80."""
    results = []
    results.append(study_cell([20, 20, 20, 20], 2, 10, "d=4 uniform"))
    results.append(study_cell([15, 25, 25, 15], 2, 10, "d=4 palindromic"))
    results.append(study_cell([0, 40, 40, 0], 2, 10, "d=4 concentrated"))
    results.append(study_cell([10, 30, 30, 10], 2, 10, "d=4 mid"))
    return results


def study_d8():
    """d=8, n_half=2, m=5. 4n*m=40, so Σc=40."""
    results = []
    results.append(study_cell([5, 5, 5, 5, 5, 5, 5, 5], 2, 5, "d=8 uniform"))
    results.append(study_cell([0, 10, 10, 0, 0, 10, 10, 0], 2, 5, "d=8 split"))
    results.append(study_cell([2, 8, 4, 6, 6, 4, 8, 2], 2, 5, "d=8 palindromic-ish"))
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("TV_W vs max_t (f*f) tightness study")
    print("=" * 80)
    all_results = {}
    all_results['d2_n1_m20'] = study_d2_n1_m20()
    all_results['d4_n2_m10'] = study_d4()
    all_results['d8_n2_m5'] = study_d8()

    # Compute summary
    summary = {}
    max_gap_A = 0.0
    max_gap_A_cell = None
    min_pct = 1.0
    min_pct_cell = None
    max_gap_C = 0.0
    max_gap_C_cell = None
    n_soundness_violated = 0
    n_tested = 0
    n_ell_gt2_beats = 0  # gap_C > 0 (cascade goes beyond ell=2)
    for set_name, results in all_results.items():
        for r in results:
            n_tested += 1
            if not r['soundness_ok']:
                n_soundness_violated += 1
            gA = r['gap_A_max_t_minus_cascade_TV']
            gC = r['gap_C_cascade_TV_minus_ell2_TV']
            pct = r['cascade_pct_of_max_t']
            if gC > 1e-9:
                n_ell_gt2_beats += 1
            if abs(gA) > max_gap_A:
                max_gap_A = abs(gA)
                max_gap_A_cell = (set_name, r['c'])
            if pct < min_pct:
                min_pct = pct
                min_pct_cell = (set_name, r['c'])
            if abs(gC) > max_gap_C:
                max_gap_C = abs(gC)
                max_gap_C_cell = (set_name, r['c'])
    summary['n_tested'] = n_tested
    summary['n_soundness_violated'] = n_soundness_violated
    summary['n_ell_gt2_beats_ell2'] = n_ell_gt2_beats
    summary['max_gap_A_max_t_minus_cascade_TV'] = max_gap_A
    summary['max_gap_A_cell'] = max_gap_A_cell
    summary['min_cascade_pct_of_max_t'] = min_pct
    summary['min_pct_cell'] = min_pct_cell
    summary['max_gap_C_cascade_TV_minus_ell2_TV'] = max_gap_C
    summary['max_gap_C_cell'] = max_gap_C_cell
    summary['interpretation'] = {
        'theorem_1': 'TV_W(c) = (2d/(ell*S^2)) * ws_W = ws_W/(4n*ell*m^2) [for fine grid, S=4nm, d=2n]. Theorem 1: max_t(f*f) >= TV_W.  This is a LOWER bound, so cascade_TV <= max_t(f*f) always.',
        'tightness_at_ell2': 'At conv-position knots, (f*f) = (2d/S^2)*conv[k] = 2 * TV_ell2[k]. So cascade is at most HALF of max_t(f*f) at single-conv knot. The factor 2 is structural.',
        'cascade_optimum_strategy': 'Cascade picks ell, s_lo to maximize ws_W/(4n*ell*m^2). Larger ell averages more conv positions => smaller per-position contribution but possibly larger sum (when surrounding positions are also large).',
        'gap_A_meaning': 'max_t(f*f) - max_W TV_W; ALWAYS >= 0 (Theorem 1).  Tight when conv is highly concentrated at one position AND ell=2 is the optimal window.',
        'gap_C_meaning': 'max_W TV_W - max_ell=2 TV; >0 means ell>=3 averaging gives a STRICTLY TIGHTER LB than peak ell=2.  This happens for SPREAD distributions (uniform, palindromic).',
    }
    all_results['summary'] = summary

    out_path = os.path.join(_dir, '_smoke_tv_vs_maxt.json')
    with open(out_path, 'w') as fp:
        json.dump(all_results, fp, indent=2, default=float)
    print(f"\n[saved] {out_path}")
    print(f"\nSUMMARY:")
    print(f"  tested {n_tested} cells.")
    print(f"  soundness violations: {n_soundness_violated} (must be 0).")
    print(f"  cells where ell>=3 STRICTLY beats ell=2 in TV: {n_ell_gt2_beats}.")
    print(f"  max |gap A| (= max_t(f*f) - cascade)  = {max_gap_A:.6e}, cell={max_gap_A_cell}")
    print(f"  min cascade/max_t(f*f) ratio          = {min_pct:.4f}, cell={min_pct_cell}")
    print(f"  max gap C (= cascade - ell=2 TV)      = {max_gap_C:.6e}, cell={max_gap_C_cell}")
