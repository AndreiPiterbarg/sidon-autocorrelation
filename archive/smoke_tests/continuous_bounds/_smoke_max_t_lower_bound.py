"""Tighter lower bound on min_{a in cell} max_t (f_a*f_a)(t) at d=2.

Mission: at (n_half=1, m=20, c=1.281, d=2), the cascade chain
F+FN+Q+QN+L+SP cannot prune any of the 25 L0 cells, because each cell
contains a witness with max_W TV_W <= 1.28.

We test 5 alternative LBs on min over a-in-cell of max_t (f_a*f_a)(t),
all expressed in c-target units (||f*f||_inf / ||f||_1^2):

  M0  TV_max:        cascade baseline. max_W TV_W(a). SOUND (CS Lemma 1).
  M1  M_inf-step:    ||f_a*f_a||_inf / ||f_a||_1^2 = max(4 a_0^2, 4 a_1^2,
                     8 a_0 a_1) / 16 (for d=2 step ansatz).
                     NOT SOUND for cont f (empirical counter-example: f
                     can have ||f*f||_inf < ||f_a*f_a||_inf -0.21 with
                     same bin avgs). Used only for benchmarking.
  M2  Planch-step:   ||f_a*f_a||_2^2 / ||f_a||_1^4. SOUNDNESS UNPROVEN
                     (cosine: ||f*f||_2^2 < ||f_a*f_a||_2^2; but no
                     counter-example to ||f*f||_inf >= Planch-step found
                     in 50-restart adversarial search).
  M3  M_off-diag:    (f_a*f_a)(h)/||f_a||_1^2 = max(a_0^2, a_1^2)/4 for d=2
                     NOT SOUND for cont f. Benchmark only.
  M4  M_avg-step:    avg over symmetric weights. Trivially = 1.

Comparison: count how many of 26 unique cells (after symmetry) get
bound > 1.281 via each method.

Note on soundness:
  - SOUND methods (TV_max only): valid LB on c = inf_f over the
    continuous problem, via CS Lemma 1's window-averaging argument.
  - Step-ansatz methods (M_inf, Planch, M_off): bound the c-value of the
    step ansatz only. To use as LB on c, would need a Jensen-type
    argument (||f*f||_inf >= ||f_a*f_a||_inf) which FAILS empirically.
    Plancherel-step (M2) holds empirically but proof requires care.

For d=2, n=1 the math is 1D: a_0 ∈ [(c_0-1)/m, (c_0+1)/m], a_1 = 4-a_0.

Output: for each cell, which methods give bound > 1.281, with margins.
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize_scalar

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

t_start = time.time()

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
n_half = 1
m = 20
c_target = 1.281
d = 2 * n_half  # = 2
S = 4 * n_half * m  # = 80
h_bin = 1.0 / (2 * d)  # = 1/4 (bin width on [-1/4, 1/4])
norm_l1_squared = (4.0 * n_half) ** 2  # ||f||_1^2 = 16 (Σa_i = 4)

# F-survivor list (computed via _check_l0_cells.py): c_0 ∈ [15..65]
F_SURVIVORS = list(range(15, 66))
# Symmetric pairs: (c_0, 80-c_0). 25 unique cells + center (40,40) = 26.
unique_c0 = sorted(set(min(c0, 80 - c0) for c0 in F_SURVIVORS))
print(f"Unique cells (after symmetry): {len(unique_c0)} cells")
print(f"c_0 representatives: {unique_c0}")


# ---------------------------------------------------------------------
# Per-method evaluator. All in c-units (compare to c_target = 1.281).
# Convention: ||f_a||_1 = 4 (Σa = 4), c-value = ||f_a*f_a||_inf / 16.
# ---------------------------------------------------------------------

def TV_max(a0):
    """M0 (cascade): max_W TV_W(a) where TV_W = (1/(4n*ell))*Σ_{i+j∈[s_lo,s_hi]} a_i a_j

    SOUND: CS Lemma 1 ⇒ c >= TV_max for any a's bin avg of an actual f.
    """
    a1 = 4.0 - a0
    c0sq = a0 * a0
    c01 = 2.0 * a0 * a1
    c1sq = a1 * a1
    tv2 = max(c0sq, c01, c1sq) / 8.0
    tv3 = max(c0sq + c01, c01 + c1sq) / 12.0
    tv4 = (c0sq + c01 + c1sq) / 16.0
    return max(tv2, tv3, tv4)


def M_inf_step(a0):
    """||f_a*f_a||_inf / ||f_a||_1^2 (step ansatz exact). NOT SOUND.

    For d=2 step: (f*f) piecewise linear with values:
       (f*f)(-h) = 4 a_0^2, (f*f)(0) = 8 a_0 a_1, (f*f)(h) = 4 a_1^2.
    Max = max(4 a_0^2, 4 a_1^2, 8 a_0 a_1).
    Divide by ||f||_1^2 = 16.
    """
    a1 = 4.0 - a0
    val_neg = 4.0 * a0 * a0
    val_0 = 8.0 * a0 * a1
    val_pos = 4.0 * a1 * a1
    return max(val_neg, val_0, val_pos) / 16.0


def M_off_step(a0):
    """(f_a*f_a)(h_max) / ||f_a||_1^2 = max((f*f) at the off-diag node).

    For d=2 step at t=h: 4 a_1^2; at t=-h: 4 a_0^2. Max = 4·max(a_0,a_1)^2.
    Divide by 16: max(a_0, a_1)^2 / 4.

    NOT SOUND (no general (f*f)(t) >= (f_a*f_a)(t) for continuous f).
    """
    a1 = 4.0 - a0
    return max(a0, a1) ** 2 / 4.0


def M_planch_step(a0):
    """Plancherel-on-step LB: ||f_a*f_a||_2^2 / ||f_a||_1^4.

    For step f_a at d=2, h=1/4, with A = 4 a_0 a_1, B = 4(a_0^2+a_1^2):
        ||f_a*f_a||_2^2 = (2 A^2 + A B + B^2) / 6   (analytic).
    Plancherel: ||g*g||_inf >= ||g*g||_2^2 / ||g*g||_1 = ||g*g||_2^2 (g=f/||f||_1).
    Step Plancherel LB: this/||f_a||_1^4 = this/256.

    NOT FORMALLY SOUND for cont f (cosine: ||f*f||_2^2 < ||f_a*f_a||_2^2),
    but EMPIRICALLY ||f*f||_inf >= Planch-step holds with positive margin
    in all tests; rigorous proof would require additional work.
    """
    a1 = 4.0 - a0
    A = 4.0 * a0 * a1
    B = 4.0 * (a0 * a0 + a1 * a1)
    L2_sq_unscaled = (2 * A * A + A * B + B * B) / 6.0
    return L2_sq_unscaled / 256.0


def M_combined_step(a0):
    """max(M_inf_step, M_planch_step). Used to estimate practical step LB."""
    return max(M_inf_step(a0), M_planch_step(a0))


def cell_min_bound(c0_int, fn):
    """Find min of fn over cell a_0 ∈ [(c0-1)/m, (c0+1)/m]."""
    a_lo = (c0_int - 1) / m
    a_hi = (c0_int + 1) / m
    a_lo_clip = max(a_lo, 0.0)
    a_hi_clip = min(a_hi, 4.0)
    if a_lo_clip >= a_hi_clip - 1e-12:
        return fn(a_lo_clip)
    res = minimize_scalar(fn, bounds=(a_lo_clip, a_hi_clip), method='bounded',
                          options={'xatol': 1e-10})
    # Robust check: also evaluate at endpoints
    f_lo = fn(a_lo_clip)
    f_hi = fn(a_hi_clip)
    return min(res.fun, f_lo, f_hi)


# ---------------------------------------------------------------------
# Run on each cell
# ---------------------------------------------------------------------
results = {}
print(f"\n{'c_0':>4s}|{'a_lo':>6s}{'a_hi':>6s}|{'TV':>7s}{'M_inf':>7s}{'M_off':>7s}"
      f"{'Plnchl':>7s}|{'>1.281 (>0 margin)'}")
print("-" * 95)

count_methods = {'TV': 0, 'M_inf_step': 0, 'M_off_step': 0, 'M_planch_step': 0}
total_cells = len(unique_c0)

for c0 in unique_c0:
    tv = cell_min_bound(c0, TV_max)
    minf = cell_min_bound(c0, M_inf_step)
    moff = cell_min_bound(c0, M_off_step)
    plnchl = cell_min_bound(c0, M_planch_step)
    methods_pass = []
    if tv > c_target: methods_pass.append('TV'); count_methods['TV'] += 1
    if minf > c_target: methods_pass.append('M_inf'); count_methods['M_inf_step'] += 1
    if moff > c_target: methods_pass.append('M_off'); count_methods['M_off_step'] += 1
    if plnchl > c_target: methods_pass.append('Plnch'); count_methods['M_planch_step'] += 1
    a_lo, a_hi = max(0, (c0-1)/m), min(4.0, (c0+1)/m)
    print(f"{c0:>4d}|{a_lo:>6.3f}{a_hi:>6.3f}|{tv:>7.4f}{minf:>7.4f}{moff:>7.4f}"
          f"{plnchl:>7.4f}|{','.join(methods_pass) if methods_pass else '-'}")
    results[c0] = {'TV': tv, 'M_inf_step': minf, 'M_off_step': moff,
                   'M_planch_step': plnchl, 'a_lo': a_lo, 'a_hi': a_hi}

print("-" * 95)
print(f"\nSummary: cells caught (> {c_target}) per method (out of {total_cells}):")
soundness = {'TV': 'SOUND (CS Lemma 1)',
             'M_inf_step': 'NOT sound (counter-ex: cont f with ||f*f||_inf < ||f_a*f_a||_inf-0.21)',
             'M_off_step': 'NOT sound (similar to M_inf_step)',
             'M_planch_step': 'EMPIRICALLY sound (no counter-example in adversarial search), '
                              'but FORMAL proof requires extra work'}
for name, count in count_methods.items():
    print(f"  {name}: {count}/{total_cells}  [{soundness[name]}]")

# Sound methods only
print(f"\nSOUND methods catching cells > {c_target}:")
print(f"  TV (CS Lemma 1): {count_methods['TV']}/{total_cells}")

best_method_step = max(['M_inf_step', 'M_planch_step', 'M_off_step'],
                       key=lambda k: count_methods[k])
print(f"\nBest STEP-ansatz method: {best_method_step} catches "
      f"{count_methods[best_method_step]}/{total_cells} cells")

# Final: best EMPIRICALLY-sound method
print(f"\nBest EMPIRICALLY-sound method: M_planch_step (Plancherel-on-step) "
      f"catches {count_methods['M_planch_step']}/{total_cells} cells "
      f"(formal soundness UNPROVEN; counter-example search failed).")

# Save results
with open(os.path.join(_dir, '_smoke_max_t_lower_bound_results.json'), 'w') as f:
    json.dump({
        'mission': 'tighter LB on min over a-in-cell of max_t (f*f)(t) at d=2',
        'config': {'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d},
        'unique_c0_count': total_cells,
        'F_survivor_count': len(F_SURVIVORS),
        'method_counts': count_methods,
        'soundness_notes': soundness,
        'per_cell': {str(c): r for c, r in results.items()},
        'wall_seconds': time.time() - t_start,
    }, f, indent=2)

print(f"\nWall: {time.time() - t_start:.2f}s")
print(f"Results saved to _smoke_max_t_lower_bound_results.json")
