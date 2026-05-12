"""
v3: bounded parameters, softmax simplex weights, no exp blowup.

Approach:
  - f represented as f = sum_i w_i * basis_i(x) on N-grid where w in simplex (softmax of params).
  - basis: piecewise-constant blocks, OR Gaussian bumps with bounded centers/widths.
  - integral(f * dx) = sum w_i * integral(basis_i * dx). We choose basis so each is normalized to 1.
  - Then ||f||_1 = sum w_i = 1 automatically.
  - Maximize c_emp = ||f*f||_2^2 / ||f*f||_inf SUBJECT TO ||f*f||_inf <= M_target.
    Use logarithmic barrier or SLSQP with constraint.

Key trick: at the maximum, ||f*f||_inf = M_target (boundary), so we maximize
  J(w) = ||f_w * f_w||_2^2 - lambda * (||f_w * f_w||_inf - M_target)^2
with lambda growing.

But actually the SIMPLER approach: just maximize ||f_w*f_w||_2^2 subject to the
"f|_w" giving ||f*f||_inf <= M_target via constraint. Use scipy SLSQP / trust-constr.

Also since c_emp is what we want, and the upper Linf is "soft", we can ALSO
just compute c_emp directly and let Linf float — then check post-hoc whether
it satisfies the M constraint. But then we need to project.

Implementation: use scipy.optimize.minimize with method='trust-constr', supplying
nonlinear constraint g(params) = M - ||f*f||_inf >= 0.
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, basinhopping
from scipy.signal import fftconvolve
import time
import json

print = lambda *a, **k: __builtins__.print(*a, **k, flush=True)

N = 256  # smaller for speed
DX = 0.5 / N
xs = np.linspace(-0.25 + DX/2, 0.25 - DX/2, N)


def conv_self(f):
    return fftconvolve(f, f, mode='full') * DX


def metrics(f):
    g = conv_self(f)
    Linf = float(np.max(g))
    L2sq = float(np.sum(g * g) * DX)
    L1 = float(np.sum(g) * DX)
    return Linf, L2sq, L1, L2sq / Linf if Linf > 0 else 0


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# === Step parameterization ===
# K blocks of equal width; w_i = softmax(z)_i; f = w_i / dx_block on i-th block.
# Then integral(f * dx) = sum w_i = 1. Good.
def f_from_steps_softmax(z, K):
    w = softmax(z)
    block = N // K
    f = np.zeros(N)
    for i in range(K):
        s = i * block
        e = (i+1) * block if i < K-1 else N
        n_pts = e - s
        # density on block = w[i] / (n_pts * DX)
        f[s:e] = w[i] / (n_pts * DX)
    return f


# === Tile parameterization (more flexible: positions+widths) ===
# K bumps; each bump: position p_i in [-0.25,0.25], width sigma_i, weight w_i.
# Use tanh and sigmoid bounds on positions/widths.
def f_from_bumps(params, K):
    # params: 3K
    z = params[:K]
    p_raw = params[K:2*K]
    s_raw = params[2*K:3*K]
    w = softmax(z)
    # positions in [-0.24, 0.24]
    p = np.tanh(p_raw) * 0.24
    # widths in [DX, 0.20]: sigma = DX + 0.20 * sigmoid(s_raw)
    sigma = DX + 0.20 / (1 + np.exp(-s_raw))
    f = np.zeros(N)
    for i in range(K):
        bump = np.exp(-0.5 * ((xs - p[i])/sigma[i])**2)
        Z = np.sum(bump) * DX
        if Z > 0:
            f += w[i] * bump / Z
    return f


# === BL-mix loader ===
_BL = None
def load_BL():
    global _BL
    if _BL is None:
        with open('delsarte_dual/restricted_holder/coeffBL.txt') as fh:
            s = fh.read().strip().strip('{}').split(',')
        _BL = np.array([int(x) for x in s], dtype=float)
    return _BL


def f_from_BL_scaled(scale_factor=1.0, head_keep=None):
    v = load_BL().copy()
    if head_keep is not None:
        v[head_keep:] = 0
    L = len(v)
    half = 0.25 * scale_factor
    bin_idx = ((xs + half) / (2*half/L)).astype(int)
    bin_idx = np.clip(bin_idx, 0, L-1)
    in_support = (xs >= -half) & (xs < half)
    f = np.where(in_support, v[bin_idx], 0.0)
    s = float(np.sum(f) * DX)
    if s <= 0:
        return None
    return f / s


# === Optimizer ===
def opt_for_M(K, parametrization, M_target, n_restarts=8, asymmetric=True, verbose=False):
    """Maximize c_emp subject to Linf <= M_target. Returns best (c_emp, f, info)."""
    rng = np.random.default_rng(42)
    if parametrization == 'step':
        n_params = K
    elif parametrization == 'bump':
        n_params = 3 * K
    else:
        raise ValueError

    def f_of(p):
        if parametrization == 'step':
            return f_from_steps_softmax(p, K)
        else:
            return f_from_bumps(p, K)

    def neg_c_emp(p):
        f = f_of(p)
        Linf, L2sq, _, c_emp = metrics(f)
        return -c_emp  # maximize c_emp

    def Linf_of(p):
        return metrics(f_of(p))[0]

    constraint = NonlinearConstraint(Linf_of, -np.inf, M_target)

    best = (-np.inf, None, None)
    for r in range(n_restarts):
        if parametrization == 'step':
            if asymmetric:
                p0 = rng.normal(0, 1, K)
            else:
                half = (K+1)//2
                tmp = rng.normal(0, 1, half)
                p0 = np.concatenate([tmp, tmp[:K-half][::-1]])
        else:  # bump
            z0 = rng.normal(0, 1, K)
            if asymmetric:
                p0_pos = rng.uniform(-1, 1, K)
            else:
                half = (K+1)//2
                tmp = rng.uniform(-1, 1, half)
                p0_pos = np.concatenate([-tmp, tmp[:K-half]])
            p0_sig = rng.normal(-1, 0.5, K)  # sigmoid(-1) ~ 0.27 -> sigma ~ 0.054
            p0 = np.concatenate([z0, p0_pos, p0_sig])

        try:
            res = minimize(neg_c_emp, p0, method='trust-constr',
                           constraints=[constraint],
                           options={'maxiter': 80, 'xtol': 1e-7, 'gtol': 1e-6, 'verbose': 0})
        except Exception as e:
            if verbose:
                print(f"  restart {r}: {e}")
            continue
        f = f_of(res.x)
        Linf, L2sq, _, c_emp = metrics(f)
        if Linf > M_target * 1.005:
            continue
        if c_emp > best[0]:
            best = (c_emp, f, {'Linf': Linf, 'L2sq': L2sq, 'restart': r})
    return best


def adversarial_BL(M_target):
    results = []
    # Indicator on full support
    ind = np.full(N, 2.0)
    for head_keep in [None, 50, 80, 120, 200, 300]:
        for sf in [0.5, 0.65, 0.80, 1.0]:
            try:
                f_BL = f_from_BL_scaled(scale_factor=sf, head_keep=head_keep)
                if f_BL is None:
                    continue
                # Mix to get Linf <= M_target. Find max alpha satisfying constraint via bisection.
                # Linf(alpha) = ||(alpha*f_BL + (1-alpha)*ind)*self|| inf, complicated.
                # Just try grid of alpha:
                for alpha in np.linspace(0.05, 1.0, 20):
                    f_mix = alpha * f_BL + (1-alpha) * ind
                    s = np.sum(f_mix) * DX
                    if s <= 0:
                        continue
                    f_mix = f_mix / s
                    Linf, L2sq, L1, c_emp = metrics(f_mix)
                    if Linf <= M_target * 1.005:
                        results.append((c_emp, f_mix,
                                        {'kind': 'BL_mix', 'head_keep': str(head_keep),
                                         'sf': sf, 'alpha': float(alpha), 'Linf': Linf}))
            except Exception:
                continue
    # Also try BL alone (truncated head only) — no mixing
    for head_keep in [40, 50, 60, 80, 100, 150, 200]:
        for sf in [0.5, 0.65, 0.8, 1.0]:
            try:
                f_BL = f_from_BL_scaled(scale_factor=sf, head_keep=head_keep)
                if f_BL is None:
                    continue
                Linf, L2sq, L1, c_emp = metrics(f_BL)
                if Linf <= M_target * 1.005:
                    results.append((c_emp, f_BL,
                                    {'kind': 'BL_pure', 'head_keep': str(head_keep),
                                     'sf': sf, 'alpha': 1.0, 'Linf': Linf}))
            except Exception:
                continue
    return results


def asymmetry(f):
    f_rev = f[::-1]
    num = float(np.sum(np.abs(f - f_rev)) * DX)
    den = float(np.sum(np.abs(f + f_rev)) * DX)
    return num / den if den > 0 else 0


def support_extent(f, threshold_frac=0.001):
    thr = threshold_frac * np.max(f) if np.max(f) > 0 else 0
    mask = f > thr
    if not mask.any():
        return (0.0, 0.0)
    idx = np.where(mask)[0]
    return float(xs[idx[0]]), float(xs[idx[-1]])


def describe_f(f):
    Linf, L2sq, L1, c_emp = metrics(f)
    asym = asymmetry(f)
    sup_lo, sup_hi = support_extent(f)
    peak_x = float(xs[np.argmax(f)])
    g = conv_self(f)
    g_xs = np.linspace(-0.5, 0.5, 2*N-1)
    peak_g_x = float(g_xs[np.argmax(g)])
    return {
        'c_emp': c_emp, 'M': Linf, 'L2sq': L2sq, 'L1': L1,
        'asymmetry': asym, 'support_lo': sup_lo, 'support_hi': sup_hi,
        'support_width': sup_hi - sup_lo,
        'peak_f_x': peak_x, 'peak_g_x': peak_g_x,
        'f_max': float(np.max(f)),
        'norm_f_sq_K': float(np.sum(f*f) * DX),
    }


def main():
    c_star = float(np.log(16) / np.pi)
    print(f"c_star = log16/pi = {c_star:.10f}")
    print(f"N={N}, dx={DX:.4e}\n")

    M_grid = [1.27, 1.30, 1.33, 1.35, 1.378]
    out = {'c_star': c_star, 'N': N, 'results': {}}
    t_start = time.time()

    configs = [
        ('step', 8), ('step', 16), ('step', 32), ('step', 64),
        ('bump', 3), ('bump', 5), ('bump', 8),
    ]

    for M in M_grid:
        print(f"=" * 60)
        print(f"M_target = {M}  (elapsed {time.time()-t_start:.1f}s)")
        print(f"=" * 60)
        best = (-np.inf, None, None)

        # 1) Adversarial BL first (cheap, often best)
        adv = adversarial_BL(M)
        for (c_emp, f, info) in adv:
            if c_emp > best[0]:
                best = (c_emp, f, info)
        if best[2] is not None:
            print(f"  ADV best: c={best[0]:.5f}, info={best[2]}")

        # 2) Trust-constr optimization
        for (param, K) in configs:
            for asym in [False, True]:
                t0 = time.time()
                try:
                    c_emp, f, info = opt_for_M(K, param, M, n_restarts=4, asymmetric=asym)
                except Exception as e:
                    print(f"  {param}-K{K}-{'asy' if asym else 'sym'}: ERR {e}")
                    continue
                if f is None:
                    continue
                tag = f"{param}-K{K}-{'asy' if asym else 'sym'}"
                print(f"  {tag:18s}: c={c_emp:.5f}, Linf={info['Linf']:.4f}, t={time.time()-t0:.1f}s")
                if c_emp > best[0]:
                    best = (c_emp, f, {'param': param, 'K': K, 'asym': asym, **info})

        c_best, f_best, info_best = best
        if f_best is None:
            print(f"  *** NO FEASIBLE for M={M}")
            continue
        desc = describe_f(f_best)
        margin = c_star - c_best
        verdict = 'PROMISING' if margin > 0.01 else ('NEAR-CRITICAL' if margin > -0.001 else 'DISPROVED')
        print(f"\n  BEST at M={M}: c_emp={c_best:.6f}, margin to c*={margin:+.6f}, {verdict}")
        print(f"  STRUCT: asym={desc['asymmetry']:.4f}, K=||f||_2^2={desc['norm_f_sq_K']:.3f}, "
              f"sup=[{desc['support_lo']:.3f},{desc['support_hi']:.3f}] (w={desc['support_width']:.3f}), "
              f"f_peak_x={desc['peak_f_x']:.3f}, source={info_best.get('kind', info_best.get('param'))}")
        out['results'][str(M)] = {
            'best_c_emp': float(c_best),
            'best_info': {k: str(v) for k, v in info_best.items()},
            'description': desc,
            'distance_to_threshold': float(margin),
            'verdict': verdict,
        }
        np.save(f'delsarte_dual/path_a_unconditional_holder/f_best_M{M}.npy', f_best)

    with open('delsarte_dual/path_a_unconditional_holder/phi_curve.json', 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nSaved phi_curve.json (elapsed {time.time()-t_start:.1f}s)")

    print("\n" + "=" * 60)
    print("PHI(M) CURVE -- SUMMARY")
    print("=" * 60)
    print(f"  c_star (threshold) = {c_star:.6f}")
    for M in M_grid:
        if str(M) not in out['results']:
            continue
        cb = out['results'][str(M)]['best_c_emp']
        margin = c_star - cb
        v = out['results'][str(M)]['verdict']
        print(f"  M={M}: Phi_emp >= {cb:.6f}, margin = {margin:+.6f}, {v}")


if __name__ == '__main__':
    main()
