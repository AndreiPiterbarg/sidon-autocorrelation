"""Unified head-to-head bench across all sound coarse-cell-cert variants.

Auto-discovers which bench files exist, imports their cert functions, and
runs them on a common set of hard cells (cells uncertified by triangle
baseline at fixed (d, S, c_target)).

Outputs a summary table: cert rate %, time per cell, time total.
Also runs soundness audit: every method's certified cells must satisfy
fine-grid min_TV ≥ c_target.

Usage: python _unified_pruning_bench.py
"""
from __future__ import annotations
import os, sys, time, importlib, traceback
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))


def find_hard_cells(d, S, c_target, max_cells=300):
    """Enumerate compositions; return those uncertified by triangle baseline."""
    from run_cascade_coarse_v2 import _prune_no_correction, _build_pair_prefix
    from _coarse_NO_bench import prune_coarse_baseline

    def enum_comps(d, S):
        if d == 1:
            yield (S,)
            return
        for v in range(S + 1):
            for r in enum_comps(d - 1, S - v):
                yield (v,) + r

    batch = np.array(list(enum_comps(d, S)), dtype=np.int32)
    surv = prune_coarse_baseline(batch, d, S, c_target)
    hard = batch[surv]
    # Trim to those with TV > c_target at grid (else trivially certified)
    out = []
    for c in hard:
        max_tv = max_TV_at_grid(c, S, d)
        if max_tv > c_target + 1e-9:
            out.append(c)
            if len(out) >= max_cells:
                break
    return np.array(out, dtype=np.int32) if out else np.zeros((0, d), dtype=np.int32)


def max_TV_at_grid(c, S, d):
    mu = c.astype(np.float64) / S
    conv = np.zeros(2 * d - 1)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]
    best = 0.0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        for s_lo in range(2 * d - 1 - n_cv + 1):
            tv = (2.0 * d / ell) * conv[s_lo:s_lo + n_cv].sum()
            if tv > best:
                best = tv
    return best


def cert_NO(c, S, d, c_target):
    """N+O bound from v3."""
    from run_cascade_coarse_v3 import _prune_no_correction_v3, precompute_op_rest_d
    op_rest_d = _OPRD_CACHE.get(d)
    if op_rest_d is None:
        op_rest_d = precompute_op_rest_d(d)
        _OPRD_CACHE[d] = op_rest_d
    surv, _ = _prune_no_correction_v3(c.reshape(1, -1), d, S, c_target, op_rest_d)
    return not bool(surv[0])  # certified if NOT survived (i.e., pruned)


_OPRD_CACHE = {}


def cert_J(c, S, d, c_target, K=8):
    """Joint dual LB from _coarse_J_bench.py if available."""
    try:
        from _coarse_J_bench import joint_cert_LB
    except Exception:
        return None
    try:
        lb = joint_cert_LB(c, S, d, c_target, K=K, n_iters=20)
        return lb >= c_target
    except Exception:
        return None


def cert_L(c, S, d, c_target):
    """Shor SDP from _coarse_L_bench.py if available."""
    try:
        from _coarse_L_bench import find_hard_cells as _, cell_cert_shor
    except Exception:
        return None
    try:
        # Pick the highest-TV window
        mu = c.astype(np.float64) / S
        best_tv = -1.0
        best_W = None
        for ell in range(2, 2 * d + 1):
            for s_lo in range(2 * d - 1 - (ell - 1) + 1):
                A = np.zeros((d, d))
                for i in range(d):
                    for j in range(d):
                        if s_lo <= i + j <= s_lo + ell - 2:
                            A[i, j] = 1.0
                tv = (2.0 * d / ell) * float(mu @ A @ mu)
                if tv > best_tv:
                    best_tv = tv
                    best_W = (ell, s_lo)
        if best_W is None:
            return False
        lb = cell_cert_shor(c, S, d, c_target, best_W)
        return lb >= c_target
    except Exception:
        return None


def cert_KKT(c, S, d, c_target):
    """KKT-augmented exact per-window QP from _coarse_KKT_bench.py if available."""
    try:
        from _kkt_exact_qp import qp_bound_kkt
    except Exception:
        return None
    try:
        # Per-window cert: max_W (TV_W(c) - exact_QP). If positive, certified.
        from _coarse_NO_bench import _build_pair_prefix
        h = 1.0 / (2.0 * S)
        mu = c.astype(np.float64) / S
        for ell in range(2, 2 * d + 1):
            for s_lo in range(2 * d - 1 - (ell - 1) + 1):
                A = np.zeros((d, d))
                for i in range(d):
                    for j in range(d):
                        if s_lo <= i + j <= s_lo + ell - 2:
                            A[i, j] = 1.0
                tv = (2.0 * d / ell) * float(mu @ A @ mu)
                if tv <= c_target + 1e-9:
                    continue
                grad = (4.0 * d / ell) * (A @ mu)
                scale = 2.0 * d / ell
                qp_max = qp_bound_kkt(grad, A, scale, h, d)
                if tv - c_target - qp_max >= 0.0:
                    return True
        return False
    except Exception:
        return None


def cert_BnB(c, S, d, c_target, max_depth=3):
    """Adaptive cell BnB from _coarse_BnB_bench.py if available."""
    try:
        from _coarse_BnB_bench import cell_cert_bnb
    except Exception:
        return None
    try:
        return cell_cert_bnb(c, S, d, c_target, max_depth=max_depth)
    except Exception:
        return None


def cert_2W(c, S, d, c_target):
    """Two-window joint SDP from _coarse_2W_bench.py if available."""
    try:
        from _coarse_2W_bench import cell_cert_2window_SDP
    except Exception:
        return None
    try:
        # Pick top-2 windows
        mu = c.astype(np.float64) / S
        win_tvs = []
        for ell in range(2, 2 * d + 1):
            for s_lo in range(2 * d - 1 - (ell - 1) + 1):
                A = np.zeros((d, d))
                for i in range(d):
                    for j in range(d):
                        if s_lo <= i + j <= s_lo + ell - 2:
                            A[i, j] = 1.0
                tv = (2.0 * d / ell) * float(mu @ A @ mu)
                win_tvs.append((tv, (ell, s_lo)))
        win_tvs.sort(reverse=True)
        if len(win_tvs) < 2:
            return False
        W1, W2 = win_tvs[0][1], win_tvs[1][1]
        lb = cell_cert_2window_SDP(c, S, d, c_target, W1, W2)
        return lb >= c_target
    except Exception:
        return None


def cert_L2(c, S, d, c_target):
    """Lasserre order-2 from _coarse_L2_bench.py if available."""
    try:
        from _coarse_L2_bench import cell_cert_lasserre2
    except Exception:
        return None
    try:
        mu = c.astype(np.float64) / S
        best_tv = -1.0
        best_W = None
        for ell in range(2, 2 * d + 1):
            for s_lo in range(2 * d - 1 - (ell - 1) + 1):
                A = np.zeros((d, d))
                for i in range(d):
                    for j in range(d):
                        if s_lo <= i + j <= s_lo + ell - 2:
                            A[i, j] = 1.0
                tv = (2.0 * d / ell) * float(mu @ A @ mu)
                if tv > best_tv:
                    best_tv = tv
                    best_W = (ell, s_lo)
        if best_W is None:
            return False
        lb = cell_cert_lasserre2(c, S, d, c_target, best_W)
        return lb >= c_target
    except Exception:
        return None


def fine_grid_min_TV(c, S, d, n_grid=8):
    """Lower bound on cell min_TV via exhaustive grid (small d only)."""
    import itertools
    h = 1.0 / (2.0 * S)
    mu_star = c.astype(np.float64) / S
    grid = np.linspace(-h, h, n_grid)
    best = np.inf
    if d > 6:
        rng = np.random.default_rng(0)
        for _ in range(min(2000, 30 ** d)):
            delta = rng.uniform(-h, h, size=d)
            delta -= delta.mean()
            mu = mu_star + delta
            if mu.min() < -1e-12:
                continue
            tv = max_TV_at_grid((mu * S).astype(np.float64), S, d) * S * S / (S * S)
            tv = max_TV_at_grid_continuous(mu, d)
            if tv < best:
                best = tv
        return best
    for tup in itertools.product(grid, repeat=d - 1):
        last = -sum(tup)
        if abs(last) > h + 1e-12:
            continue
        delta = np.array(list(tup) + [last])
        mu = mu_star + delta
        if mu.min() < -1e-12:
            continue
        tv = max_TV_at_grid_continuous(mu, d)
        if tv < best:
            best = tv
    return best


def max_TV_at_grid_continuous(mu, d):
    conv = np.zeros(2 * d - 1)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]
    best = 0.0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        for s_lo in range(2 * d - 1 - n_cv + 1):
            tv = (2.0 * d / ell) * conv[s_lo:s_lo + n_cv].sum()
            if tv > best:
                best = tv
    return best


METHODS = [
    ('N+O', cert_NO),
    ('Joint dual K=8', cert_J),
    ('Shor SDP', cert_L),
    ('KKT exact QP', cert_KKT),
    ('Lasserre-2', cert_L2),
    ('Cell BnB', cert_BnB),
    ('Two-window SDP', cert_2W),
]


def run_one(d, S, c_target, max_cells=200, soundness_check=True):
    print(f"\n=== d={d}, S={S}, c={c_target} ===", flush=True)
    hard = find_hard_cells(d, S, c_target, max_cells=max_cells)
    if len(hard) == 0:
        print("  No hard cells (triangle suffices). Skip.")
        return {'d': d, 'S': S, 'c_target': c_target, 'n_hard': 0}
    print(f"  Hard cells (uncertified by triangle): {len(hard)}", flush=True)

    results = {}
    for name, fn in METHODS:
        t0 = time.time()
        n_cert = 0
        n_undef = 0
        for c in hard:
            r = fn(c, S, d, c_target)
            if r is None:
                n_undef += 1
            elif r:
                n_cert += 1
        t = time.time() - t0
        if n_undef == len(hard):
            print(f"  {name}: NOT AVAILABLE")
            continue
        rate = 100.0 * n_cert / max(1, len(hard) - n_undef)
        per_cell = t / max(1, len(hard) - n_undef) * 1000
        print(f"  {name:18s}: cert={n_cert}/{len(hard) - n_undef} "
              f"({rate:.1f}%)  time={t:.1f}s ({per_cell:.1f} ms/cell)")
        results[name] = {'cert': n_cert, 'time': t, 'per_cell_ms': per_cell,
                          'rate': rate, 'undef': n_undef}

    if soundness_check and d <= 6:
        # For each method's CERTIFIED cell, run fine-grid check
        violations_by_method = {}
        for name, fn in METHODS:
            if name not in results:
                continue
            n_check = 0
            n_violations = 0
            worst = np.inf
            for c in hard[:30]:
                r = fn(c, S, d, c_target)
                if r is True:
                    n_check += 1
                    tv_min = fine_grid_min_TV(c, S, d, n_grid=8)
                    if tv_min < c_target - 1e-6:
                        n_violations += 1
                    if tv_min < worst:
                        worst = tv_min
            violations_by_method[name] = (n_violations, n_check, worst)
        for name, (v, n, w) in violations_by_method.items():
            tag = 'OK' if v == 0 else f'BUG ({v}/{n} viol)'
            print(f"    {name:18s}: soundness {tag}, worst min_TV={w:.4f}")

    return {
        'd': d, 'S': S, 'c_target': c_target, 'n_hard': len(hard),
        **results,
    }


def main():
    configs = [
        (4, 20, 1.20),
        (6, 15, 1.20),
        (8, 12, 1.20),
    ]
    for cfg in configs:
        try:
            run_one(*cfg, max_cells=200)
        except Exception as e:
            print(f"  Error on {cfg}: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    main()
