"""Palindromic box certification — diagnostic and soundness analysis.

See `_palindromic_box_cert_proof.txt` for the proof.

Summary: for palindromic-canonical centers c, restricting the cell
variation polytope to the palindromic subspace (delta_i = delta_{d-1-i})
gives a SMALLER cell_var (better certificate) but is UNSOUND because
the cell minimizer of T may be NON-palindromic.

This module:
  1. Computes `cell_var_full(c)` = standard linear cell variation.
  2. Computes `cell_var_sym(c)` = cell variation restricted to the
     palindromic subspace. ALWAYS <= cell_var_full.
  3. Verifies the soundness gap: builds an explicit non-palindromic
     mu in Cell with T(mu) < T(c/S) - cell_var_sym, demonstrating
     that the sym bound would CERTIFY a cell where T actually drops
     more than the sym bound predicts.
  4. Bench on 30 uncertified cells at d=4, S=200, c=1.281.
"""
import math
import os
import sys
import time
import json

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_this_dir, 'cloninger-steinerberger', 'cpu'))


# ---------------------------------------------------------------------
# Building blocks (mirror the kernel in run_cascade_coarse.py).
# ---------------------------------------------------------------------

def autoconv_int(c_int):
    """Integer auto-convolution of integer mass vector c_int (sums to S)."""
    d = len(c_int)
    conv = np.zeros(2 * d - 1, dtype=np.int64)
    for i in range(d):
        ci = int(c_int[i])
        if ci == 0:
            continue
        conv[2 * i] += ci * ci
        for j in range(i + 1, d):
            conv[i + j] += 2 * ci * int(c_int[j])
    return conv


def best_window(c_int, S, c_target):
    """Return (ell, s, ws_int, TV, margin) for the best window (largest TV)
    among windows that prune (TV >= c_target). If none prune, returns None."""
    d = len(c_int)
    conv = autoconv_int(c_int)
    conv_len = 2 * d - 1
    eps = 1e-9
    inv_2d = 1.0 / (2.0 * d)
    S2 = float(S) * float(S)
    best = None
    for ell in range(2, 2 * d + 1):
        thr = int(c_target * float(ell) * S2 * inv_2d - eps)
        n_cv = ell - 1
        for s in range(conv_len - n_cv + 1):
            ws = int(conv[s:s + n_cv].sum())
            if ws > thr:
                tv = ws * 2.0 * d / (S2 * ell)
                margin = tv - c_target
                if best is None or tv > best[3]:
                    best = (ell, s, ws, tv, margin)
    return best


def gradient(c_int, S, ell, s):
    """grad_i TV_W at mu = c_int / S (vector of length d)."""
    d = len(c_int)
    scale = 4.0 * d / float(ell)
    grad = np.zeros(d, dtype=np.float64)
    for i in range(d):
        g = 0.0
        for j in range(d):
            k = i + j
            if s <= k <= s + ell - 2:
                g += float(c_int[j]) / float(S)
        grad[i] = g * scale
    return grad


def cell_var_full(grad, S):
    """Standard pair-extremes cell variation (full polytope).

    cell_var_full = max_{delta in Cell0} |grad . delta|
    where Cell0 = { delta : |delta_i| <= h, sum_i delta_i = 0 }
    with h = 1/(2S). Closed form: pair sorted gradient extremes.
    """
    d = len(grad)
    g = np.sort(grad)
    cv = 0.0
    for k in range(d // 2):
        cv += g[d - 1 - k] - g[k]
    return cv / (2.0 * S)


def cell_var_sym(grad, S):
    """Cell variation restricted to palindromic delta.

    Cell0_sym = { delta in Cell0 : delta_i = delta_{d-1-i} }.
    Effective vars: delta_i for i in [0, ceil(d/2) - 1], with
      - if d even: pairs (i, d-1-i), i in [0, d/2 - 1]; constraint
        2 * sum_i delta_i = 0  =>  sum_{i=0..d/2-1} delta_i = 0
        Effective gradient on each pair is (g_i + g_{d-1-i}). And the
        per-pair extent is +-h. So cell_var_sym = max_{e in {+1,-1}^{d/2},
        sum e_i = 0} sum_i e_i * (g_i + g_{d-1-i}) * h.
      - if d odd: middle index m = (d-1)/2 has 2 * delta_m absorbed
        once. We handle both via direct LP.

    NOTE: This is UNSOUND as a substitute for cell_var_full — it is a
    diagnostic only.
    """
    d = len(grad)
    h = 1.0 / (2.0 * S)
    if d == 1:
        return 0.0

    # Reduce: in palindromic delta, free vars are delta_i for i < d/2,
    # plus delta_{(d-1)/2} if d odd. Effective gradient on free var i:
    #   e_grad[i] = grad[i] + grad[d - 1 - i] for i < d // 2
    #   e_grad[mid] = grad[mid]  (only one factor) when d odd
    # Effective constraint: 2 * sum_{i < d/2} delta_i + (d-odd) * delta_mid = 0.
    half = d // 2
    e_grad = np.array([grad[i] + grad[d - 1 - i] for i in range(half)],
                      dtype=np.float64)
    coef = np.full(half, 2.0)  # constraint coefficient
    bounds = np.full(half, h)  # |delta_i| <= h
    if d % 2 == 1:
        mid = d // 2
        e_grad = np.concatenate([e_grad, [grad[mid]]])
        coef = np.concatenate([coef, [1.0]])
        bounds = np.concatenate([bounds, [h]])

    # max sum e_grad[i] * delta[i] s.t. -bounds[i] <= delta[i] <= bounds[i],
    # sum coef[i] * delta[i] = 0.
    # Solve via dual LP / vertex enumeration.
    # By LP theory, optimum is at a vertex: at most 1 var fractional.
    # Enumerate sign patterns sigma in {+1, -1}^k for k = len(e_grad);
    # at vertex, all but one var saturated to +-bounds, and the
    # remaining var pinned by sum constraint. Take max obj.
    from scipy.optimize import linprog
    n = len(e_grad)
    # Solve max e_grad^T delta = -min (-e_grad)^T delta
    # subject to coef^T delta = 0, -bounds <= delta <= bounds.
    A_eq = coef.reshape(1, n)
    b_eq = np.array([0.0])
    bnds = [(-float(b), float(b)) for b in bounds]
    res = linprog(c=-e_grad, A_eq=A_eq, b_eq=b_eq, bounds=bnds,
                  method='highs')
    if not res.success:
        return cell_var_full(grad, S)  # fall back conservatively
    # Symmetric polytope: |obj_max| = |obj_min|; cell_var = max value.
    return abs(res.fun)


# ---------------------------------------------------------------------
# Cascade-style certificate net.
# ---------------------------------------------------------------------

def quadratic_remainder(d, ell, s, S):
    """|R_W| <= (2d/ell) * N_pairs / (4 S^2)."""
    n_pairs = 0
    for k in range(s, s + ell - 1):
        cnt = min(k + 1, d)
        if k > d - 1:
            cnt = min(cnt, 2 * d - 1 - k)
        n_pairs += cnt
    return (2.0 * d / float(ell)) * n_pairs / (4.0 * float(S) * float(S))


def cert_net(c_int, S, c_target, mode='full'):
    """Box-cert net = margin - cell_var(mode) - |R|."""
    d = len(c_int)
    bw = best_window(c_int, S, c_target)
    if bw is None:
        return None  # not even pruned at center
    ell, s, ws, tv, margin = bw
    grad = gradient(c_int, S, ell, s)
    if mode == 'full':
        cv = cell_var_full(grad, S)
    elif mode == 'sym':
        cv = cell_var_sym(grad, S)
    else:
        raise ValueError(mode)
    R = quadratic_remainder(d, ell, s, S)
    return {'ell': ell, 's': s, 'tv': tv, 'margin': margin,
            'cell_var': cv, 'R': R, 'net': margin - cv - R}


def palindromic_box_cert(c_int, S, c_target):
    """Full + sym diagnostics for one center.

    Returns dict { full: {...}, sym: {...},
                   reduction: 1 - cv_sym/cv_full,
                   net_full, net_sym, sym_helps }
    """
    f = cert_net(c_int, S, c_target, mode='full')
    s = cert_net(c_int, S, c_target, mode='sym')
    if f is None or s is None:
        return None
    rd = 1.0 - (s['cell_var'] / f['cell_var']) if f['cell_var'] > 0 else 0.0
    return {
        'full': f, 'sym': s,
        'reduction': rd,
        'net_full': f['net'], 'net_sym': s['net'],
        'sym_helps': (f['net'] < 0 and s['net'] >= 0),
    }


# ---------------------------------------------------------------------
# Soundness counter-witness search:
# build a NON-palindromic mu in Cell(c) with measurable TV drop > sym bound.
# ---------------------------------------------------------------------

def find_witness(c_int, S, c_target, n_trials=200, rng=None):
    """Search for non-palindromic mu in Cell(c) with TV(mu) < c_target +
    cell_var_sym, refuting the symmetric-only bound.

    Returns dict with witness if found, else None.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = len(c_int)
    bw = best_window(c_int, S, c_target)
    if bw is None:
        return None
    ell, s, ws, tv0, margin = bw
    grad = gradient(c_int, S, ell, s)
    cv_sym = cell_var_sym(grad, S)
    cv_full = cell_var_full(grad, S)
    h = 1.0 / (2.0 * S)

    # Sample anti-palindromic perturbations: delta_i = -delta_{d-1-i}, |delta_i| <= h, sum = 0.
    # If d even: anti-palindromic delta automatically has sum = 0.
    #            free var: delta_i for i < d/2, with delta_{d-1-i} = -delta_i.
    # If d odd: middle delta_m = 0, free vars = delta_i for i < d/2.
    half = d // 2
    best_mu = None
    best_tv = None

    for trial in range(n_trials):
        # random in [-h, h]^half
        a = (2.0 * rng.random(half) - 1.0) * h
        delta = np.zeros(d)
        for i in range(half):
            delta[i] = a[i]
            delta[d - 1 - i] = -a[i]
        mu = c_int.astype(np.float64) / float(S) + delta
        if np.any(mu < -1e-12):
            continue
        # Use SAME window (ell, s) — check TV at this mu
        # TV = (2d/ell) * sum_{i+j: s<=i+j<=s+ell-2} mu_i mu_j
        tv = 0.0
        for i in range(d):
            for j in range(d):
                k = i + j
                if s <= k <= s + ell - 2:
                    tv += mu[i] * mu[j]
        tv *= 2.0 * d / float(ell)
        # Check the strongest window over all W
        # NB: T(mu) >= TV_W(mu) for our W; but to be precise, we want T(mu).
        # Conservative: take T(mu) >= this TV_W.
        if best_tv is None or tv < best_tv:
            best_tv = tv
            best_mu = mu.copy()

    if best_mu is None:
        return None
    # Evaluate full T(mu) (max over all windows)
    tv_max_at_witness = 0.0
    win_at_witness = None
    for elln in range(2, 2 * d + 1):
        for sn in range(2 * d - elln + 1):
            tv = 0.0
            for i in range(d):
                for j in range(d):
                    k = i + j
                    if sn <= k <= sn + elln - 2:
                        tv += best_mu[i] * best_mu[j]
            tv *= 2.0 * d / float(elln)
            if tv > tv_max_at_witness:
                tv_max_at_witness = tv
                win_at_witness = (elln, sn)

    drop = tv0 - tv_max_at_witness
    return {
        'best_mu': best_mu,
        'tv_at_witness': float(tv_max_at_witness),
        'best_window_at_witness': win_at_witness,
        'tv_at_center': float(tv0),
        'drop': float(drop),
        'cv_sym': float(cv_sym),
        'cv_full': float(cv_full),
        'soundness_violated_sym': float(drop) > float(cv_sym) + 1e-12,
        'soundness_violated_full': float(drop) > float(cv_full) + 1e-9,
    }


# ---------------------------------------------------------------------
# Bench: 30 uncertified cells at d=4, S=200, c=1.281.
# ---------------------------------------------------------------------

def find_uncert_cells(d, S, c_target, n_target=30):
    """Iterate over palindromic-canonical compositions and return up to
    n_target cells where net_full < 0 (uncertified by full bound)."""
    out = []
    # Enumerate compositions of S in d non-neg ints with c sorted such
    # that c <= rev(c) lexicographically (palindromic canonical).
    # For small d, we can enumerate directly.
    def gen(d, S):
        if d == 1:
            yield (S,)
            return
        for v in range(S + 1):
            for rest in gen(d - 1, S - v):
                yield (v,) + rest

    seen = set()
    for c in gen(d, S):
        # canonical filter: c <= rev(c) lex
        rev = c[::-1]
        if c > rev:
            continue
        c_arr = np.array(c, dtype=np.int32)
        bw = best_window(c_arr, S, c_target)
        if bw is None:
            continue  # not even pruned at the grid point — skip
        f = cert_net(c_arr, S, c_target, mode='full')
        if f is None:
            continue
        if f['net'] < 0:
            key = c
            if key in seen:
                continue
            seen.add(key)
            out.append(c_arr)
            if len(out) >= n_target:
                return out
    return out


def bench(d=4, S=200, c_target=1.281, n_target=30):
    print(f"\n{'='*60}")
    print(f"PALINDROMIC BOX CERT BENCH")
    print(f"  d={d}, S={S}, c_target={c_target}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    cells = find_uncert_cells(d, S, c_target, n_target=n_target)
    n = len(cells)
    print(f"\nFound {n} uncertified palindromic-canonical cells "
          f"({time.time() - t0:.1f}s)", flush=True)

    if n == 0:
        print("  No uncertified cells; bench has nothing to compare.")
        return {'n': 0}

    rows = []
    n_sym_pass = 0
    n_full_pass = 0
    n_witness = 0
    for i, c_int in enumerate(cells):
        diag = palindromic_box_cert(c_int, S, c_target)
        if diag is None:
            continue
        wit = find_witness(c_int, S, c_target, n_trials=300)

        diag['c'] = c_int.tolist()
        diag['cv_full'] = diag['full']['cell_var']
        diag['cv_sym'] = diag['sym']['cell_var']
        diag['witness_drop'] = wit['drop'] if wit else None
        diag['violates_sym'] = wit['soundness_violated_sym'] if wit else None
        diag['violates_full'] = wit['soundness_violated_full'] if wit else None
        rows.append(diag)
        if diag['net_sym'] >= 0:
            n_sym_pass += 1
        if diag['net_full'] >= 0:
            n_full_pass += 1
        if wit and wit['soundness_violated_sym']:
            n_witness += 1
        is_pal = list(c_int) == list(c_int[::-1])
        print(f"  [{i+1}/{n}] c={c_int.tolist()} "
              f"pal={is_pal} cv_full={diag['cv_full']:.4f} "
              f"cv_sym={diag['cv_sym']:.4f} "
              f"red={diag['reduction']*100:.1f}% "
              f"net_full={diag['net_full']:+.4f} "
              f"net_sym={diag['net_sym']:+.4f} "
              f"witness_violates_sym={wit['soundness_violated_sym'] if wit else 'N/A'}",
              flush=True)

    summary = {
        'd': d, 'S': S, 'c_target': c_target,
        'n_cells': n,
        'n_full_pass': n_full_pass,
        'n_sym_pass': n_sym_pass,
        'n_witnesses_violating_sym': n_witness,
        'mean_reduction_cell_var': float(np.mean(
            [r['reduction'] for r in rows])) if rows else 0.0,
        'rows': rows,
    }
    print(f"\nSUMMARY")
    print(f"  Cells: {n}")
    print(f"  Box-cert pass with FULL cell_var: {n_full_pass}/{n}")
    print(f"  Box-cert pass with SYM cell_var (UNSOUND): {n_sym_pass}/{n}")
    print(f"  Mean cell_var reduction (sym vs full): "
          f"{summary['mean_reduction_cell_var']*100:.1f}%")
    print(f"  Witnesses: {n_witness}/{n} cells exhibit "
          f"non-palindromic mu with drop > cv_sym (UNSOUND CONFIRMED)")
    return summary


def main():
    summary = bench(d=4, S=200, c_target=1.281, n_target=30)
    out_path = os.path.join(_this_dir, '_palindromic_box_cert_results.json')
    # Strip non-serializable numpy in rows
    def serialize(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, dict):
            return {k: serialize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [serialize(x) for x in o]
        return o

    with open(out_path, 'w') as f:
        json.dump(serialize(summary), f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
