"""AGENT B probe — FW direction analysis on stagnant cells from _v6_bottleneck.json.

For 2-3 stagnant cells, instrument FW iter 1 and report:
 - warm-start vertex k0 = argmax_k L_single_k
 - FW direction s_idx = argmax_W g_W where g_W = f_W(eps*, X*)
 - whether s_idx == k0  (=> FW immediately re-snaps to warm-start vertex)

NO production edits. Stand-alone analysis script.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v6 as v6


def probe_full_fw(c, d, S, c_target, K=3, max_iters=6):
    """Re-run full FW loop and report per-iter (lambda, lb_t, s_idx, subgrad gap)."""
    cell = v3.Cell.from_integer_composition(np.array(c, dtype=np.int64), S=S)
    cell_cache = v3.CellCache.build(cell)
    bundle = v6.get_bundle(v3.build_all_windows(d))
    mw_vec = v6.mwQ_vec(cell, bundle, c_target, cell_cache.mu_star)
    idxs = v6.select_L_candidates_v6(mw_vec, bundle, K=K)
    top_windows = [bundle.windows[i] for i in idxs]
    template = v6.get_sdp_template_v6(d)
    mu_star = cell_cache.mu_star

    L_single = []
    for W in top_windows:
        m_W = W.Q_coef * float(mu_star @ W.A @ mu_star) - c_target
        g_W = W.grad_coef * (W.A @ mu_star)
        Q_W = W.Q_coef * W.A
        lb, _ = template.solve(cell_cache.lo_eps, cell_cache.hi_eps, mu_star, Q_W, g_W, m_W)
        L_single.append(lb if lb is not None else -np.inf)
    best_k = int(np.argmax(L_single))
    best_lb = float(L_single[best_k])
    lam = np.eye(K)[best_k]
    log = []
    prev_lb = best_lb
    for t in range(1, max_iters + 1):
        Q_lam, g_lam, m_lam = v6._aggregate(top_windows, lam, mu_star, c_target)
        lb_t, info_t = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                        mu_star, Q_lam, g_lam, m_lam)
        if lb_t is None: break
        eps_val = info_t.get('eps'); X_val = info_t.get('X')
        if eps_val is None: break
        g_sub = np.array([v6._fW_value(W, mu_star, eps_val, X_val, c_target) for W in top_windows])
        s_idx = int(np.argmax(g_sub))
        log.append({
            't': t,
            'lam': [round(float(x),4) for x in lam],
            'lb_t': float(lb_t),
            's_idx': s_idx,
            'g_sub': [float(x) for x in g_sub],
            'gap_W_minus_lam_argmax': float(g_sub.max() - g_sub[best_k]),
        })
        if lb_t > best_lb: best_lb = float(lb_t)
        if t > 1 and abs(lb_t - prev_lb) < 1e-7: break
        prev_lb = lb_t
        eta = 2.0 / (t + 2.0)
        lam = (1.0 - eta) * lam + eta * np.eye(K)[s_idx]
    return {'c': list(c), 'd': d, 'L_single': L_single,
              'best_k': best_k, 'best_lb_final': best_lb, 'log': log}


def probe_cell(c, d, S, c_target, K=3):
    cell = v3.Cell.from_integer_composition(np.array(c, dtype=np.int64), S=S)
    if v6.is_cell_empty(cell):
        return {'skipped': 'empty'}
    cell_cache = v3.CellCache.build(cell)
    bundle = v6.get_bundle(v3.build_all_windows(d))
    mw_vec = v6.mwQ_vec(cell, bundle, c_target, cell_cache.mu_star)
    idxs = v6.select_L_candidates_v6(mw_vec, bundle, K=K)
    top_windows = [bundle.windows[i] for i in idxs]
    template = v6.get_sdp_template_v6(d)

    # Compute L_single + retain (eps*, X*) for each
    L_single = []
    per_W = []
    for W in top_windows:
        m_W = W.Q_coef * float(cell_cache.mu_star @ W.A @ cell_cache.mu_star) - c_target
        g_W = W.grad_coef * (W.A @ cell_cache.mu_star)
        Q_W = W.Q_coef * W.A
        lb, info = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                    cell_cache.mu_star, Q_W, g_W, m_W)
        L_single.append(lb if lb is not None else -np.inf)
        per_W.append((lb, info))
    best_k = int(np.argmax(L_single))
    best_lb = float(L_single[best_k])

    # === Simulate FW iter 1 starting from warm-start vertex ===
    lam = np.eye(K)[best_k]
    Q_lam, g_lam, m_lam = v6._aggregate(top_windows, lam, cell_cache.mu_star, c_target)
    lb_t, info_t = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                    cell_cache.mu_star, Q_lam, g_lam, m_lam)
    # because lam = e_{best_k}, this lb_t should equal L_single[best_k] to MOSEK tol
    eps_val = info_t.get('eps')
    X_val = info_t.get('X')
    # subgradient at warm-start optimizer
    g_subgrad = np.array([v6._fW_value(W, cell_cache.mu_star, eps_val, X_val, c_target)
                            for W in top_windows])
    s_idx = int(np.argmax(g_subgrad))

    # === SADDLE-POINT CHECK ===
    # If lam=e_{best_k}, the optimizer (eps*, X*) minimizes f_{best_k}.
    # subgrad g_W = f_W(eps*, X*).  At a saddle point of max_λ min_{ε,X} Σ λ_W f_W,
    # we need g_{best_k} >= g_W for all W (so that no FW direction increases).
    # g_{best_k} == L_single[best_k] = lb_t (envelope identity).
    # If s_idx == best_k, the warm-start IS the unconstrained Lagrangian saddle.

    # === Closed-form min-max (3-point grid line search) ===
    # Try lam_alpha = (1-alpha)*e_{best_k} + alpha*e_s_idx for alpha in {0,0.1,0.5,1.0}
    # and report the values lb(alpha).
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    lb_alpha = []
    if s_idx != best_k:
        for a in alphas:
            lam_a = (1 - a) * np.eye(K)[best_k] + a * np.eye(K)[s_idx]
            Q_a, g_a, m_a = v6._aggregate(top_windows, lam_a, cell_cache.mu_star, c_target)
            lb_a, _ = template.solve(cell_cache.lo_eps, cell_cache.hi_eps,
                                       cell_cache.mu_star, Q_a, g_a, m_a)
            lb_alpha.append((a, lb_a))

    return {
        'c': list(c), 'd': d, 'S': S,
        'L_single': [float(x) for x in L_single],
        'best_k_idx': best_k,
        'best_lb': best_lb,
        'lb_t_iter1': float(lb_t) if lb_t is not None else None,
        'g_subgrad': g_subgrad.tolist(),
        's_idx_iter1': s_idx,
        's_equals_warm': (s_idx == best_k),
        'subgrad_gap_W_minus_warm': float(g_subgrad[s_idx] - g_subgrad[best_k]),
        'line_search': lb_alpha,
    }


def main():
    bn = json.load(open('_v6_bottleneck.json'))
    c_target = 1.281
    out = []
    # pick 2 stagnant rows from each d  +  improved rows for contrast
    for k in ['rows4','rows6','rows8']:
        rows = bn.get(k,[])
        stag = [r for r in rows
                if r.get('L_best') is not None and r.get('Lj_best') is not None
                and abs(r['L_best'] - r['Lj_best']) < 1e-9
                and r.get('Lj_iters', 0) > 0]
        improved = [r for r in rows
                    if r.get('L_best') is not None and r.get('Lj_best') is not None
                    and (r['Lj_best'] - r['L_best']) > 1e-4]
        # pick 2 with most-negative L_best (closest to certification gap)
        stag.sort(key=lambda r: r['L_best'], reverse=True)
        for r in stag[:2]:
            print(f"[STAG] d={r['d']} c={r['c']} L_best={r['L_best']:.5f}")
            res = probe_cell(r['c'], r['d'], r['S'], c_target)
            res['tag'] = 'stagnant'
            out.append(res)
            print('   s_idx==warm?', res.get('s_equals_warm'),
                    'subgrad_gap=', round(res.get('subgrad_gap_W_minus_warm', 0), 6))
        for r in improved[:1]:
            print(f"[IMP ] d={r['d']} c={r['c']} L_best={r['L_best']:.5f} -> Lj_best={r['Lj_best']:.5f}")
            res = probe_cell(r['c'], r['d'], r['S'], c_target)
            res['tag'] = 'improved'
            out.append(res)
            print('   s_idx==warm?', res.get('s_equals_warm'),
                    'subgrad_gap=', round(res.get('subgrad_gap_W_minus_warm', 0), 6))
            if res.get('line_search'):
                print('   line_search:', [(a, None if v is None else round(v,5))
                                            for a, v in res['line_search']])

    # Now run FULL FW loop on the 2 known FW-improving cells (rows6 / rows8)
    print('\n=== Full FW trace on the 2 truly-FW-improving cells ===')
    cases = [
        {'c': [7,5,2,3,4,9], 'd':6, 'S':20},
        {'c': [6,1,3,1,3,0,1,1], 'd':8, 'S':20},
    ]
    for case in cases:
        # auto-pick S from bottleneck file
        bn_d = bn.get(f'rows{case["d"]}',[])
        for r in bn_d:
            if r['c'] == case['c']:
                case['S'] = r['S']
                break
        res = probe_full_fw(case['c'], case['d'], case['S'], c_target)
        out.append({'tag': 'full_fw_trace', **res})
        print(f"\n d={case['d']} c={case['c']}  L_single={[round(x,4) for x in res['L_single']]}  best_k={res['best_k']}  final={res['best_lb_final']:.5f}")
        for entry in res['log']:
            print(f"    t={entry['t']}  lam={entry['lam']}  lb_t={entry['lb_t']:.5f}  s_idx={entry['s_idx']}  gap={entry['gap_W_minus_lam_argmax']:.5f}")

    json.dump(out, open('_audit_B_probe.json','w'), indent=2)
    print('\nwrote _audit_B_probe.json')


if __name__ == '__main__':
    main()
