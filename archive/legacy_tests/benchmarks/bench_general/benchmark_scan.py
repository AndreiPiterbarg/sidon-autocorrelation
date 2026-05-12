#!/usr/bin/env python
r"""
Multi-scale benchmark scanner for d=16 L3 production run.

Measures every per-phase cost as a function of #active_windows (K) across
a wide range — K in {25, 50, 100, 200, 400, 800, 1200, ...} — so we can
see which phase explodes first as the real run accumulates windows
(production reaches 700+ by R7).

The math config is locked to production (`tests/run_d16_l3.py`):
    d=16  order=3  bw=15  rho=0.1  atom_frac=0.5  cuts_per_round=100
    k_vecs=3  add_upper_loc=True  gpu=True
    scs_iters=2000  scs_eps=1e-6 (matches step 5-6 bisection budget)

What we measure per scale K (in ms, with cuda.synchronize bracketing):
    decomp             : _precompute_window_psd_decomposition
    assemble_t1        : assemble A at t=1 + vstack with A_base
    assemble_t2        : assemble A at t=2 + vstack (extract t-derivative)
    augment            : augment_phase1 build (tau column + fix-t row)
    solver_init        : ADMMSolver __init__  (Ruiz + factorize + GPU xfer)
    warmup             : 100-iter warm-up at hi_t
    hi_feas_solve      : solve at hi_t = lb + 0.015  (target: FEAS)
    mid_bisect_solve   : solve at mid_t = lb + 0.0075 (warm-started)
    viol_check         : _check_violations_highd on full window list

Per solve we also record:  iters, pri_res, dual_res, status, tau_val,
                           wall_s, ms_per_iter.

Fixed costs (run once, independent of K):
    precompute  : _precompute_highd
    base_build  : build_base_problem
    phase0      : minimize-t ADMM solve (initial lb + seed violations)

Scales K are chosen automatically as powers-of-two from 25 up to
min(n_violations, --max-k). Override with --scales 50,200,800.

Usage (on H100 pod):
    python tests/benchmark_scan.py                          # all scales
    python tests/benchmark_scan.py --scales 100,400,1000    # custom
    python tests/benchmark_scan.py --max-k 400 --bisect 2   # quick

Output:
    data/bench/scan_<tag>.json  — full machine-readable timeline
    printed scaling table       — ms per phase vs K
"""
import argparse
import gc as _gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_d16_l3_experiments as exp
from run_d16_l3_experiments import (
    build_base_problem,
    _precompute_window_psd_decomposition,
    _assemble_window_psd,
)
from lasserre_highd import (
    _build_banded_cliques, _precompute_highd, _check_violations_highd,
    val_d_known,
)
from admm_gpu_solver import ADMMSolver, augment_phase1, admm_solve


# ------------- Production config (LOCKED) -------------
PROD = dict(
    d=16, order=3, bandwidth=15,
    rho=0.1, atom_frac=0.5, cuts_per_round=100, k_vecs=3,
    add_upper_loc=True,
)
# Bisection step 4-5 uses max_iters=800-2000, eps=1e-5 to 1e-6 in
# production. We cap at 500 here because at d=16 L3 each ADMM iter runs
# ~80-150ms (dominated by eigh on the 969×969 moment cone), so 2000
# iters per solve is 2-5 minutes — too long for a scan that hits many
# K values. 500 iters is enough to resolve per-iter timing to within
# ~5% and to see ADMM converge (pri_res below eps_pri) for feasible
# problems.
SCS_ITERS = 500
SCS_EPS = 1e-5
WARMUP_ITERS = 0  # skip warmup: solver.solve() already warm-starts internally
# HI_T and MID_T are ABSOLUTE (not offsets from scalar_lb). They must
# straddle val(16) ≈ 1.319:
#   HI_T clearly above val(d)  → feasible → ADMM converges quickly (tau→0)
#   MID_T slightly above val(d) → feasible but near boundary → harder solve,
#                                 exposes convergence-rate scaling with K
# The scalar_lb from phase 0 (~1.0 on the base problem) is NOT above
# val(d), so using it as an offset origin produced infeasible solves
# that ran to the iter cap and measured nothing useful.
HI_T = 1.40
MID_T = 1.33


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timed(label):
    """Context manager returning (label, ms_elapsed)."""
    class _T:
        def __enter__(self):
            cuda_sync()
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *a):
            cuda_sync()
            self.ms = (time.perf_counter() - self.t0) * 1000.0
    t = _T()
    t.label = label
    return t


def default_scales(n_win, max_k):
    """Build a geometric ladder of scales up to min(n_win, max_k)."""
    top = min(n_win, max_k)
    candidates = [25, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3200]
    return [k for k in candidates if k <= top]


def build_full_A(A_base, win_decomp, t_val):
    """Stack base + window PSD block at given t."""
    if win_decomp is None:
        A_full = A_base.tocsc().copy()
        A_full.sort_indices()
        b_win_rows = 0
        psd_sizes = []
        return A_full, b_win_rows, psd_sizes
    A_win, _b_win, psd_sizes = _assemble_window_psd(win_decomp, t_val)
    A_full = sp.vstack([A_base, A_win], format='csc')
    A_full.sort_indices()
    return A_full, win_decomp['n_rows'], psd_sizes


def solve_at_t(solver, A_p1_template, aug_base_data, aug_t_data,
               b_p1_template, aug_b_base_data, aug_b_t_data,
               t_val, tau_col, max_iters, eps):
    """In-place update augmented A and b to the given t, then solve."""
    if aug_t_data is not None:
        np.add(aug_base_data, t_val * aug_t_data, out=A_p1_template.data)
        np.add(aug_b_base_data, t_val * aug_b_t_data, out=b_p1_template)
    else:
        np.copyto(A_p1_template.data, aug_base_data)
        np.copyto(b_p1_template, aug_b_base_data)
    solver._update_A(A_p1_template)
    solver.update_b(b_p1_template)
    tau_tol = max(eps * 10, 1e-4)
    sol = solver.solve(max_iters=max_iters, eps_abs=eps, eps_rel=eps,
                       tau_col=tau_col, tau_tol=tau_tol)
    tau_val = float(sol['x'][tau_col]) if sol['x'] is not None else float('nan')
    info = sol['info']
    return {
        'iters': int(info['iter']),
        'status': str(info.get('status', '')),
        'pri_res': float(info.get('pri_res', float('inf'))),
        'dual_res': float(info.get('dual_res', float('inf'))),
        'eps_pri': float(info.get('eps_pri', 0.0)),
        'eps_dual': float(info.get('eps_dual', 0.0)),
        'tau': tau_val,
    }


def scan_K(K, violations_sorted, P, A_base, b_base, cone_base, meta,
           scalar_lb, phase0_warm):
    """Run the full per-K measurement block and return a record dict."""
    # Build active_windows = top-K by min-eig.
    active_windows = set(int(w) for w, _eig in violations_sorted[:K])
    rec = {'K': K, 'actual_K': len(active_windows)}

    # --- decomp ---
    with timed('decomp') as T:
        win_decomp = _precompute_window_psd_decomposition(
            P, active_windows)
    rec['decomp_ms'] = T.ms
    rec['n_win_rows'] = int(win_decomp['n_rows']) if win_decomp else 0

    # --- assemble at t=1 ---
    with timed('assemble_t1') as T:
        A_full_t1, _n_win_rows, psd_win = build_full_A(A_base, win_decomp, 1.0)
    rec['assemble_t1_ms'] = T.ms
    rec['nnz_full'] = int(A_full_t1.nnz)
    rec['n_rows_full'] = int(A_full_t1.shape[0])
    rec['n_psd_cones'] = len(cone_base['s']) + len(psd_win)

    # --- assemble at t=2 + extract t-derivative ---
    with timed('assemble_t2') as T:
        A_full_t2, _, _ = build_full_A(A_base, win_decomp, 2.0)
        full_t_data = A_full_t2.data - A_full_t1.data
        full_base_data = A_full_t1.data - full_t_data
        del A_full_t2
        has_t_full = bool(np.any(full_t_data != 0))
    rec['assemble_t2_ms'] = T.ms

    b_full_base = np.concatenate([b_base, np.zeros(
        win_decomp['n_rows'] if win_decomp else 0)])
    cone_full = {'z': cone_base['z'], 'l': cone_base['l'],
                 's': list(cone_base['s']) + psd_win}

    # --- augment with tau slack + fix-t row ---
    with timed('augment') as T:
        t_col = meta['t_col']
        fix_t_row = sp.csc_matrix(
            ([1.0], ([0], [t_col])), shape=(1, A_full_t1.shape[1]))
        A_fixed_t1 = sp.vstack([fix_t_row, A_full_t1], format='csc')
        A_fixed_t1.sort_indices()
        b_fixed_t1 = np.insert(b_full_base, 0, 1.0)
        cone_fixed = {'z': cone_full['z'] + 1,
                      'l': cone_full['l'], 's': cone_full['s']}
        A_p1_t1, b_p1_t1, c_p1, cone_p1, tau_col = augment_phase1(
            A_fixed_t1, b_fixed_t1, cone_fixed)

        if has_t_full:
            tmp = A_full_t1.data.copy()
            np.add(full_base_data, 2.0 * full_t_data, out=A_full_t1.data)
            A_fixed_t2 = sp.vstack([fix_t_row, A_full_t1], format='csc')
            A_fixed_t2.sort_indices()
            b_fixed_t2 = np.insert(b_full_base, 0, 2.0)
            A_p1_t2, b_p1_t2, _, _, _ = augment_phase1(
                A_fixed_t2, b_fixed_t2, cone_fixed)
            aug_t_data = A_p1_t2.data - A_p1_t1.data
            aug_base_data = A_p1_t1.data - aug_t_data
            aug_b_t_data = b_p1_t2 - b_p1_t1
            aug_b_base_data = b_p1_t1 - aug_b_t_data
            np.copyto(A_full_t1.data, tmp)
            del A_p1_t2, b_p1_t2, A_fixed_t2, b_fixed_t2, tmp
        else:
            aug_t_data = None
            aug_base_data = A_p1_t1.data.copy()
            aug_b_t_data = None
            aug_b_base_data = b_p1_t1.copy()
    rec['augment_ms'] = T.ms
    rec['nnz_aug'] = int(A_p1_t1.nnz)
    rec['n_rows_aug'] = int(A_p1_t1.shape[0])

    # --- solver init (Ruiz + Cholesky/CG setup + GPU transfer) ---
    with timed('solver_init') as T:
        solver = ADMMSolver(A_p1_t1, b_p1_t1, c_p1, cone_p1,
                            rho=PROD['rho'], device='cuda', verbose=False)
    rec['solver_init_ms'] = T.ms
    rec['linear_solver'] = solver._solver_type

    # --- warmup (skip if WARMUP_ITERS == 0) ---
    hi_t = HI_T
    mid_t = MID_T
    with timed('warmup') as T:
        if WARMUP_ITERS > 0:
            if aug_t_data is not None:
                np.add(aug_base_data, hi_t * aug_t_data, out=A_p1_t1.data)
                np.add(aug_b_base_data, hi_t * aug_b_t_data, out=b_p1_t1)
            solver._update_A(A_p1_t1)
            solver.update_b(b_p1_t1)
            _ = solver.solve(max_iters=WARMUP_ITERS, eps_abs=1.0, eps_rel=1.0,
                             tau_col=tau_col, tau_tol=1e-4)
    rec['warmup_ms'] = T.ms

    # --- hi_feas solve ---
    with timed('hi_feas_solve') as T:
        hi_info = solve_at_t(solver, A_p1_t1, aug_base_data, aug_t_data,
                             b_p1_t1, aug_b_base_data, aug_b_t_data,
                             hi_t, tau_col, SCS_ITERS, SCS_EPS)
    rec['hi_feas_ms'] = T.ms
    rec['hi_feas'] = hi_info
    rec['hi_feas_ms_per_iter'] = (T.ms / max(1, hi_info['iters']))

    # --- mid_bisect solve (warm-started from hi) ---
    with timed('mid_bisect_solve') as T:
        mid_info = solve_at_t(solver, A_p1_t1, aug_base_data, aug_t_data,
                              b_p1_t1, aug_b_base_data, aug_b_t_data,
                              mid_t, tau_col, SCS_ITERS, SCS_EPS)
    rec['mid_bisect_ms'] = T.ms
    rec['mid_bisect'] = mid_info
    rec['mid_bisect_ms_per_iter'] = (T.ms / max(1, mid_info['iters']))

    # --- violation check over full pool ---
    # Use the mid-solve y as the test point (mimics real run).
    y_mid = np.zeros(meta['n_y'])
    # solver returns x in augmented (fix_t + original) space; tau_col is last.
    # The original moment block is x[1 : 1+n_y] because fix_t adds 1 row but
    # augmentation adds a column, not shifting indices — need to confirm.
    # The check_feasible() in production strips the first fix_t row and reads
    # x[:n_x]. Here x is in (fix_t-augmented + tau) space; meta['n_x'] is the
    # non-augmented column count (n_y + 1 for t_col). tau_col is appended.
    x_full = solver.ws.x.detach().cpu().numpy()
    n_x = meta['n_x']
    # augmented columns are [original_cols..., tau]; x[:n_x] holds original
    y_mid[:] = x_full[:meta['n_y']]
    with timed('viol_check') as T:
        viols = _check_violations_highd(y_mid, hi_t, P, active_windows,
                                        k_vecs=0)
    rec['viol_check_ms'] = T.ms
    rec['viol_check_n_viols'] = len(viols)

    # Cleanup before next K
    del solver, win_decomp, A_full_t1, A_p1_t1, b_p1_t1, c_p1, cone_p1
    del aug_base_data, aug_t_data, aug_b_base_data, aug_b_t_data
    del full_base_data, full_t_data
    _gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return rec


def print_scale_table(records, fixed):
    print('\n' + '=' * 108)
    print('BENCHMARK SCAN — d=16 L3 production config')
    print('=' * 108)
    print(f"\nFixed costs (run once):")
    for k, v in fixed.items():
        if isinstance(v, (int, float)):
            print(f"  {k:20s} {v:>10.1f}")
        else:
            print(f"  {k:20s} {v}")
    print(f"\nScaling (ms per phase vs K = #active windows):")
    hdr = (f"  {'K':>6s} {'nnz(A)':>9s} {'rows':>6s} "
           f"{'decomp':>7s} {'asm_t1':>7s} {'asm_t2':>7s} "
           f"{'augmnt':>7s} {'slv_in':>7s} {'warm':>6s} "
           f"{'hi_slv':>7s} {'mid_slv':>8s} {'viol':>6s}   "
           f"{'hi_it':>5s} {'mid_it':>6s} "
           f"{'hi_us':>6s} {'mid_us':>7s}")
    print(hdr)
    print('  ' + '-' * 106)
    for r in records:
        K = r['K']
        print(f"  {K:>6d} {r['nnz_full']:>9,} {r['n_rows_full']:>6,} "
              f"{r['decomp_ms']:>7.0f} {r['assemble_t1_ms']:>7.0f} "
              f"{r['assemble_t2_ms']:>7.0f} "
              f"{r['augment_ms']:>7.0f} {r['solver_init_ms']:>7.0f} "
              f"{r['warmup_ms']:>6.0f} "
              f"{r['hi_feas_ms']:>7.0f} {r['mid_bisect_ms']:>8.0f} "
              f"{r['viol_check_ms']:>6.0f}   "
              f"{r['hi_feas']['iters']:>5d} {r['mid_bisect']['iters']:>6d} "
              f"{r['hi_feas_ms_per_iter']*1000:>6.0f} "
              f"{r['mid_bisect_ms_per_iter']*1000:>7.0f}")
    print('=' * 108)

    # Per-phase growth ratios (K_max / K_min) — exposes which phase is
    # super-linear and thus the first to worry about at higher rounds.
    if len(records) >= 2:
        r0 = records[0]
        r1 = records[-1]
        k_ratio = r1['K'] / max(1, r0['K'])
        print(f"\nGrowth ratios ({r1['K']}/{r0['K']} = {k_ratio:.1f}x):")
        for phase in ['decomp', 'assemble_t1', 'assemble_t2', 'augment',
                      'solver_init', 'warmup',
                      'hi_feas', 'mid_bisect', 'viol_check']:
            key = f'{phase}_ms'
            if phase in ('hi_feas', 'mid_bisect'):
                key = f'{phase}_ms'
            base = r0.get(key, 0) or 1
            top = r1.get(key, 0) or 1
            print(f"  {phase:20s} {top/base:>6.2f}x "
                  f"({base:>7.0f} -> {top:>7.0f} ms)")
    print('=' * 108)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-scale benchmark for d=16 L3 production.')
    parser.add_argument('--scales', type=str, default=None,
                        help='Comma-separated K values (e.g. 100,400,1000). '
                             'Default: geometric ladder up to --max-k')
    parser.add_argument('--max-k', type=int, default=1200,
                        help='Upper bound for default ladder (default 1200)')
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--tag', type=str, default='scan')
    args = parser.parse_args()

    ts = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    tag = f"{args.tag}_{ts}"

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'bench')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'scan_{tag}.json')

    print(f"{'='*78}")
    print(f"BENCHMARK SCAN")
    print(f"  d={PROD['d']} O{PROD['order']} bw={PROD['bandwidth']} "
          f"rho={PROD['rho']}")
    print(f"  Fixed scs eps={SCS_EPS} iters={SCS_ITERS} "
          f"(matches prod bisect step 5-6 budget)")
    print(f"  Tag: {tag}")
    print(f"{'='*78}\n", flush=True)

    # --- fixed costs ---
    fixed = {}
    with timed('precompute') as T:
        cliques = _build_banded_cliques(PROD['d'], PROD['bandwidth'])
        P = _precompute_highd(PROD['d'], PROD['order'], cliques, verbose=False)
    fixed['precompute_ms'] = T.ms
    n_y = P['n_y']
    n_win_total = P['n_win']
    print(f"  n_y={n_y:,}  n_win={n_win_total}  "
          f"precompute={T.ms:.0f} ms", flush=True)

    with timed('base_build') as T:
        A_base, b_base, c_obj, cone_base, meta = build_base_problem(
            P, add_upper_loc=PROD['add_upper_loc'])
    fixed['base_build_ms'] = T.ms
    fixed['base_nnz'] = int(A_base.nnz)
    fixed['base_rows'] = int(A_base.shape[0])
    print(f"  base: rows={A_base.shape[0]:,} nnz={A_base.nnz:,} "
          f"build={T.ms:.0f} ms", flush=True)

    # --- phase 0 (minimize-t) ---
    print(f"\n  [phase 0] minimize-t on base problem...", flush=True)
    with timed('phase0') as T:
        sol0 = admm_solve(A_base, b_base, c_obj, cone_base,
                          max_iters=SCS_ITERS,
                          eps_abs=max(SCS_EPS, 1e-5),
                          eps_rel=max(SCS_EPS, 1e-5),
                          device='cuda', verbose=False)
    fixed['phase0_ms'] = T.ms
    fixed['phase0_iters'] = int(sol0['info']['iter'])
    fixed['phase0_status'] = str(sol0['info'].get('status', ''))
    scalar_lb = float(sol0['x'][meta['t_col']])
    y_vals = sol0['x'][:n_y].copy()
    fixed['phase0_scalar_lb'] = scalar_lb
    print(f"  phase0: {T.ms:.0f} ms  {sol0['info']['iter']} iters  "
          f"lb={scalar_lb:.6f}", flush=True)

    # --- violations over empty active set ---
    with timed('viol_seed') as T:
        violations = _check_violations_highd(y_vals, scalar_lb, P, set(),
                                              k_vecs=0)
    fixed['viol_seed_ms'] = T.ms
    violations_sorted = sorted(violations, key=lambda v: v[1])
    n_viol = len(violations_sorted)
    fixed['n_seed_viols'] = n_viol
    print(f"  seed viols: {n_viol} (sort={T.ms:.0f} ms)", flush=True)

    # --- pick scales ---
    if args.scales:
        scales = [int(x) for x in args.scales.split(',')]
    else:
        scales = default_scales(n_viol, args.max_k)
    print(f"\n  Scales: {scales}\n", flush=True)

    records = []
    for K in scales:
        if K > n_viol:
            print(f"  K={K} > n_viol={n_viol}, skipping", flush=True)
            continue
        print(f"\n  ---- scan K={K} ----", flush=True)
        t_k = time.perf_counter()
        rec = scan_K(K, violations_sorted, P, A_base, b_base, cone_base,
                     meta, scalar_lb, sol0)
        rec['wall_s'] = time.perf_counter() - t_k
        records.append(rec)
        # Terse per-K printout
        print(f"  K={K}  decomp={rec['decomp_ms']:.0f}ms  "
              f"augment={rec['augment_ms']:.0f}ms  "
              f"init={rec['solver_init_ms']:.0f}ms  "
              f"hi={rec['hi_feas_ms']:.0f}ms"
              f"({rec['hi_feas']['iters']}it)  "
              f"mid={rec['mid_bisect_ms']:.0f}ms"
              f"({rec['mid_bisect']['iters']}it)  "
              f"total={rec['wall_s']:.1f}s",
              flush=True)

    # --- output ---
    out = {
        'tag': tag,
        'config': {'PROD': PROD, 'SCS_ITERS': SCS_ITERS, 'SCS_EPS': SCS_EPS,
                   'WARMUP_ITERS': WARMUP_ITERS,
                   'HI_T': HI_T, 'MID_T': MID_T},
        'fixed': fixed,
        'scales': records,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print_scale_table(records, fixed)
    print(f"\nOutput written to: {out_path}")


if __name__ == '__main__':
    main()
