"""Sweep certified Lasserre over (d, order) and report sizes + bounds.

Usage:
    python3.11 scripts/run_certify_sweep.py sizes       # size only
    python3.11 scripts/run_certify_sweep.py d=10,3      # single run
    python3.11 scripts/run_certify_sweep.py sweep       # 4..12 L3
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.set_int_max_str_digits(10**7)
except AttributeError:
    pass


def report_sizes():
    from lasserre.precompute import _precompute
    for d, k in [(8, 3), (10, 3), (12, 3), (14, 3), (16, 3)]:
        try:
            P = _precompute(d, k, verbose=False)
            print(f'd={d} order={k}: n_y={P["n_y"]}  n_basis={P["n_basis"]}  '
                  f'n_loc={P["n_loc"]}  n_win={P["n_win"]}')
        except Exception as e:
            print(f'd={d} order={k}: FAILED ({e})')
            break


def run_one(d: int, order: int, out_dir: Path = Path('data')):
    from certified_lasserre.optimal_lambda import certify_with_optimal_lambda
    t_start = time.time()
    print(f'\n===== d={d}, order={order} =====', flush=True)
    res = certify_with_optimal_lambda(d=d, order=order, verbose=False, top_lam=5)
    print(f'  t*            = {res.t_star_float:.9f}')
    print(f'  lb_rig        = {res.lb_rig_decimal}')
    print(f'  joint_time    = {res.joint_solver_time:.1f}s')
    print(f'  cert_time     = {res.certify_time:.1f}s')
    print(f'  total_time    = {time.time() - t_start:.1f}s')
    print(f'  residual_l1   = {float(res.residual_l1):.3e}')
    print(f'  safety_loss   = {float(res.safety_loss):.3e}')
    print(f'  top lam windows:')
    for wi, (ell, slo), weight in res.lam_win_top[:5]:
        print(f'    window {wi:5d} (ell={ell}, s_lo={slo}): {weight:.4f}')

    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        'd': d, 'order': order,
        't_star_float': res.t_star_float,
        'lb_rig_decimal': res.lb_rig_decimal,
        'lb_rig_num_den': [int(res.lb_rig.numerator), int(res.lb_rig.denominator)],
        'residual_l1_float': float(res.residual_l1),
        'safety_loss_float': float(res.safety_loss),
        'joint_solver_time': res.joint_solver_time,
        'certify_time': res.certify_time,
        'total_time': res.total_time,
        'top_windows': [(wi, list(ws), w) for wi, ws, w in res.lam_win_top[:10]],
    }
    fn = out_dir / f'certify_d{d}_o{order}.json'
    fn.write_text(json.dumps(out, indent=2))
    print(f'  written to {fn}')
    return res


def sweep():
    # Progressive: small to large, write JSON at each step
    for d, k in [(4, 3), (6, 3), (8, 3), (10, 3), (12, 3)]:
        try:
            run_one(d, k)
        except Exception as e:
            print(f'  d={d} order={k} FAILED: {type(e).__name__}: {e}', flush=True)
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'sizes'
    if cmd == 'sizes':
        report_sizes()
    elif cmd == 'sweep':
        sweep()
    elif ',' in cmd:
        d, k = [int(x) for x in cmd.split('=')[-1].split(',')]
        run_one(d, k)
    else:
        print(f'unknown command: {cmd}')
        sys.exit(1)
