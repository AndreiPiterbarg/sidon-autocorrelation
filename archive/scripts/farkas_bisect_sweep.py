"""Bisect Farkas infeasibility certification across d values.

Usage:
    python3.11 scripts/farkas_bisect_sweep.py single d=4 order=3 t_test=1.090
    python3.11 scripts/farkas_bisect_sweep.py bisect d=4 order=3
    python3.11 scripts/farkas_bisect_sweep.py sweep           # full scale
"""
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.set_int_max_str_digits(10**7)
except AttributeError:
    pass

from certified_lasserre.farkas_certify import (
    farkas_certify_at, farkas_certify_bisect,
)


# Targets (from lasserre/core.py val_d_known and Boyer-Li 2025)
TARGETS = {
    4: (1.05, 1.102),
    6: (1.10, 1.171),
    8: (1.15, 1.205),
    10: (1.20, 1.241),
    12: (1.24, 1.271),
    14: (1.25, 1.284),
    16: (1.27, 1.319),   # to beat Boyer-Li 1.2802
    18: (1.28, 1.315),
    20: (1.29, 1.320),
}


def run_bisect(d, order, t_lo, t_hi, tol=1e-5, max_steps=20, out_dir=Path('data')):
    print(f'\n============================================================', flush=True)
    print(f'# Farkas bisect: d={d}, order={order}, bracket=[{t_lo}, {t_hi}]', flush=True)
    print(f'============================================================', flush=True)
    t_start = time.time()
    try:
        res = farkas_certify_bisect(
            d=d, order=order,
            t_lo=t_lo, t_hi=t_hi,
            tol=tol, max_bisect=max_steps,
            verbose=True,
        )
        print(f'\n!!! d={d} FINAL CERTIFIED LB: {res.lb_rig_decimal}', flush=True)
        print(f'  total_time={time.time() - t_start:.1f}s', flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = {
            'd': d, 'order': order,
            'lb_rig_decimal': res.lb_rig_decimal,
            'lb_rig_num_den': [int(res.lb_rig.numerator), int(res.lb_rig.denominator)],
            'mu0_float': res.mu0_float,
            'residual_l1_float': res.residual_l1_float,
            'safety_margin_float': res.safety_margin_float,
            'solver_time': res.solver_time,
            'round_time': res.round_time,
            'total_time': time.time() - t_start,
        }
        fn = out_dir / f'farkas_d{d}_o{order}.json'
        fn.write_text(json.dumps(out, indent=2))
        print(f'  JSON: {fn}', flush=True)
        return res
    except Exception as e:
        print(f'  FAILED at d={d}: {type(e).__name__}: {e}', flush=True)
        import traceback; traceback.print_exc()
        return None


def sweep():
    # Progressive: small to large. Stop on first failure.
    for d in [4, 6, 8, 10, 12, 14, 16]:
        t_lo, t_hi = TARGETS[d]
        try:
            run_bisect(d=d, order=3, t_lo=t_lo, t_hi=t_hi, tol=1e-4, max_steps=15)
        except Exception as e:
            print(f'Stopping sweep at d={d}: {e}', flush=True)
            break


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'sweep'
    if cmd == 'sweep':
        sweep()
    elif cmd == 'single':
        d = int([a for a in sys.argv if a.startswith('d=')][0][2:])
        order = int([a for a in sys.argv if a.startswith('order=')][0][6:])
        t_test = float([a for a in sys.argv if a.startswith('t_test=')][0][7:])
        res, cert = farkas_certify_at(d=d, order=order, t_test=t_test, verbose=True)
        print(f'\nResult: {res.status}  lb_rig={res.lb_rig_decimal}  '
              f'total_time={res.total_time:.1f}s')
    elif cmd == 'bisect':
        d = int([a for a in sys.argv if a.startswith('d=')][0][2:])
        order = int([a for a in sys.argv if a.startswith('order=')][0][6:])
        t_lo, t_hi = TARGETS[d]
        t_lo = float([a for a in sys.argv if a.startswith('t_lo=')][0][5:]) \
               if any(a.startswith('t_lo=') for a in sys.argv) else t_lo
        t_hi = float([a for a in sys.argv if a.startswith('t_hi=')][0][5:]) \
               if any(a.startswith('t_hi=') for a in sys.argv) else t_hi
        run_bisect(d=d, order=order, t_lo=t_lo, t_hi=t_hi)
    else:
        print('Usage: sweep | single d=D order=K t_test=T | bisect d=D order=K [t_lo=L t_hi=H]')
