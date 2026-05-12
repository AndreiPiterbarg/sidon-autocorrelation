#!/usr/bin/env python
"""Run a single Lasserre highd config. For pod deployment.

Usage:
    python tests/run_single_config.py --d 14 --order 3 --bw 13
    python tests/run_single_config.py --d 16 --order 3 --bw 12
"""
import sys
import os
import time
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import solve_highd_sparse

val_d_known = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
    32: 1.336, 64: 1.384, 128: 1.420,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--bw', type=int, required=True)
    parser.add_argument('--cg-rounds', type=int, default=10)
    parser.add_argument('--bisect', type=int, default=8)
    args = parser.parse_args()

    print(f"Single config run: d={args.d} O{args.order} bw={args.bw}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    try:
        import mosek
        print(f"MOSEK: {mosek.Env.getversion()}")
    except Exception as e:
        print(f"MOSEK: {e}")

    # Memory monitoring
    import threading
    def mem_monitor():
        import subprocess as _sp
        while True:
            try:
                r = _sp.run(['free', '-g'], capture_output=True, text=True, timeout=5)
                for line in r.stdout.strip().split('\n'):
                    if 'Mem' in line:
                        parts = line.split()
                        used = int(parts[2])
                        total = int(parts[1])
                        print(f"  [MEM] {used}/{total} GB used ({100*used/total:.0f}%)",
                              flush=True)
            except Exception:
                pass
            time.sleep(30)
    t_mon = threading.Thread(target=mem_monitor, daemon=True)
    t_mon.start()

    # System info
    try:
        import subprocess as _sp
        r = _sp.run(['free', '-g'], capture_output=True, text=True, timeout=5)
        print(f"System RAM: {r.stdout.strip().split(chr(10))[1]}")
    except Exception:
        pass
    print(flush=True)

    t0 = time.time()
    r = solve_highd_sparse(
        d=args.d, order=args.order, bandwidth=args.bw,
        max_cg_rounds=args.cg_rounds, n_bisect=args.bisect,
        verbose=True,
    )
    elapsed = time.time() - t0

    vd = val_d_known.get(args.d, 0)
    gc = (r['lb'] - 1) / (vd - 1) * 100 if vd > 1 else 0
    sound = r['lb'] <= vd + 1e-6 if vd > 0 else True

    print(f"\n{'='*70}")
    print(f"FINAL: d={args.d} O{args.order} bw={args.bw}")
    print(f"  lb = {r['lb']:.10f}")
    print(f"  val({args.d}) = {vd}")
    print(f"  gap_closure = {gc:.2f}%")
    print(f"  n_y = {r['n_y']:,}")
    print(f"  time = {elapsed:.1f}s = {elapsed/3600:.2f}hr")
    print(f"  sound = {sound}")
    if vd > 0 and r['lb'] > 1.2802:
        print(f"  *** NEW RECORD: lb={r['lb']:.6f} > 1.2802 ***")
    print(f"{'='*70}")

    out = {
        'd': args.d, 'order': args.order, 'bw': args.bw,
        'lb': r['lb'], 'gap_closure': gc, 'n_y': r['n_y'],
        'elapsed': elapsed, 'sound': sound,
        'timestamp': datetime.now().isoformat(),
    }
    tag = f"d{args.d}_o{args.order}_bw{args.bw}"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'data', f'result_{tag}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
