"""SOUND cascade-based prover for general C_{1a}.

Uses run_cascade.run_cascade() which enumerates full d-dim canonical
compositions (`b <= rev(b)` lex) at L0 — this covers all f up to the
f <-> f-reversed symmetry, which preserves max(f*f).  Sound for general
C_{1a}, NOT just C_{1a}^{sym}.

Bench-based prior runs (palindromic enumeration) only proved bounds on
C_{1a}^{sym}, which is known to be >= 1.42429.  Those did NOT improve on
the published C&S 2017 bound of C_{1a} >= 1.2802 for general f.

Strategy: cascade from small d0 (where full enum is feasible) up through
L1, L2, ...  At each level apply F + Q post-filters (Q's LP becomes too
expensive at d > ~16, so cascade silently degrades to F-only at deep
levels).

For each (n_half, m, c_target):
  - Launch run_cascade.py with --use_F --use_Q (+ --skip_sdp).
  - Hard timeout per c.
  - Capture L0/L1/.../survivor counts; "closure" = 0 survivors at deepest
    level reached.

Run dir: /home/ubuntu/sound_cascade_<TS>/
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from math import comb

ROOT = '/home/ubuntu'
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
from pruning import correction


C_UPPER = 1.5029
ENV = os.environ.copy()
ENV['MOSEKLM_LICENSE_FILE'] = '/home/ubuntu/mosek/mosek.lic'


def n_full_compositions(d, S):
    return comb(S + d - 1, d - 1)


def run_one(rundir, c_target, n_half, m, max_levels, timeout):
    """Launch run_cascade.py at (n_half, m, c_target) with full enum.
    Returns dict of stats."""
    d0 = 2 * n_half
    S0 = 4 * n_half * m  # full L0 sum
    n_total = n_full_compositions(d0, S0)
    tag = f"c{int(round(c_target*10000)):05d}_n{n_half}_m{m}"
    log_path = os.path.join(rundir, f"log_{tag}.log")
    output_dir = os.path.join(rundir, f"data_{tag}")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'python3', '-u',
        os.path.join(ROOT, 'cloninger-steinerberger', 'cpu', 'run_cascade.py'),
        '--n_half', str(n_half),
        '--m', str(m),
        '--c_target', str(c_target),
        '--max_levels', str(max_levels),
        '--workers', '64',
        '--use_F',
        '--use_Q',
        '--skip_sdp',
        '--output_dir', output_dir,
    ]
    print(f"  [{tag}] launch  d0={d0} S0={S0:,}  L0_total={n_total:,} "
            f"timeout={timeout}s", flush=True)
    t0 = time.time()
    with open(log_path, 'w') as logf:
        try:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False, env=ENV)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1
    wall = time.time() - t0

    out = {
        'c_target': c_target, 'n_half': n_half, 'm': m,
        'd0': d0, 'S0': S0, 'L0_compositions': n_total,
        'wall_sec': round(wall, 2),
        'log_path': log_path, 'data_dir': output_dir,
        'returncode': rc, 'max_levels': max_levels,
    }
    if rc == -1:
        out['status'] = 'TIMEOUT'
    elif rc != 0:
        out['status'] = f'EXIT_{rc}'
    else:
        # Parse log for closure outcome.  Look for 'PROVEN' / final survivor count.
        try:
            with open(log_path) as f:
                txt = f.read()
        except Exception:
            txt = ''
        out['log_tail'] = txt[-1500:] if txt else ''
        if 'PROVEN' in txt or '0 survivors' in txt[-2000:] or 'no survivors at' in txt[-2000:]:
            out['status'] = 'CLOSED'
        else:
            out['status'] = 'NOT_CLOSED_OR_INCONCLUSIVE'
    return out


def main():
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    rundir = os.path.join(ROOT, f'sound_cascade_{ts}')
    os.makedirs(rundir, exist_ok=True)

    # Configs in priority order: low c (sanity) first, then push.
    plan = [
        # (c_target, n_half, m, max_levels, timeout_sec)
        (1.20, 2, 15, 4,  900),    # sanity: should close
        (1.20, 2, 20, 4,  900),    # sanity
        (1.25, 2, 15, 4,  1800),
        (1.25, 2, 20, 4,  1800),
        (1.27, 2, 20, 4,  1800),
        (1.28, 2, 20, 5,  3600),
        (1.281, 2, 20, 5, 3600),  # PUSH past published 1.2802
    ]

    summary_path = os.path.join(rundir, 'summary.json')
    summary = {'started_utc': ts, 'plan': plan, 'results': []}

    def save():
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"Run dir: {rundir}", flush=True)
    save()

    for (c_target, n_half, m, max_levels, timeout) in plan:
        if c_target + correction(m, n_half) >= C_UPPER:
            print(f"({n_half},{m},c={c_target}): VACUOUS, skip", flush=True)
            continue
        print(f"\n===== c_target={c_target}, n_half={n_half}, m={m}, "
                f"max_levels={max_levels} =====", flush=True)
        r = run_one(rundir, c_target, n_half, m, max_levels, timeout)
        summary['results'].append(r)
        save()
        print(f"  -> {r['status']}  wall={r['wall_sec']}s", flush=True)
        if r['status'] != 'CLOSED' and c_target >= 1.27:
            # Sanity ones (c <= 1.25) we keep going even on inconclusive.
            # For high c, stop on first failure.
            print(f"  --- Stopping push at c={c_target} ({r['status']})", flush=True)
            break

    save()
    print(f"\nDONE.  summary -> {summary_path}", flush=True)


if __name__ == '__main__':
    main()
