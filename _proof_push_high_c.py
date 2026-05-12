"""Push c_target higher and higher until vacuous or no closure.

Continues from the 4-target sweep — that already closed up through c=1.29.
This script targets c ∈ [1.295, ..., 1.49], advancing through bigger m
(bigger m gives smaller correction → can target larger c).

Per c, try configs in cost order, advance to next c on first Q=0 closure.
Stop entirely when one c has no closure under any feasible config.

All logs / JSONs under /home/ubuntu/proof_push_high_<TS>/.
Driver runs unattended.  Send `pkill -f proof_push_high` to halt.
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
# Push higher and higher; stop when no closure at a given c.
C_TARGETS = [
    1.295,
    1.30,
    1.31,
    1.32,
    1.33,
    1.34,
    1.35,
    1.36,
    1.37,
    1.38,
    1.39,
    1.40,
    1.42,
    1.45,
    1.48,
]

# (n_half, m) configs ordered by computational cost ascending.
# Will skip vacuous (c + corr(m, n_half) >= 1.5029) configs.
CONFIGS = [
    (5, 10),    # d=10,  ~4.6 M
    (5, 15),    # d=10,  ~22.5 M
    (6, 10),    # d=12,  ~234 M
    (5, 20),    # d=10,  ~46 M
    (6, 15),    # d=12,  ~1.95 B
    (5, 30),    # d=10,  ~150 M
    (5, 50),    # d=10,  ~1.4 B
    (6, 20),    # d=12,  ~6.7 B  — slow, only if smaller fail
    (7, 10),    # d=14,  ~36 B   — VERY slow, last resort
]


def n_compositions(n_half, m):
    half_sum = 2 * n_half * m
    return comb(half_sum + n_half - 1, n_half - 1)


def estimate_timeout(n_half, m):
    nc = n_compositions(n_half, m)
    base = 30
    return int(base + nc / 6_000_000)


def vacuous(c_target, n_half, m):
    return c_target + correction(m, n_half) >= C_UPPER


def run_one(rundir: str, c_target: float, n_half: int, m: int) -> dict:
    tag = f"c{int(round(c_target*10000)):05d}_n{n_half}_m{m}"
    log_path = os.path.join(rundir, f"log_{tag}.log")
    json_path = os.path.join(rundir, f"out_{tag}.json")
    timeout = estimate_timeout(n_half, m) + 60
    timeout = min(timeout, 2400)  # 40-min hard cap

    t0 = time.time()
    cmd = [
        'python3', '-u', os.path.join(ROOT, '_L_bench.py'),
        '--n_half', str(n_half),
        '--m', str(m),
        '--c_target', str(c_target),
        '--solver', 'CLARABEL',
        '--order', '1',
        '--max_l', '0',
        '--audit', '0',
        '--out', json_path,
    ]
    print(f"  [{tag}] launching ...  timeout={timeout}s "
            f"({n_compositions(n_half,m):,} comps)", flush=True)
    with open(log_path, 'w') as logf:
        try:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1
    wall = time.time() - t0

    out = {
        'c_target': c_target, 'n_half': n_half, 'm': m,
        'd': 2 * n_half, 'wall_sec': round(wall, 2),
        'log_path': log_path, 'json_path': json_path,
        'returncode': rc,
        'n_compositions': n_compositions(n_half, m),
    }
    if rc == -1:
        out['status'] = 'TIMEOUT'
        return out
    if rc != 0:
        out['status'] = f'EXIT_{rc}'
        return out
    if not os.path.exists(json_path):
        out['status'] = 'NO_JSON'
        return out
    try:
        with open(json_path) as f:
            jdata = json.load(f)
        r = jdata[0] if isinstance(jdata, list) else jdata
    except Exception as e:
        out['status'] = f'JSON_ERR:{e}'
        return out

    out['n_processed'] = r.get('n_processed', -1)
    out['surv_F'] = r.get('surv_F', -1)
    out['surv_Q'] = r.get('surv_Q', -1)
    if out['surv_Q'] == 0 and out['surv_F'] >= 0:
        out['status'] = 'CLOSED'
    else:
        out['status'] = 'NOT_CLOSED'
    return out


def main():
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    rundir = os.path.join(ROOT, f'proof_push_high_{ts}')
    os.makedirs(rundir, exist_ok=True)
    summary = {
        'started_utc': ts,
        'c_targets': C_TARGETS,
        'configs_tried': CONFIGS,
        'host': os.uname().nodename,
        'attempts': [],
        'closures': {},
    }
    summary_path = os.path.join(rundir, 'summary.json')
    txt_path = os.path.join(rundir, 'summary.txt')

    def save():
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        with open(txt_path, 'w') as f:
            f.write(f"Push-c sweep — started {ts}\n")
            f.write(f"Host: {summary['host']}\n\n")
            f.write(f"{'c_target':>9} {'n':>3} {'m':>3} {'d':>3} "
                    f"{'comps':>14} {'F_surv':>8} {'Q_surv':>8} "
                    f"{'wall_s':>8} {'status'}\n")
            f.write('-' * 90 + '\n')
            for a in summary['attempts']:
                cps = a.get('n_processed', a.get('n_compositions', -1))
                cps_s = f"{cps:,}" if cps >= 0 else 'n/a'
                f.write(f"{a['c_target']:>9.4f} {a['n_half']:>3} "
                        f"{a['m']:>3} {a['d']:>3} {cps_s:>14} "
                        f"{a.get('surv_F', '?'):>8} "
                        f"{a.get('surv_Q', '?'):>8} "
                        f"{a['wall_sec']:>8.1f} {a['status']}\n")
            f.write('\n=== Closures ===\n')
            for c, info in summary['closures'].items():
                f.write(f"  c={c}: closed by ({info['n_half']},{info['m']}) "
                        f"d={info['d']} F={info['surv_F']} Q={info['surv_Q']} "
                        f"in {info['wall_sec']}s\n")
            failed = [c for c in C_TARGETS
                      if str(c) not in summary['closures']
                      and float(c) <= summary.get('first_failure', float('inf'))]
            f.write(f"  c-targets without closure: {failed}\n")

    print(f"Run dir: {rundir}", flush=True)
    save()

    t_start = time.time()

    for c_target in C_TARGETS:
        print(f"\n===== c_target = {c_target} =====", flush=True)
        feasible_any = False
        closed = False
        for nh, m in CONFIGS:
            if vacuous(c_target, nh, m):
                continue
            feasible_any = True
            comps = n_compositions(nh, m)
            corr = correction(m, nh)
            print(f"  ({nh},{m},d={2*nh}): {comps:,} comps, "
                    f"corr={corr:.5f} thresh={c_target+corr:.5f}", flush=True)
            r = run_one(rundir, c_target, nh, m)
            summary['attempts'].append(r)
            print(f"  -> {r['status']}  F={r.get('surv_F','?')} "
                    f"Q={r.get('surv_Q','?')}  wall={r['wall_sec']}s", flush=True)
            save()
            if r['status'] == 'CLOSED':
                summary['closures'][str(c_target)] = r
                closed = True
                save()
                print(f"  *** CLOSURE: c={c_target} at "
                        f"(n_half={nh}, m={m}, d={2*nh}) ***", flush=True)
                break
            # If config has very few Q-survivors, the next bigger config will
            # almost certainly close.  Continue.
        if not feasible_any:
            print(f"  --- NO FEASIBLE CONFIG (all vacuous).  STOPPING.",
                    flush=True)
            summary['first_failure'] = c_target
            break
        if not closed:
            print(f"  --- NO CLOSURE for c={c_target}.  STOPPING.", flush=True)
            summary['first_failure'] = c_target
            break

    summary['elapsed_sec'] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== DONE.  total wall {summary['elapsed_sec']}s ===", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    closed_cs = sorted([float(c) for c in summary['closures']])
    if closed_cs:
        print(f"Highest c closed: {max(closed_cs)}", flush=True)


if __name__ == '__main__':
    main()
