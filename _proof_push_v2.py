"""Proof sweep v2 — smart config order, generous timeouts.

Vacuity bound: c + correction(m) < 1.5029 where correction = 2/m + 1/m^2.
   m=15  ->  c < 1.3651
   m=20  ->  c < 1.4004
   m=30  ->  c < 1.4351
   m=50  ->  c < 1.4625

Strategy: for each c, try big-d configs that have actual pruning power.
Skip the (5, *) configs entirely — empirically they leave too many F-survivors
at high c so Q's LP loop runs forever.

Lesson from v1: at c=1.32 the (6,15,d=12) needed > 375s to finish (timeout).
Increase timeouts substantially.

Closures from v1 (recorded; not re-run):
   c=1.295 -> (6,15,d=12), 309.7s
   c=1.30  -> (6,15,d=12), 313.7s
   c=1.31  -> (6,15,d=12), 320.9s
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

# Resume from c=1.32 (1.295, 1.30, 1.31 already closed in v1).
C_TARGETS = [
    1.32, 1.33, 1.34, 1.35, 1.36,
    1.365,                         # near m=15 vacuity limit
    1.37, 1.38, 1.39, 1.40,
    1.41, 1.42, 1.43,
    1.44, 1.45, 1.46,
]

# (n_half, m, timeout_sec) — big-d only, generous timeouts.
CONFIGS = [
    (6, 15,  900),     # d=12, 1.71B comps  (tight: ran 320s at c=1.31)
    (6, 20, 1800),     # d=12, 2.4B comps   (~7 min predicted)
    (6, 30, 3600),     # d=12, 9.0B comps   (~25 min)
    (6, 50, 18000),    # d=12, 67B comps    (~3 hr — only if needed)
    (7, 15, 7200),     # d=14, 56B comps    (last resort)
]


def n_compositions(n_half, m):
    half_sum = 2 * n_half * m
    return comb(half_sum + n_half - 1, n_half - 1)


def vacuous(c_target, n_half, m):
    return c_target + correction(m, n_half) >= C_UPPER


def run_one(rundir, c_target, n_half, m, timeout):
    tag = f"c{int(round(c_target*10000)):05d}_n{n_half}_m{m}"
    log_path = os.path.join(rundir, f"log_{tag}.log")
    json_path = os.path.join(rundir, f"out_{tag}.json")
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
    out['status'] = 'CLOSED' if out['surv_Q'] == 0 and out['surv_F'] >= 0 else 'NOT_CLOSED'
    return out


def main():
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    rundir = os.path.join(ROOT, f'proof_push_v2_{ts}')
    os.makedirs(rundir, exist_ok=True)
    summary = {
        'started_utc': ts,
        'c_targets': C_TARGETS,
        'configs_tried': [(n, m) for n, m, _ in CONFIGS],
        'host': os.uname().nodename,
        'attempts': [],
        'closures': {},
        'prior_closures_v1': {
            '1.295': {'n_half': 6, 'm': 15, 'd': 12, 'wall_sec': 309.7,
                      'surv_F': 0, 'surv_Q': 0,
                      'json_path': '/home/ubuntu/proof_push_high_20260507_124729/out_c12950_n6_m15.json'},
            '1.30':  {'n_half': 6, 'm': 15, 'd': 12, 'wall_sec': 313.7,
                      'surv_F': 0, 'surv_Q': 0,
                      'json_path': '/home/ubuntu/proof_push_high_20260507_124729/out_c13000_n6_m15.json'},
            '1.31':  {'n_half': 6, 'm': 15, 'd': 12, 'wall_sec': 320.9,
                      'surv_F': 10, 'surv_Q': 0,
                      'json_path': '/home/ubuntu/proof_push_high_20260507_124729/out_c13100_n6_m15.json'},
        },
    }
    summary_path = os.path.join(rundir, 'summary.json')
    txt_path = os.path.join(rundir, 'summary.txt')

    def save():
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        with open(txt_path, 'w') as f:
            f.write(f"Push-c v2 — started {ts}\n")
            f.write(f"Host: {summary['host']}\n\n")
            f.write("Prior closures (v1):\n")
            for c, info in summary['prior_closures_v1'].items():
                f.write(f"  c={c}: ({info['n_half']},{info['m']}) "
                        f"d={info['d']} F={info['surv_F']} Q={info['surv_Q']} "
                        f"in {info['wall_sec']}s\n")
            f.write("\n")
            f.write(f"{'c_target':>9} {'n':>3} {'m':>3} {'d':>3} "
                    f"{'comps':>14} {'F_surv':>8} {'Q_surv':>8} "
                    f"{'wall_s':>8} {'status'}\n")
            f.write('-' * 90 + '\n')
            for a in summary['attempts']:
                cps = a.get('n_processed', a.get('n_compositions', -1))
                cps_s = f"{cps:,}" if cps >= 0 else 'n/a'
                f.write(f"{a['c_target']:>9.4f} {a['n_half']:>3} "
                        f"{a['m']:>3} {a['d']:>3} {cps_s:>14} "
                        f"{a.get('surv_F','?'):>8} "
                        f"{a.get('surv_Q','?'):>8} "
                        f"{a['wall_sec']:>8.1f} {a['status']}\n")
            f.write('\n=== Closures (v2) ===\n')
            for c, info in summary['closures'].items():
                f.write(f"  c={c}: closed by ({info['n_half']},{info['m']}) "
                        f"d={info['d']} F={info['surv_F']} Q={info['surv_Q']} "
                        f"in {info['wall_sec']}s\n")
            if 'first_failure' in summary:
                f.write(f"  *** First c with NO closure: "
                        f"{summary['first_failure']} (stopped). ***\n")

    print(f"Run dir: {rundir}", flush=True)
    save()
    t_start = time.time()

    for c_target in C_TARGETS:
        print(f"\n===== c_target = {c_target} =====", flush=True)
        any_feasible = False
        closed = False
        for nh, m, timeout in CONFIGS:
            if vacuous(c_target, nh, m):
                continue
            any_feasible = True
            comps = n_compositions(nh, m)
            corr = correction(m, nh)
            print(f"  ({nh},{m},d={2*nh}): {comps:,} comps, "
                    f"corr={corr:.5f} thresh={c_target+corr:.5f}", flush=True)
            r = run_one(rundir, c_target, nh, m, timeout)
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
        if not any_feasible:
            print(f"  --- ALL VACUOUS.  STOPPING.", flush=True)
            summary['first_failure'] = c_target
            save()
            break
        if not closed:
            print(f"  --- NO CLOSURE for c={c_target}.  STOPPING.", flush=True)
            summary['first_failure'] = c_target
            save()
            break

    summary['elapsed_sec'] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== DONE.  v2 wall {summary['elapsed_sec']}s ===", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    closed_cs = sorted([float(c) for c in list(summary['closures'].keys()) +
                          list(summary['prior_closures_v1'].keys())])
    if closed_cs:
        print(f"Highest c closed (overall): {max(closed_cs)}", flush=True)


if __name__ == '__main__':
    main()
