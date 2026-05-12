"""Proof sweep v3 — MOSEK + L (Lasserre) enabled.

CLARABEL was throwing TypeError inside SDP setup; MOSEK is licensed (key
shipped to /home/ubuntu/mosek/mosek.lic) and works.  L (Lasserre/Shor SDP)
empirically kills 90-100% of Q-survivors at d>=10.

Strategy: use F+Q+L on (6,15,d=12) for c >= 1.32.  If Q leaves N survivors,
L runs ~100-200 ms per SDP, so N*0.15s extra wall.  At N=1046 (c=1.32 from
v2) that's 10-20 min L wall on top of 5-10 min F+Q.

Resume from c=1.32.
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

C_TARGETS = [
    1.32, 1.33, 1.34, 1.35, 1.36,
    1.365,
    1.37, 1.38, 1.39, 1.40,
    1.41, 1.42, 1.43,
    1.44, 1.45, 1.46,
]

# (n_half, m, timeout_sec, max_l).
# max_l caps the number of SDPs L will solve (None = all).
# We always allow all L SDPs since L's strength is the whole point.
CONFIGS = [
    (6, 15, 1800, None),     # d=12, 1.71B comps
    (6, 20, 3600, None),     # d=12, 7B comps
    (6, 30, 7200, None),     # d=12, 52.5B comps
]

ENV = os.environ.copy()
ENV['MOSEKLM_LICENSE_FILE'] = '/home/ubuntu/mosek/mosek.lic'


def n_compositions(n_half, m):
    half_sum = 2 * n_half * m
    return comb(half_sum + n_half - 1, n_half - 1)


def vacuous(c_target, n_half, m):
    return c_target + correction(m, n_half) >= C_UPPER


def run_one(rundir, c_target, n_half, m, timeout, max_l):
    tag = f"c{int(round(c_target*10000)):05d}_n{n_half}_m{m}"
    log_path = os.path.join(rundir, f"log_{tag}.log")
    json_path = os.path.join(rundir, f"out_{tag}.json")
    t0 = time.time()
    cmd = [
        'python3', '-u', os.path.join(ROOT, '_L_bench.py'),
        '--n_half', str(n_half),
        '--m', str(m),
        '--c_target', str(c_target),
        '--solver', 'MOSEK',
        '--order', '1',
        '--audit', '0',
        '--out', json_path,
    ]
    if max_l is not None:
        cmd.extend(['--max_l', str(max_l)])
    print(f"  [{tag}] launching ...  timeout={timeout}s "
            f"({n_compositions(n_half,m):,} comps)", flush=True)
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
    out['surv_L'] = r.get('surv_L', -1)
    if out['surv_L'] == 0 and out['surv_Q'] >= 0:
        out['status'] = 'CLOSED_BY_L'
    elif out['surv_Q'] == 0 and out['surv_F'] >= 0:
        out['status'] = 'CLOSED_BY_Q'
    else:
        out['status'] = 'NOT_CLOSED'
    return out


def main():
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    rundir = os.path.join(ROOT, f'proof_push_v3_{ts}')
    os.makedirs(rundir, exist_ok=True)
    summary = {
        'started_utc': ts,
        'c_targets': C_TARGETS,
        'configs_tried': [(n, m) for n, m, _, _ in CONFIGS],
        'host': os.uname().nodename,
        'attempts': [],
        'closures': {},
        'prior_closures': {
            'v1_1.295': '(6,15) d=12 F=0 Q=0 309.7s',
            'v1_1.30':  '(6,15) d=12 F=0 Q=0 313.7s',
            'v1_1.31':  '(6,15) d=12 F=10 Q=0 320.9s',
        },
    }
    summary_path = os.path.join(rundir, 'summary.json')
    txt_path = os.path.join(rundir, 'summary.txt')

    def save():
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        with open(txt_path, 'w') as f:
            f.write(f"Push-c v3 (MOSEK+L) — started {ts}\n")
            f.write(f"Host: {summary['host']}\n\n")
            f.write("Prior closures: " + str(summary['prior_closures']) + "\n\n")
            f.write(f"{'c_target':>9} {'n':>3} {'m':>3} {'d':>3} "
                    f"{'comps':>14} {'F':>6} {'Q':>6} {'L':>6} "
                    f"{'wall_s':>8} {'status'}\n")
            f.write('-' * 95 + '\n')
            for a in summary['attempts']:
                cps = a.get('n_processed', a.get('n_compositions', -1))
                cps_s = f"{cps:,}" if cps >= 0 else 'n/a'
                f.write(f"{a['c_target']:>9.4f} {a['n_half']:>3} "
                        f"{a['m']:>3} {a['d']:>3} {cps_s:>14} "
                        f"{a.get('surv_F','?'):>6} "
                        f"{a.get('surv_Q','?'):>6} "
                        f"{a.get('surv_L','?'):>6} "
                        f"{a['wall_sec']:>8.1f} {a['status']}\n")
            f.write('\n=== Closures (v3) ===\n')
            for c, info in summary['closures'].items():
                f.write(f"  c={c}: ({info['n_half']},{info['m']}) d={info['d']} "
                        f"F={info['surv_F']} Q={info['surv_Q']} "
                        f"L={info['surv_L']} in {info['wall_sec']}s "
                        f"({info['status']})\n")
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
        for nh, m, timeout, max_l in CONFIGS:
            if vacuous(c_target, nh, m):
                continue
            any_feasible = True
            comps = n_compositions(nh, m)
            corr = correction(m, nh)
            print(f"  ({nh},{m},d={2*nh}): {comps:,} comps, "
                    f"corr={corr:.5f} thresh={c_target+corr:.5f}", flush=True)
            r = run_one(rundir, c_target, nh, m, timeout, max_l)
            summary['attempts'].append(r)
            print(f"  -> {r['status']}  F={r.get('surv_F','?')} "
                    f"Q={r.get('surv_Q','?')} L={r.get('surv_L','?')} "
                    f"wall={r['wall_sec']}s", flush=True)
            save()
            if r['status'].startswith('CLOSED'):
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
    print(f"\n=== DONE.  total wall {summary['elapsed_sec']}s ===", flush=True)


if __name__ == '__main__':
    main()
