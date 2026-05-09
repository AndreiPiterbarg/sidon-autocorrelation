"""Proof sweep — find configs that close L0 at given c_target.

For each c_target, sweep (n_half, m) configs in order of cost.
Run _L_bench.py with --max_l 0 (skip Lasserre SDP — CLARABEL broken on pod;
F+Q is enough to certify closure when surv_Q == 0).

Per (c, n, m):
   - Hard timeout (proportional to expected work)
   - Parse JSON output, record F/Q survivor counts
   - If surv_Q == 0  -> CLOSURE for this c, advance to next c.
   - If surv_Q  > 0  -> not a closure; advance to next (n, m).

All logs + JSONs under /home/ubuntu/proof_sweep_<TS>/.
Final summary table written to summary.json + summary.txt.

The bench's F+Q math is sound by construction (LP-tight per-window
correction; Q is a tightening of F via multi-window LP duality).  Soundness
audited empirically across 12M+ compositions in the historical bench files.
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
C_TARGETS = [1.2805, 1.281, 1.285, 1.29]

# (n_half, m) configs ordered by computational cost ascending.
# d = 2*n_half.  Palindromic enumeration: half has n_half parts summing to 2*n_half*m.
# n_processed = C(2*n_half*m + n_half - 1, n_half - 1).
CONFIGS = [
    (5, 5),    # d=10,  316,251 comps      ~0.3 s
    (6, 5),    # d=12,  8,259,888 comps    ~1.3 s
    (5, 8),    # d=10,  ~1.85 M comps      ~2 s
    (5, 10),   # d=10,  ~3.5 M comps       ~3 s
    (6, 8),    # d=12,  ~79 M comps        ~10-20 s
    (5, 15),   # d=10,  ~16 M comps        ~10 s
    (6, 10),   # d=12,  ~234 M comps       ~30-60 s
    (7, 5),    # d=14,  ~1.5 B comps       ~3-6 min
    (5, 20),   # d=10,  ~46 M comps        ~30 s
    (6, 15),   # d=12,  ~1.95 B comps      ~3-6 min
    (8, 5),    # d=16,  ~1.3e10 comps      probably too big — try if time
]


def n_compositions(n_half, m):
    """Total palindromic compositions: half has n_half parts summing to 2*n_half*m."""
    half_sum = 2 * n_half * m
    return comb(half_sum + n_half - 1, n_half - 1)


def estimate_timeout(n_half, m):
    """Rough wall budget per (n,m) config in seconds, based on 6 M comps/sec."""
    nc = n_compositions(n_half, m)
    base = 30  # JIT warmup + Q LP overhead
    return int(base + nc / 6_000_000)


def vacuous(c_target, n_half, m):
    return c_target + correction(m, n_half) >= C_UPPER


def run_one(rundir: str, c_target: float, n_half: int, m: int) -> dict:
    """Run _L_bench.py at (c, n, m) with --max_l 0 (skip SDP).
    Returns dict with all reported counts + status."""
    tag = f"c{int(round(c_target*10000)):05d}_n{n_half}_m{m}"
    log_path = os.path.join(rundir, f"log_{tag}.log")
    json_path = os.path.join(rundir, f"out_{tag}.json")
    timeout = estimate_timeout(n_half, m) + 60  # safety margin
    timeout = min(timeout, 1500)  # hard cap 25 min per config

    t0 = time.time()
    cmd = [
        'python3', '-u', os.path.join(ROOT, '_L_bench.py'),
        '--n_half', str(n_half),
        '--m', str(m),
        '--c_target', str(c_target),
        '--solver', 'CLARABEL',
        '--order', '1',
        '--max_l', '0',  # skip SDP entirely (we only need F + Q)
        '--audit', '0',  # skip audit (not needed for proof; saves time)
        '--out', json_path,
    ]
    print(f"  [{tag}] launching ...  timeout={timeout}s", flush=True)
    with open(log_path, 'w') as logf:
        try:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1
            print(f"  [{tag}] *** TIMEOUT >{timeout}s — killing", flush=True)
    wall = time.time() - t0

    out = {
        'c_target': c_target, 'n_half': n_half, 'm': m,
        'd': 2 * n_half, 'wall_sec': round(wall, 2),
        'log_path': log_path, 'json_path': json_path,
        'returncode': rc,
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
    rundir = os.path.join(ROOT, f'proof_sweep_{ts}')
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
            f.write(f"Proof sweep — started {ts}\n")
            f.write(f"Host: {summary['host']}\n\n")
            f.write(f"{'c_target':>9} {'n':>3} {'m':>3} {'d':>3} "
                    f"{'comps':>14} {'F_surv':>8} {'Q_surv':>8} "
                    f"{'wall_s':>8} {'status'}\n")
            f.write('-' * 90 + '\n')
            for a in summary['attempts']:
                cps = a.get('n_processed', -1)
                cps_s = f"{cps:,}" if cps >= 0 else 'n/a'
                f.write(f"{a['c_target']:>9.4f} {a['n_half']:>3} "
                        f"{a['m']:>3} {a['d']:>3} {cps_s:>14} "
                        f"{a.get('surv_F', '?'):>8} "
                        f"{a.get('surv_Q', '?'):>8} "
                        f"{a['wall_sec']:>8.1f} {a['status']}\n")
            f.write('\n=== Closures ===\n')
            for c, info in summary['closures'].items():
                f.write(f"  c={c}: closed by ({info['n_half']},{info['m']}) d={info['d']} "
                        f"with F={info['surv_F']} Q={info['surv_Q']} in {info['wall_sec']}s\n")
            f.write(f"  c-targets without closure: "
                    f"{[c for c in C_TARGETS if c not in summary['closures']]}\n")

    print(f"Run dir: {rundir}", flush=True)
    save()

    t_start = time.time()

    for c_target in C_TARGETS:
        print(f"\n===== c_target = {c_target} =====", flush=True)
        for nh, m in CONFIGS:
            if vacuous(c_target, nh, m):
                print(f"  ({nh},{m}): VACUOUS at c={c_target}, skip", flush=True)
                continue
            comps = n_compositions(nh, m)
            print(f"  ({nh},{m},d={2*nh}): {comps:,} comps "
                    f"corr={correction(m, nh):.5f} "
                    f"thresh={c_target+correction(m, nh):.5f}", flush=True)
            r = run_one(rundir, c_target, nh, m)
            summary['attempts'].append(r)
            print(f"  -> {r['status']}  F={r.get('surv_F','?')} "
                    f"Q={r.get('surv_Q','?')}  wall={r['wall_sec']}s", flush=True)
            save()
            if r['status'] == 'CLOSED':
                print(f"  *** CLOSURE: c={c_target} closed at "
                        f"(n_half={nh}, m={m}, d={2*nh}) ***", flush=True)
                summary['closures'][str(c_target)] = r
                save()
                break  # advance to next c
        else:
            print(f"  --- no closure found for c={c_target} in tried configs", flush=True)

    summary['elapsed_sec'] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== DONE.  total wall {summary['elapsed_sec']}s ===", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == '__main__':
    main()
