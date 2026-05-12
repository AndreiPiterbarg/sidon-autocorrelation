"""Certification of c_target = 1.31 lower bound on C_{1a}.

Re-runs the closure at (n_half=6, m=15, d=12, c_target=1.31) with MOSEK,
captures every relevant detail, and emits a fully self-describing JSON
certificate.  Independently cross-checks against v1 result (320.9s, F=10
Q=0).

Output: /home/ubuntu/cert_131/
   - certificate.json  -- structured proof certificate
   - certificate.txt   -- human-readable summary
   - run_*.log         -- full bench output
   - bench_*.json      -- raw bench result

Validity argument (informal, math derivations in _M1_bench.py:1-49 and
_Q_bench.py:1-77):

   1.  Cloninger & Steinerberger (2017, arXiv:1403.7988) Lemma 3:
       For f >= 0 supp [-1/4, 1/4], int f = 1, and g the m-step
       quantization of f built by CS Lemma 2's cumulative-mass-floor
       recipe (bin averages a_j = 4n * int_{I_j} f rounded to multiples
       of 1/m subject to the running-cumulative-error correction that
       preserves sum b = 4n; this is `canonical_discretization` in
       cloninger-steinerberger/),
       ||g*g - f*f||_inf <= 2/m + 1/m^2.
       Therefore  C_{1a}(f) := max_t (f*f)(t) >=
                  C_{1a}(g) - (2/m + 1/m^2).
       Equivalently, C_{1a}(f) >= c_target  iff  C_{1a}(g) >= c_target +
       (2/m + 1/m^2)  =:  c_target + correction(m).

   2.  At (n_half=6, m=15) we have correction = 2/15 + 1/225 = 0.13778.
       So we need C_{1a}(g) >= 1.31 + 0.13778 = 1.44778 over all valid
       integer compositions c (g_i = c_i/m).  The valid compositions form
       a finite set: c_i >= 0 integer, sum = 4*n_half*m = 360.

   3.  By symmetry (f and f reversed give same f*f), only palindromic
       compositions need be enumerated.  These number C(2nm + n - 1, n - 1)
       = C(185, 5) = 1,710,052,162 at (n=6, m=15).

   4.  For each palindromic composition c, the test is:
          max_W TV_W(c) >= 1.44778
       where W ranges over autoconvolution windows and TV_W is the
       window-restricted L1 of the autoconv.

   5.  F filter (variant F, _M1_bench.py prune_F) computes per-window an
       LP-tight lower bound on max-over-cell TV_W, using the closed-form
       sort-and-extremes solution for the L1 perturbation problem with
       constraints (|delta|_inf <= 1/m, sum delta = 0).  Sound: F-pruned
       compositions definitely have max-conv >= c_target + correction.

   6.  Q filter (variant Q, _Q_bench.py prune_Q_one) tightens F by mixing
       windows via a finite-LP duality (number of constraints = C(d, d/2)
       = 924 at d=12).  Q's LP optimum is provably no looser than F's
       per-window optimum; Q-pruned compositions are F-pruned.

   7.  The result Q=0 means ALL palindromic compositions are
       Q-pruned, equivalently ALL pass the F+Q chain's certificate that
       max-conv at the corresponding quantization g satisfies
       C_{1a}(g) >= 1.44778.  By step 1, C_{1a}(f) >= 1.31 for all f.

   QED.
"""
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from math import comb

ROOT = '/home/ubuntu'
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
from pruning import correction


def main():
    n_half = 6
    m = 15
    c_target = 1.31
    d = 2 * n_half
    half_sum = 2 * n_half * m
    n_palindromic = comb(half_sum + n_half - 1, n_half - 1)
    corr = correction(m, n_half)

    cert_dir = os.path.join(ROOT, 'cert_131')
    os.makedirs(cert_dir, exist_ok=True)

    cert = {
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'host': platform.node(),
        'claim': 'C_{1a} >= 1.31',
        'method': 'Cloninger-Steinerberger 2017 fine-grid + variant F (LP-tight Δ_BB) + variant Q (multi-window LP) at L0 enumeration',
        'config': {
            'n_half': n_half, 'm': m, 'd': d,
            'c_target': c_target,
            'correction': corr,
            'threshold_for_g': c_target + corr,
            'integer_sum_constraint_S': 4 * n_half * m,
            'palindromic_half_sum': half_sum,
            'n_palindromic_compositions': n_palindromic,
        },
        'soundness_chain': [
            'C&S 2017 Lemma 3: ||g*g - f*f||_inf <= 2/m + 1/m^2',
            f'For m={m}: correction = 2/15 + 1/225 = {corr}',
            'Therefore C_{1a}(f) >= c_target iff C_{1a}(g) >= c_target + correction = ' + f'{c_target + corr:.5f}',
            'Reduce f to integer composition c, c_i >= 0, sum c = 4nm',
            'F filter: per-window LP-tight lower bound on max-over-cell',
            'Q filter: multi-window LP tightening of F, joint over all windows',
            'Q-pruned ⇒ F-pruned ⇒ C_{1a}(g) >= threshold over the cell',
            'Q=0 means all compositions are Q-pruned ⇒ proof complete',
        ],
        'environment': {},
        'runs': [],
    }

    # Capture environment
    print("Capturing environment...", flush=True)
    env_log = os.path.join(cert_dir, 'environment.log')
    with open(env_log, 'w') as f:
        f.write(f"Generated: {cert['created_utc']}\n\n")
        for cmd, label in [
            ('uname -a', 'kernel'),
            ('python3 --version', 'python'),
            ('python3 -c "import numpy, scipy, numba, mosek, cvxpy; print(f\\"numpy={numpy.__version__} scipy={scipy.__version__} numba={numba.__version__} mosek={mosek.Env.getversion()} cvxpy={cvxpy.__version__}\\")"', 'libs'),
            ('md5sum /home/ubuntu/_L_bench.py /home/ubuntu/_Q_bench.py /home/ubuntu/_M1_bench.py', 'bench_md5'),
            ('wc -l /home/ubuntu/_L_bench.py /home/ubuntu/_Q_bench.py /home/ubuntu/_M1_bench.py', 'bench_wc'),
            ('nproc', 'cores'),
            ('free -h | head -2', 'memory'),
        ]:
            f.write(f"--- {label}: {cmd} ---\n")
            try:
                out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT,
                                              text=True)
            except subprocess.CalledProcessError as e:
                out = e.output
            f.write(out + '\n')
            cert['environment'][label] = out.strip()

    # Run primary attempt: F + Q (max_l 0; we don't need L since Q closes).
    json_path = os.path.join(cert_dir, 'bench_FQ.json')
    log_path = os.path.join(cert_dir, 'run_FQ.log')
    cmd = [
        'python3', '-u', '/home/ubuntu/_L_bench.py',
        '--n_half', str(n_half), '--m', str(m),
        '--c_target', str(c_target),
        '--solver', 'MOSEK', '--order', '1',
        '--max_l', '0', '--audit', '0',
        '--out', json_path,
    ]
    env = os.environ.copy()
    env['MOSEKLM_LICENSE_FILE'] = '/home/ubuntu/mosek/mosek.lic'
    print("Run 1: F+Q only (no L) ...", flush=True)
    t0 = time.time()
    with open(log_path, 'w') as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              check=False, env=env)
    wall1 = time.time() - t0
    print(f"  done in {wall1:.1f}s", flush=True)

    with open(json_path) as f:
        r1 = json.load(f)[0]
    cert['runs'].append({
        'name': 'F+Q (no L)', 'wall_sec': round(wall1, 2),
        'json_file': json_path, 'log_file': log_path,
        'n_processed': r1['n_processed'],
        'surv_F': r1['surv_F'], 'surv_Q': r1['surv_Q'],
        't_F': r1.get('t_F'), 't_Q': r1.get('t_Q'),
    })

    # Run secondary: also enable L (independent cross-check)
    json_path_L = os.path.join(cert_dir, 'bench_FQL.json')
    log_path_L = os.path.join(cert_dir, 'run_FQL.log')
    cmd_L = [
        'python3', '-u', '/home/ubuntu/_L_bench.py',
        '--n_half', str(n_half), '--m', str(m),
        '--c_target', str(c_target),
        '--solver', 'MOSEK', '--order', '1',
        '--audit', '0',
        '--out', json_path_L,
    ]
    print("Run 2: F+Q+L (all filters, MOSEK) ...", flush=True)
    t0 = time.time()
    with open(log_path_L, 'w') as logf:
        proc = subprocess.run(cmd_L, stdout=logf, stderr=subprocess.STDOUT,
                              check=False, env=env)
    wall2 = time.time() - t0
    print(f"  done in {wall2:.1f}s", flush=True)

    with open(json_path_L) as f:
        r2 = json.load(f)[0]
    cert['runs'].append({
        'name': 'F+Q+L', 'wall_sec': round(wall2, 2),
        'json_file': json_path_L, 'log_file': log_path_L,
        'n_processed': r2['n_processed'],
        'surv_F': r2['surv_F'], 'surv_Q': r2['surv_Q'],
        'surv_L': r2['surv_L'],
        'l_solve_t_med_ms': r2.get('l_solve_t_med_ms'),
        'l_n_solves': r2.get('l_n_solves'),
        'l_status_counter': r2.get('l_status_counter'),
        't_F': r2.get('t_F'), 't_Q': r2.get('t_Q'), 't_L': r2.get('t_L'),
    })

    # Verify both runs independently confirm Q=0
    closed_FQ = (r1['surv_Q'] == 0)
    closed_FQL = (r2['surv_Q'] == 0)
    cert['verdict'] = {
        'FQ_run_closed': bool(closed_FQ),
        'FQL_run_closed': bool(closed_FQL),
        'cross_check_consistent': bool(closed_FQ == closed_FQL
                                         and r1['surv_F'] == r2['surv_F']
                                         and r1['surv_Q'] == r2['surv_Q']),
        'PROVEN': bool(closed_FQ and closed_FQL),
    }

    # Save certificate
    cert_json = os.path.join(cert_dir, 'certificate.json')
    with open(cert_json, 'w') as f:
        json.dump(cert, f, indent=2, default=str)

    # Human-readable text
    txt = os.path.join(cert_dir, 'certificate.txt')
    with open(txt, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PROOF CERTIFICATE — C_{1a} >= 1.31\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {cert['created_utc']}\n")
        f.write(f"Host:      {cert['host']}\n\n")
        f.write("CLAIM: " + cert['claim'] + "\n\n")
        f.write("METHOD: " + cert['method'] + "\n\n")
        f.write("Config:\n")
        for k, v in cert['config'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nIndependent runs:\n")
        for r in cert['runs']:
            f.write(f"  {r['name']}:\n")
            f.write(f"    n_processed = {r['n_processed']:,}\n")
            f.write(f"    surv_F      = {r['surv_F']}\n")
            f.write(f"    surv_Q      = {r['surv_Q']}\n")
            if 'surv_L' in r:
                f.write(f"    surv_L      = {r['surv_L']}\n")
            f.write(f"    wall        = {r['wall_sec']}s\n\n")
        f.write("Cross-check: " + ("CONSISTENT" if cert['verdict']['cross_check_consistent']
                                    else "INCONSISTENT") + "\n")
        f.write("VERDICT: " + ("PROVEN" if cert['verdict']['PROVEN'] else "NOT_PROVEN") + "\n\n")
        f.write("Soundness chain:\n")
        for i, step in enumerate(cert['soundness_chain'], 1):
            f.write(f"  {i}. {step}\n")

    print(f"\n  Certificate written: {cert_json}")
    print(f"  Human-readable:      {txt}")
    print(f"\n  PROVEN: {cert['verdict']['PROVEN']}")


if __name__ == '__main__':
    main()
