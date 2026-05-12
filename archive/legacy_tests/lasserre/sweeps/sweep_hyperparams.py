#!/usr/bin/env python
"""Comprehensive hyperparameter sweep for the Lasserre L3 CG pipeline at d=16.

Tests configurations that were hardcoded and never measured:

 ADMM internals (env-var overrides in admm_gpu_solver.py):
 SIDON_AA_MEM -- Anderson history window size (default 5)
 SIDON_AA_INTERVAL -- AA application frequency (in iters) (default 10)
 SIDON_AA_BETA -- AA damping coefficient (default 0.85)
 SIDON_AA_RST -- AA periodic restart period (default 100)

 CLI args on run_d16_l3.py:
 --rho -- ADMM penalty parameter (default 0.1)
 --atom-frac -- atom-ranked fraction of cuts (default 0.5)
 --cuts-per-round -- window budget per CG round (default 100)
 --k-vecs -- spectral eigvecs per window check (default 3)

STRATEGY
--------
Phase 1 (Round 0 only, ~110s each): measures ADMM convergence quality
without triggering cut generation. Any ADMM-internal parameter that
affects convergence shows up in Round 0 iter count and scalar bound.

Phase 2 (Round 0 + 1, ~300s each): measures cut-selection quality.
`atom_frac` and `cuts_per_round` only matter once CG rounds start.

BUDGET
------
Target: <= 1 hour total. With 8 Round-0 configs × 110s ~ 15 min plus
5 Round-0+1 configs × 300s ~ 25 min plus baseline ~ 5 min => ~45 min.

Each subprocess is bounded by `timeout` with 30% headroom.

USAGE
-----
 # On the GPU pod:
 cd /workspace/sidon-autocorrelation
 python tests/sweep_hyperparams.py \
 --out data/sweep_report.json

OUTPUT
------
 - One JSON file per run in /tmp/sweep_<label>.log
 - Aggregate JSON at --out path
 - Markdown table printed to stdout at the end
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_CLI = {
 '--d': 16,
 '--order': 3,
 '--bw': 15,
 '--scs-iters': 20000,
 '--scs-eps': '1e-5',
 '--k-vecs': 3,
 '--cuts-per-round': 100,
 '--rho': '0.1',
 '--atom-frac': '0.5',
}

# Phase 1: ADMM-internal sweeps, Round 0 only (--cg-rounds 0)
# Baseline is implicit: DEFAULT_CLI + no env overrides.
PHASE1_CONFIGS: List[Dict[str, Any]] = [
 # ---- Anderson memory (currently 5, untested) ----
 {'label': 'aa_mem=2', 'env': {'SIDON_AA_MEM': '2'}, 'cli': {}},
 {'label': 'aa_mem=3', 'env': {'SIDON_AA_MEM': '3'}, 'cli': {}},
 {'label': 'aa_mem=8', 'env': {'SIDON_AA_MEM': '8'}, 'cli': {}},

 # ---- Anderson interval (currently 10, untested) ----
 {'label': 'aa_int=5', 'env': {'SIDON_AA_INTERVAL': '5'}, 'cli': {}},
 {'label': 'aa_int=20', 'env': {'SIDON_AA_INTERVAL': '20'}, 'cli': {}},
 {'label': 'aa_int=1e9', 'env': {'SIDON_AA_INTERVAL': '1000000000'}, 'cli': {}},

 # ---- Anderson damping (currently 0.85, untested) ----
 {'label': 'aa_beta=0.7', 'env': {'SIDON_AA_BETA': '0.7'}, 'cli': {}},
 {'label': 'aa_beta=1.0', 'env': {'SIDON_AA_BETA': '1.0'}, 'cli': {}},

 # ---- Anderson periodic restart (currently 100, untested) ----
 {'label': 'aa_rst=50', 'env': {'SIDON_AA_RST': '50'}, 'cli': {}},
 {'label': 'aa_rst=200', 'env': {'SIDON_AA_RST': '200'}, 'cli': {}},

 # ---- rho (previously swept at d=32 L2; confirm 0.1 is still optimal at d=16 L3) ----
 {'label': 'rho=0.05', 'env': {}, 'cli': {'--rho': '0.05'}},
 {'label': 'rho=0.2', 'env': {}, 'cli': {'--rho': '0.2'}},
]

# Phase 2: Cut-selection sweeps, Round 0 + Round 1 (--cg-rounds 1)
PHASE2_CONFIGS: List[Dict[str, Any]] = [
 # ---- Atom ranking blend (currently 0.5, never swept) ----
 {'label': 'atom=0.0', 'env': {}, 'cli': {'--atom-frac': '0.0'}}, # pure eig
 {'label': 'atom=0.25', 'env': {}, 'cli': {'--atom-frac': '0.25'}},
 {'label': 'atom=0.75', 'env': {}, 'cli': {'--atom-frac': '0.75'}},
 {'label': 'atom=1.0', 'env': {}, 'cli': {'--atom-frac': '1.0'}}, # pure atom

 # ---- Cuts per round (100 default, 50 faster, 200 denser) ----
 {'label': 'cuts=50', 'env': {}, 'cli': {'--cuts-per-round': '50'}},
 {'label': 'cuts=200', 'env': {}, 'cli': {'--cuts-per-round': '200'}},
]


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

SCALAR_BOUND_RE = re.compile(r"Scalar bound\s*=\s*([0-9.]+)\s*\(([0-9.]+)s,\s*([0-9]+)\s*iters\)")
LB_CHECKPOINT_RE = re.compile(r"lb=([0-9.]+)\s*\([^)]*\)\s*gc=([\-0-9.]+)%")
FINAL_RE = re.compile(r"FINAL:.*")
PAIRWISE_RE = re.compile(r"Pairwise localizing:\s*(\d+)/(\d+)")


def run_one(
 label: str,
 env_overrides: Dict[str, str],
 cli_overrides: Dict[str, Any],
 cg_rounds: int,
 n_bisect: int,
 workdir: Path,
 run_script: str,
 timeout_s: int,
 verbose: bool = True,
) -> Dict[str, Any]:
 """Execute one sweep configuration and parse its output."""
 cli: Dict[str, Any] = {**DEFAULT_CLI, **cli_overrides}
 cli['--cg-rounds'] = cg_rounds
 cli['--bisect'] = n_bisect
 if '--gpu' not in cli:
 cli['--gpu'] = None # flag

 # Build env
 env = os.environ.copy()
 env.update({k: str(v) for k, v in env_overrides.items()})

 # Build command
 parts = [sys.executable, '-u', run_script]
 for k, v in cli.items():
 if v is None:
 parts.append(k)
 else:
 parts.extend([k, str(v)])

 log_file = workdir / f"sweep_{label.replace('=', '_').replace('.', 'p')}.log"
 log_file.parent.mkdir(parents=True, exist_ok=True)

 if verbose:
 print(f"\n── {label} ─────────────────────────────────────────", flush=True)
 env_str = " ".join(f"{k}={v}" for k, v in env_overrides.items()) or "(no env)"
 cli_str = " ".join(
 f"{k} {v}" if v is not None else k
 for k, v in cli_overrides.items()) or "(default)"
 print(f" env: {env_str}", flush=True)
 print(f" cli: {cli_str}", flush=True)
 print(f" log: {log_file}", flush=True)

 t0 = time.time()
 try:
 r = subprocess.run(
 parts, env=env, timeout=timeout_s,
 capture_output=True, text=True, cwd=workdir.parent,
 )
 stdout = r.stdout
 stderr = r.stderr
 rc = r.returncode
 except subprocess.TimeoutExpired as te:
 stdout = te.stdout.decode() if isinstance(te.stdout, bytes) else (te.stdout or "")
 stderr = te.stderr.decode() if isinstance(te.stderr, bytes) else (te.stderr or "")
 rc = -1
 elapsed = time.time() - t0

 log_file.write_text(stdout + ("\n--- STDERR ---\n" + stderr if stderr else ""))

 # Parse
 scalar_bound: Optional[float] = None
 scalar_time: Optional[float] = None
 scalar_iters: Optional[int] = None
 m = SCALAR_BOUND_RE.search(stdout)
 if m:
 scalar_bound = float(m.group(1))
 scalar_time = float(m.group(2))
 scalar_iters = int(m.group(3))

 # Round 1 lb (last Checkpoint match before FINAL)
 cg1_lb: Optional[float] = None
 cg1_gc: Optional[float] = None
 for cm in LB_CHECKPOINT_RE.finditer(stdout):
 cg1_lb = float(cm.group(1))
 cg1_gc = float(cm.group(2))

 pairwise = PAIRWISE_RE.search(stdout)
 n_pairwise = int(pairwise.group(1)) if pairwise else None

 result = {
 'label': label,
 'env_overrides': env_overrides,
 'cli_overrides': cli_overrides,
 'cg_rounds': cg_rounds,
 'n_bisect': n_bisect,
 'wall_s': round(elapsed, 1),
 'returncode': rc,
 'scalar_bound': scalar_bound,
 'scalar_time_s': scalar_time,
 'scalar_iters': scalar_iters,
 'cg1_lb': cg1_lb,
 'cg1_gc_pct': cg1_gc,
 'n_pairwise_cones': n_pairwise,
 'log_file': str(log_file),
 }

 if verbose:
 ok_mark = "" if rc == 0 else ("×" if rc != -1 else "T")
 print(f" [{ok_mark}] scalar={scalar_bound} iters={scalar_iters} "
 f"cg1_lb={cg1_lb} gc={cg1_gc}% wall={elapsed:.1f}s", flush=True)
 return result


def main() -> None:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--run-script', default='tests/run_d16_l3.py',
 help='Path to solver CLI (default tests/run_d16_l3.py)')
 ap.add_argument('--workdir', default='/tmp/sidon_sweep',
 help='Directory for per-run log files')
 ap.add_argument('--out', default='data/sweep_hyperparams.json',
 help='Aggregate JSON result path')
 ap.add_argument('--phase1', action='store_true', default=True,
 help='Run Phase 1 (ADMM internals, Round 0 only)')
 ap.add_argument('--phase2', action='store_true', default=True,
 help='Run Phase 2 (cut selection, Round 0+1)')
 ap.add_argument('--baseline-only', action='store_true',
 help='Just run the baseline config -- for sanity')
 ap.add_argument('--skip-baseline', action='store_true',
 help='Skip the baseline run (e.g. already have numbers)')
 ap.add_argument('--phase1-bisect', type=int, default=0,
 help='Bisect count for Phase 1 (0=Round 0 only)')
 ap.add_argument('--phase2-bisect', type=int, default=5,
 help='Bisect count for Phase 2 (default 5)')
 ap.add_argument('--phase1-timeout', type=int, default=240,
 help='Per-config timeout for Phase 1 (seconds)')
 ap.add_argument('--phase2-timeout', type=int, default=540,
 help='Per-config timeout for Phase 2 (seconds)')
 args = ap.parse_args()

 workdir = Path(args.workdir)
 workdir.mkdir(parents=True, exist_ok=True)

 repo_root = Path(__file__).resolve().parent.parent
 run_script = str(repo_root / args.run_script) if not Path(args.run_script).is_absolute() else args.run_script

 all_results: List[Dict[str, Any]] = []

 print(f"Sweep started at {datetime.now().isoformat()}")
 print(f" Repo: {repo_root}")
 print(f" Run script: {run_script}")
 print(f" Workdir: {workdir}")
 print(f" Out: {args.out}")
 print(f" Phase 1 budget: {args.phase1_timeout}s per config, {len(PHASE1_CONFIGS)} configs")
 print(f" Phase 2 budget: {args.phase2_timeout}s per config, {len(PHASE2_CONFIGS)} configs")

 t_sweep = time.time()

 # Baseline
 if not args.skip_baseline:
 baseline = run_one(
 'BASELINE', env_overrides={}, cli_overrides={},
 cg_rounds=1, n_bisect=args.phase2_bisect,
 workdir=workdir, run_script=run_script,
 timeout_s=args.phase2_timeout,
 )
 baseline['phase'] = 'baseline'
 all_results.append(baseline)

 if args.baseline_only:
 _write_report(all_results, args.out, t_sweep)
 return

 # Phase 1
 if args.phase1:
 for cfg in PHASE1_CONFIGS:
 r = run_one(
 cfg['label'], cfg['env'], cfg['cli'],
 cg_rounds=0, n_bisect=args.phase1_bisect,
 workdir=workdir, run_script=run_script,
 timeout_s=args.phase1_timeout,
 )
 r['phase'] = 1
 all_results.append(r)
 _write_report(all_results, args.out, t_sweep)

 # Phase 2
 if args.phase2:
 for cfg in PHASE2_CONFIGS:
 r = run_one(
 cfg['label'], cfg['env'], cfg['cli'],
 cg_rounds=1, n_bisect=args.phase2_bisect,
 workdir=workdir, run_script=run_script,
 timeout_s=args.phase2_timeout,
 )
 r['phase'] = 2
 all_results.append(r)
 _write_report(all_results, args.out, t_sweep)

 total_elapsed = time.time() - t_sweep
 print(f"\n╔══ SWEEP COMPLETE ══╗")
 print(f" Total time: {total_elapsed:.0f}s = {total_elapsed/60:.1f} min")
 print(f" Configs run: {len(all_results)}")
 print(f" Report: {args.out}")
 _print_summary(all_results)


def _write_report(results: List[Dict[str, Any]], out_path: str, t_sweep: float) -> None:
 """Incrementally flush the aggregate JSON so partial progress survives crashes."""
 Path(out_path).parent.mkdir(parents=True, exist_ok=True)
 payload = {
 'generated_at': datetime.now().isoformat(),
 'elapsed_s': round(time.time() - t_sweep, 1),
 'n_configs': len(results),
 'results': results,
 }
 Path(out_path).write_text(json.dumps(payload, indent=2))


def _print_summary(results: List[Dict[str, Any]]) -> None:
 """Markdown table summary sorted by usefulness per phase."""
 print()
 print("## Phase 1 -- ADMM internals (Round 0 only)")
 p1 = [r for r in results if r.get('phase') == 1]
 if p1:
 p1.sort(key=lambda r: -(r.get('scalar_bound') or 0))
 print(f"{'label':<14} | {'scalar_lb':>10} | {'iters':>6} | {'sec':>6}")
 print("-" * 44)
 for r in p1:
 sb = r.get('scalar_bound') or 0
 it = r.get('scalar_iters') or 0
 tm = r.get('scalar_time_s') or 0
 print(f"{r['label']:<14} | {sb:>10.6f} | {it:>6} | {tm:>6.1f}")

 print()
 print("## Phase 2 -- Cut selection (Round 0 + Round 1)")
 p2 = [r for r in results if r.get('phase') == 2]
 if p2:
 p2.sort(key=lambda r: -(r.get('cg1_lb') or 0))
 print(f"{'label':<14} | {'cg1_lb':>10} | {'cg1_gc':>7} | {'wall':>6}")
 print("-" * 44)
 for r in p2:
 lb = r.get('cg1_lb') or 0
 gc = r.get('cg1_gc_pct') or 0
 wl = r.get('wall_s') or 0
 print(f"{r['label']:<14} | {lb:>10.6f} | {gc:>6.1f}% | {wl:>6.0f}s")

 print()
 print("## Baseline")
 bl = [r for r in results if r.get('phase') == 'baseline']
 for r in bl:
 print(f" scalar_lb={r.get('scalar_bound')} iters={r.get('scalar_iters')}")
 print(f" cg1_lb={r.get('cg1_lb')} gc={r.get('cg1_gc_pct')}% wall={r.get('wall_s')}s")


if __name__ == '__main__':
 main()
