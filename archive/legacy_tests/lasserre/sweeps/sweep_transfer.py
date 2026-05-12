#!/usr/bin/env python
"""Deep-CG transfer validation sweep.

Goal: confirm that parameter winners from `sweep_hyperparams.py` (which
only measures R0 and R1) still win at deeper CG rounds where production
runtime is actually spent.

Selects the top-K winners of the initial sweep and re-runs them at two
tiers of CG depth:

 T-mid (all configs): cg-rounds=5, bisect=8, scs-eps=1e-6 ~15 min each
 T-deep (top 2 + base): cg-rounds=10, bisect=10, scs-eps=1e-6 ~35 min each

See TRANSFER_SWEEP_PLAN.md for the full rationale.

USAGE
-----
 python tests/sweep_transfer.py --initial-report data/sweep_report.json \
 --out data/transfer_sweep_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------- Parsing helpers (mirror sweep_hyperparams.py) ------------
SCALAR_BOUND_RE = re.compile(r"Scalar bound\s*=\s*([0-9.]+)\s*\(([0-9.]+)s,\s*([0-9]+)\s*iters\)")
CHECKPOINT_RE = re.compile(r"Checkpoint.*?cg(\d+).*?lb=([0-9.]+)\s*\([^)]*\)\s*gc=([\-0-9.]+)%")


def select_configs(report_path: Path, k_p1: int = 3, k_p2: int = 2) -> List[Dict[str, Any]]:
 """Select baseline + top-k Phase-1 (by scalar_bound) + top-k Phase-2 (by cg1_lb).

 Drops any Phase-1 config whose scalar_iters exceeds the baseline's (these
 are slower-and-better configs whose speed advantage vanishes with deeper
 CG rounds, so they are not robust winners).
 """
 payload = json.loads(Path(report_path).read_text())
 rows = payload['results']

 baseline = next(r for r in rows if r.get('phase') == 'baseline')
 base_iters = baseline.get('scalar_iters') or 99999

 p1 = [r for r in rows if r.get('phase') == 1 and r.get('returncode') == 0]
 p2 = [r for r in rows if r.get('phase') == 2 and r.get('returncode') == 0]

 # Phase-1 winners: higher scalar_bound, iters not worse than baseline
 p1_sorted = [r for r in p1 if (r.get('scalar_iters') or 99999) <= base_iters * 1.05]
 p1_sorted.sort(key=lambda r: -(r.get('scalar_bound') or 0))
 p1_top = p1_sorted[:k_p1]

 # Phase-2 winners by cg1_lb
 p2_sorted = sorted(p2, key=lambda r: -(r.get('cg1_lb') or 0))
 p2_top = p2_sorted[:k_p2]

 configs: List[Dict[str, Any]] = []
 configs.append({
 'label': 'BASELINE',
 'env_overrides': {},
 'cli_overrides': {},
 'source': 'baseline',
 })
 for r in p1_top:
 configs.append({
 'label': r['label'],
 'env_overrides': dict(r.get('env_overrides') or {}),
 'cli_overrides': dict(r.get('cli_overrides') or {}),
 'source': 'phase1',
 'initial_scalar_bound': r.get('scalar_bound'),
 'initial_scalar_iters': r.get('scalar_iters'),
 })
 for r in p2_top:
 lab = r['label']
 if any(c['label'] == lab for c in configs):
 continue # dedupe if same config appeared in both phases
 configs.append({
 'label': lab,
 'env_overrides': dict(r.get('env_overrides') or {}),
 'cli_overrides': dict(r.get('cli_overrides') or {}),
 'source': 'phase2',
 'initial_cg1_lb': r.get('cg1_lb'),
 })
 return configs


def run_one(
 label: str,
 env_overrides: Dict[str, str],
 cli_overrides: Dict[str, Any],
 cg_rounds: int,
 bisect: int,
 scs_eps: str,
 scs_iters: int,
 workdir: Path,
 run_script: str,
 timeout_s: int,
 verbose: bool = True,
) -> Dict[str, Any]:
 defaults = {
 '--d': 16, '--order': 3, '--bw': 15,
 '--scs-iters': scs_iters,
 '--scs-eps': scs_eps,
 '--k-vecs': 3,
 '--cuts-per-round': 100,
 '--rho': '0.1',
 '--atom-frac': '0.5',
 }
 cli = {**defaults, **cli_overrides}
 cli['--cg-rounds'] = cg_rounds
 cli['--bisect'] = bisect
 cli['--gpu'] = None # flag

 env = os.environ.copy()
 env.update({k: str(v) for k, v in env_overrides.items()})

 parts = [sys.executable, '-u', run_script]
 for k, v in cli.items():
 parts.append(k)
 if v is not None:
 parts.append(str(v))

 log_file = workdir / f"transfer_{label.replace('=', '_').replace('.', 'p')}_cg{cg_rounds}_b{bisect}.log"
 log_file.parent.mkdir(parents=True, exist_ok=True)

 if verbose:
 print(f"\n── {label} @ cg={cg_rounds},bisect={bisect} ──────────", flush=True)
 env_str = " ".join(f"{k}={v}" for k, v in env_overrides.items()) or "(no env)"
 cli_str = " ".join(f"{k} {v}" for k, v in cli_overrides.items()) or "(default)"
 print(f" env: {env_str}", flush=True)
 print(f" cli: {cli_str}", flush=True)
 print(f" log: {log_file}", flush=True)

 t0 = time.time()
 try:
 r = subprocess.run(
 parts, env=env, timeout=timeout_s,
 capture_output=True, text=True, cwd=workdir.parent,
 )
 stdout, stderr, rc = r.stdout, r.stderr, r.returncode
 except subprocess.TimeoutExpired as te:
 stdout = te.stdout.decode() if isinstance(te.stdout, bytes) else (te.stdout or "")
 stderr = te.stderr.decode() if isinstance(te.stderr, bytes) else (te.stderr or "")
 rc = -1
 elapsed = time.time() - t0
 log_file.write_text(stdout + ("\n--- STDERR ---\n" + stderr if stderr else ""))

 # Parse scalar
 scalar_bound: Optional[float] = None
 scalar_iters: Optional[int] = None
 m = SCALAR_BOUND_RE.search(stdout)
 if m:
 scalar_bound = float(m.group(1))
 scalar_iters = int(m.group(3))

 # Parse per-round lb (map of round -> lb, gc)
 rounds: Dict[int, Dict[str, float]] = {}
 for cm in CHECKPOINT_RE.finditer(stdout):
 rn = int(cm.group(1))
 rounds[rn] = {'lb': float(cm.group(2)), 'gc_pct': float(cm.group(3))}

 per_round = [
 {'round': k, 'lb': rounds[k]['lb'], 'gc_pct': rounds[k]['gc_pct']}
 for k in sorted(rounds.keys())
 ]
 final_lb = per_round[-1]['lb'] if per_round else scalar_bound
 final_gc = per_round[-1]['gc_pct'] if per_round else None

 result = {
 'label': label,
 'env_overrides': env_overrides,
 'cli_overrides': cli_overrides,
 'cg_rounds': cg_rounds,
 'bisect': bisect,
 'scs_eps': scs_eps,
 'wall_s': round(elapsed, 1),
 'returncode': rc,
 'scalar_bound': scalar_bound,
 'scalar_iters': scalar_iters,
 'per_round': per_round,
 'final_lb': final_lb,
 'final_gc_pct': final_gc,
 'log_file': str(log_file),
 }

 if verbose:
 ok = "" if rc == 0 else ("×" if rc != -1 else "T")
 last_str = (f" R{per_round[-1]['round']}={per_round[-1]['lb']:.4f}"
 f" gc={per_round[-1]['gc_pct']:.1f}%") if per_round else ""
 print(f" [{ok}] scalar={scalar_bound}{last_str} wall={elapsed:.0f}s", flush=True)
 return result


def _write(results: List[Dict[str, Any]], out_path: str, t0: float) -> None:
 Path(out_path).parent.mkdir(parents=True, exist_ok=True)
 Path(out_path).write_text(json.dumps({
 'generated_at': datetime.now().isoformat(),
 'elapsed_s': round(time.time() - t0, 1),
 'n_results': len(results),
 'results': results,
 }, indent=2))


def _ranking_table(results: List[Dict[str, Any]], tier: str) -> None:
 print(f"\n## {tier} — per-round lb (sorted by final_lb)")
 sub = [r for r in results if r.get('tier') == tier]
 if not sub:
 return
 sub.sort(key=lambda r: -(r.get('final_lb') or 0))
 header = f"{'label':<14} | " + " | ".join(f"R{k}".rjust(8) for k in range(11)) + f" | {'wall':>5}s"
 print(header)
 print("-" * len(header))
 for r in sub:
 rd = {p['round']: p['lb'] for p in r.get('per_round', [])}
 cells = []
 for k in range(11):
 if k in rd:
 cells.append(f"{rd[k]:8.4f}")
 else:
 cells.append(" " * 8)
 print(f"{r['label']:<14} | " + " | ".join(cells) + f" | {r.get('wall_s') or 0:>5.0f}")


def main() -> None:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--initial-report', default='data/sweep_report.json')
 ap.add_argument('--out', default='data/transfer_sweep_report.json')
 ap.add_argument('--run-script', default='tests/run_d16_l3.py')
 ap.add_argument('--workdir', default='/tmp/sidon_transfer')
 ap.add_argument('--k-phase1', type=int, default=3)
 ap.add_argument('--k-phase2', type=int, default=2)
 # T-mid
 ap.add_argument('--mid-cg-rounds', type=int, default=5)
 ap.add_argument('--mid-bisect', type=int, default=8)
 ap.add_argument('--mid-timeout', type=int, default=1500) # 25 min
 # T-deep
 ap.add_argument('--deep-cg-rounds', type=int, default=10)
 ap.add_argument('--deep-bisect', type=int, default=10)
 ap.add_argument('--deep-timeout', type=int, default=3000) # 50 min
 ap.add_argument('--deep-top-k', type=int, default=2)
 ap.add_argument('--skip-deep', action='store_true')
 ap.add_argument('--scs-eps', default='1e-6')
 ap.add_argument('--scs-iters', type=int, default=20000)
 args = ap.parse_args()

 workdir = Path(args.workdir)
 workdir.mkdir(parents=True, exist_ok=True)
 repo_root = Path(__file__).resolve().parent.parent
 run_script = (str(repo_root / args.run_script)
 if not Path(args.run_script).is_absolute()
 else args.run_script)

 configs = select_configs(
 Path(args.initial_report),
 k_p1=args.k_phase1, k_p2=args.k_phase2,
 )

 print(f"Transfer sweep started at {datetime.now().isoformat()}")
 print(f" Repo: {repo_root}")
 print(f" Run script: {run_script}")
 print(f" Workdir: {workdir}")
 print(f" Out: {args.out}")
 print(f" Initial report: {args.initial_report}")
 print(f" Selected configs:")
 for c in configs:
 print(f" - {c['label']:<14} (source={c['source']}) "
 f"env={c['env_overrides']} cli={c['cli_overrides']}")

 t_sweep = time.time()
 results: List[Dict[str, Any]] = []

 # T-mid: all configs
 print(f"\n=== T-mid (cg={args.mid_cg_rounds}, bisect={args.mid_bisect}) ===")
 for c in configs:
 r = run_one(
 c['label'], c['env_overrides'], c['cli_overrides'],
 cg_rounds=args.mid_cg_rounds, bisect=args.mid_bisect,
 scs_eps=args.scs_eps, scs_iters=args.scs_iters,
 workdir=workdir, run_script=run_script,
 timeout_s=args.mid_timeout,
 )
 r['tier'] = 'T-mid'
 r['source_config'] = c
 results.append(r)
 _write(results, args.out, t_sweep)

 # T-deep: baseline + top k from T-mid (by final_lb)
 if not args.skip_deep:
 print(f"\n=== T-deep (cg={args.deep_cg_rounds}, bisect={args.deep_bisect}) ===")
 mid = [r for r in results if r.get('tier') == 'T-mid']
 mid.sort(key=lambda r: -(r.get('final_lb') or 0))
 deep_labels = {'BASELINE'}
 for r in mid:
 if len(deep_labels) >= args.deep_top_k + 1:
 break
 deep_labels.add(r['label'])

 deep_configs = [c for c in configs if c['label'] in deep_labels]
 for c in deep_configs:
 r = run_one(
 c['label'], c['env_overrides'], c['cli_overrides'],
 cg_rounds=args.deep_cg_rounds, bisect=args.deep_bisect,
 scs_eps=args.scs_eps, scs_iters=args.scs_iters,
 workdir=workdir, run_script=run_script,
 timeout_s=args.deep_timeout,
 )
 r['tier'] = 'T-deep'
 r['source_config'] = c
 results.append(r)
 _write(results, args.out, t_sweep)

 total = time.time() - t_sweep
 print(f"\n╔══ TRANSFER SWEEP COMPLETE ══╗")
 print(f" Total time: {total:.0f}s = {total/60:.1f} min")
 print(f" Configs run: {len(results)}")
 print(f" Report: {args.out}")
 _ranking_table(results, 'T-mid')
 _ranking_table(results, 'T-deep')


if __name__ == '__main__':
 main()
