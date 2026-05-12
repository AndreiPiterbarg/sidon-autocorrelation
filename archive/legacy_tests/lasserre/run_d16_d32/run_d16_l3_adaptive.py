#!/usr/bin/env python
"""Adaptive per-round lb search with config-ladder fallback + Z/2 symmetry.

Goal
----
Push the Lasserre L3 lower bound as high as possible within a wall-clock
budget by treating EACH CG round as an independent optimisation
sub-problem and trying multiple configurations per round until one
improves lb meaningfully (or the ladder is exhausted).

Design
------
 state
 ┌──────────────────────────────┐
 │ ckpt_cg<N-1>.{json,npz} ─────┼── starting state for round N
 └──────────────────────────────┘

 round N:
 before_lb ← lb at ckpt_cg<N-1>
 for tier in ORDERED_LADDER:
 delete any stale ckpt_cg<N>
 run run_d16_l3.py with:
 --resume --cg-rounds N (resume to ckpt<N-1>, run ONE more)
 + tier-specific CLI (cuts_per_round, atom_frac, bisect, ...)
 + env SIDON_Z2_SYMMETRY=1 (if --z2 active)
 after_lb ← lb at ckpt_cg<N>
 if after_lb > before_lb + IMPROVEMENT_EPS:
 commit (keep the ckpt) and break
 else:
 DELETE ckpt_cg<N> (stale attempt)
 if all tiers stalled:
 # Record a placeholder ckpt so --resume advances to round N+1.
 # It carries the previous lb to preserve monotonicity.
 create placeholder ckpt_cg<N> with best_lb = before_lb

Soundness
---------
Each tier solves the SAME relaxation (one more CG round added to the
existing checkpoint), so every lb produced is a valid lower bound on
val(d). Committing the maximum across tiers preserves the invariant
lb_cg<N> ≥ lb_cg<N-1>, and lb_cg<N> ≤ val(d) always.

Z/2 symmetry is soundness-preserving too: see lasserre/z2_symmetry.py
module docstring for the proof sketch (the Sidon problem admits a
σ-symmetric extremiser, so val_σ(d) = val(d), and lb_σ ≤ val_σ(d)
⇒ lb_σ ≤ val(d)).

Usage
-----
 python tests/run_d16_l3_adaptive.py \
 --d 16 --order 3 --bw 15 \
 --max-rounds 30 \
 --data-dir data \
 --z2 # enable Z/2 symmetry
 [--start-round N] # resume from round N (default 1)

Outputs
-------
 data/ckpt_d16_o3_bw15_scs_cg<N>.{json,npz} — official round ckpts
 data/adaptive_<tag>.log — human-readable log
 data/adaptive_<tag>_tier_attempts.jsonl — full per-tier attempt log
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Config ladder
# -----------------------------------------------------------------------------

@dataclass
class Tier:
 """One configuration to try for a single CG round."""
 name: str
 cuts_per_round: int
 atom_frac: float
 bisect: int
 eps_mult: float = 1.0 # multiplier on the base scs_eps
 iters_mult: float = 1.0 # multiplier on base scs_iters
 notes: str = ''

 def cli_flags(self, base_eps: float, base_iters: int) -> List[str]:
 eps = base_eps * self.eps_mult
 iters = int(base_iters * self.iters_mult)
 return [
 '--cuts-per-round', str(self.cuts_per_round),
 '--atom-frac', str(self.atom_frac),
 '--bisect', str(self.bisect),
 '--scs-eps', f'{eps:.2e}',
 '--scs-iters', str(iters),
 ]


# Ordered cheapest → most aggressive. The per-round loop commits the first
# tier that beats the previous lb by IMPROVEMENT_EPS. Empirically we expect
# tier 1 or 2 to succeed most rounds; tiers 3-5 only trigger on plateau rounds.
DEFAULT_LADDER: List[Tier] = [
 Tier('default', cuts_per_round=50, atom_frac=0.5, bisect=15,
 eps_mult=1.0, iters_mult=1.0, notes='baseline from sweep'),
 Tier('double-cuts', cuts_per_round=100, atom_frac=0.5, bisect=15,
 eps_mult=1.0, iters_mult=1.0, notes='more constraint pressure'),
 Tier('atom-heavy', cuts_per_round=150, atom_frac=1.0, bisect=15,
 eps_mult=1.0, iters_mult=1.0, notes='atom-driven cut selection'),
 Tier('aggressive', cuts_per_round=200, atom_frac=0.0, bisect=18,
 eps_mult=0.1, iters_mult=2.0,
 notes='eig-only cuts, tighter eps, more iters, deeper bisect'),
 Tier('kitchen-sink', cuts_per_round=300, atom_frac=0.5, bisect=20,
 eps_mult=0.1, iters_mult=3.0,
 notes='max cuts, tight eps, max bisect; last resort'),
]


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

def ckpt_stem(data_dir: Path, tag: str, cg: int) -> Path:
 return data_dir / f'ckpt_{tag}_cg{cg}'


def ckpt_exists(data_dir: Path, tag: str, cg: int) -> bool:
 stem = ckpt_stem(data_dir, tag, cg)
 return stem.with_suffix('.json').exists()


def load_ckpt_lb(data_dir: Path, tag: str, cg: int) -> Optional[float]:
 json_path = ckpt_stem(data_dir, tag, cg).with_suffix('.json')
 if not json_path.exists():
 return None
 try:
 meta = json.loads(json_path.read_text())
 return float(meta.get('best_lb', meta.get('scalar_lb', 0.0)))
 except Exception:
 return None


def delete_ckpt(data_dir: Path, tag: str, cg: int) -> None:
 stem = ckpt_stem(data_dir, tag, cg)
 for suf in ('.json', '.npz', '.json.tmp', '.npz.tmp'):
 p = stem.with_suffix(suf)
 if p.exists():
 p.unlink()


def backup_ckpt(data_dir: Path, tag: str, cg: int,
 suffix: str = '.backup') -> None:
 """Snapshot ckpt_cg<N>.{json,npz} to a backup location."""
 stem = ckpt_stem(data_dir, tag, cg)
 for suf in ('.json', '.npz'):
 p = stem.with_suffix(suf)
 if p.exists():
 shutil.copy2(p, str(p) + suffix)


def restore_ckpt(data_dir: Path, tag: str, cg: int,
 suffix: str = '.backup') -> None:
 """Restore ckpt from its .backup sibling. Deletes the backup after."""
 stem = ckpt_stem(data_dir, tag, cg)
 for suf in ('.json', '.npz'):
 p = stem.with_suffix(suf)
 bp = Path(str(p) + suffix)
 if bp.exists():
 shutil.move(str(bp), p)


def save_placeholder_ckpt(data_dir: Path, tag: str, cg: int,
 best_lb: float, source_cg: int,
 d: int, order: int, bandwidth: int) -> None:
 """Write a placeholder ckpt for round cg when every tier stalled.

 The placeholder carries the previous lb (unchanged) and a marker
 'stalled=True' so downstream tooling can tell it apart from a real
 round. It shares the last feasible warm-start arrays from source_cg,
 so the next resume step is no less warm than the stalled round was.
 """
 src_json = ckpt_stem(data_dir, tag, source_cg).with_suffix('.json')
 src_npz = ckpt_stem(data_dir, tag, source_cg).with_suffix('.npz')
 if not (src_json.exists() and src_npz.exists()):
 raise RuntimeError(
 f"Cannot create placeholder ckpt at cg{cg}: source cg{source_cg} "
 f"missing ({src_json}).")

 meta = json.loads(src_json.read_text())
 meta = dict(meta)
 meta['cg_round'] = int(cg)
 meta['best_lb'] = float(best_lb)
 meta['stalled'] = True
 meta['stalled_from'] = int(source_cg)
 meta['timestamp'] = datetime.now().isoformat()
 meta['d'] = int(d)
 meta['order'] = int(order)
 meta['bandwidth'] = int(bandwidth)

 dst_json = ckpt_stem(data_dir, tag, cg).with_suffix('.json')
 dst_npz = ckpt_stem(data_dir, tag, cg).with_suffix('.npz')
 # Copy the warm-start arrays; write JSON last so the stem appears
 # atomically from the loader's perspective.
 shutil.copy2(src_npz, dst_npz)
 dst_json.write_text(json.dumps(meta, indent=2, default=str))


# -----------------------------------------------------------------------------
# Subprocess runner
# -----------------------------------------------------------------------------

LB_RE = re.compile(r"lb=([0-9.]+)\s*\([^)]*\)\s*gc=([\-0-9.]+)%")
FINAL_RE = re.compile(r"FINAL:.*?lb = ([0-9.]+)", re.DOTALL)


def run_one_round_subprocess(
 runner: Path,
 d: int, order: int, bandwidth: int,
 target_round: int,
 tier: Tier,
 base_eps: float, base_iters: int,
 base_rho: float,
 data_dir: Path,
 log_path: Path,
 enable_z2: bool,
 timeout_s: int,
 verbose: bool = True,
) -> Dict[str, Any]:
 """Invoke run_d16_l3.py with --resume --cg-rounds target_round.

 Because the runner loads the highest-numbered ckpt and advances from
 there, the subprocess will perform EXACTLY ONE new round (the one
 between ckpt_cg<target_round-1> and ckpt_cg<target_round>) provided
 that we've deleted any stale cg<target_round> ckpt beforehand.

 Returns a dict with:
 returncode, wall_s, lb (last parsed), log_tail, stdout_path
 """
 cli = [
 sys.executable, '-u', str(runner),
 '--d', str(d), '--order', str(order), '--bw', str(bandwidth),
 '--cg-rounds', str(target_round),
 '--rho', str(base_rho),
 '--gpu',
 '--resume', '--data-dir', str(data_dir),
 ]
 cli.extend(tier.cli_flags(base_eps, base_iters))

 env = os.environ.copy()
 if enable_z2:
 env['SIDON_Z2_SYMMETRY'] = '1'

 if verbose:
 print(f" tier={tier.name!r}: {' '.join(cli[5:])}", flush=True)

 t0 = time.time()
 try:
 with open(log_path, 'a') as fh:
 fh.write(f"\n=== tier={tier.name} target_round={target_round} "
 f"started={datetime.now().isoformat()} ===\n")
 fh.flush()
 r = subprocess.run(
 cli, env=env, stdout=fh, stderr=subprocess.STDOUT,
 timeout=timeout_s,
 )
 rc = r.returncode
 except subprocess.TimeoutExpired:
 rc = -1

 wall = time.time() - t0

 # Parse tail of the log for the latest lb line.
 tail = ''
 final_lb: Optional[float] = None
 try:
 with open(log_path, 'r') as fh:
 contents = fh.read()
 # Look at the section we just appended.
 split_idx = contents.rfind(
 f"=== tier={tier.name} target_round={target_round}")
 section = contents[split_idx:] if split_idx >= 0 else contents
 tail = section[-2000:]
 # First try FINAL (run completed normally).
 fm = FINAL_RE.search(section)
 if fm:
 final_lb = float(fm.group(1))
 else:
 # Fall back to the last Checkpoint line.
 lb_hits = list(LB_RE.finditer(section))
 if lb_hits:
 final_lb = float(lb_hits[-1].group(1))
 except Exception:
 pass

 return {
 'returncode': rc,
 'wall_s': round(wall, 1),
 'lb_parsed_from_log': final_lb,
 'log_tail': tail,
 }


# -----------------------------------------------------------------------------
# Main per-round loop
# -----------------------------------------------------------------------------

def main() -> None:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--d', type=int, default=16)
 ap.add_argument('--order', type=int, default=3)
 ap.add_argument('--bw', type=int, default=15)
 ap.add_argument('--max-rounds', type=int, default=30)
 ap.add_argument('--start-round', type=int, default=0,
 help='First round to (re)compute. 0 means start from '
 'scalar bound (round 0 = no CG, just minimise t).')
 ap.add_argument('--data-dir', type=str, default='data')
 ap.add_argument('--rho', type=float, default=0.1)
 ap.add_argument('--base-eps', type=float, default=1e-6)
 ap.add_argument('--base-iters', type=int, default=30000)
 ap.add_argument('--improvement-eps', type=float, default=1e-4,
 help='Minimum lb improvement to consider a tier "successful".')
 ap.add_argument('--tier-timeout', type=int, default=3600,
 help='Per-tier subprocess timeout in seconds (default 1h).')
 ap.add_argument('--z2', action='store_true',
 help='Enable Z/2 time-reversal symmetry injection.')
 ap.add_argument('--tag', type=str, default=None,
 help='Checkpoint tag suffix; default "_scs".')
 ap.add_argument('--stop-on-plateau', type=int, default=3,
 help='Stop if this many consecutive rounds stall all tiers.')
 ap.add_argument('--runner', type=str,
 default='tests/run_d16_l3.py',
 help='Path to the single-round runner.')
 args = ap.parse_args()

 repo_root = Path(__file__).resolve().parent.parent
 runner = (repo_root / args.runner
 if not Path(args.runner).is_absolute()
 else Path(args.runner))
 data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() \
 else repo_root / args.data_dir
 data_dir.mkdir(parents=True, exist_ok=True)

 tag_stem = args.tag or '_scs'
 tag = f'd{args.d}_o{args.order}_bw{args.bw}{tag_stem}'
 log_path = data_dir / f'adaptive_{tag}.log'
 tier_log_path = data_dir / f'adaptive_{tag}_tier_attempts.jsonl'

 # Announce
 banner = (
 "=" * 72
 + f"\n adaptive per-round lb search — tag={tag}\n"
 + f" d={args.d} order={args.order} bw={args.bw}\n"
 + f" max_rounds={args.max_rounds} start_round={args.start_round}\n"
 + f" Z/2 symmetry={'ON' if args.z2 else 'off'}\n"
 + f" improvement_eps={args.improvement_eps:.1e}\n"
 + f" data_dir={data_dir}\n"
 + f" started={datetime.now().isoformat()}\n"
 + "=" * 72)
 print(banner, flush=True)
 with open(log_path, 'a') as fh:
 fh.write(banner + '\n')

 # Determine the best completed round (if any).
 last_completed = args.start_round - 1
 for cg in range(0, args.max_rounds + 1):
 if ckpt_exists(data_dir, tag, cg):
 last_completed = max(last_completed, cg)
 prev_lb = load_ckpt_lb(data_dir, tag, last_completed) if last_completed >= 0 \
 else None

 consecutive_plateau = 0
 t_wall = time.time()

 for target_round in range(max(1, last_completed + 1),
 args.max_rounds + 1):

 before_lb = prev_lb
 if before_lb is None:
 # No prior ckpt: invoke with NO --resume so a fresh Round-0
 # scalar solve happens and ckpt_cg1 is produced.
 # We still go through the tier loop to try multiple configs,
 # but the subprocess internally does cg_rounds=target_round
 # rounds from scratch. In practice callers should pass
 # --start-round 1 after running a scalar-bound seed separately.
 print(f"[round {target_round}] no prior ckpt — starting fresh.",
 flush=True)

 print(f"\n── round {target_round} (prev lb = "
 f"{before_lb if before_lb is None else f'{before_lb:.6f}'}) ──",
 flush=True)

 committed = False
 tier_results: List[Dict[str, Any]] = []

 for tier in DEFAULT_LADDER:
 # Clean any stale ckpt at this round before attempting.
 if ckpt_exists(data_dir, tag, target_round):
 delete_ckpt(data_dir, tag, target_round)

 result = run_one_round_subprocess(
 runner=runner,
 d=args.d, order=args.order, bandwidth=args.bw,
 target_round=target_round,
 tier=tier,
 base_eps=args.base_eps, base_iters=args.base_iters,
 base_rho=args.rho,
 data_dir=data_dir,
 log_path=log_path,
 enable_z2=args.z2,
 timeout_s=args.tier_timeout,
 )
 attempt = {
 'round': target_round,
 'tier': asdict(tier),
 'before_lb': before_lb,
 'result': result,
 'timestamp': datetime.now().isoformat(),
 }

 after_lb = load_ckpt_lb(data_dir, tag, target_round)
 attempt['after_lb'] = after_lb
 improved = (
 after_lb is not None
 and (before_lb is None
 or after_lb > before_lb + args.improvement_eps))
 attempt['improved'] = bool(improved)
 tier_results.append(attempt)

 with open(tier_log_path, 'a') as fh:
 fh.write(json.dumps(attempt, default=str) + '\n')

 status = ' committed' if improved else '× stalled'
 print(f" → after_lb={after_lb if after_lb is None else f'{after_lb:.6f}'} "
 f"{status} wall={result['wall_s']}s",
 flush=True)

 if improved:
 committed = True
 prev_lb = after_lb
 break
 else:
 # Delete the failed attempt so the next tier starts from
 # ckpt_cg<target_round-1> (via --resume).
 if ckpt_exists(data_dir, tag, target_round):
 delete_ckpt(data_dir, tag, target_round)

 if committed:
 consecutive_plateau = 0
 else:
 consecutive_plateau += 1
 # All tiers stalled — save a placeholder so downstream
 # rounds can --resume, carrying prev_lb forward untouched.
 if before_lb is not None and last_completed >= 0:
 save_placeholder_ckpt(
 data_dir, tag, target_round,
 best_lb=before_lb, source_cg=last_completed,
 d=args.d, order=args.order, bandwidth=args.bw,
 )
 print(f" all tiers stalled; placeholder ckpt saved "
 f"(lb held at {before_lb:.6f}).",
 flush=True)

 if consecutive_plateau >= args.stop_on_plateau:
 print(f"\n⏹ stopping: {consecutive_plateau} consecutive "
 f"rounds stalled all tiers — structural plateau.",
 flush=True)
 break

 last_completed = target_round

 total = time.time() - t_wall
 print(f"\n=== DONE total={total:.0f}s = {total/60:.1f}min "
 f"final_lb={prev_lb} rounds_completed={last_completed} ===",
 flush=True)


if __name__ == '__main__':
 main()
