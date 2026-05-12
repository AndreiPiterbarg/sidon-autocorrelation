"""Publication-grade orchestrator: BnB cascade → instant dump → K-ladder SDP.

PIPELINE (all phases write to a single timestamped run directory)
-----------------------------------------------------------------
1. PHASE 1 — `cert_pipeline.bnb_phase.run_one_bnb_phase`
   Spawn `interval_bnb.parallel.parallel_branch_and_bound` as subprocess
   with `INTERVAL_BNB_INSTANT_DUMP=1`. Track `[par]` log lines for
   in_flight history. Stuck-detector triggers SIGINT; workers DUMP
   their local_stack to disk and `os._exit(0)` immediately (no further
   cascade processing → no boxes lost). Master drains shared mp.Queue
   to `master_queue.npz`.

2. SURVIVOR LOAD — Read all dump files (worker_w*.npz + master_queue.npz).
   - Compute LP value per box (cheap `bound_epigraph_lp_float`)
   - Skip boxes already LP-cert (LP_val ≥ target) — these don't need SDP
   - Compute canonical hash + volume per surviving box
   - Build SurvivorBox list, persist to `survivors_initial.npz`

3. PHASE 2 — `cert_pipeline.k_ladder.run_k_ladder`
   For each K in [0, 16, 32, 64, 128]:
     a. CALIBRATE on 2 boxes (serial), measure peak RSS
     b. AUTO-TUNE n_parallel from RAM headroom + cores
     c. SWEEP all remaining survivors in parallel pool
     d. FILTER: cert vs. survivor for next K stage
   Stops at K=128. Reports final_survivors.

4. AUDIT REPORT — Print final summary with cascade vs SDP cert counts,
   any remaining survivors that need split-and-recurse.

OUTPUT DIRECTORY LAYOUT
-----------------------
runs/{tag}/
  config.json                # full config snapshot
  bnb_phase/
    bnb.log
    bnb_metrics.jsonl
    bnb_summary.json
    dumps/
      bnb_dump_w*.npz
      bnb_dump_master_queue.npz
  survivors_initial.npz      # post-LP-filter survivors
  survivors_initial.json     # human-readable summary
  k_ladder/
    stage_K0/
    stage_K16/
    ...
    stage_K128/
    final_summary.json
    final_survivors.npz
  pipeline_summary.json       # full rollup (initial → cascade → SDP per K)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from cert_pipeline.bnb_phase import (
    BnBPhaseConfig, BnBPhaseResult, run_one_bnb_phase,
)
from cert_pipeline.k_ladder import (
    DEFAULT_K_LADDER, LadderResult, SurvivorBox,
    run_k_ladder, survivors_to_npz, survivors_from_npz,
)
from cert_pipeline.box_journal import canonical_box_hash


# ==========================================================================
# Helpers
# ==========================================================================

def _utc_tag() -> str:
    return datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')


def _git_state(repo_root: Path) -> Dict[str, str]:
    """Capture git rev + diff for reproducibility."""
    import subprocess
    out: Dict[str, str] = {}
    try:
        out['git_rev'] = subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        out['git_rev'] = 'unknown'
    try:
        diff = subprocess.check_output(
            ['git', '-C', str(repo_root), 'diff', '--stat', 'HEAD'],
            stderr=subprocess.DEVNULL, timeout=10).decode()
        out['git_diff_stat'] = diff
    except Exception:
        out['git_diff_stat'] = ''
    return out


def _load_dump_boxes(dump_files: Sequence[str]) -> List[Dict[str, Any]]:
    """Load (lo, hi, depth, src) tuples from BnB dump npz files."""
    out: List[Dict[str, Any]] = []
    for fp in dump_files:
        try:
            data = np.load(str(fp), allow_pickle=True)
            keys = list(data.files)
            if 'lo' not in keys or 'hi' not in keys:
                continue
            lo_arr = data['lo']
            hi_arr = data['hi']
            deps = data['depths'] if 'depths' in keys else None
            n = lo_arr.shape[0] if lo_arr.ndim == 2 else 1
            if lo_arr.ndim == 1:
                lo_arr = lo_arr[None, :]
                hi_arr = hi_arr[None, :]
            for i in range(n):
                out.append({
                    'lo': lo_arr[i].astype(np.float64),
                    'hi': hi_arr[i].astype(np.float64),
                    'depth': int(deps[i]) if deps is not None else 0,
                    'src': f'{Path(fp).name}#{i}',
                })
        except Exception as e:
            print(f"  load fail {fp}: {type(e).__name__}: {e}", flush=True)
    return out


def _filter_lp_failing_survivors(
    raw_boxes: List[Dict[str, Any]], d: int, target: float,
    lp_too_far_margin: float = 0.10,
) -> Tuple[List[SurvivorBox], List[SurvivorBox]]:
    """Filter raw dump boxes → (sdp_eligible, lp_too_far_skipped).

    A box B has three possible classifications based on its epigraph LP value:
      - LP_value >= target               → cascade should have caught it; cert
      - LP_value in [target - margin, target) → "stress zone" SDP-eligible
      - LP_value < target - margin       → "LP-too-far": skipping the SDP because
        no K is likely to cert; box must be SPLIT (re-fed to BnB) instead.

    The LP gives a LOWER bound on val_B. McCormick relaxation looseness
    is O(hw²) for boxes of half-width hw. If LP is far below target, it
    means either:
      (a) val_B truly is well below target (mathematically impossible
          for d=22 t=1.2805 since val(22)≈1.309 > target — so val_B
          must be ≥ 1.309 always), OR
      (b) the box is too LARGE and its McCormick LP is loose. SDP at
          finite K won't fix this — only SPLITTING the box can.

    So: LP-too-far boxes need split-and-recurse, not bigger K.

    Returns (sdp_eligible, lp_too_far). Both are SurvivorBox lists with
    full int endpoints + LP value preserved for downstream re-injection.
    """
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_epigraph import bound_epigraph_lp_float
    from interval_bnb.box import SCALE as _SCALE

    windows = build_windows(d)
    sdp_eligible: List[SurvivorBox] = []
    lp_too_far: List[SurvivorBox] = []
    n_lp_cert = 0
    n_empty = 0
    n_lp_too_far = 0
    n_lp_failing = 0
    threshold_too_far = target - lp_too_far_margin
    for box in raw_boxes:
        lo = box['lo']
        hi = box['hi']
        if lo.shape[0] != d or lo.sum() > 1.0 or hi.sum() < 1.0:
            n_empty += 1
            continue
        try:
            lp = float(bound_epigraph_lp_float(lo, hi, windows, d))
        except Exception:
            lp = float('nan')
        if np.isfinite(lp) and lp >= target:
            n_lp_cert += 1
            continue
        lo_int = [int(round(x * _SCALE)) for x in lo]
        hi_int = [int(round(x * _SCALE)) for x in hi]
        bhash = canonical_box_hash(lo_int, hi_int)
        vol = float(np.prod(hi - lo))
        sb = SurvivorBox(
            hash=bhash, lo_int=lo_int, hi_int=hi_int,
            depth=int(box.get('depth', 0)), volume=vol,
            lp_val=lp if np.isfinite(lp) else None,
            src=box.get('src', ''),
        )
        if np.isfinite(lp) and lp < threshold_too_far:
            n_lp_too_far += 1
            lp_too_far.append(sb)
            continue
        # In the SDP-eligible "stress zone" target-margin <= LP < target.
        n_lp_failing += 1
        sdp_eligible.append(sb)
    print(f"  [filter] raw={len(raw_boxes)}  "
          f"empty/infeasible={n_empty}  "
          f"lp_cert={n_lp_cert}  "
          f"lp_too_far(LP<{threshold_too_far:.4f})={n_lp_too_far}  "
          f"sdp_eligible({threshold_too_far:.4f}<=LP<{target})={n_lp_failing}",
          flush=True)
    return sdp_eligible, lp_too_far


def _dedupe_survivors_by_hash(survivors: List[SurvivorBox]) -> List[SurvivorBox]:
    """Keep one entry per canonical hash."""
    seen: Dict[str, SurvivorBox] = {}
    for s in survivors:
        if s.hash not in seen:
            seen[s.hash] = s
    return list(seen.values())


# ==========================================================================
# Top-level orchestrator
# ==========================================================================

@dataclass
class PipelineConfig:
    """Top-level configuration for ONE end-to-end run."""
    d: int
    target_str: str                 # e.g. "1.2805"
    output_root: str = 'runs'
    tag: str = ''                   # auto-generated if empty
    bnb: Optional[BnBPhaseConfig] = None
    k_ladder: List[int] = field(default_factory=lambda: list(DEFAULT_K_LADDER))
    sdp_time_limit_s: float = 600.0
    # LP-too-far cutoff: boxes with LP < (target - margin) are SKIPPED
    # from the K-ladder (they cannot be certified at any K because the
    # box is too LARGE for the McCormick LP relaxation to give a tight
    # lower bound; only SPLITTING the box can help). They are saved to
    # survivors_lp_too_far.npz for split-and-recurse in a future iter.
    lp_too_far_margin: float = 0.10


def run_pipeline(cfg: PipelineConfig, repo_root: Path) -> Dict[str, Any]:
    """Run the publication-grade pipeline end-to-end."""
    if not cfg.tag:
        cfg.tag = f'd{cfg.d}_t{cfg.target_str.replace(".", "p")}_{_utc_tag()}'
    run_dir = Path(cfg.output_root) / cfg.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Snapshot config + git state ----
    if cfg.bnb is None:
        cfg.bnb = BnBPhaseConfig(d=cfg.d, target_str=cfg.target_str)
    config_dict = {
        'd': cfg.d,
        'target': cfg.target_str,
        'tag': cfg.tag,
        'k_ladder': cfg.k_ladder,
        'sdp_time_limit_s': cfg.sdp_time_limit_s,
        'bnb': asdict(cfg.bnb),
        'machine': {
            'cores': os.cpu_count(),
            'platform': sys.platform,
        },
        'utc_start': datetime.datetime.utcnow().isoformat() + 'Z',
        'python': sys.version,
    }
    config_dict.update({'git': _git_state(repo_root)})
    (run_dir / 'config.json').write_text(json.dumps(config_dict, indent=2,
                                                     default=str))
    print(f"\n{'#'*70}\n# Pipeline run: {cfg.tag}\n# Output: {run_dir}\n{'#'*70}",
          flush=True)

    pipeline_t0 = time.time()

    # ---- PHASE 1: BnB cascade with instant dump ----
    bnb_dir = run_dir / 'bnb_phase'
    bnb_result = run_one_bnb_phase(cfg.bnb, repo_root, bnb_dir)
    print(f"\n[phase1] BnB done. trigger={bnb_result.trigger_reason} "
          f"dumped_boxes={bnb_result.n_dumped_total}", flush=True)

    # ---- SURVIVOR LOAD + LP FILTER ----
    print("\n[phase1.5] loading & filtering survivors...", flush=True)
    raw = _load_dump_boxes(bnb_result.dump_files)
    target_f = float(cfg.target_str)
    survivors, lp_too_far = _filter_lp_failing_survivors(
        raw, cfg.d, target_f, lp_too_far_margin=cfg.lp_too_far_margin)
    # Deduplicate (different worker dumps may emit the same box).
    survivors = _dedupe_survivors_by_hash(survivors)
    lp_too_far = _dedupe_survivors_by_hash(lp_too_far)
    survivors_to_npz(survivors, str(run_dir / 'survivors_initial.npz'))
    survivors_to_npz(lp_too_far, str(run_dir / 'survivors_lp_too_far.npz'))
    survivor_meta = {
        'n_raw': len(raw),
        'n_unique_sdp_eligible': len(survivors),
        'n_unique_lp_too_far': len(lp_too_far),
        'lp_too_far_margin': cfg.lp_too_far_margin,
        'lp_value_summary_sdp_eligible': {
            'min': float(min((s.lp_val for s in survivors
                              if s.lp_val is not None), default=float('nan'))),
            'max': float(max((s.lp_val for s in survivors
                              if s.lp_val is not None), default=float('nan'))),
            'count': sum(1 for s in survivors if s.lp_val is not None),
        },
        'lp_value_summary_lp_too_far': {
            'min': float(min((s.lp_val for s in lp_too_far
                              if s.lp_val is not None), default=float('nan'))),
            'max': float(max((s.lp_val for s in lp_too_far
                              if s.lp_val is not None), default=float('nan'))),
            'count': sum(1 for s in lp_too_far if s.lp_val is not None),
        },
        'depth_summary': {
            'sdp_min': min((s.depth for s in survivors), default=0),
            'sdp_max': max((s.depth for s in survivors), default=0),
            'too_far_min': min((s.depth for s in lp_too_far), default=0),
            'too_far_max': max((s.depth for s in lp_too_far), default=0),
        },
    }
    (run_dir / 'survivors_initial.json').write_text(json.dumps(
        survivor_meta, indent=2, default=str))
    print(f"  [filter] {len(survivors)} SDP-eligible survivors → K-ladder", flush=True)
    print(f"  [filter] {len(lp_too_far)} LP-too-far → "
          f"survivors_lp_too_far.npz (need split + re-iter)", flush=True)

    if not survivors and not lp_too_far:
        print("\n[done] No survivors at all — cascade closed everything.",
              flush=True)
        summary = {
            'config': config_dict,
            'phase1': asdict(bnb_result),
            'phase2': None,
            'final_survivors_count': 0,
            'verdict': 'CERT_VIA_CASCADE_ALONE',
            'total_wall_s': time.time() - pipeline_t0,
        }
        (run_dir / 'pipeline_summary.json').write_text(
            json.dumps(summary, indent=2, default=str))
        return summary

    if not survivors:
        print(f"\n[done] {len(lp_too_far)} LP-too-far survivors but NONE "
              f"in K-ladder zone — write report and exit (need re-BnB).",
              flush=True)
        summary = {
            'config': config_dict,
            'phase1': asdict(bnb_result),
            'phase2': None,
            'lp_too_far_count': len(lp_too_far),
            'verdict': 'INCOMPLETE_NEEDS_REITERATION',
            'total_wall_s': time.time() - pipeline_t0,
        }
        (run_dir / 'pipeline_summary.json').write_text(
            json.dumps(summary, indent=2, default=str))
        return summary

    # ---- PHASE 2: K-ladder over survivors ----
    ladder_dir = run_dir / 'k_ladder'
    ladder_result = run_k_ladder(
        cfg.d, target_f, survivors, ladder_dir, repo_root,
        k_ladder=cfg.k_ladder, time_limit_s=cfg.sdp_time_limit_s,
    )

    # ---- ROLLUP ----
    n_final = len(ladder_result.final_survivors)
    if n_final == 0:
        verdict = 'CERTIFIED_FULL_PIPELINE'
    else:
        verdict = 'INCOMPLETE_FINAL_SURVIVORS'
    summary: Dict[str, Any] = {
        'config': config_dict,
        'phase1_bnb': {
            'trigger_reason': bnb_result.trigger_reason,
            'elapsed_s': bnb_result.elapsed_s,
            'n_dumped_total': bnb_result.n_dumped_total,
            'cert_count_at_trigger': bnb_result.cert_count_at_trigger,
        },
        'phase1_5_survivor_filter': survivor_meta,
        'phase2_k_ladder': {
            'K_ladder': ladder_result.K_ladder,
            'total_input': ladder_result.total_input,
            'total_cert': ladder_result.total_cert,
            'final_survivor_count': n_final,
            'wall_s': ladder_result.total_wall_s,
            'stages': [
                {
                    'K': s.K,
                    'n_input': s.n_input,
                    'n_cert': s.n_cert,
                    'n_fail': s.n_fail,
                    'n_exception': s.n_exception,
                    'max_rss_gb_observed': s.max_rss_gb_observed,
                    'n_parallel': s.n_parallel,
                    'threads_per_proc': s.threads_per_proc,
                    'sweep_wall_s': s.sweep_wall_s,
                    'avg_solve_s': s.avg_solve_s,
                    'available_ram_gb_at_start': s.available_ram_gb_at_start,
                }
                for s in ladder_result.stages
            ],
        },
        'final_survivors_count': n_final,
        'verdict': verdict,
        'total_wall_s': time.time() - pipeline_t0,
        'utc_end': datetime.datetime.utcnow().isoformat() + 'Z',
    }
    (run_dir / 'pipeline_summary.json').write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"\n{'#'*70}", flush=True)
    print(f"# PIPELINE END: {verdict}", flush=True)
    print(f"#   total wall: {summary['total_wall_s']:.0f}s "
          f"({summary['total_wall_s']/60:.1f} min)", flush=True)
    print(f"#   final survivors (after K=128): {n_final}", flush=True)
    print(f"#   output: {run_dir}", flush=True)
    print(f"{'#'*70}", flush=True)
    return summary


# ==========================================================================
# CLI
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Publication-grade BnB+SDP pipeline for val(d) ≥ target.')
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--target', type=str, required=True,
                    help='e.g. "1.2805"')
    ap.add_argument('--output-root', type=str, default='runs')
    ap.add_argument('--tag', type=str, default='',
                    help='auto-generated if empty')
    ap.add_argument('--bnb-workers', type=int, default=16)
    ap.add_argument('--bnb-init-split-depth', type=int, default=22)
    ap.add_argument('--bnb-warmup-s', type=int, default=300)
    ap.add_argument('--bnb-stuck-window-s', type=int, default=300)
    ap.add_argument('--bnb-time-budget-s', type=int, default=7200)
    ap.add_argument('--k-ladder', type=str, default='0,16,32,64,128',
                    help='comma-separated K values')
    ap.add_argument('--sdp-time-limit-s', type=float, default=600.0)
    ap.add_argument('--lp-too-far-margin', type=float, default=0.10,
                    help='boxes with LP < (target - margin) are skipped '
                         'from K-ladder and saved to survivors_lp_too_far.npz')
    args = ap.parse_args()

    bnb_cfg = BnBPhaseConfig(
        d=args.d, target_str=args.target,
        workers=args.bnb_workers,
        init_split_depth=args.bnb_init_split_depth,
        warmup_s=args.bnb_warmup_s,
        stuck_window_s=args.bnb_stuck_window_s,
        time_budget_s=args.bnb_time_budget_s,
    )
    cfg = PipelineConfig(
        d=args.d, target_str=args.target,
        output_root=args.output_root, tag=args.tag,
        bnb=bnb_cfg,
        k_ladder=[int(k) for k in args.k_ladder.split(',')],
        sdp_time_limit_s=args.sdp_time_limit_s,
        lp_too_far_margin=args.lp_too_far_margin,
    )
    run_pipeline(cfg, _REPO)


if __name__ == '__main__':
    main()
