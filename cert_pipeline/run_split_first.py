"""SPLIT-FIRST orchestrator — guaranteed-termination cert pipeline.

ARCHITECTURE
============
Replaces the K-ladder approach (which thrashes at K=64/K=128 on hard
boxes) with iterative SPLIT-then-SDP. Each iter halves the McCormick
LP gap by ~4× per dimension via splits, so any box with val_B > target
WILL eventually reach hw small enough that LP or K=0 SDP certs it.

  ITER 1:
    A. BnB cascade (X minutes) → dump in-flight boxes + master queue
    B. LP-cert filter the dumps (drops boxes already cert by LP)
    C. Save survivors_iter_001.npz

  ITER 2..N:
    A. Load survivors_iter_(N-1).npz
    B. SPLIT each box to depth D (= 2^D children/box)
    C. LP-cert filter the children (most cert by LP at smaller hw)
    D. K=0 SDP sweep (parallel pool, auto-tuned)
    E. K=16 fallback on K=0 failures (single retry)
    F. Save survivors_iter_N.npz; dropped to next iter
    G. STOP if no survivors OR if hit MAX_ITERS

WHY THIS GUARANTEES TERMINATION (modulo MAX_ITERS)
==================================================
McCormick LP gap on a box of half-width hw is `O(hw²)`. Each split of
the widest axis halves hw → gap shrinks 4×. After D splits, gap shrinks
4^D. For target=1.2805 vs val(d=22) ≈ 1.309 (gap_to_target=0.028), once
LP_gap_from_val_B is below 0.028, LP cert succeeds.

Initial McCormick LP gap on a hw=0.5 box is ~0.06–0.10. After 4 splits
(gap shrinks 256×), the LP gap is ~2e-4 ≪ 0.028 → LP almost always
certs. K=0 SDP closes the rest.

CHECKPOINTS + RESUMABILITY
==========================
After EVERY phase write to disk:
  iter_NNN/
    children_after_split.npz    # input to LP filter
    children_after_lp.npz       # input to K-ladder (LP-failing only)
    survivors_after_K0.npz      # K=0 sweep failures
    survivors_after_K16.npz     # K=16 sweep failures (= iter survivors)
    iter_summary.json           # cert/fail counts
    sdp/per_box/<hash>.json     # full per-box SDP results
  pending_survivors.npz         # MASTER pointer to current survivor pool
                                # (atomic-replaced after each iter)
  pipeline_summary.json         # rolling rollup

So `pending_survivors.npz` is ALWAYS the authoritative current state.
Pull it down to local at any time to inspect or pivot.

LOCAL BACKUP (USER-DRIVEN)
==========================
Run on your LOCAL machine (separate terminal):

    python -m cert_pipeline.local_pull --pod ubuntu@HOST --run RUN_DIR \\
        --interval-s 300 --local-dir ./runs_local

This rsyncs the run dir from the pod to local every 5 min so you don't
lose progress if the pod dies. (Or just run rsync manually whenever.)

CLI
===
  python -m cert_pipeline.run_split_first \\
      --d 22 --target 1.2805 \\
      --bnb-time-budget-s 1800 \\
      --split-depth 4 \\
      --max-iters 6 \\
      --k-ladder 0,16
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
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from cert_pipeline.bnb_phase import (
    BnBPhaseConfig, BnBPhaseResult, run_one_bnb_phase,
)
from cert_pipeline.k_ladder import (
    SurvivorBox, run_k_ladder, survivors_to_npz, survivors_from_npz,
)
from cert_pipeline.kill_survivors import (
    split_survivors, lp_cert_filter, MathInsufficient,
)
from cert_pipeline.box_journal import canonical_box_hash


# After this many split-then-SDP iterations on a single lineage with no
# cert, declare the lineage stuck and abort with MATH_INSUFFICIENT_AT_d.
# This catches both (a) box geometry where SDP relaxation can never close
# (val_B at the local minimum is exactly target) and (b) val(d) < target
# (proof attempt cannot succeed at this d). Either way the answer is
# raise d / lower target — not throw more compute at it.
STUCK_ITERS_THRESHOLD = 3


# ==========================================================================
# Helpers
# ==========================================================================

def _utc_tag() -> str:
    return datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')


def _git_state(repo_root: Path) -> Dict[str, str]:
    import subprocess
    out: Dict[str, str] = {}
    try:
        out['git_rev'] = subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        out['git_rev'] = 'unknown'
    return out


def _load_dump_boxes(dump_files: List[str]) -> List[Dict[str, Any]]:
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
            if lo_arr.ndim == 1:
                lo_arr = lo_arr[None, :]
                hi_arr = hi_arr[None, :]
            n = lo_arr.shape[0]
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


def _filter_all_lp_failing(raw_boxes: List[Dict[str, Any]],
                            d: int, target: float) -> List[SurvivorBox]:
    """Convert raw dumps to SurvivorBox list, dropping LP-cert ones.

    Crucially does NOT filter LP-too-far — those go through the same
    split-and-SDP loop (splitting ALWAYS helps because LP gap → 0).
    """
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_epigraph import bound_epigraph_lp_float
    from interval_bnb.box import SCALE as _SCALE
    windows = build_windows(d)
    out: List[SurvivorBox] = []
    n_lp_cert = 0
    n_empty = 0
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
        out.append(SurvivorBox(
            hash=bhash, lo_int=lo_int, hi_int=hi_int,
            depth=int(box.get('depth', 0)), volume=vol,
            lp_val=lp if np.isfinite(lp) else None,
            src=box.get('src', ''),
        ))
    print(f"  [filter] raw={len(raw_boxes)}  empty={n_empty}  "
          f"lp_cert={n_lp_cert}  lp_failing→pool={len(out)}",
          flush=True)
    return out


def _dedupe_by_hash(survivors: List[SurvivorBox]) -> List[SurvivorBox]:
    seen: Dict[str, SurvivorBox] = {}
    for s in survivors:
        if s.hash not in seen:
            seen[s.hash] = s
    return list(seen.values())


# ==========================================================================
# Main split-first loop
# ==========================================================================

@dataclass
class IterSummary:
    iter: int
    n_input: int
    n_after_split: int
    n_after_lp: int
    n_lp_cert: int
    n_sdp_input: int
    n_sdp_cert: int
    n_survivors: int
    wall_s: float
    sdp_stage_summaries: List[Dict[str, Any]] = field(default_factory=list)


def write_atomic_npz(survivors: List[SurvivorBox], path: Path) -> None:
    """Write npz atomically (write to temp, then rename).

    np.savez auto-appends `.npz` if the target path doesn't end in
    `.npz`, so the tmp basename MUST end in `.npz` (otherwise np.savez
    writes to `<tmp>.npz` and the rename misses).
    """
    tmp_basename = path.stem + '.tmp.npz'           # e.g. pending_survivors.tmp.npz
    tmp = path.with_name(tmp_basename)
    survivors_to_npz(survivors, str(tmp))
    tmp.replace(path)


def update_pipeline_summary(run_dir: Path, payload: Dict[str, Any]) -> None:
    """Atomically update pipeline_summary.json."""
    tmp = run_dir / 'pipeline_summary.json.tmp'
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(run_dir / 'pipeline_summary.json')


def run_split_first_iter(iter_n: int, d: int, target_f: float,
                          pending: List[SurvivorBox],
                          iter_dir: Path, repo_root: Path,
                          split_depth: int, k_ladder: List[int],
                          sdp_time_limit_s: float) -> IterSummary:
    """Run one split-then-SDP iteration."""
    iter_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\n=== ITER {iter_n} === ({len(pending)} pending, "
          f"split_depth={split_depth})", flush=True)

    # Phase A: SPLIT
    children = split_survivors(pending, split_depth)
    write_atomic_npz(children, iter_dir / 'children_after_split.npz')
    print(f"  [split] {len(pending)} → {len(children)} children", flush=True)

    # Phase B: LP-CERT FILTER
    sdp_eligible, n_lp_cert = lp_cert_filter(children, d, target_f)
    write_atomic_npz(sdp_eligible, iter_dir / 'children_after_lp.npz')
    print(f"  [lp]    {n_lp_cert} LP-cert  →  {len(sdp_eligible)} need SDP",
          flush=True)

    # Phase C: K-LADDER (typically just K=0 + K=16)
    if not sdp_eligible:
        print(f"  [iter {iter_n}] all children LP-cert. ITER COMPLETE.",
              flush=True)
        summary = IterSummary(
            iter=iter_n, n_input=len(pending),
            n_after_split=len(children), n_after_lp=0,
            n_lp_cert=n_lp_cert, n_sdp_input=0,
            n_sdp_cert=0, n_survivors=0,
            wall_s=time.time() - t0,
        )
        (iter_dir / 'iter_summary.json').write_text(
            json.dumps(asdict(summary), indent=2, default=str))
        return summary

    ladder = run_k_ladder(d, target_f, sdp_eligible,
                           iter_dir / 'k_ladder', repo_root,
                           k_ladder=k_ladder,
                           time_limit_s=sdp_time_limit_s)
    n_sdp_cert = ladder.total_cert
    survivors_out = ladder.final_survivors
    write_atomic_npz(survivors_out, iter_dir / 'survivors_after_kladder.npz')

    summary = IterSummary(
        iter=iter_n, n_input=len(pending),
        n_after_split=len(children), n_after_lp=len(sdp_eligible),
        n_lp_cert=n_lp_cert, n_sdp_input=len(sdp_eligible),
        n_sdp_cert=n_sdp_cert, n_survivors=len(survivors_out),
        wall_s=time.time() - t0,
        sdp_stage_summaries=[
            {'K': s.K, 'n_input': s.n_input, 'n_cert': s.n_cert,
             'n_fail': s.n_fail, 'max_rss_gb': s.max_rss_gb_observed,
             'n_parallel': s.n_parallel,
             'sweep_wall_s': s.sweep_wall_s}
            for s in ladder.stages
        ],
    )
    (iter_dir / 'iter_summary.json').write_text(
        json.dumps(asdict(summary), indent=2, default=str))
    print(f"  [iter {iter_n}] DONE  in={len(pending)} children={len(children)} "
          f"lp_cert={n_lp_cert} sdp_cert={n_sdp_cert} survivors={len(survivors_out)} "
          f"wall={summary.wall_s:.0f}s", flush=True)
    return summary


def _build_root_pool(d: int) -> List[SurvivorBox]:
    """Construct a single SurvivorBox covering the entire (d-1)-simplex
    with the standard half-symmetry cut mu_0 in [0, 1/2].

    This is the SOUND starting point when --skip-bnb is on: the union of
    this single box's children (via splits) IS the simplex. No cascade
    means no INSTANT_DUMP race-condition window where boxes could be
    silently lost — every box that ever exists is on disk in some
    iter_NNN/children_after_split.npz.
    """
    from interval_bnb.box import Box, SCALE as _SCALE
    init = Box.initial(d, sym_cuts=[(0, d - 1)])
    lo_int = list(init.lo_int) if init.lo_int is not None else [0] * d
    hi_int = list(init.hi_int) if init.hi_int is not None else [_SCALE] * d
    bhash = canonical_box_hash(lo_int, hi_int)
    vol = float(np.prod([(h - l) / _SCALE for l, h in zip(lo_int, hi_int)]))
    return [SurvivorBox(
        hash=bhash, lo_int=lo_int, hi_int=hi_int,
        depth=0, volume=vol, lp_val=None,
        src='root_simplex',
        iters_survived=0,
    )]


def main():
    ap = argparse.ArgumentParser(
        description='Split-first iterative cert pipeline (guaranteed term).')
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--target', type=str, required=True,
                    help='e.g. "1.2805"')
    ap.add_argument('--output-root', type=str, default='runs')
    ap.add_argument('--tag', type=str, default='',
                    help='auto-generated if empty')
    ap.add_argument('--resume', type=str, default='',
                    help='Resume from existing run dir; skips BnB phase '
                         'and starts iter loop from pending_survivors.npz')
    # --skip-bnb DEFAULTS TO ON: bypasses the cascade entirely and
    # starts split-first from the root simplex box. This is the SOUND
    # path — the cascade's INSTANT_DUMP can lose in-flight boxes when
    # workers are stuck inside C extensions when SIGINT fires.
    # Set --skip-bnb=0 to run the (faster but harder-to-soundness-prove)
    # cascade-then-split-first variant.
    ap.add_argument('--skip-bnb', type=int, default=1,
                    help='1 = skip cascade, start from root simplex (DEFAULT, '
                         'sound). 0 = run cascade first, then split-first '
                         'on the dumps.')
    ap.add_argument('--bnb-workers', type=int, default=16)
    ap.add_argument('--bnb-init-split-depth', type=int, default=22)
    ap.add_argument('--bnb-warmup-s', type=int, default=300)
    ap.add_argument('--bnb-stuck-window-s', type=int, default=600,
                    help='Cascade plateau window (10 min default)')
    ap.add_argument('--bnb-time-budget-s', type=int, default=1800,
                    help='Cascade hard cap (30 min default)')
    ap.add_argument('--split-depth', type=int, default=4,
                    help='Splits per box per iter (top-N widest axes; '
                         '=> 2^N children/box). 4 is the sweet spot at d=22 '
                         '— enough binding pairs touched per iter to drive '
                         'LP cert rate above 50%, low enough to keep child '
                         'count tractable.')
    ap.add_argument('--max-iters', type=int, default=12,
                    help='Maximum split-then-SDP iterations. Higher than '
                         'the cascade-feed default (6) because root-simplex '
                         'mode needs more iters to reach cert-friendly hw.')
    ap.add_argument('--k-ladder', type=str, default='0,16,32',
                    help='Comma-separated K values. Default includes K=32 '
                         'as last-resort SDP tier.')
    ap.add_argument('--sdp-time-limit-s', type=float, default=600.0)
    ap.add_argument('--stuck-iters-threshold', type=int,
                    default=STUCK_ITERS_THRESHOLD,
                    help='Abort if any box lineage survives this many '
                         'split-then-SDP iters without cert. Distinguishes '
                         'math-failure (val_B too tight) from config-failure '
                         '(time / parallelism).')
    args = ap.parse_args()

    target_f = float(args.target)
    k_ladder = [int(k) for k in args.k_ladder.split(',')]

    # ---- Setup run dir ----
    if args.resume:
        run_dir = Path(args.resume)
        assert run_dir.exists(), f"resume dir {run_dir} does not exist"
        print(f"\n#### RESUMING from {run_dir} ####", flush=True)
    else:
        if not args.tag:
            args.tag = f'd{args.d}_t{args.target.replace(".", "p")}_split_{_utc_tag()}'
        run_dir = Path(args.output_root) / args.tag
        run_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot config + git
        config = {
            'd': args.d, 'target': args.target, 'tag': args.tag,
            'skip_bnb': bool(args.skip_bnb),
            'bnb_workers': args.bnb_workers,
            'bnb_init_split_depth': args.bnb_init_split_depth,
            'bnb_warmup_s': args.bnb_warmup_s,
            'bnb_stuck_window_s': args.bnb_stuck_window_s,
            'bnb_time_budget_s': args.bnb_time_budget_s,
            'split_depth': args.split_depth,
            'max_iters': args.max_iters,
            'k_ladder': k_ladder,
            'sdp_time_limit_s': args.sdp_time_limit_s,
            'stuck_iters_threshold': args.stuck_iters_threshold,
            'utc_start': datetime.datetime.utcnow().isoformat() + 'Z',
            'machine': {'cores': os.cpu_count(), 'platform': sys.platform},
            'python': sys.version,
            'git': _git_state(_REPO),
        }
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2,
                                                         default=str))
        print(f"\n#### SPLIT-FIRST PIPELINE ####")
        print(f"  d={args.d} target={args.target} tag={args.tag}")
        print(f"  output: {run_dir}")
        print(f"  iters max: {args.max_iters}, split_depth: {args.split_depth}")
        print(f"  k_ladder: {k_ladder}")

    pipeline_t0 = time.time()
    iter_logs: List[Dict[str, Any]] = []

    # ---- INITIAL POOL: root-simplex (skip-bnb=ON) OR cascade dumps ----
    pending_path = run_dir / 'pending_survivors.npz'
    if args.resume and pending_path.exists():
        pending = survivors_from_npz(str(pending_path))
        print(f"\n[resume] loaded {len(pending)} pending from {pending_path}")
    elif args.skip_bnb:
        # SOUND PATH: start from the single root-simplex box. Every
        # subsequent box exists on disk in iter_NNN/children_after_split.npz
        # so soundness == cert + dropped-by-simplex-filter + final survivors.
        print(f"\n[skip-bnb] starting from root simplex (no cascade) "
              f"— soundness path", flush=True)
        pending = _build_root_pool(args.d)
        write_atomic_npz(pending, pending_path)
        print(f"[skip-bnb] initial pool = {len(pending)} root box "
              f"→ {pending_path.name}", flush=True)
    else:
        print(f"\n[bnb] running cascade (skip-bnb=0) — "
              f"NOTE: INSTANT_DUMP can lose in-flight boxes if workers "
              f"are stuck in C extensions; soundness depends on cascade "
              f"accounting balancing (check bnb_summary.json).", flush=True)
        bnb_cfg = BnBPhaseConfig(
            d=args.d, target_str=args.target,
            workers=args.bnb_workers,
            init_split_depth=args.bnb_init_split_depth,
            warmup_s=args.bnb_warmup_s,
            stuck_window_s=args.bnb_stuck_window_s,
            time_budget_s=args.bnb_time_budget_s,
            instant_dump=True,
        )
        bnb_dir = run_dir / 'bnb_phase'
        bnb_result = run_one_bnb_phase(bnb_cfg, _REPO, bnb_dir)
        print(f"\n[bnb] done. trigger={bnb_result.trigger_reason} "
              f"dumped={bnb_result.n_dumped_total}", flush=True)

        # LP-cert filter the dumps to get initial pending pool
        raw = _load_dump_boxes(bnb_result.dump_files)
        pending = _filter_all_lp_failing(raw, args.d, target_f)
        pending = _dedupe_by_hash(pending)
        write_atomic_npz(pending, pending_path)
        print(f"[bnb] {len(pending)} survivors → {pending_path.name}",
              flush=True)

    # ---- ITERATIVE SPLIT-THEN-SDP LOOP ----
    iter_n = 0
    math_failure_msg: Optional[str] = None
    stuck_failure_msg: Optional[str] = None
    while pending and iter_n < args.max_iters:
        iter_n += 1
        iter_dir = run_dir / f'iter_{iter_n:03d}'
        if iter_dir.exists() and (iter_dir / 'survivors_after_kladder.npz').exists():
            # Resume case: this iter already done, load its output
            survs = survivors_from_npz(
                str(iter_dir / 'survivors_after_kladder.npz'))
            print(f"\n[resume] iter {iter_n} already done → "
                  f"{len(survs)} survivors", flush=True)
            pending = survs
            write_atomic_npz(pending, pending_path)
            # Reload its summary
            try:
                summary = json.loads(
                    (iter_dir / 'iter_summary.json').read_text())
                iter_logs.append(summary)
            except Exception:
                pass
            continue

        # Run the iter — surface MathInsufficient cleanly.
        try:
            summary = run_split_first_iter(
                iter_n, args.d, target_f, pending, iter_dir, _REPO,
                args.split_depth, k_ladder, args.sdp_time_limit_s)
        except MathInsufficient as e:
            # A box hit the dyadic-2^60 grid floor without certifying.
            # f(mu) at that point is < target → val(d) < target →
            # raise d or lower the target.
            math_failure_msg = (
                f'MATH_INSUFFICIENT_AT_d={args.d}: {e}. '
                f'val(d) < target={args.target}. Raise d or lower target.'
            )
            print(f"\n[FATAL] {math_failure_msg}", flush=True)
            break
        iter_logs.append(asdict(summary))

        # Update pending for next iter
        next_path = iter_dir / 'survivors_after_kladder.npz'
        if next_path.exists():
            pending = survivors_from_npz(str(next_path))
        else:
            pending = []
        write_atomic_npz(pending, pending_path)

        # ---- STUCK-LINEAGE DETECTION (#4 / #5 from the audit) ----
        # If any surviving box has been through >= STUCK_ITERS_THRESHOLD
        # split-then-SDP iters without getting certified, abort. The
        # box is either at a true val_B = target geometry (relaxation
        # gap will never close at this d) or val(d) < target. Either
        # way, more compute won't help — escalate d.
        max_iters_survived = max((b.iters_survived for b in pending),
                                  default=0)
        if max_iters_survived >= args.stuck_iters_threshold:
            stuck_lineage = [b for b in pending
                              if b.iters_survived >= args.stuck_iters_threshold]
            stuck_failure_msg = (
                f'STUCK_AT_d={args.d}: {len(stuck_lineage)}/{len(pending)} '
                f'boxes have survived {max_iters_survived} iters of '
                f'split-then-SDP without cert (threshold='
                f'{args.stuck_iters_threshold}). Raise d or relax target. '
                f'Sample stuck box: hash={stuck_lineage[0].hash} '
                f'depth={stuck_lineage[0].depth}.'
            )
            print(f"\n[FATAL] {stuck_failure_msg}", flush=True)
            # Persist the stuck-lineage list for inspection.
            survivors_to_npz(stuck_lineage,
                              str(run_dir / 'stuck_lineage.npz'))
            break

        # Update rolling pipeline summary
        update_pipeline_summary(run_dir, {
            'tag': args.tag if not args.resume else run_dir.name,
            'd': args.d, 'target': args.target,
            'iters_done': iter_n,
            'pending_count': len(pending),
            'max_iters_survived_in_pool': max_iters_survived,
            'total_wall_s': time.time() - pipeline_t0,
            'iter_logs': iter_logs,
            'verdict': ('CERTIFIED_FULL_PIPELINE' if not pending
                        else f'{len(pending)}_PENDING_AFTER_ITER_{iter_n}'),
        })

    # ---- ROLLUP ----
    total_wall = time.time() - pipeline_t0
    final_n = len(pending)
    if math_failure_msg is not None:
        verdict = math_failure_msg
    elif stuck_failure_msg is not None:
        verdict = stuck_failure_msg
    elif final_n == 0:
        verdict = 'CERTIFIED_FULL_PIPELINE'
    elif iter_n >= args.max_iters:
        verdict = f'INCOMPLETE_HIT_MAX_ITERS_{final_n}_REMAINING'
    else:
        verdict = f'INCOMPLETE_{final_n}_REMAINING'
    final = {
        'tag': args.tag if not args.resume else run_dir.name,
        'd': args.d, 'target': args.target,
        'iters_done': iter_n, 'iters_max': args.max_iters,
        'pending_count': final_n,
        'total_wall_s': total_wall,
        'iter_logs': iter_logs,
        'verdict': verdict,
        'utc_end': datetime.datetime.utcnow().isoformat() + 'Z',
    }
    update_pipeline_summary(run_dir, final)
    print(f"\n{'#'*70}")
    print(f"# PIPELINE END  iters={iter_n}/{args.max_iters}  remaining={final_n}")
    print(f"# verdict: {verdict}")
    print(f"# wall: {total_wall:.0f}s ({total_wall/60:.1f} min)")
    print(f"# output: {run_dir}")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
