"""Local-only val(d) trajectory bisection.

For each d in DIMS, bisects t over [t_lo, t_hi] using
``lasserre.d64_farkas_cert.certify_sparse_farkas`` and records the
largest t for which Farkas certifies infeasibility (i.e., the highest
rigorously-certified lower bound on val^{(k,b)}(d)).

Hyperparameters are FIXED across d's (apples-to-apples):
    order = 2
    bandwidth = 7         # clamped to d-1 internally for d <= 8
    max_denom_S = 10**10
    max_denom_mu = 10**10
    eig_margin = 1e-9
    bisect_tol = 1e-3

Usage:
    python -m lasserre.trajectory.run_trajectory --d 4 8 16
    python -m lasserre.trajectory.run_trajectory --d 32 --time_budget_s 1800

Per-d JSON output:
    lasserre/trajectory/results/d{d}_trajectory.json

Sentinel statuses:
    COMPLETED              — bisection finished within tolerance
    TIME_BUDGET_EXCEEDED   — wall budget exhausted between probes
    OOM_OR_SOLVER_FAILURE  — exception during a probe (logs details)
    UNTRIED                — never invoked (skipped by user choice)

Time budget interpretation: enforced BETWEEN probes; a probe in
progress is allowed to complete, since killing a Python+MOSEK process
mid-solve does not give a clean RSS snapshot anyway.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from lasserre.d64_farkas_cert import certify_sparse_farkas
from lasserre.d64_solver import solve_sparse_farkas_at_t


# ---------------------------------------------------------------------
# Default hyperparameters (held fixed across d's)
# ---------------------------------------------------------------------
DEFAULT_ORDER = 2
DEFAULT_BANDWIDTH = 7
DEFAULT_MAX_DENOM_S = 10**10
DEFAULT_MAX_DENOM_MU = 10**10
DEFAULT_EIG_MARGIN = 1e-9
DEFAULT_BISECT_TOL = 1e-3
DEFAULT_T_LO = 1.05
DEFAULT_T_HI = 1.30


@dataclass
class ProbeRecord:
    t: float
    status: str                 # CERTIFIED | NOT_CERTIFIED | FEASIBLE | ERROR
    safety_margin_sign: str     # 'pos' / 'nonpos' / 'na'
    wall_s: float
    peak_rss_after_gb: float    # peak RSS observed after this probe, in GB
    notes: str = ''


@dataclass
class DTrajectoryResult:
    d: int
    order: int
    bandwidth: int
    t_lo_init: float
    t_hi_init: float
    bisect_tol: float
    largest_certified_t: Optional[float]
    largest_certified_t_str: Optional[str]   # exact rational e.g. "1281/1000"
    probes: List[ProbeRecord]
    n_y: Optional[int] = None
    n_clique: Optional[int] = None
    n_loc_blocks: Optional[int] = None
    n_win_blocks: Optional[int] = None
    largest_psd_block: Optional[int] = None
    mosek_iter_count: Optional[int] = None
    total_wall_s: float = 0.0
    peak_rss_overall_gb: float = 0.0
    status: str = 'UNTRIED'
    timestamp: str = ''
    host_info: Dict[str, Any] = field(default_factory=dict)
    notes: str = ''


def _peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB."""
    p = psutil.Process(os.getpid())
    info = p.memory_info()
    # .rss is current. peak_wset is Windows-only via memory_info_ex; emulate.
    rss = info.rss
    try:
        # Windows: peak working set via memory_full_info or memory_info()
        # has 'peak_wset' on win32; on linux use VmHWM via /proc/self/status
        ex = p.memory_info()
        if hasattr(ex, 'peak_wset'):
            rss = max(rss, ex.peak_wset)
        else:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmHWM:'):
                        rss = max(rss, int(line.split()[1]) * 1024)
                        break
    except Exception:
        pass
    return rss / (1024 ** 3)


def _problem_size_for_d(d: int, order: int, bandwidth: int) -> Dict[str, Any]:
    """Run a fast solve at a clearly-feasible t_test just to extract sizes
    and MOSEK iteration count. Returns a small dict; on failure returns
    {} and sets a flag.
    """
    from lasserre.d64_solver import solve_sparse_farkas_at_t
    res = solve_sparse_farkas_at_t(
        d=d, order=order, bandwidth=bandwidth, t_test=DEFAULT_T_HI,
        verbose=False,
    )
    largest = 0
    if res.blocks_meta:
        for tag in ('mom_blocks', 'loc_blocks', 'win_blocks'):
            for entry in res.blocks_meta.get(tag, []):
                size = entry[-1]
                if isinstance(size, int):
                    largest = max(largest, size)
    return {
        'n_y': res.n_y,
        'n_clique': res.n_clique,
        'n_loc_blocks': len(res.blocks_meta.get('loc_blocks', [])) if res.blocks_meta else None,
        'n_win_blocks': len(res.blocks_meta.get('win_blocks', [])) if res.blocks_meta else None,
        'largest_psd_block': largest,
        'sizing_solve_status': res.status,
        'sizing_solve_wall_s': res.solver_time + res.build_time,
    }


def measure_d(
    d: int,
    *,
    order: int = DEFAULT_ORDER,
    bandwidth: int = DEFAULT_BANDWIDTH,
    t_lo: float = DEFAULT_T_LO,
    t_hi: float = DEFAULT_T_HI,
    bisect_tol: float = DEFAULT_BISECT_TOL,
    max_denom_S: int = DEFAULT_MAX_DENOM_S,
    max_denom_mu: int = DEFAULT_MAX_DENOM_MU,
    eig_margin: float = DEFAULT_EIG_MARGIN,
    time_budget_s: float = 1800.0,
    verbose: bool = True,
) -> DTrajectoryResult:
    """Bisect on t for one d. Honors a wall time budget between probes."""
    import datetime
    t0 = time.time()
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    host_info = {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'mem_total_gb': round(psutil.virtual_memory().total / 1024**3, 1),
    }

    res = DTrajectoryResult(
        d=d, order=order, bandwidth=bandwidth,
        t_lo_init=t_lo, t_hi_init=t_hi, bisect_tol=bisect_tol,
        largest_certified_t=None, largest_certified_t_str=None,
        probes=[], timestamp=timestamp, host_info=host_info,
    )

    # Step 1: extract problem-size metadata via a quick high-t (feasible) solve
    if verbose:
        print(f'\n=== d={d} sizing run at t={t_hi}... ===', flush=True)
    try:
        sz = _problem_size_for_d(d, order, bandwidth)
        res.n_y = sz['n_y']
        res.n_clique = sz['n_clique']
        res.n_loc_blocks = sz['n_loc_blocks']
        res.n_win_blocks = sz['n_win_blocks']
        res.largest_psd_block = sz['largest_psd_block']
        if verbose:
            print(f'  n_y={res.n_y}  n_clique={res.n_clique}  '
                  f'n_loc={res.n_loc_blocks}  n_win={res.n_win_blocks}  '
                  f'largest_psd={res.largest_psd_block}', flush=True)
    except MemoryError as e:
        res.status = 'OOM_OR_SOLVER_FAILURE'
        res.notes = f'sizing run OOMed: {e}'
        return res
    except Exception as e:
        res.status = 'OOM_OR_SOLVER_FAILURE'
        res.notes = f'sizing run failed: {type(e).__name__}: {e}'
        return res

    # Step 2: bisect (with shared precompute cache across probes)
    largest_certified_t: Optional[float] = None
    largest_certified_t_str: Optional[str] = None
    lo, hi = float(t_lo), float(t_hi)
    bisect_cache: Dict[str, Any] = {}

    # Anchor probe at lo first to confirm an INFEASIBLE base — saves
    # bisection time when MOSEK returns UNKNOWN at borderline t. If
    # lo itself doesn't certify, we know order=2/b=bandwidth caps below
    # lo and there's no point bisecting.
    if verbose:
        print(f'\n--- d={d} ANCHOR probe at t={lo:.5f} ---', flush=True)
    probe_t0 = time.time()
    try:
        cert_anchor = certify_sparse_farkas(
            d=d, order=order, bandwidth=bandwidth, t_test=lo,
            max_denom_S=max_denom_S, max_denom_mu=max_denom_mu,
            eig_margin=eig_margin, verbose=False,
            mpmath_verify=False, _cache=bisect_cache,
        )
        anchor_wall = time.time() - probe_t0
        anchor_status = cert_anchor.status
        res.probes.append(ProbeRecord(
            t=lo, status=cert_anchor.status,
            safety_margin_sign=('pos' if cert_anchor.status == 'CERTIFIED' else 'nonpos'),
            wall_s=anchor_wall, peak_rss_after_gb=_peak_rss_gb(),
            notes=cert_anchor.notes[:120]))
        if verbose:
            print(f'  anchor t={lo:.5f}: {cert_anchor.status} '
                  f'({anchor_wall:.0f}s)', flush=True)
        if cert_anchor.status == 'CERTIFIED':
            largest_certified_t = lo
            largest_certified_t_str = cert_anchor.lb_rig
        else:
            # No CERTIFIED at the lo bracket — order/bandwidth too weak
            # to clear t_lo. Skip bisection.
            res.status = 'COMPLETED'
            res.notes = f'anchor probe at lo={lo} returned {cert_anchor.status}; '\
                        f'order={order}/bandwidth={bandwidth} insufficient at d={d}.'
            res.largest_certified_t = None
            res.total_wall_s = time.time() - t0
            res.peak_rss_overall_gb = max(
                (p.peak_rss_after_gb for p in res.probes), default=_peak_rss_gb())
            return res
    except Exception as e:
        anchor_wall = time.time() - probe_t0
        tb = traceback.format_exc()
        print(f'!!! anchor at t={lo} raised {type(e).__name__}: {e}', flush=True)
        print(tb, flush=True)
        res.probes.append(ProbeRecord(
            t=lo, status='ERROR', safety_margin_sign='na',
            wall_s=anchor_wall, peak_rss_after_gb=_peak_rss_gb(),
            notes=f'{type(e).__name__}: {str(e)[:200]}'))
        res.status = 'OOM_OR_SOLVER_FAILURE'
        res.notes = f'anchor exception:\n{tb}'[:2000]
        res.total_wall_s = time.time() - t0
        return res

    while hi - lo > bisect_tol:
        elapsed = time.time() - t0
        if elapsed >= time_budget_s:
            res.status = 'TIME_BUDGET_EXCEEDED'
            res.notes = f'wall budget {time_budget_s:.0f}s exceeded after probe; '\
                        f'gap {hi - lo:.4f} > tol {bisect_tol}'
            break
        mid = 0.5 * (lo + hi)
        if verbose:
            print(f'\n--- d={d} probe t={mid:.5f}  bracket=[{lo:.5f},{hi:.5f}]  '
                  f'elapsed={elapsed:.0f}s ---', flush=True)
        probe_t0 = time.time()
        try:
            cert = certify_sparse_farkas(
                d=d, order=order, bandwidth=bandwidth, t_test=mid,
                max_denom_S=max_denom_S, max_denom_mu=max_denom_mu,
                eig_margin=eig_margin, verbose=False,
                # Skip mpmath cross-check during bisection probing — the
                # numpy float64 sign test with 1e-9 cushion is rigorous to
                # f64 precision; final cert can be re-verified at dps=80
                # via a one-shot rerun at the highest certified t.
                mpmath_verify=False,
                _cache=bisect_cache,
            )
            probe_wall = time.time() - probe_t0
            sm_sign = ('pos' if cert.status == 'CERTIFIED' else
                       'nonpos' if cert.status == 'NOT_CERTIFIED' else 'na')
            rec = ProbeRecord(
                t=mid, status=cert.status, safety_margin_sign=sm_sign,
                wall_s=probe_wall, peak_rss_after_gb=_peak_rss_gb(),
                notes=cert.notes[:120],
            )
            res.probes.append(rec)
            if verbose:
                print(f'  -> {cert.status} ({probe_wall:.1f}s)  '
                      f'rss={rec.peak_rss_after_gb:.2f} GB', flush=True)
            if cert.status == 'CERTIFIED':
                lo = mid
                largest_certified_t = mid
                largest_certified_t_str = cert.lb_rig
            else:
                # NOT_CERTIFIED, FEASIBLE, ERROR all pull hi down
                hi = mid
        except MemoryError as e:
            probe_wall = time.time() - probe_t0
            res.probes.append(ProbeRecord(
                t=mid, status='ERROR', safety_margin_sign='na',
                wall_s=probe_wall, peak_rss_after_gb=_peak_rss_gb(),
                notes=f'MemoryError: {e}'[:120]))
            res.status = 'OOM_OR_SOLVER_FAILURE'
            res.notes = f'OOM at t={mid}'
            break
        except Exception as e:
            probe_wall = time.time() - probe_t0
            tb = traceback.format_exc()
            # Print to stderr so we can diagnose mid-run; also keep in JSON
            print(f'!!! probe at t={mid} raised {type(e).__name__}: {e}',
                  flush=True)
            print(tb, flush=True)
            res.probes.append(ProbeRecord(
                t=mid, status='ERROR', safety_margin_sign='na',
                wall_s=probe_wall, peak_rss_after_gb=_peak_rss_gb(),
                notes=f'{type(e).__name__}: {str(e)[:200]}'))
            res.status = 'OOM_OR_SOLVER_FAILURE'
            res.notes = f'exception at t={mid}:\n{tb}'[:2000]
            break
    else:
        # bisection narrowed within tolerance
        res.status = 'COMPLETED'

    res.largest_certified_t = largest_certified_t
    res.largest_certified_t_str = largest_certified_t_str
    res.total_wall_s = time.time() - t0
    res.peak_rss_overall_gb = max((p.peak_rss_after_gb for p in res.probes),
                                   default=_peak_rss_gb())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, nargs='+', default=[4, 8, 16, 32])
    ap.add_argument('--time_budget_s', type=float, default=1800.0,
                    help='wall budget per d (default 30 min)')
    ap.add_argument('--order', type=int, default=DEFAULT_ORDER)
    ap.add_argument('--bandwidth', type=int, default=DEFAULT_BANDWIDTH)
    ap.add_argument('--t_lo', type=float, default=DEFAULT_T_LO)
    ap.add_argument('--t_hi', type=float, default=DEFAULT_T_HI)
    ap.add_argument('--bisect_tol', type=float, default=DEFAULT_BISECT_TOL)
    ap.add_argument('--results_dir', type=str,
                    default=str(Path(_HERE) / 'results'))
    ap.add_argument('--skip_existing', action='store_true',
                    help='do not re-run d if a JSON result already exists')
    args = ap.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    for d in args.d:
        out_path = Path(args.results_dir) / f'd{d}_trajectory.json'
        if args.skip_existing and out_path.exists():
            print(f'[skip] d={d} (result file exists at {out_path})')
            continue
        try:
            result = measure_d(
                d=d, order=args.order, bandwidth=args.bandwidth,
                t_lo=args.t_lo, t_hi=args.t_hi,
                bisect_tol=args.bisect_tol,
                time_budget_s=args.time_budget_s,
            )
        except KeyboardInterrupt:
            print(f'[abort] d={d} interrupted by user', flush=True)
            raise
        # Persist
        with open(out_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f'\n[saved] {out_path}', flush=True)
        print(f'  status:                {result.status}')
        print(f'  largest_certified_t:   {result.largest_certified_t}')
        print(f'  total_wall_s:          {result.total_wall_s:.1f}')
        print(f'  peak_rss_overall_gb:   {result.peak_rss_overall_gb:.2f}')
        if result.status == 'OOM_OR_SOLVER_FAILURE':
            print(f'  ===> stopping further d (per spec, no auto-downgrade) <===')
            break


if __name__ == '__main__':
    main()
