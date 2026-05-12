"""V2 Pilot runner for the FULL design: Legendre + S_3 x Z/2 orbit reduction +
Christoffel-Darboux nu-side dualization (SOS Gram matrices on [-1/2, 1/2]).

Sweeps a (k, N) ladder, records numerical objective lambda, lift Delta,
and timing/memory.  Saves per-run JSON to results/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from lasserre.threepoint_full import (
    build_2pt_full, build_3pt_full, solve, BuildInfo,
)


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _bi_to_dict(bi: BuildInfo) -> Dict:
    return dict(
        k=bi.k, N=bi.N, with_3pt=bi.with_3pt,
        n_orbits_y=bi.n_orbits_y, n_orbits_g=bi.n_orbits_g, n_orbits_m=bi.n_orbits_m,
        block_sizes=list(bi.block_sizes),
        n_constraints=bi.n_constraints,
        build_seconds=bi.build_seconds, solve_seconds=bi.solve_seconds,
        peak_mem_mb=bi.peak_mem_mb,
        status=bi.status, objective=bi.objective,
    )


def run_pair(k: int, N: int, *, solver: str, verbose: bool = False) -> Dict:
    print(f"  building 2-point ...", flush=True)
    p2, info2, _ = build_2pt_full(k, N)
    print(f"    build {info2.build_seconds:.2f} s, blocks {info2.block_sizes}, "
          f"orbits g={info2.n_orbits_g} m={info2.n_orbits_m}", flush=True)
    print(f"  solving 2-point ({solver}) ...", flush=True)
    info2 = solve(p2, info2, solver=solver, verbose=verbose)
    print(f"    -> status={info2.status}, lambda^2pt = {info2.objective}, "
          f"solve {info2.solve_seconds:.2f} s, peakmem +{info2.peak_mem_mb:.0f} MB",
          flush=True)

    print(f"  building 3-point ...", flush=True)
    p3, info3, _ = build_3pt_full(k, N)
    print(f"    build {info3.build_seconds:.2f} s, blocks {info3.block_sizes}, "
          f"orbits y={info3.n_orbits_y}", flush=True)
    print(f"  solving 3-point ({solver}) ...", flush=True)
    info3 = solve(p3, info3, solver=solver, verbose=verbose)
    print(f"    -> status={info3.status}, lambda^3pt = {info3.objective}, "
          f"solve {info3.solve_seconds:.2f} s, peakmem +{info3.peak_mem_mb:.0f} MB",
          flush=True)

    delta = None
    if info2.objective is not None and info3.objective is not None:
        delta = info3.objective - info2.objective

    return dict(
        k=k, N=N,
        two_pt=_bi_to_dict(info2),
        three_pt=_bi_to_dict(info3),
        delta=delta,
        solver=solver,
    )


def run_ladder(ladder: List[Tuple[int, int]], *, solver: str, verbose: bool = False) -> List[Dict]:
    out: List[Dict] = []
    for (k, N) in ladder:
        avail_gb = psutil.virtual_memory().available / 1e9
        print(f"\n=== (k={k}, N={N})  available RAM {avail_gb:.1f} GB ===", flush=True)
        record = run_pair(k, N, solver=solver, verbose=verbose)
        out.append(record)
        json.dump(record, open(RESULTS_DIR / f"threepoint_full_k{k}_N{N}.json", "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=str, default="2,2;3,4;4,6;5,8",
                    help="Semicolon-separated k,N pairs (need 2k >= N).")
    ap.add_argument("--solver", type=str, default="MOSEK")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ladder: List[Tuple[int, int]] = []
    for tok in args.ladder.split(";"):
        k, N = tok.split(",")
        ladder.append((int(k), int(N)))

    print(f"Pilot ladder (FULL design): {ladder}, solver={args.solver}")
    records = run_ladder(ladder, solver=args.solver, verbose=args.verbose)

    print("\n\n=== SUMMARY ===")
    print(f"{'k':>3} {'N':>3} {'lambda^2pt':>14} {'lambda^3pt':>14} {'Delta':>14} "
          f"{'t_2pt':>8} {'t_3pt':>8} {'orbits_y':>9} {'blk3D':>6}")
    for r in records:
        l2 = r["two_pt"]["objective"]
        l3 = r["three_pt"]["objective"]
        d = r["delta"]
        t2 = r["two_pt"]["solve_seconds"]
        t3 = r["three_pt"]["solve_seconds"]
        oy = r["three_pt"]["n_orbits_y"]
        blk3D = r["three_pt"]["block_sizes"][0] if r["three_pt"]["block_sizes"] else 0
        s = f"{r['k']:>3} {r['N']:>3} "
        s += f"{l2:>14.8f} " if isinstance(l2, float) else f"{'FAIL':>14} "
        s += f"{l3:>14.8f} " if isinstance(l3, float) else f"{'FAIL':>14} "
        s += f"{d:>14.3e} " if isinstance(d, float) else f"{'-':>14} "
        s += f"{t2:>8.2f} {t3:>8.2f} {oy:>9} {blk3D:>6}"
        print(s)
    json.dump(records, open(RESULTS_DIR / "threepoint_full_summary.json", "w"), indent=2)
    print(f"\nWrote {RESULTS_DIR / 'threepoint_full_summary.json'}")


if __name__ == "__main__":
    main()
