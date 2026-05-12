#!/usr/bin/env python
"""Parse a result JSON from lasserre_sdpnalplus.py and print a verdict.

The SDPNAL+ driver writes one JSON per run at
    data/sdpnal_<tag>_<timestamp>/result_d<d>_o<o>_bw<bw>_<mode>.json

This script reads that file (or auto-finds the most recent one), compares
the gap closure against the expected baseline for the configuration, and
prints a structured verdict:

  GO       — exceeds the expected gc → proceed to the next rung
  MARGINAL — close to but below the expected gc; run the next rung anyway
  NO-GO    — significantly below; the thesis that SDPNAL+ recovers
             MOSEK-class tightness is wrong; pivot

Usage:
  python tests/interpret_result.py path/to/result.json
  python tests/interpret_result.py --auto-find
  python tests/interpret_result.py --auto-find --d 6 --order 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime


# Reference points (all from PROBLEM_STATE.md + user audit):
# - d=4 L3 full MOSEK: 99.25%
# - d=6 L3 full MOSEK: 99.38%
# - d=8 L3 full MOSEK: unmeasured (MOSEK timed out)
# - d=8 L3 SCS-direct audit: 76.4% (known loose)
_REFS = {
    (4, 3): {'mosek_gc': 99.25, 'note': 'MOSEK full-L3 reference'},
    (6, 3): {'mosek_gc': 99.38, 'note': 'MOSEK full-L3 reference'},
    (8, 3): {'mosek_gc': None,
             'scs_direct_gc': 76.4,
             'note': 'no MOSEK; SCS-direct was 76.4% (likely loose)'},
    (14, 3): {'mosek_gc': None, 'note': 'unmeasured, target lb > 1.2802'},
    (16, 3): {'mosek_gc': None, 'note': 'production run, target lb > 1.2802'},
}


def _find_most_recent(pattern='sdpnal_*'):
    """Find the most recent sdpnal_* subdirectory in data/ that has a
    result_*.json inside."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    data_dir = os.path.join(repo_root, 'data')
    candidates = glob.glob(os.path.join(data_dir, pattern))
    best = None
    best_mtime = -1
    for c in candidates:
        if not os.path.isdir(c):
            continue
        results = glob.glob(os.path.join(c, 'result_*.json'))
        if not results:
            continue
        for r in results:
            if os.path.getmtime(r) > best_mtime:
                best = r
                best_mtime = os.path.getmtime(r)
    return best


def _filter_by_d_order(path, d, order):
    """True if `path` points to a result for the given (d, order)."""
    if d is None and order is None:
        return True
    fname = os.path.basename(path)
    if d is not None and f'd{d}_' not in fname:
        return False
    if order is not None and f'o{order}_' not in fname:
        return False
    return True


def _classify(measured_gc, expected_gc):
    """Return ('GO' | 'MARGINAL' | 'NO-GO', margin, banner_color)."""
    if expected_gc is None:
        # No reference — classify absolutely.
        if measured_gc >= 95.0:
            return 'GO', None, 'GREEN'
        if measured_gc >= 80.0:
            return 'MARGINAL', None, 'YELLOW'
        return 'NO-GO', None, 'RED'
    margin = measured_gc - expected_gc
    if margin >= -0.5:
        return 'GO', margin, 'GREEN'
    if margin >= -5.0:
        return 'MARGINAL', margin, 'YELLOW'
    return 'NO-GO', margin, 'RED'


def _banner(verdict, colour):
    width = 70
    bar = '=' * width
    return f"\n{bar}\nVERDICT: {verdict}   (see analysis below)\n{bar}"


def interpret(json_path, verbose=True):
    """Load a result JSON and print a structured analysis + verdict.
    Returns an integer exit code."""
    with open(json_path) as f:
        data = json.load(f)

    d = data.get('d')
    order = data.get('order')
    bw = data.get('bandwidth')
    mode = data.get('mode', '?')
    lb = data.get('lb')
    val_d = data.get('val_d')
    measured_gc = data.get('gap_closure')
    elapsed = data.get('elapsed')
    sound = data.get('sound', None)

    ref = _REFS.get((d, order), {})
    expected_gc = ref.get('mosek_gc')

    print(f"\nResult file: {json_path}")
    print(f"Config:      d={d}  L{order}  bw={bw}  mode={mode}")
    print(f"Solver:      {data.get('solver', '?')}")
    print(f"Elapsed:     {elapsed}")
    print(f"Sound:       {sound}")
    if 'history' in data and data['history']:
        print(f"Bisection:   {len(data['history'])} steps, "
              f"last verdict = {data['history'][-1].get('verdict', '?')}")

    print()
    print("Measured:")
    print(f"  lb          = {lb}")
    print(f"  val(d)      = {val_d}")
    print(f"  gap closure = {measured_gc}")
    if 'scalar_lb' in data and data['scalar_lb'] is not None:
        print(f"  scalar_lb   = {data['scalar_lb']}")

    print()
    print("Reference:")
    if expected_gc is not None:
        print(f"  expected gc = {expected_gc}% ({ref['note']})")
    else:
        print(f"  expected gc = (unmeasured) — {ref.get('note', '')}")
        if 'scs_direct_gc' in ref:
            print(f"  SCS-direct  = {ref['scs_direct_gc']}% "
                  f"(known loose, should beat this)")

    if measured_gc is None:
        print()
        print("No gap closure in result — likely a feasibility-mode run.")
        cert = data.get('certified')
        if cert:
            print(f"  Feasibility certified: lb > {data.get('target')}")
            return 0
        if cert is False:
            print("  Feasibility NOT certified at the target; run bisection "
                  "mode to see how close we got.")
            return 2
        return 1

    verdict, margin, colour = _classify(measured_gc, expected_gc)
    print(_banner(verdict, colour))

    print()
    if verdict == 'GO':
        if expected_gc is not None:
            print(f"Measured gc {measured_gc:.2f}% matches MOSEK baseline "
                  f"{expected_gc}% within {abs(margin):.2f} pp.")
        else:
            print(f"Measured gc {measured_gc:.2f}% >= 95% — healthy.")
        print()
        # Next-step suggestions by d
        if d == 4:
            print("NEXT: run d=6 L3 acceptance.")
            print("  python tests/lasserre_sdpnalplus.py --d 6 --order 3 --bw 5 "
                  "--mode bisection --sdpnal-tol 1e-7")
        elif d == 6:
            print("NEXT: run d=8 L3 pilot (unmeasured against MOSEK).")
            print("  python tests/lasserre_sdpnalplus.py --d 8 --order 3 --bw 7 "
                  "--mode bisection --sdpnal-tol 1e-7")
        elif d == 8:
            print("NEXT: commit to d=16 production via bisection OR direct ")
            print("      feasibility proof at target=1.2802.")
            print("  # Fast path — single feasibility solve at target:")
            print("  python tests/lasserre_sdpnalplus.py --d 16 --order 3 --bw 15 "
                  "--mode feasibility --target 1.2802")
            print("  # Full bisection (slower, gives certified bisection lb):")
            print("  python tests/lasserre_sdpnalplus.py --d 16 --order 3 --bw 15 "
                  "--mode bisection --target 1.2802 --n-bisect 12")
        elif d == 16:
            if lb and lb > 1.2802:
                print(f"*** RECORD BEATEN: lb = {lb} > 1.2802 ***")
                print("NEXT: draft proof artefact. Verify soundness "
                      "(y[0]=1, consistency, Z/2 equalities hold on solution).")
            else:
                print("lb below target despite clean solve. Either the "
                      "relaxation ceiling is under 1.2802 at d=16 L3, or the "
                      "run hasn't bisected close enough. Check history "
                      "for last bracket; consider --n-bisect 20.")
    elif verdict == 'MARGINAL':
        print(f"gc {measured_gc:.2f}% is {abs(margin):.2f} pp below "
              f"baseline {expected_gc}%. Proceed but expect the margin "
              f"compounds at larger d.")
    else:  # NO-GO
        if expected_gc is not None:
            print(f"gc {measured_gc:.2f}% is {abs(margin):.2f} pp below "
                  f"baseline {expected_gc}%.")
        print()
        print("IMPLICATION: SDPNAL+ is not recovering MOSEK-class tightness "
              "on this problem. The solver-class thesis is incorrect.")
        print()
        print("PIVOT OPTIONS (in order of ROI):")
        print("  1. Hybrid cascade + local SOS certificate")
        print("     — leverages CS μ* already proving 1.2802")
        print("  2. Minimax dual via Sion (lasserre_colgen.py is stubbed)")
        print("  3. TSSOS term sparsity (lasserre_tssos.jl is stubbed)")
        return 2

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Interpret SDPNAL+ driver result JSON.')
    parser.add_argument('path', nargs='?', default=None,
                        help='Path to result_*.json.  If omitted, use --auto-find.')
    parser.add_argument('--auto-find', action='store_true',
                        help='Scan data/sdpnal_*/ for the most recent result.')
    parser.add_argument('--d', type=int, default=None,
                        help='(with --auto-find) filter by d')
    parser.add_argument('--order', type=int, default=None,
                        help='(with --auto-find) filter by order')
    args = parser.parse_args()

    path = args.path
    if path is None and args.auto_find:
        path = _find_most_recent()
        if path is None:
            print("No result_*.json found under data/sdpnal_*/.")
            return 3
        if not _filter_by_d_order(path, args.d, args.order):
            # Walk alternatives ordered by mtime.
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..'))
            all_results = glob.glob(os.path.join(
                repo_root, 'data', 'sdpnal_*', 'result_*.json'))
            all_results.sort(key=os.path.getmtime, reverse=True)
            path = None
            for r in all_results:
                if _filter_by_d_order(r, args.d, args.order):
                    path = r
                    break
            if path is None:
                print(f"No result matching d={args.d} order={args.order} "
                      f"found under data/sdpnal_*/.")
                return 3
    if path is None:
        parser.error("either pass a path or use --auto-find")

    if not os.path.exists(path):
        print(f"ERROR: {path} does not exist")
        return 3

    return interpret(path)


if __name__ == '__main__':
    sys.exit(main())
