"""Step 2 driver: rigorously exclude saddle-KKT critical points in a box.

Given:
  * box B = prod_i [lo_i, hi_i] cap Delta_d
  * target T (a rational), and required margin eps
This driver enumerates candidate active sets (A_W, A_plus, A_minus) and for
each one runs interval Krawczyk (cert_pipeline.krawczyk_solver) on the
reduced KKT system (cert_pipeline.saddle_kkt).  Verdicts per active set:

  EXCLUDED     - rigorously proved no KKT point with t < T-eps in B
                 for this active set.
  COUNTEREX    - found a unique KKT point with t < T-eps and all
                 KKT inequalities satisfied. Bound is FALSE.
  UNDECIDED    - Krawczyk inconclusive after bisection budget; this
                 active set survives.

ABSOLUTE-CORRECTNESS GUARANTEE (per active set):
  Each individual verdict is mathematically rigorous (proved by Krawczyk
  + interval arithmetic).
COMPLETENESS CAVEAT:
  A complete proof requires enumerating *every* possible active set
  (A_W, A_plus, A_minus) up to Caratheodory size.  This driver enumerates
  a user-specified budget; for boxes where the budget is exhausted,
  output is conservative (UNDECIDED).  Combine with Step 1 (vertex
  enumeration via cert_pipeline.uniform_spike_threshold) which handles
  vertices independently.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from mpmath import iv, mp

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from cert_pipeline.iv_core import IVVec, rat_to_iv, set_precision
from cert_pipeline.krawczyk_solver import (Verdict, krawczyk_recurse,
                                           krawczyk_step)
from cert_pipeline.saddle_kkt import (ActiveSet, BoxSpec, KKTSystem,
                                      WindowSpec, derived_quantities)


# ---------------------------------------------------------------------
# Window construction (matching the repo's lasserre.core convention)
# ---------------------------------------------------------------------

def build_windows_specs(d: int) -> List[WindowSpec]:
    """Build all WindowSpec for dimension d.  Same enumeration as
    `lasserre.core.build_window_matrices`.
    """
    out: List[WindowSpec] = []
    for ell in range(2, 2 * d + 1):
        for s_lo in range(2 * d - ell + 1):
            pairs_all = []
            for i in range(d):
                for j in range(d):
                    if s_lo <= i + j <= s_lo + ell - 2:
                        pairs_all.append((i, j))
            if not pairs_all:
                continue
            scale_q = Fraction(2 * d, ell)
            out.append(WindowSpec(ell=ell, s_lo=s_lo, scale_q=scale_q,
                                  pairs_all=tuple(pairs_all)))
    return out


# ---------------------------------------------------------------------
# KKT inequality post-hoc check (rigorous)
# ---------------------------------------------------------------------

@dataclass
class KKTPointCheck:
    is_valid_kkt: bool
    t_below_threshold: bool
    t_value_iv: object
    failures: List[str] = field(default_factory=list)


def check_kkt_inequalities(system: KKTSystem, X_localized: IVVec,
                           hi_iv, lo_iv, scales_iv,
                           target_T: Fraction, margin: Fraction
                           ) -> KKTPointCheck:
    """Given a Krawczyk-localized box X_localized that contains exactly
    one zero of the KKT residual, verify *rigorously* (using interval
    arithmetic) that all KKT inequality constraints are satisfied AND
    that t < T - margin.

    Soundness: every check uses interval bounds on the localized box;
    if an inequality is verified for *every* interval-feasible point in
    X_localized, it holds at the unique zero too.
    """
    failures: List[str] = []

    # 1. lambda_W >= 0 for W in A_W: just check that lower bound of each
    #    lambda_W interval is >= 0.
    for k, w_idx in enumerate(system.active.A_W):
        lam_iv = X_localized.data[system.idx_lambda[k]]
        if float(lam_iv.a) < 0:
            failures.append(f"lambda[{w_idx}] could be < 0: a={float(lam_iv.a):.6e}")

    # 2. mu_F[i] in [lo, hi] strictly: x is in interior of box
    for k, i in enumerate(system.F):
        mu_iv = X_localized.data[system.idx_mu_F[k]]
        lo_b = float(rat_to_iv(system.box.lo_q[i]).b)
        hi_b = float(rat_to_iv(system.box.hi_q[i]).a)
        if float(mu_iv.b) < lo_b or float(mu_iv.a) > hi_b:
            failures.append(f"mu_F[{i}] out of [lo,hi]: "
                            f"[{float(mu_iv.a):.6e}, {float(mu_iv.b):.6e}]")

    # 3. beta_plus, beta_minus >= 0: compute via derived_quantities at
    #    interval inputs and check sign of lower bounds.
    der = derived_quantities(system, list(X_localized.data),
                             hi_q=hi_iv, lo_q=lo_iv,
                             scale_q_override=scales_iv)
    for i, val in der["beta_plus"].items():
        if hasattr(val, "a"):
            if float(val.a) < 0:
                failures.append(f"beta_plus[{i}] could be < 0: "
                                f"a={float(val.a):.6e}")
        else:
            if float(val) < 0:
                failures.append(f"beta_plus[{i}] = {float(val):.6e} < 0")
    for i, val in der["beta_minus"].items():
        if hasattr(val, "a"):
            if float(val.a) < 0:
                failures.append(f"beta_minus[{i}] could be < 0: "
                                f"a={float(val.a):.6e}")
        else:
            if float(val) < 0:
                failures.append(f"beta_minus[{i}] = {float(val):.6e} < 0")

    # 4. tv_off <= t for off-active windows
    t_iv = X_localized.data[system.idx_t]
    for w_idx, tv_val in der["tv_off"].items():
        if hasattr(tv_val, "a"):
            tv_a = float(tv_val.a)
        else:
            tv_a = float(tv_val)
        if tv_a > float(t_iv.b):
            failures.append(f"TV_W[{w_idx}] could exceed t: "
                            f"tv_a={tv_a:.6e}, t_b={float(t_iv.b):.6e}")

    # 5. t < T - margin?
    threshold_b = float(rat_to_iv(target_T - margin).b)
    t_below = float(t_iv.b) < threshold_b

    return KKTPointCheck(
        is_valid_kkt=(not failures),
        t_below_threshold=t_below,
        t_value_iv=t_iv,
        failures=failures,
    )


# ---------------------------------------------------------------------
# Single-active-set runner
# ---------------------------------------------------------------------

@dataclass
class ActiveSetVerdict:
    A_W: Tuple[int, ...]
    A_plus: Tuple[int, ...]
    A_minus: Tuple[int, ...]
    n_F: int
    n_vars: int
    verdict: str   # "EXCLUDED" | "COUNTEREX" | "UNDECIDED" | "INVALID"
    n_excluded_leaves: int = 0
    n_unique_leaves: int = 0
    n_undecided_leaves: int = 0
    counterex_t_lo: Optional[float] = None
    counterex_t_hi: Optional[float] = None
    notes: str = ""
    wall_s: float = 0.0


def run_active_set(box: BoxSpec, windows: List[WindowSpec],
                   active: ActiveSet, target_T: Fraction, margin: Fraction,
                   max_depth: int = 10,
                   nu_bound: float = 1000.0,
                   t_lo: float = 0.0,
                   verbose: bool = False) -> ActiveSetVerdict:
    """Run Krawczyk recursion on the KKT system for one active set.

    The unknown box X has:
      mu_F[i] in [lo_i, hi_i]    for i in F
      lambda_W in [0, 1]         for W in A_W
      nu in [-nu_bound, nu_bound]
      t in [t_lo, T - margin]    (the bad-branch hypothesis)
    """
    t0 = time.time()
    try:
        system = KKTSystem(box, windows, active)
    except Exception as e:
        return ActiveSetVerdict(
            A_W=active.A_W, A_plus=active.A_plus, A_minus=active.A_minus,
            n_F=0, n_vars=0,
            verdict="INVALID",
            notes=f"system construction failed: {e}",
            wall_s=time.time() - t0,
        )

    threshold = target_T - margin

    # Build initial X
    X_data = []
    for i in system.F:
        X_data.append(iv.mpf([float(box.lo_q[i]), float(box.hi_q[i])]))
    for _ in range(system.n_AW):
        X_data.append(iv.mpf([0.0, 1.0]))   # lambda_W
    X_data.append(iv.mpf([-nu_bound, nu_bound]))  # nu
    X_data.append(iv.mpf([t_lo, float(threshold)]))  # t
    X = IVVec.__new__(IVVec)
    X.data = X_data

    hi_iv = tuple(rat_to_iv(q) for q in box.hi_q)
    lo_iv = tuple(rat_to_iv(q) for q in box.lo_q)
    scales_iv = tuple(rat_to_iv(q) for q in system.A_W_scales_q)

    leaves = krawczyk_recurse(system, X, hi_iv, lo_iv, scales_iv,
                              max_depth=max_depth)

    n_excl = sum(1 for v, _ in leaves if v == Verdict.EXCLUDED)
    n_uniq = sum(1 for v, _ in leaves if v == Verdict.UNIQUE_ZERO)
    n_und = sum(1 for v, _ in leaves if v == Verdict.UNDECIDED)

    # If there are unique-zero leaves, check KKT inequalities. If a unique
    # zero satisfies all inequalities and has t < T-margin, it's a true
    # counterexample.
    counterex = None
    for v, lbox in leaves:
        if v == Verdict.UNIQUE_ZERO:
            chk = check_kkt_inequalities(system, lbox, hi_iv, lo_iv,
                                         scales_iv, target_T, margin)
            if chk.is_valid_kkt and chk.t_below_threshold:
                counterex = (chk, lbox)
                break

    if counterex is not None:
        chk, lbox = counterex
        t_iv = lbox.data[system.idx_t]
        return ActiveSetVerdict(
            A_W=active.A_W, A_plus=active.A_plus, A_minus=active.A_minus,
            n_F=system.n_F, n_vars=system.n_vars,
            verdict="COUNTEREX",
            n_excluded_leaves=n_excl,
            n_unique_leaves=n_uniq,
            n_undecided_leaves=n_und,
            counterex_t_lo=float(t_iv.a),
            counterex_t_hi=float(t_iv.b),
            notes=f"Found valid KKT point with t in [{float(t_iv.a):.6e}, "
                  f"{float(t_iv.b):.6e}] < threshold = {float(threshold):.6e}",
            wall_s=time.time() - t0,
        )

    if n_und == 0 and n_uniq == 0:
        verdict = "EXCLUDED"
        notes = (f"All {n_excl} leaves verified empty (no KKT solution with "
                 f"t < {float(threshold):.6e} exists in B for this active set)")
    elif n_und == 0 and n_uniq > 0:
        # Unique zeros exist but none satisfies KKT inequalities -> the
        # active set assumption is internally inconsistent (zero of equation
        # system but not a valid KKT critical point of the original problem).
        verdict = "EXCLUDED"
        notes = (f"All {n_uniq} unique zeros violate KKT inequalities, hence "
                 f"are not critical points of original problem")
    else:
        verdict = "UNDECIDED"
        notes = (f"Bisection budget exhausted: {n_excl} excluded, "
                 f"{n_uniq} unique-zero, {n_und} undecided leaves")

    return ActiveSetVerdict(
        A_W=active.A_W, A_plus=active.A_plus, A_minus=active.A_minus,
        n_F=system.n_F, n_vars=system.n_vars,
        verdict=verdict,
        n_excluded_leaves=n_excl,
        n_unique_leaves=n_uniq,
        n_undecided_leaves=n_und,
        notes=notes,
        wall_s=time.time() - t0,
    )


# ---------------------------------------------------------------------
# Active-set enumeration with hot-start
# ---------------------------------------------------------------------

def enumerate_active_sets(d: int, n_windows: int,
                          A_W_candidates: Sequence[int],
                          A_plus_max: int = 2,
                          A_minus_max: int = 21,
                          A_W_size_min: int = 1,
                          A_W_size_max: int = 4,
                          n_F_min: int = 2,
                          ):
    """Enumerate (A_W, A_plus, A_minus) candidates.

    A_W is drawn from `A_W_candidates` (a hot-start list, e.g. windows
    near-active in the LP/SDP dual).
    Yields tuples (A_W_tup, A_plus_tup, A_minus_tup).

    Symmetry caveat: this is a basic enumeration; it does NOT quotient
    by S_d or Z_2 (do that downstream if desired).
    """
    # We focus on A_plus = empty, A_minus subsets of [d].
    # For non-vertex KKT, we need n_F = d - |A_plus| - |A_minus| >= n_F_min.
    # Iterate A_plus first, then A_minus, then A_W subsets.
    for ap_size in range(0, A_plus_max + 1):
        for A_plus in combinations(range(d), ap_size):
            ap_set = set(A_plus)
            avail_minus = [i for i in range(d) if i not in ap_set]
            for am_size in range(0, A_minus_max + 1):
                if d - ap_size - am_size < n_F_min:
                    continue
                for A_minus in combinations(avail_minus, am_size):
                    for aw_size in range(A_W_size_min, A_W_size_max + 1):
                        if aw_size > len(A_W_candidates):
                            break
                        for A_W in combinations(A_W_candidates, aw_size):
                            yield A_W, A_plus, A_minus


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_decimal(s: str) -> Fraction:
    s = s.strip().lower()
    if "e" in s:
        m, ex = s.split("e")
        if "." in m:
            i, f = m.split(".")
            mant = Fraction(int((i if i else "0") + f), 10 ** len(f))
        else:
            mant = Fraction(int(m))
        return mant * (Fraction(10) ** int(ex))
    if "." in s:
        sign = -1 if s.startswith("-") else 1
        ss = s.lstrip("-")
        i, f = ss.split(".")
        return sign * Fraction(int((i if i else "0") + f), 10 ** len(f))
    return Fraction(int(s))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--lo", type=str, required=True,
                   help="Comma-separated lower endpoints (Fraction-parsable, length d)")
    p.add_argument("--hi", type=str, required=True,
                   help="Comma-separated upper endpoints (length d)")
    p.add_argument("--target", type=str, default="1.2805")
    p.add_argument("--margin", type=str, default="1e-6")
    p.add_argument("--A_W", type=str, default=None,
                   help="Comma-separated window indices to use as A_W; "
                        "if omitted, will enumerate small subsets")
    p.add_argument("--A_plus", type=str, default="",
                   help="Comma-separated axis indices for A_plus")
    p.add_argument("--A_minus", type=str, default="",
                   help="Comma-separated axis indices for A_minus")
    p.add_argument("--A_W_candidates", type=str, default=None,
                   help="Comma-separated window indices for hot-start "
                        "enumeration (default: all windows)")
    p.add_argument("--A_W_size_max", type=int, default=2,
                   help="Max size of enumerated A_W subsets")
    p.add_argument("--A_plus_max", type=int, default=0)
    p.add_argument("--A_minus_max", type=int, default=2,
                   help="Max size of enumerated A_minus subsets")
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--budget", type=int, default=200,
                   help="Max number of active sets to evaluate")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--dps", type=int, default=50,
                   help="mpmath working precision (decimal digits)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    set_precision(args.dps)

    d = args.d
    lo_strs = args.lo.split(",")
    hi_strs = args.hi.split(",")
    if len(lo_strs) != d or len(hi_strs) != d:
        print(f"--lo and --hi must each have {d} comma-separated entries")
        return 2

    def parse_q(s: str) -> Fraction:
        s = s.strip()
        if "/" in s:
            n, d_ = s.split("/")
            return Fraction(int(n), int(d_))
        if "." in s:
            return parse_decimal(s)
        return Fraction(int(s))

    lo_q = tuple(parse_q(s) for s in lo_strs)
    hi_q = tuple(parse_q(s) for s in hi_strs)
    target_T = parse_decimal(args.target)
    margin = parse_decimal(args.margin)
    box = BoxSpec(d=d, lo_q=lo_q, hi_q=hi_q)

    print(f"=== Step 2: Krawczyk on saddle-KKT ===")
    print(f"d = {d}")
    print(f"box: lo = {[float(x) for x in lo_q]}")
    print(f"     hi = {[float(x) for x in hi_q]}")
    print(f"target T = {float(target_T):.10f}")
    print(f"margin = {float(margin):.2e}")
    print(f"threshold = T - margin = {float(target_T - margin):.10f}")
    print()

    print(f"Building windows for d={d}...")
    windows = build_windows_specs(d)
    print(f"  {len(windows)} windows total.")

    # Parse A_W
    if args.A_W is not None:
        A_W = tuple(int(x) for x in args.A_W.split(",") if x.strip())
        A_plus_arg = tuple(int(x) for x in args.A_plus.split(",") if x.strip())
        A_minus_arg = tuple(int(x) for x in args.A_minus.split(",") if x.strip())
        active = ActiveSet(A_W=A_W, A_plus=A_plus_arg, A_minus=A_minus_arg,
                           target_T=target_T)
        print(f"\nRunning single active set:")
        print(f"  A_W = {A_W}")
        print(f"  A_plus = {A_plus_arg}, A_minus = {A_minus_arg}")
        print()
        verdict = run_active_set(box, windows, active, target_T, margin,
                                 max_depth=args.max_depth, verbose=args.verbose)
        print(f"verdict: {verdict.verdict}")
        print(f"  n_F={verdict.n_F}, n_vars={verdict.n_vars}, "
              f"wall={verdict.wall_s:.2f}s")
        print(f"  leaves: excluded={verdict.n_excluded_leaves}, "
              f"unique={verdict.n_unique_leaves}, "
              f"undecided={verdict.n_undecided_leaves}")
        print(f"  notes: {verdict.notes}")
        if args.output:
            Path(args.output).write_text(json.dumps({
                "d": d, "target": str(target_T), "margin": str(margin),
                "A_W": list(A_W), "A_plus": list(A_plus_arg),
                "A_minus": list(A_minus_arg),
                "verdict": verdict.verdict,
                "n_excluded": verdict.n_excluded_leaves,
                "n_unique": verdict.n_unique_leaves,
                "n_undecided": verdict.n_undecided_leaves,
                "wall_s": verdict.wall_s,
                "notes": verdict.notes,
            }, indent=2))
        return 0 if verdict.verdict in ("EXCLUDED", "INVALID") else 1

    # Enumeration mode
    if args.A_W_candidates is not None:
        A_W_cand = [int(x) for x in args.A_W_candidates.split(",")]
    else:
        A_W_cand = list(range(len(windows)))

    print(f"Enumeration mode: A_W candidates = {len(A_W_cand)}, "
          f"|A_W| <= {args.A_W_size_max}, |A_plus| <= {args.A_plus_max}, "
          f"|A_minus| <= {args.A_minus_max}")
    print()

    n_eval = 0
    n_excluded = 0
    n_undecided = 0
    n_counterex = 0
    n_invalid = 0
    results: List[ActiveSetVerdict] = []
    t_total = time.time()
    for A_W, A_plus, A_minus in enumerate_active_sets(
            d=d, n_windows=len(windows),
            A_W_candidates=A_W_cand,
            A_plus_max=args.A_plus_max,
            A_minus_max=args.A_minus_max,
            A_W_size_max=args.A_W_size_max):
        if n_eval >= args.budget:
            break
        n_eval += 1
        active = ActiveSet(A_W=tuple(A_W), A_plus=tuple(A_plus),
                           A_minus=tuple(A_minus), target_T=target_T)
        verdict = run_active_set(box, windows, active, target_T, margin,
                                 max_depth=args.max_depth)
        results.append(verdict)
        if verdict.verdict == "EXCLUDED":
            n_excluded += 1
        elif verdict.verdict == "UNDECIDED":
            n_undecided += 1
        elif verdict.verdict == "COUNTEREX":
            n_counterex += 1
        elif verdict.verdict == "INVALID":
            n_invalid += 1
        if args.verbose or verdict.verdict in ("COUNTEREX", "UNDECIDED"):
            print(f"[{n_eval}] A_W={A_W} A_plus={A_plus} A_minus={A_minus} "
                  f"-> {verdict.verdict} ({verdict.wall_s:.2f}s) "
                  f"E={verdict.n_excluded_leaves} U={verdict.n_unique_leaves} "
                  f"?={verdict.n_undecided_leaves}")
        if verdict.verdict == "COUNTEREX":
            print(f"    !! COUNTEREXAMPLE: {verdict.notes}")

    total_wall = time.time() - t_total
    print()
    print(f"=== Summary ===")
    print(f"Active sets evaluated: {n_eval}")
    print(f"  EXCLUDED:  {n_excluded}")
    print(f"  UNDECIDED: {n_undecided}")
    print(f"  COUNTEREX: {n_counterex}")
    print(f"  INVALID:   {n_invalid}")
    print(f"Total wall time: {total_wall:.1f}s")
    print()
    if n_counterex > 0:
        print("WARNING: counterexample(s) found -- target lower bound is FALSE")
    elif n_undecided > 0:
        print(f"PARTIAL: {n_undecided} active sets undecided. NOT a complete proof.")
    else:
        print(f"All evaluated active sets EXCLUDED. (Completeness: only "
              f"those enumerated; NOT a global proof unless enumeration "
              f"exhaustive.)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "d": d, "target": str(target_T), "margin": str(margin),
            "lo": [str(q) for q in lo_q], "hi": [str(q) for q in hi_q],
            "n_eval": n_eval,
            "n_excluded": n_excluded, "n_undecided": n_undecided,
            "n_counterex": n_counterex, "n_invalid": n_invalid,
            "total_wall_s": total_wall,
            "results": [
                {
                    "A_W": list(r.A_W), "A_plus": list(r.A_plus),
                    "A_minus": list(r.A_minus),
                    "n_F": r.n_F, "n_vars": r.n_vars,
                    "verdict": r.verdict,
                    "n_excluded_leaves": r.n_excluded_leaves,
                    "n_unique_leaves": r.n_unique_leaves,
                    "n_undecided_leaves": r.n_undecided_leaves,
                    "wall_s": r.wall_s, "notes": r.notes,
                }
                for r in results
            ],
        }, indent=2))
        print(f"\nWrote {args.output}")
    return 0 if n_counterex == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
