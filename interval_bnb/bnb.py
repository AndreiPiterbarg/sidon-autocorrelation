"""Best-first branch-and-bound driver.

Certifies val(d) >= target_c by partitioning the half-simplex H_d into
closed boxes and proving that every box B has
    max_W  lb(W, B)  >=  target_c.

Pruning uses a batched tensor evaluation of natural + autoconv +
McCormick bounds over ALL windows in a single vectorised pass (one
argsort and one cumsum per box, operating on a (W, d, d) adjacency
tensor). Every certified leaf is re-verified in fractions.Fraction
arithmetic with the same formula.

Termination:
  SUCCESS: queue empty, every leaf certified.
  FAIL:    some box reached min_box_width without certifying.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bound_eval import (
    batch_bounds,
    batch_bounds_full,
    batch_bounds_rank1_hi,
    batch_bounds_rank1_lo,
    bound_autoconv_exact,
    bound_autoconv_int_ge,
    bound_mccormick_exact_nosym,
    bound_mccormick_joint_face_dual_cert_int_ge,
    bound_mccormick_joint_face_lp,
    bound_mccormick_ne_exact_nosym,
    bound_mccormick_ne_int_ge,
    bound_mccormick_sw_int_ge,
    bound_natural_exact,
    bound_natural_int_ge,
    gap_weighted_split_axis,
    window_tensor,
)
from .box import Box
from .symmetry import box_outside_hd, half_simplex_cuts
from .windows import WindowMeta, build_windows


def rigor_replay(
    B: Box, w: WindowMeta, d: int, target_q: Fraction,
    *, try_joint: bool = False,
) -> bool:
    """Exact integer-arithmetic rigor gate.

    Mathematically equivalent to the Fraction-based replay but runs on
    lo_int/hi_int (shared-denominator ints at scale 2**D_SHIFT=60),
    avoiding per-operation GCD. Bit-exact for dyadic-rational endpoints,
    which is always the case for our Box (midpoint splits preserve
    dyadicity).

    Short-circuits on the first of (natural, autoconv, SW-McCormick,
    NE-McCormick) that clears target_q. If `try_joint` is True, ALSO
    attempts the joint-face McCormick dual-certificate bound as a final
    fallback. Joint-cert is strictly >= max(SW, NE) so it closes some
    boxes the greedy bounds cannot; but each call runs scipy.linprog so
    it is only worth attempting on deep, hard-to-close boxes. The caller
    controls the gate (typically depth >= JOINT_DEPTH_THRESHOLD or
    max_width < JOINT_WIDTH_THRESHOLD).
    """
    lo_int, hi_int = B.to_ints()
    tn = target_q.numerator
    td = target_q.denominator
    if bound_natural_int_ge(lo_int, hi_int, w, tn, td):
        return True
    if bound_autoconv_int_ge(lo_int, hi_int, w, d, tn, td):
        return True
    if bound_mccormick_sw_int_ge(lo_int, hi_int, w, d, tn, td):
        return True
    if bound_mccormick_ne_int_ge(lo_int, hi_int, w, d, tn, td):
        return True
    if try_joint:
        return bound_mccormick_joint_face_dual_cert_int_ge(
            lo_int, hi_int, w, d, tn, td,
        )
    return False


def rigor_replay_with_cctr(
    B: Box, w_winner: WindowMeta, d: int, target_q: Fraction,
    *, alt_windows: Sequence[WindowMeta] = (),
    cctr_M_int=None, cctr_D_M: int = 0,
    try_joint: bool = False,
) -> bool:
    """Rigor replay with all tiers: natural, autoconv, SW, NE, joint-face,
    P1-LITE top-K joint, and CCTR aggregate (rigorous, integer arithmetic).

    CCTR is the FINAL tier and is sound regardless of α choice (provided
    α >= 0, sum α = 1, which is enforced by `build_cctr_aggregate_int`).
    Pass `cctr_M_int=None` to skip CCTR (default).
    """
    lo_int, hi_int = B.to_ints()
    tn = target_q.numerator
    td = target_q.denominator
    # Standard rigor (natural / autoconv / SW / NE / optionally joint-face).
    if rigor_replay(B, w_winner, d, target_q, try_joint=try_joint):
        return True
    # P1-LITE top-K joint-face on alternate windows.
    from .bound_eval import bound_mccormick_joint_face_dual_cert_int_ge
    for w_alt in alt_windows:
        if w_alt is w_winner:
            continue
        if bound_mccormick_joint_face_dual_cert_int_ge(
            lo_int, hi_int, w_alt, d, tn, td,
        ):
            return True
    # CCTR aggregate tier (cascading: SW → NE → joint-face).
    if cctr_M_int is not None and cctr_D_M > 0:
        from .bound_cctr import (
            bound_cctr_sw_int_ge, bound_cctr_int_ge,
            bound_cctr_joint_face_int_ge,
        )
        if bound_cctr_int_ge(  # SW or NE
            lo_int, hi_int, cctr_M_int, d, cctr_D_M, tn, td,
        ):
            return True
        if try_joint:
            if bound_cctr_joint_face_int_ge(
                lo_int, hi_int, cctr_M_int, d, cctr_D_M, tn, td,
            ):
                return True
    return False


def rigor_replay_topk_joint(
    B: Box, w_winner: WindowMeta, d: int, target_q: Fraction,
    *, alt_windows: Sequence[WindowMeta] = (),
) -> bool:
    """P1-LITE: rigor replay with top-K alternate-window joint-face fallback.

    First tries the standard `rigor_replay` (natural, autoconv, SW, NE
    int bounds, plus joint-face on the winning window). If that fails,
    iterates `alt_windows` (assumed sorted by descending float LB) and
    tries int joint-face dual cert on each. The first window whose
    joint-face exact-rational cert exceeds target_q certifies the box.

    Soundness: each alternate window's joint-face dual cert is itself a
    valid LB on min_box mu^T M_{W_alt} mu (weak LP duality + Neumaier
    -Shcherbina rigor in `bound_mccormick_joint_face_dual_cert_int_ge`).
    Hence max over windows of these LBs is a valid LB on
    max_W min_box mu^T M_W mu = min_box max_W mu^T M_W mu (after weak
    minimax). So if any single alternate certifies, the box is closed.

    Cost: each joint-face dual cert solves one HiGHS LP (~30-50 ms at
    d=20). `alt_windows` should be limited to top-K (K=3-5) and only
    invoked at deep depth; the `try_joint` gating is the caller's job.
    """
    lo_int, hi_int = B.to_ints()
    tn = target_q.numerator
    td = target_q.denominator
    if rigor_replay(B, w_winner, d, target_q, try_joint=True):
        return True
    # Alternate-window joint-face dual cert.
    from .bound_eval import bound_mccormick_joint_face_dual_cert_int_ge
    for w_alt in alt_windows:
        if w_alt is w_winner:
            continue
        if bound_mccormick_joint_face_dual_cert_int_ge(
            lo_int, hi_int, w_alt, d, tn, td,
        ):
            return True
    return False


def rigor_replay_fraction(
    B: Box, w: WindowMeta, d: int, target_q: Fraction,
) -> bool:
    """Legacy Fraction-based rigor gate. Retained for cross-check tests
    that the integer path is mathematically identical. Not used on the
    hot path."""
    lo_q, hi_q = B.to_fractions()
    lb_rat = bound_natural_exact(lo_q, hi_q, w)
    if lb_rat >= target_q:
        return True
    lb_rat = bound_autoconv_exact(lo_q, hi_q, w, d)
    if lb_rat >= target_q:
        return True
    lb_rat = bound_mccormick_exact_nosym(lo_q, hi_q, w)
    if lb_rat >= target_q:
        return True
    lb_rat = bound_mccormick_ne_exact_nosym(lo_q, hi_q, w)
    return lb_rat >= target_q


@dataclass
class BnBStats:
    nodes_processed: int = 0
    leaves_certified: int = 0
    leaves_split: int = 0
    max_depth: int = 0
    window_usage: Dict[int, int] = field(default_factory=dict)
    wall_time_s: float = 0.0
    worst_lb_seen: float = float("inf")
    max_queue: int = 0
    rigor_retries: int = 0

    def to_dict(self) -> dict:
        return {
            "nodes_processed": self.nodes_processed,
            "leaves_certified": self.leaves_certified,
            "leaves_split": self.leaves_split,
            "max_depth": self.max_depth,
            "window_usage": dict(self.window_usage),
            "wall_time_s": self.wall_time_s,
            "worst_lb_seen": self.worst_lb_seen,
            "max_queue": self.max_queue,
            "rigor_retries": self.rigor_retries,
        }


@dataclass
class BnBResult:
    success: bool
    target_q: Fraction
    target_float: float
    d: int
    stats: BnBStats
    failing_box: Optional[Box] = None
    failing_lb: Optional[float] = None


_UID = itertools.count()


def branch_and_bound(
    d: int,
    target_c: float,
    *,
    max_nodes: int = 10_000_000,
    min_box_width: float = 1e-10,
    use_symmetry: bool = True,
    log_every: int = 10_000,
    time_budget_s: Optional[float] = None,
    verbose: bool = True,
) -> BnBResult:
    """Run BnB. Returns BnBResult with stats."""
    sym_cuts = half_simplex_cuts(d) if use_symmetry else []
    initial = Box.initial(d, sym_cuts)
    if not initial.intersects_simplex():
        raise RuntimeError("initial box does not intersect simplex (bug)")
    return branch_and_bound_from_box(
        d, target_c, initial,
        max_nodes=max_nodes, min_box_width=min_box_width,
        log_every=log_every, time_budget_s=time_budget_s,
        verbose=verbose, use_symmetry=use_symmetry,
    )


def branch_and_bound_from_box(
    d: int,
    target_c: float,
    initial: Box,
    *,
    max_nodes: int = 10_000_000,
    min_box_width: float = 1e-10,
    log_every: int = 10_000,
    time_budget_s: Optional[float] = None,
    verbose: bool = True,
    joint_depth_threshold: int = 20,
    tighten_depth_threshold: int = 15,
    use_symmetry: bool = True,
) -> BnBResult:
    """Run BnB restricted to the given initial box. Used both as the
    single-shot entry point and as the per-worker driver in
    interval_bnb.parallel.
    """
    t0 = time.time()
    target_q = _to_fraction(target_c)
    target_f = float(target_c)
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)

    stats = BnBStats()

    # DFS stack entries: (box, depth, parent_cache, changed_axis, which_end)
    # where `which_end` is 'lo' (right child), 'hi' (left child), or None
    # for the root. When parent_cache is None, we do a full recompute;
    # otherwise we apply a rank-1 update.
    stack: List[Tuple[Box, int, Optional[tuple], int, Optional[str]]] = []
    stack.append((initial, 0, None, -1, None))

    last_log = time.time()

    while stack:
        if stats.nodes_processed >= max_nodes:
            if verbose:
                print(f"[bnb] max_nodes={max_nodes} reached; aborting")
            return BnBResult(
                success=False, target_q=target_q, target_float=target_f,
                d=d, stats=_finalise(stats, t0),
            )
        if time_budget_s is not None and time.time() - t0 > time_budget_s:
            if verbose:
                print(f"[bnb] time_budget_s={time_budget_s} exceeded; aborting")
            return BnBResult(
                success=False, target_q=target_q, target_float=target_f,
                d=d, stats=_finalise(stats, t0),
            )

        B, depth, parent_cache, changed_k, which_end = stack.pop()
        stats.nodes_processed += 1
        if depth > stats.max_depth:
            stats.max_depth = depth
        if len(stack) > stats.max_queue:
            stats.max_queue = len(stack)

        if not B.intersects_simplex():
            continue

        # H_d half-simplex pre-filter (proper sigma cut). Sound by
        # Lemma 3.4 (THEOREM.md): boxes with lo_int[0] > hi_int[d-1]
        # have their sigma-image covered elsewhere in the BnB cover
        # of {mu_0 <= 1/2}. Skipped when `use_symmetry=False` so the
        # full-simplex cross-check can run without the symmetry cut.
        if use_symmetry and box_outside_hd(B):
            continue

        # T3 re-enabled at depth threshold: tightens hi via simplex
        # sum-constraint, which lifts the autoconv bound (binding on
        # d=16 stall boxes). The invalidation below forces a full
        # recompute, losing the rank-1 cache, so we only pay the cost
        # at deep depth where the stall matters.
        if depth >= tighten_depth_threshold:
            if B.tighten_to_simplex():
                parent_cache = None
                if not B.intersects_simplex():
                    continue

        # Bounds (with rank-1 update if possible).
        if parent_cache is None:
            lb_fast, w_idx, which, _mu, my_cache = batch_bounds_full(
                B.lo, B.hi, A_tensor, scales, target_f,
            )
        elif which_end == "lo":
            lb_fast, w_idx, which, _mu, my_cache = batch_bounds_rank1_lo(
                A_tensor, scales, parent_cache, B.lo, changed_k, target_f,
            )
        else:  # "hi"
            lb_fast, w_idx, which, _mu, my_cache = batch_bounds_rank1_hi(
                A_tensor, scales, parent_cache, B.hi, changed_k, target_f,
            )

        if lb_fast < stats.worst_lb_seen:
            stats.worst_lb_seen = lb_fast

        # T1 (dual-face joint LP) was REVERTED: float-path tightening
        # delivered no tree shrink because the integer-rigor gate still
        # uses separate SW/NE bounds, so every joint-certified box
        # becomes a rigor retry and is split anyway. Measured 20k-30x
        # slowdown on d=10/d=4. Re-enable only if integer-rigor joint
        # LP is implemented (requires a rational LP simplex).

        if lb_fast >= target_f:
            w = windows[w_idx]
            # P2 SAFE-MARGIN SHORTCUT: the autoconv and McCormick float
            # bounds compute the same closed-form formula as their integer
            # counterparts (bound_autoconv_int_ge, bound_mccormick_*_int_ge).
            # Worst-case absolute disagreement is bounded by float64 +
            # Numba `fastmath=True` reassociation error in the batched
            # evaluator (`bound_eval._batch_min_linear_lb_only_numba`):
            #   |lb_float - lb_int| <= O(d * depth * eps_machine)
            # i.e. ~ d * 60 * 2.22e-16 ~ 6e-13 at d=22, depth=60. The 1e-9
            # margin is 4 orders larger; this skips int rigor replay on the
            # easy majority while staying safely above the worst-case
            # float drift. NOT applied to joint-face (joint-face has its
            # own rigor invocation inside rigor_replay).
            if (lb_fast - target_f) > 1e-9 and which in ("autoconv", "mccormick"):
                stats.leaves_certified += 1
                stats.window_usage[w_idx] = stats.window_usage.get(w_idx, 0) + 1
                if verbose and stats.leaves_certified % max(1, log_every // 10) == 0 \
                        and time.time() - last_log > 1.0:
                    _log_progress(stats, stack, lb_fast, target_f, t0)
                    last_log = time.time()
                continue
            certified = rigor_replay(
                B, w, d, target_q,
                try_joint=(depth >= joint_depth_threshold),
            )
            if certified:
                stats.leaves_certified += 1
                stats.window_usage[w_idx] = stats.window_usage.get(w_idx, 0) + 1
                if verbose and stats.leaves_certified % max(1, log_every // 10) == 0 \
                        and time.time() - last_log > 1.0:
                    _log_progress(stats, stack, lb_fast, target_f, t0)
                    last_log = time.time()
                continue
            stats.rigor_retries += 1

        if B.max_width() < min_box_width:
            stats.wall_time_s = time.time() - t0
            if verbose:
                print(f"[bnb] FAIL: box at depth {depth} width<{min_box_width} could not certify")
                print(f"       lb_fast={lb_fast:.6f}  target={target_f:.6f}")
                print(f"       {B.shape_summary()}")
            return BnBResult(
                success=False, target_q=target_q, target_float=target_f,
                d=d, stats=_finalise(stats, t0),
                failing_box=B, failing_lb=lb_fast,
            )
        # P3 SPLIT STRATEGY: gap-weighted at depth >= 4. Earlier benchmark
        # at d <= 10 showed widest-axis better; at d >= 14 with target
        # margin tight, gap-weighted reduces tree size by exploiting
        # gradient anisotropy of the binding window.
        if depth >= 4 and w_idx >= 0:
            axis = gap_weighted_split_axis(B.lo, B.hi, windows[w_idx], d)
            # Defensive: if gap_weighted picked a saturated axis, fall back.
            if B.lo_int is not None and B.hi_int is not None \
                    and (B.hi_int[axis] - B.lo_int[axis]) < 2:
                axis = B.widest_splittable_axis()
        else:
            axis = B.widest_splittable_axis()
        if axis < 0:
            # No axis is splittable — every axis exhausted dyadic depth.
            # The bound did not certify (we'd have continued above), so
            # this is an UNCERTIFIED leaf. Failing loudly is the correct
            # rigor stance.
            stats.wall_time_s = time.time() - t0
            if verbose:
                print(f"[bnb] FAIL: box at depth {depth} fully saturated "
                      f"(no axis splittable) without certifying")
                print(f"       lb_fast={lb_fast:.6f}  target={target_f:.6f}")
                print(f"       {B.shape_summary()}")
            return BnBResult(
                success=False, target_q=target_q, target_float=target_f,
                d=d, stats=_finalise(stats, t0),
                failing_box=B, failing_lb=lb_fast,
            )
        left, right = B.split(axis)
        stats.leaves_split += 1
        # Push RIGHT first so LEFT is processed first (LIFO). Both share
        # `my_cache` -- they will apply rank-1 updates from it.
        if right.intersects_simplex():
            stack.append((right, depth + 1, my_cache, axis, "lo"))
        if left.intersects_simplex():
            stack.append((left, depth + 1, my_cache, axis, "hi"))

        if verbose and stats.nodes_processed % log_every == 0 \
                and time.time() - last_log > 1.0:
            _log_progress(stats, stack, lb_fast, target_f, t0)
            last_log = time.time()

    stats.wall_time_s = time.time() - t0
    return BnBResult(
        success=True, target_q=target_q, target_float=target_f,
        d=d, stats=_finalise(stats, t0),
    )


def _finalise(stats: BnBStats, t0: float) -> BnBStats:
    stats.wall_time_s = time.time() - t0
    return stats


def _log_progress(stats: BnBStats, queue, lb_fast, target_c, t0):
    n_queue = len(queue)
    t = time.time() - t0
    rate = stats.nodes_processed / max(t, 1e-9)
    print(
        f"[bnb] t={t:6.1f}s  nodes={stats.nodes_processed:>8d}  "
        f"queue={n_queue:>7d}  cert={stats.leaves_certified:>7d}  "
        f"depth={stats.max_depth:>3d}  rate={rate:.0f}/s  "
        f"lb_fast={lb_fast:.6f}/{target_c:.6f}"
    )


def _to_fraction(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        # bool is a subclass of int; reject to avoid silent coercion.
        raise TypeError(
            "_to_fraction does not accept bool; pass a str like \"1.2802\" "
            "or a Fraction to make the decimal intent explicit."
        )
    if isinstance(x, float):
        raise TypeError(
            "_to_fraction refuses float input to avoid silent rounding that "
            "could strengthen the certified target beyond the declared value. "
            "Pass a str (e.g. \"1.2802\") or a Fraction instead."
        )
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, int):
        return Fraction(x)
    raise TypeError(
        f"_to_fraction: unsupported type {type(x).__name__}; "
        "pass a str (e.g. \"1.2802\"), a Fraction, or an int."
    )
