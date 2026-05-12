"""Work-stealing parallel branch-and-bound.

Design (see plan file `now-lets-focus-on-iridescent-lollipop.md`):

* Each worker has a PRIVATE DFS stack (fast, in-process, preserves
  rank-1 cache locality between parent and child).
* A SHARED `mp.Queue` holds unclaimed boxes. Workers pull a BATCH when
  their stack is empty and donate the SHALLOW (oldest) HALF of their
  stack when `len(stack) >= donate_trigger()`. Batched IPC amortises
  pickle overhead.
* Termination is Safra-style: a shared `in_flight` counter tracks the
  number of boxes that have been created but NOT YET CLOSED (certified
  or declared infeasible). When a box splits into two children, the
  counter increments by +1 (one input removed, two created). When a box
  certifies or is infeasible, -1. `in_flight == 0` is a one-time
  monotone event observable without races.
* `mp.get_context("fork")` lets workers inherit the pre-built
  `A_tensor` via copy-on-write on Linux.

Correctness: identical boxes get processed (just on different cores),
rigor gate (Fraction replay) is per-box and process-local. The single
"silent failure" risk -- min_box_width hit without certification -- is
broadcast via an `mp.Event` that every worker polls at every loop
iteration.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import signal
import sys
import time
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.box import Box  # noqa: E402
from interval_bnb.symmetry import box_outside_hd, half_simplex_cuts  # noqa: E402


# ---------------------------------------------------------------------
# Starter partition
# ---------------------------------------------------------------------

def _split_initial(d: int, depth: int, sym_cuts) -> List[Box]:
    """Breadth-first midpoint split of the initial (half-)simplex box
    `depth` times. Drops sub-boxes that do not intersect the simplex,
    or that lie strictly outside the half-simplex H_d (proper sigma
    cut: see interval_bnb/symmetry.py:box_outside_hd).

    The H_d cut is the proper coordinate-coupled symmetry reduction:
    for any orbit {mu, sigma(mu)}, exactly one representative survives
    (or both, if mu_0 = mu_{d-1}). This roughly halves the search.
    """
    root = Box.initial(d, sym_cuts)
    boxes: List[Box] = [root]
    apply_hd_cut = bool(sym_cuts)
    for _ in range(depth):
        new_boxes: List[Box] = []
        for B in boxes:
            ax = B.widest_axis()
            left, right = B.split(ax)
            if left.intersects_simplex() and not (apply_hd_cut and box_outside_hd(left)):
                new_boxes.append(left)
            if right.intersects_simplex() and not (apply_hd_cut and box_outside_hd(right)):
                new_boxes.append(right)
        boxes = new_boxes
    return boxes


# ---------------------------------------------------------------------
# Worker-side globals (set once per process on fork/spawn)
# ---------------------------------------------------------------------

_W_STATE: dict = {}  # filled in _worker_main on first entry


def _worker_main(
    worker_id: int,
    d: int,
    target_c,
    min_box_width: float,
    max_nodes_per_worker: int,
    pull_batch_max: int,
    donate_threshold_floor: int,
    # Shared primitives -- pickled across fork/spawn.
    queue: mp.Queue,
    in_flight: "mp.sharedctypes.Synchronized",
    idle_count: "mp.sharedctypes.Synchronized",
    cert_count: "mp.sharedctypes.Synchronized",
    node_count: "mp.sharedctypes.Synchronized",
    closed_vol: "mp.sharedctypes.Synchronized",  # volume of closed boxes
    failed_event: "mp.synchronize.Event",
    done_event: "mp.synchronize.Event",
    stats_queue: mp.Queue,
    cctr_M_int=None,  # numpy (d, d) int matrix or None — first aggregate
    cctr_D_M: int = 0,  # integer denom for cctr_M_int (0 = disabled)
    multi_cctr_M_ints=None,  # list of (d,d) int matrices, multi-α aggregates
    multi_cctr_D_Ms=None,    # parallel list of denoms
):
    """Long-lived worker: pull from shared queue, process local DFS,
    donate when stack grows large, exit when done/failed signalled."""
    # Install SIGINT handler so Ctrl-C from the master cleanly flips
    # `failed_event`. The main loop observes this on the next iteration
    # and exits via the `try/finally` cleanup. Avoid `sys.exit()` from
    # inside the signal handler — it can fire mid-`queue.put` and leave
    # internal queue locks held, deadlocking sibling workers.
    signal.signal(signal.SIGINT, lambda *_: failed_event.set())

    # Lazy imports of the bnb-internal functions so the fork happens
    # with the module preloaded but per-process caches remain clean.
    from interval_bnb.bound_eval import (
        batch_bounds_full,
        batch_bounds_rank1_hi,
        batch_bounds_rank1_lo,
        eval_all_window_lbs_from_cached,
        gap_weighted_split_axis,
        shor_bound_float,
        window_tensor,
        _adjacency_matrix,
    )
    from interval_bnb.bound_cctr import bound_cctr_sw_float_lp
    from interval_bnb.bnb import (
        rigor_replay, rigor_replay_topk_joint, rigor_replay_with_cctr,
    )
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_anchor import (
        build_anchor_data, bound_anchor_int_ge,
        build_multi_anchor_data, bound_anchor_multi_int_ge,
        build_centroid_anchor_cache, bound_anchor_centroid_int_ge,
    )

    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)
    target_q = target_c if isinstance(target_c, Fraction) else Fraction(str(target_c))

    # Anchor-cut setup: load mu_star_d{d}.npz if present and build the
    # supporting-hyperplane data once. Activated by INTERVAL_BNB_ANCHOR_DEPTH
    # env var (default 999 = disabled).
    anchor_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_ANCHOR_DEPTH", "999"
    ))
    multi_anchors = None  # list of anchor dicts (mu*, sigma(mu*))
    centroid_cache = None  # per-window lambda_min / A_W cache
    anchor_disable_reason = None
    if anchor_depth_threshold < 999:
        anchor_npz_path = os.path.join(_REPO, f"mu_star_d{d}.npz")
        if not os.path.isfile(anchor_npz_path):
            anchor_disable_reason = (
                f"file not found: {anchor_npz_path}"
            )
        else:
            try:
                _data = np.load(anchor_npz_path, allow_pickle=True)
                if 'mu_star' in _data.files:
                    mu_star_arr = np.asarray(
                        _data['mu_star'], dtype=np.float64,
                    )
                elif 'mu' in _data.files:
                    mu_star_arr = np.asarray(_data['mu'], dtype=np.float64)
                else:
                    mu_star_arr = None
                    anchor_disable_reason = (
                        f"npz missing 'mu_star' / 'mu' key "
                        f"(found: {list(_data.files)})"
                    )
                if mu_star_arr is not None:
                    if mu_star_arr.shape != (d,):
                        anchor_disable_reason = (
                            f"mu_star shape {mu_star_arr.shape}, "
                            f"expected ({d},)"
                        )
                    else:
                        # Multi-anchor: builds {mu*, sigma(mu*)}.
                        # Sigma(mu*) is in H_d when mu_0 > mu_{d-1};
                        # without this, anchor never fires on H_d boxes
                        # (root cause of d=22 anchor 0/251763).
                        multi_anchors = build_multi_anchor_data(
                            d, mu_star_arr, windows=windows,
                        )
                        # Per-box centroid anchor: rebuilds the supporting
                        # hyperplane at each box's midpoint with curvature
                        # concession. Tighter than the global anchor on
                        # boxes far from mu*.
                        centroid_cache = build_centroid_anchor_cache(
                            d, windows=windows,
                        )
            except Exception as exc:
                anchor_disable_reason = f"load/build failed: {exc!r}"
                multi_anchors = None
                centroid_cache = None
    anchor_enabled = multi_anchors is not None
    if (anchor_depth_threshold < 999) and (not anchor_enabled):
        try:
            print(
                f"[w{worker_id}] anchor disabled: {anchor_disable_reason}",
                file=sys.stderr,
            )
        except Exception:
            pass
    local_anchor_attempts = 0
    local_anchor_certs = 0
    local_centroid_attempts = 0
    local_centroid_certs = 0
    target_f = float(target_q)

    # Depth threshold above which rigor_replay also tries the joint-face
    # McCormick dual certificate (closes boxes where max(SW, NE) < target
    # but the tighter joint LP crosses it). Below threshold, greedy bounds
    # + splitting is cheaper than invoking scipy.linprog per box.
    joint_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_JOINT_DEPTH", "20"
    ))
    # Depth threshold above which we tighten the box to its simplex
    # intersection before evaluating bounds. This tightens `hi` via the
    # simplex sum-constraint (new_hi[i] = min(hi[i], 1 - (lo_sum - lo[i])))
    # and `lo` symmetrically. The tightened box has the same simplex
    # intersection as the original, so any bound computed on it is still
    # valid, but `autoconv` (which uses (sum hi)^2 and hi^T A_W hi as
    # coefficients rather than just as constraints) gets strictly tighter.
    # At d=16 t=1.25 the BnB stalls because autoconv plateaus around 1.1
    # on deep boxes; tightening hi closes that gap.
    tighten_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_TIGHTEN_DEPTH", "4"
    ))
    # P1-LITE: at depth >= topk_joint_depth_threshold, try int joint-face
    # dual cert on top-K alternate windows when the winning window's
    # standard rigor_replay fails. Each alternate is one HiGHS LP solve
    # (~30-50 ms at d=20), so we only run on deep boxes where the
    # tree-shrink dominates the LP cost.
    # Default depth=99 (disabled). Use d-aware setting in run scripts:
    # d <= 12: don't bother (margin is generous, standard bounds suffice).
    # d >= 14: enable at depth >= 24-30 depending on d.
    topk_joint_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_TOPK_JOINT_DEPTH", "99"
    ))
    topk_joint_K = int(os.environ.get("INTERVAL_BNB_TOPK_JOINT_K", "3"))
    # P3: gap-weighted split axis threshold. Default 999 (disabled);
    # opt-in via INTERVAL_BNB_GAP_DEPTH.
    gap_split_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_GAP_DEPTH", "999"
    ))
    # PC-aligned (variance-weighted) split threshold. Empirical d=14
    # diagnostic showed widest-axis catastrophically wastes splits at
    # depth ≥ 20 — splits the low-std plateau, leaves the 8-D manifold
    # spread intact. Variance-weighted (mid·(1−mid)·width) targets the
    # transition zone where the McCormick lift is weakest. Default 25
    # (just above where stuck residual appears).
    pc_split_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_PC_DEPTH", "25"
    ))
    # CCTR LP-tier float pre-filter: skip joint/RLT LP if float-SW LB is
    # below target by more than this margin. Each LP costs 50-300 ms;
    # this filter is critical to keep BnB throughput up.
    cctr_lp_filter = float(os.environ.get(
        "INTERVAL_BNB_CCTR_LP_FILTER", "0.05"
    ))
    # EPIGRAPH LP — the bound that closes the minimax-maximin gap. Per
    # box: O(d^2) variables, O(d^2 + |W|) constraints; ~50-300ms per LP
    # at d=10-20. ONLY invoke at deep depth where standard/CCTR have
    # failed AND the box is small enough that the LP min is informative.
    # Float pre-filter (epigraph_lp_float) gates the rigor cert. Default
    # disabled via env var; opt-in with INTERVAL_BNB_EPIGRAPH_DEPTH=12.
    epigraph_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_EPIGRAPH_DEPTH", "999"
    ))
    epigraph_filter = float(os.environ.get(
        "INTERVAL_BNB_EPIGRAPH_FILTER", "0.10"
    ))
    # PER-BOX CENTROID ANCHOR: ~98ms per box at d=24 (Python iteration
    # over W * |pairs| products in argmax_window_at_centroid). Sound
    # safety-net cut for boxes where the epi LP fails. Empirically the
    # centroid cert rate is ~0% for half-width > 0.003 and ~100% for
    # half-width <= 0.0015 (Agent 1 measurement). To avoid paying its
    # cost on shallow boxes that won't benefit, gate it on a depth
    # threshold separate from the cheap multi-anchor (which always runs
    # at depth >= INTERVAL_BNB_ANCHOR_DEPTH).
    centroid_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_CENTROID_DEPTH", "999"
    ))
    # LP-BINDING SPLIT-AXIS heuristic: when the epigraph LP just solved
    # (and failed to certify), use its McCormick-face dual marginals to
    # pick the split axis whose tightening will most lift the LP value
    # next time. Default 999 (disabled). Recommended: 28 (deeper than
    # the cross-box variance heuristic at 25). Activates only when the
    # epigraph LP was solved AND ineqlin marginals are available.
    lp_split_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_LP_SPLIT_DEPTH", "999"
    ))
    # BOUNDARY-CONCENTRATED SPLIT heuristic (HIGHEST priority when active).
    # Targets the d>=22 BnB stall mode: stuck boxes have many axes pinned
    # at lo[i]<=1e-12 ("boundary axes" — mu_i=0 face of the simplex), and
    # the LP gap is dominated by (free, free) axis pair products. The
    # existing LP-binding / PC-variance / widest cascade can pick a wide
    # boundary axis whose split does not change the LP gap; the resulting
    # children inherit the same gap and the BnB random-walks. This tier,
    # when the box has >= boundary_axis_count_threshold boundary axes,
    # forces the split onto a FREE axis (lo[i] > 1e-12), which is what
    # actually tightens the LP gap on those products.
    #
    # Soundness: unaffected. The BnB is sound for ANY axis choice (the
    # split partitions the parent's feasible region into two children
    # whose union equals the parent). This heuristic only changes the
    # ORDER of work, not what is searched.
    #
    # Default 999 (disabled). Recommended at d>=22 stall: 30 (above the
    # PC-variance threshold of 25 so it overrides on deep boxes).
    boundary_split_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_BOUNDARY_SPLIT_DEPTH", "999"
    ))
    # Min number of axes at lo<=1e-12 to trigger the heuristic. Default
    # d//2 — i.e. half or more of axes pinned to the simplex boundary.
    boundary_axis_count_threshold = int(os.environ.get(
        "INTERVAL_BNB_BOUNDARY_AXIS_COUNT", str(d // 2)
    ))
    # LASSERRE ORDER-2 SDP ESCALATION (publication-rigor tier 3):
    # Fires AFTER the epigraph LP and centroid anchor on residual boxes
    # the cheap cascade could not close. Closes the McCormick gap by
    # going one rung up the Lasserre hierarchy (~O(w^4) gap vs LP's
    # O(w^2)). Cost ~1-3s per box at d=30 via MOSEK Fusion.
    # Default depth=999 (disabled). Recommended at d=30: 30 (after epi
    # LP at depth 24-30 has had its chance).
    sdp_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_SDP_DEPTH", "999"
    ))
    # Pre-filter for SDP: only invoke if the epi LP value is within
    # `sdp_filter` of the target. Boxes the epi LP says are far from
    # cert are unlikely to be closed by the SDP relaxation either.
    # Default 0.02 (the empirical SDP-vs-LP gap on residual boxes).
    sdp_filter = float(os.environ.get(
        "INTERVAL_BNB_SDP_FILTER", "0.02"
    ))
    # Per-box MOSEK time limit in seconds. Sound under-approximation
    # on timeout (returns False).
    sdp_time_limit_s = float(os.environ.get(
        "INTERVAL_BNB_SDP_TIME_LIMIT_S", "5.0"
    ))
    # Build float aggregate matrix M_float once per worker (for the
    # cheap pre-filter). Reconstructs M_float from the integer M_int
    # (avoids passing two large arrays via fork/spawn).
    cctr_M_float = None
    if cctr_M_int is not None and cctr_D_M > 0:
        cctr_M_float = np.zeros((d, d), dtype=np.float64)
        for ii in range(d):
            for jj in range(d):
                cctr_M_float[ii, jj] = float(cctr_M_int[ii, jj]) / cctr_D_M

    # Multi-α: build float versions of all aggregates for cheap pre-filter.
    multi_M_floats = None
    if (multi_cctr_M_ints is not None and multi_cctr_D_Ms is not None
            and len(multi_cctr_M_ints) > 0):
        multi_M_floats = []
        for k, (Mi, Dm) in enumerate(zip(multi_cctr_M_ints, multi_cctr_D_Ms)):
            Mf = np.zeros((d, d), dtype=np.float64)
            for ii in range(d):
                for jj in range(d):
                    Mf[ii, jj] = float(Mi[ii, jj]) / Dm
            multi_M_floats.append(Mf)

    local_stack: List[tuple] = []  # entries: (Box, depth, parent_cache, axis, which_end)
    local_nodes = 0
    local_cert = 0
    local_vol = 0.0
    local_max_depth = 0
    # CCTR diagnostics
    local_cctr_attempts = 0
    local_cctr_sw_ne_certs = 0
    local_cctr_joint_attempts = 0
    local_cctr_joint_certs = 0
    local_cctr_rlt_attempts = 0
    local_cctr_rlt_certs = 0
    # Epigraph diagnostics
    local_epi_attempts = 0
    local_epi_certs = 0
    # SDP escalation diagnostics
    local_sdp_attempts = 0
    local_sdp_certs = 0
    # Lazy MOSEK escalation cache (built on first SDP attempt; ~1s at
    # d=30). None until first invocation; then a dict from
    # `build_sdp_escalation_cache(d, windows)`.
    sdp_cache = None
    # PC1-tracker: cross-box variance EMA over stuck-box centroids.
    # Uses Welford's online mean+variance recursion (numerically stable).
    # Updated only on boxes that reach pc_split_depth_threshold (i.e. the
    # ones that are actually causing splits late). Score axis = (cross-box
    # variance per axis) * (in-box width). Only active once stuck_count >= 5.
    stuck_mean = np.zeros(d, dtype=np.float64)
    stuck_M2 = np.zeros(d, dtype=np.float64)
    stuck_count = 0
    donate_period = random.randint(50, 150)
    since_last_donate_check = 0
    rng = random.Random(worker_id * 7919 + 1)

    def _publish_stats():
        nonlocal local_nodes, local_cert, local_vol
        with node_count.get_lock():
            node_count.value += local_nodes
        with cert_count.get_lock():
            cert_count.value += local_cert
        with closed_vol.get_lock():
            closed_vol.value += local_vol
        local_nodes = 0
        local_cert = 0
        local_vol = 0.0

    try:
        while not done_event.is_set() and not failed_event.is_set():
            # ----- Refill from shared queue if empty -----
            if not local_stack:
                with idle_count.get_lock():
                    idle_count.value += 1
                try:
                    batch = queue.get(timeout=0.1)
                except Exception:
                    batch = None
                with idle_count.get_lock():
                    idle_count.value -= 1
                if batch is None:
                    # Queue empty. Check termination.
                    with in_flight.get_lock():
                        if in_flight.value == 0:
                            done_event.set()
                            break
                    # Otherwise spin: another worker is still splitting.
                    continue
                # `batch` is a list of (Box, depth) tuples (no parent_cache -- we
                # lost it at donation; recompute via batch_bounds_full).
                for item in batch:
                    if isinstance(item, tuple) and len(item) >= 2:
                        B, depth = item[0], item[1]
                    else:
                        B = item
                        depth = 0
                    local_stack.append((B, depth, None, -1, None))
                continue

            # ----- Process one node -----
            B, depth, parent_cache, changed_k, which_end = local_stack.pop()
            local_nodes += 1
            if depth > local_max_depth:
                local_max_depth = depth
            # Periodic progress publish so the master sees counters advance.
            # Lowered to 500 nodes (vs 5_000) for fast feedback on
            # LP-bottlenecked deep boxes. Trade-off: more lock contention
            # on shared counters, but with epigraph at 19ms/box this is
            # ~1 publish per 10s/worker — negligible overhead.
            if local_nodes % 500 == 0:
                _publish_stats()
            if local_nodes >= max_nodes_per_worker:
                # Safety cap (shouldn't normally fire).
                failed_event.set()
                break

            if not B.intersects_simplex():
                local_vol += B.volume()
                with in_flight.get_lock():
                    in_flight.value -= 1
                continue

            # H_d half-simplex pre-filter: drop boxes strictly outside
            # H_d = {mu_0 <= mu_{d-1}}. Sound by Lemma 3.4 (THEOREM.md).
            # The dropped box's sigma-image lies in H_d and is covered
            # by another sibling that survives this cut.
            if box_outside_hd(B):
                local_vol += B.volume()
                with in_flight.get_lock():
                    in_flight.value -= 1
                continue

            # T3 (simplex-tightened box) re-enabled at depth threshold.
            # On stall boxes at d=16 the autoconv bound is binding (~1.1
            # vs target 1.25); tightening hi via sum-constraint is the
            # only cheap way to lift autoconv. Invalidates parent_cache
            # because lo/hi change, forcing a full batch_bounds_full
            # recompute on the tightened endpoints.
            if depth >= tighten_depth_threshold:
                if B.tighten_to_simplex():
                    parent_cache = None
                    if not B.intersects_simplex():
                        # Tightening collapsed the simplex intersection
                        # (should be rare; means box was barely feasible).
                        local_vol += B.volume()
                        with in_flight.get_lock():
                            in_flight.value -= 1
                        continue

            if parent_cache is None:
                lb_fast, w_idx, which, _, my_cache = batch_bounds_full(
                    B.lo, B.hi, A_tensor, scales, target_f,
                )
            elif which_end == "lo":
                lb_fast, w_idx, which, _, my_cache = batch_bounds_rank1_lo(
                    A_tensor, scales, parent_cache, B.lo, changed_k, target_f,
                )
            else:
                lb_fast, w_idx, which, _, my_cache = batch_bounds_rank1_hi(
                    A_tensor, scales, parent_cache, B.hi, changed_k, target_f,
                )

            if lb_fast >= target_f and w_idx >= 0:
                w = windows[w_idx]
                # P2: SAFE-MARGIN SHORTCUT — skip Fraction/integer rigor when
                # float margin is comfortably above the worst-case float64
                # accumulated error. Both autoconv and McCormick float bounds
                # use the same closed-form formula as the integer paths
                # (verified by inspection of bound_autoconv_int_ge,
                # bound_mccormick_sw_int_ge, bound_mccormick_ne_int_ge).
                # Worst-case absolute error at d=20: ~d^2 * eps * |scale * x|
                # ~= 400 * 2.22e-16 * 4 ≈ 3.5e-13. Threshold 1e-9 is 4 orders
                # safer. Only applied to autoconv/mccormick paths from
                # _eval_bounds_from_cached (NOT joint-face, which is invoked
                # separately inside rigor_replay).
                if (lb_fast - target_f) > 1e-9 and which in ("autoconv", "mccormick"):
                    local_cert += 1
                    local_vol += B.volume()
                    with in_flight.get_lock():
                        in_flight.value -= 1
                    continue
                # Try standard rigor first (always cheap).
                cert_now = False
                # CCTR enabled when caller passed cctr_M_int (any depth).
                cctr_active = (cctr_M_int is not None and cctr_D_M > 0)
                if depth >= topk_joint_depth_threshold:
                    # P1-LITE: top-K alternate joint-face cert.
                    # P4 pre-filter: cheap eigenvalue Shor on winning window.
                    shor_lb = shor_bound_float(B.lo, B.hi, w, d)
                    if shor_lb >= target_f - 0.05:
                        Alo, Ahi, loAlo, hiAhi, lo_sum_c, hi_sum_c, lo_c, hi_c = my_cache
                        lb_all = eval_all_window_lbs_from_cached(
                            lo_c, hi_c, A_tensor, scales,
                            Alo, Ahi, loAlo, hiAhi, lo_sum_c, hi_sum_c,
                        )
                        order = np.argsort(-lb_all)
                        alt_idx_list = [int(i) for i in order[:topk_joint_K]
                                         if lb_all[int(i)] >= target_f - 1e-3]
                        alt_windows_list = [windows[i] for i in alt_idx_list]
                        if rigor_replay_with_cctr(
                            B, w, d, target_q,
                            alt_windows=alt_windows_list,
                            cctr_M_int=cctr_M_int if cctr_active else None,
                            cctr_D_M=cctr_D_M if cctr_active else 0,
                            try_joint=True,
                        ):
                            cert_now = True
                else:
                    # Standard tier + CCTR (no top-K joint), no Shor filter.
                    if rigor_replay_with_cctr(
                        B, w, d, target_q,
                        alt_windows=(),
                        cctr_M_int=cctr_M_int if cctr_active else None,
                        cctr_D_M=cctr_D_M if cctr_active else 0,
                        try_joint=(depth >= joint_depth_threshold),
                    ):
                        cert_now = True
                if cert_now:
                    local_cert += 1
                    local_vol += B.volume()
                    with in_flight.get_lock():
                        in_flight.value -= 1
                    continue
                # Rigor refused -- split further below.

            # MULTI-ANCHOR CUT: OR over {mu*, sigma(mu*)} supporting
            # hyperplanes (sigma-image makes the cut work in H_d when
            # mu* is not). Cheap (~10us per box: two greedy LPs over d
            # axes each), sound globally with curvature concession for
            # non-PSD A_{W*}. Always tried at this depth — fast filter.
            #
            # The PER-BOX CENTROID ANCHOR is the EXPENSIVE last-resort
            # cut and is moved AFTER the epigraph LP — it fires only on
            # boxes the epi LP couldn't close. See comment block after
            # the epigraph LP for soundness + cost analysis.
            if anchor_enabled and depth >= anchor_depth_threshold:
                lo_int_, hi_int_ = B.to_ints()
                tn_, td_ = target_q.numerator, target_q.denominator
                local_anchor_attempts += 1
                if bound_anchor_multi_int_ge(
                    lo_int_, hi_int_,
                    multi_anchors,
                    tn_, td_,
                ):
                    local_anchor_certs += 1
                    local_cert += 1
                    local_vol += B.volume()
                    with in_flight.get_lock():
                        in_flight.value -= 1
                    continue

            # CCTR LAST-CHANCE: try multiple α aggregates, each individually
            # sound. Pick the best one per box (highest float-SW LB) and
            # cascade SW/NE → joint-face → RLT in integer arithmetic.
            #
            # Multi-α path: when `multi_cctr_M_ints` is non-empty, iterate
            # over all aggregates (SW/NE first; joint/RLT only on the
            # most-promising one to limit LP cost).
            #
            # Single-α path: legacy (cctr_M_int set, multi_cctr_M_ints None).
            if multi_M_floats is not None and len(multi_M_floats) > 0:
                from interval_bnb.bound_cctr import (
                    bound_cctr_int_ge, bound_cctr_joint_face_int_ge,
                    bound_cctr_rlt_int_ge,
                )
                lo_int_, hi_int_ = B.to_ints()
                tn_, td_ = target_q.numerator, target_q.denominator
                local_cctr_attempts += 1
                # Step 1: cheap SW+NE int cert across ALL aggregates. First
                # one to certify wins.
                cert_now = False
                for k_agg in range(len(multi_cctr_M_ints)):
                    if bound_cctr_int_ge(
                        lo_int_, hi_int_,
                        multi_cctr_M_ints[k_agg], d, multi_cctr_D_Ms[k_agg],
                        tn_, td_,
                    ):
                        local_cctr_sw_ne_certs += 1
                        cert_now = True
                        break
                if cert_now:
                    local_cert += 1
                    local_vol += B.volume()
                    with in_flight.get_lock():
                        in_flight.value -= 1
                    continue
                # Step 2: at deep depth, find the BEST aggregate by float-SW
                # and run joint-face + RLT only on that one (saves LP cost).
                if depth >= joint_depth_threshold:
                    best_lb = float("-inf")
                    best_idx = -1
                    for k_agg, Mf in enumerate(multi_M_floats):
                        sw_f = bound_cctr_sw_float_lp(B.lo, B.hi, Mf, d)
                        if sw_f > best_lb:
                            best_lb = sw_f
                            best_idx = k_agg
                    if (best_idx >= 0
                            and best_lb >= target_f - cctr_lp_filter):
                        local_cctr_joint_attempts += 1
                        if bound_cctr_joint_face_int_ge(
                            lo_int_, hi_int_,
                            multi_cctr_M_ints[best_idx], d,
                            multi_cctr_D_Ms[best_idx], tn_, td_,
                        ):
                            local_cctr_joint_certs += 1
                            local_cert += 1
                            local_vol += B.volume()
                            with in_flight.get_lock():
                                in_flight.value -= 1
                            continue
                        # RLT on the best aggregate.
                        local_cctr_rlt_attempts += 1
                        if bound_cctr_rlt_int_ge(
                            lo_int_, hi_int_,
                            multi_cctr_M_ints[best_idx], d,
                            multi_cctr_D_Ms[best_idx], tn_, td_,
                        ):
                            local_cctr_rlt_certs += 1
                            local_cert += 1
                            local_vol += B.volume()
                            with in_flight.get_lock():
                                in_flight.value -= 1
                            continue
            # EPIGRAPH LP — final tier that closes the minimax-maximin gap.
            # SINGLE LP solve (extract value AND cert AND dual marginals
            # in one call — no double-solve).
            epi_ineqlin = None  # captured for LP-binding split heuristic
            if depth >= epigraph_depth_threshold:
                from interval_bnb.bound_epigraph import (
                    bound_epigraph_int_ge_with_marginals,
                )
                cert, lp_val, epi_ineqlin = (
                    bound_epigraph_int_ge_with_marginals(
                        B.lo, B.hi, windows, d, target_f,
                    )
                )
                if np.isfinite(lp_val):
                    local_epi_attempts += 1
                    if cert:
                        local_epi_certs += 1
                        local_cert += 1
                        local_vol += B.volume()
                        with in_flight.get_lock():
                            in_flight.value -= 1
                        continue

            # PER-BOX CENTROID ANCHOR (LAST-RESORT). Sound: builds a
            # supporting hyperplane at the box midpoint mu_c using the
            # box's argmax window W*(B), with curvature concession for
            # non-PSD A_{W*(B)}. Only fires when the epi LP did NOT
            # cert and the box is small enough (depth gate) for the
            # curvature concession to be tighter than the missed margin.
            # Cost ~98ms/call at d=24 (Python argmax over windows).
            if (anchor_enabled and depth >= centroid_depth_threshold
                    and centroid_cache is not None):
                lo_int_, hi_int_ = B.to_ints()
                tn_, td_ = target_q.numerator, target_q.denominator
                local_centroid_attempts += 1
                if bound_anchor_centroid_int_ge(
                    lo_int_, hi_int_,
                    tn_, td_,
                    centroid_cache,
                ):
                    local_centroid_certs += 1
                    local_cert += 1
                    local_vol += B.volume()
                    with in_flight.get_lock():
                        in_flight.value -= 1
                    continue
            if cctr_M_int is not None and cctr_D_M > 0:
                # Legacy single-α path (kept for backwards compat).
                from interval_bnb.bound_cctr import (
                    bound_cctr_int_ge, bound_cctr_joint_face_int_ge,
                    bound_cctr_rlt_int_ge,
                )
                lo_int_, hi_int_ = B.to_ints()
                tn_, td_ = target_q.numerator, target_q.denominator
                local_cctr_attempts += 1
                if bound_cctr_int_ge(
                    lo_int_, hi_int_, cctr_M_int, d, cctr_D_M, tn_, td_,
                ):
                    local_cctr_sw_ne_certs += 1
                    local_cert += 1
                    local_vol += B.volume()
                    with in_flight.get_lock():
                        in_flight.value -= 1
                    continue
                if depth >= joint_depth_threshold:
                    sw_float = bound_cctr_sw_float_lp(
                        B.lo, B.hi, cctr_M_float, d,
                    )
                    if sw_float >= target_f - cctr_lp_filter:
                        local_cctr_joint_attempts += 1
                        if bound_cctr_joint_face_int_ge(
                            lo_int_, hi_int_, cctr_M_int, d, cctr_D_M, tn_, td_,
                        ):
                            local_cctr_joint_certs += 1
                            local_cert += 1
                            local_vol += B.volume()
                            with in_flight.get_lock():
                                in_flight.value -= 1
                            continue
                        local_cctr_rlt_attempts += 1
                        if bound_cctr_rlt_int_ge(
                            lo_int_, hi_int_, cctr_M_int, d, cctr_D_M, tn_, td_,
                        ):
                            local_cctr_rlt_certs += 1
                            local_cert += 1
                            local_vol += B.volume()
                            with in_flight.get_lock():
                                in_flight.value -= 1
                            continue

            # LASSERRE ORDER-2 SDP ESCALATION (publication-rigor tier 3).
            # Last tier before splitting. Sound by Lasserre 2001 Thm 4.2:
            # the order-2 moment SDP is a valid lower bound on
            # min_{mu in B cap Delta_d} max_W mu^T A_W mu. Cushion-based
            # rigor cert; see `bound_sdp_escalation.py` docstring for
            # the soundness derivation and cushion size justification.
            #
            # Pre-filter (epi LP value within `sdp_filter` of target)
            # avoids paying the ~1-3s SDP cost on boxes the LP says are
            # far from cert. The epi LP must have been solved (tier
            # active and finite lp_val captured). Without that, we skip.
            if (depth >= sdp_depth_threshold
                    and depth >= epigraph_depth_threshold
                    and epi_ineqlin is not None):
                # `lp_val` is captured above when the epi LP solved; reuse.
                # We need an explicit reference to the value here.
                # (epi LP block sets `cert, lp_val, epi_ineqlin`; on this
                # path `cert` was False so the LP value is below target.)
                if np.isfinite(lp_val) and lp_val >= target_f - sdp_filter:
                    if sdp_cache is None:
                        from interval_bnb.bound_sdp_escalation import (
                            build_sdp_escalation_cache,
                        )
                        sdp_cache = build_sdp_escalation_cache(d, windows)
                    from interval_bnb.bound_sdp_escalation import (
                        bound_sdp_escalation_int_ge,
                    )
                    lo_int_, hi_int_ = B.to_ints()
                    tn_, td_ = target_q.numerator, target_q.denominator
                    local_sdp_attempts += 1
                    if bound_sdp_escalation_int_ge(
                        lo_int_, hi_int_, windows, d,
                        tn_, td_,
                        cache=sdp_cache,
                        time_limit_s=sdp_time_limit_s,
                        n_threads=1,
                    ):
                        local_sdp_certs += 1
                        local_cert += 1
                        local_vol += B.volume()
                        with in_flight.get_lock():
                            in_flight.value -= 1
                        continue

            if B.max_width() < min_box_width:
                failed_event.set()
                break

            # SPLIT STRATEGY:
            #   * depth < gap_split_depth_threshold: widest-axis
            #   * gap <= depth < pc_split_depth_threshold: gap-weighted axis
            #     (existing logic — picks axis whose split most tightens the
            #     winning window's McCormick bound)
            #   * depth >= pc_split_depth_threshold: VARIANCE-WEIGHTED axis,
            #     i.e. split on argmax_i [mid_i · (1 − mid_i) · width_i].
            #     This approximates PCA-aligned splitting: at the d=14 stall
            #     the empirical agent found widest-axis splits the wrong
            #     coords (low-std plateau) and leaves the high-std manifold
            #     directions intact. The variance term `mid·(1−mid)` heavily
            #     weights the transition zone (μ ≈ 0.05–0.5) where the
            #     McCormick lift is weakest, while `width` ensures we make
            #     real progress on box volume. Empirically: redirects ~3×
            #     more splits to high-variance axes.
            # Splittability mask: an axis with integer width <= 1 cannot
            # be split exactly under D_SHIFT-bit dyadic arithmetic and
            # would raise inside Box.split. ANY axis chooser below must
            # respect this mask. (Hit at d=20 init_split_depth>=26 where
            # some axes saturate while others remain wide.)
            lo_int_arr = np.asarray(B.lo_int, dtype=object) if B.lo_int is not None else None
            hi_int_arr = np.asarray(B.hi_int, dtype=object) if B.hi_int is not None else None
            if lo_int_arr is not None and hi_int_arr is not None:
                int_widths = hi_int_arr - lo_int_arr
                splittable = np.array([int(w) >= 2 for w in int_widths], dtype=bool)
            else:
                splittable = (B.hi - B.lo) > 0
            if not splittable.any():
                # Fully saturated box: every axis exhausted dyadic depth
                # (int_width <= 1) without a certificate. Soundness
                # requires us to FAIL LOUDLY rather than silently close
                # the box. With the default `min_box_width=1e-10` (>>
                # 2^-60) the upstream `B.max_width() < min_box_width`
                # check fires first and this branch is unreachable; we
                # keep it as a defensive guard against future config
                # changes that could lower min_box_width below the
                # dyadic resolution.
                failed_event.set()
                break

            # BOUNDARY-CONCENTRATED split heuristic (HIGHEST priority).
            # When the box has >= boundary_axis_count_threshold axes pinned
            # at lo[i] <= 1e-12 (mu_i==0 face of the simplex), prefer
            # splitting a FREE axis (lo[i] > 1e-12) — these are the axes
            # whose product terms dominate the LP gap, and splitting a
            # boundary axis is wasted work (LP gap unchanged on the
            # surviving free-axis pair products). Among free splittable
            # axes, pick the widest. Falls back to widest splittable when
            # no free axis is splittable. Soundness is unaffected (any
            # valid axis split is sound; this only reorders work).
            axis = None
            if depth >= boundary_split_depth_threshold:
                boundary_mask = B.lo <= 1e-12
                n_boundary = int(np.count_nonzero(boundary_mask))
                if n_boundary >= boundary_axis_count_threshold:
                    free_mask = ~boundary_mask
                    free_splittable = free_mask & splittable
                    if free_splittable.any():
                        widths_b = B.hi - B.lo
                        ws_free = np.where(
                            free_splittable, widths_b, -np.inf
                        )
                        cand = int(np.argmax(ws_free))
                        if np.isfinite(ws_free[cand]):
                            axis = cand
                    # else: fall through to existing cascade (widest
                    # splittable will run as the final fallback below).

            # LP-BINDING split heuristic (next priority when active).
            # Uses the just-returned ineqlin from the epigraph LP to score
            # axes by Σ_j (|λ_SW(i,j)|+|λ_NE|+|λ_NW|+|λ_SE|) · width[i].
            # Soundness is unaffected (any valid axis split is sound).
            if (axis is None
                    and depth >= lp_split_depth_threshold
                    and epi_ineqlin is not None):
                from interval_bnb.bound_epigraph import lp_binding_axis_score
                widths = B.hi - B.lo
                lp_score = lp_binding_axis_score(epi_ineqlin, widths, d)
                lp_score = np.where(splittable, lp_score, -np.inf)
                cand = int(np.argmax(lp_score))
                if np.isfinite(lp_score[cand]) and lp_score[cand] > 0:
                    axis = cand

            if axis is not None:
                pass  # boundary or LP-binding heuristic chose `axis`
            elif depth >= pc_split_depth_threshold and w_idx >= 0:
                widths = B.hi - B.lo
                mid = (B.lo + B.hi) * 0.5
                # Welford update: track cross-box centroid variance per axis.
                # M2[i] = sum_n (mid_i^(n) - mean_i)^2 (running),
                # var[i] = M2[i] / count.
                stuck_count += 1
                delta = mid - stuck_mean
                stuck_mean += delta / stuck_count
                delta2 = mid - stuck_mean
                stuck_M2 += delta * delta2
                if stuck_count >= 5:
                    # cross-box variance per axis (preferred: targets the
                    # axes where stuck-box centroids actually disagree)
                    cross_var = stuck_M2 / stuck_count
                    score = cross_var * widths
                    score = np.where(splittable, score, -np.inf)
                    axis = int(np.argmax(score))
                    if not np.isfinite(score[axis]) or score[axis] <= 0:
                        # tracker degenerate: fall back to in-box heuristic
                        var_proxy = mid * (1.0 - mid)
                        score = var_proxy * widths
                        score = np.where(splittable, score, -np.inf)
                        axis = int(np.argmax(score))
                        if not np.isfinite(score[axis]) or score[axis] <= 0:
                            # widest splittable axis
                            ws = np.where(splittable, widths, -np.inf)
                            axis = int(np.argmax(ws))
                else:
                    # warm-up: use in-box variance heuristic
                    var_proxy = mid * (1.0 - mid)
                    score = var_proxy * widths
                    score = np.where(splittable, score, -np.inf)
                    axis = int(np.argmax(score))
                    if not np.isfinite(score[axis]) or score[axis] <= 0:
                        ws = np.where(splittable, widths, -np.inf)
                        axis = int(np.argmax(ws))
            elif depth >= gap_split_depth_threshold and w_idx >= 0:
                axis = gap_weighted_split_axis(B.lo, B.hi, windows[w_idx], d)
                if not splittable[axis]:
                    ws = np.where(splittable, B.hi - B.lo, -np.inf)
                    axis = int(np.argmax(ws))
            else:
                # widest splittable axis (avoid saturated axes)
                ws = np.where(splittable, B.hi - B.lo, -np.inf)
                axis = int(np.argmax(ws))
            left, right = B.split(axis)
            delta = 1  # net change: +2 children - 1 closed parent
            if not left.intersects_simplex():
                delta -= 1
                local_vol += left.volume()
            if not right.intersects_simplex():
                delta -= 1
                local_vol += right.volume()
            with in_flight.get_lock():
                in_flight.value += delta
            if right.intersects_simplex():
                local_stack.append((right, depth + 1, my_cache, axis, "lo"))
            if left.intersects_simplex():
                local_stack.append((left, depth + 1, my_cache, axis, "hi"))

            # ----- Donate shallow half if stack grew large -----
            # trigger goes DOWN as more workers are idle (be more generous
            # donating when others have nothing; hoard only when everyone
            # is busy and donations would be wasted IPC).
            since_last_donate_check += 1
            if since_last_donate_check >= donate_period:
                since_last_donate_check = 0
                donate_period = rng.randint(10, 40)  # tight check period
                idle_now = max(0, idle_count.value)
                # BUG FIX: trigger formula previously hardcoded 64
                # (assumed pod's 64-core config), starving smaller-worker
                # runs. Now: when ANY worker is idle, donate eagerly down
                # to the floor; only hoard a buffer of 4*floor when no
                # idle. This restores the "redistribute aggressively"
                # behavior at any worker count.
                if idle_now > 0:
                    trigger = donate_threshold_floor
                else:
                    trigger = max(donate_threshold_floor * 4, 16)
                if len(local_stack) >= trigger:
                    # SMALL BATCHES → MORE RECEIVERS. Previously donated
                    # 50-90% of stack in ONE batch → only 1 worker grabs it.
                    # With 192 workers idle, we want each donation to feed
                    # a DIFFERENT worker. Donate small batches of size
                    # = max(2, len/8) per call. Many small batches over
                    # time spread work across all idle workers.
                    batch_size = max(2, len(local_stack) // 8)
                    batch_size = min(batch_size, len(local_stack) - 1)
                    if batch_size >= 1:
                        donation = local_stack[:batch_size]
                        local_stack = local_stack[batch_size:]
                        donated_payload = [(B, depth) for (B, depth, _, _, _) in donation]
                        try:
                            queue.put(donated_payload, timeout=1.0)
                        except Exception:
                            local_stack = donation + local_stack

    finally:
        _publish_stats()
        # Optional: dump remaining in_flight boxes to a per-worker .npz file
        # for post-mortem analysis of stuck/stall pathologies.
        # Activated by env var INTERVAL_BNB_DUMP_BOXES=<prefix>; each worker
        # writes <prefix>_w{worker_id}.npz with arrays lo, hi, depths.
        _dump_prefix = os.environ.get('INTERVAL_BNB_DUMP_BOXES', '')
        if _dump_prefix and local_stack:
            try:
                _los = np.asarray([B.lo for (B, _, _, _, _) in local_stack],
                                   dtype=np.float64)
                _his = np.asarray([B.hi for (B, _, _, _, _) in local_stack],
                                   dtype=np.float64)
                _deps = np.asarray([dep for (_, dep, _, _, _) in local_stack],
                                    dtype=np.int64)
                np.savez(f'{_dump_prefix}_w{worker_id}.npz',
                         lo=_los, hi=_his, depths=_deps)
            except Exception:
                pass
        if local_max_depth > 0:
            try:
                stats_queue.put_nowait({
                    "worker_id": worker_id,
                    "max_depth": local_max_depth,
                    "cctr_attempts": local_cctr_attempts,
                    "cctr_sw_ne_certs": local_cctr_sw_ne_certs,
                    "cctr_joint_attempts": local_cctr_joint_attempts,
                    "cctr_joint_certs": local_cctr_joint_certs,
                    "cctr_rlt_attempts": local_cctr_rlt_attempts,
                    "cctr_rlt_certs": local_cctr_rlt_certs,
                    "epi_attempts": local_epi_attempts,
                    "epi_certs": local_epi_certs,
                    "anchor_attempts": local_anchor_attempts,
                    "anchor_certs": local_anchor_certs,
                    "centroid_attempts": local_centroid_attempts,
                    "centroid_certs": local_centroid_certs,
                    "sdp_attempts": local_sdp_attempts,
                    "sdp_certs": local_sdp_certs,
                })
            except Exception:
                pass


# ---------------------------------------------------------------------
# Master-side driver
# ---------------------------------------------------------------------

def parallel_branch_and_bound(
    d: int,
    target_c,
    *,
    workers: Optional[int] = None,
    init_split_depth: int = 10,
    min_box_width: float = 1e-10,
    max_nodes: int = 10**10,
    pull_batch_max: int = 64,
    donate_threshold_floor: int = 16,
    time_budget_s: Optional[float] = None,
    use_symmetry: bool = True,
    enable_cctr: bool = False,
    enable_multi_cctr: bool = False,
    cctr_mu_star=None,
    multi_cctr_strategies: tuple = ('kkt', 'uniform_active', 'all_windows'),
    verbose: bool = True,
    **_ignored,
):
    """Run work-stealing parallel BnB. Returns a result dict.

    Parameters
    ----------
    d : int                ambient dimension
    target_c : number      the c to certify (Fraction, str, or float)
    workers : int          number of worker processes (default cpu_count-1)
    init_split_depth : int seeds ~2**depth starter boxes
    min_box_width : float  failure threshold
    max_nodes : int        global node cap (each worker enforces a share)
    pull_batch_max : int   max boxes per shared-queue get
    donate_threshold_floor : int  min stack size before a worker donates
    time_budget_s : float  soft wall-clock cap
    """
    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")

    sym = half_simplex_cuts(d) if use_symmetry else []
    t0 = time.time()
    starter_boxes = _split_initial(d, init_split_depth, sym)
    n_starter = len(starter_boxes)
    workers = workers or max(1, mp.cpu_count() - 1)
    if verbose:
        print(f"[par] d={d}  target={target_c}  init_boxes={n_starter}  "
              f"workers={workers}  init_split_depth={init_split_depth}")

    # CCTR aggregate(s): build once if enabled.
    cctr_M_int = None
    cctr_D_M = 0
    multi_cctr_M_ints = None
    multi_cctr_D_Ms = None
    if enable_multi_cctr:
        from interval_bnb.cctr_setup import setup_multi_cctr
        if verbose:
            print(f"[par] MULTI-CCTR: building aggregates "
                  f"({list(multi_cctr_strategies)})")
        aggregates = setup_multi_cctr(
            d, mu_star=cctr_mu_star,
            include=multi_cctr_strategies,
        )
        multi_cctr_M_ints = [ag['M_int'] for ag in aggregates]
        multi_cctr_D_Ms = [ag['D_M'] for ag in aggregates]
        if verbose:
            for ag in aggregates:
                print(f"[par] MULTI-CCTR aggregate '{ag['name']}': "
                      f"|active|={len(ag['active_idx'])}, "
                      f"val_max={ag['val_max']:.6f}")
    elif enable_cctr:
        from interval_bnb.cctr_setup import setup_cctr
        if verbose:
            print(f"[par] CCTR: building aggregate M_int from KKT-correct mu*")
        cctr_data = setup_cctr(d, mu_star=cctr_mu_star)
        cctr_M_int = cctr_data['M_int']
        cctr_D_M = cctr_data['D_M']
        if verbose:
            print(f"[par] CCTR: |active|={len(cctr_data['active_idx'])}, "
                  f"val_max(mu*)={cctr_data['val_max']:.6f}, "
                  f"residual={cctr_data['kkt_residual']:.4e}, "
                  f"D_M={cctr_D_M}")

    # Shared primitives
    queue = ctx.Queue(maxsize=0)
    in_flight = ctx.Value("i", n_starter)
    idle_count = ctx.Value("i", 0)
    cert_count = ctx.Value("i", 0)
    node_count = ctx.Value("i", 0)
    closed_vol = ctx.Value("d", 0.0)
    failed_event = ctx.Event()
    done_event = ctx.Event()
    stats_queue = ctx.Queue()

    # Total search volume = sum of starter box volumes. This is the
    # denominator of the ETA fraction.
    total_volume = float(sum(b.volume() for b in starter_boxes))

    # Seed the queue in CHUNKS so the queue machinery is primed. We send
    # each starter box as a singleton list so workers can batch-pull.
    # Larger chunks reduce queue IPC but may starve fast workers at start.
    seed_chunk = max(1, n_starter // (workers * 4) + 1)
    for i in range(0, n_starter, seed_chunk):
        chunk = [(b, 0) for b in starter_boxes[i:i + seed_chunk]]
        queue.put(chunk)

    # Max nodes per worker (approximate -- just a safety cap)
    max_nodes_per_worker = max(1, max_nodes // workers)

    procs: List[mp.Process] = []
    target_pass = target_c if isinstance(target_c, Fraction) else str(target_c)
    for wid in range(workers):
        p = ctx.Process(
            target=_worker_main,
            args=(
                wid, d, target_pass, min_box_width, max_nodes_per_worker,
                pull_batch_max, donate_threshold_floor,
                queue, in_flight, idle_count, cert_count, node_count,
                closed_vol, failed_event, done_event, stats_queue,
                cctr_M_int, cctr_D_M,
                multi_cctr_M_ints, multi_cctr_D_Ms,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    last_log = time.time()
    last_nodes = 0
    try:
        while True:
            if done_event.wait(timeout=1.0):
                break
            if failed_event.is_set():
                break
            if time_budget_s is not None and time.time() - t0 > time_budget_s:
                failed_event.set()
                if verbose:
                    print(f"[par] time_budget_s={time_budget_s} exceeded; aborting")
                break
            # Progress log.
            t = time.time() - t0
            if verbose and t - (last_log - t0) > 5.0:
                n = node_count.value
                c = cert_count.value
                inf_ = in_flight.value
                cvol = closed_vol.value
                try:
                    qs = queue.qsize()
                except NotImplementedError:
                    qs = -1
                rate = (n - last_nodes) / max(1.0, time.time() - last_log)
                idle_n = idle_count.value
                # Volume-based progress + ETA.
                frac = min(1.0, cvol / total_volume) if total_volume > 0 else 0.0
                if frac > 1e-6:
                    eta_s = (1.0 - frac) * t / frac
                    eta_tag = f"eta={eta_s:7.0f}s" if eta_s < 99999 else "eta= >1day"
                else:
                    eta_tag = "eta=  --"
                print(f"[par] t={t:7.1f}s  nodes={n:>10d}  cert={c:>10d}  "
                      f"in_flight={inf_:>6d}  queue={qs:>4d}  "
                      f"active={workers - idle_n:>2d}/{workers}  "
                      f"rate={rate:>7.0f}/s  "
                      f"progress={100*frac:9.5f}%  {eta_tag}")
                last_log = time.time()
                last_nodes = n
    finally:
        # Signal workers to stop and join.
        # NOTE: This `done_event.set()` is purely a worker-shutdown signal,
        # NOT a success vote. Success is determined ONLY by `failed_event`
        # being unset (line below). If a worker hit `failed_event` (e.g.
        # min_box_width without cert) and broke out leaving local_stack
        # non-empty, those boxes are "lost" — but the master correctly
        # reports failure via `ok = not failed_event.is_set()`. Soundness
        # therefore depends on `failed_event` being correctly set on any
        # uncertified termination, which it is at every break path.
        done_event.set()
        # Drain the shared queue and dump it BEFORE joining workers, so
        # we capture all in_flight boxes that were waiting in the queue
        # (not just those in worker local DFS stacks). Activated by
        # INTERVAL_BNB_DUMP_BOXES=<prefix>; writes <prefix>_master_queue.npz.
        # Soundness: queue items are batches of (Box, depth) tuples that
        # workers haven't claimed yet — they are still in_flight.
        _dump_prefix = os.environ.get('INTERVAL_BNB_DUMP_BOXES', '')
        if _dump_prefix:
            try:
                _q_los: list = []
                _q_his: list = []
                _q_deps: list = []
                # done_event is set so workers will stop pulling from queue
                # at their next loop guard check; drain now is race-free.
                while True:
                    try:
                        batch = queue.get_nowait()
                    except Exception:
                        break
                    for item in batch:
                        if isinstance(item, tuple) and len(item) >= 2:
                            B, depth = item[0], item[1]
                        else:
                            B = item
                            depth = 0
                        _q_los.append(B.lo)
                        _q_his.append(B.hi)
                        _q_deps.append(depth)
                if _q_los:
                    np.savez(f'{_dump_prefix}_master_queue.npz',
                              lo=np.asarray(_q_los, dtype=np.float64),
                              hi=np.asarray(_q_his, dtype=np.float64),
                              depths=np.asarray(_q_deps, dtype=np.int64))
                    if verbose:
                        print(f"[par] master drained queue: "
                              f"{len(_q_los)} boxes -> "
                              f"{_dump_prefix}_master_queue.npz", flush=True)
            except Exception as e:
                if verbose:
                    print(f"[par] queue-drain dump failed: {e}", flush=True)

        # First pass: short join to catch workers already at the loop
        # guard. Second pass: longer wait for workers mid-LP-solve so
        # their `_publish_stats` flush in the `finally` block can land
        # before we resort to `terminate()` (which bypasses `finally`
        # and loses any local_vol / local_cert / local_nodes accumulated
        # since the last 500-node publish).
        for p in procs:
            p.join(timeout=5.0)
        for p in procs:
            if p.is_alive():
                p.join(timeout=30.0)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

    ok = not failed_event.is_set()
    stats_items = []
    while True:
        try:
            stats_items.append(stats_queue.get_nowait())
        except Exception:
            break
    max_depth = max((s["max_depth"] for s in stats_items), default=0)
    cctr_stats = {
        "attempts": sum(s.get("cctr_attempts", 0) for s in stats_items),
        "sw_ne_certs": sum(s.get("cctr_sw_ne_certs", 0) for s in stats_items),
        "joint_attempts": sum(s.get("cctr_joint_attempts", 0) for s in stats_items),
        "joint_certs": sum(s.get("cctr_joint_certs", 0) for s in stats_items),
        "rlt_attempts": sum(s.get("cctr_rlt_attempts", 0) for s in stats_items),
        "rlt_certs": sum(s.get("cctr_rlt_certs", 0) for s in stats_items),
    }
    epi_stats = {
        "attempts": sum(s.get("epi_attempts", 0) for s in stats_items),
        "certs": sum(s.get("epi_certs", 0) for s in stats_items),
    }
    anchor_stats = {
        "attempts": sum(s.get("anchor_attempts", 0) for s in stats_items),
        "certs": sum(s.get("anchor_certs", 0) for s in stats_items),
    }
    centroid_stats = {
        "attempts": sum(s.get("centroid_attempts", 0) for s in stats_items),
        "certs": sum(s.get("centroid_certs", 0) for s in stats_items),
    }
    sdp_stats = {
        "attempts": sum(s.get("sdp_attempts", 0) for s in stats_items),
        "certs": sum(s.get("sdp_certs", 0) for s in stats_items),
    }

    elapsed = time.time() - t0
    cvol_final = closed_vol.value
    # SOUNDNESS NOTE: `coverage_fraction` is for progress display only.
    # The denominator `total_volume` is the sum of starter-box
    # HYPERRECTANGLE volumes (Box.volume), not vol(B ∩ Δ_d), so the
    # ratio is informational. Soundness of val(d) >= target is
    # established by  `success == True`  i.e.  failed_event NOT set
    # AND in_flight_final == 0, NOT by coverage_fraction.
    result = {
        "success": ok,
        "target_q": str(target_c) if isinstance(target_c, Fraction) else str(target_c),
        "d": d,
        "elapsed_s": elapsed,
        "total_nodes": node_count.value,
        "total_leaves_certified": cert_count.value,
        "closed_volume": cvol_final,
        "total_volume": total_volume,
        "coverage_fraction": cvol_final / total_volume if total_volume > 0 else 0.0,
        "in_flight_final": in_flight.value,
        "max_depth": max_depth,
        "workers": workers,
        "init_boxes": n_starter,
    }
    result["cctr_stats"] = cctr_stats
    result["epi_stats"] = epi_stats
    result["anchor_stats"] = anchor_stats
    result["centroid_stats"] = centroid_stats
    result["sdp_stats"] = sdp_stats
    if verbose:
        print(f"[par] DONE success={ok} elapsed={elapsed:.1f}s  "
              f"nodes={result['total_nodes']}  cert={result['total_leaves_certified']}  "
              f"coverage={100*result['coverage_fraction']:.3f}%  "
              f"in_flight_final={result['in_flight_final']}")
        if cctr_stats["attempts"] > 0:
            print(f"[par] CCTR: attempts={cctr_stats['attempts']}, "
                  f"SW/NE certs={cctr_stats['sw_ne_certs']}, "
                  f"joint: {cctr_stats['joint_certs']}/{cctr_stats['joint_attempts']}, "
                  f"RLT: {cctr_stats['rlt_certs']}/{cctr_stats['rlt_attempts']}")
        if epi_stats["attempts"] > 0:
            print(f"[par] EPIGRAPH: certs={epi_stats['certs']}/{epi_stats['attempts']}")
        if anchor_stats["attempts"] > 0:
            print(f"[par] ANCHOR: certs={anchor_stats['certs']}/{anchor_stats['attempts']}")
        if centroid_stats["attempts"] > 0:
            print(f"[par] CENTROID: certs={centroid_stats['certs']}/{centroid_stats['attempts']}")
        if sdp_stats["attempts"] > 0:
            print(f"[par] SDP-ESCALATION: certs={sdp_stats['certs']}/{sdp_stats['attempts']}")
    return result
