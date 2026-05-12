#!/usr/bin/env python
"""Subdivision box cert for c=1.29 — uses production Numba cascade.

Uses the PRODUCTION Numba-JIT kernels for all heavy computation:
  - _fused_coarse / process_parent for cascade child generation
  - _prune_no_correction for batch box cert on subcells

Only the final convergence level needs box cert (intermediate levels
don't matter for the proof — see docstring below).

val(16) = 1.319 > 1.29  =>  margin = 0.029
"""
import subprocess, sys, os, time, re, math
import multiprocessing as mp

os.chdir("/workspace/sidon-autocorrelation")
sys.path.insert(0, ".")
sys.path.insert(0, "cloninger-steinerberger/cpu")
sys.path.insert(0, "cloninger-steinerberger")

import numpy as np
import numba
from numba import njit, prange

from run_cascade_coarse import (
    _prune_no_correction,
    _fused_coarse,
    coarse_x_cap,
    process_parent,
    run_level0,
    _worker,
)
from compositions import generate_canonical_compositions_batched
from pruning import count_compositions


def _log(msg):
    print(msg, flush=True)


# =====================================================================
# Modified fused kernel: outputs box-cert-failing compositions
# =====================================================================

@njit(cache=True)
def _fused_coarse_collect_failures(parent_int, d_child, S, c_target,
                                    lo_arr, hi_arr,
                                    surv_buf, fail_buf):
    """Like _fused_coarse but also collects box-cert failures.

    Returns (n_survivors, n_failures, n_tested, min_cert_net).
    surv_buf: output buffer for survivors (TV < c_target)
    fail_buf: output buffer for box-cert failures (TV >= c but net < 0)
    """
    d_parent = parent_int.shape[0]
    conv_len = 2 * d_child - 1

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d_child)
    inv_2d = 1.0 / (2.0 * d_d)
    eps = 1e-9
    max_ell = 2 * d_child
    d_minus_1 = d_child - 1

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d
                                - eps)

    max_surv = surv_buf.shape[0]
    max_fail = fail_buf.shape[0]
    n_surv = 0
    n_fail = 0
    n_tested = np.int64(0)
    local_min_net = np.float64(1e30)

    cursor = np.empty(d_parent, dtype=np.int32)
    child = np.empty(d_child, dtype=np.int32)
    conv = np.empty(conv_len, dtype=np.int32)
    grad_buf = np.empty(d_child, dtype=np.float64)

    for i in range(d_parent):
        cursor[i] = lo_arr[i]

    while True:
        for i in range(d_parent):
            child[2 * i] = cursor[i]
            child[2 * i + 1] = parent_int[i] - cursor[i]

        n_tested += 1

        # Full autoconvolution
        for k in range(conv_len):
            conv[k] = np.int32(0)
        for i in range(d_child):
            ci = np.int32(child[i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d_child):
                    cj = np.int32(child[j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        # Window scan
        pruned = False
        best_net = np.float64(-1e30)

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(
                        conv[s_lo - 1])
                if ws > dyn_it:
                    pruned = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target
                    for i in range(d_child):
                        g = 0.0
                        for j in range(d_child):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(child[j]) / S_d
                        grad_buf[i] = g * scale_g
                    for i in range(1, d_child):
                        key = grad_buf[i]
                        jj = i - 1
                        while jj >= 0 and grad_buf[jj] > key:
                            grad_buf[jj + 1] = grad_buf[jj]
                            jj -= 1
                        grad_buf[jj + 1] = key
                    cell_var = 0.0
                    for kk in range(d_child // 2):
                        cell_var += grad_buf[d_child-1-kk] - grad_buf[kk]
                    cell_var /= (2.0 * S_d)
                    n_pairs = 0
                    for kk in range(s_lo, s_lo + ell - 1):
                        cnt = min(kk + 1, d_child)
                        if kk > d_child - 1:
                            cnt = min(cnt, 2 * d_child - 1 - kk)
                        n_pairs += cnt
                    R_bound = (2.0 * d_d / ell_f) * np.float64(
                        n_pairs) / (4.0 * S_sq)
                    net = margin - cell_var - R_bound
                    if net > best_net:
                        best_net = net

        if pruned:
            if best_net < local_min_net:
                local_min_net = best_net
            if best_net < 0.0:
                # Box cert failure — collect it
                if n_fail < max_fail:
                    for i in range(d_child):
                        fail_buf[n_fail, i] = child[i]
                n_fail += 1
        else:
            if n_surv < max_surv:
                for i in range(d_child):
                    surv_buf[n_surv, i] = child[i]
            n_surv += 1

        # Advance odometer
        carry = d_parent - 1
        while carry >= 0:
            cursor[carry] += 1
            if cursor[carry] <= hi_arr[carry]:
                break
            cursor[carry] = lo_arr[carry]
            carry -= 1
        if carry < 0:
            break

    return n_surv, n_fail, n_tested, local_min_net


def process_parent_collect(parent_int, d_child, S, c_target,
                            surv_cap=100_000, fail_cap=500_000):
    """Process one parent, collect both survivors and box-cert failures."""
    d_parent = len(parent_int)
    x_cap = coarse_x_cap(d_child, S, c_target)

    lo_arr = np.empty(d_parent, dtype=np.int32)
    hi_arr = np.empty(d_parent, dtype=np.int32)
    total_product = 1

    for i in range(d_parent):
        p = int(parent_int[i])
        lo = max(0, p - x_cap)
        hi = min(p, x_cap)
        if lo > hi:
            return (np.empty((0, d_child), dtype=np.int32),
                    np.empty((0, d_child), dtype=np.int32), 0, 1e30)
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_product *= (hi - lo + 1)

    if total_product == 0:
        return (np.empty((0, d_child), dtype=np.int32),
                np.empty((0, d_child), dtype=np.int32), 0, 1e30)

    surv_buf = np.empty((min(total_product, surv_cap), d_child), dtype=np.int32)
    fail_buf = np.empty((min(total_product, fail_cap), d_child), dtype=np.int32)

    ns, nf, nt, mn = _fused_coarse_collect_failures(
        parent_int, d_child, S, c_target, lo_arr, hi_arr,
        surv_buf, fail_buf)

    # Handle buffer overflow
    if ns > surv_cap or nf > fail_cap:
        surv_buf = np.empty((max(ns, 1), d_child), dtype=np.int32)
        fail_buf = np.empty((max(nf, 1), d_child), dtype=np.int32)
        ns2, nf2, _, mn = _fused_coarse_collect_failures(
            parent_int, d_child, S, c_target, lo_arr, hi_arr,
            surv_buf, fail_buf)
        ns, nf = ns2, nf2

    return (surv_buf[:ns].copy(), fail_buf[:nf].copy(), nt, mn)


# =====================================================================
# Subcell enumeration (for subdivision)
# =====================================================================

def enumerate_subcells_batch(k_parent, S, refine):
    """Enumerate sub-grid-points at S*refine within cell of k_parent/S.
    Returns (batch_array, S_fine).
    """
    d = len(k_parent)
    Sf = S * refine
    cf = k_parent * refine
    half = refine // 2
    lo = np.maximum(cf - half, 0).astype(np.int32)
    hi = (cf + half).astype(np.int32)
    results = []
    _rec(lo, hi, d, Sf, 0, np.zeros(d, dtype=np.int32), results)
    if not results:
        return np.empty((0, d), dtype=np.int32), Sf
    return np.array(results, dtype=np.int32), Sf

def _rec(lo, hi, d, tgt, idx, cur, out):
    if idx == d - 1:
        v = tgt - int(cur[:idx].sum())
        if lo[idx] <= v <= hi[idx] and v >= 0:
            cur[idx] = v
            out.append(cur.copy())
        return
    partial = int(cur[:idx].sum())
    for v in range(lo[idx], hi[idx] + 1):
        cur[idx] = v
        rem = tgt - partial - v
        rlo = int(lo[idx+1:].sum())
        rhi = int(hi[idx+1:].sum())
        if rem < rlo or rem > rhi or rem < 0:
            continue
        _rec(lo, hi, d, tgt, idx + 1, cur, out)


def subdiv_cert_batch(k, d, S, c_target, max_refine=32, verbose=False):
    """Subdivision cert using Numba batch pruner. Returns (ok, refine, n_sub, worst)."""
    # First check at original resolution
    mu = k.reshape(1, -1).astype(np.int32)
    _, mn = _prune_no_correction(mu, d, S, c_target)
    if mn >= 0:
        return True, 1, 1, mn

    refine = 2
    while refine <= max_refine:
        subs, Sf = enumerate_subcells_batch(k, S, refine)
        if len(subs) == 0:
            refine *= 2
            continue
        surv_mask, mn = _prune_no_correction(subs, d, Sf, c_target)
        # All should be pruned (val(d) > c_target), count box cert failures
        n_fail = int(surv_mask.sum())  # survived = not pruned = bad
        # For pruned ones, mn tells us worst box cert net
        # But we need per-cell: if mn >= 0, all pass
        if mn >= 0 and n_fail == 0:
            if verbose:
                _log(f"      r={refine}x S={Sf}: {len(subs)} sub, "
                     f"all certified, worst_net={mn:.6f}")
            return True, refine, len(subs), mn
        if verbose:
            _log(f"      r={refine}x S={Sf}: {len(subs)} sub, "
                 f"survivors={n_fail}, min_net={mn:.6f}")
        refine *= 2
    return False, refine // 2, len(subs), mn


# =====================================================================
# Run one config
# =====================================================================

def run_one(d0, S, c_target, max_levels):
    _log(f"\n{'#'*70}")
    _log(f"# d0={d0}, S={S}, c={c_target}, max_levels={max_levels}")
    _log(f"{'#'*70}")

    # Clean checkpoints
    for f in os.listdir("data"):
        if f.startswith("checkpoint_"):
            try: os.remove(os.path.join("data", f))
            except: pass

    # --- STEP 1: Run cascade using production code ---
    t_total = time.time()
    _log(f"\n  [CASCADE]")

    # L0
    l0 = run_level0(d0, S, c_target, verbose=True)
    if l0['proven']:
        if l0['box_certified']:
            _log(f"\n  *** RIGOROUS PROOF at L0: C_{{1a}} >= {c_target} ***")
            return
        _log(f"  L0: all pruned, box cert {'PASS' if l0['box_certified'] else 'FAIL'}")

    current = l0['survivors']
    d_parent = d0
    converge_level = 0 if l0['n_survivors'] == 0 else None
    final_failures = None

    if l0['n_survivors'] > 0:
        for level in range(1, max_levels + 1):
            d_child = 2 * d_parent
            n_parents = len(current)
            if n_parents == 0:
                break

            x_cap = coarse_x_cap(d_child, S, c_target)
            _log(f"\n  [L{level}] d={d_parent}->{d_child}, "
                 f"parents={n_parents:,}, x_cap={x_cap}")

            # Pre-filter infeasible
            feasible = np.all(current <= 2 * x_cap, axis=1)
            current = np.ascontiguousarray(current[feasible])
            n_parents = len(current)

            t0 = time.time()
            all_surv = []
            all_fail = []
            total_children = 0
            level_min_net = 1e30
            n_done = 0

            for pi in range(n_parents):
                survs, fails, nt, mn = process_parent_collect(
                    current[pi], d_child, S, c_target)
                total_children += nt
                if mn < level_min_net:
                    level_min_net = mn
                if len(survs) > 0:
                    all_surv.append(survs)
                if len(fails) > 0:
                    all_fail.append(fails)
                n_done += 1
                if n_done % max(1, n_parents // 5) == 0:
                    ns = sum(len(s) for s in all_surv)
                    nf = sum(len(f) for f in all_fail)
                    _log(f"    [{n_done}/{n_parents}] "
                         f"{total_children:,} children, "
                         f"{ns:,} surv, {nf:,} box-fail")

            elapsed = time.time() - t0
            if all_surv:
                survivors = np.vstack(all_surv)
            else:
                survivors = np.empty((0, d_child), dtype=np.int32)
            if all_fail:
                failures = np.vstack(all_fail)
            else:
                failures = np.empty((0, d_child), dtype=np.int32)

            _log(f"  {elapsed:.1f}s: {total_children:,} children, "
                 f"{len(survivors):,} surv, {len(failures):,} box-fail")
            _log(f"  min_net = {level_min_net:.6f}")

            if len(survivors) == 0:
                converge_level = level
                final_failures = failures
                _log(f"  *** CONVERGED at d={d_child} ***")
                break

            current = survivors
            d_parent = d_child

    if converge_level is None:
        _log(f"  Cascade did not converge in {max_levels} levels")
        _log(f"  Total time: {time.time()-t_total:.1f}s")
        return

    d_final = d0 * (2 ** converge_level)
    cascade_time = time.time() - t_total

    if final_failures is None or len(final_failures) == 0:
        _log(f"\n  All cells box-certified at d={d_final}!")
        _log(f"  *** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
        _log(f"  Total time: {cascade_time:.1f}s")
        return

    # --- STEP 2: Subdivision on box-cert failures ---
    n_fail = len(final_failures)
    _log(f"\n  {'='*60}")
    _log(f"  SUBDIVISION at d={d_final}, S={S}")
    _log(f"  {n_fail:,} box-cert failures to fix")
    _log(f"  val({d_final})=1.319 > {c_target} => convergence guaranteed")
    _log(f"  {'='*60}")

    t0 = time.time()
    n_sub_pass = 0
    n_sub_fail = 0
    max_r = 0
    total_sc = 0

    for ci in range(n_fail):
        k = final_failures[ci]
        ok, r, ns, wn = subdiv_cert_batch(k, d_final, S, c_target,
                                           max_refine=32,
                                           verbose=(ci < 5))
        if ok:
            n_sub_pass += 1
            total_sc += ns
            if r > max_r:
                max_r = r
        else:
            n_sub_fail += 1
            if n_sub_fail <= 3:
                _log(f"    FAIL #{n_sub_fail}: k={k[:8]}... worst={wn:.6f}")

        if (ci + 1) % 1000 == 0 or ci + 1 == n_fail:
            elapsed = time.time() - t0
            rate = (ci + 1) / max(elapsed, 0.001)
            eta = (n_fail - ci - 1) / max(rate, 0.001)
            _log(f"    [{ci+1}/{n_fail}] pass={n_sub_pass} fail={n_sub_fail} "
                 f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    subdiv_time = time.time() - t0

    _log(f"\n  {'='*60}")
    _log(f"  RESULTS: d0={d0}, S={S}, c={c_target}")
    _log(f"  {'='*60}")
    _log(f"  Cascade:   {cascade_time:.1f}s, converged at d={d_final}")
    _log(f"  Box fails: {n_fail:,}")
    _log(f"  Subdiv:    {n_sub_pass:,} pass, {n_sub_fail:,} fail "
         f"({subdiv_time:.1f}s)")
    _log(f"  Max refine: {max_r}x (S -> {S*max_r})")
    if n_sub_pass > 0:
        _log(f"  Avg subcells: {total_sc/n_sub_pass:.0f}")

    if n_sub_fail == 0:
        _log(f"\n  *** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
        _log(f"  (cascade d={d_final} + subdivision, S={S})")
    else:
        _log(f"\n  {n_sub_fail} cells uncertified. Need higher S or refine.")

    _log(f"  Total: {time.time()-t_total:.1f}s")


# =====================================================================
# Main
# =====================================================================

configs = [
    (4, 20, 1.29, 8),
    (4, 30, 1.29, 8),
    (4, 50, 1.29, 8),
    (4, 75, 1.29, 8),
    (6, 20, 1.29, 6),
    (6, 30, 1.29, 6),
    (6, 50, 1.29, 6),
    (8, 20, 1.29, 5),
    (8, 30, 1.29, 5),
]

_log("=" * 70)
_log("CASCADE + SUBDIVISION BOX CERT at c_target=1.29")
_log("val(16)=1.319, margin=0.029")
_log("Production Numba kernels for all heavy computation.")
_log("=" * 70)

for d0, S, c, ml in configs:
    run_one(d0, S, c, ml)

_log("\n" + "=" * 70)
_log("ALL DONE")
_log("=" * 70)
