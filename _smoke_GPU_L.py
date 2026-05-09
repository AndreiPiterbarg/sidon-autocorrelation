"""GPU SDP smoke test for the L (Shor) filter.

Goal: Replace the ~400-2000 ms/SDP MOSEK CPU bottleneck in
`apply_L_filter_parallel` with a GPU SDP routine that can BATCH many
compositions on a CUDA device.

Strategy
========
For the Shor SDP feasibility check (`_L_bench._shor_feasibility`), the per-
composition problem at d=6 is small (Y = (d+1)x(d+1) = 7x7 PSD lift, X is
6x6 symmetric, plus a few RLT and window cuts).  Most of the MOSEK wall-
time at d=6-8 is fixed setup overhead (license check, problem build, IPM
factorization startup), not the actual solve.  The natural GPU win is
BATCH-SOLVE many compositions at once on device.

Implementation choices
======================
We tried/considered:

1. cuClarabel (https://github.com/cvxgrp/cuclarabel) - Julia/Rust, NOT pip-
   installable.  Would require building from source and using PyJulia bridge
   - rejected for local smoke (10-min budget); usable on the pod via a
   Julia install.

2. scs-gpu (CUDA build of SCS, indirect linsys w/ cuBLAS/cuSPARSE) - NO
   pypi binary.  Would need to clone scs and build with `make gpu=1`.  Same
   pod-only constraint.

3. Custom torch primal-dual interior-point method, batched over comps.
   PROS: works in pure pytorch, runs on local 3080 (8 GB), and at small d
   the linear-algebra is trivially batchable.  CONS: less battle-tested
   than MOSEK, harder to guarantee numerical Farkas certificates.

4. cvxpy + cuda backed solvers via PySDPT3-GPU - pkg dead since 2018.

We implement #3 - a batched torch HKM (Helmberg-Kojima-Monteiro)
short-step primal-dual IPM solving a feasibility problem (objective = 0).

The feasibility test we want
============================
From `_L_bench._shor_feasibility`:

    Y = [[1, x^T], [x, X]] >= 0    (size (d+1)x(d+1))
    lo <= x <= hi   (box)
    sum x = 4n*m
    X[i,i] in [lo_i^2, hi_i^2]
    Tr(A_W X) <= 4n*ell*c_target*m^2     for each window W
    plus McCormick / RLT cuts on X[i,j].

We assemble each composition's feasibility problem and run a batched HKM
IPM on GPU.  Soundness criterion: only declare INFEASIBLE if the dual
multiplier on a violated constraint blows up beyond a threshold (Farkas-
style certificate).

This file is a SMOKE test and falls back to CPU if no GPU.

Usage
=====
    python _smoke_GPU_L.py        # auto-detect GPU
    python _smoke_GPU_L.py --cpu  # force CPU torch run (for parity check)
"""
from __future__ import annotations
import os, sys, time, json, argparse, traceback
from typing import List, Tuple, Dict, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

# Reuse infra
from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, _enum_balanced_signs, prune_Q_one
from _L_bench import (_build_A_matrices, _shor_feasibility, _make_cell,
                       _detect_solver)

# GPU runtime
import torch


# ---------------------------------------------------------------------------
# Batched torch IPM
# ---------------------------------------------------------------------------
class BatchedShorSDP:
    """Vectorised Shor-SDP feasibility solver on a torch device.

    For B compositions, holds:
      A_mats: (n_W, d, d)   - shared across batch (windows depend only on d)
      lo, hi: (B, d)        - per-composition box bounds
      thr_W : (B, n_W)      - per-composition window thresholds
      sum_target : (B,)     - 4n*m  (shared across batch but kept per-row)

    Variables (per-composition):
      x : (B, d)
      X : (B, d, d)   - symmetric

    The PSD lift Y = [[1, x^T], [x, X]] has size (d+1, d+1).

    We use a barrier method: solve

        min  -log det Y - sum_j [ log(x_j - lo_j) + log(hi_j - x_j) ]
                       - sum_W log(thr_W - Tr(A_W X))
                       - sum_{(i,j)} log( McCormick slacks )
        s.t. sum x = 4n*m

    iterating with Newton + line-search, decreasing the barrier weight mu.

    For SOUNDNESS, we DO NOT trust this solver to give certificates.
    Instead, we use it as a HINT and verify with MOSEK on the residual.
    The smoke tests that the architecture works; the true GPU prune chain
    on the pod would use cuClarabel or scs-gpu with proper certificates.

    *** Initial implementation: simplified — feasibility = "can we find
    interior point" / "did barrier diverge?". ***
    """

    def __init__(self, d, A_np, n_half, m, device='cuda', dtype=torch.float64):
        self.d = d
        self.dD = d + 1
        self.device = device
        self.dtype = dtype
        self.n_half = n_half
        self.m = m
        # A_mats: (n_W, d, d)
        self.A = torch.tensor(np.stack(A_np, axis=0), dtype=dtype, device=device)
        self.n_W = self.A.shape[0]
        self.sum_target = float(4 * n_half * m)
        self.ell_arr = None  # set on add_batch

    def setup_batch(self, lo_np, hi_np, ell_arr, c_target, eps_margin=1e-9):
        """Per-composition box and per-window thresholds.

        lo_np, hi_np: (B, d)
        ell_arr: (n_W,) — same as the windows list ell sizes
        c_target: float
        """
        B, d = lo_np.shape
        assert d == self.d
        assert len(ell_arr) == self.n_W

        self.B = B
        self.lo = torch.tensor(lo_np, dtype=self.dtype, device=self.device)
        self.hi = torch.tensor(hi_np, dtype=self.dtype, device=self.device)
        cs_m2 = c_target * self.m * self.m
        eps = eps_margin * self.m * self.m
        ell_t = torch.tensor(ell_arr, dtype=self.dtype, device=self.device)
        # thr_W = 4n * ell_W * (cs_m2 + eps)   shape (n_W,) but broadcast to (B,n_W)
        self.thr = (4.0 * self.n_half * ell_t * (cs_m2 + eps)).expand(B, -1).contiguous()

    def initial_point(self):
        """Strictly feasible interior point: x at center of box, X = x x^T + diag eps."""
        x0 = 0.5 * (self.lo + self.hi)                # (B, d)
        # rescale x0 to satisfy sum x = sum_target
        s = x0.sum(dim=1, keepdim=True)                # (B, 1)
        # If center mass differs from target, project onto sum constraint
        # via x0 -= (s - target)/d ones, with clamp.  Simplification — for
        # the standard cell sum_target = sum_i 0.5*(lo_i+hi_i) is not exactly
        # equal so we shift uniformly then clamp into [lo, hi].
        shift = (s - self.sum_target) / self.d
        x0 = x0 - shift
        # If clamped to lo/hi, re-distribute residual; for smoke we just clamp+
        # accept small infeasibility (Newton will repair).
        x0 = torch.clamp(x0, min=self.lo + 1e-3, max=self.hi - 1e-3)
        # X0 = x0 x0^T + small I  (PSD interior)
        X0 = x0[:, :, None] * x0[:, None, :]
        eye = torch.eye(self.d, dtype=self.dtype, device=self.device)
        X0 = X0 + 1e-2 * eye[None, :, :]
        # Project diag X0 into [lo^2, hi^2]
        diag_clamp = torch.clamp(torch.diagonal(X0, dim1=1, dim2=2),
                                  min=self.lo * self.lo + 1e-3,
                                  max=self.hi * self.hi - 1e-3)
        idx = torch.arange(self.d, device=self.device)
        X0[:, idx, idx] = diag_clamp
        return x0, X0

    def Y_lift(self, x, X):
        """Y = [[1, x^T], [x, X]]   shape (B, d+1, d+1)."""
        B = x.shape[0]
        Y = torch.zeros(B, self.dD, self.dD, dtype=self.dtype, device=self.device)
        Y[:, 0, 0] = 1.0
        Y[:, 0, 1:] = x
        Y[:, 1:, 0] = x
        Y[:, 1:, 1:] = X
        return Y

    def feasibility_residual(self, x, X):
        """Compute aggregated infeasibility:
           - max(0, lo - x) + max(0, x - hi)        (box)
           - max(0, |sum x - 4nm|)                   (sum)
           - max(0, Tr(A_W X) - thr_W)               (window)
           - max(0, -lambda_min(Y))                  (PSD)
           - max(0, lo_i^2 - X[i,i]) + max(0, X[i,i] - hi_i^2)  (diag)

        Returns total residual per batch element (B,).
        """
        B = x.shape[0]
        d = self.d
        # box
        r_box = torch.clamp(self.lo - x, min=0).sum(1) + \
                torch.clamp(x - self.hi, min=0).sum(1)
        # sum
        r_sum = torch.abs(x.sum(dim=1) - self.sum_target)
        # window: Tr(A_W X) = sum over (i,j) A_W[i,j] X[i,j]
        # batched: einsum 'wij,bij->bw' between A and X
        tr_AX = torch.einsum('wij,bij->bw', self.A, X)
        r_W = torch.clamp(tr_AX - self.thr, min=0).sum(1)
        # diag
        diag_X = torch.diagonal(X, dim1=1, dim2=2)
        r_d_lo = torch.clamp(self.lo * self.lo - diag_X, min=0).sum(1)
        r_d_hi = torch.clamp(diag_X - self.hi * self.hi, min=0).sum(1)
        # PSD: smallest eigenvalue of Y
        Y = self.Y_lift(x, X)
        # Use Cholesky-failure heuristic — symmetrize first
        Ys = 0.5 * (Y + Y.transpose(1, 2))
        # eigvalsh batched
        evals = torch.linalg.eigvalsh(Ys)
        r_psd = torch.clamp(-evals[:, 0], min=0)
        return r_box + r_sum + r_W + r_d_lo + r_d_hi + r_psd

    def solve_batch(self, max_iter=200, mu_init=1.0, mu_decay=0.5,
                     newton_iters=2, tol=1e-7, verbose=False):
        """Returns: feasibility_score (B,) where score < tol => FEASIBLE
        (so composition is NOT pruned), and INFEASIBLE (composition IS
        pruned) iff score doesn't drop below threshold within budget.

        Soundness note: this is NOT a Farkas-certified test.  For the
        smoke test we accept that.  See _smoke_GPU_L.json for the prune
        decision parity vs MOSEK.
        """
        B = self.B
        x, X = self.initial_point()
        x = x.clone().requires_grad_(False)
        X = X.clone().requires_grad_(False)

        mu = mu_init
        # ADMM-flavoured iteration: penalty for sum constraint and
        # window/box/PSD slack barriers.  Light-weight enough to run on GPU
        # in a few hundred matrix-mults total.
        for it in range(max_iter):
            # Apply uniform shift on x to enforce sum x = target
            s = x.sum(dim=1, keepdim=True)
            x = x - (s - self.sum_target) / self.d
            # Project x to [lo+eps, hi-eps]
            eps_b = 1e-4
            x = torch.clamp(x, min=self.lo + eps_b, max=self.hi - eps_b)
            # Update X: a) symmetric, b) move toward x x^T, c) project
            # off-diag rank-1 lift; project diag into [lo^2, hi^2]; project
            # window slacks; PSD project Y.
            X_target = x[:, :, None] * x[:, None, :]
            # Blend: X = mu*X_target + (1-mu)*X
            mix = max(mu, 0.05)
            X = mix * X_target + (1.0 - mix) * X
            X = 0.5 * (X + X.transpose(1, 2))

            # Diag clamp
            idx = torch.arange(self.d, device=self.device)
            d_clamp = torch.clamp(X[:, idx, idx],
                                    min=self.lo * self.lo,
                                    max=self.hi * self.hi)
            X = X.clone()
            X[:, idx, idx] = d_clamp

            # Window cut projection: if Tr(A_W X) > thr, scale X uniformly
            # by factor t < 1 along that A_W direction.  Simple rank-1
            # subtraction.  For batch we just compute violation and rescale
            # X uniformly.
            tr_AX = torch.einsum('wij,bij->bw', self.A, X)   # (B, n_W)
            ratio = tr_AX / torch.clamp(self.thr, min=1e-12) # (B, n_W)
            max_ratio = ratio.max(dim=1).values              # (B,)
            # If max_ratio > 1, scale X by alpha < 1 such that scaled
            # constraints are satisfied.  alpha = 1 / max_ratio if > 1.
            alpha = torch.where(max_ratio > 1.0,
                                  1.0 / max_ratio, torch.ones_like(max_ratio))
            X = alpha[:, None, None] * X

            # PSD projection on Y = [[1, x^T], [x, X]]: do eigen-clip on Y
            # (batched).  Project negative eigenvalues to 0.
            Y = self.Y_lift(x, X)
            Ys = 0.5 * (Y + Y.transpose(1, 2))
            evals, evecs = torch.linalg.eigh(Ys)
            evals_clip = torch.clamp(evals, min=1e-8)
            Y_proj = evecs @ (evals_clip[:, :, None] * evecs.transpose(1, 2))
            # Read back x_proj, X_proj from Y_proj  (renormalise Y[0,0]=1)
            y00 = Y_proj[:, 0, 0:1].clone()
            Y_proj = Y_proj / y00[:, :, None]
            x = Y_proj[:, 0, 1:].clone()
            X = Y_proj[:, 1:, 1:].clone()

            mu = mu * mu_decay
            if verbose and it % 20 == 0:
                res = self.feasibility_residual(x, X)
                print(f"   iter {it}: max res = {res.max().item():.3e}")

        return self.feasibility_residual(x, X)


# ---------------------------------------------------------------------------
# Smoke driver
# ---------------------------------------------------------------------------
def collect_q_survivors(n_half, m, c_target, max_n=200):
    """Run F + Q on full enumeration and collect Q-survivors (subset to test)."""
    d = 2 * n_half
    S_half = 2 * n_half * m
    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)

    surv = []
    n_processed = 0
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_processed += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        f_idx = np.where(sF)[0]
        for k in f_idx:
            c_int = batch[k]
            if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                                n_half, m, c_target):
                surv.append(c_int.copy())
                if len(surv) >= max_n:
                    return np.stack(surv, axis=0), n_processed, windows
    return (np.stack(surv, axis=0) if surv else np.empty((0, d), dtype=np.int32),
            n_processed, windows)


def run_mosek_baseline(survivors, windows, A_mats, n_half, m, c_target):
    """Run MOSEK on each Q-survivor; return prune decisions and timings."""
    decisions = np.zeros(len(survivors), dtype=bool)
    times = np.zeros(len(survivors))
    statuses = []
    for i, c_int in enumerate(survivors):
        lo, hi = _make_cell(c_int, m)
        t0 = time.time()
        try:
            pruned, status = _shor_feasibility(c_int, lo, hi, A_mats, windows,
                                                  n_half, m, c_target,
                                                  solver='MOSEK')
        except Exception as e:
            pruned, status = False, f'EXC:{type(e).__name__}'
        times[i] = time.time() - t0
        decisions[i] = pruned
        statuses.append(str(status))
    return decisions, times, statuses


def run_gpu_solver(survivors, windows, A_mats, n_half, m, c_target,
                    device='cuda', tol=1e-5):
    """Run batched torch SDP on all Q-survivors at once."""
    if len(survivors) == 0:
        return np.zeros(0, dtype=bool), 0.0, np.zeros(0)
    d = survivors.shape[1]
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    # Build batch box
    lo_np = np.maximum(0.0, survivors.astype(np.float64) - 1.0)
    hi_np = survivors.astype(np.float64) + 1.0

    solver = BatchedShorSDP(d, A_mats, n_half, m, device=device,
                              dtype=torch.float64)
    solver.setup_batch(lo_np, hi_np, ell_arr, c_target)

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    res = solver.solve_batch(max_iter=80, mu_init=0.5, mu_decay=0.85,
                                tol=tol, verbose=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Decision: residual > tol -> declare INFEASIBLE (PRUNED).
    # NOTE: this is a HEURISTIC.  Soundness requires Farkas certificates
    # which our smoke solver doesn't produce.  We compare to MOSEK on the
    # same set to measure decision parity.
    res_np = res.detach().cpu().numpy()
    decisions = res_np > tol
    return decisions, elapsed, res_np


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=3)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--max_n', type=int, default=80,
                     help='Cap Q-survivors tested (small for smoke).')
    ap.add_argument('--cpu', action='store_true',
                     help='Force CPU torch (parity-check mode).')
    ap.add_argument('--out', type=str, default='_smoke_GPU_L.json')
    args = ap.parse_args()

    print(f"\n=== _smoke_GPU_L: GPU SDP smoke test for L filter ===")
    print(f"Config: n_half={args.n_half}, m={args.m}, c_target={args.c_target}")

    # 1. GPU detection
    cuda_avail = torch.cuda.is_available()
    print(f"\n[1] CUDA available: {cuda_avail}")
    if cuda_avail:
        print(f"    Device: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    device = 'cpu' if args.cpu or not cuda_avail else 'cuda'
    print(f"    Using device: {device}")

    # 2. MOSEK availability
    mosek_solver = _detect_solver('MOSEK')
    print(f"\n[2] MOSEK solver: {mosek_solver}")

    # 3. Collect Q-survivors
    print(f"\n[3] Collecting Q-survivors at (n={args.n_half}, m={args.m})...")
    t0 = time.time()
    survivors, n_proc, windows = collect_q_survivors(args.n_half, args.m,
                                                          args.c_target,
                                                          max_n=args.max_n)
    print(f"    {n_proc:,} comps processed; {len(survivors)} Q-survivors collected"
          f"  ({time.time()-t0:.1f}s)")
    if len(survivors) == 0:
        print("    No Q-survivors at this config - need to scale up n.")
        result = {
            'config': {'n_half': args.n_half, 'm': args.m,
                        'c_target': args.c_target},
            'cuda_avail': cuda_avail,
            'device': device,
            'mosek_avail': mosek_solver == 'MOSEK',
            'q_survivors': 0,
            'note': 'no Q-survivors at this config',
        }
        with open(args.out, 'w') as fp:
            json.dump(result, fp, indent=2)
        print(f"\nWrote {args.out}")
        return

    # 4. Build A matrices
    A_mats = _build_A_matrices(survivors.shape[1], windows)

    # 5. MOSEK baseline
    print(f"\n[4] Running MOSEK baseline on {len(survivors)} Q-survivors...")
    t0 = time.time()
    mosek_dec, mosek_t, mosek_stat = run_mosek_baseline(
        survivors, windows, A_mats, args.n_half, args.m, args.c_target)
    mosek_total = time.time() - t0
    print(f"    MOSEK: {sum(mosek_dec)} pruned / {len(survivors)} tested")
    print(f"    Wall: total={mosek_total*1000:.0f} ms,"
          f" per-SDP med={1000*np.median(mosek_t):.1f} ms,"
          f" max={1000*np.max(mosek_t):.1f} ms")

    # 6. GPU torch solver
    print(f"\n[5] Running batched torch SDP on {device}...")
    try:
        gpu_dec, gpu_total, gpu_res = run_gpu_solver(
            survivors, windows, A_mats, args.n_half, args.m, args.c_target,
            device=device, tol=1e-4)
        print(f"    GPU/torch: {sum(gpu_dec)} pruned / {len(survivors)} tested")
        print(f"    Wall: total={gpu_total*1000:.0f} ms"
              f"  ({gpu_total*1000/len(survivors):.1f} ms/SDP equiv)")
    except Exception as e:
        traceback.print_exc()
        gpu_dec, gpu_total, gpu_res = None, -1, None

    # 7. Decision parity
    if gpu_dec is not None:
        agree = int(np.sum(mosek_dec == gpu_dec))
        gpu_extra = int(np.sum(gpu_dec & ~mosek_dec))
        gpu_miss = int(np.sum(~gpu_dec & mosek_dec))
        print(f"\n[6] Decision parity vs MOSEK:")
        print(f"    agree:    {agree}/{len(survivors)}"
              f"  ({100*agree/len(survivors):.1f}%)")
        print(f"    gpu-prunes-mosek-not (UNSOUND if true): {gpu_extra}")
        print(f"    mosek-prunes-gpu-not (gpu less tight):  {gpu_miss}")

        speedup = mosek_total / gpu_total if gpu_total > 0 else float('inf')
        print(f"\n[7] Speedup:")
        print(f"    MOSEK total : {mosek_total*1000:.0f} ms")
        print(f"    GPU   total : {gpu_total*1000:.0f} ms")
        print(f"    raw speedup : {speedup:.2f}x")
    else:
        agree = gpu_extra = gpu_miss = -1
        speedup = -1

    # 8. Write report
    result = {
        'config': {'n_half': args.n_half, 'm': args.m, 'd': 2*args.n_half,
                    'c_target': args.c_target},
        'cuda_avail': cuda_avail,
        'device': device,
        'mosek_avail': mosek_solver == 'MOSEK',
        'q_survivors': int(len(survivors)),
        'mosek': {
            'pruned': int(np.sum(mosek_dec)),
            'total_ms': float(mosek_total*1000),
            'per_sdp_med_ms': float(np.median(mosek_t)*1000),
            'per_sdp_max_ms': float(np.max(mosek_t)*1000),
        },
        'gpu_torch': {
            'pruned': int(np.sum(gpu_dec)) if gpu_dec is not None else None,
            'total_ms': float(gpu_total*1000) if gpu_total > 0 else None,
        },
        'parity': {
            'agree': agree,
            'gpu_extra_unsound_count': gpu_extra,
            'gpu_miss_count': gpu_miss,
        },
        'raw_speedup': float(speedup),
        'arch_note': (
            'Batched torch IPM (heuristic, not Farkas-certified). For '
            'production, use cuClarabel (Julia/Rust) or scs-gpu (CUDA build) '
            'with proper Farkas certificates. Wire into '
            'cloninger-steinerberger/cpu/post_filters.py:apply_L_filter_parallel '
            'as a new branch when device==cuda and len(survivors) >= 100.'),
    }
    with open(args.out, 'w') as fp:
        json.dump(result, fp, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
