"""Cluster-SDP smoke test: joint Shor SDP over Hamming-1 cluster of cells.

Idea: a SINGLE SDP over the convex-hull box of a cluster of geometrically
adjacent cells gives a SOUND lower bound on TV minimum across the union.
If cluster_LB > c_target then EVERY cell in the cluster is prunable.

Soundness:  min over union <= min over each cell, so cluster_LB <= each
cell's true LB, but cluster's SDP-INFEASIBLE => union infeasible (no
feasible x in any cell with all TV_W <= c_target). Therefore EVERY cell
in cluster gets pruned.

Compare vs per-cell Shor SDP (`_L_bench.prune_L_one`):
  (a) cells individually L-prunable
  (b) cells the cluster prunes that L misses

Config: n_half=3, m=10, c_target=1.28, d=6.

Output: _smoke_cluster_SDP.json
"""
from __future__ import annotations
import os, sys, time, json
from itertools import product
from collections import defaultdict
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows
from _L_bench import _build_A_matrices, _detect_solver, prune_L_one


def _hamming1_neighbor_pairs(survivors_arr):
    """Return list of (i, j) such that survivors[i] and survivors[j] differ
    by a palindromic minimal mass-conserving move:

      Move 1 (interior, 4 coord change): swap +/-1 between (a, b) AND mirrors
        c'[a] = c[a]-1, c'[b] = c[b]+1, c'[d-1-a] = c[d-1-a]-1, c'[d-1-b] = c[d-1-b]+1
        (skip if a == d-1-b, since that would cancel)

      Move 2 (cross-mirror, 2 coord change): a + (d-1-a) self-pair vs b + (d-1-b)
        c'[a] -= 1, c'[d-1-a] -= 1, c'[b] += 1, c'[d-1-b] += 1
        (when a != d-1-a and b != d-1-b and {a,d-1-a} ∩ {b,d-1-b} = ∅)

    Both preserve palindromy (c[i]=c[d-1-i]) and total mass.
    Each move is the smallest palindromic L1-distance-2 step."""
    n = len(survivors_arr)
    key_to_idx = {tuple(c.tolist()): i for i, c in enumerate(survivors_arr)}
    d = survivors_arr.shape[1]
    pairs_set = set()

    for i, c in enumerate(survivors_arr):
        # Move 1: shift between (a,b) and shift mirrored (d-1-a, d-1-b)
        # i.e. c[a]-=1, c[b]+=1, c[d-1-a]-=1, c[d-1-b]+=1
        # If a + (d-1-a) overlaps with b + (d-1-b), special case.
        for a in range(d):
            ar = d - 1 - a
            for b in range(d):
                br = d - 1 - b
                if a == b:
                    continue
                # Compute delta vector
                delta = np.zeros(d, dtype=np.int32)
                delta[a] -= 1
                delta[b] += 1
                delta[ar] -= 1
                delta[br] += 1
                # Skip null move
                if not delta.any():
                    continue
                neighbor = c + delta
                if (neighbor < 0).any():
                    continue
                key = tuple(neighbor.tolist())
                if key in key_to_idx:
                    j = key_to_idx[key]
                    if j != i:
                        pairs_set.add((min(i, j), max(i, j)))
    return sorted(pairs_set)


def _greedy_cluster(survivors_arr, pairs):
    """Greedy clustering: pick a center, group all Hamming-1 neighbors that
    are F-survivors. A composition can only belong to one cluster.

    Returns: list of clusters (list of indices into survivors_arr),
             and the unclustered (singleton) indices.
    """
    n = len(survivors_arr)
    # adjacency
    adj = defaultdict(set)
    for (i, j) in pairs:
        adj[i].add(j)
        adj[j].add(i)

    # Order centers by degree (highest first) to grow large clusters
    deg = [(len(adj[i]), i) for i in range(n)]
    deg.sort(reverse=True)

    used = [False] * n
    clusters = []
    for (_, i) in deg:
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in adj[i]:
            if not used[j]:
                cluster.append(j)
                used[j] = True
        clusters.append(cluster)
    return clusters


def _cluster_box(cluster_compositions):
    """Convex-hull box: lo[k] = max(0, min_c(c[k]) - 1), hi[k] = max_c(c[k]) + 1.
    Each cell is c-1 <= x <= c+1, so union's hull is min(c)-1 <= x <= max(c)+1."""
    cs = np.array(cluster_compositions)
    c_min = cs.min(axis=0)
    c_max = cs.max(axis=0)
    lo = np.maximum(0.0, c_min - 1.0)
    hi = c_max + 1.0
    return lo, hi


def _shor_cluster_feasibility(lo, hi, A_mats, windows, n_half, m, c_target,
                                solver='MOSEK', tol=1e-9, eps_margin=1e-9):
    """Per-cluster Shor SDP feasibility. Same as _shor_feasibility but with
    a (potentially larger) box [lo, hi] (the convex hull of the cluster).

    Returns (cluster_pruned: bool, status_str).
    """
    import cvxpy as cp

    d = len(lo)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)

    ones11 = np.ones((1, 1))
    Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                  [cp.reshape(x, (d, 1), order='C'), X]])

    cons = [Y >> 0]
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == nm]

    for i in range(d):
        cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
        cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
        cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]

    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]

    for A_mat, (ell, _) in zip(A_mats, windows):
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        if solver == 'MOSEK':
            prob.solve(solver='MOSEK', verbose=False,
                        mosek_params={
                            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                        })
        elif solver == 'CLARABEL':
            prob.solve(solver='CLARABEL', verbose=False,
                        eps_abs=tol, eps_rel=tol, max_iter=300)
        else:
            return False, f'NO_SOLVER:{solver}'
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'

    return prob.status == 'infeasible', prob.status


def main():
    n_half, m, c_target = 3, 10, 1.28
    d = 2 * n_half
    S_half = 2 * n_half * m

    print(f"\n=== Cluster-SDP smoke test ===")
    print(f"n_half={n_half}, m={m}, c_target={c_target}, d={d}")

    solver = _detect_solver('MOSEK')
    print(f"solver = {solver}")

    windows, _ = _build_windows(d)
    A_mats = _build_A_matrices(d, windows)
    print(f"n_windows = {len(windows)}")

    # JIT warmup
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    # Get F-survivors
    print("\n--- Step 1: Get F-survivors ---")
    t0 = time.time()
    survivors = []
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        sF = prune_F(batch, n_half, m, c_target)
        f_idx = np.where(sF)[0]
        for idx in f_idx:
            survivors.append(batch[idx].copy())
    survivors_arr = np.array(survivors, dtype=np.int32)
    print(f"  F-survivors: {len(survivors_arr)} (in {time.time()-t0:.2f}s)")

    # Cluster by Hamming-1 adjacency
    print("\n--- Step 2: Hamming-1 clustering ---")
    t0 = time.time()
    pairs = _hamming1_neighbor_pairs(survivors_arr)
    clusters = _greedy_cluster(survivors_arr, pairs)
    n_in_clusters = sum(len(c) for c in clusters if len(c) >= 2)
    n_clusters_ge2 = sum(1 for c in clusters if len(c) >= 2)
    print(f"  pairs: {len(pairs)}")
    print(f"  total clusters: {len(clusters)}")
    print(f"  clusters of size >= 2: {n_clusters_ge2}")
    print(f"  cells in size>=2 clusters: {n_in_clusters}")
    cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
    print(f"  largest cluster sizes: {cluster_sizes[:10]}")
    print(f"  ({time.time()-t0:.2f}s)")

    # Run per-cell L SDP and per-cluster SDP
    print("\n--- Step 3: Per-cell L SDP (baseline) on all F-survivors ---")
    t0 = time.time()
    cell_pruned = np.zeros(len(survivors_arr), dtype=bool)
    for i, c_int in enumerate(survivors_arr):
        pruned, status = prune_L_one(c_int, A_mats, windows, n_half, m,
                                       c_target, solver=solver, order=1)
        cell_pruned[i] = pruned
    n_cell_pruned = int(cell_pruned.sum())
    print(f"  cells L-pruned individually: {n_cell_pruned} / {len(survivors_arr)}"
          f"  ({time.time()-t0:.2f}s)")

    print("\n--- Step 4: Per-cluster joint SDP ---")
    t0 = time.time()
    cluster_results = []
    cells_pruned_by_cluster = np.zeros(len(survivors_arr), dtype=bool)
    extra_over_cell = 0
    n_clusters_pruned = 0

    for cl_idx, cluster in enumerate(clusters):
        if len(cluster) < 2:
            continue
        cluster_comps = [survivors_arr[i] for i in cluster]
        lo, hi = _cluster_box(cluster_comps)
        pruned, status = _shor_cluster_feasibility(lo, hi, A_mats, windows,
                                                       n_half, m, c_target,
                                                       solver=solver)
        if pruned:
            n_clusters_pruned += 1
            for i in cluster:
                cells_pruned_by_cluster[i] = True
                if not cell_pruned[i]:
                    extra_over_cell += 1
        cluster_results.append({
            'size': len(cluster),
            'pruned': bool(pruned),
            'status': status,
            'lo': lo.tolist(),
            'hi': hi.tolist(),
        })
    print(f"  clusters >=2 tested: {n_clusters_ge2}, pruned: {n_clusters_pruned}")
    print(f"  cells pruned by cluster-SDP: {int(cells_pruned_by_cluster.sum())}")
    print(f"  extra prunes (cluster prunes but cell-SDP doesn't): {extra_over_cell}")
    print(f"  ({time.time()-t0:.2f}s)")

    # Save results
    out = {
        'config': {'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d},
        'solver': solver,
        'n_survivors_F': len(survivors_arr),
        'n_pairs_hamming1': len(pairs),
        'n_clusters_total': len(clusters),
        'n_clusters_ge2': n_clusters_ge2,
        'cluster_sizes_top10': cluster_sizes[:10],
        'n_cells_in_clusters': n_in_clusters,
        'n_cells_L_pruned_individually': n_cell_pruned,
        'n_clusters_pruned_jointly': n_clusters_pruned,
        'n_cells_pruned_by_cluster_SDP': int(cells_pruned_by_cluster.sum()),
        'extra_prunes_cluster_over_cell': extra_over_cell,
        'cluster_details': cluster_results[:50],  # first 50 only
    }
    out_path = os.path.join(_dir, '_smoke_cluster_SDP.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Verdict
    if extra_over_cell > 0:
        print(f"\nVERDICT: CLUSTER_SDP_PRUNES_{extra_over_cell}_EXTRA_CELLS")
    else:
        print(f"\nVERDICT: NO_GAIN")


if __name__ == '__main__':
    main()
