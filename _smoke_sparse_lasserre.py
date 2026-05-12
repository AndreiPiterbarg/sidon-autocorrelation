"""Smoke test: correlative sparsity for per-composition Lasserre SDP.

Builds the correlation graph (nodes = variable indices 0..d-1, edges (i,j) iff
some window W = (ell, s_lo) contains the pair (i.e. s_lo <= i+j <= s_lo+ell-2)).

Question: is the graph complete?  If yes, no chordal sparsity decomposition.

This is the per-cell SDP from _L_bench.py:_lasserre2_feasibility (lines 247-381).
Window set comes from _Q_bench.py:_build_windows: ell in [2, 2d], s_lo in
[0, conv_len - n_cv].  Sums i+j range over [s_lo, s_lo+ell-2].
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _Q_bench import _build_windows


def correlation_graph(d, windows):
    """edges[(i,j)] = True iff some window covers the pair (i,j)."""
    adj = np.zeros((d, d), dtype=bool)
    for (ell, s_lo) in windows:
        s_hi = s_lo + ell - 2
        for i in range(d):
            for j in range(d):
                if s_lo <= i + j <= s_hi:
                    adj[i, j] = True
                    adj[j, i] = True
    return adj


def report(d):
    windows, _ = _build_windows(d)
    adj = correlation_graph(d, windows)

    n_edges = int(np.sum(adj) - np.trace(adj)) // 2  # undirected, no self-loop
    n_max = d * (d - 1) // 2
    is_complete_offdiag = (n_edges == n_max)
    # Diagonal also (i,i) self-loop: every window with s_lo<=2i<=s_hi has it.
    diag_full = bool(np.all(np.diag(adj)))

    # Density
    density = n_edges / max(1, n_max)

    # Largest clique = d (whole vertex set) iff complete.
    print(f"=== d={d} ===")
    print(f"  num windows         : {len(windows)}")
    print(f"  num undirected edges: {n_edges} / {n_max}")
    print(f"  density             : {density:.4f}")
    print(f"  diagonal full       : {diag_full}")
    print(f"  graph is COMPLETE   : {is_complete_offdiag and diag_full}")

    # Even if complete, check if any single window induces sparsity
    # (sparsity could be exploited if we DECOMPOSE windows by pair).
    # For SDP, the relevant constraint is: x_i x_j appears as a single y_{e_i+e_j}
    # in the moment matrix.  Since every (i,j) appears in some window, every
    # off-diagonal of M_2 is constrained, so M_2 is dense — no decomposition.

    # Per-window sparsity (just for curiosity)
    per_w_density = []
    for (ell, s_lo) in windows:
        s_hi = s_lo + ell - 2
        npairs = sum(1 for i in range(d) for j in range(d)
                     if s_lo <= i + j <= s_hi)
        per_w_density.append(npairs / (d * d))
    print(f"  per-window pair density: min={min(per_w_density):.3f} "
          f"max={max(per_w_density):.3f} mean={np.mean(per_w_density):.3f}")

    return is_complete_offdiag and diag_full


if __name__ == '__main__':
    t0 = time.time()
    for d in [4, 6, 8, 10, 12, 14]:
        cmpl = report(d)
    print(f"\nelapsed: {time.time() - t0:.2f}s")
