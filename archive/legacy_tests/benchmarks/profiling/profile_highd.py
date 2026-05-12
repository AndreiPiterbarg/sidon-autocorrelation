#!/usr/bin/env python
"""Profile bottlenecks in lasserre_highd.py for d=128, L2.

Measures time and memory for each phase without running MOSEK.
"""
import numpy as np
import time
import sys
import os
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lasserre_fusion import (
    enum_monomials, _make_hash_bases, _hash_monos,
    _build_hash_table, _hash_lookup,
    build_window_matrices, collect_moments,
)
from lasserre_enhanced import _build_banded_cliques, _build_clique_basis

def sizeof_fmt(num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def profile_phase(name, func):
    """Time a function and report peak memory delta."""
    tracemalloc.start()
    t0 = time.time()
    result = func()
    elapsed = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  {name}: {elapsed:.2f}s, peak_mem={sizeof_fmt(peak)}")
    return result, elapsed


def main():
    d = 128
    order = 2
    bandwidth = 16

    print(f"{'='*70}")
    print(f"PROFILING d={d}, order={order} (L2), bw={bandwidth}")
    print(f"{'='*70}\n")

    # Phase 1: Enumerate monomials
    print("Phase 1: Monomial enumeration")

    def enum_all():
        basis = enum_monomials(d, order)
        loc_basis = enum_monomials(d, order - 1)
        consist_mono = enum_monomials(d, 2 * order - 1)
        deg4_mono = enum_monomials(d, 2 * order)
        return basis, loc_basis, consist_mono, deg4_mono

    (basis, loc_basis, consist_mono, deg4_mono), t1 = profile_phase(
        "enum_monomials", enum_all)
    print(f"    basis={len(basis)}, loc_basis={len(loc_basis)}, "
          f"consist_mono={len(consist_mono)}, deg4_mono={len(deg4_mono)}")

    # Phase 2: collect_moments
    print("\nPhase 2: collect_moments")
    def do_collect():
        return collect_moments(d, order, basis, loc_basis, consist_mono)
    (mono_list, idx), t2 = profile_phase("collect_moments", do_collect)
    n_y = len(mono_list)
    print(f"    n_y={n_y:,}")

    # Phase 3: Hash table
    print("\nPhase 3: Build hash table")
    max_comp = 2 * order
    bases = _make_hash_bases(d, max_comp)
    def do_hash_table():
        return _build_hash_table(mono_list, bases)
    (sorted_h, sort_o), t3 = profile_phase("_build_hash_table", do_hash_table)

    # Phase 4: Window matrices
    print("\nPhase 4: Window matrices")
    def do_windows():
        return build_window_matrices(d)
    (windows, M_mats), t4 = profile_phase("build_window_matrices", do_windows)
    print(f"    n_win={len(windows)}")

    # Phase 5: Moment matrix hash lookup (AB_hash)
    print("\nPhase 5: Moment matrix index lookup (B x B)")
    B_arr = np.array(basis, dtype=np.int64)
    B_hash = _hash_monos(B_arr, bases)
    def do_AB_lookup():
        AB_hash = B_hash[:, None] + B_hash[None, :]
        return _hash_lookup(AB_hash, sorted_h, sort_o)
    _, t5 = profile_phase("AB_hash lookup", do_AB_lookup)
    print(f"    queries={len(basis)**2:,} ({len(basis)}x{len(basis)})")

    # Phase 6: Localizing indices (ab_eiej_idx) — THE BIG ONE
    print("\nPhase 6: ab_eiej_idx (localizing 4D index array)")
    LB_arr = np.array(loc_basis, dtype=np.int64)
    AB_loc = LB_arr[:, None, :] + LB_arr[None, :, :]
    AB_loc_hash = _hash_monos(AB_loc, bases)
    n_loc = len(loc_basis)

    def do_abij():
        EE_hash = bases[:, None] + bases[None, :]  # (d, d)
        ABIJ_hash = (AB_loc_hash[:, :, None, None]
                     + EE_hash[None, None, :, :])   # (n_loc, n_loc, d, d)
        print(f"    ABIJ_hash shape: {ABIJ_hash.shape}, "
              f"size={sizeof_fmt(ABIJ_hash.nbytes)}")
        return _hash_lookup(ABIJ_hash, sorted_h, sort_o)
    ab_eiej_idx, t6 = profile_phase("ab_eiej_idx", do_abij)
    print(f"    ab_eiej_idx shape: {ab_eiej_idx.shape}, "
          f"size={sizeof_fmt(ab_eiej_idx.nbytes)}")

    # Phase 7: Localizing per-bin lookups (loc_picks)
    print("\nPhase 7: Per-bin localizing lookups (d={d} bins)")
    def do_loc_picks():
        loc_picks = []
        for i_var in range(d):
            h = AB_loc_hash + bases[i_var]
            picks = _hash_lookup(h, sorted_h, sort_o)
            loc_picks.append(picks)
        return loc_picks
    _, t7 = profile_phase("loc_picks", do_loc_picks)

    # Phase 8: Consistency lookups
    print("\nPhase 8: Consistency index lookups")
    consist_arr = np.array(consist_mono, dtype=np.int64)
    def do_consist():
        consist_hash = _hash_monos(consist_arr, bases)
        consist_idx = _hash_lookup(consist_hash[:, None] * 0 + consist_hash[:, None],
                                    sorted_h, sort_o)  # dummy
        # Real version:
        consist_hash2 = _hash_monos(consist_arr, bases)
        consist_idx2 = _hash_lookup(consist_hash2, sorted_h, sort_o)
        consist_ei_hash = consist_hash2[:, None] + bases[None, :]
        consist_ei_idx = _hash_lookup(consist_ei_hash, sorted_h, sort_o)
        return consist_idx2, consist_ei_idx
    _, t8 = profile_phase("consist lookups", do_consist)
    print(f"    consist_mono={len(consist_mono)}, "
          f"consist_ei shape=({len(consist_mono)}, {d})")

    # Phase 9: Clique construction
    print("\nPhase 9: Clique basis construction")
    cliques = _build_banded_cliques(d, bandwidth)
    def do_clique_basis():
        all_cb = []
        for clique in cliques:
            cb = _build_clique_basis(clique, order, d)
            all_cb.append(cb)
        return all_cb
    all_cb, t9 = profile_phase("clique bases", do_clique_basis)
    print(f"    n_cliques={len(cliques)}, basis_sizes={[len(cb) for cb in all_cb[:3]]}...")

    # Phase 10: _collect_used_indices simulation
    print("\nPhase 10: Simulating _collect_used_indices")
    def do_collect_used():
        """Simulate the set-based collection."""
        used = set()
        # Clique PSD picks
        for c_idx, clique in enumerate(cliques):
            cb_arr = _build_clique_basis(clique, order, d)
            cb_hash = _hash_monos(cb_arr, bases)
            AB_hash = cb_hash[:, None] + cb_hash[None, :]
            picks = _hash_lookup(AB_hash, sorted_h, sort_o).ravel()
            used.update(int(p) for p in picks if p >= 0)

        # Localizing picks
        for i_var in range(d):
            c_idx2 = 0  # simplified
            clique = cliques[c_idx2]
            cb_arr = _build_clique_basis(clique, order - 1, d)
            cb_hash = _hash_monos(cb_arr, bases)
            AB_ei_hash = cb_hash[:, None] + cb_hash[None, :] + bases[i_var]
            picks = _hash_lookup(AB_ei_hash, sorted_h, sort_o).ravel()
            used.update(int(p) for p in picks if p >= 0)
        return used
    used, t10 = profile_phase("collect_used_indices", do_collect_used)
    print(f"    used={len(used):,}")

    # Phase 11: Batch violation check simulation
    print("\nPhase 11: Batch violation check (einsum cost)")
    n_check = 100  # typical number of windows to check
    def do_violation_sim():
        # Simulate y_abij creation
        y_fake = np.random.rand(n_y).astype(np.float64)
        safe_idx = np.clip(ab_eiej_idx, 0, n_y - 1)
        y_abij = y_fake[safe_idx]
        y_abij[ab_eiej_idx < 0] = 0.0

        # Simulate einsum for n_check windows
        M_stack = np.random.rand(n_check, d, d)
        L_q = np.einsum('abij,wij->wab', y_abij, M_stack)
        return L_q
    _, t11 = profile_phase(f"violation check ({n_check} windows)", do_violation_sim)

    # Phase 12: Window matrix memory
    print("\nPhase 12: Window matrix storage")
    win_mem = sum(M.nbytes for M in M_mats)
    print(f"    {len(M_mats)} window matrices: {sizeof_fmt(win_mem)}")

    # Summary
    print(f"\n{'='*70}")
    print(f"TIMING SUMMARY (d={d}, L{order})")
    print(f"{'='*70}")
    times = [
        ("Monomial enumeration", t1),
        ("collect_moments", t2),
        ("Hash table build", t3),
        ("Window matrices", t4),
        ("AB_hash lookup (moment matrix)", t5),
        ("ab_eiej_idx (4D localizing)", t6),
        ("Per-bin loc_picks", t7),
        ("Consistency lookups", t8),
        ("Clique bases", t9),
        ("collect_used_indices", t10),
        (f"Violation check ({n_check} wins)", t11),
    ]
    total = sum(t for _, t in times)
    for name, t in sorted(times, key=lambda x: -x[1]):
        pct = 100 * t / total if total > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  {t:7.2f}s ({pct:5.1f}%) {name}  {bar}")
    print(f"  {'─'*50}")
    print(f"  {total:7.2f}s  TOTAL (precompute only, no MOSEK)")

    # Memory summary
    print(f"\n{'='*70}")
    print(f"MEMORY SUMMARY")
    print(f"{'='*70}")
    print(f"  ab_eiej_idx:      {sizeof_fmt(ab_eiej_idx.nbytes)}")
    print(f"  Window matrices:  {sizeof_fmt(win_mem)}")
    print(f"  Hash table:       {sizeof_fmt(sorted_h.nbytes + sort_o.nbytes)}")
    print(f"  mono_list (est):  {sizeof_fmt(n_y * d * 8)}")
    print(f"  idx dict (est):   ~{sizeof_fmt(n_y * 200)}")
    total_mem = (ab_eiej_idx.nbytes + win_mem + sorted_h.nbytes + sort_o.nbytes
                 + n_y * d * 8 + n_y * 200)
    print(f"  ──────────────────────────────")
    print(f"  ESTIMATED TOTAL:  {sizeof_fmt(total_mem)}")
    print(f"  Budget:           256 GB RAM")
    print(f"  Headroom:         {sizeof_fmt(256 * 1024**3 - total_mem)}")


if __name__ == '__main__':
    main()
