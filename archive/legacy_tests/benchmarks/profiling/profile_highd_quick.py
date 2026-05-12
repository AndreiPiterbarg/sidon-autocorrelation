#!/usr/bin/env python
"""Quick profiling at d=32 and d=64 to extrapolate d=128 bottlenecks."""
import numpy as np
import time
import sys
import os

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


def profile_d(d, order=2, bandwidth=16):
    print(f"\n{'='*60}")
    print(f"  d={d}, order={order}, bw={bandwidth}")
    print(f"{'='*60}")

    times = {}

    # 1. Enumerate
    t0 = time.time()
    basis = enum_monomials(d, order)
    loc_basis = enum_monomials(d, order - 1)
    consist_mono = enum_monomials(d, 2 * order - 1)
    times['enum'] = time.time() - t0
    print(f"  enum: {times['enum']:.3f}s  basis={len(basis)} loc={len(loc_basis)} consist={len(consist_mono)}")

    # 2. collect_moments
    t0 = time.time()
    mono_list, idx = collect_moments(d, order, basis, loc_basis, consist_mono)
    times['collect'] = time.time() - t0
    n_y = len(mono_list)
    print(f"  collect_moments: {times['collect']:.3f}s  n_y={n_y:,}")

    # 3. Hash table
    bases = _make_hash_bases(d, 2 * order)
    t0 = time.time()
    sorted_h, sort_o = _build_hash_table(mono_list, bases)
    times['hash_table'] = time.time() - t0
    print(f"  hash_table: {times['hash_table']:.3f}s")

    # 4. Windows
    t0 = time.time()
    windows, M_mats = build_window_matrices(d)
    times['windows'] = time.time() - t0
    print(f"  windows: {times['windows']:.3f}s  n_win={len(windows)}")

    # 5. AB lookup (moment matrix)
    B_arr = np.array(basis, dtype=np.int64)
    B_hash = _hash_monos(B_arr, bases)
    t0 = time.time()
    AB_hash = B_hash[:, None] + B_hash[None, :]
    _hash_lookup(AB_hash, sorted_h, sort_o)
    times['AB_lookup'] = time.time() - t0
    print(f"  AB_lookup: {times['AB_lookup']:.3f}s  queries={len(basis)**2:,}")

    # 6. ab_eiej_idx
    n_loc = len(loc_basis)
    LB_arr = np.array(loc_basis, dtype=np.int64)
    AB_loc = LB_arr[:, None, :] + LB_arr[None, :, :]
    AB_loc_hash = _hash_monos(AB_loc, bases)

    t0 = time.time()
    EE_hash = bases[:, None] + bases[None, :]
    ABIJ_hash = AB_loc_hash[:, :, None, None] + EE_hash[None, None, :, :]
    abij_size = ABIJ_hash.nbytes
    ab_eiej_idx = _hash_lookup(ABIJ_hash, sorted_h, sort_o)
    times['abij'] = time.time() - t0
    print(f"  ab_eiej_idx: {times['abij']:.3f}s  shape={ab_eiej_idx.shape} "
          f"mem={sizeof_fmt(ab_eiej_idx.nbytes)}")

    # 7. loc_picks
    t0 = time.time()
    for i_var in range(d):
        h = AB_loc_hash + bases[i_var]
        _hash_lookup(h, sorted_h, sort_o)
    times['loc_picks'] = time.time() - t0
    print(f"  loc_picks: {times['loc_picks']:.3f}s  ({d} bins)")

    # 8. Consistency
    consist_arr = np.array(consist_mono, dtype=np.int64)
    t0 = time.time()
    consist_hash = _hash_monos(consist_arr, bases)
    consist_idx = _hash_lookup(consist_hash, sorted_h, sort_o)
    consist_ei_hash = consist_hash[:, None] + bases[None, :]
    consist_ei_idx = _hash_lookup(consist_ei_hash, sorted_h, sort_o)
    times['consist'] = time.time() - t0
    print(f"  consist: {times['consist']:.3f}s  rows={len(consist_mono)} "
          f"ei_shape={consist_ei_idx.shape}")

    # 9. Cliques + used indices
    cliques = _build_banded_cliques(d, bandwidth)
    t0 = time.time()
    used = set()
    for c_idx, clique in enumerate(cliques):
        cb_arr = _build_clique_basis(clique, order, d)
        cb_hash = _hash_monos(cb_arr, bases)
        AB_h = cb_hash[:, None] + cb_hash[None, :]
        picks = _hash_lookup(AB_h, sorted_h, sort_o).ravel()
        used.update(int(p) for p in picks if p >= 0)
    for i_var in range(d):
        c_idx2 = 0
        clique = cliques[c_idx2]
        cb_arr = _build_clique_basis(clique, order - 1, d)
        cb_hash = _hash_monos(cb_arr, bases)
        AB_ei_h = cb_hash[:, None] + cb_hash[None, :] + bases[i_var]
        picks = _hash_lookup(AB_ei_h, sorted_h, sort_o).ravel()
        used.update(int(p) for p in picks if p >= 0)
    times['used_idx'] = time.time() - t0
    print(f"  collect_used: {times['used_idx']:.3f}s  used={len(used):,}")

    # 10. einsum simulation
    n_check = min(100, len(windows))
    t0 = time.time()
    y_fake = np.random.rand(n_y)
    safe_idx = np.clip(ab_eiej_idx, 0, n_y - 1)
    y_abij = y_fake[safe_idx]
    M_stack = np.random.rand(n_check, d, d)
    L_q = np.einsum('abij,wij->wab', y_abij, M_stack)
    times['einsum'] = time.time() - t0
    print(f"  einsum ({n_check} wins): {times['einsum']:.3f}s")

    # 11. F_scipy construction
    E_arr = np.eye(d, dtype=np.int64)
    EE_deg2 = E_arr[:, None, :] + E_arr[None, :, :]
    EE_deg2_hash = _hash_monos(EE_deg2, bases)
    idx_ij = _hash_lookup(EE_deg2_hash, sorted_h, sort_o)

    t0 = time.time()
    f_r, f_c, f_v = [], [], []
    for w in range(len(windows)):
        Mw = M_mats[w]
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) == 0:
            continue
        mi = idx_ij[nz_i, nz_j]
        valid = mi >= 0
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        f_r.extend([w] * n_valid)
        f_c.extend(mi[valid].tolist())
        f_v.extend(Mw[nz_i[valid], nz_j[valid]].tolist())
    times['F_scipy'] = time.time() - t0
    print(f"  F_scipy: {times['F_scipy']:.3f}s")

    # Summary
    total = sum(times.values())
    print(f"\n  TOTAL: {total:.2f}s")
    print(f"  Memory: ab_eiej={sizeof_fmt(ab_eiej_idx.nbytes)}, "
          f"M_mats={sizeof_fmt(sum(M.nbytes for M in M_mats))}")

    return times, d


results = []
for d in [16, 32, 64]:
    bw = min(16, d // 2)
    t, dd = profile_d(d, order=2, bandwidth=bw)
    results.append((dd, t))

# Extrapolate to d=128
print(f"\n\n{'='*60}")
print("EXTRAPOLATION to d=128")
print(f"{'='*60}")
if len(results) >= 2:
    d1, t1 = results[-2]
    d2, t2 = results[-1]
    ratio = d2 / d1
    for key in t2:
        if t1[key] > 0.001:
            growth = t2[key] / t1[key]
            est_128 = t2[key] * (128/d2) ** (np.log(growth)/np.log(ratio))
            print(f"  {key:20s}: d={d1}->{t1[key]:.3f}s, d={d2}->{t2[key]:.3f}s, "
                  f"growth={growth:.1f}x, est_d128={est_128:.1f}s")
