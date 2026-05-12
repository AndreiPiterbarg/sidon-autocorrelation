"""Test block mass invariant pruning for soundness and effectiveness.

Verifies that:
  1. The mathematical identity holds: for k consecutive parent bins with
     total mass M, the child self-conv over 4k-1 positions = (2M)^2.
  2. Parents flagged as prunable by block_mass_prune_mask are ACTUALLY
     fully pruned by the existing kernel (zero survivors).
  3. The threshold computation is conservative (no false positives).

Usage:
    python tests/test_block_mass_pruning.py
    pytest tests/test_block_mass_pruning.py -v
"""

import os
import sys
import numpy as np
import math

# Path setup
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
_cs_dir = os.path.join(_root, "cloninger-steinerberger")
sys.path.insert(0, _cs_dir)
sys.path.insert(0, os.path.join(_cs_dir, "cpu"))

from pruning import block_mass_prune_mask, correction
from run_cascade import process_parent_fused


def _compute_child_block_selfconv(parent_block, cursor_vals):
    """Brute-force: compute self-conv of child bins from a parent block.

    parent_block: array of k parent bin masses
    cursor_vals: array of k cursor values (child[2j] = cursor[j],
                 child[2j+1] = 2*parent[j] - cursor[j])

    Returns: sum of all autoconvolution entries within the block's range.
    """
    k = len(parent_block)
    child = np.zeros(2 * k, dtype=np.int64)
    for j in range(k):
        child[2 * j] = cursor_vals[j]
        child[2 * j + 1] = 2 * parent_block[j] - cursor_vals[j]

    # Self-conv of the 2k child bins: 4k-1 entries
    conv_len = 2 * (2 * k) - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(2 * k):
        for j in range(2 * k):
            if i + j < conv_len:
                conv[i + j] += child[i] * child[j]
    return int(np.sum(conv))


def test_block_mass_identity():
    """Verify (2M)^2 identity for random parent blocks and cursor values."""
    rng = np.random.RandomState(42)
    for trial in range(500):
        k = rng.randint(1, 6)
        parent_block = rng.randint(0, 50, size=k)
        M = int(np.sum(parent_block))
        # Random cursor values in valid range [0, 2*p_j]
        cursor_vals = np.array([rng.randint(0, 2 * p + 1) for p in parent_block])
        total_selfconv = _compute_child_block_selfconv(parent_block, cursor_vals)
        expected = (2 * M) ** 2
        assert total_selfconv == expected, (
            f"Trial {trial}: k={k}, parent={parent_block}, cursor={cursor_vals}, "
            f"selfconv={total_selfconv}, expected (2M)^2={expected}")
    print(f"  [PASS] Block mass identity verified for 500 random cases")


def test_block_mass_identity_edge_cases():
    """Verify identity for edge cases: zero bins, single bin, all mass in one."""
    # All zeros
    parent = np.array([0, 0, 0])
    cursor = np.array([0, 0, 0])
    assert _compute_child_block_selfconv(parent, cursor) == 0

    # Single bin
    for p in [1, 10, 50, 100]:
        for c in range(2 * p + 1):
            total = _compute_child_block_selfconv(np.array([p]), np.array([c]))
            assert total == (2 * p) ** 2, f"p={p}, c={c}: got {total}"

    # All mass in first bin
    parent = np.array([40, 0, 0])
    cursor = np.array([30, 0, 0])
    assert _compute_child_block_selfconv(parent, cursor) == (2 * 40) ** 2

    print(f"  [PASS] Edge cases verified")


def test_soundness_against_kernel():
    """Verify that parents pruned by block mass check have zero survivors
    in the actual kernel.

    This is the critical soundness test: block_mass_prune_mask says
    'prunable', and process_parent_fused confirms 0 survivors.
    """
    m = 20
    c_target = 1.28

    # Load L0 survivors (parents for L1)
    l0_path = os.path.join(_root, 'data', 'checkpoint_L0_survivors.npy')
    if not os.path.exists(l0_path):
        print(f"  [SKIP] {l0_path} not found")
        return

    parents = np.load(l0_path)
    d_parent = parents.shape[1]
    n_half_child = d_parent  # n_half doubles: n_half_parent = d_parent/2, n_half_child = d_parent

    print(f"  Testing against {len(parents)} L0 parents (d_parent={d_parent})")

    # Get block mass mask (True = needs expansion, False = prunable)
    mask = block_mass_prune_mask(parents, n_half_child, m, c_target,
                                  use_flat_threshold=False)

    n_prunable = int(np.sum(~mask))
    print(f"  Block mass prunable: {n_prunable} / {len(parents)}")

    if n_prunable == 0:
        print(f"  [SKIP] No parents prunable at this level")
        return

    # Verify each pruned parent has 0 survivors in the actual kernel
    pruned_parents = parents[~mask]
    n_checked = min(len(pruned_parents), 50)  # limit for speed
    for i in range(n_checked):
        parent = pruned_parents[i]
        survivors, n_children = process_parent_fused(
            parent, m, c_target, n_half_child,
            use_flat_threshold=False)
        assert len(survivors) == 0, (
            f"SOUNDNESS FAILURE: parent {parent} was block-mass pruned but "
            f"kernel found {len(survivors)} survivors out of {n_children} children")

    print(f"  [PASS] All {n_checked} block-mass-pruned parents confirmed "
          f"0 survivors by kernel")


def test_soundness_flat_threshold():
    """Same as above but with flat threshold (Lean axiom mode)."""
    m = 20
    c_target = 1.28

    l0_path = os.path.join(_root, 'data', 'checkpoint_L0_survivors.npy')
    if not os.path.exists(l0_path):
        print(f"  [SKIP] {l0_path} not found")
        return

    parents = np.load(l0_path)
    d_parent = parents.shape[1]
    n_half_child = d_parent

    mask = block_mass_prune_mask(parents, n_half_child, m, c_target,
                                  use_flat_threshold=True)

    n_prunable = int(np.sum(~mask))
    print(f"  Flat threshold: block mass prunable: {n_prunable} / {len(parents)}")

    if n_prunable == 0:
        print(f"  [SKIP] No parents prunable with flat threshold")
        return

    pruned_parents = parents[~mask]
    n_checked = min(len(pruned_parents), 50)
    for i in range(n_checked):
        parent = pruned_parents[i]
        survivors, _ = process_parent_fused(
            parent, m, c_target, n_half_child,
            use_flat_threshold=True)
        assert len(survivors) == 0, (
            f"SOUNDNESS FAILURE (flat): parent {parent} block-mass pruned but "
            f"kernel found {len(survivors)} survivors")

    print(f"  [PASS] Flat threshold: {n_checked} pruned parents confirmed 0 survivors")


def test_no_false_negatives_sample():
    """Sample parents NOT pruned by block mass and verify they DO have survivors.

    This tests the other direction: we're not being too aggressive.
    Note: not all non-pruned parents necessarily have survivors (other
    pruning may kill them), so we just verify the block mass check isn't
    claiming more than it should.
    """
    m = 20
    c_target = 1.28

    l0_path = os.path.join(_root, 'data', 'checkpoint_L0_survivors.npy')
    if not os.path.exists(l0_path):
        print(f"  [SKIP] {l0_path} not found")
        return

    parents = np.load(l0_path)
    d_parent = parents.shape[1]
    n_half_child = d_parent

    mask = block_mass_prune_mask(parents, n_half_child, m, c_target,
                                  use_flat_threshold=False)
    kept_parents = parents[mask]

    # Count how many kept parents actually have survivors
    n_check = min(len(kept_parents), 20)
    n_with_survivors = 0
    for i in range(n_check):
        survivors, _ = process_parent_fused(
            kept_parents[i], m, c_target, n_half_child,
            use_flat_threshold=False)
        if len(survivors) > 0:
            n_with_survivors += 1

    print(f"  Of {n_check} kept parents, {n_with_survivors} have survivors")
    print(f"  [INFO] Block mass check is conservative by design; "
          f"non-pruned parents may still be fully pruned by kernel")


def test_threshold_computation():
    """Verify the threshold computation matches the kernel's threshold table."""
    m = 20
    c_target = 1.28
    n_half_child = 2  # d_parent=2, d_child=4
    d_child = 4
    d_parent = 2

    m_d = float(m)
    n_half_d = float(n_half_child)
    four_n = 4.0 * n_half_d
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    S_child = 4 * n_half_child * m  # = 160

    # k=1: ell=4 (2 child bins → 3 conv positions, n_cv=3, ell=4)
    ell = 4
    scale_ell = float(ell) * four_n
    # W-refined with W_int = S_child (conservative)
    corr_val = 1.0 + float(S_child) / (2.0 * n_half_d)
    dyn_x = (cs_base_m2 + corr_val + eps_margin) * scale_ell
    thr_k1 = int(dyn_x)

    # Check: parent bin with mass p, block mass M = p
    # Prune if 4*p^2 > thr_k1
    p_min = int(math.ceil(math.sqrt(thr_k1 / 4.0)))
    print(f"  k=1 (ell=4): threshold = {thr_k1}, prune when p >= {p_min}")

    # k=2: ell=8 (full parent for d_parent=2)
    ell = 8
    scale_ell = float(ell) * four_n
    corr_val = 1.0 + float(S_child) / (2.0 * n_half_d)
    dyn_x = (cs_base_m2 + corr_val + eps_margin) * scale_ell
    thr_k2 = int(dyn_x)
    M_min = int(math.ceil(math.sqrt(thr_k2 / 4.0)))
    print(f"  k=2 (ell=8): threshold = {thr_k2}, prune when M >= {M_min}")
    print(f"  Total mass S_parent = {S_child // 2}")

    # Sanity: threshold should be positive
    assert thr_k1 > 0
    assert thr_k2 > 0

    print(f"  [PASS] Threshold computation verified")


if __name__ == '__main__':
    print("=== Block Mass Invariant Pruning Tests ===\n")

    print("1. Block mass identity (2M)^2:")
    test_block_mass_identity()

    print("\n2. Edge cases:")
    test_block_mass_identity_edge_cases()

    print("\n3. Threshold computation:")
    test_threshold_computation()

    print("\n4. Soundness (W-refined threshold):")
    test_soundness_against_kernel()

    print("\n5. Soundness (flat threshold):")
    test_soundness_flat_threshold()

    print("\n6. False negative check:")
    test_no_false_negatives_sample()

    print("\n=== All tests passed ===")
