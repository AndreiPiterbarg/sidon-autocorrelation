"""Test whether max_W TV_W is monotonically non-decreasing under refinement.

CRITICAL QUESTION for cascade soundness:
If parent at d bins has max TV = V, do ALL children at 2d bins
(ν with ν_{2i} + ν_{2i+1} = μ_i) have max TV >= V?

If YES: the cascade can prune parents without correction (sound).
If NO: we need a correction term to account for the TV decrease.

This test exhaustively checks refinements at small d, and randomly
samples at larger d.
"""
import numpy as np
import sys
import time

sys.path.insert(0, '.')
from estimate_min_max_tv import _max_tv, _autoconv


def all_splits(parent_mass, n_splits):
    """Enumerate all ways to split parent_mass into n_splits integer parts.

    Each split has child[2i] + child[2i+1] = 2*parent[i] (preserving
    total in integer space for S=round(1/delta)).
    """
    S_parent = int(round(sum(parent_mass)))
    d_parent = len(parent_mass)

    # For each parent bin i, child[2i] ranges from 0 to parent[i]*granularity
    # With integer masses: child[2i] ranges from 0 to parent_int[i]
    # child[2i+1] = parent_int[i] - child[2i]
    parent_int = [int(round(p * n_splits)) for p in parent_mass]
    total_int = sum(parent_int)

    ranges = [list(range(0, pi + 1)) for pi in parent_int]

    # Cartesian product
    from itertools import product
    for combo in product(*ranges):
        child = np.zeros(2 * d_parent, dtype=np.float64)
        for i in range(d_parent):
            child[2 * i] = combo[i] / n_splits
            child[2 * i + 1] = parent_int[i] / n_splits - combo[i] / n_splits
        yield child


def test_exhaustive_small():
    """Exhaustively test all parents and all children at small d and S."""
    print("=" * 60)
    print("EXHAUSTIVE TEST: refinement monotonicity at small d")
    print("=" * 60)

    violations = 0
    total_tested = 0
    worst_decrease = 0.0
    worst_parent = None
    worst_child = None

    for S in [4, 6, 8, 10]:
        for d_parent in [2, 3, 4]:
            # Enumerate all parent compositions of S into d_parent parts
            from itertools import combinations_with_replacement
            parent_compositions = []

            def gen_compositions(n, k, prefix=[]):
                if k == 1:
                    parent_compositions.append(prefix + [n])
                    return
                for i in range(n + 1):
                    gen_compositions(n - i, k - 1, prefix + [i])

            gen_compositions(S, d_parent)

            n_parents = len(parent_compositions)
            d_child = 2 * d_parent

            print(f"\n  S={S}, d_parent={d_parent} ({n_parents} parents):")

            level_violations = 0
            level_worst = 0.0
            level_total = 0

            for parent_int in parent_compositions:
                parent_mu = np.array([c / S for c in parent_int], dtype=np.float64)
                parent_tv, _, _ = _max_tv(parent_mu, d_parent)

                # Enumerate ALL children
                for child_mu in all_splits(parent_mu, S):
                    child_tv, _, _ = _max_tv(child_mu, d_child)
                    level_total += 1

                    if child_tv < parent_tv - 1e-10:
                        decrease = parent_tv - child_tv
                        level_violations += 1
                        if decrease > level_worst:
                            level_worst = decrease
                        if decrease > worst_decrease:
                            worst_decrease = decrease
                            worst_parent = (d_parent, parent_mu.copy(), parent_tv)
                            worst_child = (d_child, child_mu.copy(), child_tv)

            total_tested += level_total
            violations += level_violations

            if level_violations > 0:
                print(f"    VIOLATIONS: {level_violations}/{level_total} "
                      f"(worst decrease: {level_worst:.6f})")
            else:
                print(f"    OK: {level_total} children tested, all >= parent TV")

    print(f"\n  TOTAL: {violations} violations out of {total_tested} tests")
    if violations > 0:
        d_p, mu_p, tv_p = worst_parent
        d_c, mu_c, tv_c = worst_child
        print(f"\n  WORST VIOLATION:")
        print(f"    Parent (d={d_p}): mu = {mu_p}, max TV = {tv_p:.6f}")
        print(f"    Child  (d={d_c}): mu = {mu_c}, max TV = {tv_c:.6f}")
        print(f"    Decrease: {tv_p - tv_c:.6f} ({(tv_p - tv_c)/tv_p*100:.2f}%)")
    return violations == 0


def test_random_large(d_parent=8, n_parents=1000, n_children_per=500, seed=42):
    """Randomly test refinement monotonicity at larger d."""
    print(f"\n{'=' * 60}")
    print(f"RANDOM TEST: d_parent={d_parent}, {n_parents} parents, "
          f"{n_children_per} children each")
    print("=" * 60)

    rng = np.random.RandomState(seed)
    violations = 0
    total = 0
    worst_decrease = 0.0
    worst_ratio = 1.0
    worst_parent = None
    worst_child = None

    for p_idx in range(n_parents):
        # Random parent on simplex
        parent_mu = rng.dirichlet(np.ones(d_parent))
        parent_tv, _, _ = _max_tv(parent_mu, d_parent)
        d_child = 2 * d_parent

        for _ in range(n_children_per):
            # Random refinement: split each parent bin randomly
            child_mu = np.zeros(d_child, dtype=np.float64)
            for i in range(d_parent):
                # Random split of parent_mu[i] into two non-negative parts
                split_frac = rng.uniform(0, 1)
                child_mu[2 * i] = parent_mu[i] * split_frac
                child_mu[2 * i + 1] = parent_mu[i] * (1 - split_frac)

            child_tv, _, _ = _max_tv(child_mu, d_child)
            total += 1

            if child_tv < parent_tv - 1e-10:
                decrease = parent_tv - child_tv
                violations += 1
                if decrease > worst_decrease:
                    worst_decrease = decrease
                    worst_ratio = child_tv / parent_tv
                    worst_parent = (d_parent, parent_mu.copy(), parent_tv)
                    worst_child = (d_child, child_mu.copy(), child_tv)

    print(f"  Results: {violations} violations out of {total} tests")
    if violations > 0:
        d_p, mu_p, tv_p = worst_parent
        d_c, mu_c, tv_c = worst_child
        print(f"  WORST VIOLATION:")
        print(f"    Parent (d={d_p}): max TV = {tv_p:.6f}")
        print(f"    Child  (d={d_c}): max TV = {tv_c:.6f}")
        print(f"    Decrease: {tv_p - tv_c:.6f} ({(tv_p-tv_c)/tv_p*100:.2f}%)")
        print(f"    Ratio: child/parent = {worst_ratio:.4f}")
    else:
        print(f"  PASS: all children have max TV >= parent max TV")

    return violations == 0


def test_adversarial(d_parent=4, S=20, seed=42, n_children=10000):
    """Targeted search for violations near the adversarial minimizer.

    These are the mass vectors where TV is LOWEST — most likely place
    for a violation.
    """
    print(f"\n{'=' * 60}")
    print(f"ADVERSARIAL TEST: near the min-max-TV optimum at d={d_parent}")
    print("=" * 60)

    # Use optimizer to find the minimizer
    from estimate_min_max_tv import optimize_for_d
    parent_tv, parent_mu = optimize_for_d(d_parent, n_restarts=20,
                                           n_iters=20000, verbose=False)
    print(f"  Parent minimizer: max TV = {parent_tv:.6f}")
    print(f"  mu = {parent_mu}")

    # Try many refinements of this specific parent
    rng = np.random.RandomState(seed)
    d_child = 2 * d_parent
    n_children = 10000
    violations = 0
    worst_decrease = 0.0

    child_tvs = []
    for _ in range(n_children):
        child_mu = np.zeros(d_child, dtype=np.float64)
        for i in range(d_parent):
            frac = rng.uniform(0, 1)
            child_mu[2 * i] = parent_mu[i] * frac
            child_mu[2 * i + 1] = parent_mu[i] * (1 - frac)

        child_tv, _, _ = _max_tv(child_mu, d_child)
        child_tvs.append(child_tv)

        if child_tv < parent_tv - 1e-10:
            violations += 1
            decrease = parent_tv - child_tv
            if decrease > worst_decrease:
                worst_decrease = decrease

    child_tvs = np.array(child_tvs)
    print(f"  Children ({n_children} tested):")
    print(f"    min  child TV: {child_tvs.min():.6f}")
    print(f"    mean child TV: {child_tvs.mean():.6f}")
    print(f"    max  child TV: {child_tvs.max():.6f}")
    print(f"    Violations: {violations}")
    if violations > 0:
        print(f"    Worst decrease: {worst_decrease:.6f}")
    else:
        print(f"    All children >= parent ({parent_tv:.6f})")

    # Also try the OPTIMAL child (minimizer at child dimension)
    child_opt_tv, child_opt_mu = optimize_for_d(d_child, n_restarts=20,
                                                  n_iters=20000, verbose=False)
    print(f"\n  Optimal child (unrestricted): max TV = {child_opt_tv:.6f}")
    print(f"  vs parent: {parent_tv:.6f}")
    if child_opt_tv < parent_tv:
        print(f"  NOTE: optimal child TV < parent TV!")
        print(f"  But this child may NOT be a refinement of this parent.")
        # Check if optimal child is a refinement of ANY parent
        # by computing parent masses
        parent_from_child = np.zeros(d_parent, dtype=np.float64)
        for i in range(d_parent):
            parent_from_child[i] = child_opt_mu[2*i] + child_opt_mu[2*i+1]
        parent_from_child_tv, _, _ = _max_tv(parent_from_child, d_parent)
        print(f"  Implied parent max TV: {parent_from_child_tv:.6f}")
        if parent_from_child_tv <= child_opt_tv + 1e-10:
            print(f"  Refinement monotonicity HOLDS for this case "
                  f"(parent TV {parent_from_child_tv:.6f} <= child TV {child_opt_tv:.6f})")
        else:
            print(f"  VIOLATION: parent TV {parent_from_child_tv:.6f} > "
                  f"child TV {child_opt_tv:.6f}")

    return violations == 0


def test_reverse_check():
    """For each child minimizer, compute its implied parent and check.

    If min_max_TV(d) > min_max_TV(2d), this means the global minimum
    DECREASES with d, but we need to check if it's achieved by a
    valid refinement.
    """
    print(f"\n{'=' * 60}")
    print("REVERSE CHECK: does the global min decrease with d?")
    print("=" * 60)

    from estimate_min_max_tv import optimize_for_d

    prev_tv = None
    for d in [2, 4, 8, 16, 32]:
        tv, mu = optimize_for_d(d, n_restarts=30, n_iters=20000, verbose=False)

        if prev_tv is not None:
            direction = "UP" if tv > prev_tv else "DOWN"
            print(f"  d={d:3d}: min max TV = {tv:.6f}  ({direction} from {prev_tv:.6f})")
        else:
            print(f"  d={d:3d}: min max TV = {tv:.6f}")

        # Check implied parent
        if d >= 4:
            d_parent = d // 2
            parent_mu = np.zeros(d_parent, dtype=np.float64)
            for i in range(d_parent):
                parent_mu[i] = mu[2*i] + mu[2*i+1]
            parent_tv, _, _ = _max_tv(parent_mu, d_parent)
            print(f"         implied parent (d={d_parent}): max TV = {parent_tv:.6f}"
                  f"  {'OK' if parent_tv <= tv + 1e-6 else 'VIOLATION!'}")

        prev_tv = tv


if __name__ == "__main__":
    # Warmup
    _max_tv(np.array([0.5, 0.5]), 2)

    t0 = time.time()

    # Test 1: Exhaustive at small d
    exhaustive_ok = test_exhaustive_small()

    # Test 2: Random at larger d
    random_ok_8 = test_random_large(d_parent=8, n_parents=500, n_children_per=200)
    random_ok_16 = test_random_large(d_parent=16, n_parents=200, n_children_per=100)

    # Test 3: Adversarial near the optimum
    for dp in [4, 8, 16]:
        test_adversarial(d_parent=dp, n_children=5000)

    # Test 4: Reverse check
    test_reverse_check()

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"SUMMARY (total time: {elapsed:.1f}s)")
    print("=" * 60)
    if exhaustive_ok:
        print("  Exhaustive test (small d): PASS — no violations found")
    else:
        print("  Exhaustive test (small d): FAIL — violations exist!")
    print()
    print("  If all tests pass: cascade is sound without correction.")
    print("  If violations found: need refinement correction term.")
