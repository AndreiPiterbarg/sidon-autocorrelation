"""Quick cascade test with numpy-vectorized pruning.

Tests intermediate grids (varying m) to find optimal m for c_target=1.28.
Uses batch convolution and vectorized window scan for ~100x speedup over
pure Python loops.
"""
import numpy as np
from itertools import product
import time
import sys


def batch_prune(children, d, m, c_target):
    """Prune a batch of compositions. Returns boolean mask of survivors."""
    n = d / 2.0
    corr = 2.0 / m + 1.0 / (m * m)
    thresh_base = c_target + corr
    N = len(children)
    if N == 0:
        return np.array([], dtype=bool)

    children = np.array(children, dtype=np.int64)

    # Compute autoconvolution for all children at once
    n_conv = 2 * d - 1
    conv = np.zeros((N, n_conv), dtype=np.int64)
    for k in range(n_conv):
        for i in range(max(0, k - d + 1), min(k + 1, d)):
            j = k - i
            conv[:, k] += children[:, i] * children[:, j]

    survived = np.ones(N, dtype=bool)

    for ell in range(2, 2 * d + 1):
        if not np.any(survived):
            break
        ncv = ell - 1
        norm = 4.0 * n * ell * m * m
        thresh_int = int(np.floor(thresh_base * norm))

        # Sliding window sums for all survivors
        active = np.where(survived)[0]
        if len(active) == 0:
            break
        conv_active = conv[active]

        for s in range(n_conv - ncv + 1):
            ws = np.sum(conv_active[:, s:s + ncv], axis=1)
            pruned_mask = ws > thresh_int
            if np.any(pruned_mask):
                survived[active[pruned_mask]] = False
                active = np.where(survived)[0]
                if len(active) == 0:
                    break
                conv_active = conv[active]

    return survived


def run_cascade(m, c_target, d0=2, max_levels=8, verbose=True):
    """Run cascade with given parameters. Returns list of (level, d, survivors)."""
    corr = 2.0 / m + 1.0 / (m * m)
    thresh = c_target + corr
    if verbose:
        print(f"CASCADE: m={m}, c_target={c_target}, corr={corr:.4f}, thresh={thresh:.4f}")

    # L0
    S0 = int(4 * (d0 / 2) * m)
    d = d0
    all_comps = []
    if d == 2:
        for c0 in range(S0 + 1):
            all_comps.append([c0, S0 - c0])
    elif d == 4:
        for c0 in range(S0 + 1):
            for c1 in range(S0 + 1 - c0):
                for c2 in range(S0 + 1 - c0 - c1):
                    all_comps.append([c0, c1, c2, S0 - c0 - c1 - c2])

    survived = batch_prune(all_comps, d, m, c_target)
    survivors = [all_comps[i] for i in range(len(all_comps)) if survived[i]]
    if verbose:
        print(f"  L0: d={d}, tested={len(all_comps)}, surv={len(survivors)}")

    results = [(0, d, len(survivors))]
    if len(survivors) == 0:
        if verbose:
            print("  PROVEN at L0!")
        return results

    # Cascade levels
    parents = np.array(survivors, dtype=np.int64)
    for level in range(1, max_levels + 1):
        d_parent = d
        d_child = 2 * d
        d = d_child

        x_cap = int(np.floor(m * np.sqrt(4 * d_child * (thresh + 1e-9))))

        all_children = []
        t0 = time.time()

        for pi, p in enumerate(parents):
            ranges_list = []
            for i in range(d_parent):
                ci = int(p[i])
                lo = max(0, 2 * ci - x_cap)
                hi = min(2 * ci, x_cap)
                ranges_list.append(range(lo, hi + 1))

            for combo in product(*ranges_list):
                child = np.zeros(d_child, dtype=np.int64)
                for i in range(d_parent):
                    child[2 * i] = combo[i]
                    child[2 * i + 1] = 2 * int(p[i]) - combo[i]
                if np.any(child < 0):
                    continue
                all_children.append(child)

                # Batch prune every 500K children to avoid OOM
                if len(all_children) >= 500000:
                    surv_mask = batch_prune(all_children, d_child, m, c_target)
                    new_surv = [all_children[j] for j in range(len(all_children)) if surv_mask[j]]
                    all_children = []
                    if not hasattr(run_cascade, '_accum'):
                        run_cascade._accum = new_surv
                    else:
                        run_cascade._accum.extend(new_surv)

            if verbose and (pi + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"    L{level} progress: {pi+1}/{len(parents)} parents, {elapsed:.0f}s", flush=True)

        # Final batch
        if hasattr(run_cascade, '_accum'):
            accumulated = run_cascade._accum
            del run_cascade._accum
        else:
            accumulated = []

        if all_children:
            surv_mask = batch_prune(all_children, d_child, m, c_target)
            new_surv = [all_children[j] for j in range(len(all_children)) if surv_mask[j]]
            accumulated.extend(new_surv)

        elapsed = time.time() - t0
        n_tested = sum(1 for _ in [])  # placeholder
        if verbose:
            print(f"  L{level}: d={d_child}, parents={len(parents)}, surv={len(accumulated)}, time={elapsed:.1f}s")

        results.append((level, d_child, len(accumulated)))

        if len(accumulated) == 0:
            if verbose:
                print(f"  PROVEN at L{level}!")
            return results

        parents = np.array(accumulated, dtype=np.int64)

    if verbose:
        print(f"  Cascade did not converge in {max_levels} levels")
    return results


if __name__ == "__main__":
    c_target = float(sys.argv[1]) if len(sys.argv) > 1 else 1.28
    m_values = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [5, 7, 10]

    for m in m_values:
        print("=" * 60)
        run_cascade(m, c_target, d0=2, max_levels=7)
        print()
