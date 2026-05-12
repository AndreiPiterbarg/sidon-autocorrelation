"""Quick expansion sweep: test L0 + cascade levels with few parents each."""
import sys, os, time, math
import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import run_level0, process_parent_fused

C_TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1.35
C_UPPER = 1.5029
N_PARENTS = 3
MAX_LEVELS = 5
MAX_CHILDREN = 500_000_000  # skip parents with more children than this


def children_count(parent, m, c_target, d_child, n_half_child):
    """Count children for a single parent without generating them."""
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(4 * d_child * thresh)))
    x_cap_cs = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
    x_cap = min(x_cap, x_cap_cs)
    x_cap = max(x_cap, 0)
    total = 1
    for b in parent:
        lo = max(0, 2 * int(b) - x_cap)
        hi = min(2 * int(b), x_cap)
        r = max(hi - lo + 1, 0)
        total *= r
        if total > MAX_CHILDREN * 10:
            return total  # early out
    return total


def sweep_config(n_half, m):
    corr = correction(m)
    eff = C_TARGET + corr
    if eff >= C_UPPER:
        print(f"  m={m:>3}  VACUOUS (eff={eff:.4f})")
        return

    t0 = time.time()
    l0 = run_level0(n_half, m, C_TARGET, verbose=False)
    l0_time = time.time() - t0
    survivors = l0['survivors']
    n0 = len(survivors)
    print(f"  m={m:>3}  corr={corr:.4f} eff={eff:.4f}  L0: {n0:,} surv ({l0_time:.1f}s)", flush=True)

    if n0 == 0:
        print(f"         PROVEN at L0!", flush=True)
        return

    current = survivors
    projected = float(n0)
    d_parent = 2 * n_half
    n_half_p = n_half

    for lvl in range(1, MAX_LEVELS + 1):
        d_child = 2 * d_parent
        n_half_c = 2 * n_half_p
        n_avail = len(current)
        if n_avail == 0:
            print(f"         L{lvl}: 0 available -> PROVEN!", flush=True)
            return

        n_sample = min(N_PARENTS, n_avail)
        idx = np.random.choice(n_avail, size=n_sample, replace=False)
        sampled = current[idx]

        # Pre-check children counts, skip huge ones
        counts = [children_count(p, m, C_TARGET, d_child, n_half_c) for p in sampled]
        feasible = [(p, c) for p, c in zip(sampled, counts) if c <= MAX_CHILDREN]
        skipped = len(sampled) - len(feasible)
        if skipped:
            print(f"         L{lvl}: skipped {skipped}/{len(sampled)} parents (>{MAX_CHILDREN:,} children)", flush=True)

        if not feasible:
            # All sampled parents too large — sample more, pick smallest
            all_counts = np.array([children_count(p, m, C_TARGET, d_child, n_half_c) for p in current[:min(200, n_avail)]])
            smallest_idx = np.argsort(all_counts)[:N_PARENTS]
            feasible = [(current[i], all_counts[i]) for i in smallest_idx if all_counts[i] <= MAX_CHILDREN]
            if not feasible:
                med = np.median(all_counts)
                print(f"         L{lvl}: ALL parents too large (median {med:.2e} children), stopping", flush=True)
                return

        total_surv = 0
        all_surv = []

        for i, (p, nc_est) in enumerate(feasible):
            t1 = time.time()
            s, nc = process_parent_fused(p, m, C_TARGET, n_half_c)
            dt = time.time() - t1
            total_surv += len(s)
            if len(s) > 0:
                all_surv.append(s)
            print(f"         L{lvl} p{i}: {len(s):,}/{nc:,} surv/children ({dt:.1f}s)", flush=True)

        n_done = len(feasible)
        expansion = total_surv / n_done
        projected = projected * expansion

        if projected < 1:
            print(f"         L{lvl}: expansion={expansion:.1f}x, projected=0 -> PROVEN!", flush=True)
            return

        print(f"         L{lvl}: expansion={expansion:,.0f}x, projected={projected:.2e}", flush=True)

        if expansion > 1e6:
            print(f"         Stopping: expansion too large for deeper levels", flush=True)
            return

        if all_surv:
            current = np.vstack(all_surv)
        else:
            current = np.empty((0, d_child), dtype=np.int32)
        d_parent = d_child
        n_half_p = n_half_c

    print(f"         NOT PROVEN after {MAX_LEVELS} levels, projected={projected:.2e}", flush=True)


if __name__ == '__main__':
    n_half = 2
    m_values = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [80, 70, 60, 50, 45, 40, 35, 30]
    print(f"Quick sweep: c_target={C_TARGET}, n_half={n_half}, {N_PARENTS} parents/level")
    print(f"Max children/parent: {MAX_CHILDREN:,}")
    print(f"Correction = 2/m + 1/m^2 (C&S Lemma 3, fine grid S=4nm)\n")

    for m in m_values:
        sweep_config(n_half, m)
        print(flush=True)
