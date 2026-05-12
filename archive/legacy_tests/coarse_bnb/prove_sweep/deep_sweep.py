"""Fast cascade probe: 1 parent per level, track expansion and project."""
import sys, os, time
import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction
from run_cascade import run_level0, process_parent_fused

C_TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1.35
MAX_LEVELS = 6
MAX_SECONDS_PER_PARENT = 30

def run(n_half, m):
    corr = correction(m)
    if C_TARGET + corr >= 1.5029:
        print(f"  m={m}: VACUOUS"); return

    l0 = run_level0(n_half, m, C_TARGET, verbose=False)
    surv = l0['survivors']
    projected = float(len(surv))
    print(f"  m={m}: L0={len(surv):,} surv", flush=True)

    d_parent = 2 * n_half
    nhp = n_half

    for lvl in range(1, MAX_LEVELS + 1):
        d_child = 2 * d_parent
        nhc = 2 * nhp
        n = len(surv)
        if n == 0:
            print(f"    L{lvl}: PROVEN (0 parents)"); return

        # pick 1 random parent
        p = surv[np.random.randint(n)]
        t0 = time.time()
        s, nc = process_parent_fused(p, m, C_TARGET, nhc)
        dt = time.time() - t0

        exp = len(s)
        projected *= exp
        status = "PROVEN!" if exp == 0 else f"proj={projected:.1e}"
        print(f"    L{lvl}: {len(s):,}/{nc:,} surv/child ({dt:.1f}s) exp={exp:,}x {status}", flush=True)

        if exp == 0: return
        if dt > MAX_SECONDS_PER_PARENT:
            print(f"    L{lvl}: parent took {dt:.0f}s > {MAX_SECONDS_PER_PARENT}s, stopping"); return

        surv = s
        d_parent = d_child
        nhp = nhc

if __name__ == '__main__':
    m_values = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [20,25,30]
    print(f"c_target={C_TARGET}, 1 parent/level, {MAX_SECONDS_PER_PARENT}s cutoff\n")
    for m in m_values:
        run(2, m)
        print()
