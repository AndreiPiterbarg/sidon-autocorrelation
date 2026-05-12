"""Quick benchmark: Python vs MATLAB-style throughput.
Uses n_half=2, m=20, c_target=1.28 (fast L0) for real timing.
Then extrapolates to m=50, n_half=3.
"""
import sys, os, time, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger'))
from cpu.run_cascade import process_parent_fused, run_level0

def matlab_style_l1(parents, d_child, n_half_child, m, c_target):
    """MATLAB-style: enumerate children as matrix, vectorized autoconv + window scan."""
    S = 4 * n_half_child * m
    gs = 1.0 / m
    d_parent = parents.shape[1]
    x = math.sqrt(c_target / d_child)
    total_ch = 0; total_surv = 0

    for pi in range(len(parents)):
        pc = parents[pi].astype(np.float64) / S
        ranges = []
        for j in range(d_parent):
            w = pc[j]
            lo = round((w - x) / gs) * gs
            hi = round(min(w, x) / gs) * gs
            r = np.arange(max(0, lo), hi + gs/2, gs)
            r = r[(r >= -1e-12) & (r <= w + 1e-12)]
            if len(r) == 0: r = np.array([max(0.0, min(w, x))])
            ranges.append(r)
        cart = 1
        for r in ranges: cart *= len(r)
        if cart == 0 or cart > 10_000_000: continue

        grids = np.meshgrid(*ranges, indexing='ij')
        flat = [g.ravel() for g in grids]
        N = len(flat[0])
        ch = np.empty((N, d_child), dtype=np.float64)
        for i in range(d_parent):
            ch[:, 2*i] = flat[i]; ch[:, 2*i+1] = pc[i] - flat[i]

        nc = 2 * d_child - 1
        conv = np.zeros((N, nc), dtype=np.float64)
        for i in range(d_child):
            for j in range(d_child):
                conv[:, i+j] += ch[:, i] * ch[:, j]

        alive = np.ones(N, dtype=bool)
        for ell in range(2, 2*d_child + 1):
            idx = np.where(alive)[0]
            if len(idx) == 0: break
            ncv = ell - 1; nw = nc - ncv + 1
            sc = conv[idx]; sch = ch[idx]
            for s in range(nw):
                ws = np.sum(sc[:, s:s+ncv], axis=1)
                TV = ws * (2*d_child) / ell
                lo_b = max(0, s-(d_child-1)); hi_b = min(d_child-1, s+ncv-1)
                W = np.sum(sch[:, lo_b:hi_b+1], axis=1)
                bound = (c_target + gs**2) + 2*gs*W
                p = TV >= bound
                if np.any(p):
                    alive[idx[p]] = False
                    idx = np.where(alive)[0]
                    if len(idx)==0: break
                    sc = conv[idx]; sch = ch[idx]
        total_ch += N; total_surv += int(np.sum(alive))
    return total_ch, total_surv

# ---- Config ----
configs = [
    (2, 20, 1.28),
    (2, 20, 1.40),
    (3, 20, 1.28),
]

for n_half, m, c_target in configs:
    d0 = 2 * n_half; d_child = 2 * d0; nhc = d_child // 2
    S = 4 * n_half * m
    print(f"\n{'='*60}")
    print(f"n_half={n_half}, m={m}, c_target={c_target}, d={d0}->{d_child}, S={S}")
    print(f"{'='*60}")

    # L0
    t0 = time.perf_counter()
    l0 = run_level0(n_half, m, c_target, verbose=False)
    t_l0 = time.perf_counter() - t0
    survivors = l0['survivors']
    print(f"L0: {l0['n_processed']:,} comps -> {len(survivors):,} survivors in {t_l0:.2f}s")

    if len(survivors) == 0:
        print("Proven at L0!"); continue

    N = min(50, len(survivors))
    parents = survivors[:N]

    # Warmup
    _ = process_parent_fused(parents[0], m, c_target, nhc)

    # Python
    py_ch = 0
    t0 = time.perf_counter()
    for i in range(N):
        s, nc = process_parent_fused(parents[i], m, c_target, nhc)
        py_ch += nc
    t_py = time.perf_counter() - t0
    print(f"Python:  {N} parents, {py_ch:,} children in {t_py:.3f}s -> {N/t_py:.1f} parents/s, {py_ch/t_py:,.0f} ch/s")

    # MATLAB-style
    t0 = time.perf_counter()
    mat_ch, mat_surv = matlab_style_l1(parents, d_child, nhc, m, c_target)
    t_mat = time.perf_counter() - t0
    print(f"MATLAB:  {N} parents, {mat_ch:,} children in {t_mat:.3f}s -> {N/t_mat:.1f} parents/s, {mat_ch/t_mat:,.0f} ch/s")

    if t_mat > 0 and t_py > 0:
        print(f"Speedup: {(N/t_py)/(N/t_mat):.1f}x (Python over MATLAB-numpy)")
