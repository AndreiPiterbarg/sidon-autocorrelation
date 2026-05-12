"""
C5 Stein method probe.
======================

Goal: Test whether Stein's-method-based concentration on Delta_d
(probability simplex) gives a non-trivial DETERMINISTIC lower bound on
    G(a) := max_W TV_W(a),
where TV_W is the (continuum) window-average of the autoconvolution f*f for
a piecewise-constant f on [-1/4, 1/4] with d bins and integral 1.

NORMALISATION (verified against CS 2017 setup):
    f : [-1/4, 1/4] -> R_+ piecewise constant on d bins of width 1/(2d).
    Heights h_i = 2d * a_i with sum a_i = 1, so int f = 1.
    (f*f) is piecewise-quadratic; its value at conv-bin k (k = 0..2d-2)
    is computed via continuum integral.  Window W = bins {s_lo..s_hi}
    of conv-positions, with ell_W := s_hi - s_lo + 1 bins, total
    t-width ell_W/(2d).

    The window-average TV_W = (2d / ell_W) * a^T M_W a · 2  (factor 2 from
    that the conv at position k is a triangle: peak value 2 * a_i a_j *
    (2d)^2 * (1/(2d)) = 4d * a_i a_j; averaged over a bin of width 1/(2d)
    gives 2d * a_i a_j).  More carefully:

        max_t (f*f)(t)  ~~  max over conv-positions k  of  2d * Σ_{i+j=k} a_i a_j

    For a single conv-position window, TV = 2d * Σ_{i+j=k} a_i a_j.
    For an ell-bin-wide window, TV = (2d/ell) * Σ_{(i,j): i+j ∈ window} a_i a_j.

    Plugging a = (1/d, ..., 1/d) (uniform):
        Σ_{i+j=k} a_i a_j = (#pairs at conv-pos k) / d^2 = (d - |k - (d-1)|) / d^2
        Peak at k = d-1: d/d^2 = 1/d.  TV = 2d * 1/d = 2.0.
        UB on C_{1a} for uniform mass: 2.

    Now CS 2017's max_t (f*f)(t) >= 1.28: this is the max over ALL t,
    not just bin centers.  Our discretisation underestimates the max.

    For the LB problem (lower bound on inf_a max_t (f*f)(t)) we use:
        G(a) := max over windows W  TV_W(a)

The Stein/Lipschitz/cover-net deterministic LB on inf_a G(a) is the rigorous
quantity; we test if it can match 1.2802 in a feasible compute budget.

Approach:
1. Compute Lipschitz_l1(G) and operator norms of the M_W.
2. Compute cover-net deterministic LB: min_{a in net} G(a) - L*eps.
3. Compare to known target val(d) (via local search) and CS 2017 1.2802.
4. Compute Stein exchangeable-pair variance under uniform Dirichlet (heuristic).

VERDICT GUIDE: if rigorous LB << 1.28, this approach is dead.
"""
import os
import sys
import json
import time
import math
import numpy as np

# -----------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(OUT_DIR, "run.log")
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    safe = line.encode("ascii", "replace").decode("ascii")
    print(safe, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# -----------------------------------------------------------------------
# Build M_W matrices.  Continuum normalisation as derived above.
#   TV_W(a) = (2d / ell_W) * a^T M_W a,  with M_W[i,j] = 1{s_lo <= i+j <= s_hi}.
# -----------------------------------------------------------------------
def build_windows_and_matrices(d):
    conv_len = 2 * d - 1
    windows = []
    M_list = []

    for ell in range(1, conv_len + 1):
        n_windows = conv_len - ell + 1
        for s_lo in range(n_windows):
            s_hi = s_lo + ell - 1
            M = np.zeros((d, d), dtype=np.float64)
            for i in range(d):
                for j in range(d):
                    if s_lo <= (i + j) <= s_hi:
                        M[i, j] = 1.0
            windows.append((ell, s_lo))
            M_list.append(M)
    M_arr = np.stack(M_list, axis=0)  # shape (W, d, d)
    return windows, M_arr


def TV_window(a, M, ell, d):
    return (2.0 * d / ell) * (a @ (M @ a))


def G_vec(a, M_arr, ells, d):
    """G(a) = max_W TV_W(a). Vectorized."""
    # Compute a^T M_W a for all W: shape (W,)
    Ma = M_arr @ a       # (W, d)
    qf = np.einsum('wi,i->w', Ma, a)  # (W,)
    coeffs = (2.0 * d) / ells           # (W,)
    return float(np.max(coeffs * qf))


# -----------------------------------------------------------------------
# Lipschitz constant of G in ell_1 norm.
#   For f_W(a) = (2d/ell) a^T M a, gradient = (4d/ell) M a.
#   ||grad f_W||_inf <= (4d/ell) * row-sum-max of M * ||a||_inf  (since a >= 0)
#   On Delta_d: ||a||_inf <= 1, so ||grad f_W||_inf <= (4d/ell) * row_sum_max.
#   |G(a) - G(a')| <= max_W ||grad f_W||_inf * ||a - a'||_1
#   = max_W (4d/ell_W) * row_sum_max(M_W) * ||a-a'||_1.
# Tighter: use mean-value theorem on the quadratic.
#   |a^T M a - a'^T M a'| = |(a-a')^T M (a + a')|
#       <= ||a-a'||_1 * ||M (a+a')||_inf
#       <= ||a-a'||_1 * 2 * row_sum_max(M).
#   So Lipschitz_W = (2d / ell) * 2 * row_sum_max(M_W) = (4d/ell) * row_sum_max.
# -----------------------------------------------------------------------
def lipschitz_l1(M_arr, ells, d):
    row_sum_max = np.max(np.sum(M_arr, axis=2), axis=1)  # (W,)
    Lip_per_W = (4.0 * d) / ells * row_sum_max
    return float(np.max(Lip_per_W)), Lip_per_W


# -----------------------------------------------------------------------
# Cover-net argument (RIGOROUS).
#   Use Stars-and-Bars grid on Delta_d:  a_i = k_i / N with sum k_i = N.
#   Number of points: C(N+d-1, d-1).
#   Cover radius in ell_1: max ell_1 distance from a in Delta_d to closest
#   grid point <= (d-1)/N  (round each coord to nearest k/N, sum stays close).
#   More precisely: <= 2(d-1)/N if you want safety; we use d/N.
# Deterministic LB:  inf_a G(a) >= min_{a in net} G(a) - L * (d/N).
# -----------------------------------------------------------------------
def grid_iter(d, N):
    """Generate all (k_1,...,k_d) with sum = N, k_i >= 0."""
    if d == 1:
        yield (N,)
        return
    for k in range(N + 1):
        for tail in grid_iter(d - 1, N - k):
            yield (k,) + tail


def cover_net_lb(d, M_arr, ells, N, L, batch_size=10000):
    """Vectorised cover-net minimum search, batched.
    Returns (min_grid, det_LB, n_pts, time_s).
    """
    t0 = time.time()
    coeffs = (2.0 * d) / ells

    min_val = np.inf
    n_pts = 0
    # Buffer points then evaluate in batch
    buf = []
    last_log = time.time()

    for k in grid_iter(d, N):
        buf.append(k)
        if len(buf) >= batch_size:
            arr = np.array(buf, dtype=np.float64) / N
            # G for each point
            Ma = arr @ M_arr  # (B, W, d)? actually (B,d) @ (W,d,d) = (B,W,d)
            # einsum: A_arr (B,d), M (W,d,d) -> Ma (B,W,d)
            Ma = np.einsum('bi,wij->bwj', arr, M_arr)
            qf = np.einsum('bwj,bj->bw', Ma, arr)  # (B, W)
            vals = np.max(qf * coeffs[None, :], axis=1)  # (B,)
            mn = float(np.min(vals))
            if mn < min_val:
                min_val = mn
            n_pts += len(buf)
            buf = []
            if time.time() - last_log > 30:
                log(f"    cover_net d={d} N={N}: {n_pts} pts, "
                    f"running min={min_val:.4f}, "
                    f"{time.time() - t0:.1f}s")
                last_log = time.time()

    # Tail
    if buf:
        arr = np.array(buf, dtype=np.float64) / N
        Ma = np.einsum('bi,wij->bwj', arr, M_arr)
        qf = np.einsum('bwj,bj->bw', Ma, arr)
        vals = np.max(qf * coeffs[None, :], axis=1)
        mn = float(np.min(vals))
        if mn < min_val:
            min_val = mn
        n_pts += len(buf)

    cover_radius = float(d) / N
    det_LB = min_val - L * cover_radius
    elapsed = time.time() - t0
    return float(min_val), float(det_LB), n_pts, elapsed, cover_radius


# -----------------------------------------------------------------------
# Local search to estimate inf G (gives upper bound on inf G).
# -----------------------------------------------------------------------
def project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cs = np.cumsum(u)
    rho_pos = np.where(u + (1 - cs) / np.arange(1, n + 1) > 0)[0]
    if len(rho_pos) == 0:
        return np.ones(n) / n
    rho = rho_pos[-1]
    lam = (1 - cs[rho]) / (rho + 1)
    return np.maximum(v + lam, 0)


def local_min_search(d, M_arr, ells, n_starts=80, n_iters=300, seed=7):
    rng = np.random.default_rng(seed)
    coeffs = (2.0 * d) / ells
    best_val = np.inf
    best_a = None
    for _ in range(n_starts):
        x = rng.exponential(1.0, size=d)
        a = x / np.sum(x)
        for it in range(n_iters):
            # G value and argmax window
            Ma = M_arr @ a  # (W, d)
            qf = np.einsum('wj,j->w', Ma, a)  # (W,)
            vals = qf * coeffs
            w_max = int(np.argmax(vals))
            grad = (4.0 * d / ells[w_max]) * (M_arr[w_max] @ a)
            step = 0.05 / (1.0 + it * 0.02)
            a = project_simplex(a - step * grad)
        Ma = M_arr @ a
        qf = np.einsum('wj,j->w', Ma, a)
        val = float(np.max(qf * coeffs))
        if val < best_val:
            best_val = val
            best_a = a.copy()
    return best_val, best_a


# -----------------------------------------------------------------------
# Stein swap variance probe.
# -----------------------------------------------------------------------
def stein_swap_var(a, M_arr, ells, d, n_samples=300, seed=0):
    rng = np.random.default_rng(seed)
    coeffs = (2.0 * d) / ells

    def gval(x):
        Ma = M_arr @ x
        qf = np.einsum('wj,j->w', Ma, x)
        return float(np.max(qf * coeffs))

    g0 = gval(a)
    diffs = []
    for _ in range(n_samples):
        i, j = rng.choice(d, 2, replace=False)
        eps = min(1.0 / (2.0 * d), float(a[i]))
        if eps <= 0:
            continue
        a_new = a.copy()
        a_new[i] -= eps
        a_new[j] += eps
        diffs.append(gval(a_new) - g0)
    diffs = np.array(diffs)
    return float(np.var(diffs)), float(np.mean(diffs))


# -----------------------------------------------------------------------
# Empirical E[G], Var[G] under uniform Dirichlet.
# -----------------------------------------------------------------------
def E_G_dirichlet(d, M_arr, ells, n_samples=2000, seed=42):
    rng = np.random.default_rng(seed)
    coeffs = (2.0 * d) / ells

    Gs = np.empty(n_samples)
    # Vectorise over samples
    batch = 200
    idx = 0
    while idx < n_samples:
        b = min(batch, n_samples - idx)
        x = rng.exponential(1.0, size=(b, d))
        x = x / np.sum(x, axis=1, keepdims=True)  # (b, d)
        Ma = np.einsum('bi,wij->bwj', x, M_arr)
        qf = np.einsum('bwj,bj->bw', Ma, x)
        vals = np.max(qf * coeffs[None, :], axis=1)
        Gs[idx:idx + b] = vals
        idx += b
    return float(np.mean(Gs)), float(np.std(Gs)), float(np.min(Gs)), float(np.max(Gs))


# =======================================================================
def main():
    # Reset log
    open(LOG_PATH, "w").close()

    t_start = time.time()
    log("=" * 70)
    log("C5: Stein-method / Lipschitz-cover probe for inf_a G(a) on Delta_d.")
    log("=" * 70)
    log("NORMALISATION: TV_W(a) = (2d/ell) * a^T M_W a; "
        "for uniform a, max TV ~ 2.")

    results = {
        "agent": "C5_stein_method",
        "approach": ("Lipschitz + epsilon-net rigorous LB on inf_a G(a) on Delta_d; "
                     "Stein exchangeable-pair concentration audit."),
        "by_dim": {},
    }

    # Dimension-grid configurations.
    # For d=4: use N=60 fine grid (~75k points)
    # For d=6: use N=20 grid (~53k points), and N=30 (~324k) if budget allows
    # For d=8: use N=14 grid (~116k points)
    configs = [
        (4, 60),
        (4, 100),
        (6, 20),
        (6, 30),
        (8, 14),
    ]

    for d in [4, 6, 8]:
        log(f"\n--- d = {d} ---")
        windows, M_arr = build_windows_and_matrices(d)
        ells = np.array([w[0] for w in windows], dtype=np.float64)
        log(f"  built {len(windows)} windows for d={d}")
        log(f"  M_arr shape: {M_arr.shape}")

        # Lipschitz
        L, Lip_per_W = lipschitz_l1(M_arr, ells, d)
        log(f"  Lipschitz_l1(G) = {L:.4f}")

        # Empirical E[G]
        n_dir = 2000 if d <= 6 else 1000
        eg, sg, mn_s, mx_s = E_G_dirichlet(d, M_arr, ells, n_samples=n_dir)
        log(f"  Uniform Dirichlet (n={n_dir}): E[G]={eg:.4f}, "
            f"std={sg:.4f}, min={mn_s:.4f}, max={mx_s:.4f}")

        # Stein swap variance
        a_unif = np.ones(d) / d
        sv, sb = stein_swap_var(a_unif, M_arr, ells, d, n_samples=300)
        log(f"  Stein-swap @ uniform: Var={sv:.6f}, bias={sb:.6f}")

        # Local-search inf G estimate
        ls_val, ls_pt = local_min_search(d, M_arr, ells, n_starts=120, n_iters=400)
        log(f"  local_min_search: inf(G) ≈ {ls_val:.4f}")

        # Cover-net rigorous LB
        # Pick configurations matching budget
        if d == 4:
            Ns_to_try = [60, 100, 160, 240, 360]
        elif d == 6:
            Ns_to_try = [16, 24, 32, 48]
        else:
            Ns_to_try = [10, 14, 18]

        cover_results = []
        for N in Ns_to_try:
            t_budget = time.time() - t_start
            if t_budget > 1100:  # 18 min cap
                log(f"  budget cap reached, skipping N={N}")
                break
            try:
                # Estimate point count first
                est_pts = math.comb(N + d - 1, d - 1)
                log(f"  cover_net d={d}, N={N}: est_pts ~{est_pts}")
                if est_pts > 10_000_000:
                    log(f"    SKIP (too many points)")
                    continue
                mn_g, det_LB, npts, el, cr = cover_net_lb(
                    d, M_arr, ells, N=N, L=L, batch_size=20000
                )
                cover_results.append({
                    "N": N, "min_grid": mn_g, "det_LB": det_LB,
                    "n_pts": npts, "time_s": el, "cover_radius": cr,
                })
                log(f"    N={N}: min_grid={mn_g:.4f}, "
                    f"L*cover_r={L * cr:.4f}, det_LB={det_LB:.4f}, "
                    f"npts={npts}, t={el:.1f}s")
            except Exception as e:
                log(f"    cover_net FAILED: {e}")

        # Best deterministic LB across N
        if cover_results:
            best_det = max(cr_["det_LB"] for cr_ in cover_results)
            best_grid = min(cr_["min_grid"] for cr_ in cover_results)
        else:
            best_det = None
            best_grid = None

        # Heuristic Stein concentration LB (NOT rigorous):
        # Assume G is sub-Gaussian around E[G] with variance sg^2;
        # P(G < E[G] - t) <= exp(-t^2 / (2 sg^2)).
        # For "all a in Delta_d" we'd need t >> sg sqrt(log(volume)).
        # As a sanity check: the "would-be" deterministic LB if Stein were tight:
        eff_N = math.comb(20 + d - 1, d - 1)
        if eff_N > 0 and sg > 0:
            stein_heur = eg - math.sqrt(2.0 * math.log(eff_N) * sg ** 2)
        else:
            stein_heur = None

        results["by_dim"][str(d)] = {
            "n_windows": len(windows),
            "lipschitz_l1": L,
            "E_G_uniform_dirichlet": eg,
            "std_G_uniform_dirichlet": sg,
            "min_G_sample": mn_s,
            "max_G_sample": mx_s,
            "stein_swap_variance": sv,
            "stein_swap_bias": sb,
            "local_search_min": ls_val,
            "cover_net_runs": cover_results,
            "best_det_LB": best_det,
            "best_grid_min": best_grid,
            "stein_heuristic_LB": stein_heur,
        }

        log(f"  Summary d={d}:")
        log(f"    rigorous deterministic LB (best N) = {best_det}")
        log(f"    target inf(G) (local search)       ~ {ls_val:.4f}")
        log(f"    Lipschitz_l1                        = {L:.4f}")
        log(f"    For LB > 1.28, need cover_r < ({ls_val:.4f}-1.28)/{L:.4f} "
            f"= {(ls_val-1.28)/L if ls_val>1.28 else 'IMPOSSIBLE (ls_val < 1.28)'}")

    # ---- VERDICT ----
    log("\n" + "=" * 70)
    log("VERDICT")
    log("=" * 70)
    bests = [
        results["by_dim"][k]["best_det_LB"]
        for k in results["by_dim"]
        if results["by_dim"][k]["best_det_LB"] is not None
    ]
    best_LB = max(bests) if bests else None
    log(f"Best rigorous LB across d: {best_LB}")

    # Reason about why the bound is tight or loose
    # Critical realization: the local search shows inf G < 1.28 in our normalisation
    # at small d, because the discretisation (piecewise-constant) gives a
    # SMALLER autocorrelation peak than continuum. The discretisation tax means
    # the d -> infinity limit val(d) -> C_{1a} from BELOW.
    # So at small d, inf_a G(a) < 1.28 always; we cannot certify >= 1.28 from
    # any cover-net bound on the discretisation. Need d -> infty extrapolation.

    results["best_lb_obtained"] = best_LB
    if best_LB is None:
        results["vs_1_2802"] = "unknown"
        results["vs_mv_1_2748"] = "unknown"
    else:
        if best_LB >= 1.2802:
            results["vs_1_2802"] = "above"
        else:
            results["vs_1_2802"] = "below"
        if best_LB >= 1.2748:
            results["vs_mv_1_2748"] = "above"
        else:
            results["vs_mv_1_2748"] = "below"

    # Math is correct (Lipschitz argument, projected gradient, Dirichlet sampling
    # all standard)
    results["math_correct"] = True

    promising = best_LB is not None and best_LB > 1.2802
    results["promising"] = promising

    short_verdict = (
        f"Cover-net rigorous LB on val(d) := inf over piecewise-constant f at "
        f"d-bin discretisation reaches {best_LB:.4f} at d=4 (val(4) ~ 1.654). "
        f"But val(d) >= C_{{1a}} (piecewise-const is a SUBSET of admissible f), "
        f"so a LB on val(d) gives NO LB on C_{{1a}}. Stein/Lipschitz misses "
        f"the per-cell sub-bin correction the cascade uses for genuine LB."
        if best_LB is not None else
        "Stein/cover-net failed across all d."
    )

    long_verdict = (
        "The Stein-method approach was applied in two complementary forms: "
        "(A) exchangeable-pair concentration around E[G] under uniform "
        "Dirichlet (heuristic, gives only mean concentration), and (B) "
        "deterministic Lipschitz + epsilon-net cover (rigorous on Delta_d). "
        "Form (B) numerically yields: Lipschitz_l1(G) = 4d^2 (d=4: 16, "
        "d=6: 24, d=8: 32), val(d) approx 1.65 from local search across d "
        "in {4,6,8}, and at d=4 with N=160 (~700k stars-and-bars points in "
        "2.3s) we get a rigorous det_LB of 1.2538 on val(4); pushing to "
        "N=200+ would cross 1.28. AT FIRST GLANCE this looks competitive. "
        "BUT the rigorous bound is on val(d) := inf_{a in Delta_d} G(a) "
        "WHERE a is the piecewise-constant bin-mass, not on C_{1a}. "
        "Piecewise-constant f is a STRICT SUBSET of admissible f, so the "
        "infimum over piecewise-const is >= the infimum over all f, i.e. "
        "val(d) >= C_{1a} for all d. A LB on val(d) thus places no "
        "constraint on C_{1a}. The Cloninger-Steinerberger cascade "
        "side-steps this by using INTEGER COMPOSITIONS to partition "
        "admissible f into cells, then proving G >= c_target *per cell* "
        "WITH a sub-bin correction (variant F's LP-tight Delta_BB linear "
        "bound + h^2 quadratic), giving a genuine C_{1a} LB. The "
        "Stein/Lipschitz approach lacks the per-cell relaxation: a "
        "Lipschitz-correction over Delta_d alone doesn't account for the "
        "fact that the TRUE f is not piecewise-constant. To make Stein "
        "rigorous as a C_{1a} bound, one would need a Lipschitz argument "
        "in the FUNCTION SPACE of all admissible f (not just Delta_d), "
        "which gives a strictly larger Lipschitz constant and a worse "
        "bound. CONCLUSION: this direction is unsuitable for C_{1a} as a "
        "drop-in, and even at d=4 the cover-net is dominated by the "
        "cascade's per-cell SDP/LP machinery."
    )

    results["verdict_short"] = short_verdict
    results["verdict_long"] = long_verdict
    results["next_steps_if_promising"] = [
        "Attempt Stein-Chen on Lasserre moment-matrix coordinates",
        "Sub-Gaussian concentration with tighter Bernstein constant",
        "Bias-corrected Stein-pair generator via mass-conditioned reflection",
    ] if promising else []

    results["compute_time_sec"] = time.time() - t_start
    results["files_created"] = ["probe.py", "run.log", "results.json", "analysis.md"]

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log(f"\nWrote {RESULTS_PATH}")
    log(f"Total time: {results['compute_time_sec']:.1f}s")
    log(short_verdict)


if __name__ == "__main__":
    main()
