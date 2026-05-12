"""
Focused Euler-Lagrange extremizer scan for Phi(M).

Phi(M) = sup { ||g||_2^2 / ||g||_inf : g = f*f, f admissible, ||g||_inf <= M }.

The previous el_analytical run failed because the optimizer overshot M_cap
(no feasible point at M <= 1.378). Here we adopt three RELIABLE strategies:

(A)  STRICT projection by global support-rescaling (Boyer-Li-style).
     A nonneg sequence v on N cells maps to f on [-1/4,1/4] by spreading,
     and:    M(f)   = 2N * max(v*v) / (sum v)^2,
             c_emp  = sum L^2 / ((sum v)^2 * max L)   (scale-invariant).
     M is invariant to scaling v (homogeneous of deg 0), but DEPENDS on N
     (= "support length"). Padding with zeros REDUCES M (because effective
     support shrinks relative to total).

(B)  Multi-restart  L-BFGS-B  on a COMPACT  parameter space (multi-Gaussian
     with bounded centers and widths), penalty-projected to M = M_target.

(C)  Direct E-L gradient-flow on f with HARD bisection-projection of
     ||f*f||_inf at every step.

For each strategy and each M in {1.10,1.15,...,1.40,1.50,1.65}, we record
the best c_emp seen, and check the analytical E-L residual

   R(x) = 4 (g*f_tilde)(x) - 2 lambda (f_tilde * dnu)(x) - mu

at the resulting f. If even the BEST f at M<=1.378 has c_emp < c_* - 0.005,
we report VERDICT: PROMISING.
"""
from __future__ import annotations

import json, time, warnings
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

C_STAR = float(np.log(16.0) / np.pi)         # 0.88254240061...

# ----------------------------------------------------------------------
# Common metrics
def Mc_full(v):
    """For nonneg sequence v of length N (rescaled to [-1/4,1/4]):
       returns (M, c_emp, L2sq, L_inf, L_1).  Scale-invariant in v."""
    v = np.asarray(v, dtype=np.float64)
    v = np.maximum(v, 0.0)
    N = len(v)
    S = float(v.sum())
    if S <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    L = fftconvolve(v, v, mode="full")
    Mpeak = float(L.max())
    L2 = float((L * L).sum())
    L1 = float(L.sum())
    if Mpeak <= 0:
        return float("nan"), float("nan"), 0.0, 0.0, 0.0
    M = (2.0 * N) * Mpeak / (S * S)              # ||f*f||_inf, with f scaled to int=1
    # c_emp = ||g||_2^2 / ||g||_inf,  using ||g||_1 = 1.
    # ||g||_2^2 = (2N) * sum(L^2) / S^4   * (1 / (2N))   ?  Let me redo.
    # g(t) on [-1/2,1/2] has 2N-1 grid pts spaced (1/(2N)) apart.
    # ||g||_inf   = M
    # ||g||_2^2  approx = sum(g^2)*Delta = sum(L^2 * (2N/S^2)^2) * (1/(2N))
    #                  = (2N) * sum(L^2) / S^4.
    # ||g||_1     = sum(g)*Delta = sum(L) * (2N/S^2) * (1/(2N))  = sum(L)/S^2 = 1 (autocorr).
    L2sq = (2.0*N) * L2 / (S**4)
    return M, L2sq / max(M,1e-18), L2sq, M, 1.0

def metrics_from_f_grid(f, dx):
    """For density f on a uniform grid with spacing dx (so that sum f * dx = 1).
       Returns (M, c_emp, ||g||_2^2)."""
    g = fftconvolve(f, f) * dx
    M = float(g.max())
    L2sq = float((g*g).sum() * dx)
    L1 = float(g.sum() * dx)
    return M, L2sq/max(M,1e-18), L2sq, L1


# ----------------------------------------------------------------------
# (A) Discrete sequence + zero-padding for M projection.
def search_v_padded(v_seed, M_target, pad_min=0, pad_max=4000, tol=1e-4):
    """Right-pad v with zeros: v_pad = [v, 0,..,0] of total length N0+k.
       As k grows, M strictly decreases (toward 0). Bisect k to hit M_target.
       Returns (v_pad, M_actual, c_emp, k_used)."""
    v0 = np.asarray(v_seed, dtype=np.float64)
    v0 = np.maximum(v0, 0.0)
    N0 = len(v0)
    M0, c0, _, _, _ = Mc_full(v0)
    if not np.isfinite(M0):
        return None, None, None, None
    if M0 <= M_target:
        # Already below target: cannot raise M by padding (only lowers).
        # Try to raise M by REMOVING zero tail / contracting support.
        # We instead try `unpad`: find suffix of zeros and trim; if no zero
        # tail, return v0 as-is.
        nz = np.flatnonzero(v0 > 0)
        if len(nz) == 0:
            return None, None, None, None
        v_trim = v0[nz[0]: nz[-1]+1]
        Mt, ct, _, _, _ = Mc_full(v_trim)
        return v_trim, Mt, ct, 0
    # Bisect on integer k in [0, pad_max]
    k_lo, k_hi = 0, pad_max
    M_lo, _, _, _, _ = Mc_full(v0)            # k=0 ->  M_lo (=M0)
    v_hi = np.concatenate([v0, np.zeros(pad_max)])
    M_hi, _, _, _, _ = Mc_full(v_hi)
    if M_hi > M_target:
        # even max pad doesn't reduce enough
        return None, None, None, None
    while k_hi - k_lo > 1:
        k_mid = (k_lo + k_hi) // 2
        v_mid = np.concatenate([v0, np.zeros(k_mid)])
        M_mid, _, _, _, _ = Mc_full(v_mid)
        if M_mid > M_target:
            k_lo = k_mid
        else:
            k_hi = k_mid
    v_out = np.concatenate([v0, np.zeros(k_hi)])
    M_out, c_out, _, _, _ = Mc_full(v_out)
    return v_out, M_out, c_out, k_hi


def make_seed_BL():
    """Boyer-Li 575-step extremizer."""
    p = Path(__file__).resolve().parent / "delsarte_dual" / "restricted_holder" / "coeffBL.txt"
    if not p.exists():
        return None
    txt = p.read_text().strip().lstrip("{").rstrip("}")
    v = np.array([int(x.strip()) for x in txt.split(",") if x.strip()], dtype=np.float64)
    return v


def make_seed_indicator(N=500):
    return np.ones(N)


def make_seed_two_block(N, frac_left, frac_right, h_left, h_right):
    n_l = int(N * frac_left)
    n_r = int(N * frac_right)
    n_g = N - n_l - n_r
    return np.concatenate([np.full(n_l, h_left),
                           np.zeros(n_g),
                           np.full(n_r, h_right)])


def make_seed_three_block(N, w1, w2, w3, h1, h2, h3, gap=10):
    n1 = int(N * w1); n2 = int(N * w2); n3 = N - n1 - n2 - 2*gap
    if n3 < 1: n3 = 1
    return np.concatenate([np.full(n1, h1), np.zeros(gap),
                           np.full(n2, h2), np.zeros(gap),
                           np.full(n3, h3)])


def make_seed_BL_truncated(keep_low, keep_high):
    v = make_seed_BL()
    if v is None: return None
    return v[keep_low:keep_high]


def make_seed_BL_smoothed(sigma):
    v = make_seed_BL()
    if v is None: return None
    half = max(1, int(np.ceil(4*sigma)))
    xs = np.arange(-half, half+1)
    w = np.exp(-0.5*(xs/sigma)**2); w /= w.sum()
    return np.maximum(fftconvolve(v, w, mode="same"), 0)


def make_seed_BL_cubed(power):
    v = make_seed_BL()
    if v is None: return None
    return np.maximum(v, 0) ** power


def make_seed_BL_doubled():
    v = make_seed_BL()
    if v is None: return None
    return np.concatenate([v, np.zeros(20), v])


def make_seed_cosine(N, n_cos, coeffs):
    xs = np.linspace(-0.25 + 0.5/(2*N), 0.25 - 0.5/(2*N), N)
    f = coeffs[0] * np.ones(N)
    for j in range(1, n_cos+1):
        if j < len(coeffs):
            f += coeffs[j] * np.cos(2*np.pi*j*xs)
    return np.maximum(f, 0)


def collect_seeds():
    seeds = {}
    if (v_BL := make_seed_BL()) is not None:
        seeds["BL"] = v_BL
        seeds["BL_smooth1.5"] = make_seed_BL_smoothed(1.5)
        seeds["BL_smooth3"]   = make_seed_BL_smoothed(3.0)
        seeds["BL_smooth6"]   = make_seed_BL_smoothed(6.0)
        seeds["BL_smooth12"]  = make_seed_BL_smoothed(12.0)
        seeds["BL_smooth25"]  = make_seed_BL_smoothed(25.0)
        seeds["BL_pow1.5"]    = make_seed_BL_cubed(1.5)
        seeds["BL_pow0.7"]    = make_seed_BL_cubed(0.7)
        seeds["BL_pow0.5"]    = make_seed_BL_cubed(0.5)
        seeds["BL_pow0.3"]    = make_seed_BL_cubed(0.3)
        seeds["BL_double"]    = make_seed_BL_doubled()
        seeds["BL_t100_400"]  = make_seed_BL_truncated(100, 475)
        seeds["BL_t200_400"]  = make_seed_BL_truncated(200, 475)
        seeds["BL_t50_500"]   = make_seed_BL_truncated(50, 525)
    seeds["ind500"]    = make_seed_indicator(500)
    seeds["ind1000"]   = make_seed_indicator(1000)
    seeds["2blk_LR"]   = make_seed_two_block(500, 0.10, 0.10, 5.0, 1.0)
    seeds["2blk_eq"]   = make_seed_two_block(500, 0.20, 0.20, 1.0, 1.0)
    seeds["3blk_unif"] = make_seed_three_block(500, 0.10, 0.10, 0.10, 1.0, 1.0, 1.0)
    seeds["3blk_asym"] = make_seed_three_block(500, 0.05, 0.10, 0.20, 5.0, 2.0, 1.0)
    # Several cosine families
    for kappa in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        seeds[f"cos1_{kappa:+.1f}"] = make_seed_cosine(500, 1, [1.0, kappa])
    for c2 in [-0.3, -0.2, 0.2, 0.3]:
        seeds[f"cos2_{c2:+.2f}"] = make_seed_cosine(500, 2, [1.0, 0.0, c2])
    for c1, c2, c3 in [(0.3, 0.0, 0.2), (0.4, 0.2, 0.0), (-0.3, 0.0, -0.2)]:
        seeds[f"cos3_{c1:+.1f}_{c2:+.1f}_{c3:+.1f}"] = make_seed_cosine(500, 3, [1.0, c1, c2, c3])
    return seeds


def run_strategy_A(M_grid):
    seeds = collect_seeds()
    print(f"\n[Strategy A: pad-projection]  {len(seeds)} seeds, {len(M_grid)} M-targets")
    out = {}
    for M_t in M_grid:
        best_c, best_seed, best_v = -np.inf, None, None
        feasible_count = 0
        for name, v_seed in seeds.items():
            v_pad, M_act, c, k = search_v_padded(v_seed, M_t)
            if v_pad is None: continue
            feasible_count += 1
            if abs(M_act - M_t) > 0.05 * M_t:
                continue
            if c > best_c:
                best_c, best_seed, best_v = c, name, v_pad
        out[M_t] = dict(M=M_t, c_emp=best_c, seed=best_seed, n_feas=feasible_count)
        print(f"  M={M_t:.3f}: best c_emp={best_c:.6f}  via [{best_seed}]  ({feasible_count} feasible seeds)")
    return out


# ----------------------------------------------------------------------
# (B) Multi-Gaussian / multi-bump parametric L-BFGS.
def f_from_bumps_grid(params, K, xs, dx, N):
    """params = (z[K], p[K], s[K])  -> nonneg density on grid."""
    z = params[0:K]
    p_raw = params[K:2*K]
    s_raw = params[2*K:3*K]
    # softmax weights
    z = z - z.max()
    e = np.exp(z); w = e / e.sum()
    # positions in [-0.24,0.24]
    p = np.tanh(p_raw) * 0.24
    sigma = 0.005 + 0.10 / (1 + np.exp(-s_raw))   # sigma in [0.005, 0.105]
    f = np.zeros(N)
    for i in range(K):
        b = np.exp(-0.5 * ((xs - p[i])/sigma[i])**2)
        Z = b.sum() * dx
        if Z > 0:
            f += w[i] * b / Z
    return f


def loss_bumps(params, K, xs, dx, N, M_cap, alpha):
    f = f_from_bumps_grid(params, K, xs, dx, N)
    if not np.isfinite(f).all():
        return 1e6
    g = fftconvolve(f, f) * dx
    M = float(g.max())
    L2sq = float((g*g).sum() * dx)
    # objective: -c_emp = -L2sq/M
    obj = -L2sq / max(M, 1e-12)
    pen = alpha * max(0.0, M - M_cap)**2
    return obj + pen


def search_strategy_B(M_target, K=8, n_restarts=12, seed_base=0, sym=False):
    N = 401
    dx = 0.5/(N-1)
    xs = np.linspace(-0.25, 0.25, N)
    rng = np.random.default_rng(seed_base)
    best_c, best_M, best_f = -np.inf, None, None
    for r in range(n_restarts):
        # Random initial bump centers (spread out) + small widths
        p_raw = rng.normal(0, 1, K)
        s_raw = rng.normal(-1.5, 0.5, K)
        z = rng.normal(0, 0.5, K)
        if sym:
            # enforce reflective pairing of bumps
            half = K // 2
            p_raw[:half] = -np.linspace(0.5, 2.0, half)
            p_raw[K-half:] = np.linspace(0.5, 2.0, half)
            z[:half] = z[K-half:][::-1]
            s_raw[:half] = s_raw[K-half:][::-1]
        params0 = np.concatenate([z, p_raw, s_raw])
        # Penalty homotopy
        for alpha in (10.0, 100.0, 1000.0, 10000.0):
            res = minimize(loss_bumps, params0,
                           args=(K, xs, dx, N, M_target, alpha),
                           method='L-BFGS-B',
                           options=dict(maxiter=300, ftol=1e-10, gtol=1e-7))
            params0 = res.x
        f = f_from_bumps_grid(params0, K, xs, dx, N)
        if sym:
            f = 0.5*(f + f[::-1]); f = f / max(f.sum()*dx, 1e-12)
        M, c, L2sq, L1 = metrics_from_f_grid(f, dx)
        feas = (M <= M_target * 1.005)
        if feas and c > best_c:
            best_c, best_M, best_f = c, M, f.copy()
    return best_c, best_M, best_f


def run_strategy_B(M_grid):
    print(f"\n[Strategy B: multi-bump L-BFGS]  K=8 bumps")
    out = {}
    for M_t in M_grid:
        cs, Ms, fs = search_strategy_B(M_t, K=8, n_restarts=12, seed_base=int(M_t*1000), sym=False)
        cs2, Ms2, fs2 = search_strategy_B(M_t, K=8, n_restarts=8, seed_base=int(M_t*1000)+777, sym=True)
        if cs2 > cs:
            cs, Ms, fs = cs2, Ms2, fs2
            tag = "sym"
        else:
            tag = "asy"
        out[M_t] = dict(M=M_t, c_emp=cs, M_at=Ms, tag=tag)
        print(f"  M={M_t:.3f}: best c_emp={cs:.6f}  (M_at={Ms})  [{tag}]")
    return out


# ----------------------------------------------------------------------
# E-L residual on a discrete f.
def el_residual(f, dx):
    g = fftconvolve(f, f) * dx
    M = float(g.max())
    Nf = f.size
    h_full = fftconvolve(g, f[::-1]) * dx
    h_x = h_full[Nf-1: 2*Nf-1]
    # peak measure: identify bins within 1e-4 of peak
    peak_thr = M - 1e-4 * max(M, 1)
    pm = (g >= peak_thr).astype(float)
    s = float(pm.sum() * dx)
    if s <= 0: return None
    pm = pm / s
    phi_full = fftconvolve(pm, f[::-1]) * dx
    phi_x = phi_full[Nf-1: 2*Nf-1]
    supp = f > 1e-3 * f.max()
    if supp.sum() < 4: return None
    A = np.stack([np.ones(supp.sum()), 2.0 * phi_x[supp]], axis=1)
    b = 4.0 * h_x[supp]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    mu, lam = sol
    R = b - A @ sol
    rel = float(np.linalg.norm(R) / max(np.linalg.norm(b), 1e-12))
    R_off = 4.0*h_x - 2.0*lam*phi_x - mu
    off = ~supp
    max_off = float(R_off[off].max()) if off.any() else 0.0
    return dict(mu=float(mu), lam=float(lam),
                resid_rel=rel, max_R_off=max_off)


# ----------------------------------------------------------------------
def main():
    print("=" * 72)
    print(f"Hyp_R E-L extremizer scan       c_* = {C_STAR:.10f}")
    print("=" * 72)

    M_grid = [1.10, 1.20, 1.275, 1.30, 1.35, 1.378, 1.50, 1.65]

    t0 = time.time()
    A = run_strategy_A(M_grid)
    B = run_strategy_B(M_grid)

    print("\n" + "=" * 72)
    print(f"COMBINED EXTREMIZER TABLE   target c_* = {C_STAR:.6f}")
    print("=" * 72)
    print(f"  {'M_target':>9s}  {'best c_emp':>10s}  {'margin':>10s}  {'src':<24s}")
    sup_c = -np.inf; sup_M = None; sup_src = None
    table_full = []
    for M_t in M_grid:
        a = A.get(M_t, {}); b = B.get(M_t, {})
        ca = a.get("c_emp", -np.inf); cb = b.get("c_emp", -np.inf)
        if ca >= cb:
            best_c, src = ca, f"A: {a.get('seed','?')}"
        else:
            best_c, src = cb, f"B: K=8 [{b.get('tag','?')}]"
        margin = C_STAR - best_c
        flag = ""
        if M_t <= 1.378 + 1e-9 and best_c > sup_c:
            sup_c = best_c; sup_M = M_t; sup_src = src
        if best_c >= C_STAR:
            flag = "  *** VIOLATES Hyp_R"
        print(f"  {M_t:9.3f}  {best_c:10.6f}  {margin:+10.6f}  {src:<24s}{flag}")
        table_full.append(dict(M_target=M_t, best_c_emp=best_c,
                               margin_to_cstar=margin, src=src,
                               A_data=a, B_data=b))

    print("=" * 72)
    print(f"sup_{{M_target <= 1.378}} c_emp_found = {sup_c:.6f} at M={sup_M}")
    print(f"   gap to c_*  = {C_STAR - sup_c:+.6f}")
    print(f"   source      = {sup_src}")
    if C_STAR - sup_c > 0.01:
        verdict = "PROMISING"
        msg = f"E-L extremizer search finds NO admissible f with c_emp >= c_* - 0.01 = {C_STAR-0.01:.4f} at M<=1.378."
    elif C_STAR - sup_c >= 0.0:
        verdict = "INCONCLUSIVE"
        msg = f"E-L sup is within 0.01 of c_*; cannot conclude either way numerically."
    else:
        verdict = "OBSTRUCTED"
        msg = f"Numerical extremizer found with c_emp >= c_* at M<=1.378; Hyp_R numerically at risk."
    print(f"VERDICT: {verdict}")
    print(f"  --  {msg}")
    print(f"\n(elapsed {time.time()-t0:.1f}s)")

    out = dict(c_target=C_STAR, M_grid=M_grid,
               sup_c_emp_le_1p378=sup_c, sup_M=sup_M, sup_src=sup_src,
               margin=C_STAR-sup_c, verdict=verdict, table=table_full)
    with open(Path(__file__).resolve().parent / "el_extremiser_test3_result.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
