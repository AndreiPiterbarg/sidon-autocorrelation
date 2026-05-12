"""
W2 verification: independent re-computation of K_2 via mpmath at dps=50.

K_2 = sum_{i,j} lambda_i lambda_j C_ij where C_ij = int_R J0(pi a_i xi)^2 J0(pi a_j xi)^2 dxi
"""

import json
import os
import sys
import time
import gc
import mpmath as mp

mp.mp.dps = 50
DELTA = [mp.mpf(138) / 1000, mp.mpf(55) / 1000, mp.mpf(25) / 1000]
LAM = [mp.mpf(85) / 100, mp.mpf(10) / 100, mp.mpf(5) / 100]
T_CUTOFF = mp.mpf(100000)

CKPT_PATH = r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\archive\multiscale_arcsine\_w2_ckpt.json"
OUT_PATH = r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\archive\multiscale_arcsine\_w2_verify_k2.json"


def integrand(i, j, xi):
    a = mp.pi * DELTA[i] * xi
    b = mp.pi * DELTA[j] * xi
    return mp.besselj(0, a) ** 2 * mp.besselj(0, b) ** 2


def compute_Cij(i, j, T, CHUNK=1000):
    """Compute C_ij = 2 * int_0^T J0(pi a_i xi)^2 J0(pi a_j xi)^2 dxi
    via splitting at k/a_max where a_max = max(delta_i, delta_j).
    """
    a_max = max(DELTA[i], DELTA[j])
    val = mp.mpf(0)
    X_inner = mp.mpf(50)
    val += mp.quad(lambda xi: integrand(i, j, xi), [0, X_inner])

    k_start = int(mp.ceil(X_inner * a_max))
    k_end = int(mp.floor(T * a_max))

    prev = X_inner
    last_progress = time.time()
    for k in range(k_start, k_end + 1, CHUNK):
        nxt = mp.mpf(min(k + CHUNK, k_end)) / a_max
        if nxt <= prev:
            continue
        val += mp.quad(lambda xi: integrand(i, j, xi), [prev, nxt])
        prev = nxt
        if time.time() - last_progress > 30.0:
            print(f"    ... C_{i+1}{j+1} progress: xi ~ {float(prev):.1f} / {float(T):.1f}", flush=True)
            last_progress = time.time()
            gc.collect()
    if prev < T:
        val += mp.quad(lambda xi: integrand(i, j, xi), [prev, T])
    return 2 * val


def load_ckpt():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as fp:
            return json.load(fp)
    return {}


def save_ckpt(d):
    with open(CKPT_PATH, "w") as fp:
        json.dump(d, fp, indent=2)


def main():
    print(f"mpmath dps = {mp.mp.dps}", flush=True)
    print(f"delta = {[float(d) for d in DELTA]}", flush=True)
    print(f"lambda = {[float(l) for l in LAM]}", flush=True)
    print(f"T_cutoff = {float(T_CUTOFF)}", flush=True)

    ckpt = load_ckpt()
    t0 = time.time()
    C = [[None, None, None] for _ in range(3)]
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for (i, j) in pairs:
        key = f"C_{i+1}{j+1}"
        if key in ckpt:
            v = mp.mpf(ckpt[key])
            print(f"{key} (cached) = {mp.nstr(v, 12)}", flush=True)
        else:
            ts = time.time()
            v = compute_Cij(i, j, T_CUTOFF)
            print(f"{key} = {mp.nstr(v, 12)}  ({time.time()-ts:.1f} s)", flush=True)
            ckpt[key] = mp.nstr(v, 30)
            save_ckpt(ckpt)
        C[i][j] = v
        if i != j:
            C[j][i] = v
        gc.collect()

    K2 = mp.mpf(0)
    for i in range(3):
        for j in range(3):
            K2 += LAM[i] * LAM[j] * C[i][j]

    sum_lam_over_delta = sum(LAM[i] / DELTA[i] for i in range(3))
    tail_bound = (mp.mpf(8) / mp.pi**2) * sum_lam_over_delta**2 / T_CUTOFF

    print(f"K_2 (mpmath, bulk on [-T,T])   = {mp.nstr(K2, 14)}", flush=True)
    print(f"Tail bound (|xi|>T)            <= {mp.nstr(tail_bound, 8)}", flush=True)
    print(f"K_2 lower (bulk only)          = {mp.nstr(K2, 14)}", flush=True)
    print(f"K_2 upper (bulk + tail)        = {mp.nstr(K2 + tail_bound, 14)}", flush=True)

    v4_lower = mp.mpf("4.788823421259034")
    v4_upper = mp.mpf("4.789630363787504")
    print(f"v4 K_2_lower = {mp.nstr(v4_lower, 14)}", flush=True)
    print(f"v4 K_2_upper = {mp.nstr(v4_upper, 14)}", flush=True)

    bulk_inside = (K2 >= v4_lower) and (K2 <= v4_upper)
    overlap = (K2 + tail_bound >= v4_lower) and (K2 <= v4_upper)
    print(f"mpmath K_2_bulk in [v4_lower, v4_upper]?  {bulk_inside}", flush=True)
    print(f"mpmath [K_2, K_2+tail] overlaps [v4_lower, v4_upper]?  {overlap}", flush=True)

    out = {
        "dps": mp.mp.dps,
        "delta": [str(d) for d in DELTA],
        "lambda": [str(l) for l in LAM],
        "T_cutoff": str(T_CUTOFF),
        "C_11": mp.nstr(C[0][0], 30),
        "C_22": mp.nstr(C[1][1], 30),
        "C_33": mp.nstr(C[2][2], 30),
        "C_12": mp.nstr(C[0][1], 30),
        "C_13": mp.nstr(C[0][2], 30),
        "C_23": mp.nstr(C[1][2], 30),
        "K_2_mpmath_bulk": mp.nstr(K2, 30),
        "tail_bound": mp.nstr(tail_bound, 12),
        "K_2_mpmath_plus_tail": mp.nstr(K2 + tail_bound, 30),
        "v4_K_2_lower": mp.nstr(v4_lower, 30),
        "v4_K_2_upper": mp.nstr(v4_upper, 30),
        "K_2_bulk_inside_v4_enclosure": bool(bulk_inside),
        "K_2_overlap_with_v4_enclosure": bool(overlap),
        "elapsed_s": time.time() - t0,
    }
    with open(OUT_PATH, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"Elapsed: {time.time()-t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
