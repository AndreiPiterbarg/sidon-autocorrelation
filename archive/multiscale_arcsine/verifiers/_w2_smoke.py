"""Smoke test: compute C_11 with a small T and check timing/value."""
import time
import mpmath as mp

mp.mp.dps = 50
DELTA = [mp.mpf(138) / 1000, mp.mpf(55) / 1000, mp.mpf(25) / 1000]


def integrand(i, j, xi):
    a = mp.pi * DELTA[i] * xi
    b = mp.pi * DELTA[j] * xi
    return mp.besselj(0, a) ** 2 * mp.besselj(0, b) ** 2


def compute_Cij_short(i, j, T):
    a_max = max(DELTA[i], DELTA[j])
    val = mp.mpf(0)
    X_inner = mp.mpf(50)
    val += mp.quad(lambda xi: integrand(i, j, xi), [0, X_inner])
    step = 1 / a_max
    k_start = int(mp.ceil(X_inner * a_max))
    k_end = int(mp.floor(T * a_max))
    CHUNK = 500
    prev = X_inner
    for k in range(k_start, k_end + 1, CHUNK):
        nxt = mp.mpf(min(k + CHUNK, k_end)) / a_max
        if nxt <= prev:
            continue
        val += mp.quad(lambda xi: integrand(i, j, xi), [prev, nxt])
        prev = nxt
    if prev < T:
        val += mp.quad(lambda xi: integrand(i, j, xi), [prev, T])
    return 2 * val


for T in [mp.mpf(1000), mp.mpf(10000)]:
    t = time.time()
    v = compute_Cij_short(0, 0, T)
    print(f"T={float(T)}  C_11 = {mp.nstr(v, 14)}  ({time.time()-t:.1f} s)")
    # ref: 0.574695 / 0.138 = 4.16446...
print("ref C_11 ~ 4.164456")
