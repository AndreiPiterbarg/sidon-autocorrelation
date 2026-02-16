"""Test value computation for branch-and-prune algorithm.

Computes max_{window} (1/(4*n*ell)) * sum_{k<=i+j<=k+ell-2} a_i*a_j
for batches of mass vectors.
"""
import numpy as np
import numba


@numba.njit(parallel=True, cache=True)
def _test_values_jit(batch_int, d, n_half, inv_m, early_stop=0.0):
    """Numba JIT: fused autoconvolution + window-max, parallelized over batch.

    Parameters
    ----------
    early_stop : float
        If > 0, stop checking windows once best exceeds this threshold.
        The returned value is a lower bound on the true max (sufficient for pruning).
    """
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    result = np.empty(B, dtype=np.float64)
    do_early = early_stop > 0.0
    scale = 4.0 * n_half * inv_m   # c -> a conversion factor (4n/m)
    inv_ell2 = 1.0 / (4.0 * n_half * 2)
    for b in numba.prange(B):
        # Quick check: if max element squared alone exceeds threshold
        # (from ell=2 diagonal), skip full autoconvolution
        if do_early:
            max_a = 0.0
            for i in range(d):
                ai = batch_int[b, i] * scale
                if ai > max_a:
                    max_a = ai
            if max_a * max_a * inv_ell2 > early_stop:
                result[b] = max_a * max_a * inv_ell2
                continue

        # Autoconvolution with symmetry: conv[k] = sum_{i+j=k} a_i*a_j
        conv = np.empty(conv_len, dtype=np.float64)
        if d == 4:
            a0 = batch_int[b, 0] * scale
            a1 = batch_int[b, 1] * scale
            a2 = batch_int[b, 2] * scale
            a3 = batch_int[b, 3] * scale
            conv[0] = a0 * a0
            conv[1] = 2.0 * a0 * a1
            conv[2] = a1 * a1 + 2.0 * a0 * a2
            conv[3] = 2.0 * (a0 * a3 + a1 * a2)
            conv[4] = a2 * a2 + 2.0 * a1 * a3
            conv[5] = 2.0 * a2 * a3
            conv[6] = a3 * a3
        elif d == 6:
            a0 = batch_int[b, 0] * scale
            a1 = batch_int[b, 1] * scale
            a2 = batch_int[b, 2] * scale
            a3 = batch_int[b, 3] * scale
            a4 = batch_int[b, 4] * scale
            a5 = batch_int[b, 5] * scale
            conv[0] = a0 * a0
            conv[1] = 2.0 * a0 * a1
            conv[2] = 2.0 * a0 * a2 + a1 * a1
            conv[3] = 2.0 * (a0 * a3 + a1 * a2)
            conv[4] = 2.0 * (a0 * a4 + a1 * a3) + a2 * a2
            conv[5] = 2.0 * (a0 * a5 + a1 * a4 + a2 * a3)
            conv[6] = 2.0 * (a1 * a5 + a2 * a4) + a3 * a3
            conv[7] = 2.0 * (a2 * a5 + a3 * a4)
            conv[8] = 2.0 * a3 * a5 + a4 * a4
            conv[9] = 2.0 * a4 * a5
            conv[10] = a5 * a5
        else:
            for k in range(conv_len):
                conv[k] = 0.0
            for i in range(d):
                ai = batch_int[b, i] * scale
                for j in range(d):
                    conv[i + j] += ai * batch_int[b, j] * scale
        # In-place prefix sums
        for k in range(1, conv_len):
            conv[k] += conv[k - 1]
        # Window max with early termination (large ell first —
        # pre-filtered configs tend to be spread out, favoring large window)
        best = 0.0
        done = False
        for ell in range(2, d + 1):
            if done:
                break
            n_cv = ell - 1
            inv_norm = 1.0 / (4.0 * n_half * ell)
            for s_lo in range(conv_len - n_cv + 1):
                s_hi = s_lo + n_cv - 1
                ws = conv[s_hi]
                if s_lo > 0:
                    ws -= conv[s_lo - 1]
                tv = ws * inv_norm
                if tv > best:
                    best = tv
                    if do_early and best > early_stop:
                        done = True
                        break
        result[b] = best
    return result


def compute_test_values_batch(batch_int, n_half, m, prune_target=0.0):
    """Compute max test value for a batch of integer mass vectors.

    Parameters
    ----------
    batch_int : (B, d) int32 array
        Integer mass coordinates (c_i = m * w_i, sum = m, S=m convention).
    n_half : int
        Paper's n.
    m : int
        Grid resolution.
    prune_target : float
        If > 0, early-stop per config once test value exceeds this threshold.

    Returns
    -------
    (B,) float64 array of test values (lower bounds if early-stopped).
    """
    _, d = batch_int.shape
    return _test_values_jit(batch_int, d, n_half, 1.0 / m, prune_target)


def compute_test_value_single(a, n_half):
    """Compute test value for a single mass vector (a-coordinates, float)."""
    a = np.asarray(a, dtype=np.float64)
    d = len(a)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += a[i] * a[j]

    cumconv = np.cumsum(conv)
    best = 0.0
    for ell in range(2, d + 1):
        n_conv_vals = ell - 1
        for s_lo in range(conv_len - n_conv_vals + 1):
            s_hi = s_lo + n_conv_vals - 1
            ws = cumconv[s_hi]
            if s_lo > 0:
                ws -= cumconv[s_lo - 1]
            tv = ws / (4.0 * n_half * ell)
            if tv > best:
                best = tv
    return best
