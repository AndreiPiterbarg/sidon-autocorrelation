"""Numerical scan of c(f) := ||f*f||_2^2 / ||f*f||_inf  for admissible f.

Conventions:
  - f >= 0 supported on [-1/4, 1/4], int f = 1.
  - g = f * f, supported on [-1/2, 1/2].
  - We work on a uniform grid on [-1/4, 1/4] with N samples and step dx=1/(2(N-1)).
  - Convolution g = f*f is computed via FFT; ||f*f||_2^2 and ||f*f||_inf likewise.
  - ||f*f||_1 = (int f)^2 = 1 by construction.

Hyp_R(c, M_max): if for every admissible f with M:=||f*f||_inf <= M_max
    we have ||f*f||_2^2 <= c * M, then C_{1a} >= 1.378 (with c = log16/pi ~ 0.8825).

For each parametric family we compute (M, c) and tabulate:
  - Symmetric Gaussians, double Gaussians, triangles
  - Asymmetric: single offset bumps, skewed Gaussians, two-bump configs
  - MV-class (cosine sum -> normalized pdf)
  - Random parametric mixtures (5-10 free params)

The crucial question: for f's with M near 1.28 (the C_{1a} regime),
is c(f) reliably < 1?  Is c(f) >= 1 for asymmetric f with M < 1.378?
"""
from __future__ import annotations
import numpy as np
from numpy.fft import rfft, irfft

# Grid: f sampled on [-1/4, 1/4]; convolution support [-1/2, 1/2].
N = 4097                          # samples on [-1/4, 1/4] (odd: index 0 is left endpt)
dx = 0.5 / (N - 1)                 # step
xs = -0.25 + dx * np.arange(N)
# Pad for circular convolution to act as linear; output length 2N - 1.
PAD = 2 * N - 1
# pad to next power-of-2 for FFT speed
PAD2 = 1 << (PAD - 1).bit_length()


def normalize(f):
    s = f.sum() * dx
    if s <= 0:
        raise ValueError("non-positive integral")
    return f / s


def autoconv(f):
    """g = f * f as continuous-function values on a uniform grid of length 2N-1.

    Output grid step = dx, so g[i] approximates (f*f)(t) at t = -0.5 + i*dx.
    The standard rectangle rule: (f*f)(t) ~= sum_k f(s_k) f(t - s_k) * dx.
    """
    F = rfft(f, n=PAD2)
    G = F * F
    g = irfft(G, n=PAD2)[:PAD] * dx
    return g


def c_param(f):
    """Return (M, c, ff_l22) with c = ||f*f||_2^2 / ||f*f||_inf  (||f*f||_1 = 1)."""
    f = normalize(np.maximum(f, 0.0).astype(np.float64))
    g = autoconv(f)
    M = g.max()
    # int g dt = (int f)^2 = 1; ||g||_2^2 = int g^2 dt
    ff_l22 = (g * g).sum() * dx
    return M, ff_l22 / M, ff_l22


# -------------------------------------------------------------------------
# Helpers for parametric families.
# -------------------------------------------------------------------------
def gaussian(mu, sigma):
    return np.exp(-0.5 * ((xs - mu) / sigma) ** 2)


def triangle(c, w):
    """Triangle centered at c with half-width w (=> support [c-w, c+w])."""
    return np.maximum(1.0 - np.abs((xs - c) / w), 0.0)


def indicator(a, b):
    return ((xs >= a) & (xs <= b)).astype(np.float64)


def smooth_bump(c, w, exponent=2):
    """Symmetric C^infty bump centered at c with support [c-w, c+w]."""
    u = (xs - c) / w
    out = np.zeros_like(xs)
    mask = np.abs(u) < 1
    out[mask] = np.exp(-1.0 / (1 - u[mask] ** 2) ** exponent)
    return out


# -------------------------------------------------------------------------
# Family 1: symmetric Gaussian (sigma sweep).
# -------------------------------------------------------------------------
def fam_sym_gaussian():
    rows = []
    for sigma in np.linspace(0.02, 0.13, 40):
        f = gaussian(0.0, sigma)
        # Gaussian mass outside [-1/4, 1/4] is truncated by support; that's OK.
        try:
            M, c, _ = c_param(f)
        except ValueError:
            continue
        rows.append(("sym_gauss", {"sigma": sigma}, M, c))
    return rows


# -------------------------------------------------------------------------
# Family 2: symmetric double Gaussian (two bumps centered at +-mu, equal sigma).
# -------------------------------------------------------------------------
def fam_double_gauss():
    rows = []
    for mu in np.linspace(0.02, 0.22, 22):
        for sigma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
            f = gaussian(mu, sigma) + gaussian(-mu, sigma)
            try:
                M, c, _ = c_param(f)
            except ValueError:
                continue
            rows.append(("dbl_gauss", {"mu": mu, "sigma": sigma}, M, c))
    return rows


# -------------------------------------------------------------------------
# Family 3: symmetric triangle.
# -------------------------------------------------------------------------
def fam_triangle():
    rows = []
    for w in np.linspace(0.05, 0.25, 21):
        f = triangle(0.0, w)
        try:
            M, c, _ = c_param(f)
        except ValueError:
            continue
        rows.append(("triangle", {"w": w}, M, c))
    return rows


# -------------------------------------------------------------------------
# Family 4: indicator function (uniform on a sub-interval).
# -------------------------------------------------------------------------
def fam_uniform():
    rows = []
    for w in np.linspace(0.05, 0.25, 21):
        f = indicator(-w, w)
        try:
            M, c, _ = c_param(f)
        except ValueError:
            continue
        rows.append(("uniform", {"halfwidth": w}, M, c))
    return rows


# -------------------------------------------------------------------------
# Family 5: ASYMMETRIC -- offset Gaussian.
# -------------------------------------------------------------------------
def fam_offset_gauss():
    rows = []
    for mu in np.linspace(-0.20, 0.20, 21):
        for sigma in [0.02, 0.04, 0.06, 0.08, 0.10]:
            f = gaussian(mu, sigma)
            try:
                M, c, _ = c_param(f)
            except ValueError:
                continue
            rows.append(("offset_gauss", {"mu": mu, "sigma": sigma}, M, c))
    return rows


# -------------------------------------------------------------------------
# Family 6: ASYMMETRIC -- two unequal bumps at different centers.
# -------------------------------------------------------------------------
def fam_two_unequal():
    rows = []
    for mu1 in np.linspace(-0.22, -0.05, 6):
        for mu2 in np.linspace(0.05, 0.22, 6):
            for sigma in [0.02, 0.04, 0.06]:
                for h1 in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                    f = h1 * gaussian(mu1, sigma) + 1.0 * gaussian(mu2, sigma)
                    try:
                        M, c, _ = c_param(f)
                    except ValueError:
                        continue
                    rows.append(("two_unequal_g",
                                  {"mu1": mu1, "mu2": mu2, "sigma": sigma, "h1": h1},
                                  M, c))
    return rows


# -------------------------------------------------------------------------
# Family 7: ASYMMETRIC -- skewed Gaussian (different left/right sigma).
# -------------------------------------------------------------------------
def fam_skewed():
    rows = []
    for mu in np.linspace(-0.15, 0.15, 11):
        for s_l in [0.03, 0.05, 0.08]:
            for s_r in [0.03, 0.05, 0.08, 0.12]:
                f = np.where(xs < mu,
                              np.exp(-0.5 * ((xs - mu) / s_l) ** 2),
                              np.exp(-0.5 * ((xs - mu) / s_r) ** 2))
                try:
                    M, c, _ = c_param(f)
                except ValueError:
                    continue
                rows.append(("skewed",
                              {"mu": mu, "s_l": s_l, "s_r": s_r}, M, c))
    return rows


# -------------------------------------------------------------------------
# Family 8: MV-class near-extremizer.
# -------------------------------------------------------------------------
MV_COEFFS_119 = [
    +2.16620392, -1.87775750, +1.05828868, -0.729790538,
    +0.428008515, +0.217832838, -0.270415201, +0.0272834790,
    -0.191721888, +0.0551862060, +0.321662512, -0.164478392,
    +0.0395478603, -0.205402785, -0.0133758316, +0.231873221,
    -0.0437967118, +0.0612456374, -0.157361919, -0.0778036253,
    +0.138714392, -0.000145201483, +0.0916539824, -0.0834020840,
    -0.101919986, +0.0594915025, -0.0119336618, +0.102155366,
    -0.0145929982, -0.0795205457, +0.00559733152, -0.0358987179,
    +0.0716132260, +0.0415425065, -0.0489180454, +0.00165425755,
    -0.0648251747, +0.0345951253, +0.0532122058, -0.0128435276,
    +0.0148814403, -0.0649404547, -0.00601344770, +0.0433784473,
    -0.000253362778, +0.0381674519, -0.0483816002, -0.0253878079,
    +0.0196933442, -0.00304861682, +0.0479203471, -0.0200930265,
    -0.0273895519, +0.00330183589, -0.0167380508, +0.0423917582,
    +0.00364690190, -0.0179916104, +0.0000731661649, -0.0299875575,
    +0.0271842526, +0.0141806855, -0.00601781076, +0.00586806100,
    -0.0332350597, +0.00923347466, +0.0147071722, -0.000742858080,
    +0.0163414270, -0.0287265671, -0.00164287280, +0.00802601605,
    -0.000762613027, +0.0218735533, -0.0178816282, -0.00658341101,
    +0.00267706547, -0.00625261247, +0.0224942824, -0.00810756022,
    -0.00568160823, +0.0000701871209, -0.0115294332, +0.0183608944,
    -0.00120567880, -0.00313147456, +0.00139083675, -0.0149312478,
    +0.0132106694, +0.00173474188, -0.000853469045, +0.00403211203,
    -0.0155352991, +0.00874711543, +0.00193998895, -0.0000271357322,
    +0.00613179585, -0.0141983972, +0.00584710551, +0.000922578333,
    -0.000216583469, +0.00707919829, -0.0118488582, +0.00439698322,
    -0.0000891346785, -0.000342086367, +0.00646355636, -0.00887555371,
    +0.00356799654, -0.000497335419, -0.000804560326, +0.00555076717,
    -0.00713560569, +0.00453679038, -0.00333261516, +0.00235463427,
    +0.000204023789, -0.00127746711, +0.000181247830,
]


def fam_mv():
    rows = []
    u = 0.638  # MV's u
    for u_var in [0.55, 0.60, 0.638, 0.65, 0.70, 0.75]:
        f = np.zeros_like(xs)
        for j, a in enumerate(MV_COEFFS_119, start=1):
            f += a * np.cos(2 * np.pi * j * xs / u_var)
        # Set negatives to zero (the cosine sum is positive on [-1/4,1/4]
        # for u=0.638; sweeping u may introduce small negatives near edges).
        f = np.maximum(f, 0.0)
        if f.sum() <= 0:
            continue
        try:
            M, c, _ = c_param(f)
        except ValueError:
            continue
        rows.append(("MV_119", {"u": u_var}, M, c))
    return rows


# -------------------------------------------------------------------------
# Family 9: random N-bump mixtures (5-10 free params).
# -------------------------------------------------------------------------
def fam_random_mixture(seed=0, n_bumps=4, n_trials=200):
    rng = np.random.default_rng(seed)
    rows = []
    for tr in range(n_trials):
        mus = rng.uniform(-0.22, 0.22, size=n_bumps)
        sigmas = rng.uniform(0.005, 0.06, size=n_bumps)
        weights = rng.uniform(0.1, 1.0, size=n_bumps)
        f = np.zeros_like(xs)
        for mu, s, w in zip(mus, sigmas, weights):
            f += w * gaussian(mu, s)
        try:
            M, c, _ = c_param(f)
        except ValueError:
            continue
        rows.append((f"rand_{n_bumps}b", {"trial": tr}, M, c))
    return rows


# -------------------------------------------------------------------------
# Family 10: BOYER-LI witness rescaled to [-1/4, 1/4] (asymmetric, M ~= 1.652).
# -------------------------------------------------------------------------
def fam_bl():
    """Rescale the Boyer-Li step function to [-1/4, 1/4]."""
    try:
        with open("delsarte_dual/restricted_holder/coeffBL.txt") as fh:
            txt = fh.read()
    except FileNotFoundError:
        return []
    import re
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    if len(nums) < 575:
        return []
    v = np.array(nums[:575], dtype=np.float64)  # f_0 on [0, 575]
    S = v.sum()
    # f_0 supported on [0, 575] integer cells. Translate to [-575/2, 575/2],
    # then rescale to [-1/4, 1/4] by dilation factor 1150.
    # On our xs grid, the interval [-1/4, 1/4] maps to indexed cells:
    #    index n (centered at  -1/4 + (n+1/2)/1150 ... no, let's just
    # discretize the rescaled step function on xs.
    f = np.zeros_like(xs)
    # cell width on xs = 1/1150 in original f_0 cell units;
    # the n-th cell of f_0 (n in [0, 575)) maps to xs in
    # [-1/4 + n/1150, -1/4 + (n+1)/1150]
    for n in range(575):
        a = -0.25 + n / 1150.0
        b = -0.25 + (n + 1) / 1150.0
        mask = (xs >= a) & (xs < b)
        f[mask] = v[n]
    try:
        M, c, _ = c_param(f)
    except ValueError:
        return []
    return [("BL_witness", {}, M, c)]


# -------------------------------------------------------------------------
# Run all and emit tables.
# -------------------------------------------------------------------------
def main():
    print(f"Grid N={N}, dx={dx:.4e}, support of f*f sampled on 2N-1={PAD} pts.")
    all_rows = []
    for fam_fn in [fam_sym_gaussian, fam_double_gauss, fam_triangle,
                   fam_uniform, fam_offset_gauss, fam_two_unequal,
                   fam_skewed, fam_mv, fam_bl]:
        all_rows.extend(fam_fn())
    for seed in range(5):
        for nb in (3, 4, 5, 7, 10):
            all_rows.extend(fam_random_mixture(seed=seed, n_bumps=nb,
                                                n_trials=80))
    print(f"Total samples: {len(all_rows)}")

    # ---------------------------------------------------------
    # Per-family summary (extremes of c, with M near 1.28 highlighted).
    # ---------------------------------------------------------
    from collections import defaultdict
    fams = defaultdict(list)
    for name, params, M, c in all_rows:
        fams[name].append((M, c, params))

    print("\n" + "=" * 96)
    print(f"{'family':<18s} {'n':>5s}  {'min M':>9s} {'max M':>9s} {'min c':>9s}"
          f" {'max c':>9s}  {'med c':>9s}  {'#(c<1)':>7s}  {'#(M<1.30)':>10s}")
    print("=" * 96)
    for name in sorted(fams):
        data = fams[name]
        Ms = np.array([m for m, _, _ in data])
        cs = np.array([c for _, c, _ in data])
        n_c_lt_1 = int(np.sum(cs < 1.0))
        n_M_low = int(np.sum(Ms < 1.30))
        print(f"{name:<18s} {len(data):>5d}  {Ms.min():>9.4f} {Ms.max():>9.4f}"
              f" {cs.min():>9.4f} {cs.max():>9.4f}  {np.median(cs):>9.4f}"
              f"  {n_c_lt_1:>7d}  {n_M_low:>10d}")
    print("=" * 96)

    # ---------------------------------------------------------
    # Show the f's that ACHIEVE small M (M < 1.30 close to C_{1a} regime).
    # These are the most relevant: are they all c < 1?
    # ---------------------------------------------------------
    print("\n--- Samples with M < 1.30 (near C_{1a} extremizer regime) ---")
    print(f"{'family':<18s}  {'M':>8s}  {'c':>8s}  {'ratio c/0.8825':>15s}  params")
    low_M = sorted([(M, c, name, p) for (name, p, M, c) in all_rows
                    if M < 1.30])
    for M, c, name, p in low_M[:50]:
        ratio = c / (np.log(16) / np.pi)
        print(f"{name:<18s}  {M:>8.4f}  {c:>8.4f}  {ratio:>15.4f}  {p}")
    print(f"... (total with M<1.30: {len(low_M)})")

    # ---------------------------------------------------------
    # Find the f's with smallest c overall (best Hyp_R candidates).
    # ---------------------------------------------------------
    print("\n--- Samples with smallest c (best for Hyp_R) ---")
    best_c = sorted(all_rows, key=lambda r: r[3])
    for name, p, M, c in best_c[:15]:
        print(f"  c={c:.4f}  M={M:.4f}  family={name}  params={p}")

    # ---------------------------------------------------------
    # Find the f's with LARGEST c at small M (Hyp_R counterexample candidates).
    # ---------------------------------------------------------
    print("\n--- ASYMMETRIC samples with LARGEST c at M < 1.40 ---")
    asym_names = {"offset_gauss", "two_unequal_g", "skewed",
                  "rand_3b", "rand_4b", "rand_5b", "rand_7b", "rand_10b",
                  "BL_witness"}
    asym_low_M = [r for r in all_rows
                  if r[0] in asym_names and r[2] < 1.40]
    asym_low_M.sort(key=lambda r: -r[3])
    for name, p, M, c in asym_low_M[:25]:
        print(f"  c={c:.4f}  M={M:.4f}  family={name}  params={p}")

    # ---------------------------------------------------------
    # Headline: count violations of Hyp_R for various c thresholds.
    # ---------------------------------------------------------
    print("\n--- Hyp_R(c, M_max=1.378) violation counts ---")
    M_cap = 1.378
    sub_M = [(M, c, name) for (name, p, M, c) in all_rows if M <= M_cap]
    print(f"Samples with M <= {M_cap}: {len(sub_M)}")
    for c_thresh in [0.85, 0.88, 0.8825, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]:
        viol = sum(1 for (M, c, _) in sub_M if c > c_thresh)
        print(f"  c > {c_thresh:.4f}: {viol:>4d} violations "
              f"({100.0*viol/max(1,len(sub_M)):.1f}%)")

    # ---------------------------------------------------------
    # Concrete extremes for the report.
    # ---------------------------------------------------------
    print("\n--- TABLE for report (selected representatives) ---")
    print(f"{'family':<18s}  {'M':>9s}  {'c=L2^2/Linf':>13s}  {'note'}")
    # min c
    name, p, M, c = min(all_rows, key=lambda r: r[3])
    print(f"{name:<18s}  {M:>9.4f}  {c:>13.4f}  smallest c globally  {p}")
    # max c overall
    name, p, M, c = max(all_rows, key=lambda r: r[3])
    print(f"{name:<18s}  {M:>9.4f}  {c:>13.4f}  largest c globally   {p}")
    # closest M to 1.28
    name, p, M, c = min(all_rows, key=lambda r: abs(r[2] - 1.28))
    print(f"{name:<18s}  {M:>9.4f}  {c:>13.4f}  M closest to 1.28    {p}")
    # closest M to 1.378
    name, p, M, c = min(all_rows, key=lambda r: abs(r[2] - 1.378))
    print(f"{name:<18s}  {M:>9.4f}  {c:>13.4f}  M closest to 1.378   {p}")


if __name__ == "__main__":
    main()
