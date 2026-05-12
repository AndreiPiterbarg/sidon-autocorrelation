"""
W5: Bochner positivity check for the 3-scale K_hat at QP frequencies.

K_hat(xi) = 0.85*J_0(pi*0.138*xi)^2 + 0.10*J_0(pi*0.055*xi)^2 + 0.05*J_0(pi*0.025*xi)^2
u = 0.638

Tasks:
1. Compute w_j = K_hat(j/u) for j=1..200; find min, j*.
2. Pure-arcsine comparison: J_0(pi*0.138*j/u)^2.
3. Verify w_j > 1e-4 (well-conditioned).
4. Bochner over integers j=1..500: K_hat(j) >= 0.
5. Pass: min_j w_j (j=1..200) > 1e-3 AND all K_hat(j) >= 0 (j=1..500).
"""
import mpmath as mp
import json

mp.mp.dps = 30

delta1 = mp.mpf("0.138")
delta2 = mp.mpf("0.055")
delta3 = mp.mpf("0.025")
c1 = mp.mpf("0.85")
c2 = mp.mpf("0.10")
c3 = mp.mpf("0.05")
u = mp.mpf("0.638")
pi = mp.pi

def K_hat(xi):
    a = c1 * mp.besselj(0, pi * delta1 * xi) ** 2
    b = c2 * mp.besselj(0, pi * delta2 * xi) ** 2
    c = c3 * mp.besselj(0, pi * delta3 * xi) ** 2
    return a + b + c

def arcsine_only(xi):
    # Single-scale (pure arcsine-ish) component
    return mp.besselj(0, pi * delta1 * xi) ** 2

# --- Step 1: w_j for j=1..200 at QP frequencies j/u ---
qp_results = []
min_w = None
min_j = None
for j in range(1, 201):
    xi = mp.mpf(j) / u
    w = K_hat(xi)
    a_only = arcsine_only(xi)
    qp_results.append({
        "j": j,
        "xi": float(xi),
        "w_j": float(w),
        "arcsine_only": float(a_only),
    })
    if min_w is None or w < min_w:
        min_w = w
        min_j = j

# --- Step 2: arcsine min for comparison ---
arc_results = [(r["j"], r["arcsine_only"]) for r in qp_results]
arc_min_val = min(arc_results, key=lambda t: t[1])
arc_min_j = arc_min_val[0]
arc_min_w = arc_min_val[1]

# --- Step 3: which j have w_j < 1e-3 ---
small_w = [(r["j"], r["w_j"]) for r in qp_results if r["w_j"] < 1e-3]
sub_1e4 = [(r["j"], r["w_j"]) for r in qp_results if r["w_j"] < 1e-4]

# --- Step 4: Bochner over integers j=1..500 ---
bochner_negatives = []
bochner_near_zero = []
bochner_min = None
bochner_min_j = None
for j in range(1, 501):
    val = K_hat(mp.mpf(j))
    if val < 0:
        bochner_negatives.append((j, float(val)))
    if val < mp.mpf("1e-6"):
        bochner_near_zero.append((j, float(val)))
    if bochner_min is None or val < bochner_min:
        bochner_min = val
        bochner_min_j = j

# --- Find top 5 smallest w_j ---
sorted_w = sorted(qp_results, key=lambda r: r["w_j"])[:10]

# --- Report ---
report = {
    "min_w_j_for_j_1_200": float(min_w),
    "argmin_j": min_j,
    "argmin_xi": float(mp.mpf(min_j) / u),
    "arcsine_only_min": float(arc_min_w),
    "arcsine_only_argmin_j": arc_min_j,
    "lift_factor": float(min_w / arc_min_w) if arc_min_w > 0 else None,
    "num_w_below_1e-3": len(small_w),
    "num_w_below_1e-4": len(sub_1e4),
    "smallest_w_below_1e-3": small_w[:20],
    "smallest_w_below_1e-4": sub_1e4[:20],
    "bochner_min_j_1_500": float(bochner_min),
    "bochner_argmin_j": bochner_min_j,
    "bochner_negatives": bochner_negatives,
    "bochner_near_zero_lt_1e-6": bochner_near_zero,
    "top10_smallest_qp": [(r["j"], r["w_j"]) for r in sorted_w],
}

# Pass criterion
pass_qp = (min_w > mp.mpf("1e-3"))
pass_bochner = (len(bochner_negatives) == 0)

print("=" * 70)
print("W5 BOCHNER CHECK")
print("=" * 70)
print(f"\nQP min w_j (j=1..200): {float(min_w):.6e} at j={min_j} (xi={float(mp.mpf(min_j)/u):.4f})")
print(f"Arcsine-only min (same j range): {float(arc_min_w):.6e} at j={arc_min_j}")
if arc_min_w > 0:
    print(f"Lift factor: {float(min_w/arc_min_w):.3f}x")
print(f"\nNum w_j < 1e-3: {len(small_w)}")
print(f"Num w_j < 1e-4: {len(sub_1e4)}")
print(f"\nTop 10 smallest QP w_j:")
for j_, w_ in report["top10_smallest_qp"]:
    print(f"  j={j_:3d}  w={w_:.6e}")
print(f"\nBochner (j=1..500): min = {float(bochner_min):.6e} at j={bochner_min_j}")
print(f"Bochner negatives: {len(bochner_negatives)}")
print(f"Bochner near-zero (<1e-6): {len(bochner_near_zero)}")
if bochner_near_zero:
    for j_, v_ in bochner_near_zero[:10]:
        print(f"  j={j_:3d}  K_hat(j)={v_:.6e}")
print(f"\nPASS QP (min_w > 1e-3):     {pass_qp}")
print(f"PASS Bochner (all >= 0):     {pass_bochner}")
print(f"\nOVERALL: {'CONFIRM' if (pass_qp and pass_bochner) else 'FLAG'}")

with open("_w5_bochner_results.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nWrote _w5_bochner_results.json")
