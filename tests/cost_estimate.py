import numpy as np
from math import prod

print("=" * 80)
print("LEVEL 1 REFINEMENT COMPUTATIONAL COST ESTIMATE")
print("=" * 80)

# ============================================================================
# QUESTION 1: For the specific min config
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 1: Min config refinement factor")
print("=" * 80)

min_config_raw = [151, 42, 0, 115, 84, 208]
print(f"Min config (raw integers): {min_config_raw}")
print(f"Sum: {sum(min_config_raw)} (should be 600)")

# Each b_i generates (2*b_i + 1) children
refinement_factors = [2*b + 1 for b in min_config_raw]
print(f"Refinement factors (2*b_i + 1): {refinement_factors}")

min_config_prod = prod(refinement_factors)
print(f"\nProduct for min config: {min_config_prod:,}")
print(f"In scientific notation: {min_config_prod:.3e}")

# ============================================================================
# QUESTION 2: Typical survivor where S=600 spread across 6 bins
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 2: Typical survivor refinement factor")
print("=" * 80)

print("\nAssuming S=600 spread across d=6 bins.")
print("We'll compute expected prod(2*b_i + 1) for various distributions.\n")

# Uniform distribution
uniform_dist = [100] * 6
uniform_prod = prod([2*b + 1 for b in uniform_dist])
print(f"Uniform (100,100,100,100,100,100):")
print(f"  Refinement factors: {[2*b+1 for b in uniform_dist]}")
print(f"  Product: {uniform_prod:,} = {uniform_prod:.3e}")

# Concentrated distribution (skewed)
concentrated_dist = [300, 100, 100, 50, 25, 25]
concentrated_prod = prod([2*b + 1 for b in concentrated_dist])
print(f"\nConcentrated (300,100,100,50,25,25):")
print(f"  Sum: {sum(concentrated_dist)}")
print(f"  Refinement factors: {[2*b+1 for b in concentrated_dist]}")
print(f"  Product: {concentrated_prod:,} = {concentrated_prod:.3e}")

# Another distribution
dist3 = [200, 150, 100, 75, 50, 25]
dist3_prod = prod([2*b + 1 for b in dist3])
print(f"\nAnother distribution (200,150,100,75,50,25):")
print(f"  Sum: {sum(dist3)}")
print(f"  Refinement factors: {[2*b+1 for b in dist3]}")
print(f"  Product: {dist3_prod:,} = {dist3_prod:.3e}")

# Random realistic distributions
print("\n--- Random sampling of S=600 distributions ---")
np.random.seed(42)
random_prods = []
for trial in range(1000):
    # Generate 6 bins that sum to 600, using Dirichlet distribution
    alphas = np.ones(6)
    fracs = np.random.dirichlet(alphas)
    bins = np.round(fracs * 600).astype(int)
    # Ensure sum is exactly 600
    diff = 600 - bins.sum()
    bins[0] += diff

    p = prod([2*b + 1 for b in bins])
    random_prods.append(p)

random_prods = np.array(random_prods)
print(f"  Min prod: {random_prods.min():,} = {random_prods.min():.3e}")
print(f"  Max prod: {random_prods.max():,} = {random_prods.max():.3e}")
print(f"  Mean prod: {random_prods.mean():,.0f} = {random_prods.mean():.3e}")
print(f"  Median prod: {np.median(random_prods):,.0f} = {np.median(random_prods):.3e}")
print(f"  25th percentile: {np.percentile(random_prods, 25):,.0f}")
print(f"  75th percentile: {np.percentile(random_prods, 75):,.0f}")

# Use geometric mean as typical value (more appropriate for products)
geometric_mean = np.exp(np.mean(np.log(random_prods)))
print(f"\n  Geometric mean: {geometric_mean:,.0f} = {geometric_mean:.3e}")

typical_prod = geometric_mean

# ============================================================================
# QUESTION 3: Total refinement count for all 1.55B parents
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 3: Total refinement count for all 1.55B parents")
print("=" * 80)

num_parents = 1_553_783_953
print(f"Number of Level 0 survivors (parents): {num_parents:,}")
print(f"Typical refinement factor per parent: {typical_prod:,.0f}")

total_children = num_parents * typical_prod
print(f"\nTotal Level 1 children configs: {total_children:.3e}")
print(f"                               = {total_children:,.0f}")

# Also show if we used uniform distribution
total_children_uniform = num_parents * uniform_prod
print(f"\nIf all parents had uniform dist (100 per bin): {total_children_uniform:.3e}")
print(f"If min config applied to all parents: {num_parents * min_config_prod:.3e}")

# ============================================================================
# QUESTION 4: Runtime at 91B configs/sec
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 4: Runtime at 91B configs/sec")
print("=" * 80)

kernel_rate = 91e9  # configs/sec
print(f"Kernel throughput: {kernel_rate:.3e} configs/sec")
print(f"                 = {kernel_rate/1e9:.1f} billion configs/sec\n")

runtime_seconds = total_children / kernel_rate
runtime_hours = runtime_seconds / 3600
runtime_days = runtime_hours / 24

print(f"Total runtime:")
print(f"  {runtime_seconds:,.0f} seconds")
print(f"  {runtime_hours:,.1f} hours")
print(f"  {runtime_days:,.2f} days")

# Cost estimate (A100 @ $1.49/hr)
hourly_cost = 1.49
total_cost = runtime_hours * hourly_cost
print(f"\nCost at $1.49/hr A100:")
print(f"  ${total_cost:,.2f}")

# ============================================================================
# QUESTION 5: Disk space needed for survivors
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 5: Disk space for Level 1 survivors")
print("=" * 80)

print(f"Assuming d=12, 48 bytes per child config")
print(f"Worst case: all {total_children:.3e} children are survivors\n")

bytes_per_config = 48
total_bytes_worst = total_children * bytes_per_config
total_gb_worst = total_bytes_worst / (1024**3)
total_tb_worst = total_gb_worst / 1024

print(f"Worst case (all children survive):")
print(f"  {total_bytes_worst:.3e} bytes")
print(f"  {total_gb_worst:,.1f} GB")
print(f"  {total_tb_worst:,.2f} TB")

# More realistic: assume 10% of children survive (pruning during phase 1)
survival_rate = 0.10
survivors_children = total_children * survival_rate
bytes_realistic = survivors_children * bytes_per_config
gb_realistic = bytes_realistic / (1024**3)

print(f"\nRealistic case (10% survival rate):")
print(f"  {survivors_children:.3e} children survive")
print(f"  {bytes_realistic:.3e} bytes")
print(f"  {gb_realistic:,.1f} GB")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

summary_data = [
    ("Min config prod(2*b_i+1)", f"{min_config_prod:,}", f"{min_config_prod:.3e}"),
    ("Typical prod(2*b_i+1)", f"{typical_prod:,.0f}", f"{typical_prod:.3e}"),
    ("Uniform prod(2*b_i+1)", f"{uniform_prod:,}", f"{uniform_prod:.3e}"),
    ("", "", ""),
    ("Level 0 parents", f"{num_parents:,}", ""),
    ("Total Level 1 children", f"{total_children:.3e}", f"({total_children/1e12:.1f}T)"),
    ("", "", ""),
    ("Kernel rate", f"{kernel_rate/1e9:.0f}B configs/sec", ""),
    ("Runtime (hours)", f"{runtime_hours:,.1f}", f"({runtime_days:.2f} days)"),
    ("Cost at $1.49/hr", f"${total_cost:,.2f}", ""),
    ("", "", ""),
    ("Disk (worst case, 100%)", f"{total_gb_worst:,.0f} GB", f"({total_tb_worst:.2f} TB)"),
    ("Disk (realistic, 10%)", f"{gb_realistic:,.0f} GB", ""),
]

print(f"\n{'Metric':<40} {'Value':<25} {'Note':<20}")
print("-" * 85)
for metric, value, note in summary_data:
    if metric:
        print(f"{metric:<40} {value:>25} {note:<20}")
    else:
        print()

print("\n" + "=" * 80)
