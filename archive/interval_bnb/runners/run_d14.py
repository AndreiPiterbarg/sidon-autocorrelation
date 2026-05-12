"""d=14 certification attempt.

Target: val(14) >= 1.2802 (record-breaking if achieved).
val(14) ~= 1.28396 so the slack is ~0.004 -- very tight.

Writes results to interval_bnb/run_d14.log and interval_bnb/tree_d14.json.
If the target is certified, emits interval_bnb/proof.md with a
Lean-formalisation-ready proof sketch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bnb import branch_and_bound  # noqa: E402


def _proof_md(d: int, target_q, stats: dict) -> str:
    return f"""# Rigorous lower bound on val({d}) via interval branch-and-bound

**Claim.** val({d}) >= {target_q} exactly, where

    val(d) := min_{{mu in Delta_d}} max_{{W in W_d}} mu^T M_W mu

and W_d, M_W are defined in `lasserre/core.py::build_window_matrices`.

**Consequence.** C_{{1a}} >= val({d}) >= {target_q}. The prior record
is C_{{1a}} >= 1.2802 (Cloninger-Steinerberger cascade).

## Proof structure

### Part 1: half-simplex orbit cover

Let sigma : Delta_d -> Delta_d be the involution
sigma(mu)_i := mu_{{d-1-i}}.  The objective f(mu) := max_W mu^T M_W mu
is sigma-invariant (the window family W_d is sigma-invariant as a set).

Define
    H_d := {{ mu in Delta_d : mu_0 <= mu_{{d-1}} }}.

*Lemma.* H_d meets every sigma-orbit.
*Proof.* For mu in Delta_d, either (i) mu_0 <= mu_{{d-1}}, so mu in H_d;
or (ii) mu_0 > mu_{{d-1}}, in which case sigma(mu)_0 = mu_{{d-1}} <
mu_0 = sigma(mu)_{{d-1}}, so sigma(mu) in H_d. Because f is constant
on orbits, min_{{mu in H_d}} f = min_{{mu in Delta_d}} f = val(d).

*Remark.* We deliberately use the SINGLE-PAIR cut {{mu_0 <= mu_{{d-1}}}}
rather than the stronger "all pairs ordered" cut
{{mu_i <= mu_{{d-1-i}} for all i}}, because the latter is NOT an orbit
cover in general (counter-example at d=4:
mu = (0.3, 0.1, 0.4, 0.2) has neither mu nor sigma(mu) respecting
both pair inequalities). The stronger cut would require the
symmetric-minimiser theorem for the discrete problem, which depends
on a non-self-contained fixed-point argument on a non-convex set.

### Part 2: Exhaustive exact-arithmetic box cover

We partition H_d into {stats.get('leaves_certified', '?')} closed dyadic
boxes {{B_k}}; every endpoint is a dyadic rational stored as an integer
with denominator 2**60, so all splits are EXACT (no floating-point
rounding) and box intersections with Delta_d are decided by exact
integer sums.

For each box B_k the rigor gate asserts, in `fractions.Fraction`, that
some window W_k satisfies

    min_{{mu in B_k cap Delta_d}}  mu^T M_{{W_k}} mu  >=  {target_q}.

The certifying bound is one of the following (we try them in the
order listed and accept the first that clears {target_q}):

1. **Natural interval:** for every (i, j) in the support of M_{{W_k}},
   mu_i mu_j >= lo_i lo_j (since mu >= lo >= 0). Summing and scaling
   gives  scale_{{W_k}} * sum_{{(i,j) in supp}} lo_i lo_j.

2. **Autoconvolution (simplex):** since (sum mu)^2 = 1 on the simplex,
   sum_{{(i,j) in supp}} mu_i mu_j = 1 - sum_{{(i,j) NOT in supp}}
   mu_i mu_j >= 1 - sum_{{(i,j) NOT in supp}} hi_i hi_j.

3. **SW McCormick LP:** (mu_i - lo_i)(mu_j - lo_j) >= 0 gives
   mu_i mu_j >= lo_j mu_i + lo_i mu_j - lo_i lo_j. Summing yields a
   LINEAR lower bound g_SW^T mu + c0_SW; the minimum of this over
   Delta_d intersect B_k is found by greedy sort (closed-form LP).

4. **NE McCormick LP:** (mu_i - hi_i)(mu_j - hi_j) >= 0 (both factors
   are non-positive in B_k) gives  mu_i mu_j >= hi_j mu_i + hi_i mu_j
   - hi_i hi_j.  Summing and minimising likewise.

All four formulas are valid pointwise lower bounds on mu^T M_W mu over
B_k cap Delta_d. Each Fraction computation is done in exact arithmetic
(Python's fractions.Fraction with integer arithmetic on a shared
denominator 2**60 for box endpoints).

### Part 3: Conclusion

By Part 1, val({d}) = min_{{mu in H_d}} f(mu). By Part 2, for every
box B_k in our cover of H_d, f(mu) >= {target_q} for all mu in
B_k cap Delta_d. Hence val({d}) >= {target_q}. Because C_{{1a}} >=
val(d) for every d, we conclude  C_{{1a}} >= {target_q}.  QED.

## Rigor audit

- **Box endpoints** are dyadic rationals; splits (lo+hi)/2 preserve
  dyadicity, and integer representation at denominator 2**60 avoids
  any float rounding error up to tree depth 60.
- **intersects_simplex** is decided by integer sums, exact.
- **Bounds on mu^T M_W mu** (natural, autoconv, SW McCormick, NE
  McCormick) all reduce to sum-of-products over pair indices and
  linear programs over the bounded simplex; each is re-verified in
  exact Fraction arithmetic before the leaf is closed.
- The **float64 fast path** is only a PRUNING heuristic; leaves
  survive only if the exact Fraction replay clears the target.
- **Half-simplex** reduction uses the single-pair cut which is a
  provable orbit cover without any convexity / Kakutani assumption.

## Tree statistics

```json
{json.dumps(stats, indent=2, default=str)}
```
"""


def main():
    ap = argparse.ArgumentParser(description="interval BnB d=14 attempt")
    ap.add_argument("--target", type=str, default="1.2802")
    ap.add_argument("--max_nodes", type=int, default=100_000_000)
    ap.add_argument("--time_budget_s", type=float, default=7 * 86400.0)
    ap.add_argument("--method", choices=["natural", "autoconv", "mccormick", "combined"],
                    default="combined")
    ap.add_argument("--log_out", type=str,
                    default=os.path.join(_HERE, "run_d14.log"))
    ap.add_argument("--stats_out", type=str,
                    default=os.path.join(_HERE, "tree_d14.json"))
    ap.add_argument("--proof_out", type=str,
                    default=os.path.join(_HERE, "proof.md"))
    args = ap.parse_args()

    # Target may be a decimal string like "1.2802" or a rational
    # "6401/5000" -- pass to branch_and_bound as a Fraction so rigor
    # compares against the EXACT user intent, not a float rounding.
    from fractions import Fraction as _F
    if "/" in args.target:
        target = _F(args.target)
    else:
        target = _F(args.target)  # Fraction parses decimal strings exactly
    print(f"[run_d14] certifying val(14) >= {target}  method={args.method}")
    t0 = time.time()
    # Pass the Fraction directly; branch_and_bound stores it as target_q
    # without going through float.
    res = branch_and_bound(
        d=14, target_c=target, verbose=True,
        log_every=100_000, time_budget_s=args.time_budget_s,
        max_nodes=args.max_nodes, method=args.method,
    )
    elapsed = time.time() - t0
    print(f"[run_d14] SUCCESS={res.success}  elapsed={elapsed:.1f}s")

    stats = res.stats.to_dict()
    stats["d"] = 14
    stats["target_c"] = target
    stats["target_rational"] = str(res.target_q)
    stats["success"] = res.success
    with open(args.stats_out, "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"[run_d14] stats written to {args.stats_out}")

    if res.success:
        with open(args.proof_out, "w") as fh:
            fh.write(_proof_md(14, res.target_q, stats))
        print(f"[run_d14] proof sketch written to {args.proof_out}")
        print(f"[run_d14] val(14) >= {res.target_q}  (rational)")
    else:
        print(f"[run_d14] target NOT certified. Last worst_lb_seen = "
              f"{stats.get('worst_lb_seen')}")


if __name__ == "__main__":
    main()
