"""Generate a markdown report from a list of RunRecord results."""
from __future__ import annotations
from typing import List, Dict, Optional
from collections import defaultdict
import math

from lasserre.polya_lp.runner import RunRecord, VAL_D_KNOWN
from lasserre.polya_lp.fit import fit_convergence, project_R_for_target


def render_table(records: List[RunRecord]) -> str:
    """Group by d, render an R-vs-alpha table per d."""
    by_d: Dict[int, List[RunRecord]] = defaultdict(list)
    for r in records:
        by_d[r.d].append(r)

    lines: List[str] = []
    for d in sorted(by_d.keys()):
        runs = sorted(by_d[d], key=lambda r: r.R)
        target = VAL_D_KNOWN.get(d, "?")
        lines.append(f"\n## d = {d}  (val(d) ≈ {target})\n")
        lines.append("| R | alpha | gap | n_vars | n_eq | nnz(A) | build_s | solve_s |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in runs:
            a = f"{r.alpha:.6f}" if r.alpha is not None else "-"
            g = f"{r.gap_to_known:.6f}" if r.gap_to_known is not None else "-"
            lines.append(
                f"| {r.R} | {a} | {g} | {r.n_vars:,} | {r.n_eq:,} | "
                f"{r.n_nonzero_A:,} | {r.build_wall_s:.2f} | {r.solve_wall_s:.2f} |"
            )
    return "\n".join(lines)


def render_convergence_fits(records: List[RunRecord], target: float = 1.281) -> str:
    """For each d, fit gap = C/R^a and project the R needed to certify
    alpha >= target."""
    by_d: Dict[int, List[RunRecord]] = defaultdict(list)
    for r in records:
        if r.alpha is None or r.val_d_known is None:
            continue
        by_d[r.d].append(r)

    lines = ["\n## Convergence-rate fits and projections\n",
             f"Target alpha = {target} (i.e. need val(d) >= {target}).\n",
             "| d | val(d) | n_R | fit a | fit C | gap@max R | R-needed for target |",
             "|---|---|---|---|---|---|---|"]
    for d in sorted(by_d.keys()):
        runs = sorted(by_d[d], key=lambda r: r.R)
        if len(runs) < 3:
            continue
        R_arr = [r.R for r in runs]
        a_arr = [r.alpha for r in runs]
        val_d = runs[0].val_d_known
        fit = fit_convergence(R_arr, a_arr, val_d, skip_first_n=1)
        if fit is None:
            continue
        gap_at_max = val_d - max(a_arr)
        if val_d < target:
            R_needed_str = f"impossible (val({d})={val_d}<{target})"
        else:
            R_needed = project_R_for_target(fit, val_d, target)
            R_needed_str = f"{R_needed:.1f}" if math.isfinite(R_needed) else "inf"
        lines.append(
            f"| {d} | {val_d} | {len(runs)} | {fit.a:.3f} | {fit.C:.4f} | "
            f"{gap_at_max:.4f} | {R_needed_str} |"
        )
    return "\n".join(lines)


def render_recommendation(records: List[RunRecord], target: float = 1.281) -> str:
    """Compare LP times vs estimated SDP times and write a verdict."""
    by_d: Dict[int, List[RunRecord]] = defaultdict(list)
    for r in records:
        if r.alpha is None:
            continue
        by_d[r.d].append(r)

    out = ["\n## Verdict\n"]

    # Find best alpha per d
    best_per_d = {}
    for d, runs in by_d.items():
        best = max(runs, key=lambda r: r.alpha)
        best_per_d[d] = best

    out.append(f"Target alpha for the project: {target}.\n")
    out.append("Best LP results so far:\n")
    out.append("| d | best LP alpha | val(d) | gap to target |")
    out.append("|---|---|---|---|")
    for d in sorted(best_per_d.keys()):
        b = best_per_d[d]
        val_d = b.val_d_known
        gap_to_target = target - b.alpha if b.alpha < target else 0.0
        out.append(f"| {d} | {b.alpha:.6f} (R={b.R}) | {val_d} | "
                   f"{(b.alpha - target):+.6f} |")
    return "\n".join(out)


def generate_report(records: List[RunRecord], path: str, target: float = 1.281) -> None:
    """Write the full markdown report."""
    parts = [
        "# Pólya / Handelman LP Hierarchy for the Sidon val(d) Lower Bound\n",
        "Empirical sweep of the Handelman LP at varying (d, R), Z/2-symmetrized,\n"
        "with variable lambda over windows.\n",
        render_table(records),
        render_convergence_fits(records, target=target),
        render_recommendation(records, target=target),
    ]
    with open(path, "w") as f:
        f.write("\n".join(parts))
