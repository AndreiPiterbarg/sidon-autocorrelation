#!/usr/bin/env python3
"""Generate publication-quality figures for proof/lower_bound_proof.tex."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT
CASCADE_STATS = ROOT.parent.parent / "data" / "cpu_cascade_20260319_201644.json"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "savefig.transparent": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


BLUE = "#1f4e79"
TEAL = "#1f9d8a"
GOLD = "#d99a2b"
CORAL = "#ca5a57"
PURPLE = "#6f5cc2"
INK = "#25313c"
GRID = "#d5dde6"
SOFT = "#edf3f8"
SOFT2 = "#f7f1e3"
LIGHT = "#fbfcfe"


def load_cascade_stats() -> list[tuple[str, int]]:
    with CASCADE_STATS.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    stages = [("$d=4$", int(payload["l0_survivors"]))]
    for level in payload["levels"]:
        stages.append((fr"$d={int(level['d_child'])}$", int(level["survivors_out"])))
    return stages


def fmt_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def setup_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def arrow(ax, start, end, **kwargs):
    kw = dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.6, color=INK)
    kw.update(kwargs)
    ax.add_patch(FancyArrowPatch(start, end, **kw))


def fig1_partition_bins():
    fig = plt.figure(figsize=(10.2, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.06)

    ax_top = fig.add_subplot(gs[0])
    setup_axes(ax_top)
    ax_top.set_xlim(-0.31, 0.31)
    ax_top.set_ylim(-0.02, 1.08)

    xs = np.linspace(-0.25, 0.25, 600)
    y = 0.18 + 0.57 * (
        0.65 * np.exp(-((xs + 0.11) / 0.085) ** 2)
        + 0.85 * np.exp(-((xs - 0.02) / 0.06) ** 2)
        + 0.60 * np.exp(-((xs - 0.145) / 0.05) ** 2)
    )
    y = y / y.max() * 0.82 + 0.08
    ax_top.plot(xs, y, color=BLUE, linewidth=2.6)
    ax_top.fill_between(xs, 0.08, y, color=BLUE, alpha=0.08)

    # Bins inside the support interval.
    bins = np.linspace(-0.25, 0.25, 9)
    colors = ["#e8eff7", "#d7e6f2"]
    for i in range(len(bins) - 1):
        x0, x1 = bins[i], bins[i + 1]
        ax_top.add_patch(
            Rectangle((x0, 0.02), x1 - x0, 0.96, facecolor=colors[i % 2], edgecolor="white", linewidth=1.2)
        )
    ax_top.plot(xs, y, color=BLUE, linewidth=2.6)
    ax_top.fill_between(xs, 0.08, y, color=BLUE, alpha=0.08)

    for x in bins:
        ax_top.plot([x, x], [0.02, 0.98], color="white", linewidth=1.1)
    for i in range(len(bins) - 1):
        xc = (bins[i] + bins[i + 1]) / 2
        ax_top.text(xc, 0.965, f"$I_{i}$", ha="center", va="top", fontsize=9, color=INK)

    ax_top.annotate(
        r"$f(x)$",
        xy=(0.195, 0.77),
        xytext=(0.26, 0.92),
        arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.2),
        ha="left",
        va="center",
        color=INK,
    )
    ax_top.text(-0.305, 1.06, "Support partition and bin averages", ha="left", va="bottom", fontsize=12.5, fontweight="bold", color=INK)
    ax_top.text(
        -0.305,
        0.88,
        r"$\operatorname{supp}(f)\subset[-\frac{1}{4},\frac{1}{4}]$",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=INK,
    )
    ax_top.text(
        0.265,
        0.05,
        r"$a_j = 4n\int_{I_j} f$",
        ha="right",
        va="bottom",
        fontsize=10,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, linewidth=0.9),
    )

    ax_mid = fig.add_subplot(gs[1])
    setup_axes(ax_mid)
    ax_mid.set_xlim(-0.31, 0.31)
    ax_mid.set_ylim(-0.42, 0.45)
    ax_mid.axhline(0, color=GRID, linewidth=1.0)

    # Stylized step averages underneath.
    heights = np.array([0.10, 0.19, 0.14, 0.26, 0.30, 0.18, 0.24, 0.12])
    for i in range(len(bins) - 1):
        x0, x1 = bins[i], bins[i + 1]
        ax_mid.add_patch(
            Rectangle(
                (x0, 0),
                x1 - x0,
                heights[i],
                facecolor=[SOFT, SOFT2][i % 2],
                edgecolor=BLUE,
                linewidth=1.0,
            )
        )
        ax_mid.text((x0 + x1) / 2, heights[i] + 0.03, f"${heights[i]:.2f}$", ha="center", va="bottom", fontsize=8, color=INK)

    ax_mid.text(-0.305, 0.41, r"$a=(a_0,\ldots,a_{d-1})\in \mathbb{R}_{\geq 0}^d$", ha="left", va="top", fontsize=9.6, color=INK)
    ax_mid.text(0.305, -0.02, r"$\sum_j a_j = 4n$", ha="right", va="top", fontsize=10, color=INK)
    ax_mid.text(0.305, -0.24, r"Normalization turns $\int f=1$ into a simplex constraint.", ha="right", va="top", fontsize=9, color="#445566")

    save(fig, "01_partition_bins")


def barycentric_to_xy(w):
    # Triangle vertices: left, right, top
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    return w[0] * v0 + w[1] * v1 + w[2] * v2


def fig2_simplex_to_lattice():
    fig, ax = plt.subplots(figsize=(7.3, 6.6))
    setup_axes(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.08, 0.98)

    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    tri = np.vstack([v0, v1, v2])
    ax.add_patch(Polygon(tri, closed=True, facecolor="#f9fbfd", edgecolor=INK, linewidth=1.8))

    m = 7
    pts = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            k = m - i - j
            w = np.array([i, j, k]) / m
            pts.append(barycentric_to_xy(w))
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=20, color="#aab9c8", alpha=0.95, zorder=2)

    # Grid lines.
    for i in range(1, m):
        t = i / m
        # lines parallel to each edge
        p1 = barycentric_to_xy(np.array([t, 0, 1 - t]))
        p2 = barycentric_to_xy(np.array([t, 1 - t, 0]))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=GRID, linewidth=0.8, zorder=1)

        p1 = barycentric_to_xy(np.array([0, t, 1 - t]))
        p2 = barycentric_to_xy(np.array([1 - t, t, 0]))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=GRID, linewidth=0.8, zorder=1)

        p1 = barycentric_to_xy(np.array([0, 1 - t, t]))
        p2 = barycentric_to_xy(np.array([1 - t, 0, t]))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=GRID, linewidth=0.8, zorder=1)

    mu = np.array([0.26, 0.39, 0.35])
    b = np.array([2, 3, 2]) / 7
    mu_xy = barycentric_to_xy(mu)
    b_xy = barycentric_to_xy(b)

    # Uncertainty box around mu in barycentric coordinates.
    eps = 0.055
    corners = np.array(
        [
            barycentric_to_xy(mu + np.array([-eps, -eps, 2 * eps])),
            barycentric_to_xy(mu + np.array([eps, -eps, 0.0])),
            barycentric_to_xy(mu + np.array([eps, eps, -2 * eps])),
            barycentric_to_xy(mu + np.array([-eps, eps, 0.0])),
        ]
    )
    ax.add_patch(Polygon(corners, closed=True, facecolor=TEAL, alpha=0.10, edgecolor=TEAL, linewidth=1.2, zorder=3))

    ax.scatter([mu_xy[0]], [mu_xy[1]], s=130, color=CORAL, edgecolor="white", linewidth=1.6, zorder=5)
    ax.scatter([b_xy[0]], [b_xy[1]], s=130, color=GOLD, edgecolor="white", linewidth=1.6, zorder=5)
    ax.plot([mu_xy[0], b_xy[0]], [mu_xy[1], b_xy[1]], color=INK, linewidth=1.5, linestyle="--", zorder=4)
    ax.text(mu_xy[0] + 0.03, mu_xy[1] + 0.03, r"$\mu$", color=CORAL, fontsize=13, fontweight="bold")
    ax.text(b_xy[0] + 0.03, b_xy[1] - 0.04, r"$b$", color="#9a6a10", fontsize=13, fontweight="bold")

    # A small coordinate inset.
    inset = fig.add_axes([0.66, 0.08, 0.27, 0.22])
    setup_axes(inset)
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.88, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="white", edgecolor=GRID, linewidth=1.0))
    inset.text(0.12, 0.78, r"continuous point", fontsize=9, color=INK, va="center")
    inset.text(0.12, 0.58, r"$\mu \in A_n$", fontsize=10, color=CORAL, va="center")
    inset.text(0.12, 0.34, r"nearest lattice point", fontsize=9, color=INK, va="center")
    inset.text(0.12, 0.14, r"$b \in B_{n,m}$", fontsize=10, color="#9a6a10", va="center")
    inset.annotate("", xy=(0.78, 0.56), xytext=(0.72, 0.55), arrowprops=dict(arrowstyle="->", linewidth=1.2, color=INK))
    inset.text(0.78, 0.56, r"$\| \mu-b \|_\infty \leq \frac{1}{m}$", fontsize=9, color=INK, va="center", ha="left")

    # Labels.
    ax.text(0.01, 0.96, "Discretization of the simplex", transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK)
    ax.text(0.01, 0.91, r"One continuous point is rounded to a nearby grid point.", transform=ax.transAxes, fontsize=10, color="#445566")
    ax.text(0.14, -0.03, r"$A_n$", fontsize=12, color=INK)
    ax.text(0.88, -0.03, r"$B_{n,m}$", fontsize=12, color=INK)
    ax.text(0.46, 0.79, r"$\|\mu-b\|_\infty\leq \frac{1}{m}$", fontsize=11, color=INK, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=GRID))
    save(fig, "02_simplex_lattice")


def fig3_reversal_canonical():
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    setup_axes(ax)
    ax.set_xlim(-0.8, 9.3)
    ax.set_ylim(-0.55, 2.55)

    ax.text(-0.72, 2.2, "Reversal symmetry and canonicalization", fontsize=14, fontweight="bold", color=INK)
    ax.text(-0.72, 2.0, r"$\operatorname{tv}(a)=\operatorname{tv}(\operatorname{rev}(a))$ reduces the search to one representative per orbit.", fontsize=9.7, color="#445566")

    example = np.array([1, 4, 2, 3, 3, 2, 4, 1]) / 4
    rev = example[::-1]
    x = np.arange(len(example))

    def draw_row(y, vals, color, label, annotate_pairs=False):
        for i, v in enumerate(vals):
            ax.add_patch(Rectangle((i + 0.22, y), 0.62, v * 0.3, facecolor=color, edgecolor="white", linewidth=1.0))
            ax.text(i + 0.53, y - 0.08, str(int(round(v * 4))), ha="center", va="top", fontsize=9, color=INK)
        ax.text(-0.42, y + 0.28, label, fontsize=11, fontweight="bold", color=INK)
        ax.text(8.02, y + 0.22, r"$\sum_i c_i = m$", fontsize=9, color="#445566")
        if annotate_pairs:
            for i in range(4):
                x0 = i + 0.53
                x1 = len(vals) - 1 - i + 0.53
                ax.plot([x0, x0], [y + 0.05, y - 0.03], color=GRID, linewidth=1.0)
                ax.plot([x1, x1], [y + 0.05, y - 0.03], color=GRID, linewidth=1.0)
                ax.text((x0 + x1) / 2, y + 0.34, f"compare $c_{i}$ vs $c_{len(vals)-1-i}$", ha="center", va="bottom", fontsize=7.7, color="#445566")

    draw_row(1.25, example, BLUE, r"$c$", annotate_pairs=True)
    draw_row(0.25, rev, GOLD, r"$\operatorname{rev}(c)$")

    arrow(ax, (8.55, 1.45), (8.55, 0.55), color=INK, linewidth=1.8)
    ax.text(8.72, 1.00, "reverse", fontsize=10, color=INK, rotation=90, va="center")

    ax.add_patch(FancyBboxPatch((0.28, -0.42), 7.1, 0.34, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=SOFT, edgecolor=GRID, linewidth=0.9))
    ax.text(0.42, -0.25, r"canonicalize: $\min(c,\operatorname{rev}(c))$.", fontsize=10.0, color=INK, va="center")
    ax.add_patch(FancyArrowPatch((3.95, -0.10), (4.72, -0.10), arrowstyle="->", mutation_scale=12, linewidth=1.3, color=INK))
    ax.text(4.90, -0.25, r"palindromes unchanged", fontsize=9.2, color="#445566", va="center")

    # Symmetry indicator.
    ax.add_patch(Arc((4.53, 1.0), 7.3, 1.8, theta1=200, theta2=-20, linewidth=1.3, color=TEAL))
    ax.text(4.53, 1.89, r"mirror symmetry", fontsize=9.8, color=TEAL, ha="center")

    save(fig, "03_reversal_canonical")


def fig4_cascade_flow():
    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    setup_axes(ax)
    ax.set_xlim(-0.4, 12.75)
    ax.set_ylim(-0.2, 4.0)
    ax.text(-0.28, 3.72, "Multiscale cascade", fontsize=14, fontweight="bold", color=INK)
    ax.text(-0.28, 3.49, "Coarse levels prune cheaply; surviving branches are refined dyadically.", fontsize=10, color="#445566")

    stages = load_cascade_stats()
    xs = np.arange(len(stages)) * 1.75 + 0.35
    levels = [label for label, _ in stages]
    survivor_counts = [count for _, count in stages]
    dot_counts = [9, 8, 7, 5, 3, 1]
    prune_counts = [22, 16, 12, 8, 5, 0]
    labels = ["asymmetry", "window scan", "energy cap", "canonicalize", "refine", "certificate"]
    colors = [BLUE, TEAL, GOLD, PURPLE, CORAL, "#2c7f62"]

    for i, x in enumerate(xs):
        box = FancyBboxPatch((x - 0.45, 1.25), 0.9, 1.38, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=LIGHT, edgecolor=GRID, linewidth=1.1)
        ax.add_patch(box)
        ax.text(x, 2.45, levels[i], ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(x, 2.18, labels[i], ha="center", va="center", fontsize=9, color="#445566")

        # Draw a few survivors and pruned nodes in each stage.
        base_y = 1.45
        # survivors stack
        for j in range(dot_counts[i]):
            y = base_y + 0.12 * j
            ax.add_patch(Circle((x - 0.15 + 0.07 * (j % 2), y), 0.035, facecolor=colors[i], edgecolor="white", linewidth=0.8, alpha=0.95))
        # pruned faint nodes
        for j in range(prune_counts[i]):
            y = base_y + 0.02 * (j % 8)
            xoff = x + 0.12 + 0.05 * (j % 3)
            ax.add_patch(Circle((xoff, y), 0.025, facecolor="#cbd5df", edgecolor="none", alpha=0.55))
        ax.text(x, 1.07, f"{fmt_count(survivor_counts[i])} survive", ha="center", va="center", fontsize=9, color=INK)

    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + 0.48, 1.95), (xs[i + 1] - 0.48, 1.95), color=INK, linewidth=1.5)
        ax.text((xs[i] + xs[i + 1]) / 2, 2.08, "refine", fontsize=8.8, color="#445566", ha="center")

    # Side legend.
    ax.add_patch(FancyBboxPatch((9.95, 0.35), 2.2, 2.75, boxstyle="round,pad=0.03,rounding_size=0.04", facecolor="white", edgecolor=GRID, linewidth=1.0))
    ax.text(10.06, 2.88, "pruning stack", fontsize=11, fontweight="bold", color=INK)
    entries = [
        ("asymmetry", BLUE),
        ("dynamic window", TEAL),
        ("single-bin cap", GOLD),
        ("canonicalization", PURPLE),
        ("terminal certificate", CORAL),
    ]
    for i, (name, c) in enumerate(entries):
        y = 2.55 - 0.42 * i
        ax.add_patch(Rectangle((10.08, y - 0.08), 0.12, 0.12, facecolor=c, edgecolor="white", linewidth=0.8))
        ax.text(10.25, y - 0.02, name, fontsize=8.9, color=INK, va="center")

    ax.text(10.06, 0.62, r"$0$ survivors at $d=128$", fontsize=9.4, color=INK)

    # Encircle the final stage.
    ax.add_patch(FancyBboxPatch((xs[-1] - 0.6, 1.02), 1.2, 1.78, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="none", edgecolor=CORAL, linewidth=1.6, linestyle="--"))
    save(fig, "04_cascade_flow")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_partition_bins()
    fig2_simplex_to_lattice()
    fig3_reversal_canonical()
    fig4_cascade_flow()


if __name__ == "__main__":
    main()
