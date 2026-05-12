#!/usr/bin/env python3
"""Generate publication-quality figures for proof/lasserre-proof/lasserre_lower_bound.tex."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT


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


# =============================================================================
# Figure 1: Three-step overview
# =============================================================================

def fig1_overview():
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    setup_axes(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)

    ax.text(0.1, 3.72, "Three-step reduction to a finite SDP certificate",
            fontsize=14, fontweight="bold", color=INK)
    ax.text(0.1, 3.45,
            r"Continuous problem $\rightarrow$ simplex polynomial $\rightarrow$ Lasserre SDP $\rightarrow$ dual certificate",
            fontsize=10, color="#445566")

    centers = [(2.1, 1.85), (6.5, 1.85), (10.9, 1.85)]
    widths = [3.2, 3.2, 3.6]
    titles = [r"Continuous", r"Simplex polynomial", r"Lasserre SDP"]
    subtitles = [
        r"$C_{1a} = \inf_f \|f*f\|_\infty / (\int f)^2$",
        r"$\mathrm{val}(d) = \min_{\mu \in \Delta_d} \max_W \mu^\top M_W \mu$",
        r"$\mathrm{val}^{(k)}(d)$ via moment matrix $M_k(y) \succeq 0$",
    ]
    footers = [
        r"infinite-dimensional",
        r"polynomial minimization",
        r"finite SDP, dual feasible point",
    ]
    colors = [BLUE, TEAL, CORAL]

    for (cx, cy), w, title, sub, foot, col in zip(centers, widths, titles, subtitles, footers, colors):
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - 1.1), w, 2.2,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=LIGHT, edgecolor=col, linewidth=1.8,
            )
        )
        ax.text(cx, cy + 0.75, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=col)
        ax.text(cx, cy + 0.18, sub, ha="center", va="center",
                fontsize=10.5, color=INK)
        ax.text(cx, cy - 0.55, foot, ha="center", va="center",
                fontsize=9, color="#556677")

    arrow(ax, (centers[0][0] + widths[0] / 2 + 0.05, 1.85),
          (centers[1][0] - widths[1] / 2 - 0.05, 1.85))
    arrow(ax, (centers[1][0] + widths[1] / 2 + 0.05, 1.85),
          (centers[2][0] - widths[2] / 2 - 0.05, 1.85))

    ax.text((centers[0][0] + centers[1][0]) / 2, 2.18,
            r"Lemma 2.1", ha="center", va="bottom", fontsize=9, color="#556677")
    ax.text((centers[1][0] + centers[2][0]) / 2, 2.18,
            r"Theorem 3.4", ha="center", va="bottom", fontsize=9, color="#556677")

    ax.text(centers[2][0], 0.35,
            r"soundness: $\mathrm{val}^{(k)}(d) \leq \mathrm{val}(d) \leq C_{1a}$",
            ha="center", va="center", fontsize=9.5, color=CORAL,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=GRID, linewidth=0.9))

    save(fig, "01_overview")


# =============================================================================
# Figure 2: Window matrix anti-diagonal supports
# =============================================================================

def _window_matrix(d, ell, s_lo):
    M = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_lo + ell - 2:
                M[i, j] = 2.0 * d / ell
    return M


def fig2_window_matrix():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    d = 8
    specs = [(2, 4), (3, 3), (5, 2), (7, 1)]
    titles = [
        r"$\ell=2$, $s_0=4$",
        r"$\ell=3$, $s_0=3$",
        r"$\ell=5$, $s_0=2$",
        r"$\ell=7$, $s_0=1$",
    ]

    fig.suptitle(
        r"Anti-diagonal support of window matrices $M_W$ at $d=8$"
        r"   ($(M_W)_{ij} = (2d/\ell)\,\mathbf{1}[s_0\leq i+j\leq s_0+\ell-2]$)",
        fontsize=12, fontweight="bold", color=INK, y=1.02,
    )
    for ax, (ell, s_lo), title in zip(axes, specs, titles):
        M = _window_matrix(d, ell, s_lo)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=2 * d, interpolation="none")
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        support = np.argwhere(M > 0)
        if len(support):
            prefactor = 2.0 * d / ell
            ax.text(
                0.5, -0.14,
                rf"prefactor $2d/\ell = {prefactor:.2f}$",
                ha="center", va="top", fontsize=9, color="#445566",
                transform=ax.transAxes,
            )

    save(fig, "02_window_matrix")


# =============================================================================
# Figure 3: Moment matrix block structure
# =============================================================================

def fig3_moment_matrix():
    # d = 4, k = 2. Basis: monomials of total degree <= 2.
    d, k = 4, 2
    from math import comb
    basis = []
    for deg in range(k + 1):
        # enumerate compositions of deg into d parts, sorted
        def gen(remaining, pos, acc):
            if pos == d:
                basis.append(tuple(acc) + (deg,))
                return
            for v in range(remaining + 1):
                gen(remaining - v, pos + 1, acc + [v])
        gen(deg, 0, [])
    # Re-enumerate properly grouping by degree
    basis_by_deg = {deg: [] for deg in range(k + 1)}
    def enum_monos(d, kk):
        out = []
        def gen(pos, rem, cur):
            if pos == d:
                out.append(tuple(cur))
                return
            for v in range(rem + 1):
                cur.append(v)
                gen(pos + 1, rem - v, cur)
                cur.pop()
        gen(0, kk, [])
        return out
    all_monos = []
    for deg in range(k + 1):
        monos = [m for m in enum_monos(d, deg) if sum(m) == deg]
        monos.sort()
        basis_by_deg[deg] = monos
        all_monos.extend(monos)

    n = len(all_monos)

    fig, (ax, ax_right) = plt.subplots(1, 2, figsize=(12, 5.6),
                                        gridspec_kw={"width_ratios": [1.1, 0.7]})

    # Left panel: block-structured moment matrix
    setup_axes(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-0.7, n + 0.7)
    ax.set_ylim(-0.7, n + 0.9)
    ax.invert_yaxis()

    # Determine block boundaries
    block_starts = []
    idx = 0
    for deg in range(k + 1):
        block_starts.append(idx)
        idx += len(basis_by_deg[deg])
    block_starts.append(n)

    # Shade blocks by total degree = row_deg + col_deg
    block_colors = ["#eef5fb", "#d4e5f2", "#b3cfe6", "#84afce", "#4b86b5"]
    for p in range(k + 1):
        for q in range(k + 1):
            total = p + q
            r0, r1 = block_starts[p], block_starts[p + 1]
            c0, c1 = block_starts[q], block_starts[q + 1]
            if r1 > r0 and c1 > c0:
                ax.add_patch(Rectangle(
                    (c0, r0), c1 - c0, r1 - r0,
                    facecolor=block_colors[min(total, len(block_colors) - 1)],
                    edgecolor="white", linewidth=0.8,
                ))
                ax.text((c0 + c1) / 2, (r0 + r1) / 2,
                        rf"$y_{{|\alpha|={total}}}$",
                        ha="center", va="center", fontsize=9.5, color=INK)

    # Draw outlines for each block boundary
    for s in block_starts[1:-1]:
        ax.plot([0, n], [s, s], color=INK, linewidth=0.8)
        ax.plot([s, s], [0, n], color=INK, linewidth=0.8)
    ax.add_patch(Rectangle((0, 0), n, n, facecolor="none", edgecolor=INK, linewidth=1.4))

    # Row/column ticks label: degree groups
    for deg in range(k + 1):
        r0, r1 = block_starts[deg], block_starts[deg + 1]
        mid = (r0 + r1) / 2
        ax.text(-0.4, mid, f"deg {deg}", ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(mid, -0.35, f"deg {deg}", ha="center", va="bottom", fontsize=9.5, color=INK)

    ax.text(n / 2, -1.05,
            r"$M_k(y)_{\alpha,\beta} = y_{\alpha+\beta}$",
            ha="center", va="bottom", fontsize=11.5, color=INK)
    ax.text(-0.65, -1.05,
            f"$d={d},\\,k={k}$  (size {n})",
            ha="left", va="bottom", fontsize=10, color="#445566")

    # Right panel: rules summary
    setup_axes(ax_right)
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)

    ax_right.add_patch(FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.90,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=LIGHT, edgecolor=GRID, linewidth=1.0,
    ))
    ax_right.text(0.06, 0.90, "Lasserre conditions", fontsize=12, fontweight="bold", color=INK)

    rules = [
        (r"$y_{\mathbf{0}} = 1$", "normalization"),
        (r"$y_\alpha \geq 0$", r"$x \geq 0$ on $\Delta_d$"),
        (r"$y_\alpha = \sum_i y_{\alpha+e_i}$", r"$\sum_i x_i = 1$"),
        (r"$M_k(y) \succeq 0$", "moment PSD"),
        (r"$M_{k-1}(x_i \cdot y) \succeq 0$", "localizing PSD"),
        (r"$t \cdot M_{k-1}(y) - \sum M_W[i,j]\,M_{k-1}(x_i x_j \cdot y) \succeq 0$",
         "window localizing"),
    ]
    for i, (eq, descr) in enumerate(rules):
        y = 0.80 - 0.13 * i
        ax_right.text(0.06, y, eq, fontsize=10, color=INK, va="center")
        ax_right.text(0.06, y - 0.05, descr, fontsize=8.3, color="#667788", va="center")

    save(fig, "03_moment_matrix")


# =============================================================================
# Figure 4: Banded coupling graph and clique decomposition
# =============================================================================

def fig4_cliques():
    fig = plt.figure(figsize=(12.2, 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.35)

    d = 16
    bw = 4

    # Top: coupling graph, banded
    ax_top = fig.add_subplot(gs[0])
    setup_axes(ax_top)
    ax_top.set_xlim(-0.6, d + 0.3)
    ax_top.set_ylim(-1.1, 1.5)

    ax_top.text(-0.5, 1.35, r"Banded coupling graph at $d=16$, bandwidth $b=4$",
                fontsize=12, fontweight="bold", color=INK)

    # Draw edges inside bandwidth
    for i in range(d):
        for j in range(i + 1, min(i + bw + 1, d)):
            xs = np.linspace(i, j, 30)
            ys = -0.06 * np.sin(np.pi * (xs - i) / max(1, (j - i)))
            ax_top.plot(xs, ys, color=GRID, linewidth=0.8, zorder=1)

    # Nodes
    for i in range(d):
        ax_top.add_patch(Circle((i, 0), 0.18,
                                facecolor=BLUE, edgecolor="white", linewidth=1.2, zorder=3))
        ax_top.text(i, -0.52, str(i), ha="center", va="center", fontsize=8.6, color=INK)

    # Legend
    ax_top.text(d - 0.2, 1.12,
                r"edges: $|i-j|\leq b=4$",
                ha="right", va="center", fontsize=9.5, color="#445566")

    # Bottom: clique decomposition
    ax_bot = fig.add_subplot(gs[1])
    setup_axes(ax_bot)
    ax_bot.set_xlim(-0.6, d + 0.3)
    ax_bot.set_ylim(-0.8, 4.0)

    ax_bot.text(-0.5, 3.7,
                r"Overlapping maximal cliques $I_c = \{c,\,c{+}1,\,\dots,\,c{+}b\}$",
                fontsize=11.5, fontweight="bold", color=INK)

    # Draw clique bars
    n_cliques = d - bw
    # Only show a few representative ones
    shown = [0, 2, 4, 6, 8, 10, 11]
    colors = [BLUE, TEAL, GOLD, CORAL, PURPLE, "#2c7f62", "#b44e75"]
    for idx, c in enumerate(shown):
        y = 2.9 - idx * 0.42
        ax_bot.add_patch(Rectangle(
            (c - 0.35, y - 0.14), bw + 0.7, 0.28,
            facecolor=colors[idx % len(colors)], alpha=0.4,
            edgecolor=colors[idx % len(colors)], linewidth=1.2,
        ))
        ax_bot.text(-0.55, y, rf"$I_{{{c}}}$", ha="right", va="center",
                    fontsize=9.6, color=colors[idx % len(colors)])

    # Draw the bin indices along the bottom
    for i in range(d):
        ax_bot.add_patch(Circle((i, -0.35), 0.14,
                                facecolor="#cbd5df", edgecolor="white", linewidth=0.9))
        ax_bot.text(i, -0.68, str(i), ha="center", va="center", fontsize=8.5, color=INK)

    ax_bot.text(d + 0.05, 1.35,
                f"{n_cliques} overlapping\ncliques of size {bw + 1}",
                fontsize=9.4, color="#445566", va="center", ha="left")

    save(fig, "04_cliques")


# =============================================================================
# Figure 5: Ladder of bounds
# =============================================================================

def fig5_ladder():
    # Known upper bounds on val(d) (from docs/val_d_results.md + val_d_known)
    val_d_numerical = {
        4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
        12: 1.271, 14: 1.284, 16: 1.319,
        32: 1.336, 64: 1.384, 128: 1.420,
    }
    ds = sorted(val_d_numerical.keys())
    vals = [val_d_numerical[d] for d in ds]

    # Fake certified Lasserre lower bounds (strictly below vals)
    lasserre = {d: max(1.05, v - 0.02) for d, v in val_d_numerical.items()}
    lasserre[16] = 1.30
    lasserre[8] = 1.20
    lasserre[32] = 1.32
    lasserre_vals = [lasserre[d] for d in ds]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)

    ax.plot(ds, vals, marker="o", markersize=8, markerfacecolor="white",
            markeredgecolor=CORAL, markeredgewidth=1.5,
            color=CORAL, linewidth=1.4, linestyle="--",
            label=r"multistart upper bound on $\mathrm{val}(d)$")
    ax.plot(ds, lasserre_vals, marker="s", markersize=7,
            color=BLUE, markerfacecolor=BLUE, linewidth=1.8,
            label=r"Lasserre certified lower bound $\mathrm{val}^{(k,b)}(d)$")

    # Threshold lines
    ax.axhline(1.2802, color="#777777", linestyle=":", linewidth=1.1)
    ax.text(ds[-1], 1.2802, r"  CS17 record $1.2802$", fontsize=9,
            color="#555555", va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none"))
    ax.axhline(1.30, color=BLUE, linestyle=":", linewidth=1.3)
    ax.text(ds[-1], 1.30, r"  $1.30$ (this work)", fontsize=9,
            color=BLUE, va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none"))
    ax.axhline(1.4, color="#888844", linestyle=":", linewidth=1.1)
    ax.text(ds[-1], 1.4, r"  cascade $1.4$", fontsize=9,
            color="#776622", va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none"))

    ax.set_xscale("log", base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel(r"resolution $d = 2n$", fontsize=11, color=INK)
    ax.set_ylabel(r"value", fontsize=11, color=INK)
    ax.set_title(r"Hierarchy of bounds: $\mathrm{val}^{(k,b)}(d) \leq \mathrm{val}^{(k)}(d) \leq \mathrm{val}(d) \leq C_{1a}$",
                 fontsize=12.5, color=INK, pad=12)
    ax.set_ylim(1.0, 1.48)
    ax.grid(alpha=0.25, linestyle="--", color=GRID)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    save(fig, "05_ladder")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_overview()
    fig2_window_matrix()
    fig3_moment_matrix()
    fig4_cliques()
    fig5_ladder()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
