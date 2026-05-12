"""Generate figures for the STAT 4830 presentation (editorial theme)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---- Editorial palette (shared with build_deck.py) ----
PAPER = "#F7F2E8"
INK = "#1C1917"
STONE = "#7C756B"
HAIRLINE = "#D6CDB8"
HIGHLIGHT = "#EEE6D2"

NAVY = "#25435F"      # Track 1 — cascade
FOREST = "#3A5445"    # Track 2 — SDP
BRICK = "#8E3A38"     # Track 3 — audit
TERRA = "#B54B2F"     # cross-section accent
GOLD = "#A88431"      # secondary accent

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 15,
    "axes.titleweight": "regular",
    "axes.labelsize": 13,
    "axes.edgecolor": STONE,
    "axes.labelcolor": INK,
    "xtick.color": STONE,
    "ytick.color": STONE,
    "text.color": INK,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.edgecolor": PAPER,
    "figure.dpi": 160,
    "savefig.dpi": 240,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def _style(ax):
    """Apply editorial spine + tick treatment."""
    ax.tick_params(length=0, pad=4)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(STONE)
        ax.spines[spine].set_linewidth(0.8)


def autoconv_intuition():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    x = np.linspace(-0.25, 0.25, 800)
    f = np.where(np.abs(x) < 0.25, np.exp(-18 * x**2), 0.0)
    f = f / np.trapezoid(f, x)
    t = np.linspace(-0.5, 0.5, 1600)
    dx = x[1] - x[0]
    fc = np.convolve(f, f) * dx
    tc = np.linspace(2 * x[0], 2 * x[-1], len(fc))

    axes[0].fill_between(x, f, color=NAVY, alpha=0.18)
    axes[0].plot(x, f, color=NAVY, lw=2.4)
    axes[0].set_xlim(-0.32, 0.32)
    axes[0].set_title("f  on  [-1/4, 1/4],   ∫ f = 1",
                      color=INK, fontsize=13)
    axes[0].axvline(-0.25, color=STONE, lw=0.6, ls=(0, (1, 3)))
    axes[0].axvline(0.25, color=STONE, lw=0.6, ls=(0, (1, 3)))
    _style(axes[0])

    peak = fc.max()
    axes[1].fill_between(tc, fc, color=TERRA, alpha=0.18)
    axes[1].plot(tc, fc, color=TERRA, lw=2.4)
    axes[1].axhline(peak, color=TERRA, lw=1.2, ls="--", alpha=0.8)
    axes[1].annotate("‖ f * f ‖∞", xy=(0.0, peak), xytext=(0.30, peak * 0.93),
                     fontsize=13, color=TERRA, ha="center")
    axes[1].set_xlim(-0.58, 0.58)
    axes[1].set_title("f * f  on  [-1/2, 1/2]",
                      color=INK, fontsize=13)
    axes[1].axvline(-0.5, color=STONE, lw=0.6, ls=(0, (1, 3)))
    axes[1].axvline(0.5, color=STONE, lw=0.6, ls=(0, (1, 3)))
    _style(axes[1])

    for ax in axes:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_autoconv_intuition.png")
    plt.close(fig)


def bounds_number_line():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(1.23, 1.55)
    ax.set_ylim(-1.3, 1.5)

    ax.hlines(0, 1.24, 1.54, color=INK, lw=1.3, zorder=1)
    for v in np.arange(1.25, 1.56, 0.05):
        ax.plot([v, v], [-0.04, 0.04], color=INK, lw=0.9)
        ax.text(v, -0.16, f"{v:.2f}", ha="center", va="top",
                fontsize=11, color=STONE)

    feasible_l, feasible_r = 1.4, 1.5029
    ax.fill_betweenx([-0.04, 0.04], feasible_l, feasible_r,
                     color=HAIRLINE, zorder=0)
    ax.text((feasible_l + feasible_r) / 2, -0.55, "remaining uncertainty",
            ha="center", fontsize=11, color=STONE, style="italic")

    markers = [
        (1.2802, "C&S 2017",       "lower bound",         STONE, 0.70, "right",  -0.002),
        (1.3,    "SDP",            "Lasserre — ours",     FOREST, 1.30, "left",   0.002),
        (1.4,    "Cascade",        "GPU — ours",          NAVY,   0.70, "center", 0.0),
        (1.5029, "Matolcsi–Vinuesa", "upper bound",       BRICK,  1.30, "center", 0.0),
    ]
    for v, title, sub, c, y, ha, dx in markers:
        ax.plot([v, v], [0.05, y - 0.18], color=c, lw=1.5, alpha=0.95)
        ax.scatter([v], [0], color=c, s=72, zorder=3,
                   edgecolors=PAPER, linewidths=1.4)
        ax.text(v + dx, y, f"{title}\n{v}", ha=ha, va="bottom",
                fontsize=12, color=c, fontweight="bold",
                fontfamily="serif")
        ax.text(v + dx, y - 0.18, sub, ha=ha, va="top",
                fontsize=9.5, color=STONE, style="italic")

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bounds_number_line.png")
    plt.close(fig)


def correction_gap():
    ds = np.array([3, 6, 12, 24, 48, 96])
    eps_mass = 0.02
    code = 2 * eps_mass + eps_mass ** 2 * np.ones_like(ds, dtype=float)
    true = 2 * (2 * ds * eps_mass) + (2 * ds * eps_mass) ** 2

    fig, ax = plt.subplots(figsize=(9, 4.3))
    width = 0.38
    idx = np.arange(len(ds))
    ax.bar(idx - width / 2, code, width, color=STONE,
           label="Code budget  (ε = mass step = 0.02)")
    ax.bar(idx + width / 2, true, width, color=BRICK,
           label="Required   (ε = height step = 2d × 0.02)")

    for i, (c, t) in enumerate(zip(code, true)):
        ratio = t / c
        ax.text(i + width / 2, t * 1.12, f"× {ratio:.0f}", ha="center",
                fontsize=11, color=BRICK, fontweight="bold",
                fontfamily="serif")
        ax.text(i - width / 2, c * 1.6, f"{c:.3f}", ha="center",
                fontsize=8.5, color=STONE)

    ax.set_xticks(idx)
    ax.set_xticklabels([f"d = {d}" for d in ds], color=INK, fontsize=11)
    ax.set_ylabel("Lemma-3 correction   (log scale, W ≈ 1)",
                  fontsize=12, color=INK)
    ax.set_title("the true correction grows as  2d · ε,   so it is too small by  ≈ 2d",
                 fontsize=12, color=STONE, pad=12, style="italic")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=11,
              labelcolor=INK)
    ax.grid(True, axis="y", which="both", color=HAIRLINE, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_correction_gap.png")
    plt.close(fig)


def lasserre_ladder():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    labels = [
        (r"$\mathrm{val}^{(k,b)}(d)$", "sparse Lasserre",     FOREST, 0.28),
        (r"$\mathrm{val}^{(k)}(d)$",   "dense Lasserre",      FOREST, 0.60),
        (r"$\mathrm{val}(d)$",         "simplex polynomial",  GOLD,   0.78),
        (r"$C_{1a}$",                  "autoconvolution",     TERRA,  1.0),
    ]
    y = 0.55
    for i, (tex, sub, c, alpha) in enumerate(labels):
        x = 0.9 + i * 2.35
        # tile
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.95, y), 1.9, 0.95,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=c, alpha=alpha * 0.22, edgecolor=c, lw=1.0,
            zorder=1))
        ax.text(x, y + 0.65, tex, ha="center", va="center",
                fontsize=18, color=c, fontweight="bold", zorder=2)
        ax.text(x, y + 0.25, sub, ha="center", va="center",
                fontsize=10.5, color=STONE, style="italic", zorder=2)
        if i < 3:
            ax.annotate("", xy=(x + 1.05, y + 0.48),
                        xytext=(x + 0.97, y + 0.48),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                        color=STONE, mutation_scale=14))
            ax.text(x + 1.01, y + 0.78, r"$\leq$", ha="center",
                    fontsize=16, color=STONE)

    ax.text(4.3, 0.2, "we solve at  k = 3,  d = 16",
            ha="center", fontsize=11, color=FOREST, style="italic",
            fontfamily="serif")

    ax.set_xlim(-0.4, 9.4)
    ax.set_ylim(0, 1.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_lasserre_ladder.png")
    plt.close(fig)


def effective_m_plot():
    ds = np.arange(2, 100)
    eps_mass = 0.02
    m_eff = 1.0 / (2 * ds * eps_mass)

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(ds, m_eff, color=BRICK, lw=2.4,
            label=r"effective  $m$   in the C&S fine grid")
    ax.axhline(1, color=STONE, lw=1.2, ls="--",
               label="Lemma 3 says nothing below 1")
    ax.axhline(50, color=GOLD, lw=1.0, ls=":", alpha=0.8,
               label=r"C&S paper's  $m = 50$")

    ax.fill_between(ds, 0.3, 1, where=(m_eff < 1),
                    color=BRICK, alpha=0.10)
    ax.set_xlabel(r"cascade resolution  $d$", color=INK, fontsize=12)
    ax.set_ylabel(r"effective  $m_{\mathrm{CS}}$",
                  color=INK, fontsize=12)
    ax.set_title("the deeper the cascade, the less the Lemma-3 correction says",
                 fontsize=12, color=STONE, pad=12, style="italic")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 80)
    ax.legend(loc="upper right", frameon=False, fontsize=11,
              labelcolor=INK)
    ax.grid(True, axis="y", which="both", color=HAIRLINE, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_effective_m.png")
    plt.close(fig)


def three_tracks():
    fig, ax = plt.subplots(figsize=(12, 4.0))
    tracks = [
        ("I",  "Cascade",        "branch-and-prune\non GPU",
         r"$C_{1a} \geq 1.4$",         NAVY),
        ("II", "Lasserre SDP",   "continuous  →  polynomial\n→  semidefinite",
         r"$C_{1a} \geq 1.3$ · certified",     FOREST),
        ("III","MATLAB audit",   "soundness check of the\nC&S artifact",
         "bug identified",               BRICK),
    ]
    for i, (tag, head, body, result, c) in enumerate(tracks):
        x = 1.6 + i * 3.6
        # thin left rule
        ax.plot([x - 1.35, x - 1.35], [0.15, 3.4], color=c, lw=2.4, zorder=1)
        # roman numeral
        ax.text(x - 1.2, 3.2, tag, ha="left", va="top",
                fontsize=30, color=c, fontfamily="serif",
                fontweight="regular", alpha=0.8)
        ax.text(x - 1.2, 2.55, head, ha="left", va="top",
                fontsize=19, color=INK, fontweight="bold",
                fontfamily="serif")
        ax.text(x - 1.2, 2.05, body, ha="left", va="top",
                fontsize=12, color=STONE)
        ax.text(x - 1.2, 0.75, result, ha="left", va="top",
                fontsize=15, color=c, fontweight="bold",
                fontfamily="serif")
    ax.set_xlim(-0.4, 12.4)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_three_tracks.png")
    plt.close(fig)


def cascade_schematic():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    levels = 5
    ys = np.linspace(3.9, 0.45, levels)
    prev_positions = [(5.0, ys[0])]
    ax.scatter(*prev_positions[0], s=150, color=NAVY, zorder=3,
               edgecolors=PAPER, linewidths=1.2)
    ax.text(prev_positions[0][0], prev_positions[0][1] + 0.28, "root",
            ha="center", fontsize=10, color=NAVY, fontfamily="serif")

    rng = np.random.default_rng(3)
    for lv in range(1, levels):
        new_positions = []
        for (px, py) in prev_positions:
            n_children = rng.integers(2, 4)
            xs = (np.linspace(px - 1.0, px + 1.0, n_children)
                  if n_children > 1 else [px])
            for cx in xs:
                cy = ys[lv]
                survive = rng.random() < (0.55 if lv < levels - 1 else 0.35)
                color = NAVY if survive else BRICK
                alpha = 0.95 if survive else 0.55
                ax.plot([px, cx], [py, cy], color=STONE, lw=0.5,
                        alpha=0.45, zorder=1)
                if survive:
                    ax.scatter([cx], [cy], s=60, color=color,
                               alpha=alpha, zorder=3,
                               edgecolors=PAPER, linewidths=0.8)
                    new_positions.append((cx, cy))
                else:
                    ax.plot([cx - 0.08, cx + 0.08], [cy - 0.08, cy + 0.08],
                            color=color, lw=1.6, alpha=alpha)
                    ax.plot([cx - 0.08, cx + 0.08], [cy + 0.08, cy - 0.08],
                            color=color, lw=1.6, alpha=alpha)
        prev_positions = new_positions or prev_positions

    ax.text(0.5, 3.9, "refine\n→ enumerate",
            fontsize=11, color=NAVY, ha="left", va="top",
            style="italic", fontfamily="serif")
    ax.text(0.5, 0.65, "prune when test value\nexceeds threshold",
            fontsize=11, color=BRICK, ha="left", va="top",
            style="italic", fontfamily="serif")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cascade_schematic.png")
    plt.close(fig)


if __name__ == "__main__":
    autoconv_intuition()
    bounds_number_line()
    correction_gap()
    lasserre_ladder()
    effective_m_plot()
    three_tracks()
    cascade_schematic()
    print("figures written to", OUT)
