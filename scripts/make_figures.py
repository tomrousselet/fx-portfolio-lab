"""Figures for the paper and the README.

Usage:
    python scripts/make_figures.py

Writes paper/figures/cumulative_excess_returns.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fxlab.data import load_rates, load_spot
from fxlab.fx_returns import monthly_dataset

FIG_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"

# Reference categorical palette, slots 1-4 in fixed order (light mode)
SERIES = [
    ("EW", "Equally weighted (dollar factor)", "#2a78d6"),
    ("AUD", "AUD (high carry)", "#eb6834"),
    ("JPY", "JPY (low carry)", "#1baf7a"),
    ("CHF", "CHF (safe haven)", "#eda100"),
]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def cumulative_excess_returns(rx):
    rx = rx.assign(EW=rx.mean(axis=1))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for code, label, color in SERIES:
        value = (1 + rx[code].dropna()).cumprod()
        ax.plot(value.index, value, color=color, lw=1.6, label=label)
        ax.annotate(
            f"{code}  {value.iloc[-1]:.2f}",
            xy=(value.index[-1], value.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=TEXT,
        )

    ax.hlines(1, rx.index[0], rx.index[-1], color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_title(
        "Value of $1 in a rolled one-month forward against the US dollar",
        loc="left", fontsize=11, color=TEXT, pad=12,
    )
    ax.set_ylabel("Cumulative value ($)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=TEXT)
    ax.margins(x=0.08)
    fig.text(
        0.01, 0.01,
        "Monthly excess returns S(t+1)/F(t) - 1, forwards implied by CIP. Data: Federal Reserve H.10 and OECD via FRED.",
        fontsize=7.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main():
    _, _, rx = monthly_dataset(load_spot(), load_rates())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "cumulative_excess_returns.png"
    cumulative_excess_returns(rx).savefig(path, facecolor=SURFACE)
    print(f"Figure written to {path.relative_to(FIG_DIR.parents[1])}")


if __name__ == "__main__":
    main()
