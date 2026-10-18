"""Figures for the paper and the README.

Usage:
    python scripts/make_figures.py

Writes paper/figures/cumulative_excess_returns.png (light theme, used in the paper)
and cumulative_excess_returns_dark.png (shown on GitHub to readers in dark mode).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fxlab.data import load_rates, load_spot
from fxlab.fx_returns import monthly_dataset

FIG_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"

SERIES = [
    ("EW", "Equally weighted (dollar factor)"),
    ("AUD", "AUD (high carry)"),
    ("JPY", "JPY (low carry)"),
    ("CHF", "CHF (safe haven)"),
]
# Reference categorical palette, slots 1-4 in fixed order, with its light and dark steps
THEMES = {
    "light": {
        "colors": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
    },
    "dark": {
        "colors": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "surface": "#0d1117", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#30363d",
    },
}


def cumulative_excess_returns(rx, theme="light"):
    t = THEMES[theme]
    SURFACE, TEXT, MUTED, GRID = t["surface"], t["text"], t["muted"], t["grid"]
    rx = rx.assign(EW=rx.mean(axis=1))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for (code, label), color in zip(SERIES, t["colors"]):
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
    for theme, suffix in [("light", ""), ("dark", "_dark")]:
        path = FIG_DIR / f"cumulative_excess_returns{suffix}.png"
        fig = cumulative_excess_returns(rx, theme)
        fig.savefig(path, facecolor=fig.get_facecolor())
        print(f"Figure written to {path.relative_to(FIG_DIR.parents[1])}")


if __name__ == "__main__":
    main()
