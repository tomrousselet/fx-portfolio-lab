"""What moves the euro: regressions of EURUSD and EURCHF on the rates and risk channels (paper, section 5).

Usage:
    python scripts/euro_drivers.py

Writes paper/tables/euro_drivers.tex and paper/figures/euro_risk_beta(.png, _dark.png).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fxlab.data import load_rates, load_spot, load_yields
from fxlab.euro import euro_drivers
from fxlab.regression import ols, rolling_coef

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "paper" / "tables" / "euro_drivers.tex"
FIG_DIR = ROOT / "paper" / "figures"

# The Swiss National Bank capped the franc at 1.20 per euro from 6 September 2011 to
# 15 January 2015: EURCHF could not respond to euro risk, so these months are excluded
# in the last specification.
SNB_FLOOR = ("2011-09-01", "2015-01-01")
SPECS = [
    ("EURUSD", ["d_short_US", "d_BTP"], False),
    ("EURUSD", ["d_short_US", "d_10y_US", "d_BTP", "d_OAT"], False),
    ("EURCHF", ["d_short_CH", "d_BTP"], False),
    ("EURCHF", ["d_short_CH", "d_BTP"], True),
]
LABELS = {
    "const": "Constant",
    "d_short_US": r"$\Delta(i_{\mathrm{EUR}} - i_{\mathrm{USD}})$, 3M",
    "d_short_CH": r"$\Delta(i_{\mathrm{EUR}} - i_{\mathrm{CHF}})$, 3M",
    "d_10y_US": r"$\Delta(y_{\mathrm{Bund}} - y_{\mathrm{UST}})$, 10Y",
    "d_BTP": r"$\Delta$ BTP--Bund, 10Y",
    "d_OAT": r"$\Delta$ OAT--Bund, 10Y",
}
THEMES = {
    "light": {"colors": ["#2a78d6", "#eb6834"], "surface": "#fcfcfb", "text": "#0b0b0b",
              "muted": "#52514e", "grid": "#e4e3df", "band": "#ecebe7"},
    "dark": {"colors": ["#3987e5", "#d95926"], "surface": "#0d1117", "text": "#ffffff",
             "muted": "#c3c2b7", "grid": "#30363d", "band": "#1f242b"},
}
EPISODES = [("2010-04", "2012-08", "Euro area\ndebt crisis"), ("2018-05", "2018-12", "Italian\nbudget"),
            ("2020-03", "2020-05", "Covid"), ("2024-06", "2024-12", "French\ndissolution")]
FLOOR_LABEL = "SNB floor\n(EURCHF fixed)"


def without_snb_floor(d: pd.DataFrame) -> pd.DataFrame:
    return d[(d.index < SNB_FLOOR[0]) | (d.index > SNB_FLOOR[1])]


def regressions(d: pd.DataFrame):
    out = []
    for dep, regs, exclude_floor in SPECS:
        sample = without_snb_floor(d) if exclude_floor else d
        label = f"{dep} ex-floor" if exclude_floor else dep
        out.append((label, regs, ols(sample[dep], sample[regs])))
    return out


def to_latex(results) -> str:
    names = ["d_short_US", "d_short_CH", "d_10y_US", "d_BTP", "d_OAT", "const"]
    cols = len(results)
    lines = [
        r"\begin{tabular}{@{}l" + "r" * cols + "@{}}",
        r"\toprule",
        " & " + " & ".join(f"({k + 1}) {dep}" for k, (dep, _, _) in enumerate(results)) + r" \\",
        r"\midrule",
    ]
    for name in names:
        coefs, tstats = [], []
        for _, _, res in results:
            if name in res.coef:
                coefs.append(f"{res.coef[name]:.2f}")
                tstats.append(f"({res.tstat[name]:.1f})")
            else:
                coefs.append("")
                tstats.append("")
        lines.append(LABELS[name] + " & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(tstats) + r" \\[2pt]")
    lines.append(r"\midrule")
    lines.append(r"$R^2$ & " + " & ".join(f"{res.r2:.2f}" for _, _, res in results) + r" \\")
    lines.append(r"Months & " + " & ".join(str(res.nobs) for _, _, res in results) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def risk_beta_figure(d: pd.DataFrame, theme: str):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for start, end, label in EPISODES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=t["band"], lw=0)
        ax.text(pd.Timestamp(start), 0.98, label, transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=t["muted"], va="top")
    ax.axvspan(pd.Timestamp(SNB_FLOOR[0]), pd.Timestamp(SNB_FLOOR[1]), facecolor="none",
               edgecolor=t["muted"], hatch="///", lw=0, alpha=0.5)
    ax.text(pd.Timestamp("2013-01-01"), 0.02, FLOOR_LABEL, transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=t["muted"], va="bottom")
    for (dep, regs), color in zip([("EURUSD", ["d_short_US", "d_BTP"]), ("EURCHF", ["d_short_CH", "d_BTP"])],
                                  t["colors"]):
        beta = rolling_coef(d[dep], d[regs], "d_BTP", window=36) / 10  # per 10 bp of spread
        ax.plot(beta.index, beta, color=color, lw=1.6, label=dep)
    ax.axhline(0, color=t["muted"], lw=0.8, ls=(0, (3, 3)))
    ax.set_title("Euro move for a 10 bp widening of the BTP-Bund spread (rolling 36 months)",
                 loc="left", fontsize=11, color=t["text"], pad=12)
    ax.set_ylabel("% change of the euro", color=t["muted"], fontsize=9)
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=t["text"], ncol=2)
    ax.margins(x=0.06)
    fig.text(0.01, 0.01, "OLS of monthly euro changes on changes in the short-rate differential and the "
             "BTP-Bund spread, window ending at each date. Data: FRED (Federal Reserve, OECD).",
             fontsize=7.5, color=t["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main():
    d = euro_drivers(load_spot(), load_rates(), load_yields())
    results = regressions(d)
    for k, (dep, regs, res) in enumerate(results, 1):
        print(f"({k}) {dep} ~ {' + '.join(regs)}   n={res.nobs}  R2={res.r2:.3f}  NW lags={res.lags}")
        print(pd.DataFrame({"coef": res.coef, "t": res.tstat}).round(3).to_string(), "\n")
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(to_latex(results))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for theme, suffix in [("light", ""), ("dark", "_dark")]:
        fig = risk_beta_figure(d, theme)
        fig.savefig(FIG_DIR / f"euro_risk_beta{suffix}.png", facecolor=fig.get_facecolor())
    print("Table and figures written to paper/")


if __name__ == "__main__":
    main()
