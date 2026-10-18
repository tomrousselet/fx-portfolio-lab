"""Descriptive statistics of monthly currency excess returns (paper, Table 2).

Usage:
    python scripts/describe_returns.py

Prints the table and writes it to paper/tables/excess_returns.tex.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fxlab.data import load_rates, load_spot
from fxlab.fx_returns import carry, monthly_dataset

TABLE_PATH = Path(__file__).resolve().parents[1] / "paper" / "tables" / "excess_returns.tex"


def max_drawdown(returns: pd.Series) -> float:
    """max_t [1 - V_t / max_{s<=t} V_s] (course, section 2.2)."""
    value = (1 + returns.dropna()).cumprod()
    return float((1 - value / value.cummax()).max())


def describe(rx: pd.DataFrame, carry_me: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name in rx.columns:
        r = rx[name].dropna()
        rows[name] = {
            "First month": r.index[0].strftime("%Y-%m"),
            "Last month": r.index[-1].strftime("%Y-%m"),
            "Months": len(r),
            "Mean (%/yr)": 100 * 12 * r.mean(),
            "Carry (%/yr)": 100 * 12 * carry_me[name].shift(1).loc[r.index].mean()
            if name in carry_me
            else np.nan,
            "Vol (%/yr)": 100 * np.sqrt(12) * r.std(),
            "Sharpe": np.sqrt(12) * r.mean() / r.std(),
            "Worst month (%)": 100 * r.min(),
            "Max DD (%)": 100 * max_drawdown(r),
        }
    return pd.DataFrame(rows).T


def to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}lllrrrrrrr@{}}",
        r"\toprule",
        r" & First & Last & Months & Mean & Carry & Vol & Sharpe & Worst & Max DD \\",
        r" & month & month & & (\%/yr) & (\%/yr) & (\%/yr) & & month (\%) & (\%) \\",
        r"\midrule",
    ]
    for name, row in table.iterrows():
        if name == "EW":
            lines.append(r"\midrule")
        carry_str = "--" if pd.isna(row["Carry (%/yr)"]) else f"{row['Carry (%/yr)']:.2f}"
        lines.append(
            f"{name} & {row['First month']} & {row['Last month']} & {row['Months']} & "
            f"{row['Mean (%/yr)']:.2f} & {carry_str} & {row['Vol (%/yr)']:.2f} & "
            f"{row['Sharpe']:.2f} & {row['Worst month (%)']:.2f} & {row['Max DD (%)']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    spot, rates = load_spot(), load_rates()
    spot_me, rates_me, rx = monthly_dataset(spot, rates)
    carry_me = carry(spot_me, rates_me)

    # Equally weighted portfolio of the available currencies, rebalanced monthly:
    # by the fundamental property (course, eq. r_p = w'r) its return is the cross-sectional mean.
    rx_with_ew = rx.assign(EW=rx.mean(axis=1))
    table = describe(rx_with_ew, carry_me)

    pd.set_option("display.width", 140)
    print(table.round(2).to_string())
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(to_latex(table))
    print(f"\nLaTeX table written to {TABLE_PATH.relative_to(TABLE_PATH.parents[2])}")


if __name__ == "__main__":
    main()
