"""Backtest of the equally weighted currency basket (dollar factor) under several cost scenarios.

Usage:
    python scripts/backtest_benchmarks.py

Prints the table and writes it to paper/tables/dollar_factor_costs.tex.
This is the first end-to-end use of the engine: data -> excess returns ->
weights -> forward backtest with costs -> performance measures.
"""

from pathlib import Path

import pandas as pd

from fxlab.costs import BASE_CASE
from fxlab.data import load_rates, load_spot
from fxlab.forwards import backtest_forwards, equal_weights
from fxlab.fx_returns import carry, monthly_dataset
from fxlab.metrics import annualized_mean, break_even_cost_scale, summary

TABLE_PATH = Path(__file__).resolve().parents[1] / "paper" / "tables" / "dollar_factor_costs.tex"
SCALES = [0, 1, 2, 5]
COLUMNS = ["Mean (%/yr)", "Vol (%/yr)", "Sharpe", "t-stat", "Skewness", "Max DD (%)", "Costs (%/yr)", "Turnover (x/yr)"]


def run():
    spot_me, rates_me, rx = monthly_dataset(load_spot(), load_rates())
    # A currency can be traded at t if its spot and both rates are known at t (carry defined).
    available = carry(spot_me, rates_me).notna()
    weights = equal_weights(available)

    rows, results = {}, {}
    for k in SCALES:
        res = backtest_forwards(rx, weights, costs=BASE_CASE.scaled(k))
        results[k] = res
        rows[f"{k} x base case"] = summary(res.returns, res.costs, res.turnover)
    table = pd.DataFrame(rows).T[COLUMNS]
    base = results[1]
    breakeven = break_even_cost_scale(annualized_mean(base.gross_returns), annualized_mean(base.costs))
    return table, breakeven


def fmt(x: float, digits: int = 2) -> str:
    """Fixed-point format that never prints a negative zero."""
    s = f"{x:.{digits}f}"
    return s[1:] if float(s) == 0 and s.startswith("-") else s


def to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Costs & Mean & Vol & Sharpe & $t$-stat & Skew. & Max DD & Costs & Turnover \\",
        r" & (\%/yr) & (\%/yr) & & & & (\%) & (\%/yr) & ($\times$/yr) \\",
        r"\midrule",
    ]
    for name, row in table.iterrows():
        label = name.replace(" x ", r" $\times$ ")
        lines.append(
            f"{label} & {fmt(row['Mean (%/yr)'])} & {fmt(row['Vol (%/yr)'])} & "
            f"{fmt(row['Sharpe'])} & {fmt(row['t-stat'])} & {fmt(row['Skewness'])} & {fmt(row['Max DD (%)'], 1)} & "
            f"{fmt(row['Costs (%/yr)'], 3)} & {fmt(row['Turnover (x/yr)'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    table, breakeven = run()
    pd.set_option("display.width", 160)
    print(table.round(3).to_string())
    print(f"\nBreak-even cost scale: {breakeven:.1f} x base case")
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(to_latex(table))
    (TABLE_PATH.parent / "dollar_factor_breakeven.tex").write_text(f"{breakeven:.1f}")
    print(f"LaTeX table written to {TABLE_PATH.relative_to(TABLE_PATH.parents[2])}")


if __name__ == "__main__":
    main()
