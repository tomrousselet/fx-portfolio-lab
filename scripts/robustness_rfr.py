"""Robustness check: CIP forwards built from overnight risk-free rates instead of 3-month interbank rates.

Usage:
    python scripts/robustness_rfr.py

Since 2018-2019 the market references are secured or overnight risk-free rates
(SOFR, ESTR, SONIA). For EUR and GBP against USD, this script recomputes carry and
excess returns with their monthly averages and compares them with the baseline
(decision D16). Writes paper/tables/rfr_robustness.tex.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fxlab.data import load_rates, load_rfr_rates, load_spot
from fxlab.fx_returns import carry, excess_returns, month_end_rates, month_end_spot

TABLE_PATH = Path(__file__).resolve().parents[1] / "paper" / "tables" / "rfr_robustness.tex"
START = "2019-10"  # first month of ESTR; SOFR starts in April 2018
END = "2026-01"  # last month of the OECD rates: after it, GBP is itself extended with SONIA


def run():
    spot_me = month_end_spot(load_spot()[["EUR", "GBP"]]).loc[START:END]
    variants = {
        "3M interbank": month_end_rates(load_rates()).reindex(spot_me.index),
        "Overnight RFR": month_end_rates(load_rfr_rates()).reindex(spot_me.index),
    }
    rx = {k: excess_returns(spot_me, r).iloc[1:] for k, r in variants.items()}
    cy = {k: carry(spot_me, r).shift(1).iloc[1:] for k, r in variants.items()}
    common = rx["3M interbank"].dropna().index.intersection(rx["Overnight RFR"].dropna().index)

    rows = {}
    for ccy in ["EUR", "GBP"]:
        a, b = rx["3M interbank"].loc[common, ccy], rx["Overnight RFR"].loc[common, ccy]
        ca, cb = cy["3M interbank"].loc[common, ccy], cy["Overnight RFR"].loc[common, ccy]
        rows[ccy] = {
            "Months": len(common),
            "Carry interbank (%/yr)": 1200 * ca.mean(),
            "Carry RFR (%/yr)": 1200 * cb.mean(),
            "Mean rx interbank (%/yr)": 1200 * a.mean(),
            "Mean rx RFR (%/yr)": 1200 * b.mean(),
            "Mean |difference| (bp/month)": 1e4 * (a - b).abs().mean(),
            "Correlation": float(np.corrcoef(a, b)[0, 1]),
        }
    return pd.DataFrame(rows).T, common


def to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r" & Months & \multicolumn{2}{c}{Carry (\%/yr)} & \multicolumn{2}{c}{Mean $rx$ (\%/yr)} & Mean $|\Delta rx|$ & Corr. \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & Interbank & RFR & Interbank & RFR & (bp/month) & \\",
        r"\midrule",
    ]
    for ccy, r in table.iterrows():
        lines.append(
            f"{ccy} & {int(r['Months'])} & {r['Carry interbank (%/yr)']:.2f} & {r['Carry RFR (%/yr)']:.2f} & "
            f"{r['Mean rx interbank (%/yr)']:.2f} & {r['Mean rx RFR (%/yr)']:.2f} & "
            f"{r['Mean |difference| (bp/month)']:.1f} & {r['Correlation']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    table, common = run()
    pd.set_option("display.width", 160)
    print(f"Sample: {common[0].date()} to {common[-1].date()}")
    print(table.round(4).to_string())
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(to_latex(table))
    print(f"LaTeX table written to {TABLE_PATH.relative_to(TABLE_PATH.parents[2])}")


if __name__ == "__main__":
    main()
