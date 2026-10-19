"""Backtest of portfolios of one-month currency forwards (paper, section 4.3).

The capital V is held as collateral and earns the US rate (optional); forward
positions require no capital. At each month end t:

    1. target notionals are set from the weights decided at t:  N_t = w_t V_t
       (weights are notional / capital, they need not sum to one: a long-short
       portfolio has weights summing to zero);
    2. moving from the previous notionals to N_t costs swap and outright
       half-spreads (costs.rebalancing_cost), paid immediately;
    3. over month t+1 the forwards earn N_t * rx_{t+1} and the collateral earns
       V i_t / 12; the forwards are then rolled.

This is the amount-propagation method of fxlab.portfolio (decision D4) applied
to notionals: the cost of every month follows from the amounts traded.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxlab.costs import CostModel, rebalancing_cost


@dataclass
class ForwardBacktestResult:
    """Monthly output, indexed by the end of each holding month (t+1).

    values:         capital V after the month
    returns:        net return of the capital over the month (excess return if no rf given)
    gross_returns:  forward P&L / capital, before costs and collateral interest
    costs:          transaction costs / capital
    turnover:       sum |N_t - N_{t-1}| / capital (outright notional traded)
    notionals:      forward notionals N_t held over the month
    """

    values: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    notionals: pd.DataFrame


def backtest_forwards(
    rx: pd.DataFrame,
    weights: pd.DataFrame,
    costs: CostModel | None = None,
    rf: pd.Series | None = None,
    initial_value: float = 1.0,
) -> ForwardBacktestResult:
    """Value a portfolio of rolled one-month forwards.

    rx:       monthly excess returns rx_{t+1}, indexed at t+1 (fx_returns.excess_returns)
    weights:  target weights decided at month end t, indexed at t, on the same monthly
              grid as rx (it must contain the month before the first rx date).
              NaN means no position.
    costs:    transaction cost model; None for no costs
    rf:       annual US rate indexed at t (decimal); if given, the collateral earns rf/12
              and `returns` are total returns, otherwise they are excess returns
    """
    weights = weights.reindex(columns=rx.columns).fillna(0.0)
    held = weights.shift(1).reindex(rx.index).fillna(0.0)  # weights decided at t, held over t+1
    if rf is not None:
        rf = rf.shift(1).reindex(rx.index)

    traded_with_no_return = (held != 0) & rx.isna()
    if traded_with_no_return.to_numpy().any():
        date, ccy = traded_with_no_return.stack().loc[lambda s: s].index[0]
        raise ValueError(f"position in {ccy} held over {date.date()} but its excess return is missing")

    value = initial_value
    previous = pd.Series(0.0, index=rx.columns)
    rows = []
    for date in rx.index:
        notional = held.loc[date] * value
        cost = rebalancing_cost(previous, notional, costs) if costs is not None else 0.0
        turnover = float((notional - previous).abs().sum())
        pnl = float((notional * rx.loc[date].fillna(0.0)).sum())
        interest = value * rf.loc[date] / 12 if rf is not None else 0.0
        new_value = value - cost + pnl + interest
        rows.append({
            "value": new_value,
            "return": new_value / value - 1,
            "gross": pnl / value,
            "cost": cost / value,
            "turnover": turnover / value,
        })
        value, previous = new_value, notional

    out = pd.DataFrame(rows, index=rx.index)
    return ForwardBacktestResult(
        values=out["value"],
        returns=out["return"],
        gross_returns=out["gross"],
        costs=out["cost"],
        turnover=out["turnover"],
        notionals=held.mul(out["value"].shift(1, fill_value=initial_value), axis=0),
    )


def equal_weights(available: pd.DataFrame) -> pd.DataFrame:
    """Long 1/N in each currency available at t (the dollar factor of Lustig et al., 2011)."""
    n = available.sum(axis=1).replace(0, np.nan)
    return available.astype(float).div(n, axis=0)
