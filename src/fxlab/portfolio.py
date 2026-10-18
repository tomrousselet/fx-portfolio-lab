"""Portfolio valuation: returns, buy-and-hold drift and rebalanced portfolios.

Implements Part 1 of the course (Milhau, Portfolio Construction, sessions 1-4).
Notation follows the course: theta = dollar amounts, w = weights, V = value.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns r_{t,t+1} = S_{t+1} / S_t - 1 (prices must be adjusted)."""
    return prices.pct_change().dropna(how="all")


def compound(returns) -> float:
    """Multi-period return: prod(1 + r_t) - 1. Returns compound, they do not add up."""
    return float(np.prod(1 + np.asarray(returns)) - 1)


def annualized_return(total_return: float, years: float) -> float:
    """(V_T / V_0)^(1 / years) - 1."""
    return (1 + total_return) ** (1 / years) - 1


@dataclass
class BacktestResult:
    """Output of a portfolio valuation.

    values:   portfolio value V_t after operations at t
    amounts:  amounts theta_{t,i} after operations at t
    weights:  weights w_{t,i} after operations at t
    traded:   total amount bought + sold at each date (turnover in dollars)
    """

    values: pd.Series
    amounts: pd.DataFrame
    weights: pd.DataFrame
    traded: pd.Series

    @property
    def returns(self) -> pd.Series:
        return self.values.pct_change().dropna()


def backtest(
    returns: pd.DataFrame,
    target_weights,
    rebalance_dates=None,
    initial_value: float = 1.0,
) -> BacktestResult:
    """Value a fixed-mix portfolio rebalanced at given dates (course section 1.8).

    Between two rebalancing dates the portfolio is buy-and-hold, so each amount
    grows with its own return: theta_{[t+1]-,i} = theta_{t,i} (1 + r_{t,t+1,i}).
    On a rebalancing date the amounts are reset to theta_{t,i} = w_i V_t.

    returns:          simple returns, one row per date t+1 (first row = first period)
    target_weights:   weights w_i (summing to 1) set at the start and on each rebalancing
    rebalance_dates:  dates (index labels of `returns`) where weights are reset;
                      None = buy-and-hold, "all" = rebalanced at every date
    initial_value:    V_0
    """
    w = np.asarray(target_weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"target weights must sum to 1, got {w.sum():.6f}")

    if rebalance_dates is None:
        rebalance = set()
    elif isinstance(rebalance_dates, str) and rebalance_dates == "all":
        rebalance = set(returns.index)
    else:
        rebalance = set(rebalance_dates)

    # Date 0 sits one step before the first return; label it with a placeholder
    # only if the index is not a DatetimeIndex.
    dates = [_previous_label(returns.index)] + list(returns.index)
    theta = w * initial_value
    amounts = [theta]
    traded = [0.0]

    for date, r in zip(returns.index, returns.to_numpy()):
        before = theta * (1 + r)  # buy-and-hold over (t, t+1)
        if date in rebalance:
            after = w * before.sum()
            traded.append(float(np.abs(after - before).sum()))
            theta = after
        else:
            traded.append(0.0)
            theta = before
        amounts.append(theta)

    amounts = pd.DataFrame(amounts, index=dates, columns=returns.columns)
    values = amounts.sum(axis=1)
    return BacktestResult(
        values=values,
        amounts=amounts,
        weights=amounts.div(values, axis=0),
        traded=pd.Series(traded, index=dates),
    )


def _previous_label(index: pd.Index):
    if isinstance(index, pd.DatetimeIndex) and index.freq is not None:
        return index[0] - index.freq
    if isinstance(index, pd.DatetimeIndex):
        return index[0] - pd.Timedelta(days=1)
    return "t0"
