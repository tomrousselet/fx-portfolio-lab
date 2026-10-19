"""Performance and risk measures of monthly return series (paper, section 4.5).

All functions take simple monthly returns. Mean and volatility are annualised
with 12 and sqrt(12); the Sharpe ratio assumes the returns are already in
excess of the risk-free rate (true for currency forward portfolios).
"""

import numpy as np
import pandas as pd

MONTHS = 12


def annualized_mean(r: pd.Series) -> float:
    return MONTHS * float(r.mean())


def annualized_vol(r: pd.Series) -> float:
    return np.sqrt(MONTHS) * float(r.std(ddof=1))


def sharpe_ratio(r: pd.Series) -> float:
    """Annualised mean / annualised volatility of excess returns."""
    return annualized_mean(r) / annualized_vol(r)


def t_stat(r: pd.Series) -> float:
    """t-statistic of the mean monthly return: mean / (std / sqrt(n)) = Sharpe x sqrt(years)."""
    r = r.dropna()
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def skewness(r: pd.Series) -> float:
    """Sample skewness (bias-adjusted); negative values signal crash risk."""
    return float(r.skew())


def excess_kurtosis(r: pd.Series) -> float:
    """Sample excess kurtosis (0 for a normal distribution); positive values mean fat tails."""
    return float(r.kurt())


def max_drawdown(r: pd.Series) -> float:
    """max_t [1 - V_t / max_{s<=t} V_s] for V the cumulative value, starting at 1 (course, section 2.2)."""
    value = pd.concat([pd.Series([1.0]), (1 + r.dropna()).cumprod()], ignore_index=True)
    return float((1 - value / value.cummax()).max())


def summary(returns: pd.Series, costs: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    """Standard table row for a strategy."""
    row = {
        "Mean (%/yr)": 100 * annualized_mean(returns),
        "Vol (%/yr)": 100 * annualized_vol(returns),
        "Sharpe": sharpe_ratio(returns),
        "t-stat": t_stat(returns),
        "Skewness": skewness(returns),
        "Excess kurtosis": excess_kurtosis(returns),
        "Max DD (%)": 100 * max_drawdown(returns),
    }
    if costs is not None:
        row["Costs (%/yr)"] = 100 * annualized_mean(costs)
    if turnover is not None:
        row["Turnover (x/yr)"] = MONTHS * float(turnover.mean())
    return row


def break_even_cost_scale(gross_mean: float, base_cost_mean: float) -> float:
    """Multiple of the base-case costs at which the mean net return falls to zero.

    Costs are proportional to the base case, so net mean ~= gross mean - k * base cost mean
    (exact up to the small effect of costs on the capital path).
    """
    if base_cost_mean <= 0:
        return np.inf
    return gross_mean / base_cost_mean
