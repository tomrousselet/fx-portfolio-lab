"""Monthly currency excess returns from spot rates and interest rates.

Convention (paper, section 2.3): S_t is in USD per unit of foreign currency,
i_t is the USD rate and i*_t the foreign rate, both in decimal per year.
A long position in a one-month forward opened at the end of month t earns

    rx_{t+1} = S_{t+1} / F_t - 1,   with   F_t = S_t (1 + i_t / 12) / (1 + i*_t / 12)  (CIP).

Timing rule (no look-ahead): everything used to open the position at t
(S_t, i_t, i*_t) is known at the end of month t; rx_{t+1} is indexed at t+1.
"""

import pandas as pd

MONTHS_PER_YEAR = 12


def month_end_spot(spot: pd.DataFrame) -> pd.DataFrame:
    """Last available daily observation of each month, indexed at the calendar month end.

    The last month is dropped if it is incomplete (data stop before its last
    business day), because its "month-end" rate would be a mid-month rate.
    """
    spot = spot.dropna(how="all")
    monthly = spot.resample("ME").last()
    last_obs = spot.index[-1]
    if last_obs < last_obs + pd.offsets.BMonthEnd(0):
        monthly = monthly.iloc[:-1]
    return monthly


def month_end_rates(rates: pd.DataFrame) -> pd.DataFrame:
    """Re-index monthly rates (FRED dates them on the 1st) to the end of the same month.

    An OECD monthly rate is an average over the month, so it is only known at
    the end of that month: dating it at the month end makes the timing explicit.
    """
    rates = rates.copy()
    rates.index = rates.index + pd.offsets.MonthEnd(0)
    return rates


def forward_rates(spot_me: pd.DataFrame, rates_me: pd.DataFrame, base: str = "USD") -> pd.DataFrame:
    """One-month forward rates implied by covered interest parity."""
    foreign = spot_me.columns
    i_base = rates_me[base].reindex(spot_me.index)
    i_fcy = rates_me[foreign].reindex(spot_me.index)
    growth_base = 1 + i_base / MONTHS_PER_YEAR
    growth_fcy = 1 + i_fcy / MONTHS_PER_YEAR
    return spot_me.mul(growth_base, axis=0) / growth_fcy


def excess_returns(spot_me: pd.DataFrame, rates_me: pd.DataFrame, base: str = "USD") -> pd.DataFrame:
    """Monthly excess returns rx_{t+1} = S_{t+1} / F_t - 1, indexed at t+1.

    NaN wherever S_t, S_{t+1}, i_t or i*_t is missing: a currency enters the
    panel as soon as all four are available (unbalanced panel).
    """
    fwd = forward_rates(spot_me, rates_me, base)
    return spot_me / fwd.shift(1) - 1


def carry(spot_me: pd.DataFrame, rates_me: pd.DataFrame, base: str = "USD") -> pd.DataFrame:
    """Forward discount S_t / F_t - 1, known at t: the excess return earned if the spot does not move."""
    return spot_me / forward_rates(spot_me, rates_me, base) - 1


def monthly_dataset(spot: pd.DataFrame, rates: pd.DataFrame, start: str = "1999-01", base: str = "USD"):
    """Month-end spot, rates and excess returns from the daily/monthly FRED data, from `start`."""
    spot_me = month_end_spot(spot).loc[start:]
    rates_me = month_end_rates(rates).reindex(spot_me.index)
    rx = excess_returns(spot_me, rates_me, base).iloc[1:]
    return spot_me, rates_me, rx
