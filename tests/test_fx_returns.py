"""Tests of the currency excess return calculations."""

import numpy as np
import pandas as pd
import pytest

from fxlab.fx_returns import (
    carry,
    excess_returns,
    forward_rates,
    month_end_rates,
    month_end_spot,
)

MONTH_ENDS = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])


@pytest.fixture
def spot_me():
    return pd.DataFrame({"EUR": [1.10, 1.12, 1.11]}, index=MONTH_ENDS)


@pytest.fixture
def rates_me():
    return pd.DataFrame({"USD": [0.05, 0.05, 0.05], "EUR": [0.03, 0.03, 0.03]}, index=MONTH_ENDS)


def test_forward_by_cip(spot_me, rates_me):
    """F = S (1 + i/12) / (1 + i*/12): the higher-yielding USD trades at a forward discount."""
    f = forward_rates(spot_me, rates_me).iloc[0, 0]
    assert f == pytest.approx(1.10 * (1 + 0.05 / 12) / (1 + 0.03 / 12))
    assert f > 1.10


def test_excess_return_by_hand(spot_me, rates_me):
    rx = excess_returns(spot_me, rates_me)
    f0 = 1.10 * (1 + 0.05 / 12) / (1 + 0.03 / 12)
    assert np.isnan(rx.iloc[0, 0])  # no position before the first month
    assert rx.iloc[1, 0] == pytest.approx(1.12 / f0 - 1)


def test_equal_rates_give_spot_return(spot_me):
    """With equal rates there is no carry: rx equals the spot return."""
    rates = pd.DataFrame({"USD": 0.04, "EUR": 0.04}, index=MONTH_ENDS)
    rx = excess_returns(spot_me, rates)
    assert rx.iloc[1, 0] == pytest.approx(1.12 / 1.10 - 1)


def test_carry_approximates_rate_differential(spot_me, rates_me):
    """Carry is about (i* - i) / 12; negative here because the EUR yields less than the USD."""
    c = carry(spot_me, rates_me).iloc[0, 0]
    assert c == pytest.approx((0.03 - 0.05) / 12, abs=1e-5)


def test_no_look_ahead(spot_me, rates_me):
    """rx_{t+1} must not depend on rates observed at t+1 or on spots after t+1."""
    base = excess_returns(spot_me, rates_me).iloc[1, 0]
    rates_later = rates_me.copy()
    rates_later.iloc[1:] = 0.20
    spot_later = spot_me.copy()
    spot_later.iloc[2] = 5.0
    assert excess_returns(spot_later, rates_later).iloc[1, 0] == pytest.approx(base)


def test_month_end_spot_uses_last_available_day():
    """31 May 2024 missing (holiday-like gap): the month-end rate is the 30 May rate."""
    daily = pd.DataFrame(
        {"EUR": [1.08, 1.09, np.nan, 1.07]},
        index=pd.to_datetime(["2024-05-29", "2024-05-30", "2024-05-31", "2024-06-28"]),
    )
    monthly = month_end_spot(daily)
    assert monthly.loc["2024-05-31", "EUR"] == 1.09


def test_incomplete_last_month_dropped():
    daily = pd.DataFrame(
        {"EUR": [1.08, 1.10]}, index=pd.to_datetime(["2024-05-31", "2024-06-14"])
    )
    monthly = month_end_spot(daily)
    assert list(monthly.index) == [pd.Timestamp("2024-05-31")]


def test_rates_dated_at_month_end():
    rates = pd.DataFrame({"USD": [0.05]}, index=pd.to_datetime(["2024-02-01"]))
    assert month_end_rates(rates).index[0] == pd.Timestamp("2024-02-29")
