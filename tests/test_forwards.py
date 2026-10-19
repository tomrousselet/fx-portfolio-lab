"""Tests of the forward portfolio backtest."""

import numpy as np
import pandas as pd
import pytest

from fxlab.costs import BP, CostModel
from fxlab.forwards import backtest_forwards, equal_weights

GRID = pd.date_range("2024-01-31", periods=4, freq="ME")  # formation dates t
RX = pd.DataFrame(
    {"AUD": [np.nan, 0.02, -0.01, 0.03], "JPY": [np.nan, -0.01, 0.00, 0.01]}, index=GRID
).iloc[1:]  # rx_{t+1} exists from the second month on
FLAT = CostModel(outright={"AUD": 2 * BP, "JPY": 2 * BP}, swap={"AUD": 1 * BP, "JPY": 1 * BP})


def constant(weights):
    return pd.DataFrame([weights] * len(GRID), index=GRID)


def test_single_long_position_earns_rx():
    res = backtest_forwards(RX, constant({"AUD": 1.0, "JPY": 0.0}))
    assert np.allclose(res.returns, RX["AUD"])
    assert res.values.iloc[-1] == pytest.approx(1.02 * 0.99 * 1.03)


def test_long_short_portfolio_return_is_weighted_sum():
    """Weights summing to zero are allowed: r_p = w' rx (course fundamental property)."""
    res = backtest_forwards(RX, constant({"AUD": 1.0, "JPY": -1.0}))
    assert np.allclose(res.returns, RX["AUD"] - RX["JPY"])


def test_collateral_earns_risk_free_rate():
    rf = pd.Series(0.06, index=GRID)
    res = backtest_forwards(RX, constant({"AUD": 1.0, "JPY": 0.0}), rf=rf)
    assert np.allclose(res.returns, RX["AUD"] + 0.06 / 12)


def test_costs_by_hand():
    """Constant 100% AUD: month 1 opens outright (2 bp), later months only roll (1 bp)."""
    res = backtest_forwards(RX, constant({"AUD": 1.0, "JPY": 0.0}), costs=FLAT)
    assert res.costs.iloc[0] == pytest.approx(2 * BP)
    v1 = 1 - 2 * BP + 0.02
    # month 2: roll the 1.0 notional opened with V0 = 1, resize to v1 (outright on the change)
    expected_cost_2 = (1.0 * 1 * BP + abs(v1 - 1.0) * 2 * BP) / v1
    assert res.costs.iloc[1] == pytest.approx(expected_cost_2)
    assert np.allclose(res.returns, res.gross_returns - res.costs)


def test_turnover_of_a_flip():
    """Long AUD then short AUD: 1 opened, then 2 traded (close 1, open 1)."""
    w = pd.DataFrame({"AUD": [1.0, -1.0, -1.0, -1.0], "JPY": 0.0}, index=GRID)
    res = backtest_forwards(RX, w)
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    v1 = 1.02
    assert res.turnover.iloc[1] == pytest.approx((1.0 + v1) / v1)


def test_no_look_ahead():
    """Weights decided at the last date are never used, and weights at t only affect month t+1."""
    w = constant({"AUD": 1.0, "JPY": 0.0})
    base = backtest_forwards(RX, w).returns
    w_changed = w.copy()
    w_changed.iloc[-1] = [-5.0, 5.0]
    assert np.allclose(backtest_forwards(RX, w_changed).returns, base)
    w_changed.iloc[1] = [0.0, 1.0]  # decided at the 2nd date: changes only the 2nd holding month
    changed = backtest_forwards(RX, w_changed).returns
    assert changed.iloc[0] == pytest.approx(base.iloc[0])
    assert changed.iloc[1] == pytest.approx(RX["JPY"].iloc[1])


def test_position_without_return_raises():
    rx = RX.copy()
    rx.iloc[1, 1] = np.nan
    with pytest.raises(ValueError):
        backtest_forwards(rx, constant({"AUD": 0.5, "JPY": 0.5}))


def test_equal_weights_follow_availability():
    available = pd.DataFrame({"A": [True, True], "B": [True, False], "C": [False, False]})
    w = equal_weights(available)
    assert w.iloc[0].tolist() == [0.5, 0.5, 0.0]
    assert w.iloc[1].tolist() == [1.0, 0.0, 0.0]
