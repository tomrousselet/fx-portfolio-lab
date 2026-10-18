"""Tests built from the worked examples of the course summary (sessions 1-4).

Each test cites the example it reproduces, so the course is the specification.
"""

import numpy as np
import pandas as pd
import pytest

from fxlab.portfolio import annualized_return, backtest, compound, simple_returns

STOCKS = ["MSFT", "GE", "AAPL", "JPM", "V"]


@pytest.fixture
def table1_prices():
    """Course Table 1: prices on 2 and 5 January 2015 (rounded to the cent)."""
    return pd.DataFrame(
        [[46.76, 25.06, 109.33, 62.49, 265.02], [46.33, 24.60, 106.25, 60.55, 259.17]],
        index=pd.to_datetime(["2015-01-02", "2015-01-05"]),
        columns=STOCKS,
    )


def test_table1_buy_and_hold(table1_prices):
    """$1,000 equally weighted -> V1 = $978.21, weights drift, return -2.18%."""
    r = simple_returns(table1_prices)
    res = backtest(r, np.full(5, 0.2), initial_value=1000)
    # Course figures use full-precision prices; ours are rounded, hence the tolerance.
    assert res.values.iloc[-1] == pytest.approx(978.21, abs=0.1)
    assert res.returns.iloc[-1] == pytest.approx(-0.0218, abs=1e-4)
    assert res.weights.iloc[-1]["MSFT"] == pytest.approx(0.2026, abs=2e-4)
    # The fundamental property: r_p = w0' r
    assert res.returns.iloc[-1] == pytest.approx(0.2 * r.iloc[0].sum())


def test_table1_rebalancing_back_to_equal_weights(table1_prices):
    """Returning to 20% weights requires trading $6.37 in total."""
    r = simple_returns(table1_prices)
    res = backtest(r, np.full(5, 0.2), rebalance_dates="all", initial_value=1000)
    assert res.traded.iloc[-1] == pytest.approx(6.37, abs=0.1)
    assert np.allclose(res.weights.iloc[-1], 0.2)


def test_midterm_q1_amount_update():
    """Q1: 23% of $150 in a stock returning 5.2% becomes $36.29."""
    r = pd.DataFrame([[0.052, 0.0]], columns=["A", "Other"])
    res = backtest(r, [0.23, 0.77], initial_value=150)
    assert res.amounts.iloc[-1]["A"] == pytest.approx(36.29, abs=0.01)


def test_midterm_q4_weight_drift():
    """Q4: $65 in A (+3%) and $22 in B (+2%) -> weight of A goes to 74.9%."""
    r = pd.DataFrame([[0.03, 0.02]], columns=["A", "B"])
    res = backtest(r, [65 / 87, 22 / 87], initial_value=87)
    assert res.weights.iloc[-1]["A"] == pytest.approx(0.749, abs=5e-4)


def test_midterm_q7_compounding():
    """Q7: +5% then -4% is +0.8%, not +1%."""
    assert compound([0.05, -0.04]) == pytest.approx(0.008)


def test_annualized_return():
    """Doubling over 10 years is about 7.18% a year."""
    assert annualized_return(1.0, 10) == pytest.approx(0.071773, abs=1e-6)


def test_single_rebalance_equals_buy_and_hold():
    """Course 1.8: rebalancing only at date 0 gives buy-and-hold."""
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (50, 3)), columns=list("ABC"))
    bh = backtest(r, [0.5, 0.3, 0.2])
    reb = backtest(r, [0.5, 0.3, 0.2], rebalance_dates=[])
    assert np.allclose(bh.values, reb.values)


def test_weights_must_sum_to_one():
    r = pd.DataFrame([[0.01, 0.02]], columns=["A", "B"])
    with pytest.raises(ValueError):
        backtest(r, [0.5, 0.6])
