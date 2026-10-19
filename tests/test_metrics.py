"""Tests of the performance measures on series computed by hand."""

import numpy as np
import pandas as pd
import pytest

from fxlab import metrics

R = pd.Series([0.01, -0.02, 0.03, 0.00])  # mean 0.005, sample std 0.0208167


def test_annualized_mean_and_vol():
    assert metrics.annualized_mean(R) == pytest.approx(0.06)
    assert metrics.annualized_vol(R) == pytest.approx(np.sqrt(12) * 0.02081666, rel=1e-6)


def test_sharpe_and_t_stat():
    """t-stat = Sharpe x sqrt(years): here 4 months = 1/3 year."""
    assert metrics.sharpe_ratio(R) == pytest.approx(0.06 / (np.sqrt(12) * 0.02081666), rel=1e-6)
    assert metrics.t_stat(R) == pytest.approx(metrics.sharpe_ratio(R) * np.sqrt(4 / 12))


def test_max_drawdown_by_hand():
    """Values 1.10, 0.88, 0.968, 1.1616: the trough 0.88 is 20% below the peak 1.10."""
    r = pd.Series([0.10, -0.20, 0.10, 0.20])
    assert metrics.max_drawdown(r) == pytest.approx(0.20)


def test_max_drawdown_counts_a_first_month_loss():
    """The starting value 1 is a peak: losing 5% in month one is a 5% drawdown."""
    assert metrics.max_drawdown(pd.Series([-0.05, 0.01])) == pytest.approx(0.05)


def test_skewness_sign():
    """Frequent small gains and a rare large loss: the carry-trade profile."""
    crashy = pd.Series([0.01] * 19 + [-0.15])
    assert metrics.skewness(crashy) < 0
    assert metrics.skewness(-crashy) > 0


def test_normal_sample_has_no_excess_kurtosis():
    rng = np.random.default_rng(0)
    assert abs(metrics.excess_kurtosis(pd.Series(rng.normal(size=100_000)))) < 0.05


def test_break_even_cost_scale():
    """Gross 2%/yr, base-case costs 0.5%/yr: costs can be 4x the base case before the mean is zero."""
    assert metrics.break_even_cost_scale(0.02, 0.005) == pytest.approx(4.0)
    assert metrics.break_even_cost_scale(0.02, 0.0) == np.inf
