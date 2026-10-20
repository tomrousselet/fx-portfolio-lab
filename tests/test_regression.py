"""Tests of OLS with Newey-West standard errors and of the euro drivers dataset."""

import numpy as np
import pandas as pd
import pytest

from fxlab.euro import euro_drivers
from fxlab.regression import newey_west_lags, ols, rolling_coef

rng = np.random.default_rng(42)
N = 5000
X = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
Y = 0.5 + 2.0 * X["a"] - 1.0 * X["b"] + rng.normal(scale=0.5, size=N)


def test_coefficients_recovered():
    res = ols(Y, X)
    assert res.coef["const"] == pytest.approx(0.5, abs=0.03)
    assert res.coef["a"] == pytest.approx(2.0, abs=0.03)
    assert res.coef["b"] == pytest.approx(-1.0, abs=0.03)


def test_exact_fit_has_r2_one():
    res = ols(1 + 3 * X["a"], X[["a"]])
    assert res.r2 == pytest.approx(1.0)
    assert res.coef["a"] == pytest.approx(3.0)


def test_white_errors_match_formula():
    """lags=0: Var(beta) = (X'X)^-1 (sum u_t^2 x_t x_t') (X'X)^-1, computed independently."""
    res = ols(Y, X, lags=0)
    Xv = np.column_stack([np.ones(N), X.to_numpy()])
    beta = np.linalg.lstsq(Xv, Y.to_numpy(), rcond=None)[0]
    u = Y.to_numpy() - Xv @ beta
    bread = np.linalg.inv(Xv.T @ Xv)
    meat = (Xv * u[:, None] ** 2).T @ Xv
    assert np.allclose(res.se.to_numpy(), np.sqrt(np.diag(bread @ meat @ bread)))


def test_hac_close_to_classical_for_iid_errors():
    """With i.i.d. homoskedastic errors, robust and classical standard errors agree."""
    res = ols(Y, X)
    classical = 0.5 / np.sqrt(N)  # sigma / sqrt(n) for a regressor of unit variance
    assert res.se["a"] == pytest.approx(classical, rel=0.1)


def test_hac_wider_with_autocorrelated_errors():
    """AR(1) regressor and errors: Newey-West standard errors exceed White's."""
    e = np.zeros(N)
    x = np.zeros(N)
    for t in range(1, N):
        e[t] = 0.8 * e[t - 1] + rng.normal()
        x[t] = 0.8 * x[t - 1] + rng.normal()
    y = pd.Series(x + e)
    xs = pd.DataFrame({"x": x})
    assert ols(y, xs, lags=12).se["x"] > 1.5 * ols(y, xs, lags=0).se["x"]


def test_newey_west_lag_rule():
    assert newey_west_lags(100) == 4
    assert newey_west_lags(330) == 5


def test_rolling_coefficient_constant_relation():
    rc = rolling_coef(1 + 3 * X["a"].iloc[:60], X[["a"]].iloc[:60], "a", window=36)
    assert len(rc) == 25
    assert np.allclose(rc, 3.0)


def test_euro_drivers_by_hand():
    """Daily spots averaged by month; changes of differentials and spreads in percentage points."""
    days = pd.to_datetime(["2024-01-10", "2024-01-20", "2024-02-10", "2024-02-20"])
    spot = pd.DataFrame({"EUR": [1.08, 1.10, 1.10, 1.12], "CHF": [1.15, 1.15, 1.15, 1.15]}, index=days)
    months = pd.to_datetime(["2024-01-01", "2024-02-01"])
    rates = pd.DataFrame({"EUR": [0.040, 0.041], "USD": [0.053, 0.053], "CHF": [0.017, 0.017]}, index=months)
    yields = pd.DataFrame({"DE": [0.022, 0.023], "US": [0.040, 0.041], "IT": [0.038, 0.041], "FR": [0.028, 0.029]},
                          index=months)
    d = euro_drivers(spot, rates, yields.reindex(columns=["US", "DE", "IT", "FR"]))
    row = d.loc["2024-02-01"]
    assert row["EURUSD"] == pytest.approx(100 * (1.11 / 1.09 - 1))
    assert row["d_short_US"] == pytest.approx(0.1)  # differential -1.3 -> -1.2 pp
    assert row["d_10y_US"] == pytest.approx(0.0)
    assert row["d_BTP"] == pytest.approx(0.2)  # 1.6 -> 1.8 pp
    assert row["d_OAT"] == pytest.approx(0.0)
