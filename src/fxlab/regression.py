"""Ordinary least squares with Newey-West (HAC) standard errors (paper, section 5).

Monthly financial series are often heteroskedastic and, when built from monthly
averages, autocorrelated; Newey-West standard errors remain valid in both cases.
Implemented with NumPy only, so that every step is visible and tested.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OLSResult:
    coef: pd.Series
    se: pd.Series
    nobs: int
    r2: float
    lags: int

    @property
    def tstat(self) -> pd.Series:
        return self.coef / self.se


def newey_west_lags(n: int) -> int:
    """Usual rule of thumb: floor(4 (n / 100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def ols(y: pd.Series, X: pd.DataFrame, lags: int | None = None, constant: bool = True) -> OLSResult:
    """Regress y on X (rows with a missing value are dropped).

    beta = (X'X)^-1 X'y; Var(beta) = (X'X)^-1 S (X'X)^-1, with S the Newey-West
    estimate of the long-run covariance of x_t u_t (Bartlett weights 1 - l/(L+1)).
    lags=0 gives White heteroskedasticity-robust standard errors.
    """
    data = pd.concat([y.rename("_y"), X], axis=1).dropna()
    yv = data["_y"].to_numpy()
    Xd = data.drop(columns="_y")
    if constant:
        Xd = pd.concat([pd.Series(1.0, index=Xd.index, name="const"), Xd], axis=1)
    Xv = Xd.to_numpy()
    n, _ = Xv.shape
    L = newey_west_lags(n) if lags is None else lags

    xtx_inv = np.linalg.inv(Xv.T @ Xv)
    beta = xtx_inv @ Xv.T @ yv
    u = yv - Xv @ beta
    g = Xv * u[:, None]  # x_t u_t
    S = g.T @ g
    for lag in range(1, L + 1):
        w = 1 - lag / (L + 1)
        gamma = g[lag:].T @ g[:-lag]
        S += w * (gamma + gamma.T)
    cov = xtx_inv @ S @ xtx_inv

    tss = ((yv - yv.mean()) ** 2).sum() if constant else (yv**2).sum()
    return OLSResult(
        coef=pd.Series(beta, index=Xd.columns),
        se=pd.Series(np.sqrt(np.diag(cov)), index=Xd.columns),
        nobs=n,
        r2=1 - (u**2).sum() / tss,
        lags=L,
    )


def rolling_coef(y: pd.Series, X: pd.DataFrame, name: str, window: int = 36) -> pd.Series:
    """Coefficient on `name` in OLS regressions over rolling windows of `window` months."""
    data = pd.concat([y.rename("_y"), X], axis=1).dropna()
    out = {}
    for end in range(window, len(data) + 1):
        chunk = data.iloc[end - window:end]
        out[chunk.index[-1]] = ols(chunk["_y"], chunk.drop(columns="_y"), lags=0).coef[name]
    return pd.Series(out, name=name)
