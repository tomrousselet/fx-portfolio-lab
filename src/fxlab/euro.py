"""What moves the euro: the rates channel and the sovereign risk channel (paper, section 5).

Monthly data on a common grid of monthly averages (dated on the 1st), because the
OECD interest rates and bond yields are monthly averages (decision D18):

    EURUSD, EURCHF   percentage change of the monthly average spot rate (+ = euro appreciation)
    rates channel    change in the short-rate differential i_EUR - i_USD (or - i_CHF), and
                     in the 10-year differential Bund - US Treasury, in percentage points
    risk channel     change in the 10-year spread of Italy (BTP) and France (OAT) over the Bund,
                     in percentage points
"""

import pandas as pd

from fxlab.data import monthly_average

PP = 100  # decimal -> percentage points


def euro_drivers(spot: pd.DataFrame, rates: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    """Monthly changes of the euro and of its candidate drivers.

    spot:    daily spot rates in USD per unit of foreign currency (data.load_spot)
    rates:   monthly 3-month rates, decimal (data.load_rates)
    yields:  monthly 10-year yields, decimal (data.load_yields)
    """
    eurusd = monthly_average(spot["EUR"])
    eurchf = monthly_average(spot["EUR"] / spot["CHF"])  # CHF per EUR
    out = pd.DataFrame({
        "EURUSD": 100 * eurusd.pct_change(),
        "EURCHF": 100 * eurchf.pct_change(),
        "d_short_US": PP * (rates["EUR"] - rates["USD"]).diff(),
        "d_short_CH": PP * (rates["EUR"] - rates["CHF"]).diff(),
        "d_10y_US": PP * (yields["DE"] - yields["US"]).diff(),
        "d_BTP": PP * (yields["IT"] - yields["DE"]).diff(),
        "d_OAT": PP * (yields["FR"] - yields["DE"]).diff(),
    })
    return out.loc["1999-02":]
