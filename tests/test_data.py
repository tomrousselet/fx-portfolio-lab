"""Tests of the data loaders on small fake FRED files (no internet needed)."""

import numpy as np
import pandas as pd
import pytest

from fxlab import data


def write_csv(raw_dir, series_id, rows):
    lines = [f"observation_date,{series_id}"] + [f"{d},{v}" for d, v in rows]
    (raw_dir / f"{series_id}.csv").write_text("\n".join(lines))


@pytest.fixture
def raw_dir(tmp_path):
    for ccy, (sid, usd_per_fcy) in data.SPOT_SERIES.items():
        value = 1.10 if usd_per_fcy else 150.0
        write_csv(tmp_path, sid, [("2024-01-02", value), ("2024-01-03", "")])
    for sid in data.RATE_SERIES.values():
        write_csv(tmp_path, sid, [("2024-01-01", 5.0)])
    return tmp_path


def test_missing_values_become_nan(raw_dir):
    s = data.read_fred_csv(raw_dir / "DEXUSEU.csv")
    assert s.iloc[0] == 1.10
    assert np.isnan(s.iloc[1])


def test_spot_converted_to_usd_per_foreign_currency(raw_dir):
    """EUR is already quoted in USD per EUR; JPY is quoted in JPY per USD and must be inverted."""
    spot = data.load_spot(raw_dir)
    assert spot.loc["2024-01-02", "EUR"] == pytest.approx(1.10)
    assert spot.loc["2024-01-02", "JPY"] == pytest.approx(1 / 150)


def test_rates_in_decimal(raw_dir):
    rates = data.load_rates(raw_dir, extend=False)
    assert rates.loc["2024-01-01", "USD"] == pytest.approx(0.05)
    assert list(rates.columns) == list(data.RATE_SERIES)


def test_ecb_csv_reader(tmp_path):
    """ECB files have many columns; only TIME_PERIOD and OBS_VALUE matter. Monthly periods map to the 1st."""
    path = tmp_path / "ECB_FM_KEY.csv"
    path.write_text("KEY,FREQ,TIME_PERIOD,OBS_VALUE,TITLE\nk,M,2026-08,2.51,x\nk,M,2026-09,2.64,x\n")
    s = data.read_ecb_csv(path)
    assert s.index[0] == pd.Timestamp("2026-08-01")
    assert s.iloc[1] == pytest.approx(2.64)


def test_monthly_average_dated_first_of_month():
    daily = pd.Series([1.0, 3.0, 5.0], index=pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-15"]))
    m = data.monthly_average(daily)
    assert m.loc["2024-01-01"] == pytest.approx(2.0)
    assert m.loc["2024-02-01"] == pytest.approx(5.0)


def test_splice_removes_the_jump():
    """Extension 0.5 below the base on the overlap: shifted up by 0.5 after the last base date."""
    idx = pd.date_range("2025-01-01", periods=6, freq="MS")
    base = pd.Series([3.0, 3.1, 3.2, np.nan, np.nan, np.nan], index=idx)
    ext = pd.Series([2.5, 2.6, 2.7, 2.8, 2.9, 3.0], index=idx)
    s = data.splice(base, ext, overlap=12)
    assert list(s.loc[:"2025-03-01"]) == [3.0, 3.1, 3.2]  # base kept where it exists
    assert s.loc["2025-04-01"] == pytest.approx(3.3)
    assert s.index[-1] == pd.Timestamp("2025-06-01")


def test_splice_needs_an_overlap():
    base = pd.Series([3.0], index=pd.to_datetime(["2025-01-01"]))
    ext = pd.Series([2.0], index=pd.to_datetime(["2025-02-01"]))
    with pytest.raises(ValueError):
        data.splice(base, ext)


def test_every_source_has_a_distinct_file():
    paths = [data.source_path(s) for s in data.all_sources()]
    assert len(paths) == len(set(paths))
