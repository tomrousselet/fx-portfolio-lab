"""Tests of the data loaders on small fake FRED files (no internet needed)."""

import numpy as np
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
    rates = data.load_rates(raw_dir)
    assert rates.loc["2024-01-01", "USD"] == pytest.approx(0.05)
    assert list(rates.columns) == list(data.RATE_SERIES)
