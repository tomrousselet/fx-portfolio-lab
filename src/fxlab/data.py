"""Download and load FX spot rates and short-term interest rates from FRED.

FRED serves every series as a CSV file at a public URL, without an API key.
Raw files are cached in data/raw/ (not versioned): only this code is versioned,
so anyone can rebuild exactly the same dataset.
"""

import urllib.request
from pathlib import Path

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Spot rates (Federal Reserve H.10, daily noon buying rates in New York).
# FRED quotes some currencies in USD per unit of foreign currency (EUR, GBP,
# AUD, NZD) and the others in foreign currency per USD. `usd_per_fcy` records
# which, so that every rate can be converted to USD per unit of foreign currency.
SPOT_SERIES = {
    "EUR": ("DEXUSEU", True),
    "GBP": ("DEXUSUK", True),
    "AUD": ("DEXUSAL", True),
    "NZD": ("DEXUSNZ", True),
    "JPY": ("DEXJPUS", False),
    "CHF": ("DEXSZUS", False),
    "CAD": ("DEXCAUS", False),
    "NOK": ("DEXNOUS", False),
    "SEK": ("DEXSDUS", False),
}

# 3-month interbank rates (OECD Main Economic Indicators), monthly, in % per year.
RATE_SERIES = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "JPY": "IR3TIB01JPM156N",
    "CHF": "IR3TIB01CHM156N",
    "CAD": "IR3TIB01CAM156N",
    "NOK": "IR3TIB01NOM156N",
    "SEK": "IR3TIB01SEM156N",
}


def all_series_ids() -> list[str]:
    return [sid for sid, _ in SPOT_SERIES.values()] + list(RATE_SERIES.values())


def download_series(series_id: str, raw_dir: Path = RAW_DIR, refresh: bool = False) -> Path:
    """Save the FRED CSV of `series_id` in raw_dir; skip it if already there."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{series_id}.csv"
    if refresh or not path.exists():
        with urllib.request.urlopen(FRED_URL.format(series_id), timeout=60) as response:
            path.write_bytes(response.read())
    return path


def read_fred_csv(path: Path) -> pd.Series:
    """Read a FRED CSV (columns: date, value). Missing values ('.' or empty) become NaN."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    series.index.name = "date"
    return series


def load_spot(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Daily spot rates in USD per unit of foreign currency, one column per currency."""
    columns = {}
    for ccy, (sid, usd_per_fcy) in SPOT_SERIES.items():
        s = read_fred_csv(raw_dir / f"{sid}.csv")
        columns[ccy] = s if usd_per_fcy else 1 / s
    return pd.DataFrame(columns)


def load_rates(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Monthly 3-month interest rates in decimal per year (0.05 = 5%), one column per currency."""
    columns = {ccy: read_fred_csv(raw_dir / f"{sid}.csv") / 100 for ccy, sid in RATE_SERIES.items()}
    return pd.DataFrame(columns)
