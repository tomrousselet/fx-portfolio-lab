"""Download and load FX spot rates, interest rates and bond yields.

Two public sources, both serving CSV files at a public URL without an API key:
FRED (Federal Reserve Bank of St. Louis) and the ECB Data Portal. A series is
identified by a *source*: ("fred", series_id) or ("ecb", flow, key).
Raw files are cached in data/raw/ (not versioned): only this code is versioned,
so anyone can rebuild exactly the same dataset.
"""

import urllib.request
from pathlib import Path

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata"
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


# Extensions of the OECD rates that stop in January 2026 (decision D17).
# EUR: the OECD series is the 3-month Euribor, published to date by the ECB.
# GBP: no 3-month interbank rate after LIBOR; extended with the monthly average of SONIA.
RATE_EXTENSIONS = {
    "EUR": ("ecb", "FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"),
    "GBP": ("fred", "IUDSOIA"),
}

# Secured or overnight risk-free reference rates, daily (robustness check, decision D16).
RFR_SERIES = {
    "USD": ("fred", "SOFR"),
    "EUR": ("ecb", "EST", "B.EU000A2X2A25.WT"),
    "GBP": ("fred", "IUDSOIA"),
}

# 10-year government bond yields (OECD MEI), monthly averages, % per year (euro drivers, section 5).
YIELD_SERIES = {
    "US": ("fred", "IRLTLT01USM156N"),
    "DE": ("fred", "IRLTLT01DEM156N"),
    "IT": ("fred", "IRLTLT01ITM156N"),
    "FR": ("fred", "IRLTLT01FRM156N"),
    "ES": ("fred", "IRLTLT01ESM156N"),
    "CH": ("fred", "IRLTLT01CHM156N"),
}


def all_sources() -> list[tuple]:
    sources = [("fred", sid) for sid, _ in SPOT_SERIES.values()]
    sources += [("fred", sid) for sid in RATE_SERIES.values()]
    sources += list(RATE_EXTENSIONS.values()) + list(RFR_SERIES.values()) + list(YIELD_SERIES.values())
    return list(dict.fromkeys(sources))  # unique, in order


def source_path(source: tuple, raw_dir: Path = RAW_DIR) -> Path:
    if source[0] == "fred":
        return raw_dir / f"{source[1]}.csv"
    _, flow, key = source
    return raw_dir / f"ECB_{flow}_{key}.csv"


def download_source(source: tuple, raw_dir: Path = RAW_DIR, refresh: bool = False) -> Path:
    """Save the CSV of `source` in raw_dir; skip it if already there."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = source_path(source, raw_dir)
    if source[0] == "fred":
        url = FRED_URL.format(source[1])
    else:
        url = ECB_URL.format(flow=source[1], key=source[2])
    if refresh or not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def download_series(series_id: str, raw_dir: Path = RAW_DIR, refresh: bool = False) -> Path:
    """Save the FRED CSV of `series_id` in raw_dir (shortcut for a FRED source)."""
    return download_source(("fred", series_id), raw_dir, refresh)


def read_fred_csv(path: Path) -> pd.Series:
    """Read a FRED CSV (columns: date, value). Missing values ('.' or empty) become NaN."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    series.index.name = "date"
    return series


def read_ecb_csv(path: Path) -> pd.Series:
    """Read an ECB Data Portal CSV (columns TIME_PERIOD and OBS_VALUE); monthly periods are dated on the 1st."""
    df = pd.read_csv(path, usecols=["TIME_PERIOD", "OBS_VALUE"])
    series = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").to_numpy(),
                       index=pd.to_datetime(df["TIME_PERIOD"]), name=path.stem)
    series.index.name = "date"
    return series.sort_index()


def read_source(source: tuple, raw_dir: Path = RAW_DIR) -> pd.Series:
    path = source_path(source, raw_dir)
    return read_fred_csv(path) if source[0] == "fred" else read_ecb_csv(path)


def monthly_average(series: pd.Series) -> pd.Series:
    """Average of the observations of each month, dated on the 1st (the FRED convention for monthly data)."""
    return series.dropna().resample("MS").mean()


def splice(base: pd.Series, extension: pd.Series, overlap: int = 12) -> pd.Series:
    """Extend `base` after its last observation with `extension`, shifted by a constant.

    The shift is the mean of base - extension over the last `overlap` months where both exist,
    so that the spliced series has no jump at the junction (decision D17).
    """
    base = base.dropna()
    common = (base - extension).dropna().iloc[-overlap:]
    if common.empty:
        raise ValueError("no overlap between the base series and its extension")
    shifted = extension.dropna() + common.mean()
    return pd.concat([base, shifted[shifted.index > base.index[-1]]])


def load_spot(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Daily spot rates in USD per unit of foreign currency, one column per currency."""
    columns = {}
    for ccy, (sid, usd_per_fcy) in SPOT_SERIES.items():
        s = read_fred_csv(raw_dir / f"{sid}.csv")
        columns[ccy] = s if usd_per_fcy else 1 / s
    return pd.DataFrame(columns)


def load_rates(raw_dir: Path = RAW_DIR, extend: bool = True) -> pd.DataFrame:
    """Monthly 3-month interest rates in decimal per year (0.05 = 5%), one column per currency.

    With extend=True, the EUR and GBP rates are extended after January 2026 (RATE_EXTENSIONS).
    """
    columns = {ccy: read_fred_csv(raw_dir / f"{sid}.csv") for ccy, sid in RATE_SERIES.items()}
    if extend:
        for ccy, source in RATE_EXTENSIONS.items():
            columns[ccy] = splice(columns[ccy], monthly_average(read_source(source, raw_dir)))
    return pd.DataFrame(columns) / 100


def load_rfr_rates(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Monthly averages of the overnight risk-free rates (SOFR, ESTR, SONIA), decimal per year."""
    columns = {ccy: monthly_average(read_source(src, raw_dir)) for ccy, src in RFR_SERIES.items()}
    return pd.DataFrame(columns) / 100


def load_yields(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Monthly 10-year government bond yields, decimal per year, one column per country."""
    return pd.DataFrame({c: read_source(src, raw_dir) for c, src in YIELD_SERIES.items()}) / 100
