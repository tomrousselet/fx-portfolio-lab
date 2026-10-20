"""Download every series used by the project (FRED and ECB) into data/raw/.

Usage:
    python scripts/download_data.py            # download missing files only
    python scripts/download_data.py --refresh  # download everything again (latest data)
"""

import argparse

from fxlab.data import all_sources, download_source, load_rates, load_spot


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download files already present")
    args = parser.parse_args()

    for source in all_sources():
        path = download_source(source, refresh=args.refresh)
        print(f"  {source[0].upper():<5} {path.name}")

    spot, rates = load_spot(), load_rates()
    print("\nSpot rates (USD per unit of foreign currency), last valid value:")
    print(spot.ffill().iloc[-1].round(4).to_string())
    print("\nFirst and last date with data:")
    summary = pd_summary(spot, rates)
    print(summary.to_string())


def pd_summary(spot, rates):
    import pandas as pd

    rows = {}
    for ccy in rates.columns:
        s = spot[ccy].dropna() if ccy in spot else None
        r = rates[ccy].dropna()
        rows[ccy] = {
            "spot from": s.index[0].date() if s is not None else "-",
            "spot to": s.index[-1].date() if s is not None else "-",
            "rate from": r.index[0].date(),
            "rate to": r.index[-1].date(),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    main()
