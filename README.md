# FX Portfolio Lab

From portfolio construction to FX structuring: a research project that looks at an FX desk from three angles.

1. **Portfolio manager** - building and backtesting G10 currency portfolios (carry, momentum, value) with minimum-variance, mean-variance and risk-parity allocation.
2. **Hedger** - finding the optimal currency hedge ratio for a euro-based investor holding foreign assets.
3. **Structurer** - pricing and comparing the FX hedging products sold to corporate treasurers (forwards, collars, participating forwards, TARFs).

The methodology, every design choice and its justification are documented in the working paper in [`paper/`](paper/).

## Project status

| Phase | Content | Status |
|---|---|---|
| 0 | Setup: repository, environment, tests, paper skeleton | done |
| 1 | Engine: data and currency excess returns (done), backtester with transaction costs | in progress |
| 2 | Currency portfolios: carry, momentum, value; GMV / MV / risk parity | planned |
| 3 | Optimal currency hedging for a EUR investor | planned |
| 4 | FX structuring: Garman-Kohlhagen, smile, structured hedging products | planned |
| 5 | Dashboard, final paper | planned |

## Repository layout

```
src/fxlab/      Python package (the library)
tests/          automated tests (pytest)
scripts/        command-line scripts (data download, tables for the paper)
notebooks/      exploratory notebooks that use the library
data/raw/       downloaded data (not versioned)
paper/          working paper (LaTeX)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

To build the paper (requires a TeX distribution such as BasicTeX), run `make` in `paper/`; the PDF is `paper/build/main.pdf`.
