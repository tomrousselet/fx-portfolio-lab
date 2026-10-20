# Architecture - FX Portfolio Lab

**Version:** 0.3 (Phase 1 and euro rates) | **Language:** Python 3.11+ | **Documentation:** [working paper](paper/)

---

## 1. Overview

The project separates three layers that never mix:

1. **Library** (`src/fxlab/`): every calculation lives in a tested Python package. Nothing is computed in a notebook or a script that is not in the library.
2. **Scripts** (`scripts/`): thin entry points that call the library to download data or produce the tables and figures of the paper.
3. **Paper** (`paper/`): the LaTeX working paper. Its tables and figures are *generated* by the scripts and never edited by hand, so the text always matches the code.

Raw data are not versioned: `scripts/download_data.py` rebuilds them from public sources with one command.

---

## 2. Data flow

```mermaid
flowchart TB
    subgraph Sources["Public sources (FRED, ECB Data Portal)"]
        H10["Federal Reserve H.10<br/>daily spot rates<br/>9 G10 currencies"]
        OECD["OECD MEI<br/>3-month rates, 10-year yields"]
        ECB["ECB and FRED<br/>Euribor, ESTR, SOFR, SONIA"]
    end

    subgraph Raw["data/raw/ (not versioned)"]
        CSV["one CSV per series<br/>DEXUSEU.csv, IR3TIB01USM156N.csv, ..."]
    end

    subgraph Lib["fxlab package"]
        DATA["data.py<br/>download, parse,<br/>USD per foreign unit"]
        RX["fx_returns.py<br/>month-end sampling<br/>CIP forwards, carry<br/>excess returns"]
        PF["portfolio.py<br/>valuation, rebalancing<br/>amounts traded"]
        FW["forwards.py + costs.py<br/>rolled forwards<br/>roll and outright costs"]
        MT["metrics.py<br/>Sharpe, t-stat, skewness,<br/>drawdown, break-even"]
        EU["euro.py + regression.py<br/>rates and risk channels<br/>Newey-West OLS"]
    end

    subgraph Out["Outputs"]
        TAB["paper/tables/*.tex"]
        FIG["paper/figures/*.png"]
        PDF["paper/fx-portfolio-lab.pdf"]
    end

    H10 -->|download_data.py| CSV
    OECD -->|download_data.py| CSV
    ECB -->|download_data.py| CSV
    CSV --> DATA
    DATA --> RX
    RX --> PF
    RX --> FW
    FW --> MT
    MT -->|backtest_benchmarks.py| TAB
    DATA --> EU
    EU -->|euro_drivers.py| TAB
    EU -->|euro_drivers.py| FIG
    RX -->|describe_returns.py| TAB
    RX -->|make_figures.py| FIG
    TAB --> PDF
    FIG --> PDF
```

---

## 3. Modules

```mermaid
graph TB
    subgraph Done["Implemented"]
        D["fxlab.data<br/>──────────<br/>SPOT_SERIES, RATE_SERIES<br/>download_series()<br/>load_spot(), load_rates()"]
        R["fxlab.fx_returns<br/>──────────<br/>month_end_spot(), month_end_rates()<br/>forward_rates(), carry()<br/>excess_returns(), monthly_dataset()"]
        P["fxlab.portfolio<br/>──────────<br/>simple_returns(), compound()<br/>backtest() -> BacktestResult"]
        C["fxlab.costs<br/>──────────<br/>CostModel, BASE_CASE<br/>rebalancing_cost()"]
        F["fxlab.forwards<br/>──────────<br/>backtest_forwards()<br/>equal_weights()"]
        M["fxlab.metrics<br/>──────────<br/>summary(), sharpe_ratio()<br/>break_even_cost_scale()"]
        G["fxlab.regression<br/>──────────<br/>ols() -> OLSResult<br/>rolling_coef()"]
        E["fxlab.euro<br/>──────────<br/>euro_drivers()"]
    end

    subgraph Planned["Planned"]
        S["signals<br/>carry, momentum, value"]
        O["optimisation<br/>GMV, mean-variance,<br/>risk parity, shrinkage"]
        H["hedging<br/>EUR investor"]
        X["options / structuring<br/>Garman-Kohlhagen, smile,<br/>Monte Carlo"]
    end

    D --> R --> F
    C --> F --> M
    P -.->|same method| F
    D --> E --> G
    R --> S --> O --> F
    R --> H
    X
```

| Module | Responsibility | Paper |
|---|---|---|
| `fxlab.data` | Download FRED and ECB series, parse missing values, convert every spot rate to USD per unit of foreign currency, extend rates by splicing | Section 3, D5, D7, D16, D17 |
| `fxlab.fx_returns` | Month-end sampling, CIP forward rates, carry, monthly excess returns with a strict timing rule | Section 4.2, D8-D11 |
| `fxlab.portfolio` | Value buy-and-hold and rebalanced portfolios by propagating amounts; report the amount traded | Section 4.1, D4 |
| `fxlab.forwards` | Value portfolios of rolled one-month forwards: notional weights, collateral, costs, turnover | Section 4.3 |
| `fxlab.costs` | Swap half-spread on the notional rolled, outright half-spread on the notional traded; base case and scaling | Section 4.4, D14, D15 |
| `fxlab.metrics` | Annualised mean and volatility, Sharpe, t-stat, skewness, kurtosis, drawdown, break-even cost scale | Section 4.5 |
| `fxlab.regression` | OLS with Newey-West standard errors, rolling coefficients | Section 6, D18 |
| `fxlab.euro` | Monthly changes of EURUSD and EURCHF and of rate differentials and sovereign spreads | Section 6, D18 |

---

## 4. Conventions

| Item | Convention |
|---|---|
| Spot rate $S_t$ | USD per unit of foreign currency (an increase is an appreciation of the foreign currency) |
| Interest rates | decimal per year (0.05 = 5%), dated at the end of the month they average |
| Returns | simple returns, never log returns, so that $r_p = w'r$ holds exactly |
| Dates | month-end timestamps; $rx_{t+1}$ is indexed at $t+1$ |
| Timing | a position opened at $t$ uses only data available at the end of month $t$ |

---

## 5. Verification

```mermaid
flowchart LR
    Change["Code change"] --> Local["pytest (local)"]
    Local --> Push["git push"]
    Push --> CI["GitHub Actions<br/>pytest on a clean machine"]
    CI -->|pass| Badge["README badge: passing"]
    CI -->|fail| Fix["fix before anything else"]
```

Each test cites the result it reproduces (a course example, a closed-form formula or a timing rule). The full list is in the paper, Appendix B. Tests never use the internet: data loaders are tested on small synthetic files.
