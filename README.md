# factor-lab

From-scratch **factor models** you can read and extend. Core goals:
- Regress asset returns on factors (Fama–French 3/5).
- Compute **rolling betas** with windowed OLS.
- Provide a simple **CLI** over yfinance + Fama–French.
- Ship with **tests**, **types**, **CI**, and a **sample factor CSV** so you can run offline.

> Use this as a portfolio piece to show econometrics, matrix algebra, and clean engineering.

## Install
```bash
python -m venv .venv
# Windows
. .venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -e ".[dev]"
pre-commit install
pytest -q
```

## CLI quickstart
```bash
# With sample monthly FF3 included in repo (offline demo)
factorlab ff3 --tickers AAPL MSFT --start 2022-01-01 --end 2023-12-31 --freq M --ff-source sample --ff-sample-path data/sample_ff3_monthly.csv

# If you want real Fama-French download and daily data (requires internet)
factorlab ff3 --tickers AAPL --start 2018-01-01 --end 2024-12-31 --freq M --ff-source download
```

## What’s inside
- `factorlab/data.py`: loaders for Fama–French factors, yfinance prices, and return processing (excess returns).
- `factorlab/models.py`: OLS + HAC (Newey–West) options, FF3 helper.
- `factorlab/rolling.py`: rolling window betas.
- `factorlab/plots.py`: matplotlib plots for exposures.
- `tests/`: synthetic tests with known coefficients + sample FF3 parsing.

## Roadmap
- Add FF5, Carhart momentum.
- Add robust covariance options (HAC lags auto).
- Add Black–Litterman under `portfolio/`.

## License
MIT
