from __future__ import annotations
from typing import Iterable, Tuple, Optional
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


def load_ff3_sample(path: str) -> pd.DataFrame:
    """Load the included sample monthly Fama–French 3-factor CSV.

    Expects columns: Date, Mkt-RF, SMB, HML, RF
    Date format: YYYY-MM (month-end implied).

    Returns a DataFrame indexed by period end (month-end), values in percent converted to decimals.
    """
    df = pd.read_csv(path)
    need = {"Date", "Mkt-RF", "SMB", "HML", "RF"}
    if not need.issubset(df.columns):
        raise ValueError("CSV missing required FF3 columns.")
    idx = pd.to_datetime(df["Date"]) + pd.tseries.offsets.MonthEnd(0)
    fac = df[["Mkt-RF", "SMB", "HML", "RF"]].astype(float) / 100.0
    fac.index = idx
    fac = fac.sort_index()
    return fac


def compute_simple_returns(prices: pd.Series | pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def resample_to_month_end(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns).resample("M").prod() - 1.0


def excess_returns(asset_returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    rf_aligned = rf.reindex(asset_returns.index).fillna(method="ffill")
    return asset_returns.sub(rf_aligned, axis=0)


def fetch_prices_yf(tickers: Iterable[str], start: str, end: str, freq: str = "D") -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance not available. Install optional dependency or use sample CSV.")
    tickers = list(tickers)
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all")
    if freq.upper().startswith("M"):
        data = data.resample("M").last()
    return data


def join_factors_and_assets(asset_returns_excess: pd.DataFrame, factors: pd.DataFrame, factor_cols: Tuple[str, ...] = ("Mkt-RF", "SMB", "HML")) -> pd.DataFrame:
    use = factors[list(factor_cols)]
    out = asset_returns_excess.join(use, how="inner").dropna()
    return out
