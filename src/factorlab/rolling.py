from __future__ import annotations
import pandas as pd
from .models import ols

def rolling_betas(y: pd.Series, X: pd.DataFrame, window: int) -> pd.DataFrame:
    X = X.reindex(y.index).dropna()
    y = y.reindex(X.index).dropna()
    X = X.reindex(y.index)
    out = []
    idx = []
    for i in range(window, len(y) + 1):
        y_w = y.iloc[i - window : i]
        X_w = X.iloc[i - window : i]
        res = ols(y_w, X_w, add_const=True)
        out.append(res["betas"])
        idx.append(y_w.index[-1])
    if not out:
        return pd.DataFrame(columns=X.columns)
    return pd.DataFrame(out, index=idx)
