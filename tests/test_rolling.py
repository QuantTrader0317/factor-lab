import numpy as np
import pandas as pd
from factorlab.rolling import rolling_betas

def test_rolling_betas_shapes():
    rng = np.random.default_rng(42)
    n = 120
    X = pd.DataFrame({
        "Mkt-RF": rng.normal(0, 1, n),
        "SMB": rng.normal(0, 1, n),
        "HML": rng.normal(0, 1, n),
    })
    beta = np.array([1.0, 0.0, 0.0])
    y = pd.Series(X.values @ beta + rng.normal(0, 0.05, n))
    y.index = pd.date_range("2020-01-31", periods=n, freq="M")
    X.index = y.index

    out = rolling_betas(y, X, window=24)
    assert out.shape[0] == n - 24 + 1
    assert list(out.columns) == ["Mkt-RF", "SMB", "HML"]
