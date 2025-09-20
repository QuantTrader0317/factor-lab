import numpy as np
import pandas as pd
from math import isclose
from factorlab.models import ols

def test_ols_recovers_betas_close():
    rng = np.random.default_rng(123)
    n = 500
    X = pd.DataFrame({
        "Mkt-RF": rng.normal(0, 1, n),
        "SMB": rng.normal(0, 1, n),
        "HML": rng.normal(0, 1, n),
    })
    true_alpha = 0.001
    true_betas = np.array([1.2, -0.3, 0.5])
    eps = rng.normal(0, 0.01, n)
    y = pd.Series(true_alpha + X.values @ true_betas + eps)

    res = ols(y, X, add_const=True)
    est = np.array([res["alpha"], *res["betas"].values])

    assert isclose(est[0], true_alpha, rel_tol=0.05, abs_tol=0.001)
    assert np.allclose(est[1:], true_betas, rtol=0.05, atol=0.05)
    assert 0.9 <= res["n"] <= 1000
